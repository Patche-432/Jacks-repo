"""
Tests for the May 2026 production-readiness fix: pre-flight margin check
and the per-symbol margin circuit breaker.

Background
----------
On 2026-05-04 the bot placed one EURUSD BUY (#791705 @ 22:08) and then
attempted two more entries which both failed with `retcode 10019
(No money)`. The order_send retries cost a tiny bit of slippage, made
the trade history noisy, and — most importantly — the bot kept silently
trying. Operators couldn't tell from a glance whether the bot was
"working but blocked" or "broken".

The fix has two parts. AgentZeroBot.execute_trade() now:
  1. Calls mt5.order_check(request) BEFORE mt5.order_send(request).
     If the broker would reject the request (no money, invalid stops,
     etc.), we never send the live order.
  2. Increments a per-symbol consecutive-failure counter on every
     pre-check rejection. After N (default 3) in a row, the symbol's
     "margin breaker" is armed, and execute_trade() returns early
     until reset_margin_breaker() is called or a future pre-check
     succeeds.

These tests pin that behaviour. Run from the repo root:
    python -m pytest tests/test_execution_margin_breaker.py -v
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _make_check(retcode: int, *, comment: str = "",
                margin_free: float = 0.0,
                margin: float = 0.0) -> types.SimpleNamespace:
    """Mimic mt5.order_check()'s MqlTradeCheckResult namedtuple."""
    return types.SimpleNamespace(
        retcode=retcode,
        comment=comment,
        balance=10_000.0,
        equity=10_000.0,
        profit=0.0,
        margin=margin,
        margin_free=margin_free,
        margin_level=1_000.0,
    )


class TestMarginPreCheck(unittest.TestCase):
    """The pre-flight check decides whether to send the order."""

    def setUp(self) -> None:
        try:
            from ai_pro import AgentZeroBot
        except ImportError as exc:
            self.skipTest(f"ai_pro unavailable: {exc}")
        self.AgentZeroBot = AgentZeroBot
        self.bot = AgentZeroBot(use_ai=False)

    def test_pre_check_returns_ok_on_done(self):
        """retcode == TRADE_RETCODE_DONE → green-light send."""
        import MetaTrader5 as mt5
        request = {"symbol": "EURUSD", "volume": 0.5, "type": mt5.ORDER_TYPE_BUY}
        with patch.object(mt5, "order_check",
                          return_value=_make_check(mt5.TRADE_RETCODE_DONE,
                                                    margin_free=5_000.0,
                                                    margin=200.0)):
            ok, info = self.bot._margin_pre_check(mt5, request)
        self.assertTrue(ok)
        self.assertTrue(info["ok"])
        self.assertFalse(info["no_money"])

    def test_pre_check_returns_ok_on_zero_retcode(self):
        """Some brokers return retcode == 0 from order_check on success
        (the SDK treats this as 'request is valid'). Must also pass."""
        import MetaTrader5 as mt5
        request = {"symbol": "EURUSD", "volume": 0.5, "type": mt5.ORDER_TYPE_BUY}
        with patch.object(mt5, "order_check",
                          return_value=_make_check(0)):
            ok, info = self.bot._margin_pre_check(mt5, request)
        self.assertTrue(ok)

    def test_pre_check_blocks_no_money(self):
        """retcode 10019 (No money) → never send."""
        import MetaTrader5 as mt5
        request = {"symbol": "EURUSD", "volume": 0.5, "type": mt5.ORDER_TYPE_BUY}
        with patch.object(mt5, "order_check",
                          return_value=_make_check(mt5.TRADE_RETCODE_NO_MONEY,
                                                    comment="No money",
                                                    margin_free=10.0,
                                                    margin=200.0)):
            ok, info = self.bot._margin_pre_check(mt5, request)
        self.assertFalse(ok)
        self.assertTrue(info["no_money"])
        self.assertEqual(info["retcode"], mt5.TRADE_RETCODE_NO_MONEY)

    def test_pre_check_blocks_when_order_check_returns_none(self):
        """A None response from the broker means we cannot validate the
        request. Defaulting to 'send anyway' would defeat the point of
        the pre-check, so we treat it as a hard block."""
        import MetaTrader5 as mt5
        request = {"symbol": "EURUSD", "volume": 0.5, "type": mt5.ORDER_TYPE_BUY}
        with patch.object(mt5, "order_check", return_value=None), \
             patch.object(mt5, "last_error", return_value=(0, "")):
            ok, info = self.bot._margin_pre_check(mt5, request)
        self.assertFalse(ok)

    def test_pre_check_blocks_when_order_check_raises(self):
        """A C-extension exception from order_check must not bubble up;
        it must be caught and turned into a clean 'do not send'."""
        import MetaTrader5 as mt5
        request = {"symbol": "EURUSD", "volume": 0.5, "type": mt5.ORDER_TYPE_BUY}
        with patch.object(mt5, "order_check",
                          side_effect=RuntimeError("boom")):
            ok, info = self.bot._margin_pre_check(mt5, request)
        self.assertFalse(ok)
        self.assertIn("boom", info["comment"])


class TestMarginBreakerArming(unittest.TestCase):
    """The breaker arms after N consecutive failures and resets on success."""

    def setUp(self) -> None:
        try:
            from ai_pro import AgentZeroBot
        except ImportError as exc:
            self.skipTest(f"ai_pro unavailable: {exc}")
        self.bot = AgentZeroBot(use_ai=False)

    def test_starts_disarmed(self):
        self.assertFalse(self.bot.is_margin_breaker_armed("EURUSD"))
        status = self.bot.margin_breaker_status()
        self.assertEqual(status["armed_symbols"], [])
        self.assertEqual(status["failure_counts"], {})

    def test_arms_after_threshold_failures(self):
        threshold = self.bot._margin_breaker_threshold
        self.assertGreaterEqual(threshold, 1)

        # threshold-1 failures → not armed yet
        for _ in range(threshold - 1):
            self.bot._record_margin_failure("EURUSD", "test reason")
            self.assertFalse(self.bot.is_margin_breaker_armed("EURUSD"))

        # one more → armed
        self.bot._record_margin_failure("EURUSD", "test reason")
        self.assertTrue(self.bot.is_margin_breaker_armed("EURUSD"))

        status = self.bot.margin_breaker_status()
        self.assertIn("EURUSD", status["armed_symbols"])
        self.assertEqual(status["failure_counts"]["EURUSD"], threshold)

    def test_failures_are_per_symbol(self):
        """A bad streak on EURUSD must not arm GBPUSD."""
        for _ in range(self.bot._margin_breaker_threshold):
            self.bot._record_margin_failure("EURUSD", "test reason")
        self.assertTrue(self.bot.is_margin_breaker_armed("EURUSD"))
        self.assertFalse(self.bot.is_margin_breaker_armed("GBPUSD"))

    def test_success_resets_counter(self):
        """A clean pre-check on the same symbol must zero the counter
        so a one-off 'No money' (e.g. account topped up between cycles)
        doesn't cripple us forever."""
        self.bot._record_margin_failure("EURUSD", "blip")
        self.bot._record_margin_failure("EURUSD", "blip")
        self.bot._record_margin_success("EURUSD")
        self.assertFalse(self.bot.is_margin_breaker_armed("EURUSD"))
        # And a third *new* failure must NOT reach the threshold instantly.
        self.bot._record_margin_failure("EURUSD", "fresh")
        self.assertFalse(self.bot.is_margin_breaker_armed("EURUSD"))

    def test_manual_reset_clears_one_symbol(self):
        for _ in range(self.bot._margin_breaker_threshold):
            self.bot._record_margin_failure("EURUSD", "x")
            self.bot._record_margin_failure("GBPUSD", "x")
        self.assertTrue(self.bot.is_margin_breaker_armed("EURUSD"))
        self.assertTrue(self.bot.is_margin_breaker_armed("GBPUSD"))

        self.bot.reset_margin_breaker("EURUSD")
        self.assertFalse(self.bot.is_margin_breaker_armed("EURUSD"))
        self.assertTrue(self.bot.is_margin_breaker_armed("GBPUSD"))

    def test_manual_reset_all(self):
        for _ in range(self.bot._margin_breaker_threshold):
            self.bot._record_margin_failure("EURUSD", "x")
            self.bot._record_margin_failure("GBPUSD", "x")
        self.bot.reset_margin_breaker(None)
        self.assertFalse(self.bot.is_margin_breaker_armed("EURUSD"))
        self.assertFalse(self.bot.is_margin_breaker_armed("GBPUSD"))


class TestExecuteTradeBreakerShortCircuit(unittest.TestCase):
    """When the breaker is armed, execute_trade() must refuse early —
    BEFORE touching the broker — and return a structured failure dict."""

    def setUp(self) -> None:
        try:
            from ai_pro import AgentZeroBot
        except ImportError as exc:
            self.skipTest(f"ai_pro unavailable: {exc}")
        self.bot = AgentZeroBot(use_ai=False)

    def test_armed_breaker_skips_send(self):
        # Arm the breaker manually
        for _ in range(self.bot._margin_breaker_threshold):
            self.bot._record_margin_failure("EURUSD", "No money")
        self.assertTrue(self.bot.is_margin_breaker_armed("EURUSD"))

        # Pretend MT5 is up so we get past the _ensure_mt5() guard, then
        # invoke execute_trade. order_send and order_check must NEVER be
        # called when the breaker is armed.
        self.bot._mt5_initialized = True
        signal = {
            "signal":      "BUY",
            "stop_loss":   1.0900,
            "take_profit": 1.1100,
            "atr":         0.0,
            "entry_price": 1.1000,
            "signal_source": "test",
        }

        import MetaTrader5 as mt5
        with patch.object(mt5, "order_check",
                          MagicMock(side_effect=AssertionError(
                              "order_check must NOT be called when armed"))), \
             patch.object(mt5, "order_send",
                          MagicMock(side_effect=AssertionError(
                              "order_send must NOT be called when armed"))):
            result = self.bot.execute_trade("EURUSD", signal, lot_size=0.5)

        self.assertFalse(result["success"])
        self.assertEqual(result.get("breaker"), "margin")


if __name__ == "__main__":
    unittest.main(verbosity=2)
