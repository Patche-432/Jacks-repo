"""
POC bias filter parity tests.

The live strategy (ai_pro.AgentZeroBot) and the backtester
(odl.backtest.AgentZeroBacktester) BOTH apply a POC bias gate:
  * BUY signals are kept only when entry_price > POC
  * SELL signals are kept only when entry_price < POC

These tests verify three invariants:

  1.  ALGORITHMIC PARITY - for the same OHLCV window, the backtester's
      `_compute_poc` produces the same value as the live strategy's
      `_compute_poc`. Without this, the backtest is no longer a faithful
      simulation of the live bot.

  2.  CONSTANT PARITY - the lookback (96 bars) and bin count (32) are
      defined on AgentZeroBot only; the backtester imports them. So even
      if a future commit retunes one of these numbers, the other can't
      silently disagree.

  3.  LIVE GATE BEHAVIOUR - the standalone `_poc_aligned` helper rejects
      misaligned setups. (Testing the full `generate_trade_signal` veto
      path requires MT5 + a live strategy fixture; that's covered in the
      backtester simulation tests instead.)

Run from the repo root:
    python -m pytest tests/test_poc_filter_parity.py -v
"""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Make the repo root importable when run as a script.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _make_ohlcv(n_bars: int = 120, seed: int = 42) -> pd.DataFrame:
    """Synthesise a deterministic OHLCV frame.

    n_bars >= 96 so the 96-bar POC window is fully populated. We use a
    fixed seed so the test is repeatable; the exact prices don't matter
    - we only care that BOTH POC implementations land on the same bin.
    """
    rng = np.random.default_rng(seed)
    base_price = 1.10
    drift = rng.normal(0, 0.0005, n_bars).cumsum()
    closes = base_price + drift
    # Highs/lows wrap each close by a small random pip noise.
    spreads = rng.uniform(0.0002, 0.0008, n_bars)
    highs = closes + spreads
    lows = closes - spreads
    opens = np.r_[closes[0], closes[:-1]]  # next bar's open == prev close
    volumes = rng.integers(80, 240, n_bars)

    t0 = datetime(2026, 1, 5, 8, 0, tzinfo=timezone.utc)
    times = [t0 + timedelta(minutes=15 * i) for i in range(n_bars)]

    return pd.DataFrame({
        "time":        times,
        "open":        opens,
        "high":        highs,
        "low":         lows,
        "close":       closes,
        "tick_volume": volumes,
        "real_volume": np.zeros(n_bars, dtype=int),
    })


class TestPOCFilterParity(unittest.TestCase):
    """Live strategy and backtester POC filter must produce identical output."""

    def setUp(self):
        # Importing AgentZeroBot pulls in MT5; on Linux CI we skip these
        # tests rather than fail. The Windows CI job will exercise them.
        try:
            from ai_pro import AgentZeroBot  # noqa: F401
        except ImportError as exc:
            msg = str(exc).lower()
            if "metatrader" in msg or "mt5" in msg or "core.mt5" in msg:
                self.skipTest(f"MT5 unavailable on this host: {exc}")
            raise

        from odl.backtest import AgentZeroBacktester
        from ai_pro import AgentZeroBot

        self.AgentZeroBot = AgentZeroBot
        self.AgentZeroBacktester = AgentZeroBacktester

    def test_constants_pinned_to_live_strategy(self):
        """The backtester must read its lookback/bins from AgentZeroBot,
        so a future tuning change can't make the two filters disagree."""
        self.assertEqual(
            self.AgentZeroBacktester.POC_BIAS_LOOKBACK,
            self.AgentZeroBot.POC_BIAS_LOOKBACK,
            "POC_BIAS_LOOKBACK has drifted between strategy and backtester",
        )
        self.assertEqual(
            self.AgentZeroBacktester.POC_BIAS_BINS,
            self.AgentZeroBot.POC_BIAS_BINS,
            "POC_BIAS_BINS has drifted between strategy and backtester",
        )

    # ---- Algorithmic parity ---------------------------------------------

    def test_compute_poc_matches_live_strategy(self):
        """Same OHLCV in -> same POC value out. The backtester delegates
        to the strategy when one is loaded, so this also verifies the
        delegation path is wired correctly."""
        df = _make_ohlcv(n_bars=120, seed=42)

        # Live strategy POC (instantiate without MT5 by passing use_ai=False
        # - the constructor is lightweight when no broker call is needed).
        strat = self.AgentZeroBot(use_ai=False)
        live_poc = strat._compute_poc(df)

        # Backtester POC, with strategy loaded - should delegate.
        bt = self.AgentZeroBacktester()
        bt._strategy = strat
        bt_poc = bt._compute_poc(df)

        self.assertGreater(live_poc, 0.0, "live POC should be non-zero on real data")
        self.assertAlmostEqual(
            live_poc, bt_poc, places=8,
            msg="backtester POC drifted from live strategy POC",
        )

    def test_compute_poc_fallback_matches_live_when_strategy_missing(self):
        """When no strategy is loaded the backtester runs its own copy of
        the algorithm; that copy must still produce the same answer the
        strategy would for the same input."""
        df = _make_ohlcv(n_bars=120, seed=7)

        strat = self.AgentZeroBot(use_ai=False)
        live_poc = strat._compute_poc(df)

        # No strategy attached - fallback path runs.
        bt = self.AgentZeroBacktester()
        self.assertIsNone(bt._strategy)
        fallback_poc = bt._compute_poc(df)

        self.assertAlmostEqual(
            live_poc, fallback_poc, places=8,
            msg="backtester fallback POC drifted from live strategy POC",
        )

    def test_compute_poc_returns_zero_on_thin_data(self):
        """Both implementations refuse to compute POC on <12 bars."""
        thin_df = _make_ohlcv(n_bars=8)
        strat = self.AgentZeroBot(use_ai=False)
        bt = self.AgentZeroBacktester()
        bt._strategy = strat
        self.assertEqual(strat._compute_poc(thin_df), 0.0)
        self.assertEqual(bt._compute_poc(thin_df), 0.0)

    # ---- Gate behaviour --------------------------------------------------

    def test_gate_keeps_buy_above_poc(self):
        strat = self.AgentZeroBot(use_ai=False)
        self.assertTrue(strat._poc_aligned("BUY", entry_price=1.1050, poc=1.1000))

    def test_gate_rejects_buy_below_poc(self):
        strat = self.AgentZeroBot(use_ai=False)
        self.assertFalse(strat._poc_aligned("BUY", entry_price=1.0950, poc=1.1000))

    def test_gate_keeps_sell_below_poc(self):
        strat = self.AgentZeroBot(use_ai=False)
        self.assertTrue(strat._poc_aligned("SELL", entry_price=1.0950, poc=1.1000))

    def test_gate_rejects_sell_above_poc(self):
        strat = self.AgentZeroBot(use_ai=False)
        self.assertFalse(strat._poc_aligned("SELL", entry_price=1.1050, poc=1.1000))

    def test_gate_passthrough_on_missing_poc(self):
        """POC == 0 means we couldn't compute it (thin data, degenerate
        window). The strategy must NOT penalise the signal in that case
        - this matches the backtester's pass-through-no-POC behaviour."""
        strat = self.AgentZeroBot(use_ai=False)
        self.assertTrue(strat._poc_aligned("BUY",  1.1050, poc=0.0))
        self.assertTrue(strat._poc_aligned("SELL", 1.0950, poc=0.0))

    def test_gate_passthrough_on_missing_entry(self):
        strat = self.AgentZeroBot(use_ai=False)
        self.assertTrue(strat._poc_aligned("BUY",  0.0, poc=1.1000))
        self.assertTrue(strat._poc_aligned("SELL", 0.0, poc=1.1000))

    # ---- Backtester _apply_vp_filter still keeps the right signals -------
    # (Tests below moved to TestPOCFilterBacktesterStandalone so they run
    # on every host — they don't need MT5 / AgentZeroBot.)


class TestPOCFilterBacktesterStandalone(unittest.TestCase):
    """Tests that exercise the backtester's POC filter without needing the
    live strategy. These run on every CI host (Linux + Windows)."""

    def test_apply_vp_filter_keeps_aligned_signals(self):
        """BUY above POC and SELL below POC are kept; misaligned ones are
        dropped. This is the function used during backtest replay."""
        from odl.backtest import AgentZeroBacktester, Signal

        bt = AgentZeroBacktester()
        ts = datetime(2026, 1, 5, 8, 0, tzinfo=timezone.utc)

        def _sig(side, entry, poc):
            return Signal(
                ts=ts, symbol="EURUSD", signal=side,
                source="CHoCH-BUY@PDL" if side == "BUY" else "CHoCH-SELL@PDH",
                confidence=0.85, percentage=85, quality="good",
                entry_price=entry, stop_loss=entry - 0.001, take_profit=entry + 0.002,
                rr_ratio=2.0, poc=poc,
            )

        signals = [
            _sig("BUY",  1.1050, 1.1000),  # aligned -> keep
            _sig("BUY",  1.0950, 1.1000),  # below POC -> drop
            _sig("SELL", 1.0950, 1.1000),  # aligned -> keep
            _sig("SELL", 1.1050, 1.1000),  # above POC -> drop
        ]
        kept, stats = bt._apply_vp_filter(signals)

        self.assertEqual(stats["input"], 4)
        self.assertEqual(stats["kept"], 2)
        self.assertEqual(stats["dropped_buy_below_poc"], 1)
        self.assertEqual(stats["dropped_sell_above_poc"], 1)
        self.assertEqual({s.signal for s in kept}, {"BUY", "SELL"})

    def test_apply_vp_filter_passes_through_when_poc_missing(self):
        """When a signal has no POC value (poc=0), the filter should let
        it through unchanged — we don't penalise the strategy for our own
        data hiccups. This matches AgentZeroBot._poc_aligned's behaviour."""
        from odl.backtest import AgentZeroBacktester, Signal

        bt = AgentZeroBacktester()
        ts = datetime(2026, 1, 5, 8, 0, tzinfo=timezone.utc)
        signals = [
            Signal(ts=ts, symbol="EURUSD", signal="BUY",
                   source="CHoCH-BUY@PDL", confidence=0.85, percentage=85,
                   quality="good", entry_price=1.1050, stop_loss=1.10,
                   take_profit=1.11, rr_ratio=2.0, poc=0.0),
        ]
        kept, stats = bt._apply_vp_filter(signals)
        self.assertEqual(len(kept), 1)
        self.assertEqual(stats["passthrough_no_poc"], 1)
        self.assertEqual(stats["dropped"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
