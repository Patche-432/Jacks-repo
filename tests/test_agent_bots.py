"""
Tests for the May 2026 agent refactor:

  * 4 deterministic pair bots (EURUSD / GBPUSD / GBPJPY / EURJPY)
  * 1 LLM-backed orchestrator (Agent 0)

The bots are the bulk of the surface area and need real coverage —
they are the new code path exercised on every poll cycle. The
orchestrator is mocked at the OllamaClient level so we don't need a
live Ollama server in CI.

Run from the repo root:
    pytest tests/test_agent_bots.py -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ai_agent import (
    EURJPYBot,
    EURUSDBot,
    GBPJPYBot,
    GBPUSDBot,
    Orchestrator,
    PairBotBase,
    PortfolioSnapshot,
    PositionContext,
    get_pair_agent,
    get_orchestrator,
)


# ──────────────────────────── helpers ──────────────────────────────────────


def _ctx(symbol: str = "EURUSD",
         side: str = "BUY",
         entry: float = 1.10000,
         cur_price: float = 1.10000,
         cur_sl: float = 1.09800,
         cur_tp: float = 1.10500,
         profit_pts: float = 0.0,
         peak_pts: float = 0.0,
         atr: float = 0.0010,        # 10 pips ATR for non-JPY
         digits: int = 5,
         trend_intact=None,
         structure_broken=None,
         fresh_structure=None,
         notes: str = "") -> PositionContext:
    """Helper to build a PositionContext with sensible test defaults."""
    return PositionContext(
        symbol=symbol, ticket=12345, side=side,
        entry=entry, cur_price=cur_price,
        cur_sl=cur_sl, cur_tp=cur_tp,
        profit_pts=profit_pts, peak_pts=peak_pts,
        atr=atr, digits=digits,
        trend_intact=trend_intact,
        structure_broken=structure_broken,
        fresh_structure=fresh_structure,
        notes=notes,
    )


# ─────────────────────────── pair bot tests ────────────────────────────────


class TestEURUSDBot(unittest.TestCase):
    """EURUSD: tight, liquid. Tighten at 1 ATR. Close on structure break."""

    def setUp(self):
        self.bot = EURUSDBot()

    def test_holds_when_no_profit(self):
        ctx = _ctx(symbol="EURUSD", profit_pts=0.0, cur_price=1.10000)
        v = self.bot.manage_position(ctx)
        self.assertEqual(v["action"], "hold")

    def test_holds_below_min_atr_profit(self):
        # 5 pips of profit on a 10-pip ATR = 0.5 ATR — below 1.0 threshold.
        ctx = _ctx(symbol="EURUSD",
                   entry=1.10000, cur_price=1.10050,
                   profit_pts=50.0)   # 50 points = 5 pips on 5-digit
        v = self.bot.manage_position(ctx)
        self.assertEqual(v["action"], "hold")

    def test_tightens_at_one_atr_profit(self):
        # 100 points = 10 pips = 1 ATR.
        ctx = _ctx(symbol="EURUSD",
                   entry=1.10000, cur_price=1.10100,
                   profit_pts=100.0,
                   cur_sl=1.09800)
        v = self.bot.manage_position(ctx)
        self.assertEqual(v["action"], "move_sl")
        # New SL = 1.10100 - 1.0 ATR = 1.10000 — must be tighter than 1.09800.
        self.assertGreater(v["new_sl"], ctx.cur_sl)
        self.assertLess(v["new_sl"], ctx.cur_price)

    def test_close_on_structure_broken(self):
        ctx = _ctx(symbol="EURUSD", structure_broken=True)
        v = self.bot.manage_position(ctx)
        self.assertEqual(v["action"], "close")
        self.assertIn("structure", v["reason"].lower())

    def test_does_not_close_on_trend_lost(self):
        # EURUSD does NOT close on trend_intact=False (only structure break).
        ctx = _ctx(symbol="EURUSD", trend_intact=False)
        v = self.bot.manage_position(ctx)
        self.assertEqual(v["action"], "hold")


class TestGBPUSDBot(unittest.TestCase):
    """Cable: volatile. Wider 1.5 ATR buffer to tolerate stop hunts."""

    def setUp(self):
        self.bot = GBPUSDBot()

    def test_does_not_tighten_at_one_atr(self):
        # 1.0 ATR profit — Cable wants 1.5 ATR before tightening.
        ctx = _ctx(symbol="GBPUSD",
                   entry=1.27000, cur_price=1.27100,
                   profit_pts=100.0)   # 1 ATR
        v = self.bot.manage_position(ctx)
        self.assertEqual(v["action"], "hold")

    def test_tightens_at_one_point_five_atr(self):
        # 1.5 ATR profit — Cable's threshold.
        ctx = _ctx(symbol="GBPUSD",
                   entry=1.27000, cur_price=1.27150,
                   profit_pts=150.0,
                   cur_sl=1.26700)   # 30 pips behind entry
        v = self.bot.manage_position(ctx)
        self.assertEqual(v["action"], "move_sl")


class TestGBPJPYBot(unittest.TestCase):
    """Highest-volatility. JPY pair => 3 digits. 1.5 ATR threshold + trail."""

    def setUp(self):
        self.bot = GBPJPYBot()

    def test_pip_size_jpy_correct(self):
        # JPY pair: digits=3, so 1 pip = 0.01, 1 point = 0.001.
        # ATR of 0.50 (50 pips) is realistic for GBPJPY.
        # 1 ATR profit = 50 pips = 500 points.
        # We want the bot to NOT tighten yet — needs 1.5 ATR.
        ctx = _ctx(symbol="GBPJPY",
                   entry=190.000, cur_price=190.500,
                   profit_pts=500.0,    # 1 ATR
                   atr=0.500, digits=3,
                   cur_sl=189.500)
        v = self.bot.manage_position(ctx)
        self.assertEqual(v["action"], "hold")

    def test_tightens_at_one_point_five_atr_jpy(self):
        # 1.5 ATR profit on JPY pair.
        ctx = _ctx(symbol="GBPJPY",
                   entry=190.000, cur_price=190.750,
                   profit_pts=750.0,
                   atr=0.500, digits=3,
                   cur_sl=189.500)
        v = self.bot.manage_position(ctx)
        self.assertEqual(v["action"], "move_sl")
        # Trail = 1.5 ATR = 0.75 below current price = 190.000.
        self.assertAlmostEqual(v["new_sl"], 190.000, places=3)


class TestEURJPYBot(unittest.TestCase):
    """Risk-sentiment driven. CLOSE_ON_TREND_BROKEN is the differentiator."""

    def setUp(self):
        self.bot = EURJPYBot()

    def test_closes_on_trend_lost_even_without_structure_break(self):
        # The whole point of EURJPY's rule: close on trend_intact=False
        # without waiting for structure_broken=True.
        ctx = _ctx(symbol="EURJPY",
                   atr=0.300, digits=3,
                   entry=160.000, cur_price=160.100,
                   profit_pts=100.0,
                   trend_intact=False,        # the trigger
                   structure_broken=None)     # explicitly NOT set
        v = self.bot.manage_position(ctx)
        self.assertEqual(v["action"], "close")
        self.assertIn("trend", v["reason"].lower())

    def test_close_on_structure_break_still_works(self):
        ctx = _ctx(symbol="EURJPY",
                   atr=0.300, digits=3,
                   structure_broken=True)
        v = self.bot.manage_position(ctx)
        self.assertEqual(v["action"], "close")


class TestPairBotHardRails(unittest.TestCase):
    """Hard rails enforced regardless of pair: SL must be tighter and on
    the correct side of the current price."""

    def test_buy_sl_must_be_below_price(self):
        bot = EURUSDBot()
        # If TRAIL_ATR_MULT or ATR caused new_sl to land above price,
        # the bot must hold instead of issuing a wrong-side SL.
        # Engineer an impossible state: tiny ATR, price barely above entry,
        # current SL already very close to price.
        ctx = _ctx(symbol="EURUSD",
                   entry=1.10000, cur_price=1.10010,
                   cur_sl=1.10005,             # already nearly at price
                   profit_pts=10.0,            # 1 ATR with atr=0.0001
                   atr=0.0001, digits=5)
        v = bot.manage_position(ctx)
        # Either it holds, or it returns a strictly tighter SL below price.
        if v["action"] == "move_sl":
            self.assertLess(v["new_sl"], ctx.cur_price)
            self.assertGreater(v["new_sl"], ctx.cur_sl)

    def test_sell_sl_must_be_above_price(self):
        bot = EURUSDBot()
        ctx = _ctx(symbol="EURUSD", side="SELL",
                   entry=1.10000, cur_price=1.09900,
                   cur_sl=1.10200, cur_tp=1.09500,
                   profit_pts=100.0,
                   atr=0.0010, digits=5)
        v = bot.manage_position(ctx)
        if v["action"] == "move_sl":
            self.assertGreater(v["new_sl"], ctx.cur_price)
            self.assertLess(v["new_sl"], ctx.cur_sl)

    def test_atr_zero_does_not_crash(self):
        # If ATR is zero (data hiccup), the bot should hold rather than
        # divide-by-zero.
        bot = EURUSDBot()
        ctx = _ctx(symbol="EURUSD", atr=0.0, profit_pts=100.0)
        v = bot.manage_position(ctx)
        self.assertEqual(v["action"], "hold")


class TestPairBotRegistry(unittest.TestCase):
    """get_pair_agent() routing must hit the right class for each symbol."""

    def test_routes_to_correct_classes(self):
        self.assertIsInstance(get_pair_agent("EURUSD"), EURUSDBot)
        self.assertIsInstance(get_pair_agent("GBPUSD"), GBPUSDBot)
        self.assertIsInstance(get_pair_agent("GBPJPY"), GBPJPYBot)
        self.assertIsInstance(get_pair_agent("EURJPY"), EURJPYBot)

    def test_unknown_symbol_falls_back_gracefully(self):
        # USDJPY isn't in the registry; should fall back to PairBotBase.
        bot = get_pair_agent("USDJPY")
        self.assertIsInstance(bot, PairBotBase)
        self.assertEqual(bot.SYMBOL, "USDJPY")

    def test_returns_singleton(self):
        a = get_pair_agent("EURUSD")
        b = get_pair_agent("EURUSD")
        self.assertIs(a, b)


# ─────────────────────────── orchestrator tests ────────────────────────────


class _FakeClient:
    """OllamaClient stub for orchestrator tests."""
    def __init__(self, response: str):
        self._response = response
        self.last_user_msg = None

    def chat(self, system, user, **kwargs):
        self.last_user_msg = user
        return self._response


class TestOrchestrator(unittest.TestCase):

    def _verdicts(self):
        return [
            {"symbol": "EURUSD", "ticket": 1, "action": "hold",
             "new_sl": None, "reason": "EURUSD bot: hold (profit 0.50 ATR)",
             "agent": "EURUSD_bot"},
            {"symbol": "GBPJPY", "ticket": 2, "action": "move_sl",
             "new_sl": 190.50,
             "reason": "GBPJPY bot: profit 1.60 ATR — trail SL 1.5 ATR",
             "agent": "GBPJPY_bot"},
        ]

    def test_empty_verdicts_returns_empty(self):
        orch = Orchestrator(client=_FakeClient(""))
        self.assertEqual(orch.orchestrate([]), [])

    def test_approve_all_when_llm_returns_nothing(self):
        # Empty / unparseable response → approve all verdicts as-is.
        orch = Orchestrator(client=_FakeClient(""))
        result = orch.orchestrate(self._verdicts())
        self.assertEqual(len(result), 2)
        self.assertEqual([v["action"] for v in result], ["hold", "move_sl"])

    def test_approve_all_when_transport_errors(self):
        class _ErrorClient:
            def chat(self, *a, **kw):
                raise RuntimeError("ollama down")
        orch = Orchestrator(client=_ErrorClient())
        result = orch.orchestrate(self._verdicts())
        self.assertEqual(len(result), 2)

    def test_veto_drops_verdict_to_hold(self):
        response = (
            '{"decisions": [{"ticket": 2, "verdict": "veto", '
            '"reason": "portfolio at limit"}]}'
        )
        orch = Orchestrator(client=_FakeClient(response))
        result = orch.orchestrate(self._verdicts())
        # Find ticket 2 in the result.
        ticket2 = next(v for v in result if v["ticket"] == 2)
        self.assertEqual(ticket2["action"], "hold")
        self.assertIn("VETO", ticket2["orchestrator_reason"])

    def test_override_close_turns_hold_into_close(self):
        response = (
            '{"decisions": [{"ticket": 1, "verdict": "override_close", '
            '"reason": "approaching daily loss limit"}]}'
        )
        orch = Orchestrator(client=_FakeClient(response))
        result = orch.orchestrate(self._verdicts())
        ticket1 = next(v for v in result if v["ticket"] == 1)
        self.assertEqual(ticket1["action"], "close")
        self.assertIn("OVERRIDE", ticket1["orchestrator_reason"])

    def test_override_close_does_not_touch_existing_close(self):
        # If the bot already wants to close, override_close is a no-op (the
        # action is already close). We only convert hold → close.
        verdicts = [
            {"symbol": "EURUSD", "ticket": 1, "action": "close",
             "new_sl": None, "reason": "structure break", "agent": "x"}
        ]
        response = ('{"decisions": [{"ticket": 1, "verdict": "override_close", '
                    '"reason": "redundant"}]}')
        orch = Orchestrator(client=_FakeClient(response))
        result = orch.orchestrate(verdicts)
        self.assertEqual(result[0]["action"], "close")

    def test_unmentioned_tickets_default_to_approve(self):
        # LLM only mentions ticket 1. Ticket 2 should pass through unchanged.
        response = (
            '{"decisions": [{"ticket": 1, "verdict": "veto", "reason": "x"}]}'
        )
        orch = Orchestrator(client=_FakeClient(response))
        result = orch.orchestrate(self._verdicts())
        ticket2 = next(v for v in result if v["ticket"] == 2)
        self.assertEqual(ticket2["action"], "move_sl")
        # With upgrade 8, approved verdicts now carry reason_code=ORC_APPROVE.
        # Verify it was approved (not vetoed or overridden).
        self.assertEqual(ticket2.get("reason_code"), "ORC_APPROVE")
        # orchestrator_reason is empty string for approvals, not absent.
        self.assertNotIn("VETO",     ticket2.get("orchestrator_reason", ""))
        self.assertNotIn("OVERRIDE", ticket2.get("orchestrator_reason", ""))

    def test_portfolio_summary_appears_in_user_message(self):
        # The LLM must see the portfolio state. Sanity-check the summary
        # actually makes it into the prompt.
        client = _FakeClient('{"decisions": []}')
        orch = Orchestrator(client=client)
        snap = PortfolioSnapshot(
            equity=10000.0, balance=10100.0, daily_pl=-100.0,
            daily_loss_limit=200.0, open_positions_count=2,
            max_positions_total=4,
        )
        orch.orchestrate(self._verdicts(), portfolio=snap)
        self.assertIn("equity=10000", client.last_user_msg)
        self.assertIn("daily_pl=-100", client.last_user_msg)


class TestAgentBackendKillSwitch(unittest.TestCase):
    """AI_BACKEND env var controls auto-trading. The bots are deterministic
    and don't need Ollama, but the orchestrator does — so the kill-switch
    is still meaningful."""

    def test_default_enabled(self):
        from ai_agent import agent_backend_enabled
        with patch.dict("os.environ", {}, clear=False):
            # Make sure AI_BACKEND isn't set.
            import os as _os
            _os.environ.pop("AI_BACKEND", None)
            self.assertTrue(agent_backend_enabled())

    def test_off_disables(self):
        from ai_agent import agent_backend_enabled
        for val in ("off", "none", "disabled", "0", "false"):
            with patch.dict("os.environ", {"AI_BACKEND": val}):
                self.assertFalse(
                    agent_backend_enabled(),
                    f"AI_BACKEND={val!r} should disable but didn't",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
