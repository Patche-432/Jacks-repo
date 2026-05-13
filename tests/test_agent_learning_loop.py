"""
End-to-end test for the agent learning loop pipeline.

What this validates:
  1. AgentLearningLoop loads insights from a JSON file
  2. apply_learned_weights() adjusts confidence when conditions match
  3. Returns base_confidence unchanged when no insights exist for a pair
  4. Returns base_confidence unchanged when insights are below threshold
  5. Auto-reloads when the insights file's mtime changes (this is the
     critical bit — the bot subprocess relies on this to pick up new
     backtests without a restart)
  6. backtest's _save_insights_for_pair writes the right shape and
     atomically updates the file
  7. ai_pro module imports cleanly with the learning-loop hook in place

Run from the repo root:
    python -m pytest tests/test_agent_learning_loop.py -v
or as a script:
    python tests/test_agent_learning_loop.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

# Make the repo root importable when run as a script.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agent_learning_loop import AgentLearningLoop  # noqa: E402


# ─────────────────────────── helpers ───────────────────────────────────────

def _write_insights(path: Path, payload: dict) -> None:
    """Write insights JSON with the same atomicity discipline the backtester uses."""
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _bump_mtime(path: Path) -> None:
    """Force the mtime forward so the watcher detects a change.

    On fast filesystems two writes inside the same second can produce the
    same mtime, which would defeat the auto-reload logic. Sleep just over
    one second to guarantee a strictly greater mtime.
    """
    time.sleep(1.05)


# ─────────────────────────── tests ─────────────────────────────────────────

class TestAgentLearningLoop(unittest.TestCase):
    """Unit tests for the live learning-loop adjustment logic."""

    def setUp(self):
        # Each test gets its own scratch directory so they don't interfere.
        import tempfile
        self.tmpdir = Path(tempfile.mkdtemp(prefix="learning_loop_test_"))
        self.insights_path = self.tmpdir / "backtest_insights.json"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # ── No insights file ──────────────────────────────────────────────

    def test_returns_base_when_no_insights_file(self):
        loop = AgentLearningLoop(root_path=str(self.tmpdir))
        self.assertFalse(loop.has_insights())
        # Should pass through base_confidence unchanged.
        out = loop.apply_learned_weights(
            "EURUSD", base_confidence=85,
            current_price=1.10, entry=1.10, sl=1.095,
        )
        self.assertEqual(out, 85)

    def test_returns_base_when_pair_missing(self):
        _write_insights(self.insights_path,
                        {"GBPJPY": {"hour_of_day": 0.30}})
        loop = AgentLearningLoop(root_path=str(self.tmpdir))
        # Insights exist for GBPJPY only — EURUSD should be untouched.
        out = loop.apply_learned_weights(
            "EURUSD", base_confidence=85,
            current_price=1.10, entry=1.10, sl=1.095,
        )
        self.assertEqual(out, 85)

    # ── Hour-of-day boost ─────────────────────────────────────────────

    def test_hour_of_day_boost_during_optimal_hours(self):
        """If hour_of_day importance is high AND the current hour is in the
        optimal trading window, confidence should be boosted."""
        _write_insights(self.insights_path,
                        {"GBPJPY": {"hour_of_day": 0.40}})
        loop = AgentLearningLoop(root_path=str(self.tmpdir))

        # Pretend "now" is 10:00 UTC — squarely inside London/NY overlap.
        class FakeDT:
            @staticmethod
            def utcnow():
                from datetime import datetime
                return datetime(2024, 1, 15, 10, 0, 0)

        with patch("agent_learning_loop.datetime", FakeDT):
            out = loop.apply_learned_weights(
                "GBPJPY", base_confidence=85,
                current_price=190.00, entry=190.00, sl=189.50,
            )
        # 0.40 * 12 = 4.8 → +5 after rounding. 85 + 5 = 90.
        self.assertGreater(out, 85, "should boost during optimal hours")
        self.assertLessEqual(out, 95, "must respect 95 ceiling")

    def test_hour_of_day_no_boost_outside_optimal(self):
        _write_insights(self.insights_path,
                        {"GBPJPY": {"hour_of_day": 0.40}})
        loop = AgentLearningLoop(root_path=str(self.tmpdir))

        class FakeDT:
            @staticmethod
            def utcnow():
                from datetime import datetime
                return datetime(2024, 1, 15, 3, 0, 0)  # 3am UTC, off-hours

        with patch("agent_learning_loop.datetime", FakeDT):
            out = loop.apply_learned_weights(
                "GBPJPY", base_confidence=85,
                current_price=190.00, entry=190.00, sl=189.50,
            )
        self.assertEqual(out, 85)

    # ── Risk-pips and POC alignment ───────────────────────────────────

    def test_risk_pips_optimal_range_boosts(self):
        _write_insights(self.insights_path,
                        {"EURUSD": {"risk_pips": 0.30}})
        loop = AgentLearningLoop(root_path=str(self.tmpdir))
        # 50 pip risk → in the 40-100 sweet spot for non-JPY.
        out = loop.apply_learned_weights(
            "EURUSD", base_confidence=70,
            current_price=1.10, entry=1.10, sl=1.0950,
        )
        # 0.30 * 8 = 2.4 → +2 rounded. Expect a bump.
        self.assertGreater(out, 70)

    def test_risk_pips_too_tight_penalised(self):
        _write_insights(self.insights_path,
                        {"EURUSD": {"risk_pips": 0.30}})
        loop = AgentLearningLoop(root_path=str(self.tmpdir))
        # 10 pip risk → below 40 → penalty branch.
        out = loop.apply_learned_weights(
            "EURUSD", base_confidence=80,
            current_price=1.10, entry=1.10, sl=1.0990,
        )
        self.assertLess(out, 80, "tight stops should be penalised")

    def test_poc_alignment_boost(self):
        _write_insights(self.insights_path,
                        {"GBPJPY": {"dist_to_poc_pips": 0.25}})
        loop = AgentLearningLoop(root_path=str(self.tmpdir))
        # JPY pip = 0.01. Distance 30 pips → < 50 → boost.
        out = loop.apply_learned_weights(
            "GBPJPY", base_confidence=85,
            current_price=190.30, entry=190.30, sl=189.80,
            poc=190.00,
        )
        self.assertGreater(out, 85)

    # ── Below-threshold importance is ignored ─────────────────────────

    def test_low_importance_does_not_boost(self):
        # 0.10 < 0.15 threshold — feature is treated as noise.
        _write_insights(self.insights_path,
                        {"EURUSD": {"hour_of_day": 0.10}})
        loop = AgentLearningLoop(root_path=str(self.tmpdir))
        out = loop.apply_learned_weights(
            "EURUSD", base_confidence=80,
            current_price=1.10, entry=1.10, sl=1.095,
        )
        self.assertEqual(out, 80)

    # ── Clamping ──────────────────────────────────────────────────────

    def test_output_clamped_to_95(self):
        _write_insights(self.insights_path,
                        {"GBPJPY": {"hour_of_day": 0.99,
                                    "dist_to_poc_pips": 0.99,
                                    "risk_pips":        0.99}})
        loop = AgentLearningLoop(root_path=str(self.tmpdir))

        class FakeDT:
            @staticmethod
            def utcnow():
                from datetime import datetime
                return datetime(2024, 1, 15, 10, 0, 0)

        with patch("agent_learning_loop.datetime", FakeDT):
            out = loop.apply_learned_weights(
                "GBPJPY", base_confidence=92,
                current_price=190.30, entry=190.30, sl=189.80,
                poc=190.00,
            )
        self.assertLessEqual(out, 95)

    def test_output_clamped_to_60(self):
        # Penalty branch only fires for tight risk + risk_pips importance,
        # but base 60 is the floor regardless.
        _write_insights(self.insights_path,
                        {"EURUSD": {"risk_pips": 0.99}})
        loop = AgentLearningLoop(root_path=str(self.tmpdir))
        out = loop.apply_learned_weights(
            "EURUSD", base_confidence=60,
            current_price=1.10, entry=1.10, sl=1.0995,
        )
        self.assertGreaterEqual(out, 60)

    # ── Auto-reload (the bit that lets the live bot pick up new backtests) ──

    def test_auto_reload_on_mtime_change(self):
        """The bot polls signals every poll_secs while the dashboard runs
        backtests in another thread. When a backtest writes a fresh
        insights file, the next signal review must see the new weights
        without a bot restart."""
        # Initial state — no insights.
        loop = AgentLearningLoop(root_path=str(self.tmpdir))
        self.assertFalse(loop.has_insights())

        first = loop.apply_learned_weights(
            "GBPJPY", base_confidence=85,
            current_price=190.00, entry=190.00, sl=189.50,
        )
        self.assertEqual(first, 85)

        # Simulate a backtest run finishing — write insights with a STRONG
        # hour-of-day weight, then bump mtime so the watcher notices.
        _bump_mtime(self.insights_path)
        _write_insights(self.insights_path,
                        {"GBPJPY": {"hour_of_day": 0.50}})

        class FakeDT:
            @staticmethod
            def utcnow():
                from datetime import datetime
                return datetime(2024, 1, 15, 10, 0, 0)

        with patch("agent_learning_loop.datetime", FakeDT):
            second = loop.apply_learned_weights(
                "GBPJPY", base_confidence=85,
                current_price=190.00, entry=190.00, sl=189.50,
            )

        self.assertGreater(second, first,
                           "auto-reload failed — bot would not see new backtest")
        self.assertTrue(loop.has_insights())
        self.assertIn("GBPJPY", loop.get_insights_for_pair("GBPJPY") and
                      loop.insights or {})

    def test_auto_clear_when_file_deleted(self):
        """If the user deletes backtest_insights.json the loop should drop
        its in-memory insights so the bot stops applying stale weights."""
        _write_insights(self.insights_path,
                        {"EURUSD": {"hour_of_day": 0.40}})
        loop = AgentLearningLoop(root_path=str(self.tmpdir))
        self.assertTrue(loop.has_insights())

        self.insights_path.unlink()
        # apply_learned_weights() triggers the file check.
        loop.apply_learned_weights(
            "EURUSD", base_confidence=80,
            current_price=1.10, entry=1.10, sl=1.095,
        )
        self.assertFalse(loop.has_insights())


class TestBacktestInsightsPersistence(unittest.TestCase):
    """Verifies the backtester's _save_insights_for_pair writes the shape
    AgentLearningLoop expects, and is multi-pair safe."""

    def setUp(self):
        import tempfile
        self.tmpdir = Path(tempfile.mkdtemp(prefix="bt_insights_test_"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_backtester_with_repo_at(self, repo_root: Path):
        """Build a minimal AgentZeroBacktester with __file__ patched so its
        save_insights writes into our temp directory (it computes the
        repo root as parent of `odl/`)."""
        # We need backtest.py importable; add real repo to sys.path.
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))

        # Stub the AI_PRO import flag so backtest.py loads even without
        # a live MT5 environment.
        from odl import backtest as bt_mod  # type: ignore

        class _StubBacktester(bt_mod.AgentZeroBacktester):
            def __init__(self, fake_repo_root: Path):
                # Skip parent __init__ — we don't need MT5 plumbing.
                self.symbols = ["EURUSD"]
                self.lookback_bars = 200
                self.max_trade_duration_bars = 96
                self.risk_per_trade_pips = 50.0
                self.lot_size = 1.0
                self.spread_pips_override = None
                self.intrabar_policy = "conservative"
                self._strategy = None
                self._mt5 = None
                self.signals = []
                self.trades = []
                self.candle_history = {}
                self._fake_repo_root = fake_repo_root

            def _save_insights_for_pair(self, symbol, feature_importance=None,
                                        results=None, trades=None):
                # Patch repo root by monkey-patching Path lookup. Easier:
                # reimplement using the parent class but with our root.
                old_file = bt_mod.__file__
                fake_pkg = self._fake_repo_root / "odl"
                fake_pkg.mkdir(parents=True, exist_ok=True)
                # Temporarily rebind __file__ on the module so the
                # `Path(__file__).resolve().parent.parent` in the parent
                # method points at our temp root.
                bt_mod.__file__ = str(fake_pkg / "backtest.py")
                try:
                    super()._save_insights_for_pair(
                        symbol,
                        feature_importance=feature_importance,
                        results=results,
                        trades=trades,
                    )
                finally:
                    bt_mod.__file__ = old_file

        return _StubBacktester(repo_root)

    # ── Helpers ───────────────────────────────────────────────────────────
    @staticmethod
    def _make_fake_trades(symbol, n_wins=20, n_losses=15,
                          avg_win_pips=40.0, avg_loss_pips=-25.0,
                          atr_pips=20.0):
        """Create a synthetic list of decided trades the tuner can chew on.
        Avoids needing MT5 + live data."""
        from datetime import datetime as _dt
        from odl.backtest import Signal, Trade, get_pip_value

        pip = get_pip_value(symbol)
        atr_price = atr_pips * pip
        trades = []
        base_ts = _dt(2024, 1, 15, 9, 0, 0)

        def _mk(side, pnl_pips, idx):
            entry = 1.10 if not symbol.endswith("JPY") else 190.00
            sl_dist = 50 * pip
            sig = Signal(
                ts=base_ts,
                symbol=symbol,
                signal=side,
                source=f"CHoCH-{side}@PDL",
                confidence=0.75,
                percentage=75,
                quality="good",
                entry_price=entry,
                stop_loss=entry - sl_dist if side == "BUY" else entry + sl_dist,
                take_profit=entry + sl_dist * 2 if side == "BUY" else entry - sl_dist * 2,
                rr_ratio=2.0,
                atr=atr_price,
            )
            t = Trade(signal=sig, entry_price=entry, entry_time=base_ts)
            t.exit_time = base_ts
            t.exit_price = entry + pnl_pips * pip if side == "BUY" \
                else entry - pnl_pips * pip
            t.profit_pips = pnl_pips
            t.profit = pnl_pips * 10.0  # rough $ figure, unused by tuner
            t.outcome = "WIN" if pnl_pips > 0 else ("LOSS" if pnl_pips < 0 else "BE")
            t.exit_reason = "tp" if pnl_pips > 0 else "sl"
            t.duration_minutes = 90
            t.rr_achieved = abs(pnl_pips / 25.0) if pnl_pips else 0.0
            return t

        for i in range(n_wins):
            trades.append(_mk("BUY" if i % 2 == 0 else "SELL", avg_win_pips, i))
        for i in range(n_losses):
            trades.append(_mk("BUY" if i % 2 == 0 else "SELL", avg_loss_pips, i))
        return trades

    @staticmethod
    def _make_fake_results(trades, win_rate=None, avg_dur_min=120.0,
                           timeout_count=0):
        wins = [t for t in trades if t.outcome == "WIN"]
        losses = [t for t in trades if t.outcome == "LOSS"]
        decided = wins + losses
        wr = win_rate if win_rate is not None else len(wins) / max(1, len(decided))
        avg_w = (sum(t.profit_pips for t in wins) / len(wins)) if wins else 0.0
        avg_l = (sum(t.profit_pips for t in losses) / len(losses)) if losses else 0.0
        rr = (sum(t.rr_achieved for t in trades) /
              max(1, sum(1 for t in trades if t.rr_achieved > 0)))
        return {
            "total_trades":         len(trades),
            "wins":                 len(wins),
            "losses":               len(losses),
            "win_rate":             wr,
            "avg_win_pips":         round(avg_w, 1),
            "avg_loss_pips":        round(avg_l, 1),
            "avg_rr_achieved":      round(rr, 2),
            "avg_duration_minutes": avg_dur_min,
            "by_exit_reason":       {"timeout": {"count": timeout_count}},
        }

    # ── Schema tests ─────────────────────────────────────────────────────────

    def test_save_writes_new_structured_shape(self):
        """After the self-tuning rewrite, each pair entry is a dict with
        keys: importances, tuned_params, backtest_at, trade_count, win_rate.
        """
        bt = self._make_backtester_with_repo_at(self.tmpdir)
        trades  = self._make_fake_trades("GBPJPY")
        results = self._make_fake_results(trades)
        fi = {
            "n_trades": len(trades),
            "lift_vs_random": 0.05,
            "importances": [
                {"feature": "hour_of_day",      "importance": 0.32},
                {"feature": "dist_to_poc_pips", "importance": 0.23},
                {"feature": "risk_pips",        "importance": 0.17},
                {"feature": "is_jpy",           "importance": 0.001},  # noise
            ],
        }
        bt._save_insights_for_pair("GBPJPY", feature_importance=fi,
                                   results=results, trades=trades)

        path = self.tmpdir / "backtest_insights.json"
        self.assertTrue(path.exists(), "insights file was not written")
        data = json.loads(path.read_text())
        self.assertIn("GBPJPY", data)
        block = data["GBPJPY"]

        # New structured keys present.
        for k in ("importances", "tuned_params", "backtest_at",
                  "trade_count", "win_rate"):
            self.assertIn(k, block, f"missing top-level key: {k}")

        # Importances: noise filtered, significant kept.
        self.assertNotIn("is_jpy", block["importances"])
        self.assertEqual(block["importances"]["hour_of_day"], 0.32)

        # Tuned params: all 7 keys present and inside their bounds.
        from odl.backtest import AgentZeroBacktester as _BT
        for key, (lo, hi) in _BT._TUNED_PARAM_BOUNDS.items():
            self.assertIn(key, block["tuned_params"], f"missing tuned key: {key}")
            v = block["tuned_params"][key]
            self.assertGreaterEqual(v, lo, f"{key}={v} below {lo}")
            self.assertLessEqual(v, hi, f"{key}={v} above {hi}")

        # Provenance.
        self.assertEqual(block["trade_count"], len(trades))
        self.assertAlmostEqual(block["win_rate"], results["win_rate"], places=4)

    def test_save_writes_tuned_params_without_ml(self):
        """sklearn unavailable / model untrained → we still want SL/TP
        scaled from raw stats. Critical for environments where sklearn
        isn't installed."""
        bt = self._make_backtester_with_repo_at(self.tmpdir)
        trades  = self._make_fake_trades("EURUSD", avg_win_pips=30, avg_loss_pips=-20)
        results = self._make_fake_results(trades)
        bt._save_insights_for_pair("EURUSD", feature_importance=None,
                                   results=results, trades=trades)

        path = self.tmpdir / "backtest_insights.json"
        self.assertTrue(path.exists())
        block = json.loads(path.read_text())["EURUSD"]
        self.assertIn("tuned_params", block)
        self.assertNotIn("importances", block,
                         "no ML → should not write importances")

    def test_save_skips_when_no_data(self):
        """Tiny / empty backtest → nothing to learn from → no write."""
        bt = self._make_backtester_with_repo_at(self.tmpdir)
        # 2 trades is below the n>=5 threshold for tuning.
        trades = self._make_fake_trades("EURUSD", n_wins=1, n_losses=1)
        results = self._make_fake_results(trades)
        bt._save_insights_for_pair("EURUSD", feature_importance=None,
                                   results=results, trades=trades)
        # No importances + insufficient trades for tuning → skip.
        self.assertFalse((self.tmpdir / "backtest_insights.json").exists())

    def test_save_preserves_other_pairs(self):
        """Running EURUSD must not erase a previous GBPJPY backtest."""
        path = self.tmpdir / "backtest_insights.json"
        # Pre-seed with another pair in the new structured shape.
        path.write_text(json.dumps({
            "EURUSD": {
                "importances": {"hour_of_day": 0.40},
                "tuned_params": {"sl_atr_mult": 2.3, "tp_atr_mult": 5.0,
                                 "atr_tolerance_mult": 1.5,
                                 "partial_close_rr": 1.0,
                                 "be_buffer_pips": 1.0,
                                 "min_atr_to_tighten": 1.0,
                                 "trail_atr_mult": 1.0},
                "backtest_at": "2024-01-01T00:00:00+00:00",
            }
        }))

        bt = self._make_backtester_with_repo_at(self.tmpdir)
        trades  = self._make_fake_trades("GBPJPY")
        results = self._make_fake_results(trades)
        fi = {"importances": [
            {"feature": "hour_of_day", "importance": 0.31}
        ]}
        bt._save_insights_for_pair("GBPJPY", feature_importance=fi,
                                   results=results, trades=trades)

        data = json.loads(path.read_text())
        self.assertIn("EURUSD", data, "EURUSD entry got wiped")
        self.assertIn("GBPJPY", data)
        # EURUSD's tuned_params survive untouched.
        self.assertEqual(data["EURUSD"]["tuned_params"]["sl_atr_mult"], 2.3)

    def test_legacy_flat_format_migrates_transparently(self):
        """Existing files written by the old flat-importances version
        should be migrated to the new wrapped shape on next write."""
        path = self.tmpdir / "backtest_insights.json"
        # Old flat shape: {symbol: {feature: score}}
        path.write_text(json.dumps({
            "EURUSD": {"hour_of_day": 0.40, "risk_pips": 0.20},
        }))

        bt = self._make_backtester_with_repo_at(self.tmpdir)
        trades  = self._make_fake_trades("GBPJPY")
        results = self._make_fake_results(trades)
        fi = {"importances": [
            {"feature": "hour_of_day", "importance": 0.31}
        ]}
        bt._save_insights_for_pair("GBPJPY", feature_importance=fi,
                                   results=results, trades=trades)

        data = json.loads(path.read_text())
        # EURUSD should have been wrapped into the new shape.
        self.assertIsInstance(data["EURUSD"], dict)
        self.assertIn("importances", data["EURUSD"])
        self.assertEqual(data["EURUSD"]["importances"]["hour_of_day"], 0.40)

    # ── Tuning logic tests ───────────────────────────────────────────────────

    def test_tuning_jpy_be_buffer_wider(self):
        """JPY pairs should get a wider BE buffer than majors because
        their typical spread is 2p vs 1p."""
        bt = self._make_backtester_with_repo_at(self.tmpdir)
        # Same trade stats for both — only the symbol differs.
        for sym in ("GBPJPY", "EURUSD"):
            trades  = self._make_fake_trades(sym)
            results = self._make_fake_results(trades, win_rate=0.5)
            tuned = bt._compute_tuned_params(sym, trades, results, None)
            setattr(self, f"_be_{sym}", tuned["be_buffer_pips"])
        self.assertGreater(getattr(self, "_be_GBPJPY"),
                           getattr(self, "_be_EURUSD"),
                           "JPY BE buffer should be wider than majors'")

    def test_tuning_high_win_rate_tightens_be_buffer(self):
        bt = self._make_backtester_with_repo_at(self.tmpdir)
        trades = self._make_fake_trades("EURUSD", n_wins=40, n_losses=10)
        results_high = self._make_fake_results(trades, win_rate=0.80)
        results_low  = self._make_fake_results(trades, win_rate=0.30)
        tuned_high = bt._compute_tuned_params("EURUSD", trades, results_high, None)
        tuned_low  = bt._compute_tuned_params("EURUSD", trades, results_low, None)
        self.assertLess(tuned_high["be_buffer_pips"],
                        tuned_low["be_buffer_pips"],
                        "high WR should produce a tighter BE buffer")

    def test_tuning_rr_floor_enforced(self):
        """Even when avg_win is small, TP must stay at >=1.5× SL so
        we don't tune ourselves into negative-expectancy territory."""
        bt = self._make_backtester_with_repo_at(self.tmpdir)
        # Trades where wins are trivially small relative to losses.
        trades = self._make_fake_trades("EURUSD",
                                        n_wins=10, n_losses=10,
                                        avg_win_pips=10.0, avg_loss_pips=-50.0)
        results = self._make_fake_results(trades)
        tuned = bt._compute_tuned_params("EURUSD", trades, results, None)
        # tp_atr_mult must be at least 1.5× sl_atr_mult.
        self.assertGreaterEqual(
            tuned["tp_atr_mult"],
            tuned["sl_atr_mult"] * 1.5 - 0.01,  # epsilon for rounding
            "R:R floor not enforced — risky tune",
        )

    def test_tuning_clamped_when_data_extreme(self):
        """A degenerate backtest (loss bigger than 10× ATR) must clamp
        sl_atr_mult to the upper bound, not propagate the absurd value."""
        bt = self._make_backtester_with_repo_at(self.tmpdir)
        trades = self._make_fake_trades("EURUSD",
                                        n_wins=10, n_losses=10,
                                        avg_win_pips=200.0,
                                        avg_loss_pips=-500.0,
                                        atr_pips=20.0)
        results = self._make_fake_results(trades)
        tuned = bt._compute_tuned_params("EURUSD", trades, results, None)
        from odl.backtest import AgentZeroBacktester as _BT
        sl_lo, sl_hi = _BT._TUNED_PARAM_BOUNDS["sl_atr_mult"]
        tp_lo, tp_hi = _BT._TUNED_PARAM_BOUNDS["tp_atr_mult"]
        self.assertLessEqual(tuned["sl_atr_mult"], sl_hi)
        self.assertLessEqual(tuned["tp_atr_mult"], tp_hi)

    def test_tuning_returns_none_with_too_few_trades(self):
        bt = self._make_backtester_with_repo_at(self.tmpdir)
        trades  = self._make_fake_trades("EURUSD", n_wins=2, n_losses=1)
        results = self._make_fake_results(trades)
        out = bt._compute_tuned_params("EURUSD", trades, results, None)
        self.assertIsNone(out, "should refuse to tune on <5 decided trades")

    def test_tuning_timeout_dominated_takes_partials_early(self):
        bt = self._make_backtester_with_repo_at(self.tmpdir)
        trades  = self._make_fake_trades("EURUSD")
        # 50% timeouts → should reduce partial_close_rr to 0.7.
        results = self._make_fake_results(trades,
                                          timeout_count=int(len(trades) * 0.5))
        tuned = bt._compute_tuned_params("EURUSD", trades, results, None)
        self.assertLessEqual(tuned["partial_close_rr"], 0.8)

    # ── End-to-end: backtester → file → learning-loop ─────────────────────────

    def test_end_to_end_tuned_params_consumed_by_loop(self):
        """The whole point: backtester writes tuned_params, learning loop
        reads them, get_tuned_params returns the ML values (not defaults).
        If this passes, the self-tuning feature works end-to-end."""
        bt = self._make_backtester_with_repo_at(self.tmpdir)
        trades  = self._make_fake_trades("GBPJPY",
                                         avg_win_pips=45, avg_loss_pips=-30)
        results = self._make_fake_results(trades)
        bt._save_insights_for_pair("GBPJPY", feature_importance=None,
                                   results=results, trades=trades)

        # Now spin up the live loop pointed at the same directory.
        loop = AgentLearningLoop(root_path=str(self.tmpdir))
        params = loop.get_tuned_params("GBPJPY")

        # All 7 keys present.
        self.assertEqual(set(params.keys()),
                         set(loop.DEFAULT_PARAMS.keys()))
        # SL is derived from |avg_loss_pips| / median_atr_pips * 1.25.
        # avg_loss_pips=-30, atr_pips=20 → 30/20*1.25 = 1.875.
        self.assertAlmostEqual(params["sl_atr_mult"], 1.88, places=1)
        # TP: max(45/20, sl*1.5) = max(2.25, 2.81) = 2.81.
        self.assertAlmostEqual(params["tp_atr_mult"], 2.81, places=1)

        all_pairs = loop.get_all_tuned_params()
        self.assertIn("GBPJPY", all_pairs)
        self.assertEqual(all_pairs["GBPJPY"]["source"], "backtest")
        self.assertEqual(all_pairs["GBPJPY"]["trade_count"], len(trades))

    def test_end_to_end_other_pairs_still_default(self):
        """After backtesting GBPJPY only, the other 3 pairs should still
        report 'default' — not be silently empty / missing."""
        bt = self._make_backtester_with_repo_at(self.tmpdir)
        trades  = self._make_fake_trades("GBPJPY")
        results = self._make_fake_results(trades)
        bt._save_insights_for_pair("GBPJPY", feature_importance=None,
                                   results=results, trades=trades)

        loop = AgentLearningLoop(root_path=str(self.tmpdir))
        all_pairs = loop.get_all_tuned_params()
        for sym in ("EURUSD", "GBPUSD", "EURJPY"):
            self.assertIn(sym, all_pairs)
            self.assertEqual(all_pairs[sym]["source"], "default")
            self.assertEqual(all_pairs[sym]["params"], loop.DEFAULT_PARAMS)

    def test_loop_bounds_match_backtester_bounds(self):
        """AgentLearningLoop.PARAM_BOUNDS must match the backtester's
        clamps exactly. Drift here would mean the loop silently
        rejects valid tuned values."""
        from odl.backtest import AgentZeroBacktester as _BT
        loop = AgentLearningLoop(root_path=str(self.tmpdir))
        self.assertEqual(
            dict(loop.PARAM_BOUNDS), dict(_BT._TUNED_PARAM_BOUNDS),
            "PARAM_BOUNDS drifted between backtester and loop — fix one or both",
        )
        self.assertEqual(
            dict(loop.DEFAULT_PARAMS), dict(_BT._TUNED_PARAM_DEFAULTS),
            "DEFAULT_PARAMS drifted between backtester and loop",
        )


class TestAiProImports(unittest.TestCase):
    """Smoke test: ai_pro must import even when the insights file is
    missing. This catches the most common breakage mode of the wiring."""

    def test_ai_pro_module_imports(self):
        # ai_pro pulls in heavy stuff (MetaTrader5) which is Windows-only.
        # On non-Windows / CI hosts we just check that the learning-loop
        # specific block didn't introduce a syntax error.
        try:
            import ai_pro  # noqa: F401
        except ImportError as exc:
            # MT5 missing is acceptable in CI — but anything else means
            # our edits broke the file.
            msg = str(exc).lower()
            if "metatrader" in msg or "mt5" in msg:
                self.skipTest(f"MT5 unavailable on this host: {exc}")
            raise


# ─────────────────────────── runner ────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
