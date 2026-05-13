"""
Tests for the May 2026 backtester-realism fix.

Background
----------
The original `simulate_trades` walked every M15 bar in the window and
treated each bar where a setup persisted as an independent trade. Live,
once a position is open, the next bar's setup is a no-op — the bot is
in `manage_position` mode, not `execute_trade` mode. That alone produced
4–5× trade-count inflation. Add the missing spread filter and Asian-
session JPY signals were *also* getting taken in backtest that live's
`max_spread_points: 30` preflight rejects.

The fix adds two gates to `simulate_trades`:
  1. Position-overlap dedupe — skip a signal if its entry_time is
     before the previous trade's exit_time on the same symbol.
  2. Spread filter — estimate the broker spread at the bar (typical
     × Asian-session multiplier) and skip if it exceeds
     `max_spread_points`.

These tests assert the gates do what they claim, without needing MT5
or a live data feed (we hand-craft Signals + a tiny price frame).

Run from the repo root:
    python -m pytest tests/test_backtest_realism_gates.py -v
"""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _make_df(n_bars: int = 60,
             base_price: float = 1.10,
             start: datetime = datetime(2026, 5, 6, 8, 0)) -> pd.DataFrame:
    """Tiny deterministic OHLCV with a slow drift up. Naive UTC time
    column to match `df["time"] = pd.to_datetime(..., unit='s')` in
    the production fetcher."""
    times  = [start + timedelta(minutes=15 * i) for i in range(n_bars)]
    closes = base_price + np.linspace(0.0, 0.0040, n_bars)  # +40 pip drift
    opens  = np.r_[closes[0], closes[:-1]]
    highs  = closes + 0.0008
    lows   = closes - 0.0008
    return pd.DataFrame({
        "time":        times,
        "open":        opens,
        "high":        highs,
        "low":         lows,
        "close":       closes,
        "tick_volume": np.full(n_bars, 100, dtype=int),
        "real_volume": np.zeros(n_bars, dtype=int),
    })


def _make_signal(ts: datetime, side: str = "BUY",
                 entry: float = 1.10, sl: float = 1.099, tp: float = 1.103,
                 atr: float = 0.0):
    from odl.backtest import Signal
    return Signal(
        ts=ts, symbol="EURUSD", signal=side,
        source="CHoCH-BUY@PDL", confidence=0.85, percentage=85,
        quality="good",
        entry_price=entry, stop_loss=sl, take_profit=tp,
        rr_ratio=2.0, poc=0.0,           # poc=0 → VP filter passthrough
        atr=atr,
        component_scores={
            "structure_strength": 0.8, "level_interaction": 0.8,
            "momentum_quality": 0.8, "spread_volatility": 0.8,
            "environment_fit": 0.8,
        },
    )


class TestPositionOverlapDedupe(unittest.TestCase):
    """Four signals on consecutive bars must collapse to one trade
    (the one whose exit closes first), not four."""

    def setUp(self) -> None:
        try:
            from odl.backtest import AgentZeroBacktester
        except ImportError as exc:
            self.skipTest(f"odl.backtest unavailable: {exc}")
        self.AgentZeroBacktester = AgentZeroBacktester

    def test_consecutive_same_setup_collapses_to_one_trade(self):
        bt = self.AgentZeroBacktester(
            enforce_position_overlap=True,
            enforce_spread_filter=False,         # isolate overlap effect
            symbols=["EURUSD"],
            lot_size=0.5,
        )
        df = _make_df(n_bars=80)
        # Four BUY signals on four consecutive bars (the kind of
        # "setup persisting" the live bot would only enter once on).
        signals = [
            _make_signal(ts=df.iloc[10]["time"]),
            _make_signal(ts=df.iloc[11]["time"]),
            _make_signal(ts=df.iloc[12]["time"]),
            _make_signal(ts=df.iloc[13]["time"]),
        ]
        trades = bt.simulate_trades("EURUSD", signals, df)
        self.assertEqual(len(trades), 1,
                         "overlap dedupe should keep only the first signal "
                         "while the trade is still active")
        stats = bt._last_skip_stats
        self.assertEqual(stats["skipped_overlap"], 3)
        self.assertEqual(stats["skipped_spread"],  0)
        self.assertEqual(stats["taken"],           1)

    def test_signal_after_exit_is_taken(self):
        """Once the first trade is out (SL/TP/timeout), the next
        signal opens a fresh trade — that's the natural trickle live
        would also produce."""
        bt = self.AgentZeroBacktester(
            enforce_position_overlap=True,
            enforce_spread_filter=False,
            symbols=["EURUSD"],
            lot_size=0.5,
            max_trade_duration_bars=4,           # force timeout fast
        )
        df = _make_df(n_bars=80)
        signals = [
            _make_signal(ts=df.iloc[10]["time"]),  # opens trade #1
            _make_signal(ts=df.iloc[20]["time"]),  # well after #1 exits
        ]
        trades = bt.simulate_trades("EURUSD", signals, df)
        self.assertEqual(len(trades), 2)
        self.assertEqual(bt._last_skip_stats["skipped_overlap"], 0)

    def test_overlap_can_be_disabled(self):
        """Setting enforce_position_overlap=False restores the legacy
        bar-by-bar inflated count — useful for max-edge research."""
        bt = self.AgentZeroBacktester(
            enforce_position_overlap=False,
            enforce_spread_filter=False,
            symbols=["EURUSD"],
            lot_size=0.5,
        )
        df = _make_df(n_bars=80)
        signals = [
            _make_signal(ts=df.iloc[10]["time"]),
            _make_signal(ts=df.iloc[11]["time"]),
            _make_signal(ts=df.iloc[12]["time"]),
            _make_signal(ts=df.iloc[13]["time"]),
        ]
        trades = bt.simulate_trades("EURUSD", signals, df)
        self.assertEqual(len(trades), 4)


class TestSpreadFilter(unittest.TestCase):
    """Asian-session JPY signals must be dropped to mirror live's
    `max_spread_points: 30` preflight."""

    def setUp(self) -> None:
        try:
            from odl.backtest import AgentZeroBacktester
        except ImportError as exc:
            self.skipTest(f"odl.backtest unavailable: {exc}")
        self.AgentZeroBacktester = AgentZeroBacktester

    def test_asian_session_jpy_signal_is_dropped(self):
        """GBPJPY at 22:00 UTC: typical 2pip × 10 = 20 pts × 2.2 multiplier
        = 44 pts > 30 threshold. Live would have rejected this, so the
        backtest must too."""
        bt = self.AgentZeroBacktester(
            enforce_position_overlap=False,
            enforce_spread_filter=True,
            max_spread_points=30,
            symbols=["GBPJPY"],
            lot_size=0.5,
        )
        # Bar at 22:00 UTC (Asian session)
        df = _make_df(n_bars=60,
                      start=datetime(2026, 5, 6, 21, 30),
                      base_price=212.0)
        # SL/TP scaled for JPY pip size
        sig = _make_signal(ts=df.iloc[10]["time"],
                           entry=212.0, sl=211.5, tp=213.0)
        sig.symbol = "GBPJPY"
        trades = bt.simulate_trades("GBPJPY", [sig], df)
        self.assertEqual(len(trades), 0)
        self.assertEqual(bt._last_skip_stats["skipped_spread"], 1)

    def test_london_session_jpy_signal_is_kept(self):
        """Same pair at 13:00 UTC: 20 pts base × 1.0 = 20 pts < 30.
        Should pass the filter and become a trade."""
        bt = self.AgentZeroBacktester(
            enforce_position_overlap=False,
            enforce_spread_filter=True,
            max_spread_points=30,
            symbols=["GBPJPY"],
            lot_size=0.5,
        )
        df = _make_df(n_bars=60,
                      start=datetime(2026, 5, 6, 12, 30),
                      base_price=212.0)
        sig = _make_signal(ts=df.iloc[10]["time"],
                           entry=212.0, sl=211.5, tp=213.0)
        sig.symbol = "GBPJPY"
        trades = bt.simulate_trades("GBPJPY", [sig], df)
        self.assertEqual(len(trades), 1)
        self.assertEqual(bt._last_skip_stats["skipped_spread"], 0)
        self.assertEqual(bt._last_skip_stats["taken"], 1)

    def test_eurusd_passes_even_in_asian_session(self):
        """EURUSD at 22:00 UTC: 1pip × 10 = 10 pts × 1.4 = 14 pts.
        Below 30 — passes the gate. Live also lets EURUSD through the
        Asian session in practice."""
        bt = self.AgentZeroBacktester(
            enforce_position_overlap=False,
            enforce_spread_filter=True,
            max_spread_points=30,
            symbols=["EURUSD"],
            lot_size=0.5,
        )
        df = _make_df(n_bars=60,
                      start=datetime(2026, 5, 6, 21, 30),
                      base_price=1.10)
        sig = _make_signal(ts=df.iloc[10]["time"])
        trades = bt.simulate_trades("EURUSD", [sig], df)
        self.assertEqual(len(trades), 1)


class TestSkipStatsExposed(unittest.TestCase):
    """The skip counters must be readable from `_last_skip_stats` so the
    dashboard / orchestrator can render them."""

    def setUp(self) -> None:
        try:
            from odl.backtest import AgentZeroBacktester
        except ImportError as exc:
            self.skipTest(f"odl.backtest unavailable: {exc}")
        self.AgentZeroBacktester = AgentZeroBacktester

    def test_stats_dict_shape(self):
        bt = self.AgentZeroBacktester(symbols=["EURUSD"])
        df = _make_df(n_bars=40)
        bt.simulate_trades("EURUSD", [_make_signal(df.iloc[5]["time"])], df)
        stats = bt._last_skip_stats
        for key in ("raw_signals", "skipped_overlap", "skipped_spread",
                    "taken", "max_spread_points",
                    "overlap_enforced", "spread_enforced"):
            self.assertIn(key, stats, f"missing stat key: {key}")
        self.assertEqual(stats["raw_signals"], 1)
        self.assertGreaterEqual(stats["taken"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
