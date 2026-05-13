"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Fortis AI Pro — Back-tester                                                 ║
║                                                                              ║
║  Runs the deterministic AgentZeroBot strategy over a historical window and  ║
║  produces per-pair results: P&L, win-rate, feature importances, tuned       ║
║  strategy params, and ML metrics.                                            ║
║                                                                              ║
║  Design principles                                                            ║
║  ─────────────────                                                            ║
║  • Zero look-ahead: signal generated at bar[i] close; entry at bar[i+1]     ║
║    open.  The last-forming bar is never used as a signal bar.                ║
║  • Conservative intrabar: when both TP and SL are touched in the same       ║
║    OHLC candle the SL is assumed to have been hit first.                     ║
║  • Realism gates: spread widening, position-overlap deduplication,           ║
║    Asian-session JPY filter, last-bar skip.                                  ║
║  • Single-exit RAW model: the whole position exits at the first SL/TP       ║
║    touch or at the final bar's close on timeout.  No partial close,         ║
║    no breakeven trail, no runner — this isolates the raw signal edge.        ║
║    Live trade-management overlay belongs in the live bot, not here.          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ── Optional imports ──────────────────────────────────────────────────────────

# trade_memory is imported lazily so the backtester works even when SQLite is
# unavailable. The _get_memory() call is only inside _save_insights_for_pair.
HAS_TRADE_MEMORY: bool = False
_get_memory = None   # type: ignore[assignment]
try:
    from odl.trade_memory import get_memory as _get_memory
    HAS_TRADE_MEMORY = True
except ImportError:
    try:
        from trade_memory import get_memory as _get_memory  # type: ignore[no-redef]
        HAS_TRADE_MEMORY = True
    except ImportError:
        pass

HAS_SKLEARN: bool = False
try:
    from sklearn.ensemble import RandomForestClassifier   # noqa: F401
    HAS_SKLEARN = True
except ImportError:
    pass

# ── Pip helpers ───────────────────────────────────────────────────────────────

def get_pip_value(symbol: str) -> float:
    """Return pip size: 0.0001 for most pairs, 0.01 for JPY pairs."""
    if "JPY" in symbol:
        return 0.01  # JPY pairs: 2 decimal places
    return 0.0001   # Standard pairs: 4 decimal places


class _TradeProxy:
    """
    Lightweight read-only proxy that wraps a flat dict row from
    TradeMemory.query_trades() so it looks like a Trade object to
    compute_feature_importance(). Only the fields that function
    actually reads are implemented.
    """
    __slots__ = ('outcome', 'entry_time', 'signal', 'rr_achieved', 'profit_pips', 'profit')

    class _SignalProxy:
        __slots__ = ('source', 'symbol', 'signal', 'confidence', 'rr_ratio',
                     'entry_price', 'stop_loss', 'take_profit', 'atr', 'poc', 'ts', 'environment')
        def __init__(self, row: dict) -> None:
            self.source      = row.get('source', '')
            self.symbol      = row.get('symbol', '')
            self.signal      = row.get('signal', 'BUY')
            self.confidence  = float(row.get('confidence', 0.0) or 0.0)
            self.rr_ratio    = float(row.get('rr_ratio',   0.0) or 0.0)
            self.entry_price = float(row.get('entry_price', 0.0) or 0.0)
            self.stop_loss   = float(row.get('sl',  0.0) or 0.0)
            self.take_profit = float(row.get('tp',  0.0) or 0.0)
            self.atr         = float(row.get('atr', 0.0) or 0.0)
            self.poc         = float(row.get('poc', 0.0) or 0.0)
            self.environment = row.get('source', '')
            raw_ts = row.get('entry_time')
            try:
                from datetime import datetime as _dt
                self.ts = _dt.fromisoformat(str(raw_ts)) if raw_ts else None
            except Exception:
                self.ts = None

    def __init__(self, row: dict) -> None:
        self.outcome     = row.get('outcome', '')
        self.profit_pips = float(row.get('profit_pips', 0.0) or 0.0)
        self.profit      = float(row.get('profit_usd',  0.0) or 0.0)
        self.rr_achieved = float(row.get('rr_achieved', 0.0) or 0.0)
        raw_et = row.get('entry_time')
        try:
            from datetime import datetime as _dt
            self.entry_time = _dt.fromisoformat(str(raw_et)) if raw_et else None
        except Exception:
            self.entry_time = None
        self.signal = _TradeProxy._SignalProxy(row)


def get_pip_usd_value(symbol: str, lot_size: float = 1.0,
                      jpy_quote_rate: float = 150.0) -> float:
    """
    USD value of one pip for `lot_size` lots.

    For USD-quote pairs (EURUSD, GBPUSD):
        1 pip = 0.0001 price units × 100,000 units/lot × lot_size
              = $10 per standard lot

    For JPY-quote pairs (EURJPY, GBPJPY):
        1 pip = 0.01 JPY × 100,000 units/lot × lot_size
        Convert: ÷ jpy_quote_rate  (≈ 150 USDJPY → ~$6.67/pip/lot)

    Parameters
    ----------
    symbol         : e.g. "EURUSD", "GBPJPY"
    lot_size       : number of lots (0.5 = mini lot × 5)
    jpy_quote_rate : approximate USDJPY rate for conversion (default 150)
    """
    pip = get_pip_value(symbol)
    units_per_lot = 100_000.0

    if "JPY" in symbol.upper():
        # JPY-quoted pair: pip value in JPY, convert to USD
        pip_usd = (pip * units_per_lot * lot_size) / jpy_quote_rate
    else:
        # USD-quoted pair (EURUSD, GBPUSD, etc.)
        pip_usd = pip * units_per_lot * lot_size

    return pip_usd


# ── Domain objects ────────────────────────────────────────────────────────────

@dataclass
class Signal:
    """A generated signal from AgentZeroBot at a specific time."""
    ts: datetime
    symbol: str
    signal: str  # "BUY", "SELL", "neutral"
    source: str
    confidence: float
    percentage: int
    quality: str
    entry_price: float
    stop_loss: float
    take_profit: float
    rr_ratio: float
    atr: float = 0.0  # ATR (price units) at the bar that produced the signal
    component_scores: Dict[str, float] = field(default_factory=dict)
    environment: str = ""  # CHoCH, Continuation, etc.
    # Snapshot of the strategy's volume-profile context at signal time —
    # used historically by the backtester's VP filter. Kept on the signal
    # for diagnostics even though the active filter is now POC-bias only.
    volume_profile: Dict[str, Any] = field(default_factory=dict)
    # Volume-profile Point of Control (price of the highest-volume bin)
    # computed by the backtester from the bar window up to and including
    # the trigger bar. 0.0 when unavailable (insufficient data).
    poc: float = 0.0


@dataclass
class Trade:
    """A completed trade with entry and exit."""
    signal: Signal
    entry_price: float
    entry_time: datetime
    
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: str = ""  # "tp" | "sl" | "timeout"
    
    profit: float = 0.0
    profit_pips: float = 0.0
    rr_achieved: float = 0.0
    outcome: str = "open"  # "WIN" | "LOSS" | "open"
    
    duration_minutes: int = 0
    
    def calculate_exit(self, exit_price: float, exit_time: datetime, reason: str, lot_size: float = 1.0) -> None:
        """Calculate exit after hitting TP, SL, or timeout."""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = reason

        # Determine pip size based on symbol (JPY pairs use 0.01, others use 0.0001)
        pip_size = get_pip_value(self.signal.symbol)

        if self.signal.signal == "BUY":
            self.profit_pips = (exit_price - self.entry_price) / pip_size
            risk_pips = (self.entry_price - self.signal.stop_loss) / pip_size
        else:  # SELL
            self.profit_pips = (self.entry_price - exit_price) / pip_size
            risk_pips = (self.signal.stop_loss - self.entry_price) / pip_size

        # Per-pip USD value (handles JPY quote pairs correctly — was previously
        # a flat $10 which overstated JPY P&L by ~50%).
        usd_per_pip = get_pip_usd_value(self.signal.symbol, lot_size=lot_size)
        self.profit = self.profit_pips * usd_per_pip
        
        self.duration_minutes = int((exit_time - self.entry_time).total_seconds() / 60)
        
        if self.profit > 0:
            self.outcome = "WIN"
        elif self.profit < 0:
            self.outcome = "LOSS"
        else:
            self.outcome = "BE"

        if risk_pips > 0:
            self.rr_achieved = abs(self.profit_pips) / risk_pips
        else:
            self.rr_achieved = 0.0


# ============================================================ #
# Backtester                                                   #
# ============================================================ #

class Backtester:
    """
    Runs the deterministic AgentZeroBot strategy over a look-back window.

    Usage
    -----
    bt = Backtester(strategy=live_strategy_instance)
    results = bt.run_backtest(symbol="EURUSD", days=30, lot_size=0.5)
    """

    # ── Default / fallback param values (mirrors live defaults) ──────────────
    _TUNED_PARAM_DEFAULTS: Dict[str, float] = {
        "atr_tolerance_mult": 1.5,
        "sl_atr_mult":        2.5,
        "tp_atr_mult":        4.5,
        "partial_close_rr":   1.0,
        "be_buffer_pips":     1.0,
        "min_atr_to_tighten": 1.0,
        "trail_atr_mult":     1.0,
    }

    # ── Clamp ranges (guards against outlier tuning results) ──────────────────
    _TUNED_PARAM_CLAMPS: Dict[str, Tuple[float, float]] = {
        "atr_tolerance_mult": (1.0, 2.5),
        "sl_atr_mult":        (1.0, 5.0),
        "tp_atr_mult":        (1.5, 10.0),
        "partial_close_rr":   (0.5, 3.0),
        "be_buffer_pips":     (0.3, 5.0),
        "min_atr_to_tighten": (0.5, 3.0),
        "trail_atr_mult":     (0.5, 2.0),
    }

    def __init__(
        self,
        strategy=None,
        intrabar_policy: str = "conservative",
        lot_size: float = 0.5,
        # ── Realism params ───────────────────────────────────────────────────
        # Typical broker spread (in price points, not pips) added on BUY entry.
        # EURUSD: 1 point = 0.00001 → 1.0 point spread ≈ 1 pip (0.0001 / 0.00001)
        # GBPJPY: 1 point = 0.001  → 2.0 point spread ≈ 2 pips
        default_spread_points_major: float = 1.0,
        default_spread_points_jpy:   float = 2.0,
        # Asian-session spread widening multiplier applied on top of the above.
        # GBPJPY ~17 pt typical → 33–35 pt during Asian session, while
        # EURUSD widens to ~1.4 pt — realistic but not catastrophic.
        asian_session_jpy_multiplier: float = 2.2,
        asian_session_other_multiplier: float = 1.4,
        # Maximum spread (points) that a signal is allowed to trade through.
        # Signals with estimated spread above this are silently skipped, the
        # same way the live bot rejects them.
        max_spread_points_jpy:   float = 30.0,
        max_spread_points_major: float = 3.0,
    ) -> None:
        self._strategy  = strategy
        self.intrabar_policy = intrabar_policy
        self.lot_size   = float(lot_size)

        self.default_spread_points_major    = float(default_spread_points_major)
        self.default_spread_points_jpy      = float(default_spread_points_jpy)
        self.asian_session_jpy_multiplier   = float(asian_session_jpy_multiplier)
        self.asian_session_other_multiplier = float(asian_session_other_multiplier)
        self.max_spread_points_jpy          = float(max_spread_points_jpy)
        self.max_spread_points_major        = float(max_spread_points_major)

        # Set by simulate_trades; inspected by tests and the SSE stream handler.
        self._last_skip_stats: Dict[str, int] = {}
        self._active_until: Dict[str, datetime] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def run_backtest(
        self,
        symbol: str,
        days: int = 7,
        lot_size: Optional[float] = None,
        progress_cb: Optional[Callable[[dict], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run the full back-test pipeline for one symbol.

        Returns a results dict consumed by the SSE stream handler and
        the dashboard's per-pair breakdown panel.
        """
        if lot_size is not None:
            self.lot_size = float(lot_size)

        def _emit(payload: dict) -> None:
            if progress_cb is not None:
                try:
                    progress_cb(payload)
                except Exception:
                    pass

        _emit({"type": "stage", "symbol": symbol, "stage": "fetching_data"})

        # ── 1. Fetch historical data ─────────────────────────────────────────
        try:
            df = self._fetch_historical(symbol, days)
        except Exception as exc:
            log.error("Historical data fetch failed for %s: %s", symbol, exc)
            return {"symbol": symbol, "error": str(exc)}

        if df is None or df.empty:
            return {"symbol": symbol, "error": "No historical data returned"}

        _emit({"type": "stage", "symbol": symbol, "stage": "generating_signals"})

        # ── 2. Generate signals ──────────────────────────────────────────────
        try:
            signals = self.generate_signals(symbol, df, days, progress_cb=progress_cb)
        except Exception as exc:
            log.error("Signal generation failed for %s: %s", symbol, exc)
            return {"symbol": symbol, "error": f"Signal generation failed: {exc}"}

        if not signals:
            return {
                "symbol": symbol,
                "error": "No signals generated — strategy found no setups in this window",
            }

        _emit({"type": "stage", "symbol": symbol, "stage": "simulating_trades"})

        # ── 3. Simulate trades ───────────────────────────────────────────────
        try:
            trades = self.simulate_trades(symbol, df, signals, progress_cb=progress_cb)
        except Exception as exc:
            log.error("Trade simulation failed for %s: %s", symbol, exc)
            return {"symbol": symbol, "error": f"Simulation failed: {exc}"}

        skip_stats   = dict(self._last_skip_stats)
        realism_stats = {
            "raw_signals": skip_stats.get("raw_signals", len(signals)),
            "skipped_overlap": skip_stats.get("skipped_overlap", 0),
            "skipped_spread":  skip_stats.get("skipped_spread",  0),
            "simulated":       len(trades),
            "closed":          sum(1 for t in trades if t.exit_time is not None),
        }

        log.info(
            "Simulated %d trades, closed %d "
            "(spread=%.1fp, intrabar=%s, skipped: overlap=%d spread=%d of %d raw)",
            len(trades), realism_stats["closed"],
            (self.default_spread_points_jpy if "JPY" in symbol
             else self.default_spread_points_major),
            self.intrabar_policy,
            realism_stats["skipped_overlap"],
            realism_stats["skipped_spread"],
            realism_stats["raw_signals"],
        )

        _emit({"type": "stage", "symbol": symbol, "stage": "analysing_results"})

        # ── 4. Analyse results ───────────────────────────────────────────────
        try:
            results = self.analyze_results(symbol, trades, df)
        except Exception as exc:
            log.error("Analysis failed for %s: %s", symbol, exc)
            return {"symbol": symbol, "error": f"Analysis failed: {exc}"}

        results["symbol"] = symbol
        results["period"] = (f"{df.iloc[0]['time'].date()} to "
                             f"{df.iloc[-1]['time'].date()}")
        results["window_start_utc"] = df.iloc[0]["time"].isoformat()
        results["window_end_utc"]   = df.iloc[-1]["time"].isoformat()
        results["requested_days"]   = int(days)
        results["vp_filter"]        = {}          # kept for API compat
        results["realism_filters"]  = realism_stats

        # ── ML: train on current run + deduplicated historical trades ──
        # Current run is saved to DB AFTER this point, so querying
        # returns only previous runs — no risk of double-counting.
        # Dedup on (entry_time, exit_time, outcome) so the same
        # simulated trade from overlapping backtest windows is only
        # counted once — prevents RF bias toward repeated patterns.
        all_trades_for_ml = list(trades)
        if HAS_TRADE_MEMORY and _get_memory is not None:
            try:
                mem_for_ml = _get_memory()
                rows = mem_for_ml.query_trades(symbol=symbol, limit=5000)
                if rows:
                    seen: set = set()
                    extra: list = []
                    for row in rows:
                        key = (
                            str(row.get('entry_time') or ''),
                            str(row.get('exit_time')  or ''),
                            str(row.get('outcome')    or ''),
                        )
                        if key in seen:
                            continue
                        seen.add(key)
                        extra.append(_TradeProxy(row))
                    if extra:
                        all_trades_for_ml = list(trades) + extra
                        log.info(
                            "%s: ML training on %d trades "
                            "(%d this run + %d unique historical, %d dupes removed)",
                            symbol, len(all_trades_for_ml),
                            len(trades), len(extra),
                            len(rows) - len(extra),
                        )
            except Exception as exc:
                log.warning("%s: could not load historical trades for ML: %s", symbol, exc)

        fi = self.compute_feature_importance(all_trades_for_ml)
        if fi is not None:
            results["feature_importance"] = fi

        try:
            fi_for_save = fi if (fi and "error" not in fi) else None
            self._save_insights_for_pair(
                symbol=symbol,
                feature_importance=fi_for_save,
                results=results,
                trades=trades,
            )
        except Exception as exc:
            log.warning("Could not save insights for %s: %s", symbol, exc)

        return results

    # ── Data fetching ─────────────────────────────────────────────────────────

    def _fetch_historical(self, symbol: str, days: int):
        """Fetch M15 OHLCV bars for the look-back window."""
        import pandas as pd

        if self._strategy is None:
            raise RuntimeError("No strategy attached — cannot fetch data")

        # How many bars do we need?
        # days × 24h × 4 bars/h = bars in window, plus a warm-up buffer
        # so the strategy can build swing points / structure before trading.
        bars_needed = int(days * 24 * 4) + 200   # 200-bar warm-up

        try:
            df = self._strategy._fetch_m15(symbol, bars_needed)
        except Exception as exc:
            raise RuntimeError(f"_fetch_m15 failed for {symbol}: {exc}") from exc

        if df is None or (hasattr(df, 'empty') and df.empty):
            raise RuntimeError(f"_fetch_m15 returned empty data for {symbol}")

        # Ensure datetime column
        if "time" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["time"]):
            df["time"] = pd.to_datetime(df["time"])

        # ATR column (needed for spread estimation and signal sizing)
        if "atr" not in df.columns:
            close = df["close"].astype(float)
            high  = df["high"].astype(float)
            low   = df["low"].astype(float)
            tr = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low  - close.shift(1)).abs(),
            ], axis=1).max(axis=1)
            df["atr"] = tr.ewm(span=14, adjust=False).mean()

        return df

    # ── Realism helpers ───────────────────────────────────────────────────────

    def _estimate_spread_points(self, symbol: str, bar_time: datetime) -> float:
        """
        Return the estimated bid-ask spread in price *points* for this bar.

        Session multiplier widens the spread during the Asian session
        (21:00–07:00 UTC) to match realistic broker conditions:
          GBPJPY ~17 pt typical → 33–35 pt during Asian session, while
          EURUSD widens to ~1.4 pt.  Without this the backtester happily
          sails through Asian-session JPY signals that live would have
          rejected due to wide spread.
        """
        is_jpy = "JPY" in symbol.upper()
        base   = (self.default_spread_points_jpy if is_jpy
                  else self.default_spread_points_major)

        # Session multiplier
        hour = bar_time.hour if hasattr(bar_time, "hour") else 0
        is_asian = (hour >= 21) or (hour < 7)
        if not is_asian:
            return base
        mult = (self.asian_session_jpy_multiplier if is_jpy
                else self.asian_session_other_multiplier)
        return base * mult

    def _apply_vp_filter(
        self,
        sig,
        spread_price: float,
    ) -> bool:
        """
        Point-of-Control bias filter.

        Returns True (keep the signal) when the signal's direction is
        consistent with the POC bias computed at signal time, or when POC
        data is unavailable.  Returns False to skip.

        BUY signals want price > POC (bullish bias — price trading above
        the volume-weighted fair-value level).
        SELL signals want price < POC (bearish bias).

        A tolerance band of ±atr_tolerance_mult × ATR is applied so we
        don't reject signals that are only marginally on the wrong side.
        """
        poc = float(getattr(sig, "poc", 0.0) or 0.0)
        if poc <= 0.0:
            return True   # no POC data — don't filter

        entry = float(getattr(sig, "entry_price", 0.0) or 0.0)
        if entry <= 0.0:
            return True

        atr = float(getattr(sig, "atr", 0.0) or 0.0)
        if atr <= 0.0:
            return True

        # Get tolerance from strategy (respects live tuned value)
        tol_mult = 1.5
        if self._strategy is not None:
            try:
                tol_mult = float(getattr(self._strategy, "atr_tolerance_mult", 1.5))
            except Exception:
                pass
        tolerance = atr * tol_mult

        side = str(getattr(sig, "signal", "")).upper()
        if side == "BUY":
            return (entry + spread_price) >= (poc - tolerance)
        if side == "SELL":
            return (entry - spread_price) <= (poc + tolerance)
        return True

    # ── Signal generation ─────────────────────────────────────────────────────

    def generate_signals(
        self,
        symbol: str,
        df,
        days: int,
        progress_cb: Optional[Callable[[dict], None]] = None,
    ) -> List[Signal]:
        """
        Walk the bar array and generate one Signal per bar where the
        strategy would have triggered.

        Key design choices
        ------------------
        • Bar i is the *signal* bar (its close triggers the decision).
        • Entry is taken at bar i+1 open — zero look-ahead.
        • The last-forming bar (df.iloc[-1]) is never used as a signal bar
          to avoid partially-formed OHLC data influencing results.
        • Historical daily / weekly levels are reconstructed from df up to
          bar i so the strategy sees the same data it would have live.
        """
        import pandas as pd

        def _emit(payload: dict) -> None:
            if progress_cb is not None:
                try:
                    progress_cb(payload)
                except Exception:
                    pass

        signals: List[Signal] = []

        # We only trade the last `days` calendar days.  Everything before
        # that is warm-up so the strategy can build its swing-point context.
        now_utc     = datetime.now(timezone.utc).replace(tzinfo=None)
        window_start = pd.Timestamp(now_utc) - pd.Timedelta(days=days)

        if self._strategy is None:
            return signals

        # Save the strategy's original live data fetchers so we can restore
        # them after mocking for historical replay.
        original_fetch  = self._strategy._fetch_m15
        original_levels = self._strategy.get_previous_day_levels

        total_bars = len(df)
        _emit({"type": "bar", "symbol": symbol,
               "bar": 0, "total": total_bars, "signals_so_far": 0})

        for i in range(12, total_bars - 1):   # -1: skip last-forming bar
            bar_time = df.iloc[i]["time"]
            if pd.Timestamp(bar_time) < window_start:
                continue

            # Slice up to and including bar i (no look-ahead)
            df_slice = df.iloc[:i + 1].copy()

            # Mock data fetcher so the strategy only sees history up to bar i
            def mock_fetch(sym, n, _df=df_slice):   # noqa: E731
                return _df.iloc[-min(n, len(_df)):]

            # Reconstruct daily / weekly levels from bars up to this point
            from_day  = df_slice["time"].dt.date
            prev_days = df_slice[from_day < from_day.iloc[-1]].groupby(from_day)
            prev_weeks = (
                df_slice.assign(_w=df_slice["time"].dt.isocalendar().week)
                        .groupby("_w")
            )

            if len(prev_days) > 0:
                last_day  = sorted(prev_days.groups.keys())[-1]
                day_group = prev_days.get_group(last_day)
                correct_levels = {
                    "high": float(day_group["high"].max()),
                    "low":  float(day_group["low"].min()),
                }
            else:
                correct_levels = {"high": 0.0, "low": 0.0}

            if len(prev_weeks) > 0:
                last_week  = sorted(prev_weeks.groups.keys())[-1]
                week_group = prev_weeks.get_group(last_week)
                self._strategy.previous_week_high = float(week_group["high"].max())
                self._strategy.previous_week_low  = float(week_group["low"].min())

            self._strategy._fetch_m15 = mock_fetch
            self._strategy.get_previous_day_levels = lambda s, **kw: correct_levels
            self._strategy.previous_day_high = correct_levels["high"]
            self._strategy.previous_day_low  = correct_levels["low"]

            try:
                try:
                    # Generate signal at this point
                    raw_signal = self._strategy.generate_trade_signal(symbol)

                    if raw_signal.get("signal") != "neutral":
                        # Compute POC from the no-lookahead window the
                        # strategy just saw.
                        poc_value = self._compute_poc(df_slice)
                        sig = Signal(
                            ts=df_slice.iloc[-1]["time"],
                            symbol=symbol,
                            signal=raw_signal.get("signal", "neutral"),
                            source=raw_signal.get("signal_source", "?"),
                            confidence=raw_signal.get("confidence", 0.0),
                            percentage=raw_signal.get("percentage", 0),
                            quality=raw_signal.get("signal_quality", "?"),
                            entry_price=raw_signal.get("entry_price", 0.0),
                            stop_loss=raw_signal.get("stop_loss", 0.0),
                            take_profit=raw_signal.get("take_profit", 0.0),
                            rr_ratio=raw_signal.get("rr_ratio", 0.0),
                            atr=float(raw_signal.get("atr") or 0.0),
                            component_scores=raw_signal.get("component_scores", {}),
                            environment=raw_signal.get("signal_source", "?").split("-")[0] if raw_signal.get("signal_source") else "?",
                            volume_profile=dict(raw_signal.get("volume_profile") or {}),
                            poc=poc_value,
                        )
                        signals.append(sig)
                finally:
                    # Restore original methods
                    self._strategy._fetch_m15 = original_fetch
                    self._strategy.get_previous_day_levels = original_levels

            except Exception as e:
                log.error("Signal gen error at bar %d: %s", i, str(e)[:200])
                if i < 55:  # Print first few errors
                    import traceback
                    traceback.print_exc()
                continue

            if i % 20 == 0:
                _emit({"type": "bar", "symbol": symbol,
                       "bar": i, "total": total_bars,
                       "signals_so_far": len(signals)})

        log.info("Generated %d signals for %s", len(signals), symbol)
        _emit({"type": "signals_done", "symbol": symbol, "signals": len(signals)})
        return signals

    # ── Realism helpers — used by simulate_trades ─────────────────────────────

    def _compute_poc(self, df_slice) -> float:
        """
        Compute the Point-of-Control (price of highest-volume bar) from
        the last 96 bars (≈24 h of M15 data).

        Returns 0.0 when volume data is unavailable or the window is too
        short to be meaningful.
        """
        window = df_slice.tail(96)
        if len(window) < 12:
            return 0.0
        if "volume" not in window.columns:
            return 0.0
        vols = window["volume"].astype(float)
        if vols.sum() <= 0:
            return 0.0
        idx = vols.idxmax()
        poc = float(window.loc[idx, "close"])
        return poc

    # ── Trade simulation ──────────────────────────────────────────────────────

    def simulate_trades(
        self,
        symbol: str,
        df,
        signals: List[Signal],
        progress_cb: Optional[Callable[[dict], None]] = None,
    ) -> List[Trade]:
        """
        Forward-simulate each signal against the OHLC data.

        Entry model
        -----------
        BUY  fills at bar[i+1] open + spread  (ask-priced, worst-case fill).
        SELL fills at bar[i+1] open            (bid-priced, fills at open).

        This matches the round-trip cost a retail account pays.

        Exit model
        ----------
        Single-exit RAW model: the whole position exits at the first
        SL/TP touch, or at the final bar's close on timeout. No partial
        close, no breakeven trail, no runner — this isolates the raw
        signal edge. Live trade-management overlay belongs in the live
        bot, not the backtest.

        Intrabar collisions
        -------------------
        When both TP and SL are touched in the same candle we can't tell
        which hit first from OHLC. Controlled by `intrabar_policy`:
            "conservative" (default): SL wins — worst realistic outcome
            "optimistic":              TP wins — old behaviour
            "neutral":                 whichever is closer to OPEN
        """
        trades: List[Trade] = []

        def _emit(payload: dict) -> None:
            if progress_cb is not None:
                try:
                    progress_cb(payload)
                except Exception:
                    pass

        total_sigs = max(1, len(signals))
        _emit({"type": "sim_begin", "symbol": symbol, "total_signals": total_sigs})

        # ── Realism-gate bookkeeping ─────────────────────────────────────
        # `_active_until[symbol]` = exit_time of the last live trade on
        # this symbol. Subsequent signals before that timestamp are
        # rejected by the position-overlap dedupe (live can't open a
        # second trade while #N is still running). Reset per call so a
        # fresh simulate_trades() invocation doesn't carry state across
        # symbols / windows.
        self._active_until: Dict[str, "datetime"] = {}
        self._last_skip_stats = {
            "raw_signals":     len(signals),
            "skipped_overlap": 0,
            "skipped_spread":  0,
        }

        def _resolve_intrabar(c_open: float, tp: float, sl: float,
                              sl_price: float, tp_price: float) -> Tuple[float, str]:
            """Return (exit_price, reason) for a same-bar TP+SL collision."""
            if self.intrabar_policy == "optimistic":
                return tp_price, "tp"
            if self.intrabar_policy == "neutral":
                if abs(c_open - tp) <= abs(c_open - sl):
                    return tp_price, "tp"
                return sl_price, "sl"
            # conservative (default): SL wins
            return sl_price, "sl"

        bar_times = df["time"].tolist()
        bar_map   = {t: i for i, t in enumerate(bar_times)}
        df_records = df.to_dict("records")

        for sig_idx, sig in enumerate(signals):
            side = str(sig.signal).upper()
            if side not in ("BUY", "SELL"):
                continue

            # ── Find entry bar ───────────────────────────────────────────
            sig_bar_idx = bar_map.get(sig.ts)
            if sig_bar_idx is None:
                # Fuzzy match — find nearest bar
                for idx, t in enumerate(bar_times):
                    if t >= sig.ts:
                        sig_bar_idx = idx
                        break
            if sig_bar_idx is None or sig_bar_idx + 1 >= len(df_records):
                continue

            entry_bar = df_records[sig_bar_idx + 1]
            entry_open = float(entry_bar["open"])
            entry_time = entry_bar["time"]

            # ── Spread estimation ────────────────────────────────────────
            spread_pts   = self._estimate_spread_points(symbol, entry_time)
            is_jpy       = "JPY" in symbol.upper()
            max_spread   = (self.max_spread_points_jpy if is_jpy
                            else self.max_spread_points_major)
            pip_size     = get_pip_value(symbol)
            # Convert spread from points to price units
            # Points for majors: 1 pt = 0.00001 (5-digit broker)
            # Points for JPY:    1 pt = 0.001   (3-digit broker)
            point_size   = 0.001 if is_jpy else 0.00001
            spread_price = spread_pts * point_size

            if spread_pts > max_spread:
                self._last_skip_stats["skipped_spread"] += 1
                continue

            # ── POC-bias filter ──────────────────────────────────────────
            # the backtester takes Asian-session JPY signals that live
            # would have rejected because the entry price already pays the
            # broker spread at this bar (typical × session multiplier)
            if not self._apply_vp_filter(sig, spread_price):
                self._last_skip_stats["skipped_spread"] += 1
                continue

            # ── Position-overlap gate ────────────────────────────────────
            active_until = self._active_until.get(symbol)
            if active_until is not None and entry_time <= active_until:
                self._last_skip_stats["skipped_overlap"] += 1
                continue

            # ── Entry price ──────────────────────────────────────────────
            if side == "BUY":
                entry_price = entry_open + spread_price
            elif side == "SELL":
                entry_price = entry_open
            else:
                continue

            if float(sig.atr or 0.0) > 0.0:
                if self._strategy is not None:
                    sl_mult = self._strategy.get_sl_multiplier(symbol)
                    tp_mult = self._strategy.tp_atr_mult
                else:
                    sl_mult = 2.5
                    tp_mult = 4.5
                if side == "BUY":
                    sl_price = entry_price - float(sig.atr) * sl_mult
                    tp_price = entry_price + float(sig.atr) * tp_mult
                else:
                    sl_price = entry_price + float(sig.atr) * sl_mult
                    tp_price = entry_price - float(sig.atr) * tp_mult
            else:
                sl_price = float(sig.stop_loss)
                tp_price = float(sig.take_profit)

            sig.entry_price = float(entry_price)
            sig.stop_loss = float(sl_price)
            sig.take_profit = float(tp_price)

            # Validate SL is on the correct side of entry (defensive —
            # bad signals otherwise create negative risk_pips / huge P&L).
            if side == "BUY" and sl_price >= entry_price:
                continue
            if side == "SELL" and sl_price <= entry_price:
                continue

            # Risk distance — used for skip-on-bad-signal guard only.
            risk_pips_check = abs(entry_price - sl_price) / pip_size
            if risk_pips_check < 1.0 or risk_pips_check > 500.0:
                continue

            # ── Forward scan for exit ────────────────────────────────────
            trade = Trade(
                signal=sig,
                entry_price=entry_price,
                entry_time=entry_time,
            )

            max_bars      = int(days * 24 * 4)   # max hold = look-back window
            max_exit_idx  = min(sig_bar_idx + 2 + max_bars, len(df_records) - 1)
            resolved      = False

            for j in range(sig_bar_idx + 2, max_exit_idx + 1):
                c = df_records[j]
                c_high = float(c["high"])
                c_low  = float(c["low"])
                c_time = c["time"]

                if side == "BUY":
                    tp_hit = c_high >= tp_price
                    sl_hit = c_low  <= sl_price
                else:
                    tp_hit = c_low  <= tp_price
                    sl_hit = c_high >= sl_price

                # Exit spread correction for SELL trades:
                # MT5 OHLC is bid-priced. Closing a SELL requires buying
                # at the ASK (= bid + spread). Without this the backtest
                # overstates SELL P&L by ~1-2 pips per trade.
                exit_spread = spread_price if side == "SELL" else 0.0

                if tp_hit and sl_hit:
                    exit_price, reason = _resolve_intrabar(
                        float(c["open"]), tp_price, sl_price, sl_price, tp_price)
                    trade.calculate_exit(exit_price + exit_spread, c_time, reason, self.lot_size)
                    resolved = True; break
                if tp_hit:
                    trade.calculate_exit(tp_price + exit_spread, c_time, "tp", self.lot_size)
                    resolved = True; break
                if sl_hit:
                    trade.calculate_exit(sl_price + exit_spread, c_time, "sl", self.lot_size)
                    resolved = True; break

            if not resolved:
                close_candle = df_records[max_exit_idx]
                exit_price = float(close_candle.get("close", entry_price))
                # Apply exit spread correction for SELL on timeout too
                exit_spread_timeout = spread_price if side == "SELL" else 0.0
                trade.calculate_exit(exit_price + exit_spread_timeout, close_candle["time"],
                                     "timeout", self.lot_size)

            if trade.profit_pips <= -20:
                log.warning(
                    "LARGE LOSS: %s %s | Entry: %.5f | SL: %.5f | TP: %.5f | "
                    "Exit: %.5f at %s | P&L: %.1f pips | Reason: %s | Env: %s",
                    side, symbol,
                    entry_price, sl_price, tp_price,
                    trade.exit_price, trade.exit_time,
                    trade.profit_pips,
                    trade.exit_reason,
                    sig.environment or "?",
                )

            trades.append(trade)
            if trade.exit_time:
                self._active_until[symbol] = trade.exit_time

            if sig_idx % 10 == 0:
                _emit({"type": "sim_progress", "symbol": symbol,
                       "done": sig_idx + 1, "total": total_sigs})

        log.info(
            "Simulated %d trades, closed %d "
            "(spread=%.1fp, intrabar=%s, skipped: overlap=%d spread=%d of %d raw)",
            len(trades),
            sum(1 for t in trades if t.exit_time),
            spread_pts,
            self.intrabar_policy,
            self._last_skip_stats["skipped_overlap"],
            self._last_skip_stats["skipped_spread"],
            self._last_skip_stats["raw_signals"],
        )
        _emit({"type": "sim_done", "symbol": symbol, "trades": len(trades)})
        return trades

    # ── Analysis ──────────────────────────────────────────────────────────────

    def analyze_results(
        self,
        symbol: str,
        trades: List[Trade],
        df=None,
    ) -> Dict[str, Any]:
        """Compute per-symbol statistics from the simulated trade list."""

        if not trades:
            return {
                "total_trades": 0, "wins": 0, "losses": 0,
                "win_rate": 0.0, "total_pnl": 0.0, "avg_pnl": 0.0,
                "profit_factor": 0.0, "avg_rr_achieved": 0.0,
                "avg_win_rr": 0.0, "avg_loss_rr": 0.0,
                "avg_win_pips": 0.0, "avg_loss_pips": 0.0,
                "max_drawdown": 0.0, "max_drawdown_pct": 0.0,
                "equity_curve": [],
                "by_quality": {}, "by_exit_reason": {},
                "by_env": {}, "by_side": {}, "by_hour": {}, "by_dow": {},
                "confidence_buckets": {}, "pnl_distribution": {},
                "component_correlations": {},
                "early_exits": {},
                "assumptions": self._build_assumptions(symbol),
            }

        wins   = [t for t in trades if t.outcome == "WIN"]
        losses = [t for t in trades if t.outcome == "LOSS"]
        decided = wins + losses
        win_rate = wins.__len__() / len(trades) if trades else 0

        win_pips  = sum(t.profit_pips for t in wins)
        loss_pips = abs(sum(t.profit_pips for t in losses))

        total_pnl = sum(t.profit for t in trades)
        avg_pnl   = total_pnl / len(trades) if trades else 0

        avg_win_pips  = (sum(t.profit_pips for t in wins) / len(wins))   if wins   else 0.0
        avg_loss_pips = (sum(t.profit_pips for t in losses) / len(losses)) if losses else 0.0

        avg_rr = (
            round(np.mean([t.rr_achieved for t in trades if t.rr_achieved > 0]), 2)
            if any(t.rr_achieved > 0 for t in trades) else 0.0
        )
        avg_win_rr = (
            round(np.mean([t.rr_achieved for t in wins if t.rr_achieved > 0]), 2)
            if any(t.rr_achieved > 0 for t in wins) else 0.0
        )
        avg_loss_rr = (
            round(np.mean([t.rr_achieved for t in losses if t.rr_achieved > 0]), 2)
            if any(t.rr_achieved > 0 for t in losses) else 0.0
        )

        # ── By signal quality ────────────────────────────────────────────────
        by_quality: Dict[str, Any] = {}
        for quality in ["strong", "good", "fair", "weak"]:
            q_trades = [t for t in trades if t.signal.quality == quality]
            if q_trades:
                q_wins = sum(1 for t in q_trades if t.outcome == "WIN")
                by_quality[quality] = {
                    "count": len(q_trades),
                    "win_rate": q_wins / len(q_trades),
                    "avg_pnl": sum(t.profit for t in q_trades) / len(q_trades),
                    "avg_rr": sum(t.rr_achieved for t in q_trades) / len(q_trades),
                }

        # Calculate average trade duration
        durations = [t.duration_minutes for t in trades if t.duration_minutes > 0]
        avg_duration = np.mean(durations) if durations else 0

        # Breakdown by exit reason
        by_exit_reason: Dict[str, Any] = {}
        for reason in ["sl", "tp", "timeout"]:
            r_trades = [t for t in trades if t.exit_reason == reason]
            if r_trades:
                r_wins = sum(1 for t in r_trades if t.outcome == "WIN")
                by_exit_reason[reason] = {
                    "count": len(r_trades),
                    "win_rate": r_wins / len(r_trades),
                    "avg_pnl": sum(t.profit for t in r_trades) / len(r_trades),
                }

        # ── Max drawdown (equity-curve based) ──────────────────────
        sorted_trades = sorted(
            [t for t in trades if t.exit_time is not None],
            key=lambda t: t.exit_time,
        )
        running_pnl = 0.0
        peak_pnl    = 0.0
        max_dd      = 0.0
        equity_curve: List[float] = []
        for t in sorted_trades:
            running_pnl += t.profit
            equity_curve.append(round(running_pnl, 2))
            if running_pnl > peak_pnl:
                peak_pnl = running_pnl
            dd = peak_pnl - running_pnl
            if dd > max_dd:
                max_dd = dd

        max_drawdown_pct = (round((max_dd / peak_pnl) * 100.0, 2) if peak_pnl > 1e-9 else 0.0)

        # ── By environment ────────────────────────────────────────────────────
        by_env: Dict[str, Any] = {}
        for t in trades:
            env = str(t.signal.source or t.signal.environment or "unknown")
            if env not in by_env:
                by_env[env] = {"count": 0, "wins": 0, "losses": 0,
                               "total_pnl": 0.0, "avg_pips": 0.0, "_pips": 0.0}
            by_env[env]["count"] += 1
            by_env[env]["total_pnl"] += t.profit
            by_env[env]["_pips"] += t.profit_pips
            if t.outcome == "WIN":   by_env[env]["wins"]   += 1
            if t.outcome == "LOSS":  by_env[env]["losses"] += 1
        for env_data in by_env.values():
            n = env_data["count"]
            env_data["win_rate"] = env_data["wins"] / n if n else 0.0
            env_data["avg_pips"] = env_data["_pips"] / n if n else 0.0
            del env_data["_pips"]

        # ── By side ───────────────────────────────────────────────────────────
        by_side: Dict[str, Any] = {}
        for side in ("BUY", "SELL"):
            s_trades = [t for t in trades if str(t.signal.signal).upper() == side]
            if s_trades:
                s_wins = sum(1 for t in s_trades if t.outcome == "WIN")
                by_side[side] = {
                    "count":     len(s_trades),
                    "wins":      s_wins,
                    "losses":    sum(1 for t in s_trades if t.outcome == "LOSS"),
                    "win_rate":  s_wins / len(s_trades),
                    "total_pnl": sum(t.profit for t in s_trades),
                    "avg_pips":  sum(t.profit_pips for t in s_trades) / len(s_trades),
                }

        # ── By hour / day-of-week ─────────────────────────────────────────────
        by_hour: Dict[str, Any] = {}
        by_dow:  Dict[str, Any] = {}
        _dow_map = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
        for t in trades:
            ts = t.entry_time or t.signal.ts
            if ts is None:
                continue
            h  = str(ts.hour)
            dw = _dow_map.get(ts.weekday(), "?")
            for bucket, key in [(by_hour, h), (by_dow, dw)]:
                if key not in bucket:
                    bucket[key] = {"count":0,"wins":0,"losses":0,"total_pnl":0.0,"avg_pips":0.0,"_p":0.0}
                bucket[key]["count"] += 1
                bucket[key]["total_pnl"] += t.profit
                bucket[key]["_p"] += t.profit_pips
                if t.outcome == "WIN":  bucket[key]["wins"]   += 1
                if t.outcome == "LOSS": bucket[key]["losses"] += 1
        for bucket in (by_hour, by_dow):
            for d in bucket.values():
                n = d["count"]
                d["win_rate"] = d["wins"] / n if n else 0.0
                d["avg_pips"] = d["_p"]   / n if n else 0.0
                del d["_p"]

        # ── Confidence buckets ────────────────────────────────────────────────
        confidence_buckets: Dict[str, Any] = {}
        _conf_ranges = [("<50",0,50),("50-60",50,60),("60-70",60,70),
                        ("70-80",70,80),("80-90",80,90),("90-100",90,101)]
        for label, lo, hi in _conf_ranges:
            c_trades = [t for t in trades
                        if lo <= (t.signal.confidence or 0) * 100 < hi]
            if c_trades:
                c_wins = sum(1 for t in c_trades if t.outcome == "WIN")
                confidence_buckets[label] = {
                    "count":     len(c_trades),
                    "wins":      c_wins,
                    "losses":    sum(1 for t in c_trades if t.outcome == "LOSS"),
                    "win_rate":  c_wins / len(c_trades),
                    "total_pnl": sum(t.profit for t in c_trades),
                    "avg_pips":  sum(t.profit_pips for t in c_trades) / len(c_trades),
                }

        # ── P&L distribution ──────────────────────────────────────────────────
        pnl_distribution: Dict[str, Any] = {}
        pip_values = [t.profit_pips for t in decided]
        if pip_values:
            nbins = max(5, min(12, len(pip_values) // 5 or 5))
            hist_min = min(pip_values)
            hist_max = max(pip_values)
            if hist_max > hist_min:
                bin_w = (hist_max - hist_min) / nbins
                bins  = []
                for bi in range(nbins):
                    x0  = hist_min + bi * bin_w
                    x1  = x0 + bin_w
                    mid = (x0 + x1) / 2
                    cnt = sum(1 for p in pip_values if x0 <= p < x1)
                    if bi == nbins - 1:
                        cnt = sum(1 for p in pip_values if x0 <= p <= x1)
                    bins.append({
                        "x0":    round(x0,  1),
                        "x1":    round(x1,  1),
                        "mid":   round(mid, 1),
                        "count": cnt,
                        "sign":  "win" if mid > 0 else ("loss" if mid < 0 else "be"),
                    })
                pnl_distribution = {
                    "bins":   bins,
                    "n":      len(pip_values),
                    "median": round(float(np.median(pip_values)), 1),
                }

        # ── Component correlations ────────────────────────────────────────────
        component_correlations: Dict[str, float] = {}
        if decided:
            outcomes_arr = np.array([1.0 if t.outcome == "WIN" else 0.0 for t in decided])
            score_keys   = set()
            for t in decided:
                score_keys.update((t.signal.component_scores or {}).keys())
            for key in score_keys:
                vals = np.array([
                    float((t.signal.component_scores or {}).get(key, 0.0))
                    for t in decided
                ])
                if vals.std() > 1e-9 and outcomes_arr.std() > 1e-9:
                    corr = float(np.corrcoef(vals, outcomes_arr)[0, 1])
                    if np.isfinite(corr):
                        component_correlations[key] = round(corr, 3)

        # ── Early exits analysis ──────────────────────────────────────────────
        early_exits: Dict[str, Any] = {}
        for threshold_candles in [3, 6]:
            threshold_minutes = threshold_candles * 15
            early = [
                t for t in trades
                if t.duration_minutes <= threshold_minutes and t.outcome == "LOSS"
            ]
            if early:
                early_exits[f"within_{threshold_candles}_candles"] = {
                    "count": len(early),
                    "pct_of_losses": len(early) / sum(1 for t in trades if t.outcome == "LOSS"),
                    "avg_loss_pips": sum(t.profit_pips for t in early) / len(early),
                    "exit_reasons": dict(
                        (reason, sum(1 for t in early if t.exit_reason == reason))
                        for reason in ["sl", "timeout"]
                    ),
                }

        # ── By quality ────────────────────────────────────────────────────────
        by_quality_detailed: Dict[str, Any] = {}
        for quality in ["strong", "good", "fair", "weak"]:
            q_trades = [t for t in trades if t.signal.quality == quality]
            if q_trades:
                q_wins = sum(1 for t in q_trades if t.outcome == "WIN")
                q_losses = sum(1 for t in q_trades if t.outcome == "LOSS")
                by_quality_detailed[quality] = {
                    "count":     len(q_trades),
                    "wins":      q_wins,
                    "losses":    q_losses,
                    "win_rate":  q_wins / len(q_trades),
                    "total_pnl": sum(t.profit for t in q_trades),
                    "avg_pips":  sum(t.profit_pips for t in q_trades) / len(q_trades),
                }

        return {
            "total_trades":          len(trades),
            "wins":                  len(wins),
            "losses":                len(losses),
            "win_rate":              round(win_rate, 4),
            "win_rate_label":        f"{win_rate * 100:.1f}%",
            "total_pnl":             round(total_pnl, 2),
            "total_pnl_label":       f"{'+'if total_pnl>=0 else ''}${total_pnl:.2f}",
            "avg_pnl":               round(avg_pnl, 2),
            "avg_pnl_label":         f"{'+'if avg_pnl>=0 else ''}${avg_pnl:.2f}",
            "profit_factor":         round(abs(win_pips / loss_pips), 2) if loss_pips != 0 else (999.0 if win_pips > 0 else 0.0),
            "avg_rr_achieved":       avg_rr,
            "avg_win_rr":            avg_win_rr,
            "avg_loss_rr":           avg_loss_rr,
            "avg_win_pips":          round(avg_win_pips,  1),
            "avg_loss_pips":         round(avg_loss_pips, 1),
            "avg_duration_minutes":  round(float(avg_duration), 1),
            "max_drawdown":          round(max_dd, 2),
            "max_drawdown_pct":      max_drawdown_pct,
            "max_drawdown_label":    f"-${max_dd:.2f}" if max_dd > 0 else "$0.00",
            "equity_curve":          equity_curve,
            "by_quality":            by_quality_detailed,
            "by_exit_reason":        by_exit_reason,
            "by_env":                by_env,
            "by_side":               by_side,
            "by_hour":               by_hour,
            "by_dow":                by_dow,
            "confidence_buckets":    confidence_buckets,
            "pnl_distribution":      pnl_distribution,
            "component_correlations":component_correlations,
            "early_exits":           early_exits,
            "assumptions":           self._build_assumptions(symbol),
        }

    def _build_assumptions(self, symbol: str) -> Dict[str, Any]:
        """Return the modelling-assumptions block shown in the dashboard."""
        is_jpy    = "JPY" in symbol.upper()
        spread_pt = (self.default_spread_points_jpy if is_jpy
                     else self.default_spread_points_major)
        return {
            "trade_management": "raw (single SL/TP exit)",
            "lot_size":         self.lot_size,
            "spread_pips":      None,   # signals "default per-pair" in the UI
            "intrabar_policy":  "conservative",
            "entry_fill":       "bid-based OHLC: BUY pays full spread, SELL fills at open",
            "pip_usd_model":    "JPY pairs converted at ~150 USDJPY",
            "commission":       "not modelled",
            "swap":             "not modelled",
            "lookahead":        "last-forming bar skipped",
        }

    # ── ML ────────────────────────────────────────────────────────────────────

    def compute_feature_importance(
        self,
        trades: List,
    ) -> Optional[Dict[str, Any]]:
        """
        Train a RandomForest on trade-level features and return feature
        importances. Purely diagnostic — does NOT feed back into live
        trading. Returns None if sklearn is unavailable or the dataset is
        too thin to be meaningful.
        """
        if not HAS_SKLEARN:
            return {"error": "sklearn not installed (pip install scikit-learn)"}

        decided = [t for t in trades if t.outcome in ("WIN", "LOSS")]
        if len(decided) < 15:
            return {"error": f"Not enough decided trades for ML ({len(decided)}<15)"}

        FEATURES = [
            "confidence",
            "rr_ratio",
            "risk_pips",
            "is_buy",
            "is_choch",
            "is_continuation",
            "anchor_pdh",
            "anchor_pdl",
            "hour_of_day",
            "dow",
            "is_jpy",
            "dist_to_poc_atr",
            "dist_to_poc_pips",
            "has_poc",
        ]

        X: List[List[float]] = []
        y: List[int] = []
        for t in decided:
            src = str(t.signal.source or "").upper()
            pip = get_pip_value(t.signal.symbol)
            risk_pips = abs(t.signal.entry_price - t.signal.stop_loss) / pip if pip else 0.0
            ts = t.entry_time or t.signal.ts
            hour = int(ts.hour) if ts is not None else 0
            dow  = int(ts.weekday()) if ts is not None else 0

            poc_val = float(t.signal.poc or 0.0)
            atr_val = float(t.signal.atr or 0.0)
            entry_val = float(t.signal.entry_price or 0.0)
            is_buy = str(t.signal.signal).upper() == "BUY"
            if poc_val > 0.0 and entry_val > 0.0:
                raw_diff = entry_val - poc_val
                signed = raw_diff if is_buy else -raw_diff
                dist_to_poc_atr  = signed / atr_val if atr_val > 0.0 else 0.0
                dist_to_poc_pips = signed / pip if pip > 0.0 else 0.0
                has_poc = 1.0
            else:
                dist_to_poc_atr  = 0.0
                dist_to_poc_pips = 0.0
                has_poc = 0.0

            X.append([
                float(t.signal.confidence or 0.0),
                float(t.signal.rr_ratio or 0.0),
                risk_pips,
                1.0 if is_buy else 0.0,
                1.0 if "CHOCH" in src else 0.0,
                1.0 if "CONTINUATION" in src else 0.0,
                1.0 if "PDH" in src else 0.0,
                1.0 if "PDL" in src else 0.0,
                float(hour),
                float(dow),
                1.0 if pip == 0.01 else 0.0,
                dist_to_poc_atr,
                dist_to_poc_pips,
                has_poc,
            ])
            y.append(1 if t.outcome == "WIN" else 0)

        import numpy as np_local
        X_arr = np_local.array(X)
        y_arr = np_local.array(y)

        # Drop constant features (RF can't use them)
        keep_idx = [i for i in range(X_arr.shape[1]) if X_arr[:, i].std() > 1e-9]
        if len(keep_idx) < 2:
            return {"error": "Too few variable features for RF"}

        X_arr = X_arr[:, keep_idx]
        kept_features = [FEATURES[i] for i in keep_idx]

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score

        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            oob_score=True,
        )
        clf.fit(X_arr, y_arr)

        oob_pred = (clf.oob_decision_function_[:, 1] >= 0.5).astype(int)
        bal_acc  = balanced_accuracy_score(y_arr, oob_pred)

        wins_actual = int(y_arr.sum())
        baseline_wr = wins_actual / len(y_arr)
        lift = bal_acc - 0.5   # lift over random 50%

        try:
            win_prec = precision_score(y_arr, oob_pred, zero_division=0)
            win_rec  = recall_score(y_arr, oob_pred, zero_division=0)
        except Exception:
            win_prec = win_rec = 0.0

        imps = sorted(
            zip(kept_features, clf.feature_importances_),
            key=lambda x: x[1], reverse=True,
        )

        return {
            "n_trades":          len(decided),
            "baseline_win_rate": round(baseline_wr, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "lift_vs_random":    round(lift, 4),
            "win_precision":     round(win_prec, 4),
            "win_recall":        round(win_rec,  4),
            "importances": [
                {"feature": f, "importance": round(float(v), 4)}
                for f, v in imps
            ],
        }

    # ── Tuned-param derivation ────────────────────────────────────────────────

    def _clamp_tuned(self, key: str, value: float) -> float:
        lo, hi = self._TUNED_PARAM_CLAMPS.get(key, (0.0, 99.0))
        return round(max(lo, min(hi, float(value))), 4)

    def _compute_tuned_params(
        self,
        symbol: str,
        trades: List[Trade],
        results: Dict[str, Any],
        feature_importance: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, float]]:
        """Derive per-pair tuned_params from backtest statistics."""
        decided = [t for t in trades if t.outcome in ("WIN", "LOSS")]
        # Allow derivation from aggregate results when individual trades are
        # unavailable (e.g. re-computing tuned_params for old backtest runs
        # that pre-date per-trade storage), as long as we have >= 5 trades.
        aggregate_trade_count = int((results or {}).get("trade_count") or 0)
        if len(decided) < 5 and aggregate_trade_count < 5:
            return None

        pip_size = get_pip_value(symbol)
        atr_pips_series = [
            float(t.signal.atr or 0.0) / pip_size
            for t in decided
            if t.signal.atr and pip_size > 0
        ]
        # Fallback: if ATR was not stored on the signal (e.g. EURJPY
        # trades generated before ATR logging was added), derive it from
        # the actual SL distance (SL = ATR × sl_mult ⇒ ATR ≈ SL / sl_mult).
        if not atr_pips_series and pip_size > 0:
            default_sl = 2.5
            atr_pips_series = [
                abs(float(t.signal.entry_price or 0.0) - float(t.signal.stop_loss or 0.0))
                / pip_size / default_sl
                for t in decided
                if t.signal.entry_price and t.signal.stop_loss
                and abs(float(t.signal.entry_price) - float(t.signal.stop_loss)) > 0
            ]
        median_atr_pips = (
            float(np.median(atr_pips_series)) if atr_pips_series else 0.0
        )

        if median_atr_pips <= 0:
            # ATR data absent (old trades pre-dating ATR logging).
            # Use a pair-specific typical M15 ATR so we can still derive
            # sensible SL/TP multipliers rather than skipping tuning entirely.
            median_atr_pips = 20.0 if "JPY" in symbol.upper() else 10.0

        def _f(key: str, default: Any = 0.0) -> float:
            v = results.get(key)
            try:
                return float(v) if v is not None else default
            except (TypeError, ValueError):
                return default

        win_rate     = _f("win_rate")
        avg_win_p    = _f("avg_win_pips")
        avg_loss_p   = _f("avg_loss_pips")
        avg_rr       = _f("avg_rr_achieved")
        avg_dur_min  = _f("avg_duration_minutes")
        by_exit      = results.get("by_exit_reason") or {}
        timeout_n    = int((by_exit.get("timeout") or {}).get("count", 0))
        timeout_pct  = (timeout_n / len(decided)) if decided else 0.0
        loss_mag     = abs(avg_loss_p)

        if loss_mag > 0:
            sl_atr_mult = (loss_mag / median_atr_pips) * 1.25
        else:
            sl_atr_mult = self._TUNED_PARAM_DEFAULTS["sl_atr_mult"]

        if avg_win_p > 0:
            tp_from_data = avg_win_p / median_atr_pips
        else:
            tp_from_data = self._TUNED_PARAM_DEFAULTS["tp_atr_mult"]
        tp_atr_mult = max(tp_from_data, sl_atr_mult * 1.5)

        atr_tolerance_mult = 1.5
        if feature_importance:
            lift = feature_importance.get("lift_vs_random")
            if lift is not None:
                try:
                    lift_f = float(lift)
                    atr_tolerance_mult = 1.5 + max(-0.3, min(0.5, lift_f * 3.0))
                except (TypeError, ValueError):
                    pass

        if timeout_pct > 0.30:
            partial_close_rr = 0.7
        elif avg_rr > 1.5:
            partial_close_rr = min(1.5, avg_rr * 0.7)
        else:
            partial_close_rr = 1.0

        spread_floor = 2.0 if "JPY" in symbol.upper() else 1.0
        if win_rate >= 0.60:
            be_buffer_pips = spread_floor * 0.7
        elif win_rate >= 0.45:
            be_buffer_pips = spread_floor
        else:
            be_buffer_pips = spread_floor * 1.5

        if avg_dur_min > 0:
            ratio = avg_dur_min / 360.0
            min_atr_to_tighten = 0.7 + max(-0.2, min(1.3, ratio - 1.0))
        else:
            min_atr_to_tighten = self._TUNED_PARAM_DEFAULTS["min_atr_to_tighten"]

        if avg_win_p > 0:
            run_ratio = avg_win_p / median_atr_pips
            trail_atr_mult = 0.7 + max(0.0, min(0.9, (run_ratio - 1.0) / 4.0 * 0.9))
        else:
            trail_atr_mult = self._TUNED_PARAM_DEFAULTS["trail_atr_mult"]

        return {
            "atr_tolerance_mult": self._clamp_tuned("atr_tolerance_mult", atr_tolerance_mult),
            "sl_atr_mult":        self._clamp_tuned("sl_atr_mult",        sl_atr_mult),
            "tp_atr_mult":        self._clamp_tuned("tp_atr_mult",        tp_atr_mult),
            "partial_close_rr":   self._clamp_tuned("partial_close_rr",   partial_close_rr),
            "be_buffer_pips":     self._clamp_tuned("be_buffer_pips",     be_buffer_pips),
            "min_atr_to_tighten": self._clamp_tuned("min_atr_to_tighten", min_atr_to_tighten),
            "trail_atr_mult":     self._clamp_tuned("trail_atr_mult",     trail_atr_mult),
        }

    def _save_insights_for_pair(
        self,
        symbol: str,
        feature_importance: Optional[Dict[str, Any]] = None,
        results: Optional[Dict[str, Any]] = None,
        trades: Optional[List[Trade]] = None,
    ) -> None:
        """
        Persist this backtest run to SQLite, then aggregate ALL historical
        runs for `symbol` and write the result to backtest_insights.json
        so the live AgentLearningLoop picks it up via its mtime watch.

        Flow
        ----
        1. Derive tuned_params from this run's statistics (unchanged logic).
        2. Save the raw run + every trade + importances + tuned_params to
           trade_memory.db (accumulates across runs — never overwrites).
        3. Call TradeMemory.aggregate_insights(symbol) to compute a
           weighted-average importance dict across ALL stored runs.
        4. Write the aggregated block to backtest_insights.json exactly as
           before, using the same atomic temp+replace pattern.

        If trade_memory is unavailable (import failed), falls back to the
        original single-run JSON write so nothing breaks.
        """
        # ── 1. Derive tuned params (same logic as before) ────────────────
        tuned: Optional[Dict[str, float]] = None
        if results is not None and trades is not None:
            try:
                tuned = self._compute_tuned_params(
                    symbol=symbol,
                    trades=trades,
                    results=results,
                    feature_importance=feature_importance,
                )
            except Exception as exc:
                log.warning("Tuning derivation for %s failed: %s", symbol, exc)

        # ── 2. Persist to SQLite ─────────────────────────────────────────
        aggregated: Dict[str, Any] = {}
        if HAS_TRADE_MEMORY and _get_memory is not None:
            try:
                mem = _get_memory()
                run_id = mem.save_run(
                    symbol=symbol,
                    results=results or {},
                    trades=trades or [],
                    feature_importance=feature_importance,
                    tuned_params=tuned,
                )
                log.info(
                    "%s: run #%d saved to trade_memory.db — aggregating across all runs",
                    symbol, run_id,
                )
                # ── 3. Aggregate across ALL stored runs ──────────────────
                aggregated = mem.aggregate_insights(symbol)
                if aggregated:
                    log.info(
                        "%s: aggregated insights from %d runs, %d total trades",
                        symbol,
                        aggregated.get("n_runs", "?"),
                        aggregated.get("trade_count", "?"),
                    )
            except Exception as exc:
                log.warning(
                    "%s: SQLite persistence failed (%s) — falling back to "
                    "single-run JSON write",
                    symbol, exc,
                )

        # If SQLite path produced aggregated data use it; otherwise fall
        # back to building the pair_block from this run only (original
        # behaviour — guarantees the live bot always gets *something*).
        if aggregated:
            pair_block = dict(aggregated)
        else:
            # ── Fallback: single-run block (original logic) ───────────────
            pair_block: Dict[str, Any] = {}

            importances_flat: Dict[str, float] = {}
            if feature_importance:
                raw_importances = feature_importance.get("importances") or []
                for entry in raw_importances:
                    if not isinstance(entry, dict):
                        continue
                    fname = entry.get("feature")
                    score = entry.get("importance")
                    if not fname or score is None:
                        continue
                    importances_flat[fname] = round(float(score), 4)

            if importances_flat:
                pair_block["importances"] = importances_flat
            if tuned:
                pair_block["tuned_params"] = tuned
            if results:
                tc = results.get("total_trades")
                if tc is not None:
                    try:
                        pair_block["trade_count"] = int(tc)
                    except (TypeError, ValueError):
                        pass
                wr = results.get("win_rate")
                if wr is not None:
                    try:
                        pair_block["win_rate"] = round(float(wr), 4)
                    except (TypeError, ValueError):
                        pass
            pair_block["backtest_at"] = datetime.now(timezone.utc).isoformat()

        non_meta_keys = {"importances", "tuned_params", "trade_count", "win_rate"}
        if not (set(pair_block.keys()) & non_meta_keys):
            log.debug("Skipping insights write for %s — nothing useful to store", symbol)
            return

        # ── 4. Atomic write to backtest_insights.json ────────────────────
        import json, tempfile, shutil
        insights_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "backtest_insights.json",
        )

        try:
            with open(insights_path, "r", encoding="utf-8") as fh:
                existing = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            existing = {}

        # Merge — the aggregated block already represents all history, so
        # we replace the existing symbol entry wholesale.
        if aggregated:
            existing[symbol] = pair_block
        else:
            merged = dict(existing.get(symbol) or {})
            merged.update(pair_block)
            existing[symbol] = merged

        tmp_fd, tmp_path = tempfile.mkstemp(
            suffix=".json",
            dir=os.path.dirname(insights_path),
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                json.dump(existing, fh, indent=2)
            shutil.move(tmp_path, insights_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            raise

        tuned_out = pair_block.get("tuned_params") or {}
        msg_parts = [f"SL={tuned_out.get('sl_atr_mult','?')}×",
                     f"TP={tuned_out.get('tp_atr_mult','?')}×",
                     f"PC={tuned_out.get('partial_close_rr','?')}R",
                     f"BE={tuned_out.get('be_buffer_pips','?')}p"]
        if aggregated:
            msg_parts.append(f"(aggregated {aggregated.get('n_runs', '?')} runs)")
        log.info(
            "Saved insights for %s | tuned: %s → %s",
            symbol, " ".join(msg_parts), insights_path,
        )


# Backwards-compatible aliases — server.py imports AgentZeroBacktester
# and calls bt.run(symbol, days=..., progress_cb=...) rather than .run_backtest()
Backtester.run = Backtester.run_backtest
AgentZeroBacktester = Backtester
