"""
╔════════════════════════════════════════════════════════════════════╗
║           AI_Pro Backtester — Historical Signal Validation         ║
║        Measure component score correlation with trade outcomes     ║
╚════════════════════════════════════════════════════════════════════╝

Tests the improved AI_Pro engine (structure detection, level states,
momentum scoring, weighted confidence) against historical M15 data.

Tracks:
  ► Signal generation metrics (by component)
  ► Trade performance (wins/losses, R:R achieved)
  ► Correlation: which components predict winning trades?
  ► False reversals caught vs. before (for comparison)

Usage:
    python backtest.py --symbol EURUSD --days 30 --plot
    python backtest.py --symbol GBPUSD --days 7 --export
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Callable, Any
import sys

import numpy as np
import pandas as pd

# Optional: sklearn for feature-importance diagnostics
try:
    from sklearn.ensemble import RandomForestClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Optional: for plotting
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Optional: AI_Pro (only needed for live backtesting)
try:
    from ai_pro import AI_Pro, _mt5_initialize
    HAS_AI_PRO = True
except ImportError:
    HAS_AI_PRO = False
    AI_Pro = None
    _mt5_initialize = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("backtest")

# ============================================================ #
# UTILITY FUNCTIONS                                            #
# ============================================================ #

def get_pip_value(symbol: str) -> float:
    """Return pip size: 0.0001 for most pairs, 0.01 for JPY pairs."""
    if "JPY" in symbol:
        return 0.01  # JPY pairs: 2 decimal places
    return 0.0001   # Standard pairs: 4 decimal places


def get_pip_usd_value(symbol: str, lot_size: float = 1.0,
                      jpy_quote_rate: float = 150.0) -> float:
    """
    USD value of one pip for `lot_size` lots.

    For pairs quoted in USD (xxx/USD): exactly $10 per pip per 1.0 lot.
    For pairs quoted in JPY (xxx/JPY): ~1000 JPY per pip per 1.0 lot, then
    converted to USD by dividing by the USD/JPY rate. We use a fixed
    `jpy_quote_rate` (default ~150) for repeatability — exact value moves
    with the market but ~$6.67/pip is a far better approximation than the
    flat $10 used previously.

    For other quote currencies, this is still approximate; extend as
    needed.
    """
    sym = symbol.upper()
    # Standard contract size for forex = 100,000 units of base currency
    contract = 100_000.0 * lot_size
    if sym.endswith("JPY"):
        pip_in_quote = contract * 0.01           # 1000 JPY per pip per lot
        return pip_in_quote / max(1.0, jpy_quote_rate)
    if sym.endswith("USD"):
        return contract * 0.0001                  # $10 per pip per lot
    # Fallback for cross pairs we don't model (CHF, AUD, etc.) — keep the
    # historical $10 default but make it explicit so the caller can override.
    return 10.0 * lot_size


def default_spread_pips(symbol: str) -> float:
    """Typical retail-broker spread for the major pairs we trade.

    Used as a cost on entry. Override via AI_ProBacktester(spread_pips=...).
    """
    sym = symbol.upper()
    if "JPY" in sym:
        return 2.0   # GBPJPY / EURJPY are wider
    return 1.0       # EURUSD / GBPUSD are typically <=1 pip on majors

# ============================================================ #
# DATA STRUCTURE                                               #
# ============================================================ #

@dataclass
class Signal:
    """A generated signal from AI_Pro at a specific time."""
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
    component_scores: Dict[str, float] = field(default_factory=dict)
    environment: str = ""  # CHoCH, Continuation, etc.


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
# BACKTESTER                                                   #
# ============================================================ #

class AI_ProBacktester:
    """
    Backtester for AI_Pro trading signals.
    
    Requires:
    1. MT5 to be running with credentials configured
    2. Historical M15 data accessible via MT5
    
    Runs:
    1. Fetches M15 candles for period
    2. For each candle, generates AI_Pro signals
    3. Tracks opening/closing of positions
    4. Measures correlation between component scores and outcomes
    """
    
    def __init__(
        self,
        symbols: List[str] = None,
        lookback_bars: int = 200,
        max_trade_duration_bars: int = 96,  # 24 hours on M15
        risk_per_trade_pips: float = 50.0,  # How many pips SL represents
        lot_size: float = 1.0,  # Lot size (0.50, 1.0, etc.)
        # ── Accuracy knobs ────────────────────────────────────────────
        # Typical broker spread charged on entry (in pips). None = use
        # default_spread_pips(symbol). Set to 0.0 to disable.
        spread_pips: Optional[float] = None,
        # When both TP and SL are touched in the same candle we can't
        # tell from OHLC which hit first. "conservative" assumes SL
        # (worst realistic outcome), "optimistic" assumes TP (old
        # behaviour), "neutral" splits 50/50.
        intrabar_policy: str = "conservative",
        # ── Trade management (matches live bot) ───────────────────────
        # When True, simulate the live 50% partial close at 1R plus
        # stop-loss trailed to breakeven (+ buffer). When False, runs
        # the raw strategy to first SL/TP — useful for pure signal
        # quality validation, but will NOT match live P&L.
        enable_trade_management: bool = True,
        partial_close_ratio: float    = 0.5,
        partial_close_rr:    float    = 1.0,
        breakeven_buffer_pips: float  = 1.0,
    ):
        self.symbols = symbols or ["EURUSD"]
        self.lookback_bars = lookback_bars
        self.max_trade_duration_bars = max_trade_duration_bars
        self.risk_per_trade_pips = risk_per_trade_pips
        self.lot_size = lot_size
        self.spread_pips_override = spread_pips
        self.intrabar_policy = intrabar_policy if intrabar_policy in {
            "conservative", "optimistic", "neutral"
        } else "conservative"
        self.enable_trade_management = bool(enable_trade_management)
        self.partial_close_ratio     = float(partial_close_ratio)
        self.partial_close_rr        = float(partial_close_rr)
        self.breakeven_buffer_pips   = float(breakeven_buffer_pips)
        
        # Import AI_Pro lazily to avoid MT5 requirement if just analyzing
        self._strategy = None
        self._mt5 = None
        
        self.signals: List[Signal] = []
        self.trades: List[Trade] = []
        self.candle_history: Dict[str, pd.DataFrame] = {}
    
    def _ensure_mt5(self):
        """Lazy-load MT5 and AI_Pro strategy."""
        if self._strategy is not None:
            return  # Already initialized
        
        if not HAS_AI_PRO:
            log.error("AI_Pro module not available. Run: pip install MetaTrader5")
            return
        
        try:
            # Try to import MetaTrader5
            try:
                import MetaTrader5 as mt5
            except ImportError:
                log.error("MetaTrader5 not installed. Run: pip install MetaTrader5")
                return
            
            # Initialize MT5
            if not _mt5_initialize():
                log.warning("MT5 initialization failed. Ensure MT5 terminal is running with credentials configured.")
                return
            
            # Create strategy without AI (for speed)
            self._strategy = AI_Pro(use_ai=False, lookback_candles=200)
            self._mt5 = mt5
            log.info("MT5 and AI_Pro initialized successfully")
        
        except Exception as e:
            log.error("Failed to initialize MT5/AI_Pro: %s", e)
    
    def fetch_data(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """
        Fetch M15 historical data covering the **most recent `days`
        calendar days**, ending at the current UTC moment.

        Uses `copy_rates_range(start, end)` so the window rolls forward
        with real time — every run gets fresh data ending now. We pull
        two extra days of warm-up so the strategy has 50+ bars of
        swing-point context before the first backtested bar.

        Falls back to `copy_rates_from_pos(0, N)` if the range query
        returns nothing (some broker feeds balk on narrow ranges over
        weekends).
        """
        self._ensure_mt5()

        if self._mt5 is None:
            log.warning("MT5 not available; cannot fetch live data")
            return None

        try:
            now_utc   = datetime.now(timezone.utc)
            warmup_d  = 2  # ~192 M15 bars → plenty of lookback for ENV logic
            start_utc = now_utc - timedelta(days=int(days) + warmup_d)

            rates = self._mt5.copy_rates_range(
                symbol, self._mt5.TIMEFRAME_M15, start_utc, now_utc
            )
            if rates is None or len(rates) < 50:
                # Broker returned nothing meaningful — fall back to the
                # bar-count query. Over-request slightly to cover the
                # weekend-gap shortfall vs. `days * 96`.
                bars_needed = max(self.lookback_bars,
                                  int((int(days) + warmup_d) * 96))
                log.info(
                    "copy_rates_range returned %s; falling back to "
                    "copy_rates_from_pos(0, %d) for %s",
                    "None" if rates is None else f"{len(rates)} bars",
                    bars_needed, symbol,
                )
                rates = self._mt5.copy_rates_from_pos(
                    symbol, self._mt5.TIMEFRAME_M15, 0, bars_needed
                )

            if rates is None or len(rates) < 50:
                log.warning("Not enough data for %s (need 50+ M15 bars)", symbol)
                return None

            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df = df.sort_values("time").reset_index(drop=True)

            first_ts = df.iloc[0]["time"]
            last_ts  = df.iloc[-1]["time"]
            log.info(
                "Fetched %d M15 bars for %s  window=[%s → %s]  (rolling "
                "%d-day request + %d-day warm-up)",
                len(df), symbol, first_ts, last_ts, int(days), warmup_d,
            )
            return df

        except Exception as e:
            log.error("Data fetch error [%s]: %s", symbol, e)
            return None
    
    def _fetch_daily(self, symbol: str, bars: int = 90) -> pd.DataFrame:
        """Fetch daily candles ending NOW. `bars` is treated as the
        calendar-day lookback (we always over-fetch a bit to be safe).
        """
        if self._mt5 is None:
            log.warning("MT5 not available; cannot fetch daily data")
            return pd.DataFrame()

        try:
            # Always fetch at least 90 days of D1 context — the strategy
            # only ever reads previous days, so extra history is free.
            count = max(int(bars), 90)
            rates = self._mt5.copy_rates_from_pos(
                symbol, self._mt5.TIMEFRAME_D1, 0, count
            )
            if rates is None or len(rates) < 1:
                log.warning("Not enough daily data for %s", symbol)
                return pd.DataFrame()

            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df = df.sort_values("time").reset_index(drop=True)

            log.info("Fetched %d daily candles for %s (ending %s)",
                     len(df), symbol, df.iloc[-1]["time"])
            return df

        except Exception as e:
            log.error("Daily data fetch error [%s]: %s", symbol, e)
            return pd.DataFrame()

    def _fetch_weekly(self, symbol: str, bars: int = 52) -> pd.DataFrame:
        """Fetch weekly candles ending NOW."""
        if self._mt5 is None:
            log.warning("MT5 not available; cannot fetch weekly data")
            return pd.DataFrame()

        try:
            count = max(int(bars), 52)
            rates = self._mt5.copy_rates_from_pos(
                symbol, self._mt5.TIMEFRAME_W1, 0, count
            )
            if rates is None or len(rates) < 1:
                log.warning("Not enough weekly data for %s", symbol)
                return pd.DataFrame()

            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df = df.sort_values("time").reset_index(drop=True)

            log.info("Fetched %d weekly candles for %s (ending %s)",
                     len(df), symbol, df.iloc[-1]["time"])
            return df

        except Exception as e:
            log.error("Weekly data fetch error [%s]: %s", symbol, e)
            return pd.DataFrame()
    
    def generate_signals(self, symbol: str, df: pd.DataFrame,
                         progress_cb: Optional[Callable[[dict], None]] = None,
                         window_start: Optional[datetime] = None) -> List[Signal]:
        """
        Generate signals for each candle in history.

        If `window_start` is provided, only bars at or after that UTC
        timestamp are evaluated as signal candidates — earlier bars are
        treated as warm-up context for the strategy's swing detection
        (which needs at least 50 prior bars). This keeps the reported
        backtest window tight to what the user asked for.
        """
        self._ensure_mt5()

        if self._strategy is None:
            log.error("Strategy not initialized")
            return []

        def _emit(payload: dict) -> None:
            if progress_cb is not None:
                try:
                    progress_cb(payload)
                except Exception:
                    pass  # never let a progress hook break the backtest
        
        # Fetch daily and weekly data once before the loop
        df_daily = self._fetch_daily(symbol, bars=90)
        if df_daily.empty:
            log.warning("No daily data available; continuing without daily level mocking")
            df_daily = None
        
        df_weekly = self._fetch_weekly(symbol, bars=52)
        if df_weekly.empty:
            log.warning("No weekly data available; continuing without weekly level mocking")
            df_weekly = None
        
        signals = []
        # Skip the very last bar — MT5 includes the currently-forming
        # candle whose close is a partial tick. Stop one bar early so
        # every bar we evaluate is fully closed (lookahead safety).
        last_closed_idx = len(df) - 1  # exclusive upper bound below

        # Find the first bar inside the user's requested window. Earlier
        # bars serve as warm-up for the strategy (need 50+ swing points),
        # but they don't contribute signals. This keeps the reported
        # period honest: if the user asks for "7 days", signals come
        # from the last 7 days, not the 9-day fetch window.
        if window_start is not None:
            # df["time"] is naive UTC; strip tzinfo on window_start to compare
            ws = (window_start.replace(tzinfo=None)
                  if window_start.tzinfo is not None else window_start)
            mask = df["time"] >= ws
            first_in_window = int(mask.values.argmax()) if mask.any() else last_closed_idx
        else:
            first_in_window = 50
        start_idx = max(50, first_in_window)

        total_bars = max(1, last_closed_idx - start_idx)
        _emit({"type": "signals_begin", "symbol": symbol, "total_bars": total_bars})

        # Simulate walking through history
        for i in range(start_idx, last_closed_idx):  # warm-up is [0..start_idx)
            # Emit progress every ~5% of the run so the UI can update live
            bars_done = i - start_idx
            if bars_done > 0 and bars_done % max(1, total_bars // 20) == 0:
                _emit({
                    "type": "bar",
                    "symbol": symbol,
                    "bar": bars_done,
                    "total": total_bars,
                    "signals_so_far": len(signals),
                })
            try:
                # Create a slice of data up to this candle
                df_slice = df.iloc[:i+1].copy()
                bar_time = df_slice.iloc[-1]["time"]
                bar_date = bar_time.date()
                
                # Get historically correct daily levels
                correct_levels = None
                if df_daily is not None and len(df_daily) > 0:
                    prev_days = df_daily[df_daily["time"].dt.date < bar_date]
                    
                    if len(prev_days) > 0:
                        prev_day = prev_days.iloc[-1]
                        correct_levels = {
                            "date":  prev_day["time"].date(),
                            "high":  float(prev_day["high"]),
                            "low":   float(prev_day["low"]),
                            "range": float(prev_day["high"]) - float(prev_day["low"]),
                        }
                
                # Get historically correct weekly levels
                correct_weekly = None
                if df_weekly is not None and len(df_weekly) > 0:
                    prev_weeks = df_weekly[df_weekly["time"].dt.date < bar_date]
                    
                    if len(prev_weeks) > 0:
                        prev_week = prev_weeks.iloc[-1]
                        correct_weekly = {
                            "date":  prev_week["time"].date(),
                            "high":  float(prev_week["high"]),
                            "low":   float(prev_week["low"]),
                            "range": float(prev_week["high"]) - float(prev_week["low"]),
                        }
                
                # Temporarily override the strategy's fetch methods. We must
                # also override _get_daily_trend_bias because the live version
                # calls mt5.copy_rates_from_pos directly — which would leak the
                # CURRENT daily trend into every replayed bar (lookahead bias).
                original_fetch = self._strategy._fetch_m15
                original_levels = self._strategy.get_previous_day_levels
                original_daily_bias = getattr(
                    self._strategy, "_get_daily_trend_bias", None
                )

                def mock_fetch(sym):
                    return df_slice

                def mock_levels(sym):
                    if correct_levels:
                        return correct_levels
                    return {"date": bar_date, "high": 0.0, "low": 0.0, "range": 0.0}

                # Compute the historically-correct daily trend at bar_time:
                # use today's daily candle's open vs the current M15 close
                # (the bar we're replaying). Falls back to "neutral" if no
                # daily data is available.
                def mock_daily_trend(sym):
                    try:
                        if df_daily is None or len(df_daily) == 0:
                            return "neutral"
                        today_rows = df_daily[df_daily["time"].dt.date == bar_date]
                        if len(today_rows) == 0:
                            # Fall back to last completed daily candle
                            prev = df_daily[df_daily["time"].dt.date < bar_date]
                            if len(prev) == 0:
                                return "neutral"
                            row = prev.iloc[-1]
                            return ("bullish" if row["close"] > row["open"]
                                    else "bearish" if row["close"] < row["open"]
                                    else "neutral")
                        today_open = float(today_rows.iloc[0]["open"])
                        cur_close  = float(df_slice.iloc[-1]["close"])
                        if cur_close > today_open:
                            return "bullish"
                        if cur_close < today_open:
                            return "bearish"
                        return "neutral"
                    except Exception:
                        return "neutral"

                self._strategy._fetch_m15 = mock_fetch
                self._strategy.get_previous_day_levels = mock_levels
                if original_daily_bias is not None:
                    self._strategy._get_daily_trend_bias = mock_daily_trend

                # Also set these directly as a safety net. Always assign
                # (even when levels are missing) so a stale value from a
                # previous iteration can't leak into the current bar.
                if correct_levels:
                    self._strategy.previous_day_high = correct_levels["high"]
                    self._strategy.previous_day_low  = correct_levels["low"]
                else:
                    self._strategy.previous_day_high = None
                    self._strategy.previous_day_low  = None

                if correct_weekly:
                    self._strategy.previous_week_high = correct_weekly["high"]
                    self._strategy.previous_week_low  = correct_weekly["low"]
                else:
                    self._strategy.previous_week_high = None
                    self._strategy.previous_week_low  = None

                try:
                    # Generate signal at this point
                    raw_signal = self._strategy.generate_trade_signal(symbol)
                    
                    if raw_signal.get("signal") != "neutral":
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
                            component_scores=raw_signal.get("component_scores", {}),
                            environment=raw_signal.get("signal_source", "?").split("-")[0] if raw_signal.get("signal_source") else "?",
                        )
                        signals.append(sig)
                finally:
                    # Restore original methods
                    self._strategy._fetch_m15 = original_fetch
                    self._strategy.get_previous_day_levels = original_levels
                    if original_daily_bias is not None:
                        self._strategy._get_daily_trend_bias = original_daily_bias
            
            except Exception as e:
                log.error("Signal gen error at bar %d: %s", i, str(e)[:200])
                if i < 55:  # Print first few errors
                    import traceback
                    traceback.print_exc()
                continue
        
        log.info("Generated %d signals for %s", len(signals), symbol)
        _emit({"type": "signals_done", "symbol": symbol, "signals": len(signals)})
        return signals
    
    def simulate_trades(self, symbol: str, signals: List[Signal],
                        df: pd.DataFrame,
                        progress_cb: Optional[Callable[[dict], None]] = None) -> List[Trade]:
        """
        Simulate trade execution and exit with realistic assumptions.

        Entry model
        -----------
        Signal fires at close of bar i; order fills at open of bar i+1.
        MT5 OHLC is BID-priced, so:
            BUY  fills at open + full spread   (ask fill)
            SELL fills at open                 (bid fill)
        This matches the round-trip cost a retail account pays.

        Exit model
        ----------
        Two modes (`enable_trade_management`):

        A) FULL MANAGEMENT (default, matches live bot):
             Phase 1 ("full"):   whole position open at initial SL.
                - SL hit            → full loss, exit.
                - 1R target hit     → close `partial_close_ratio` (50%) at
                                      1R, trail SL to BE + buffer, switch
                                      to runner phase.
                - TP hit first      → rare (rr>1) but supported: full TP.
             Phase 2 ("runner"): (1 − ratio) of size open, SL at BE.
                - BE SL hit         → runner exits at BE (small gain from
                                      the buffer).
                - TP hit            → runner exits at TP.
                - Timeout           → runner exits at last bar's close.

             Total P&L = ratio * partial_pips + (1 − ratio) * runner_pips
             (both scaled by lot_size when converting to USD).

        B) RAW (enable_trade_management=False):
             Single-exit model — whole position exits at first SL/TP/timeout.
             Useful for signal-quality testing; will NOT match live P&L.

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

        times = df["time"].to_list()
        pip_size = get_pip_value(symbol)
        spread_pips = (self.spread_pips_override
                       if self.spread_pips_override is not None
                       else default_spread_pips(symbol))
        spread_price = spread_pips * pip_size
        usd_per_pip  = get_pip_usd_value(symbol, lot_size=self.lot_size)

        import bisect
        times_sorted = times  # df already sorted by time

        def _resolve_intrabar(c_open: float, tp: float, sl: float,
                              sl_price: float, tp_price: float) -> Tuple[float, str]:
            """Return (exit_price, reason) for a same-bar TP+SL collision."""
            if self.intrabar_policy == "optimistic":
                return tp_price, "tp"
            if self.intrabar_policy == "neutral":
                return (tp_price, "tp") if abs(tp - c_open) < abs(sl - c_open) \
                    else (sl_price, "sl")
            return sl_price, "sl"  # conservative (default)

        for idx, sig in enumerate(signals):
            if idx > 0 and idx % max(1, total_sigs // 10) == 0:
                _emit({
                    "type": "sim_progress",
                    "symbol": symbol,
                    "done": idx,
                    "total": total_sigs,
                })

            # Find the first bar whose time >= sig.ts; entry is the next bar.
            pos = bisect.bisect_left(times_sorted, sig.ts)
            if pos >= len(df) - 1:
                continue

            entry_idx  = pos + 1
            entry_row  = df.iloc[entry_idx]
            entry_open = float(entry_row["open"])
            entry_time = entry_row["time"]
            side       = str(sig.signal).upper()

            # MT5 OHLC is bid-priced → BUY fills at ask (+full spread),
            # SELL fills at bid (no adjust). Charges spread once per round
            # trip in the correct direction.
            if side == "BUY":
                entry_price = entry_open + spread_price
            elif side == "SELL":
                entry_price = entry_open
            else:
                continue

            sl_price = float(sig.stop_loss)
            tp_price = float(sig.take_profit)

            # Validate SL is on the correct side of entry (defensive —
            # bad signals otherwise create negative risk_pips / huge P&L).
            if side == "BUY" and sl_price >= entry_price:
                continue
            if side == "SELL" and sl_price <= entry_price:
                continue

            # Risk distance and 1R / BE targets
            if side == "BUY":
                risk_abs     = entry_price - sl_price
                one_r_target = entry_price + risk_abs * self.partial_close_rr
                be_sl        = entry_price + self.breakeven_buffer_pips * pip_size
            else:
                risk_abs     = sl_price - entry_price
                one_r_target = entry_price - risk_abs * self.partial_close_rr
                be_sl        = entry_price - self.breakeven_buffer_pips * pip_size

            if risk_abs <= 0:
                continue

            trade = Trade(signal=sig, entry_price=entry_price, entry_time=entry_time)

            max_exit_idx = min(entry_idx + self.max_trade_duration_bars,
                               len(df) - 1)

            # ── RAW mode ─────────────────────────────────────────────
            if not self.enable_trade_management:
                resolved = False
                for j in range(entry_idx, max_exit_idx + 1):
                    c = df.iloc[j]
                    c_high = float(c["high"]); c_low = float(c["low"])
                    c_time = c["time"]

                    if side == "BUY":
                        tp_hit = c_high >= tp_price
                        sl_hit = c_low  <= sl_price
                    else:
                        tp_hit = c_low  <= tp_price
                        sl_hit = c_high >= sl_price

                    if tp_hit and sl_hit:
                        exit_price, reason = _resolve_intrabar(
                            float(c["open"]), tp_price, sl_price, sl_price, tp_price)
                        trade.calculate_exit(exit_price, c_time, reason, self.lot_size)
                        resolved = True; break
                    if tp_hit:
                        trade.calculate_exit(tp_price, c_time, "tp", self.lot_size)
                        resolved = True; break
                    if sl_hit:
                        trade.calculate_exit(sl_price, c_time, "sl", self.lot_size)
                        resolved = True; break

                if not resolved:
                    close_candle = df.iloc[max_exit_idx]
                    exit_price = float(close_candle.get("close", entry_price))
                    trade.calculate_exit(exit_price, close_candle["time"],
                                         "timeout", self.lot_size)
                trades.append(trade)
                continue

            # ── FULL MANAGEMENT mode (default) ───────────────────────
            ratio          = self.partial_close_ratio
            phase          = "full"
            current_sl     = sl_price
            partial_hit    = False  # True once 1R partial-close has fired
            partial_pips   = 0.0    # pips realised from the partial leg
            exit_price     = None
            exit_time      = None
            exit_reason    = None
            runner_pips    = 0.0

            for j in range(entry_idx, max_exit_idx + 1):
                c = df.iloc[j]
                c_high = float(c["high"]); c_low = float(c["low"])
                c_open = float(c["open"]); c_time = c["time"]

                if phase == "full":
                    if side == "BUY":
                        sl_hit = c_low  <= current_sl
                        r1_hit = c_high >= one_r_target
                        tp_hit = c_high >= tp_price
                    else:
                        sl_hit = c_high >= current_sl
                        r1_hit = c_low  <= one_r_target
                        tp_hit = c_low  <= tp_price

                    # Initial-SL vs 1R collision → conservative SL
                    if sl_hit and r1_hit:
                        if side == "BUY":
                            runner_pips = (current_sl - entry_price) / pip_size
                        else:
                            runner_pips = (entry_price - current_sl) / pip_size
                        exit_price, exit_time, exit_reason = current_sl, c_time, "sl"
                        break

                    if sl_hit:
                        if side == "BUY":
                            runner_pips = (current_sl - entry_price) / pip_size
                        else:
                            runner_pips = (entry_price - current_sl) / pip_size
                        exit_price, exit_time, exit_reason = current_sl, c_time, "sl"
                        break

                    # TP before 1R is only possible if rr < 1 (unusual).
                    # Close the whole position at TP.
                    if tp_hit and not r1_hit:
                        if side == "BUY":
                            runner_pips = (tp_price - entry_price) / pip_size
                        else:
                            runner_pips = (entry_price - tp_price) / pip_size
                        exit_price, exit_time, exit_reason = tp_price, c_time, "tp"
                        break

                    if r1_hit:
                        # 50% closes at 1R, SL trails to BE, runner continues.
                        if side == "BUY":
                            partial_pips = (one_r_target - entry_price) / pip_size
                        else:
                            partial_pips = (entry_price - one_r_target) / pip_size
                        partial_hit = True
                        phase       = "runner"
                        current_sl  = be_sl
                        # Intentionally do NOT re-check runner exits on the
                        # trigger bar — we can't resolve post-1R intrabar
                        # order from OHLC. Resume next bar.
                        continue

                    # Nothing hit → next bar
                    continue

                # phase == "runner"
                if side == "BUY":
                    sl_hit = c_low  <= current_sl
                    tp_hit = c_high >= tp_price
                else:
                    sl_hit = c_high >= current_sl
                    tp_hit = c_low  <= tp_price

                if sl_hit and tp_hit:
                    xp, xr = _resolve_intrabar(c_open, tp_price, current_sl,
                                               current_sl, tp_price)
                    if side == "BUY":
                        runner_pips = (xp - entry_price) / pip_size
                    else:
                        runner_pips = (entry_price - xp) / pip_size
                    exit_price, exit_time, exit_reason = xp, c_time, "be+" + xr
                    break

                if tp_hit:
                    if side == "BUY":
                        runner_pips = (tp_price - entry_price) / pip_size
                    else:
                        runner_pips = (entry_price - tp_price) / pip_size
                    exit_price, exit_time, exit_reason = tp_price, c_time, "partial+tp"
                    break

                if sl_hit:
                    if side == "BUY":
                        runner_pips = (current_sl - entry_price) / pip_size
                    else:
                        runner_pips = (entry_price - current_sl) / pip_size
                    exit_price, exit_time, exit_reason = current_sl, c_time, "partial+be"
                    break

            # Timeout branch — close remainder at final bar's close
            if exit_price is None:
                close_candle = df.iloc[max_exit_idx]
                cx = float(close_candle.get("close", entry_price))
                if side == "BUY":
                    runner_pips = (cx - entry_price) / pip_size
                else:
                    runner_pips = (entry_price - cx) / pip_size
                exit_price  = cx
                exit_time   = close_candle["time"]
                exit_reason = ("partial+timeout" if phase == "runner"
                               else "timeout")

            # Compose the trade's final numbers. If the partial close
            # never fired (exit happened in "full" phase), the whole
            # position exits at runner_pips — no ratio weighting.
            if partial_hit:
                total_pips = ratio * partial_pips + (1.0 - ratio) * runner_pips
            else:
                total_pips = runner_pips
            trade.exit_price      = exit_price
            trade.exit_time       = exit_time
            trade.exit_reason     = exit_reason
            trade.profit_pips     = total_pips
            trade.profit          = total_pips * usd_per_pip
            trade.duration_minutes = int(
                (exit_time - entry_time).total_seconds() / 60)
            if trade.profit > 0:
                trade.outcome = "WIN"
            elif trade.profit < 0:
                trade.outcome = "LOSS"
            else:
                trade.outcome = "BE"
            # RR achieved relative to initial full-position risk
            if risk_abs > 0:
                risk_pips = risk_abs / pip_size
                trade.rr_achieved = abs(total_pips) / risk_pips
            else:
                trade.rr_achieved = 0.0

            if trade.profit_pips <= -20:
                log.warning(
                    "LARGE LOSS: %s %s | Entry: %.5f | SL: %.5f | TP: %.5f | "
                    "Exit: %.5f at %s | P&L: %+.1f pips | Reason: %s | Env: %s",
                    side, symbol, entry_price, sl_price, tp_price,
                    exit_price, exit_time, total_pips, exit_reason,
                    sig.environment
                )

            trades.append(trade)

        log.info("Simulated %d trades, closed %d  (spread=%.1fp, intrabar=%s, "
                 "management=%s)",
                 len(trades),
                 len([t for t in trades if t.outcome != "open"]),
                 spread_pips, self.intrabar_policy,
                 "on" if self.enable_trade_management else "off")
        _emit({"type": "sim_done", "symbol": symbol, "trades": len(trades)})
        return trades
    
    def analyze_early_exits(self, trades: List[Trade]) -> Dict:
        """Analyze trades that exit quickly (early reversals)."""
        early_exits = {}
        
        for threshold_minutes in [75, 150, 300]:  # 5, 10, 20 candles
            threshold_candles = threshold_minutes // 15
            early = []
            
            for t in trades:
                if t.duration_minutes <= threshold_minutes and t.outcome == "LOSS":
                    early.append(t)
            
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
        
        return early_exits
    
    def analyze_results(self, trades: List[Trade]) -> Dict:
        """Compute statistics and correlations."""
        if not trades:
            return {"error": "No trades"}
        
        outcomes = [t.outcome for t in trades]
        wins = sum(1 for t in trades if t.outcome == "WIN")
        losses = sum(1 for t in trades if t.outcome == "LOSS")
        win_rate = wins / len(trades) if trades else 0
        
        total_pnl = sum(t.profit for t in trades)
        avg_pnl = total_pnl / len(trades) if trades else 0
        
        win_pips = sum(t.profit_pips for t in trades if t.outcome == "WIN")
        loss_pips = sum(t.profit_pips for t in trades if t.outcome == "LOSS")
        avg_win = win_pips / wins if wins > 0 else 0
        avg_loss = loss_pips / losses if losses > 0 else 0
        
        early_exits = self.analyze_early_exits(trades)
        
        # Correlation: component scores vs outcome
        correlations = {}
        for component in ["structure_strength", "level_interaction", 
                         "momentum_quality", "spread_volatility", "environment_fit"]:
            scores = []
            outcomes_binary = []
            for t in trades:
                if component in t.signal.component_scores:
                    scores.append(t.signal.component_scores[component])
                    outcomes_binary.append(1 if t.outcome == "WIN" else 0)
            
            if len(scores) > 1:
                corr = np.corrcoef(scores, outcomes_binary)[0, 1]
                correlations[component] = float(corr) if not np.isnan(corr) else 0.0
        
        # Breakdown by signal quality
        by_quality = {}
        for quality in ["weak", "fair", "good", "strong"]:
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
        by_exit_reason = {}
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
        # Sort trades chronologically by exit_time (that's when P&L
        # lands). Walk the cumulative curve, track running peak, and
        # record the deepest peak-to-trough gap. MDD is reported as a
        # positive magnitude (dollar amount lost from peak) — the UI
        # formats it as a negative for visual consistency with a loss.
        #
        # We also emit `equity_curve` as a lightweight list of
        # {t, pnl, cum} points so the server can aggregate across pairs
        # into a portfolio-level drawdown (simple sum of per-pair MDDs
        # would over-count because pair drawdowns don't occur at the
        # same time in reality).
        sorted_trades = sorted(
            [t for t in trades if t.exit_time is not None],
            key=lambda t: t.exit_time,
        )
        running_pnl  = 0.0
        peak_pnl     = 0.0
        max_dd       = 0.0
        dd_trough_at = None  # timestamp of the worst drawdown point
        equity_curve: List[Dict[str, Any]] = []
        for t in sorted_trades:
            running_pnl += float(t.profit or 0.0)
            if running_pnl > peak_pnl:
                peak_pnl = running_pnl
            dd = peak_pnl - running_pnl
            if dd > max_dd:
                max_dd = dd
                dd_trough_at = t.exit_time
            # Represent timestamp as ISO so JSON round-trip is safe.
            try:
                ts_iso = t.exit_time.isoformat()
            except Exception:
                ts_iso = str(t.exit_time)
            equity_curve.append({
                "t":   ts_iso,
                "pnl": round(float(t.profit or 0.0), 2),
                "cum": round(running_pnl, 2),
            })

        max_drawdown     = round(max_dd, 2)
        max_drawdown_pct = (round((max_dd / peak_pnl) * 100.0, 2)
                            if peak_pnl > 1e-9 else 0.0)

        # ── Correlation-with-outcome breakdowns (for the UI charts) ─
        # These are the “what correlated with winning vs losing” surfaces.
        # Each dict is { bucket_label: {count, wins, losses, win_rate,
        # avg_pips, total_pnl} }. The UI plots them as bar charts.
        def _bucket_stats(group: List[Trade]) -> Dict[str, float]:
            n       = len(group)
            wins_   = sum(1 for t in group if t.outcome == "WIN")
            losses_ = sum(1 for t in group if t.outcome == "LOSS")
            pips    = [t.profit_pips for t in group]
            pnl     = sum(t.profit for t in group)
            return {
                "count":     n,
                "wins":      wins_,
                "losses":    losses_,
                "win_rate":  (wins_ / n) if n else 0.0,
                "avg_pips":  round(float(np.mean(pips)), 2) if pips else 0.0,
                "total_pnl": round(float(pnl), 2),
            }

        # BY ENVIRONMENT  (CHoCH-BUY@PDL, Continuation-SELL@PDH, …)
        # signal.source is free-form so we normalise here rather than in
        # the hot loop — stripping whitespace, upper-casing trailing "@XXX".
        def _env_key(src: str) -> str:
            if not src:
                return "?"
            s = str(src).strip()
            # Collapse variants like "CHoCH-BUY@PDL (aligned)" to the head
            head = s.split(" ")[0]
            return head or "?"

        env_groups: Dict[str, List[Trade]] = {}
        for t in trades:
            env_groups.setdefault(_env_key(t.signal.source), []).append(t)
        by_env = {k: _bucket_stats(g) for k, g in env_groups.items()}

        # BY SIDE (BUY vs SELL)
        by_side: Dict[str, Dict[str, float]] = {}
        for side in ("BUY", "SELL"):
            grp = [t for t in trades
                   if str(t.signal.signal).upper() == side]
            if grp:
                by_side[side] = _bucket_stats(grp)

        # BY HOUR OF DAY (UTC, entry time → 0..23)
        # Empty hours are omitted — the UI handles missing keys.
        hour_groups: Dict[int, List[Trade]] = {}
        for t in trades:
            et = t.entry_time
            if et is None:
                continue
            try:
                h = int(et.hour)
            except Exception:
                continue
            hour_groups.setdefault(h, []).append(t)
        by_hour = {str(h): _bucket_stats(g)
                   for h, g in sorted(hour_groups.items())}

        # BY DAY OF WEEK (Mon=0 .. Fri=4 in practice; FX closes Sat/Sun)
        dow_groups: Dict[int, List[Trade]] = {}
        for t in trades:
            et = t.entry_time
            if et is None:
                continue
            try:
                d = int(et.weekday())
            except Exception:
                continue
            dow_groups.setdefault(d, []).append(t)
        DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        by_dow = {DOW_NAMES[d]: _bucket_stats(g)
                  for d, g in sorted(dow_groups.items())
                  if 0 <= d < 7}

        # CONFIDENCE BUCKETS — fixed 10-percentile edges so two runs are
        # directly comparable even if one has a narrower confidence
        # distribution. Edges are on the 0–1 scale (signal.confidence).
        conf_edges = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
        conf_labels = ["<50", "50-60", "60-70", "70-80", "80-90", "90-100"]
        conf_groups: Dict[str, List[Trade]] = {lbl: [] for lbl in conf_labels}
        for t in trades:
            c = float(t.signal.confidence or 0.0)
            for i in range(len(conf_labels)):
                if conf_edges[i] <= c < conf_edges[i + 1]:
                    conf_groups[conf_labels[i]].append(t)
                    break
        confidence_buckets = {lbl: _bucket_stats(g)
                              for lbl, g in conf_groups.items() if g}

        # P&L DISTRIBUTION — histogram of profit_pips.
        # Bins are anchored to the observed range so wide runs don't end
        # up with one massive bin. We cap at 12 bins so the chart stays
        # legible; fewer if the dataset is thin.
        pnl_distribution: Dict[str, Any] = {}
        pip_values = [float(t.profit_pips) for t in trades
                      if t.profit_pips is not None]
        if pip_values:
            lo = min(pip_values); hi = max(pip_values)
            # Pad so boundaries aren't on extreme samples
            if hi - lo < 1e-6:
                lo -= 1.0; hi += 1.0
            pad = (hi - lo) * 0.05
            lo -= pad; hi += pad
            nbins = max(5, min(12, len(pip_values) // 5 or 5))
            counts, edges = np.histogram(pip_values, bins=nbins, range=(lo, hi))
            bins_out = []
            for i in range(nbins):
                e0 = float(edges[i]); e1 = float(edges[i + 1])
                mid = (e0 + e1) / 2.0
                bins_out.append({
                    "x0":    round(e0, 2),
                    "x1":    round(e1, 2),
                    "mid":   round(mid, 2),
                    "count": int(counts[i]),
                    "sign":  "win" if mid > 0 else ("loss" if mid < 0 else "be"),
                })
            pnl_distribution = {
                "bins":   bins_out,
                "min":    round(lo, 2),
                "max":    round(hi, 2),
                "n":      len(pip_values),
                "median": round(float(np.median(pip_values)), 2),
            }

        return {
            "total_trades": len(trades),
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "total_pnl": round(total_pnl, 2),
            "avg_pnl_per_trade": round(avg_pnl, 2),
            "avg_win_pips": round(avg_win, 1),
            "avg_loss_pips": round(avg_loss, 1),
            # Profit factor: gross win pips / |gross loss pips|. When there
            # are no losses, conventional reporting uses "infinity"; we send
            # a high sentinel (999) which the UI renders as "∞".
            "profit_factor": (round(abs(win_pips / loss_pips), 2)
                              if loss_pips != 0
                              else (999.0 if win_pips > 0 else 0.0)),
            "avg_rr_achieved": round(np.mean([t.rr_achieved for t in trades if t.rr_achieved > 0]), 2),
            "avg_duration_minutes": round(avg_duration, 1),
            "max_drawdown":     max_drawdown,
            "max_drawdown_pct": max_drawdown_pct,
            "equity_curve":     equity_curve,
            "component_correlations": correlations,
            "by_quality": by_quality,
            "by_exit_reason": by_exit_reason,
            "by_env": by_env,
            "by_side": by_side,
            "by_hour": by_hour,
            "by_dow": by_dow,
            "confidence_buckets": confidence_buckets,
            "pnl_distribution": pnl_distribution,
            "early_exits": early_exits,
            # Make modelling assumptions explicit so the UI / consumer
            # can show what the numbers actually represent.
            "assumptions": {
                "lot_size":            self.lot_size,
                "spread_pips":         (self.spread_pips_override
                                        if self.spread_pips_override is not None
                                        else None),
                "intrabar_policy":     self.intrabar_policy,
                "entry_fill":          ("bid-based OHLC: BUY pays full spread, "
                                        "SELL fills at open"),
                "trade_management":    ("1R partial close "
                                        f"({int(self.partial_close_ratio*100)}%) "
                                        f"+ SL→BE ({self.breakeven_buffer_pips}p)"
                                        if self.enable_trade_management
                                        else "raw (single SL/TP exit)"),
                "pip_usd_model":       "JPY pairs converted at ~150 USDJPY",
                "commission":          "not modelled",
                "swap":                "not modelled",
                "lookahead":           "last-forming bar skipped",
            },
        }
    
    def run(self, symbol: str, days: int = 30,
            progress_cb: Optional[Callable[[dict], None]] = None) -> Dict:
        """Run complete backtest. Optional progress_cb receives stage/bar events."""
        log.info("=" * 60)
        log.info("Backtest: %s (%d days)", symbol, days)
        log.info("=" * 60)

        def _emit(payload: dict) -> None:
            if progress_cb is not None:
                try:
                    progress_cb(payload)
                except Exception:
                    pass

        try:
            # Anchor point for "the last N days" — computed ONCE so every
            # stage (data fetch, window clip, reporting) agrees on the
            # same "now". Any drift here would show up as 1-bar off-by-one
            # errors between the fetched window and the reported period.
            now_utc      = datetime.now(timezone.utc)
            window_start = now_utc - timedelta(days=int(days))

            _emit({"type": "stage", "symbol": symbol, "stage": "fetching_data"})
            df = self.fetch_data(symbol, days=days)
            if df is None or len(df) < 50:
                log.error("Could not fetch sufficient data for %s", symbol)
                return {"error": f"Insufficient data for {symbol}", "symbol": symbol}

            self.candle_history[symbol] = df

            _emit({"type": "stage", "symbol": symbol, "stage": "generating_signals"})
            # Generate signals, clipping evaluation to bars inside the
            # rolling window. Warm-up bars (2 days of prefix) still feed
            # into the strategy's lookback but do not produce signals.
            signals = self.generate_signals(symbol, df,
                                            progress_cb=progress_cb,
                                            window_start=window_start)
            if not signals:
                log.warning("No signals generated for %s", symbol)
                return {
                    "symbol": symbol,
                    "period": (f"{window_start.date()} to "
                               f"{now_utc.date()}"),
                    "total_trades": 0,
                    "error": "No signals generated",
                }

            self.signals.extend(signals)

            _emit({"type": "stage", "symbol": symbol, "stage": "simulating_trades"})
            trades = self.simulate_trades(symbol, signals, df, progress_cb=progress_cb)
            self.trades.extend(trades)

            _emit({"type": "stage", "symbol": symbol, "stage": "analyzing"})
            # Analyze
            results = self.analyze_results(trades)
            results["symbol"] = symbol
            # Report the rolling window the user actually requested, not
            # the wider fetched slice (which includes 2-day warm-up).
            results["period"] = (f"{window_start.date()} to "
                                 f"{now_utc.date()}")
            results["window_start_utc"] = window_start.isoformat()
            results["window_end_utc"]   = now_utc.isoformat()
            results["requested_days"]   = int(days)

            # ML diagnostic: feature importance on the trades we just closed
            fi = self.compute_feature_importance(trades)
            if fi is not None:
                results["feature_importance"] = fi

            return results

        except Exception as e:
            log.exception("Backtest error: %s", e)
            return {"error": str(e), "symbol": symbol}

    # ---------------------------------------------------------------- #
    # Feature-importance diagnostic (sklearn — optional)               #
    # ---------------------------------------------------------------- #
    def compute_feature_importance(self, trades: List[Trade]) -> Optional[Dict[str, Any]]:
        """
        Train a RandomForest on trade-level features and return feature
        importances. Purely diagnostic — does NOT feed back into live
        trading. Returns None if sklearn is unavailable or the dataset is
        too thin to be meaningful.

        Features are derived from fields that actually vary per trade:
        setup one-hots (CHoCH vs Continuation, level anchored to PDH/PDL),
        hour/day of entry, confidence, risk-reward ratio, risk size in
        pips. Metric is balanced accuracy (robust to class imbalance).
        """
        if not HAS_SKLEARN:
            return {"error": "sklearn not installed (pip install scikit-learn)"}

        decided = [t for t in trades if t.outcome in ("WIN", "LOSS")]
        if len(decided) < 20:
            return {"error": f"Not enough decided trades for ML ({len(decided)}<20)"}

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

            row = [
                float(t.signal.confidence or 0.0),
                float(t.signal.rr_ratio or 0.0),
                float(risk_pips),
                1.0 if str(t.signal.signal).upper() == "BUY" else 0.0,
                1.0 if "CHOCH" in src else 0.0,
                1.0 if "CONTINUATION" in src else 0.0,
                1.0 if src.endswith("@PDH") else 0.0,
                1.0 if src.endswith("@PDL") else 0.0,
                float(hour),
                float(dow),
                1.0 if "JPY" in str(t.signal.symbol).upper() else 0.0,
            ]
            X.append(row)
            y.append(1 if t.outcome == "WIN" else 0)

        X_arr = np.array(X, dtype=float)
        y_arr = np.array(y, dtype=int)

        if len(set(y_arr.tolist())) < 2:
            only = "WIN" if y_arr[0] == 1 else "LOSS"
            return {"error": f"All trades ended in {only} — cannot fit classifier"}

        # Drop zero-variance columns so they don't clutter the chart with 0.000.
        # Use a small epsilon to also drop near-constant columns from float noise.
        variances = X_arr.var(axis=0)
        keep_idx = [i for i, v in enumerate(variances) if v > 1e-9]
        if len(keep_idx) < 2:
            return {"error": "Fewer than 2 features have variance — nothing to learn."}
        X_arr = X_arr[:, keep_idx]
        active_features = [FEATURES[i] for i in keep_idx]

        try:
            clf = RandomForestClassifier(
                n_estimators=300,
                max_depth=5,
                min_samples_leaf=3,
                random_state=42,
                oob_score=True,
                bootstrap=True,
                class_weight="balanced",
                n_jobs=1,
            )
            clf.fit(X_arr, y_arr)

            # Build per-feature importance list
            importances = [
                {"feature": name, "importance": round(float(imp), 4)}
                for name, imp in zip(active_features, clf.feature_importances_)
            ]
            importances.sort(key=lambda r: r["importance"], reverse=True)

            baseline_wr = float(y_arr.mean())

            # Balanced accuracy via OOB predictions (robust to class imbalance).
            # oob_decision_function_ gives per-sample class probabilities for
            # out-of-bag trees; pick argmax to get predicted class.
            balanced = None
            win_precision = None
            win_recall = None
            try:
                from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score
                oob_proba = getattr(clf, "oob_decision_function_", None)
                if oob_proba is not None:
                    valid = ~np.isnan(oob_proba).any(axis=1)
                    if valid.sum() >= 10:
                        y_pred = oob_proba[valid].argmax(axis=1)
                        y_true = y_arr[valid]
                        balanced = float(balanced_accuracy_score(y_true, y_pred))
                        # Precision / recall for the WIN class (label=1)
                        if (y_pred == 1).any():
                            win_precision = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
                        win_recall = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
            except Exception:
                pass

            raw_oob = float(getattr(clf, "oob_score_", float("nan")))

            return {
                "n_trades": len(decided),
                "baseline_win_rate": round(baseline_wr, 4),
                "balanced_accuracy": round(balanced, 4) if balanced is not None else None,
                "win_precision": round(win_precision, 4) if win_precision is not None else None,
                "win_recall": round(win_recall, 4) if win_recall is not None else None,
                # lift: how much better than random guessing on a balanced dataset.
                # 0.5 = coin flip. Positive means the model has real signal.
                "lift_vs_random": round(balanced - 0.5, 4) if balanced is not None else None,
                # Keep these for backwards-compat with older clients that read them
                "oob_accuracy": round(raw_oob, 4) if not np.isnan(raw_oob) else None,
                "lift_vs_baseline": None,  # deprecated — was misleading on imbalanced data
                "importances": importances,
                "model": "RandomForestClassifier(n=300, depth=5, class_weight=balanced)",
                "note": "Features derived from signal source + timing + confidence; "
                        "zero-variance features dropped.",
            }
        except Exception as exc:
            log.exception("feature-importance training failed")
            return {"error": f"sklearn training failed: {exc}"}
    
    def print_results(self, results: Dict) -> None:
        """Pretty-print backtest results."""
        print("\n" + "=" * 70)
        print(f"  {results.get('symbol', '?')} — {results.get('period', '?')}")
        print("=" * 70)
        print(f"Trades:       {results['total_trades']} total "
              f"({results['wins']} wins, {results['losses']} losses)")
        print(f"Win Rate:     {results['win_rate']:.1%}")
        print(f"P&L:          ${results['total_pnl']:+.2f} "
              f"(${results['avg_pnl_per_trade']:+.2f}/trade)")
        print(f"Avg Win/Loss: +{results['avg_win_pips']:.0f}p / {results['avg_loss_pips']:.0f}p")
        print(f"Profit Factor: {results['profit_factor']:.2f}x")
        print(f"Avg R:R Achieved: {results['avg_rr_achieved']:.2f}")
        print(f"Avg Trade Duration: {results.get('avg_duration_minutes', 0):.0f} minutes")
        
        if results.get("component_correlations"):
            print("\nComponent Score Correlations with Wins:")
            for comp, corr in results["component_correlations"].items():
                direction = "↑" if corr > 0.3 else "↓" if corr < -0.3 else "→"
                print(f"  {direction} {comp:25} {corr:+.3f}")
        
        if results.get("by_quality"):
            print("\nPerformance by Signal Quality:")
            for quality in ["weak", "fair", "good", "strong"]:
                if quality in results["by_quality"]:
                    data = results["by_quality"][quality]
                    print(f"  {quality:8} ({data['count']:3} trades): "
                          f"{data['win_rate']:.0%} WR, "
                          f"${data['avg_pnl']:+.1f} avg, "
                          f"{data['avg_rr']:.2f} RR")
        
        if results.get("by_exit_reason"):
            print("\nPerformance by Exit Reason:")
            for reason in ["tp", "sl", "timeout"]:
                if reason in results["by_exit_reason"]:
                    data = results["by_exit_reason"][reason]
                    reason_name = {"tp": "Take Profit", "sl": "Stop Loss", "timeout": "Timeout"}.get(reason, reason)
                    print(f"  {reason_name:12} ({data['count']:3} trades): "
                          f"{data['win_rate']:.0%} WR, "
                          f"${data['avg_pnl']:+.1f} avg")
        
        if results.get("early_exits"):
            print("\n🚩 Early Exit Analysis (Quick Reversals):")
            for threshold, data in results["early_exits"].items():
                candles = threshold.split("_")[1]
                print(f"  Losses within {candles} candles: {data['count']} trades "
                      f"({data['pct_of_losses']:.0%} of all losses), "
                      f"avg {data['avg_loss_pips']:.0f} pips")
                if data["exit_reasons"].get("sl"):
                    print(f"    └─ {data['exit_reasons']['sl']} via SL, "
                          f"{data['exit_reasons'].get('timeout', 0)} via timeout")
        
        print("=" * 70 + "\n")
    
    def export_trades(self, symbol: str, filepath: str = None) -> str:
        """Export trade log to JSON."""
        if filepath is None:
            filepath = f"backtest_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        trades_data = []
        for t in self.trades:
            if t.signal.symbol == symbol:
                trades_data.append({
                    "entry_time": t.entry_time.isoformat(),
                    "exit_time": t.exit_time.isoformat() if t.exit_time else None,
                    "signal": t.signal.signal,
                    "source": t.signal.source,
                    "entry": round(t.entry_price, 5),
                    "exit": round(t.exit_price or t.entry_price, 5),
                    "sl": round(t.signal.stop_loss, 5),
                    "tp": round(t.signal.take_profit, 5),
                    "pips": round(t.profit_pips, 1),
                    "pnl": round(t.profit, 2),
                    "outcome": t.outcome,
                    "rr_achieved": round(t.rr_achieved, 2),
                    "exit_reason": t.exit_reason,
                    "signal_quality": t.signal.quality,
                    "components": t.signal.component_scores,
                })
        
        Path(filepath).write_text(json.dumps(trades_data, indent=2))
        log.info("Exported %d trades to %s", len(trades_data), filepath)
        return filepath


# ============================================================ #
# ENTRY POINT                                                  #
# ============================================================ #

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="AI_Pro backtester — validate signal generation"
    )
    parser.add_argument("--symbol", default="EURUSD", help="Symbol to backtest")
    parser.add_argument("--days", type=int, default=7, help="Days of history")
    parser.add_argument("--lot-size", type=float, default=1.0, help="Lot size per trade (e.g., 0.50)")
    parser.add_argument("--export", action="store_true", help="Export trade log")
    parser.add_argument("--plot", action="store_true", help="Plot results (requires matplotlib)")
    
    args = parser.parse_args()
    
    if not HAS_AI_PRO:
        print("\n" + "!"*70)
        print("ERROR: AI_Pro module not found in current directory")
        print("Make sure ai_pro.py is in the same folder as backtest.py")
        print("!"*70 + "\n")
        return
    
    backtest = AI_ProBacktester(lot_size=args.lot_size)
    results = backtest.run(args.symbol, days=args.days)
    
    if "error" in results:
        log.error("Backtest failed: %s", results["error"])
        print(f"\nError: {results['error']}")
        print("\nTroubleshooting:")
        print("1. Is MT5 terminal running?")
        print("2. Have you configured login/password in the dashboard?")
        print("3. Check that you have at least 50 bars of M15 data available")
        return
    
    backtest.print_results(results)
    
    if args.export:
        filepath = backtest.export_trades(args.symbol)
        log.info("Trade log exported: %s", filepath)
    
    if args.plot and HAS_MATPLOTLIB:
        log.info("Plotting not yet implemented (contribute?)")
    elif args.plot:
        log.warning("matplotlib not installed; skipping plots")




if __name__ == "__main__":
    main()
