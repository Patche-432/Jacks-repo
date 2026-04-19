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
        
        # Determine pip value based on symbol (JPY pairs use 0.01, others use 0.0001)
        pip_value = get_pip_value(self.signal.symbol)
        
        if self.signal.signal == "BUY":
            self.profit_pips = (exit_price - self.entry_price) / pip_value
            risk_pips = (self.entry_price - self.signal.stop_loss) / pip_value
        else:  # SELL
            self.profit_pips = (self.entry_price - exit_price) / pip_value
            risk_pips = (self.signal.stop_loss - self.entry_price) / pip_value
        
        # Calculate profit based on lot size: $10 per pip per 1.0 lot
        self.profit = self.profit_pips * (10.0 * lot_size)
        
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
    ):
        self.symbols = symbols or ["EURUSD"]
        self.lookback_bars = lookback_bars
        self.max_trade_duration_bars = max_trade_duration_bars
        self.risk_per_trade_pips = risk_per_trade_pips
        self.lot_size = lot_size
        
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
        """Fetch M15 historical data for specified number of days."""
        self._ensure_mt5()
        
        if self._mt5 is None:
            log.warning("MT5 not available; cannot fetch live data")
            return None
        
        try:
            # Calculate bars needed: ~96 bars per trading day (24h * 60min / 15min)
            # But use higher multiplier to account for market hours only
            bars_needed = max(self.lookback_bars, days * 96)
            
            rates = self._mt5.copy_rates_from_pos(
                symbol, self._mt5.TIMEFRAME_M15, 0, bars_needed
            )
            if rates is None or len(rates) < 10:
                log.warning("Not enough data for %s (requested %d bars)", symbol, bars_needed)
                return None
            
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df = df.sort_values("time").reset_index(drop=True)
            
            log.info("Fetched %d M15 candles for %s (covering ~%d days)", len(df), symbol, len(df) // 96)
            return df
        
        except Exception as e:
            log.error("Data fetch error [%s]: %s", symbol, e)
            return None
    
    def _fetch_daily(self, symbol: str, bars: int = 90) -> pd.DataFrame:
        """Fetch daily historical data for the backtest period."""
        if self._mt5 is None:
            log.warning("MT5 not available; cannot fetch daily data")
            return pd.DataFrame()
        
        try:
            rates = self._mt5.copy_rates_from_pos(
                symbol, self._mt5.TIMEFRAME_D1, 0, bars
            )
            if rates is None or len(rates) < 1:
                log.warning("Not enough daily data for %s", symbol)
                return pd.DataFrame()
            
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df = df.sort_values("time").reset_index(drop=True)
            
            log.info("Fetched %d daily candles for %s", len(df), symbol)
            return df
        
        except Exception as e:
            log.error("Daily data fetch error [%s]: %s", symbol, e)
            return pd.DataFrame()
    
    def _fetch_weekly(self, symbol: str, bars: int = 52) -> pd.DataFrame:
        """Fetch weekly historical data for the backtest period."""
        if self._mt5 is None:
            log.warning("MT5 not available; cannot fetch weekly data")
            return pd.DataFrame()
        
        try:
            rates = self._mt5.copy_rates_from_pos(
                symbol, self._mt5.TIMEFRAME_W1, 0, bars
            )
            if rates is None or len(rates) < 1:
                log.warning("Not enough weekly data for %s", symbol)
                return pd.DataFrame()
            
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df = df.sort_values("time").reset_index(drop=True)
            
            log.info("Fetched %d weekly candles for %s", len(df), symbol)
            return df
        
        except Exception as e:
            log.error("Weekly data fetch error [%s]: %s", symbol, e)
            return pd.DataFrame()
    
    def generate_signals(self, symbol: str, df: pd.DataFrame,
                         progress_cb: Optional[Callable[[dict], None]] = None) -> List[Signal]:
        """Generate signals for each candle in history."""
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
        total_bars = max(1, len(df) - 50)
        _emit({"type": "signals_begin", "symbol": symbol, "total_bars": total_bars})

        # Simulate walking through history
        for i in range(50, len(df)):  # Start after 50 candles (need lookback)
            # Emit progress every ~5% of the run so the UI can update live
            if (i - 50) > 0 and (i - 50) % max(1, total_bars // 20) == 0:
                _emit({
                    "type": "bar",
                    "symbol": symbol,
                    "bar": i - 50,
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
                
                # Temporarily override the strategy's fetch methods
                original_fetch = self._strategy._fetch_m15
                original_levels = self._strategy.get_previous_day_levels
                
                def mock_fetch(sym):
                    return df_slice
                
                def mock_levels(sym):
                    if correct_levels:
                        return correct_levels
                    return {"date": bar_date, "high": 0.0, "low": 0.0, "range": 0.0}
                
                self._strategy._fetch_m15 = mock_fetch
                self._strategy.get_previous_day_levels = mock_levels
                
                # Also set these directly as a safety net
                if correct_levels:
                    self._strategy.previous_day_high = correct_levels["high"]
                    self._strategy.previous_day_low = correct_levels["low"]
                
                # Set weekly levels as attributes if available
                if correct_weekly:
                    self._strategy.previous_week_high = correct_weekly["high"]
                    self._strategy.previous_week_low = correct_weekly["low"]
                
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
        """Simulate trade execution and exit."""
        trades = []

        def _emit(payload: dict) -> None:
            if progress_cb is not None:
                try:
                    progress_cb(payload)
                except Exception:
                    pass

        total_sigs = max(1, len(signals))
        _emit({"type": "sim_begin", "symbol": symbol, "total_signals": total_sigs})

        for idx, sig in enumerate(signals):
            if idx > 0 and idx % max(1, total_sigs // 10) == 0:
                _emit({
                    "type": "sim_progress",
                    "symbol": symbol,
                    "done": idx,
                    "total": total_sigs,
                })
            # Find candle index closest to signal time
            entry_idx = None
            for i, row in df.iterrows():
                if row["time"] >= sig.ts:
                    entry_idx = i
                    break
            
            if entry_idx is None or entry_idx >= len(df) - 1:
                continue  # Can't simulate trade
            
            # Simulate trade execution on next candle
            entry_candle = df.iloc[entry_idx]
            entry_time = entry_candle["time"]
            entry_price = entry_candle.get("close", sig.entry_price)
            
            trade = Trade(
                signal=sig,
                entry_price=entry_price,
                entry_time=entry_time,
            )
            
            # Look for exit (TP, SL, or timeout)
            max_exit_idx = min(entry_idx + self.max_trade_duration_bars, len(df) - 1)
            
            for j in range(entry_idx + 1, max_exit_idx + 1):
                candle = df.iloc[j]
                candle_high = candle["high"]
                candle_low = candle["low"]
                candle_time = candle["time"]
                
                # Check TP hit
                if sig.signal == "BUY" and candle_high >= sig.take_profit:
                    trade.calculate_exit(sig.take_profit, candle_time, "tp", self.lot_size)
                    break
                elif sig.signal == "SELL" and candle_low <= sig.take_profit:
                    trade.calculate_exit(sig.take_profit, candle_time, "tp", self.lot_size)
                    break
                
                # Check SL hit
                if sig.signal == "BUY" and candle_low <= sig.stop_loss:
                    trade.calculate_exit(sig.stop_loss, candle_time, "sl", self.lot_size)
                    break
                elif sig.signal == "SELL" and candle_high >= sig.stop_loss:
                    trade.calculate_exit(sig.stop_loss, candle_time, "sl", self.lot_size)
                    break
            
            # If no exit, close at timeout
            if trade.outcome == "open":
                close_candle = df.iloc[max_exit_idx]
                exit_price = close_candle.get("close", entry_price)
                trade.calculate_exit(exit_price, close_candle["time"], "timeout", self.lot_size)
            
            # Log large losses for investigation
            if trade.profit_pips <= -20:
                log.warning(
                    "LARGE LOSS DETECTED: %s %s | Entry: %.5f | SL: %.5f | TP: %.5f | "
                    "Exit: %.5f at %s | P&L: %+.0f pips | Reason: %s | Env: %s",
                    sig.signal, symbol, entry_price, sig.stop_loss, sig.take_profit,
                    trade.exit_price, trade.exit_time, trade.profit_pips, trade.exit_reason,
                    sig.environment
                )
            
            trades.append(trade)
        
        log.info("Simulated %d trades, closed %d", len(trades),
                 len([t for t in trades if t.outcome != "open"]))
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
        
        return {
            "total_trades": len(trades),
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "total_pnl": round(total_pnl, 2),
            "avg_pnl_per_trade": round(avg_pnl, 2),
            "avg_win_pips": round(avg_win, 1),
            "avg_loss_pips": round(avg_loss, 1),
            "profit_factor": round(abs(win_pips / loss_pips), 2) if loss_pips != 0 else 0,
            "avg_rr_achieved": round(np.mean([t.rr_achieved for t in trades if t.rr_achieved > 0]), 2),
            "avg_duration_minutes": round(avg_duration, 1),
            "component_correlations": correlations,
            "by_quality": by_quality,
            "by_exit_reason": by_exit_reason,
            "early_exits": early_exits,
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
            _emit({"type": "stage", "symbol": symbol, "stage": "fetching_data"})
            # Fetch data
            df = self.fetch_data(symbol, days=days)
            if df is None or len(df) < 50:
                log.error("Could not fetch sufficient data for %s", symbol)
                return {"error": f"Insufficient data for {symbol}", "symbol": symbol}

            self.candle_history[symbol] = df

            _emit({"type": "stage", "symbol": symbol, "stage": "generating_signals"})
            # Generate signals
            signals = self.generate_signals(symbol, df, progress_cb=progress_cb)
            if not signals:
                log.warning("No signals generated for %s", symbol)
                return {
                    "symbol": symbol,
                    "period": f"{df.iloc[0]['time'].date()} to {df.iloc[-1]['time'].date()}",
                    "total_trades": 0,
                    "error": "No signals generated",
                }

            self.signals.extend(signals)

            _emit({"type": "stage", "symbol": symbol, "stage": "simulating_trades"})
            # Simulate trades
            trades = self.simulate_trades(symbol, signals, df, progress_cb=progress_cb)
            self.trades.extend(trades)

            _emit({"type": "stage", "symbol": symbol, "stage": "analyzing"})
            # Analyze
            results = self.analyze_results(trades)
            results["symbol"] = symbol
            results["period"] = f"{df.iloc[0]['time'].date()} to {df.iloc[-1]['time'].date()}"

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

        # Drop zero-variance columns so they don't clutter the chart with 0.000
        variances = X_arr.var(axis=0)
        keep_idx = [i for i, v in enumerate(variances) if v > 0]
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
