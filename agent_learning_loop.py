"""
Agent Learning Loop — Apply backtest insights to live signal generation.

This module provides the learning loop that:
1. Loads backtest feature importance weights from backtest_insights.json
2. Auto-reloads when the file is updated by a fresh backtest run
3. Applies weights to dynamically adjust per-pair confidence scores
4. Makes the agent adaptive per pair / hour / context

Pipeline shape (end-to-end):
    odl/backtest.py   — runs backtest, computes feature importances,
                        writes backtest_insights.json (atomic temp+rename)
    backtest_insights.json — repo-root file, keyed by symbol
    agent_learning_loop.AgentLearningLoop — reads the file, watches mtime
    ai_pro.AgentZeroBot.generate_trade_signal — calls apply_learned_weights
                        once per ENV (ENV1..ENV4) before publishing the signal

If the insights file is missing or empty (no backtest has been run yet) the
loop simply returns base_confidence unchanged — the strategy keeps working
exactly as it did before. The learning loop is purely additive.
"""

import json
import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any

log = logging.getLogger("agent_learning")


class AgentLearningLoop:
    """Manages loading and applying learned weights from backtests.

    Thread-safety note: the live trading bot calls apply_learned_weights()
    from each pair's poll cycle. The mtime check + dict swap is a single
    Python attribute assignment, which is atomic at the GIL level — readers
    will see either the old dict or the new dict, never a half-merged one.
    No explicit lock is needed.
    """

    # ── Sentinel showing the loop has never read the file ───────────────
    _MTIME_NEVER_LOADED = -1.0

    def __init__(self, root_path: Optional[str] = None):
        self.insights: Dict[str, Dict[str, float]] = {}
        if root_path is None:
            root_path = os.path.dirname(os.path.abspath(__file__))
        self.insights_file = os.path.join(root_path, "backtest_insights.json")
        # Track the file's last-seen mtime so we can detect a fresh
        # backtest write and reload without a manual reload_insights() call.
        self._last_mtime: float = self._MTIME_NEVER_LOADED
        self.load_insights()

    # ------------------------------------------------------------------ #
    # File I/O                                                           #
    # ------------------------------------------------------------------ #

    def _maybe_reload_from_disk(self) -> None:
        """Reload insights from disk if the file's mtime has changed.

        Called from apply_learned_weights() before every adjustment so a
        backtest run that finishes mid-session is picked up by the next
        signal review without restarting the bot. The cost is one os.stat
        per call — negligible compared to the rest of a poll cycle.
        """
        try:
            st = os.stat(self.insights_file)
        except FileNotFoundError:
            # No backtest has been run yet, or the file was deleted. Make
            # sure we don't keep stale insights in memory either.
            if self.insights:
                log.info("backtest_insights.json disappeared — clearing insights")
                self.insights = {}
                self._last_mtime = self._MTIME_NEVER_LOADED
            return
        except Exception as exc:
            log.debug("insights stat failed: %s", exc)
            return

        if st.st_mtime > self._last_mtime:
            self.load_insights()

    def load_insights(self) -> None:
        """Load feature importance weights from backtest_insights.json.

        Records the file's mtime so future calls to _maybe_reload_from_disk
        only re-read when the backtest writes a new version.
        """
        try:
            if os.path.exists(self.insights_file):
                with open(self.insights_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.insights = data if isinstance(data, dict) else {}
                try:
                    self._last_mtime = os.stat(self.insights_file).st_mtime
                except Exception:
                    pass
                if self.insights:
                    log.info("Loaded backtest insights for %d pairs (%s)",
                             len(self.insights),
                             ", ".join(sorted(self.insights.keys())))
            else:
                log.debug("No insights file yet at %s", self.insights_file)
                self.insights = {}
                self._last_mtime = self._MTIME_NEVER_LOADED
        except Exception as exc:
            # Corrupt JSON or permission error — log and fall back to "no
            # insights". The bot keeps trading on base confidences.
            log.warning("Could not load backtest insights: %s", exc)
            self.insights = {}

    def reload_insights(self) -> None:
        """Force a reload from disk (e.g. immediately after a backtest)."""
        self.load_insights()

    # ------------------------------------------------------------------ #
    # The actual adjustment                                              #
    # ------------------------------------------------------------------ #

    def apply_learned_weights(self, symbol: str, base_confidence: int,
                              current_price: float, entry: float, sl: float,
                              poc: Optional[float] = None) -> int:
        """
        Dynamically adjust confidence using learned feature weights.

        Args:
            symbol:           Trading pair (e.g., "GBPJPY")
            base_confidence:  Base confidence (85 for CHoCH, 65–85 for Continuation)
            current_price:    Current market price (for hour/POC context)
            entry:            Entry price for the signal
            sl:               Stop loss price
            poc:              Point of Control from volume profile (optional)

        Returns:
            Adjusted confidence in the [60, 95] range. When no insights
            are available for `symbol`, returns base_confidence unchanged.
        """
        # Always check for a fresh insights file first — cheap, and means a
        # newly-finished backtest takes effect on the very next signal.
        self._maybe_reload_from_disk()

        if symbol not in self.insights:
            return base_confidence

        insights = self.insights[symbol]
        bonus = 0.0

        # ──── HOUR OF DAY ────────────────────────────────────────────
        # If hour_of_day importance > 0.15, it's a strong predictor for
        # this pair — reward signals that fire during typical
        # London/NY overlap hours.
        if insights.get("hour_of_day", 0) > 0.15:
            try:
                current_hour = datetime.utcnow().hour
                optimal_hours = {8, 9, 10, 11, 13, 14, 15, 16}
                if current_hour in optimal_hours:
                    bonus += insights["hour_of_day"] * 12
            except Exception as exc:
                log.debug("hour-of-day check failed: %s", exc)

        # ──── DISTANCE TO POC ────────────────────────────────────────
        # If price is well-aligned with the day's volume profile,
        # boost confidence. JPY pairs use 0.01 pip size.
        if insights.get("dist_to_poc_pips", 0) > 0.15 and poc is not None:
            try:
                pip_size = 0.01 if "JPY" in symbol else 0.0001
                dist_pips = abs(current_price - poc) / pip_size
                if dist_pips < 50:
                    bonus += insights["dist_to_poc_pips"] * 10
            except Exception as exc:
                log.debug("POC distance check failed: %s", exc)

        # ──── RISK PIPS ──────────────────────────────────────────────
        # Award when SL distance is in the typical "good range" for
        # the pair, penalise when it's too tight (likely to get clipped
        # by spread noise).
        if insights.get("risk_pips", 0) > 0.15:
            try:
                pip_size = 0.01 if "JPY" in symbol else 0.0001
                risk_pips = abs(entry - sl) / pip_size
                if 40 <= risk_pips <= 100:
                    bonus += insights["risk_pips"] * 8
                elif risk_pips < 40:
                    bonus -= 5
            except Exception as exc:
                log.debug("risk-pips check failed: %s", exc)

        # ──── APPLY BONUS ────────────────────────────────────────────
        adjusted = base_confidence + int(round(bonus))
        # Clamp to safe range so no single feature can push confidence
        # below the strategy's reject threshold or above 95%.
        return max(60, min(95, adjusted))

    # ------------------------------------------------------------------ #
    # Tuned strategy parameters (self-adjusting from backtest)           #
    # ------------------------------------------------------------------ #
    #
    # Per-pair strategy tunables derived from backtest results. The pair
    # bots read these every poll cycle and adjust their entry filter,
    # SL/TP distances, partial-close trigger and BE buffer accordingly.
    #
    # Defaults below match the dashboard's previous global slider values
    # and are used whenever a pair has no backtest insights yet, or its
    # tuned values fall outside safety bounds. This keeps the bot trading
    # on a sensible baseline before the first backtest has run.

    DEFAULT_PARAMS: Dict[str, float] = {
        "atr_tolerance_mult": 1.5,   # Entry filter: range/ATR threshold
        "sl_atr_mult":        2.5,   # SL distance in ATR multiples
        "tp_atr_mult":        4.5,   # TP distance in ATR multiples
        "partial_close_rr":   1.0,   # R:R at which to take partials
        "be_buffer_pips":     1.0,   # Pips buffer when moving SL to BE
        "min_atr_to_tighten": 1.0,   # ATR profit before trail kicks in
        "trail_atr_mult":     1.0,   # Trailing SL distance in ATRs
    }

    # Safety bounds — tuned values outside these clamps fall back to
    # the default. Prevents one bad backtest from making the bot trade
    # with absurd SL/TP that would either never fire or fire instantly.
    PARAM_BOUNDS: Dict[str, tuple] = {
        "atr_tolerance_mult": (0.5, 3.0),
        "sl_atr_mult":        (1.0, 5.0),
        "tp_atr_mult":        (1.5, 10.0),
        "partial_close_rr":   (0.5, 3.0),
        "be_buffer_pips":     (0.0, 10.0),
        "min_atr_to_tighten": (0.5, 3.0),
        "trail_atr_mult":     (0.5, 3.0),
    }

    def get_tuned_params(self, symbol: str) -> Dict[str, float]:
        """Return the live, ML-tuned strategy parameters for a pair.

        Reads `tuned_params` from the pair's insights block if a recent
        backtest produced one. Falls back to DEFAULT_PARAMS for any field
        that's missing or out of bounds. Auto-reloads from disk so a
        fresh backtest's tuning is picked up on the very next call —
        this is what makes the bot self-adjusting.

        The backtester writes a `tuned_params` dict per symbol like:
            {
              "GBPJPY": {
                "importances": {...},      # existing feature weights
                "tuned_params": {
                  "atr_tolerance_mult": 1.6,
                  "sl_atr_mult":        2.3,
                  "tp_atr_mult":        5.1,
                  ...
                }
              }
            }
        """
        self._maybe_reload_from_disk()

        out = dict(self.DEFAULT_PARAMS)
        sym_data = self.insights.get(symbol.upper()) or {}
        tuned    = sym_data.get("tuned_params") or {}

        for key, default in self.DEFAULT_PARAMS.items():
            v = tuned.get(key)
            if v is None:
                continue
            try:
                v = float(v)
            except Exception:
                continue
            lo, hi = self.PARAM_BOUNDS[key]
            # Clamp silently — a value barely out of bounds is more useful
            # than falling all the way back to the default.
            out[key] = max(lo, min(hi, v))

        return out

    def get_all_tuned_params(self) -> Dict[str, Dict[str, float]]:
        """Per-pair tuned params for every pair we know about.

        Used by /api/agent/tuning to render the dashboard's per-pair
        strategy tiles. Always includes the four standard pairs so the
        UI can render four tiles even before any backtest has run.
        """
        self._maybe_reload_from_disk()
        pairs = set(["EURUSD", "GBPUSD", "GBPJPY", "EURJPY"]) | set(self.insights.keys())
        out = {}
        for sym in sorted(pairs):
            sym_data = self.insights.get(sym) or {}
            has_tuning = bool(sym_data.get("tuned_params"))
            out[sym] = {
                "params":      self.get_tuned_params(sym),
                "source":      "backtest" if has_tuning else "default",
                "backtest_at": sym_data.get("backtest_at"),
                "trade_count": sym_data.get("trade_count"),
                "win_rate":    sym_data.get("win_rate"),
            }
        return out

    # ------------------------------------------------------------------ #
    # Diagnostics                                                        #
    # ------------------------------------------------------------------ #

    def get_insights_for_pair(self, symbol: str) -> Dict[str, float]:
        """Return learned weights for a specific pair (for diagnostics)."""
        return self.insights.get(symbol, {})

    def has_insights(self) -> bool:
        """Check if any insights are currently loaded."""
        return bool(self.insights)

    def status(self) -> Dict[str, Any]:
        """Snapshot of the loop's current state — useful for /api endpoints
        or the dashboard so the operator can see what the agent is using."""
        return {
            "insights_file":  self.insights_file,
            "file_exists":    os.path.exists(self.insights_file),
            "last_mtime":     (None if self._last_mtime <= 0
                               else datetime.utcfromtimestamp(self._last_mtime).isoformat()),
            "pairs":          sorted(self.insights.keys()),
            "feature_count":  {sym: len(feats)
                               for sym, feats in self.insights.items()},
        }


# ────────────────────────────────────────────────────────────────────── #
# Module-level singleton                                                  #
# ────────────────────────────────────────────────────────────────────── #
#
# The bot creates exactly one of these. Multiple PairAgent instances and
# the strategy itself all share it. apply_learned_weights() is the hot
# path; it auto-detects new insight files via mtime, so a fresh backtest
# is reflected on the next signal review with zero IPC.

_learning_loop: Optional[AgentLearningLoop] = None


def get_learning_loop(root_path: Optional[str] = None) -> AgentLearningLoop:
    """Get or create the module-level learning-loop singleton."""
    global _learning_loop
    if _learning_loop is None:
        _learning_loop = AgentLearningLoop(root_path)
    return _learning_loop


def reload_learning_loop() -> None:
    """Force-reload the singleton's insights from disk.

    Generally not needed — apply_learned_weights() auto-reloads on file
    mtime change. Provided for explicit reload from REPL / tests / a
    dashboard endpoint that wants to react synchronously to a backtest
    completion.
    """
    global _learning_loop
    if _learning_loop is not None:
        _learning_loop.reload_insights()
    else:
        _learning_loop = AgentLearningLoop()
