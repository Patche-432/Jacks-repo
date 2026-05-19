"""
ai_agent.py - Multi-agent trading system.

ARCHITECTURE (post-refactor — May 2026)
=======================================

  Agent 0 (Orchestrator, LLM-backed)
      One LLM brain. Sees all 4 pair-bot verdicts each cycle plus a
      portfolio snapshot, then decides what actually gets executed. Can
      VETO any pair-bot action when portfolio risk demands it (daily
      loss breach, position cap, drawdown trajectory). Can also OVERRIDE
      a "hold" into a "close" when the portfolio is in trouble. It does
      NOT issue tightens or new SL values — that's the bots' job.

  Agents 1-4 (Pair bots, deterministic code — no LLM)
      Four small Python classes, one per pair (EURUSD / GBPUSD / GBPJPY
      / EURJPY). Each encodes that pair's tactical management rules in
      Python: when to tighten the stop, when to close early, how much
      ATR room to give. They produce the SAME verdict shape the LLM
      pair agents used to produce, so callers don't change. Dropping
      the LLM here is a big computing-power win — managing 4 positions
      no longer costs 4 Ollama calls per poll cycle.

WHY THIS SHAPE
==============

The previous design had the LLM doing both signal review AND per-pair
position management. Two problems:

  * One LLM trying to be a specialist on 4 pairs simultaneously
    regresses to generic advice. Encoding each pair's known behaviour
    deterministically preserves intent.
  * Most position-management decisions are mechanical (tighten at
    1R, close on structure break). The LLM was being asked to do
    arithmetic; it was slow at it and sometimes hallucinated numbers.

The LLM is now used only where its reasoning genuinely adds value:
weighing 4 simultaneous bot verdicts against portfolio state.

HARD RAILS (CODE-ENFORCED, REGARDLESS OF LLM OPINION)
=====================================================

  1. Pair bots can only TIGHTEN a stop or request an EARLY close.
     They cannot loosen a stop, widen a TP, or override the broker's
     hard SL.
  2. Agent 0 can VETO (suppress) any verdict and can OVERRIDE a hold
     to a close. It cannot issue a TIGHTEN by itself, propose new SL
     values, or override the same hard rails the bots respect.
  3. Strategy POC bias gate is upstream of all this. Misaligned
     signals never reach any agent.

PUBLIC SURFACE (used by ai_pro.py)
==================================

  • get_pair_agent(symbol).manage_position(ctx) -> verdict dict
        Routes to the right deterministic bot. Same signature
        as before; ai_pro.py callers don't change.
  • get_orchestrator().orchestrate(verdicts, portfolio) -> dict
        New: feeds bot verdicts + portfolio state to the LLM and
        returns the executed plan (approvals, vetoes, overrides).
  • ollama_health() -> dict
        Unchanged; still used by the dashboard.
  • agent_backend_enabled() -> bool
        Unchanged; AI_BACKEND=off disables auto-trading.

CONFIG
======
  OLLAMA_URL      (default http://localhost:11434)
  OLLAMA_MODEL    (default qwen2.5:14b-instruct)
  AGENT_TIMEOUT_S (default 90; 14b on CPU needs 40-60s, this gives headroom)
  AI_BACKEND      ("off" / "none" / "disabled" => kill switch)
"""

from __future__ import annotations

import json as _json
import logging
import os
import re
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional

# Memory bank — imported lazily so ai_agent works even if agent_memory.py
# is missing (e.g. very first checkout before the file exists).
try:
    from agent_memory import get_memory as _get_agent_memory
    HAS_AGENT_MEMORY = True
except ImportError:
    HAS_AGENT_MEMORY = False
    _get_agent_memory = None  # type: ignore[assignment]

log = logging.getLogger("ai_agent")


# ========================================================================== #
# HTTP helpers                                                                #
# ========================================================================== #

def _http_post_json(url: str, payload: dict, timeout: float) -> dict:
    body = _json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    try:
        return _json.loads(raw.decode("utf-8", errors="replace"))
    except Exception as exc:
        raise RuntimeError(f"non-JSON response from {url}: {exc}") from exc


def _http_get_json(url: str, timeout: float) -> dict:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    try:
        return _json.loads(raw.decode("utf-8", errors="replace"))
    except Exception as exc:
        raise RuntimeError(f"non-JSON response from {url}: {exc}") from exc


# ========================================================================== #
# Ollama HTTP client (used only by the orchestrator now)                     #
# ========================================================================== #

class OllamaClient:
    """Thin wrapper around Ollama's /api/chat endpoint with JSON mode."""

    def __init__(self,
                 url: Optional[str] = None,
                 model: Optional[str] = None,
                 timeout: Optional[float] = None) -> None:
        self.url = (url or os.getenv("OLLAMA_URL", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:14b-instruct")
        self.timeout = float(timeout or os.getenv("AGENT_TIMEOUT_S", "90"))

    def chat(self,
             system: str,
             user: str,
             *,
             temperature: float = 0.0,
             expect_json: bool = True,
             max_tokens: int = 512) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_tokens),
            },
        }
        if expect_json:
            payload["format"] = "json"
        data = _http_post_json(
            f"{self.url}/api/chat",
            payload=payload,
            timeout=self.timeout,
        )
        return (data.get("message") or {}).get("content", "") or ""


# ========================================================================== #
# Back-compat shims (legacy callers still import these from ai_agent)        #
# ========================================================================== #

@dataclass
class MarketContext:
    """Back-compat shim. Bias decisions live in the strategy POC gate now."""
    daily_open: float = 0.0
    current_price: float = 0.0
    detail: str = "bias gate moved to strategy POC filter"
    d1: str = "n/a"
    h4: str = "n/a"
    h1: str = "n/a"
    score: int = 0
    reason: str = ""


HTFBias = MarketContext  # legacy alias


def fetch_market_context(symbol: str) -> MarketContext:
    """Back-compat shim — agent layer no longer consumes this."""
    ctx = MarketContext()
    try:
        import MetaTrader5 as mt5
    except Exception:
        ctx.detail = "mt5 import failed"
        return ctx
    try:
        d1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, 1)
    except Exception:
        d1 = None
    if d1 is None or len(d1) < 1:
        ctx.detail = "no D1 data"
        return ctx
    ctx.daily_open = float(d1[-1]["open"])
    try:
        tk = mt5.symbol_info_tick(symbol)
        ctx.current_price = float(tk.bid) if tk is not None else float(d1[-1]["close"])
    except Exception:
        ctx.current_price = float(d1[-1]["close"])
    ctx.detail = f"strategy-POC-gated (legacy DO={ctx.daily_open:.5f} px={ctx.current_price:.5f})"
    return ctx


compute_htf_bias = fetch_market_context  # legacy alias


# ========================================================================== #
# JSON extraction helpers                                                     #
# ========================================================================== #

_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _extract_json_dict(raw: str) -> Optional[dict]:
    if not raw:
        return None
    raw = raw.strip()
    try:
        parsed = _json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    for m in _JSON_RE.finditer(raw):
        try:
            parsed = _json.loads(m.group(0))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return None


def _coerce_bool(v, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "y", "1", "approve", "approved")
    return default


def _coerce_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


# ========================================================================== #
# PositionContext - the shape pair bots read for a live trade                #
# ========================================================================== #

@dataclass
class PositionContext:
    symbol: str
    ticket: int
    side: str                 # BUY or SELL
    entry: float
    cur_price: float
    cur_sl: float
    cur_tp: float
    profit_pts: float
    peak_pts: float
    atr: float
    digits: int
    trend_intact: Optional[bool] = None
    structure_broken: Optional[bool] = None
    fresh_structure: Optional[bool] = None
    notes: str = ""


# ========================================================================== #
# Pair bots (Agents 1-4) - deterministic, no LLM                             #
# ========================================================================== #
#
# The base class implements the common machinery:
#   * convert raw price moves to ATR multiples
#   * enforce hard rails (only tighten, never loosen; correct side of price)
#   * defer pair-specific *thresholds* and *signals* to subclasses
#
# Each subclass overrides a small set of constants. This makes it easy
# to tune one pair without touching another, and easy to read what
# differs at a glance.
#
# Verdict shape (kept identical to the old LLM PairAgent so ai_pro.py
# callers don't change):
#     {"action": "hold" | "close" | "move_sl",
#      "new_sl": float | None,
#      "reason": str}


class PairBotBase:
    """
    Deterministic position manager for one pair.

    Per-pair tunables (override in subclasses):

      MIN_ATR_PROFIT_TO_TIGHTEN  How far in profit (in ATRs) before we'll
                                 consider tightening. Cable & GBPJPY want
                                 a wider buffer — their spread noise is
                                 large enough to chop a too-tight stop.
      TRAIL_ATR_MULT             How far the new SL sits from the current
                                 price when we tighten (ATR multiple).
                                 Volatile pairs use bigger multiples so
                                 normal retracements don't knock us out.
      CLOSE_ON_TREND_BROKEN      Close immediately when trend_intact is
                                 False, even before structure_broken
                                 fires. Useful for spike-prone pairs.
      CLOSE_ON_STRUCTURE_BROKEN  Close as soon as fresh structure breaks
                                 against the trade. Default True for all
                                 pairs.
    """

    SYMBOL: str = ""

    # --- per-pair tunables (subclasses override) ---
    MIN_ATR_PROFIT_TO_TIGHTEN: float = 1.0
    TRAIL_ATR_MULT:            float = 1.0
    CLOSE_ON_TREND_BROKEN:     bool  = False
    CLOSE_ON_STRUCTURE_BROKEN: bool  = True

    # --- methods ---

    def manage_position(
        self,
        ctx: PositionContext,
        tuned_params: Optional[dict] = None,
    ) -> dict:
        """Decide what to do with one open position. Verdict only;
        execution is the orchestrator + ai_pro's job.

        tuned_params: per-pair dict from AgentLearningLoop.get_tuned_params().
        When supplied, overrides the class-level MIN_ATR_PROFIT_TO_TIGHTEN
        and TRAIL_ATR_MULT and activates break-even logic using the
        backtest-derived be_buffer_pips / partial_close_rr values.
        B/E is intentionally off when tuned_params is None (tests + fallback).
        """
        tp = tuned_params or {}
        min_atr    = float(tp.get("min_atr_to_tighten") or self.MIN_ATR_PROFIT_TO_TIGHTEN)
        trail_mult = float(tp.get("trail_atr_mult")     or self.TRAIL_ATR_MULT)
        # B/E only activates when the caller supplies backtest-tuned params.
        be_trigger: Optional[float] = float(tp["partial_close_rr"]) if tuned_params and "partial_close_rr" in tp else None
        be_buffer:  Optional[float] = float(tp["be_buffer_pips"])   if tuned_params and "be_buffer_pips"   in tp else None

        if ctx.atr > 0:
            price_unit_profit = ctx.profit_pts * (10.0 ** (-ctx.digits))
            atr_profit = price_unit_profit / ctx.atr
        else:
            atr_profit = 0.0

        # ----- Close-early checks (highest priority) -------------------
        if self.CLOSE_ON_STRUCTURE_BROKEN and ctx.structure_broken is True:
            return {
                "action": "close",
                "atr_profit": round(atr_profit, 2),
                "reason": f"{self.SYMBOL} bot: structure broken — close",
            }

        if self.CLOSE_ON_TREND_BROKEN and ctx.trend_intact is False:
            return {
                "action": "close",
                "atr_profit": round(atr_profit, 2),
                "reason": f"{self.SYMBOL} bot: trend lost (spike-prone pair) — close",
            }

        # ----- Collect candidate SL tightens (both trail and B/E) -----
        # Run both checks; the one that produces the tightest SL wins.
        # This means once trail is deep enough to beat B/E, trail takes
        # over automatically — no special-casing needed.
        candidates: list[tuple[float, str]] = []

        if atr_profit >= min_atr:
            trail_sl = self._propose_trailing_sl(ctx, trail_mult)
            if trail_sl is not None and self._sl_is_tighter(ctx, trail_sl):
                candidates.append((trail_sl, f"trail SL {trail_mult:.1f}× ATR"))

        if be_trigger is not None and be_buffer is not None and atr_profit >= be_trigger:
            be_sl = self._propose_be_sl(ctx, be_buffer)
            if be_sl is not None and self._sl_is_tighter(ctx, be_sl):
                candidates.append((be_sl, f"B/E +{be_buffer:.1f}p"))

        if candidates:
            is_buy = ctx.side.upper() == "BUY"
            best_sl, best_reason = max(
                candidates,
                key=lambda x: x[0] if is_buy else -x[0],
            )
            return {
                "action": "move_sl",
                "new_sl": best_sl,
                "atr_profit": round(atr_profit, 2),
                "reason": f"{self.SYMBOL} bot: profit {atr_profit:.2f} ATR — {best_reason}",
            }

        # ----- Default: hold ------------------------------------------
        return {
            "action": "hold",
            "atr_profit": round(atr_profit, 2),
            "reason": f"{self.SYMBOL} bot: hold (profit {atr_profit:.2f} ATR)",
        }

    # -- helpers -------------------------------------------------------

    def _propose_trailing_sl(
        self,
        ctx: PositionContext,
        trail_atr_mult: Optional[float] = None,
    ) -> Optional[float]:
        """Default trail: cur_price ± (trail_atr_mult * ATR)."""
        if ctx.atr <= 0:
            return None
        mult   = trail_atr_mult if trail_atr_mult is not None else self.TRAIL_ATR_MULT
        offset = ctx.atr * mult
        if ctx.side.upper() == "BUY":
            new_sl = ctx.cur_price - offset
        else:
            new_sl = ctx.cur_price + offset
        return round(new_sl, ctx.digits)

    def _propose_be_sl(self, ctx: PositionContext, be_buffer_pips: float) -> Optional[float]:
        """Break-even SL: entry ± (be_buffer_pips × pip_size).
        Puts the stop just past entry so the trade becomes risk-free."""
        pip_size = 0.01 if "JPY" in ctx.symbol else 0.0001
        offset = be_buffer_pips * pip_size
        if ctx.side.upper() == "BUY":
            return round(ctx.entry + offset, ctx.digits)
        return round(ctx.entry - offset, ctx.digits)

    @staticmethod
    def _sl_is_tighter(ctx: PositionContext, new_sl: float) -> bool:
        """Hard rail: SL must be on the right side of price AND tighter
        than (or equal to, but not looser than) the current SL."""
        is_buy = ctx.side.upper() == "BUY"
        # Wrong side of price
        if is_buy and new_sl >= ctx.cur_price:
            return False
        if (not is_buy) and new_sl <= ctx.cur_price:
            return False
        # Looser than current SL
        if ctx.cur_sl:
            if is_buy and new_sl <= ctx.cur_sl:
                return False
            if (not is_buy) and new_sl >= ctx.cur_sl:
                return False
        return True


class EURUSDBot(PairBotBase):
    """Tight, liquid pair. Respects structure well in London/NY overlap.
    Cuts fast on structure breaks. Trails aggressively after 1R."""
    SYMBOL = "EURUSD"
    MIN_ATR_PROFIT_TO_TIGHTEN = 1.0
    TRAIL_ATR_MULT            = 1.0


class GBPUSDBot(PairBotBase):
    """Cable: volatile, prone to stop hunts. Wider tighten buffer; sharp
    structure breaks trigger immediate close."""
    SYMBOL = "GBPUSD"
    MIN_ATR_PROFIT_TO_TIGHTEN = 1.5  # don't tighten until well in profit
    TRAIL_ATR_MULT            = 1.25


class GBPJPYBot(PairBotBase):
    """Highest-volatility pair. Wide ATR; normal retracements look like
    reversals. Hold through noise; trail by 1.5 ATR minimum to avoid wicks.
    Reward runners — this pair trends strongly once a level breaks."""
    SYMBOL = "GBPJPY"
    MIN_ATR_PROFIT_TO_TIGHTEN = 1.5
    TRAIL_ATR_MULT            = 1.5


class EURJPYBot(PairBotBase):
    """Risk-sentiment driven; can gap and spike. Treat trend_intact=False
    as a hard close signal (don't wait for structure_broken)."""
    SYMBOL = "EURJPY"
    MIN_ATR_PROFIT_TO_TIGHTEN = 1.0
    TRAIL_ATR_MULT            = 1.25
    CLOSE_ON_TREND_BROKEN     = True  # the distinguishing rule


# Module-level registry — one bot per symbol, created on first use.
_PAIR_BOT_REGISTRY: dict[str, PairBotBase] = {}
_PAIR_BOT_CLASSES: dict[str, type[PairBotBase]] = {
    "EURUSD": EURUSDBot,
    "GBPUSD": GBPUSDBot,
    "GBPJPY": GBPJPYBot,
    "EURJPY": EURJPYBot,
}


def get_pair_agent(symbol: str) -> PairBotBase:
    """Return (creating if needed) the deterministic bot for this symbol.

    Public name kept as `get_pair_agent` for back-compat with ai_pro.py
    callers — they expect a `manage_position(ctx)` method, which the
    new bots provide with the same verdict shape.

    Falls back to PairBotBase for unknown symbols so the bot never
    crashes when a non-core pair is traded; the operator gets generic
    management.
    """
    sym = symbol.upper()
    if sym not in _PAIR_BOT_REGISTRY:
        cls = _PAIR_BOT_CLASSES.get(sym, PairBotBase)
        bot = cls()
        if cls is PairBotBase:
            # Generic fallback bot — assign the symbol so logs are clear.
            bot.SYMBOL = sym
        _PAIR_BOT_REGISTRY[sym] = bot
    return _PAIR_BOT_REGISTRY[sym]


# ========================================================================== #
# Agent 0 - Orchestrator (LLM-backed)                                        #
# ========================================================================== #
#
# Sees all 4 pair-bot verdicts each cycle plus a portfolio snapshot, then
# decides what actually executes. Authority: VETO + OVERRIDE.
#
#   * VETO    : suppress any verdict the bots produced
#   * OVERRIDE: turn a "hold" into a "close" when portfolio is in trouble
#
# The orchestrator does NOT issue tightens or propose new SL values —
# that's the bots' job. It can only choose between:
#     - approve as-is
#     - veto (drop)
#     - override (only "hold" -> "close")

ORCHESTRATOR_SYSTEM_PROMPT = (
    "You are AGENT ZERO, supervisor for a 4-pair FX bot trading CHoCH + daily-levels.\n"
    "\n"
    "TWO JOBS: hold winners longer. Cut losers early.\n"
    "\n"
    "Each trade shows:\n"
    "  cur_R      — current profit as R-multiple (negative = in loss)\n"
    "  peak_R     — best profit this trade has reached\n"
    "  gave-back  — how much of peak has been given back (peak_R - cur_R)\n"
    "  to-TP      — R remaining to the original take-profit\n"
    "  PA         — structure/trend state from price action detectors\n"
    "  bot        — what the deterministic pair bot decided\n"
    "\n"
    "VETO the bot (hold the trade) when:\n"
    "  • to-TP ≥ 1R AND structure intact — bot is exiting with significant room left\n"
    "  • gave-back < 0.5R AND structure intact — normal retracement noise, not a reversal\n"
    "\n"
    "OVERRIDE to close (cut the loser) when:\n"
    "  • Structure AND trend are both broken AND gave-back ≥ 1R from peak\n"
    "  • Trade is in loss (cur_R < 0) AND structure is broken — no reason to hold\n"
    "  • Price has reversed through entry AND structure broken — entry reason is gone\n"
    "\n"
    "APPROVE in all other cases — the bots are correct more often than not.\n"
    "Only act when the R-to-go or peak-drawdown numbers make the right call obvious.\n"
    "\n"
    "Use entry context shown per trade:\n"
    "  signal=CHoCH: highest-quality setup — hold through noise, raise bar for override_close.\n"
    "  signal=Continuation: already partially played out — lower patience if structure breaks.\n"
    "  conf≥75: high-conviction entry — give more room.\n"
    "  conf<65: marginal entry — cut sooner on same signals.\n"
    "  risk_free=yes (SL past break-even): almost never override_close — downside is zero.\n"
    "  open_cycles: trade held long at low R = setup failing to play out — consider override_close.\n"
    "\n"
    "First reason briefly using what you know about FX behaviour, then decide. Reply JSON only:\n"
    "  {\"reasoning\": \"<20w on what the price action and session suggest>\","
    " \"decisions\": [{\"ticket\": <int>, \"verdict\": \"approve|veto|override_close\", \"reason\": \"<15w\"}, ...]}\n"
)


@dataclass
class PortfolioSnapshot:
    """Lightweight portfolio state passed to the orchestrator each cycle.

    Built by ai_pro.py from MT5 + the rules engine; the orchestrator
    only needs the high-signal numbers to make a portfolio call.
    """
    equity:               float = 0.0
    balance:              float = 0.0
    daily_pl:             float = 0.0          # realised + open today
    daily_loss_limit:     float = 0.0          # 0 if disabled
    daily_loss_breach:    bool  = False        # already breached?
    open_positions_count: int   = 0
    max_positions_total:  int   = 0            # 0 if disabled
    notes:                str   = ""

    def summary(self) -> str:
        """Compact one-line summary the LLM can absorb cheaply."""
        parts = [
            f"equity={self.equity:.2f}",
            f"daily_pl={self.daily_pl:+.2f}",
        ]
        if self.daily_loss_limit > 0:
            parts.append(
                f"daily_limit=-{self.daily_loss_limit:.2f}"
                f"{' BREACHED' if self.daily_loss_breach else ''}"
            )
        parts.append(
            f"open={self.open_positions_count}"
            f"{('/' + str(self.max_positions_total)) if self.max_positions_total else ''}"
        )
        if self.notes:
            parts.append(self.notes)
        return " ".join(parts)


class Orchestrator:
    """Agent 0 — the LLM-backed portfolio orchestrator."""

    name = "Agent 0"
    _ML_CONTEXT_CACHE: dict = {"path": None, "mtime": -1.0, "text": ""}

    _SESSION_WINDOW: int = 8   # cycles to track for session character
    _ATR_WINDOW:     int = 20  # cycles to track for volatility regime

    def __init__(self, client=None):
        self.client = client or OllamaClient()
        self._wm:              dict = {}  # working memory: ticket -> state dict
        self._session_tracker: list = []  # recent cycle aggregate ΔR values
        self._atr_history:     dict = {}  # symbol -> list of recent ATR values

    # ------------------------------------------------------------------ #
    # Helpers                                                               #
    # ------------------------------------------------------------------ #

    # Maps top feature name \u2192 plain-English instruction the LLM can act on.
    _FEATURE_HINTS: dict = {
        "hour_of_day":      "session timing drives outcomes",
        "confidence":       "signal confidence predicts wins \u2014 high-confidence entries deserve more patience",
        "rr_ratio":         "original RR quality matters \u2014 only high-RR setups run to target",
        "dist_to_poc_atr":  "distance to daily level matters \u2014 trades near POC are more reliable",
        "dist_to_poc_pips": "distance to daily level matters \u2014 trades near POC are more reliable",
        "dow":              "day of week matters \u2014 mid-week setups tend to outperform Mon/Fri",
        "is_choch":         "CHoCH quality matters \u2014 confirmed CHoCH entries outperform continuations",
        "is_continuation":  "continuation vs reversal type matters",
    }

    @classmethod
    def _load_ml_context(cls) -> str:
        """Return ML pair tendencies as actionable instructions, cached by file mtime.

        Placed in the system prompt (not user message) so the LLM treats it as
        standing behavioural guidance rather than per-cycle data to parse.
        """
        try:
            import os as _os, json as _j
            path = _os.path.join(
                _os.path.dirname(_os.path.abspath(__file__)),
                "backtest_insights.json",
            )
            if not _os.path.exists(path):
                return ""
            mtime = _os.path.getmtime(path)
            cache = cls._ML_CONTEXT_CACHE
            if cache["path"] == path and cache["mtime"] == mtime:
                return cache["text"]
            with open(path, "r", encoding="utf-8") as fh:
                ins = _j.load(fh)

            lines: list = []
            session_sensitive: list = []  # pairs where hour_of_day is top predictor

            for sym, d in sorted(ins.items()):
                wr    = d.get("win_rate") or 0.0
                n     = d.get("trade_count", 0)
                imps  = d.get("importances") or {}
                top   = max(imps, key=imps.get) if imps else None
                tp    = d.get("tuned_params") or {}
                sl_mult = tp.get("sl_atr_mult", 0.0)

                if wr >= 0.75:
                    wr_label  = "high WR"
                    wr_action = "be patient \u2014 raise the bar for override_close"
                elif wr >= 0.55:
                    wr_label  = "moderate WR"
                    wr_action = "follow bot signals \u2014 intervene only on clear structure failure"
                else:
                    wr_label  = "low WR"
                    wr_action = "lower conviction \u2014 override_close sooner on any structure weakness"

                hint = cls._FEATURE_HINTS.get(top, f"{top} is most predictive") if top else ""

                sl_note = ""
                if sl_mult >= 4.0:
                    sl_note = (f" Wide SL ({sl_mult:.1f}\u00d7ATR) means normal retracements"
                               f" look large \u2014 do not override_close on noise alone.")

                line = f"  {sym} WR={wr*100:.0f}% (n={n}, {wr_label}): {wr_action}."
                if hint:
                    line += f" Key predictor: {hint}."
                if sl_note:
                    line += sl_note
                lines.append(line)

                if top == "hour_of_day":
                    session_sensitive.append(sym)

            if not lines:
                text = ""
            else:
                text = (
                    "PAIR TENDENCIES (backtest-derived \u2014 calibrate patience per pair):\n"
                    + "\n".join(lines)
                )
                if session_sensitive:
                    text += (
                        f"\n  SESSION RULE: {', '.join(session_sensitive)} outcome depends strongly"
                        f" on time of day. In Asian/Overnight session these pairs produce weaker"
                        f" setups \u2014 lower your bar for override_close."
                        f" Cross-reference with the SESSION label you see in TRADES."
                    )
                text += (
                    "\nHigh-WR pairs deserve more patience before you override;"
                    " low-WR pairs get override_close earlier on the same signals."
                )

            cache.update({"path": path, "mtime": mtime, "text": text})
            return text
        except Exception as exc:
            log.debug("ML insights load failed: %s", exc)
            return ""

    def _build_system_prompt(self) -> str:
        """Build system prompt: base rules → ML pair tendencies → agent memory."""
        prompt = ORCHESTRATOR_SYSTEM_PROMPT
        ml_ctx = self._load_ml_context()
        if ml_ctx:
            prompt = prompt + "\n\n" + ml_ctx
        if HAS_AGENT_MEMORY and _get_agent_memory is not None:
            try:
                mem_ctx = _get_agent_memory().build_context()
                if mem_ctx:
                    prompt = prompt + "\n" + mem_ctx
            except Exception as exc:
                log.debug("agent_memory build_context failed: %s", exc)
        return prompt

    @staticmethod
    def _session_label(utc_hour: int) -> str:
        """Map UTC hour to a readable FX session name."""
        if 22 <= utc_hour or utc_hour < 7:
            return "Asian/Overnight"
        if 7 <= utc_hour < 10:
            return "London-Open"
        if 10 <= utc_hour < 12:
            return "London-Mid"
        if 12 <= utc_hour < 17:
            return "NY/London-Overlap"
        return "NY-Afternoon"

    @staticmethod
    def _build_user_message(verdicts, portfolio, market_ctx: str = "") -> str:
        """Format bot verdicts + trader-relevant metrics as the LLM user message."""
        import datetime as _dt
        now     = _dt.datetime.now(_dt.timezone.utc)
        session = Orchestrator._session_label(now.hour)

        lines = []
        for v in verdicts:
            atr       = v.get("atr") or 0.0
            entry     = v.get("entry")
            cur_price = v.get("cur_price")
            cur_tp    = v.get("cur_tp")
            peak_pts  = v.get("peak_pts")
            point     = v.get("point") or 0.0
            direction = (v.get("direction") or "?").upper()
            notes     = v.get("notes") or ""

            # R values \u2014 all computed in price units so units are consistent.
            # atr is price-unit ATR; profit_pts is broker points, so we use
            # entry/cur_price directly rather than profit_pts/atr (wrong units).
            cur_r: Optional[float] = None
            peak_r: Optional[float] = None
            r_to_go: Optional[float] = None

            if atr > 0 and entry is not None and cur_price is not None:
                price_profit = (cur_price - entry) if direction == "BUY" else (entry - cur_price)
                cur_r = price_profit / atr

            if atr > 0 and peak_pts is not None and point > 0:
                peak_r = (peak_pts * point) / atr

            if atr > 0 and cur_tp is not None and cur_price is not None:
                r_to_go = abs(cur_tp - cur_price) / atr

            gave_back = (peak_r - cur_r) if (peak_r is not None and cur_r is not None) else None

            cur_r_s  = f"{cur_r:+.1f}R"        if cur_r   is not None else "?R"
            peak_s   = f" peak={peak_r:+.1f}R"  if peak_r  is not None else ""
            gave_s   = (f" gave-back={gave_back:.1f}R"
                        if gave_back is not None and gave_back > 0.1 else "")
            tp_s     = f" to-TP={r_to_go:.1f}R" if r_to_go is not None else " to-TP=?"

            entry_s = f"{entry:.5f}"     if entry     is not None else "?"
            price_s = f"{cur_price:.5f}" if cur_price is not None else "?"

            struct_broken = v.get("structure_broken")
            trend_intact  = v.get("trend_intact")
            struct_s = "STRUCTURE_BROKEN" if struct_broken else "struct_ok"
            trend_s  = "TREND_BROKEN" if trend_intact is False else "trend_ok"

            short_notes = (notes
                           .replace("trend_reason=", "trend=")
                           .replace("struct_reason=", "struct="))[:100].strip()

            new_sl = v.get("new_sl")
            sl_s = f"{new_sl:.5f}" if isinstance(new_sl, (int, float)) else "\u2014"

            # Risk-free flag \u2014 SL at or beyond break-even
            cur_sl_val = v.get("cur_sl") or 0.0
            is_risk_free = False
            if cur_sl_val and entry is not None:
                is_risk_free = (cur_sl_val >= entry if direction == "BUY"
                                else cur_sl_val <= entry)

            # Entry context
            sig_type     = v.get("signal_type") or ""
            confidence   = v.get("confidence")
            dist_poc     = v.get("dist_to_poc_atr")
            wm_cycles    = v.get("wm_cycles_held", 0)

            entry_parts: list = []
            if sig_type:
                entry_parts.append(f"signal={sig_type}")
            if confidence is not None:
                entry_parts.append(f"conf={confidence}%")
            if dist_poc is not None:
                entry_parts.append(f"dist_poc={dist_poc:.1f}R")
            if is_risk_free:
                entry_parts.append("RISK-FREE")
            if wm_cycles > 0:
                entry_parts.append(f"open {wm_cycles} cycles")
            entry_line = ("\n    entry: " + " | ".join(entry_parts)) if entry_parts else ""

            # Working memory \u2014 only shown when there is something meaningful to say
            wm_delta    = v.get("wm_delta_r")
            wm_sb_str   = v.get("wm_struct_streak", 0)
            wm_ti_str   = v.get("wm_trend_streak", 0)
            wm_last_dec = v.get("wm_last_decision", "")
            wm_last_rsn = (v.get("wm_last_reason") or "")[:50]

            wm_parts: list = []
            if wm_delta is not None:
                word = "falling" if wm_delta < -0.1 else "rising" if wm_delta > 0.1 else "flat"
                wm_parts.append(f"\u0394R={wm_delta:+.2f} ({word})")
            if wm_sb_str >= 2:
                wm_parts.append(f"struct_broken {wm_sb_str} cycles")
            if wm_ti_str >= 2:
                wm_parts.append(f"trend_broken {wm_ti_str} cycles")
            if wm_last_dec:
                wm_parts.append(
                    f"prev={wm_last_dec}" + (f': "{wm_last_rsn}"' if wm_last_rsn else "")
                )
            wm_line = ("\n    wm: " + " | ".join(wm_parts)) if wm_parts else ""

            lines.append(
                f"  #{v['ticket']} {v['symbol']} {direction} "
                f"entry={entry_s} px={price_s} | "
                f"{cur_r_s}{peak_s}{gave_s}{tp_s}\n"
                f"    PA: {struct_s} | {trend_s}"
                + (f" | {short_notes}" if short_notes else "")
                + entry_line
                + wm_line + "\n"
                f"    bot: {v['action']} sl={sl_s} \u2014 {(v.get('reason') or '')[:80]}"
            )

        portfolio_line = portfolio.summary() if portfolio is not None else ""
        msg = (
            f"SESSION: {session} (UTC {now.hour:02d}h)"
            + (f" | {portfolio_line}" if portfolio_line else "") + "\n"
            + (f"MARKET: {market_ctx}\n" if market_ctx else "")
            + f"\nTRADES ({len(verdicts)})\n"
            + "\n".join(lines)
        )
        return msg

    # ------------------------------------------------------------------ #
    # Working memory                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_cur_r(v: dict) -> Optional[float]:
        """Current R in price units from a verdict dict. None when inputs missing."""
        atr       = v.get("atr") or 0.0
        entry     = v.get("entry")
        cur_price = v.get("cur_price")
        direction = (v.get("direction") or "").upper()
        if atr > 0 and entry is not None and cur_price is not None and direction in ("BUY", "SELL"):
            price_profit = (cur_price - entry) if direction == "BUY" else (entry - cur_price)
            return price_profit / atr
        return None

    def _enrich_with_working_memory(self, verdicts: list) -> list:
        """Return copies of verdicts with per-ticket working memory deltas attached."""
        enriched = []
        for v in verdicts:
            ticket = v.get("ticket")
            mem    = self._wm.get(ticket, {})
            cur_r  = self._compute_cur_r(v)

            sb = v.get("structure_broken")
            ti = v.get("trend_intact")
            sb_streak = (mem.get("sb_streak", 0) + 1) if sb         else 0
            ti_streak = (mem.get("ti_streak", 0) + 1) if ti is False else 0

            prev_r  = mem.get("prev_r")
            delta_r = (cur_r - prev_r) if (cur_r is not None and prev_r is not None) else None

            enriched.append({
                **v,
                "wm_prev_r":        prev_r,
                "wm_delta_r":       delta_r,
                "wm_struct_streak": sb_streak,
                "wm_trend_streak":  ti_streak,
                "wm_last_decision": mem.get("last_decision", ""),
                "wm_last_reason":   mem.get("last_reason", ""),
                "wm_cycles_held":   mem.get("cycles_held", 0),
            })
        return enriched

    def _wm_needs_review(self, enriched: list) -> bool:
        """True when working memory flags a trade that warrants LLM review even if all bots hold.

        Catches two cases the fast-path would otherwise miss:
          • Trade falling fast (>0.5R drop in one cycle) — may be a loser to cut
          • Persistent structure/trend break (3+ cycles) — not noise, a real reversal
        """
        for v in enriched:
            delta_r = v.get("wm_delta_r")
            if delta_r is not None and delta_r < -0.5:
                return True
            if v.get("wm_struct_streak", 0) >= 3:
                return True
            if v.get("wm_trend_streak", 0) >= 3:
                return True
        return False

    def _update_working_memory(self, enriched: list, llm_out: list) -> None:
        """Persist this cycle's state and orchestrator decisions into working memory."""
        decided = {v["ticket"]: v for v in llm_out}
        active  = {v["ticket"] for v in enriched}

        for v in enriched:
            ticket = v["ticket"]
            cur_r  = self._compute_cur_r(v)
            dec    = decided.get(ticket, {})
            rc     = dec.get("reason_code", "")
            if rc == "ORC_VETO":
                last_dec = "veto"
            elif rc == "ORC_OVERRIDE":
                last_dec = "override_close"
            else:
                last_dec = "approve"
            last_reason = (dec.get("orchestrator_reason") or "")[:80]

            self._wm[ticket] = {
                "prev_r":        cur_r,
                "sb_streak":     v.get("wm_struct_streak", 0),
                "ti_streak":     v.get("wm_trend_streak", 0),
                "last_decision": last_dec,
                "last_reason":   last_reason,
                "cycles_held":   v.get("wm_cycles_held", 0) + 1,
            }

        # Prune tickets that are no longer open
        for t in [t for t in self._wm if t not in active]:
            del self._wm[t]

    # ------------------------------------------------------------------ #
    # Session character + volatility regime                                #
    # ------------------------------------------------------------------ #

    def _update_session_tracker(self, enriched: list) -> None:
        """Append aggregate portfolio ΔR for this cycle to the session window."""
        deltas = [v["wm_delta_r"] for v in enriched if v.get("wm_delta_r") is not None]
        if deltas:
            self._session_tracker.append(sum(deltas))
            if len(self._session_tracker) > self._SESSION_WINDOW:
                self._session_tracker.pop(0)

    def _session_character(self) -> str:
        """Return a one-line session sentiment string, or '' if not enough data."""
        if len(self._session_tracker) < 3:
            return ""
        n   = len(self._session_tracker)
        neg = sum(1 for d in self._session_tracker if d < -0.1)
        pos = sum(1 for d in self._session_tracker if d >  0.1)
        if neg >= n * 0.6:
            return f"Session: {neg}/{n} cycles adverse — choppy/deteriorating conditions"
        if pos >= n * 0.6:
            return f"Session: {pos}/{n} cycles improving — trending conditions"
        return f"Session: mixed ({pos} improving, {neg} deteriorating of {n} cycles)"

    def _update_atr_history(self, enriched: list) -> None:
        """Maintain a rolling ATR history per symbol."""
        for v in enriched:
            sym = v.get("symbol", "")
            atr = v.get("atr") or 0.0
            if sym and atr > 0:
                hist = self._atr_history.setdefault(sym, [])
                hist.append(atr)
                if len(hist) > self._ATR_WINDOW:
                    hist.pop(0)

    def _atr_regime(self, enriched: list) -> str:
        """Return volatility regime string for pairs that deviate from their baseline."""
        parts = []
        seen  = set()
        for v in enriched:
            sym  = v.get("symbol", "")
            atr  = v.get("atr") or 0.0
            hist = self._atr_history.get(sym, [])
            if sym in seen or not atr or len(hist) < 5:
                continue
            seen.add(sym)
            baseline = sum(hist[:-1]) / len(hist[:-1])
            ratio    = atr / baseline if baseline > 0 else 1.0
            if ratio >= 1.3:
                parts.append(f"{sym} {ratio:.1f}×ATR (high vol)")
            elif ratio <= 0.75:
                parts.append(f"{sym} {ratio:.1f}×ATR (low vol)")
        return ("Volatility: " + ", ".join(parts)) if parts else ""

    def _build_market_context(self, enriched: list) -> str:
        """Compile session character, ATR regime, and today's results into one string."""
        parts = []
        sc = self._session_character()
        if sc:
            parts.append(sc)
        ar = self._atr_regime(enriched)
        if ar:
            parts.append(ar)
        if HAS_AGENT_MEMORY and _get_agent_memory is not None:
            try:
                ts = _get_agent_memory().get_today_summary()
                if ts:
                    parts.append(ts)
            except Exception:
                pass
        return " | ".join(parts)

    @staticmethod
    def _apply_decisions(verdicts, decisions) -> list:
        """Apply orchestrator decisions to bot verdicts; return executable plan."""
        by_ticket = {}
        for d in decisions:
            try:
                by_ticket[int(d.get("ticket"))] = d
            except Exception:
                continue

        out = []
        for v in verdicts:
            d = by_ticket.get(v["ticket"], {})
            verdict_str = str(d.get("verdict", "approve")).lower().strip()
            reason = str(d.get("reason", "")).strip()[:200]

            if verdict_str == "veto":
                log.info(
                    "[orchestrator] VETO ticket #%d %s %s \u2014 %s",
                    v["ticket"], v["symbol"], v["action"], reason or "no reason",
                )
                out.append({
                    **v,
                    "action": "hold",
                    "new_sl": None,
                    "orchestrator_reason": f"VETO: {reason}",
                    "reason_code": "ORC_VETO",
                })
                continue

            if verdict_str in ("override_close", "override") and v["action"] == "hold":
                log.info(
                    "[orchestrator] OVERRIDE ticket #%d %s hold->close \u2014 %s",
                    v["ticket"], v["symbol"], reason or "no reason",
                )
                out.append({
                    **v,
                    "action": "close",
                    "new_sl": None,
                    "orchestrator_reason": f"OVERRIDE hold->close: {reason}",
                    "reason_code": "ORC_OVERRIDE",
                })
                continue

            out.append({**v, "reason_code": "ORC_APPROVE"})
        return out

    # ------------------------------------------------------------------ #
    # Main entry point                                                      #
    # ------------------------------------------------------------------ #

    def orchestrate(
        self,
        verdicts: list,
        portfolio=None,
    ) -> list:
        """
        Take the bot verdicts produced this cycle and return the
        executable plan after orchestrator review.

        Input verdict shape (one per ticket):
            {"symbol": str, "ticket": int,
             "action": "hold|close|move_sl",
             "new_sl": float|None,
             "reason": str,
             "agent": str}

        Output shape:
            same shape, with:
             - vetoed verdicts converted to holds
             - holds turned into closes when overridden
             - "reason_code" on every output verdict
               (ORC_APPROVE / ORC_VETO / ORC_OVERRIDE)
             - "orchestrator_reason" on touched verdicts

        Failure mode: if the LLM call errors or returns malformed JSON,
        ALL verdicts are approved as-is. We do not block trade
        management on orchestrator availability.
        """
        if not verdicts:
            return []

        portfolio = portfolio or PortfolioSnapshot()

        # \u2500\u2500 Operator directives: absolute priority, bypass LLM \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        d = get_directives()
        directive_out: list = []
        remaining:     list = []
        for v in verdicts:
            sym = (v.get("symbol") or "").upper()
            if sym in d.close_symbols:
                directive_out.append({**v, "action": "close",
                                      "reason_code": "ORC_FORCE_CLOSE",
                                      "orchestrator_reason": "operator directive: force close"})
            elif sym in d.freeze_symbols:
                directive_out.append({**v, "action": "hold",
                                      "reason_code": "ORC_FREEZE",
                                      "orchestrator_reason": "operator directive: frozen"})
            else:
                remaining.append(v)

        if not remaining:
            return directive_out

        # Enrich with per-ticket working memory deltas before any decision
        enriched = self._enrich_with_working_memory(remaining)

        # Update session and volatility trackers every cycle (even on fast path)
        self._update_session_tracker(enriched)
        self._update_atr_history(enriched)

        # Build market context once per cycle so the dashboard can show it
        # regardless of whether the LLM was actually called.
        market_ctx = self._build_market_context(enriched)

        # Fast path: all bots hold, no breach, and no WM alert \u2014 skip LLM.
        if (all(v.get("action") == "hold" for v in enriched)
                and not portfolio.daily_loss_breach
                and not self._wm_needs_review(enriched)):
            fast_out = [{**v, "reason_code": "ORC_APPROVE",
                         "market_context": market_ctx} for v in enriched]
            self._update_working_memory(enriched, fast_out)
            return directive_out + fast_out

        # Append operator notes to portfolio context if set
        if d.notes:
            portfolio = PortfolioSnapshot(**{
                **portfolio.__dict__,
                "notes": (portfolio.notes + " | OPERATOR: " + d.notes).strip(" |"),
            })

        system_prompt = self._build_system_prompt()
        user          = self._build_user_message(enriched, portfolio, market_ctx=market_ctx)

        try:
            raw = self.client.chat(system_prompt, user, expect_json=True, max_tokens=512)
        except Exception as exc:
            log.warning(
                "orchestrator transport error: %s \u2014 approving all verdicts as-is", exc,
            )
            fast_out = [{**v, "reason_code": "ORC_APPROVE",
                         "market_context": market_ctx} for v in enriched]
            self._update_working_memory(enriched, fast_out)
            return directive_out + fast_out

        parsed    = _extract_json_dict(raw) or {}
        decisions = parsed.get("decisions") or []
        if not isinstance(decisions, list):
            log.warning("orchestrator returned non-list decisions; approving all as-is")
            fast_out = [{**v, "reason_code": "ORC_APPROVE",
                         "market_context": market_ctx} for v in enriched]
            self._update_working_memory(enriched, fast_out)
            return directive_out + fast_out

        llm_out = self._apply_decisions(enriched, decisions)

        # Attach the LLM's reasoning and the cycle's market context to each
        # output so the dashboard can show them alongside the decisions.
        llm_reasoning = (parsed.get("reasoning") or "").strip()[:200]
        for v in llm_out:
            if llm_reasoning:
                v["llm_reasoning"] = llm_reasoning
            if market_ctx:
                v["market_context"] = market_ctx

        self._update_working_memory(enriched, llm_out)
        out = directive_out + llm_out

        if HAS_AGENT_MEMORY and _get_agent_memory is not None:
            try:
                _get_agent_memory().record_decisions(remaining, llm_out)
            except Exception as exc:
                log.debug("agent_memory record_decisions failed: %s", exc)

        return out
# Module-level singleton.
_orchestrator: Optional[Orchestrator] = None


def get_orchestrator() -> Orchestrator:
    """Return the shared orchestrator (Agent 0). Lazy-instantiated."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator


# ========================================================================== #
# Operator directives — freeze / force-close / exposure cap                  #
# ========================================================================== #

@dataclass
class GlobalDirectives:
    """Operator-level overrides applied before any bot or LLM logic.

    freeze_symbols  : hold only — no tighten, no close.
    close_symbols   : force-close immediately, bypasses orchestrator.
    max_exposure_pct: not yet enforced in bots (future); stored for display.
    notes           : forwarded verbatim into the orchestrator LLM prompt.
    """
    freeze_symbols:   set   = field(default_factory=set)
    close_symbols:    set   = field(default_factory=set)
    max_exposure_pct: float = 0.0
    notes:            str   = ""

    def summary(self) -> str:
        parts = []
        if self.freeze_symbols:
            parts.append(f"freeze={sorted(self.freeze_symbols)}")
        if self.close_symbols:
            parts.append(f"force-close={sorted(self.close_symbols)}")
        if self.max_exposure_pct > 0:
            parts.append(f"max_exp={self.max_exposure_pct:.0f}%")
        if self.notes:
            parts.append(f"note={self.notes[:40]!r}")
        return "; ".join(parts) if parts else "no active directives"


_directives: GlobalDirectives = GlobalDirectives()
_directives_lock = threading.Lock()


def get_directives() -> GlobalDirectives:
    with _directives_lock:
        return _directives


def set_directives(d: GlobalDirectives) -> None:
    global _directives
    with _directives_lock:
        _directives = d
    log.info("GlobalDirectives updated: %s", d.summary())


# ========================================================================== #
# Back-compat shims for callers still importing the old names                #
# ========================================================================== #
#
# Several places in ai_pro.py and the test suite still import AgentZero,
# get_agent_zero, and PairAgent. Keep skinny shims so a partial migration
# doesn't break imports. The real logic is in Orchestrator + PairBotBase.

class AgentZero:
    """Deprecated. Kept only so legacy imports still resolve.

    Calls to .review() now noop into a permissive verdict — the entry
    review path was removed in this refactor (strategy POC gate is the
    sole bias filter). If you actually want orchestrator behaviour,
    use get_orchestrator() instead.
    """

    name = "Agent Zero (deprecated shim)"

    def __init__(self, client: Optional[OllamaClient] = None) -> None:
        self._noop_warned = False

    def review(self, item: Any, *, symbol: str = "", df: Any = None) -> dict:
        if not self._noop_warned:
            log.warning(
                "ai_agent.AgentZero.review() called but entry review was "
                "removed; returning permissive default. Update caller to "
                "drop the AI entry-review path."
            )
            self._noop_warned = True
        # Return something the old caller shape expects.
        if isinstance(item, PositionContext):
            return {"action": "hold",
                    "reason": "AgentZero deprecated — pair bot handles positions"}
        return {"approve": True, "confidence": 0.5,
                "reason": "AgentZero deprecated — POC gate is the bias filter",
                "htf_detail": "deprecated"}


def get_agent_zero() -> AgentZero:
    """Deprecated. Returns the shim above. New code should call
    get_orchestrator() (cross-pair) or get_pair_agent(symbol) (per-pair)."""
    return AgentZero()


# `PairAgent` was the LLM-backed per-pair manager. Pair bots replace it.
# Keep the name as an alias so old test imports don't fail.
PairAgent = PairBotBase


# ========================================================================== #
# Backend kill-switch                                                         #
# ========================================================================== #

def agent_backend_enabled() -> bool:
    """True if the agent backend is enabled. Set AI_BACKEND to off / none /
    disabled / 0 / false to disable auto-trading. Pair bots are deterministic
    and don't need Ollama; this kill-switch exists so the operator can run
    strategy-only (no auto-trade) without uninstalling anything."""
    val = os.getenv("AI_BACKEND", "agent").strip().lower()
    return val not in ("off", "none", "disabled", "0", "false")


# ========================================================================== #
# Ollama health probe (unchanged surface)                                    #
# ========================================================================== #

def ollama_health() -> dict:
    """Best-effort probe of the Ollama server. Used by the dashboard.

    Note: pair bots don't need Ollama any more — only the orchestrator
    does. So an Ollama outage degrades the system gracefully:
        * pair bots continue to make hold/tighten/close decisions
        * the orchestrator falls back to "approve all verdicts" on
          transport error, so positions are still managed
        * only the cross-pair veto/override capability is lost
    """
    client = OllamaClient()
    out = {
        "reachable": False,
        "model_loaded": False,
        "url": client.url,
        "model": client.model,
        "error": None,
    }
    try:
        data = _http_get_json(
            f"{client.url}/api/tags",
            timeout=min(3.0, client.timeout),
        )
        out["reachable"] = True
        try:
            tags = data.get("models") or []
            wanted = client.model.split(":")[0].lower()
            out["model_loaded"] = any(
                wanted in str(t.get("name", "")).lower() for t in tags
            )
        except Exception as exc:
            out["error"] = f"tags parse: {exc!s}"
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        out["error"] = f"cannot reach Ollama at {client.url}: {reason}"
    except Exception as exc:
        out["error"] = str(exc)
    return out
