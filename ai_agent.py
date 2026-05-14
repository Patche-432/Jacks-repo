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
  OLLAMA_MODEL    (default qwen2.5:3b-instruct)
  AGENT_TIMEOUT_S (default 60)
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
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
        self.timeout = float(timeout or os.getenv("AGENT_TIMEOUT_S", "60"))

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
    "You are AGENT ZERO, supervisor for a 4-pair forex bot trading CHoCH + daily-levels strategy.\n"
    "\n"
    "PRIMARY MISSION: protect the entry reason. Every trade was opened because price showed a "
    "Change of Character (CHoCH) at a key daily level. Your only job is to assess whether "
    "that entry reason is still valid or has been invalidated by current price action.\n"
    "\n"
    "DEFAULT: APPROVE. The bots are correct unless price action has clearly spoken against the trade.\n"
    "\n"
    "Authority — judge each trade individually on its own price action:\n"
    "  APPROVE      → entry reason still intact. Price respecting structure. Let the bot run its plan.\n"
    "  VETO         → bot wants to close or tighten, but price action shows the setup is still valid\n"
    "                  (structure holding, no opposing CHoCH, still within the daily level range).\n"
    "                  Block the premature exit — the edge has not played out yet.\n"
    "  OVERRIDE     → bot is holding, but price action has INVALIDATED the entry reason:\n"
    "                  opposing CHoCH printed, price has broken back through the entry level,\n"
    "                  or the original directional bias is structurally broken.\n"
    "                  Force close — the reason this trade exists is gone.\n"
    "\n"
    "Never act on portfolio metrics, daily P&L, or what other pairs are doing. "
    "Each decision is based solely on whether that pair's price action still supports its entry.\n"
    "\n"
    "Reply JSON only, one line:\n"
    "  {\"decisions\": [{\"ticket\": <int>, \"verdict\": \"approve|veto|override_close\", \"reason\": \"<20w\"}, ...]}\n"
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

    def __init__(self, client=None):
        self.client = client or OllamaClient()

    # ------------------------------------------------------------------ #
    # Helpers                                                               #
    # ------------------------------------------------------------------ #

    @classmethod
    def _load_ml_context(cls) -> str:
        """Return ML insights block string, cached by file mtime."""
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
            ml_lines = []
            tp_lines = []
            for sym, d in sorted(ins.items()):
                wr   = d.get("win_rate") or 0.0
                imps = d.get("importances") or {}
                top  = max(imps, key=imps.get) if imps else None
                tv   = imps.get(top, 0.0) if top else 0.0
                n    = d.get("trade_count", 0)
                ml_lines.append(
                    f"  {sym}: WR={wr*100:.0f}% n={n}"
                    + (f" top={top}({tv:.2f})" if top else "")
                )
                tp = d.get("tuned_params") or {}
                if tp:
                    tp_lines.append(
                        f"  {sym}: SL={tp.get('sl_atr_mult', 2.5):.2f}\u00d7"
                        f" TP={tp.get('tp_atr_mult', 4.5):.2f}\u00d7"
                        f" BE={tp.get('be_buffer_pips', 1.0):.1f}p"
                        f" trail={tp.get('trail_atr_mult', 1.0):.2f}\u00d7"
                    )
            text = ""
            if ml_lines:
                text = (
                    "\nML INSIGHTS (backtest-derived):\n"
                    + "\n".join(ml_lines)
                    + "\n  Higher WR = give more patience before overriding holds."
                    + "\n  Top feature = what predicts wins for this pair."
                )
                if tp_lines:
                    text += (
                        "\n\nTUNED SL/TP (live bot uses these per pair):\n"
                        + "\n".join(tp_lines)
                        + "\n  Wider SL = position needs more room, don't override too early."
                    )
            cache.update({"path": path, "mtime": mtime, "text": text})
            return text
        except Exception as exc:
            log.debug("ML insights load failed: %s", exc)
            return ""

    def _build_system_prompt(self) -> str:
        """Build system prompt, optionally appending agent-memory context."""
        prompt = ORCHESTRATOR_SYSTEM_PROMPT
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
    def _build_user_message(verdicts, portfolio, ml_context: str) -> str:
        """Format bot verdicts + price action context as the LLM user message."""
        import datetime as _dt
        now     = _dt.datetime.now(_dt.timezone.utc)
        session = Orchestrator._session_label(now.hour)

        lines = []
        for v in verdicts:
            new_sl    = v.get("new_sl")
            sl_str    = f"{new_sl:.5f}" if isinstance(new_sl, (int, float)) else "\u2014"
            entry     = v.get("entry")
            cur_price = v.get("cur_price")
            atr       = v.get("atr") or 0.0
            direction = v.get("direction", "?")
            profit_pts= v.get("profit_pts")

            entry_str  = f"{entry:.5f}"     if entry     is not None else "?"
            price_str  = f"{cur_price:.5f}" if cur_price is not None else "?"
            profit_str = (f"{profit_pts:+.0f}pts"
                          f"({profit_pts/atr:.1f}R)" if profit_pts is not None and atr > 0
                          else f"{profit_pts:+.0f}pts" if profit_pts is not None else "?")

            struct_broken   = v.get("structure_broken")
            trend_intact    = v.get("trend_intact")
            fresh_structure = v.get("fresh_structure")

            struct_str = (
                "STRUCTURE_BROKEN" if struct_broken
                else "fresh_structure" if fresh_structure
                else "structure_ok"
            )
            trend_str = (
                "TREND_BROKEN" if trend_intact is False
                else "trend_ok"
            )

            lines.append(
                f"  ticket={v['ticket']} {v['symbol']} {direction} "
                f"entry={entry_str} price={price_str} P&L={profit_str}\n"
                f"    price_action: {struct_str} | {trend_str}\n"
                f"    bot_verdict:  action={v['action']} new_sl={sl_str}\n"
                f"    bot_reason:   {(v.get('reason') or '')[:120]}"
            )

        msg = (
            f"SESSION: {session} (UTC {now.hour:02d}h)\n\n"
            f"TRADES ({len(verdicts)})\n"
            + "\n".join(lines)
        )
        if ml_context:
            msg += ml_context
        return msg

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

        # Fast path: all remaining bots say hold and no breach \u2014 skip LLM.
        if all(v.get("action") == "hold" for v in remaining) and not portfolio.daily_loss_breach:
            return directive_out + [{**v, "reason_code": "ORC_APPROVE"} for v in remaining]

        # Append operator notes to portfolio context if set
        if d.notes:
            portfolio = PortfolioSnapshot(**{
                **portfolio.__dict__,
                "notes": (portfolio.notes + " | OPERATOR: " + d.notes).strip(" |"),
            })

        system_prompt = self._build_system_prompt()
        ml_context    = self._load_ml_context()
        user          = self._build_user_message(remaining, portfolio, ml_context)

        try:
            raw = self.client.chat(system_prompt, user, expect_json=True, max_tokens=400)
        except Exception as exc:
            log.warning(
                "orchestrator transport error: %s \u2014 approving all verdicts as-is", exc,
            )
            return directive_out + [{**v, "reason_code": "ORC_APPROVE"} for v in remaining]

        parsed    = _extract_json_dict(raw) or {}
        decisions = parsed.get("decisions") or []
        if not isinstance(decisions, list):
            log.warning("orchestrator returned non-list decisions; approving all as-is")
            return directive_out + [{**v, "reason_code": "ORC_APPROVE"} for v in remaining]

        llm_out = self._apply_decisions(remaining, decisions)
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
