"""
ai_agent.py — Agent Zero, the single local-Ollama trader in the pipeline.

Design principles:

  1. HARD SL ALWAYS FIRES (code-enforced).  While a trade is live, Agent
     Zero can only TIGHTEN a stop or request an EARLY close.  He cannot
     loosen a stop, widen a TP, or override the broker's hard stop —
     that authority has been removed in code.

  2. AGENT ZERO IS A DAY-TRADER, NOT A BIAS GATE.  Directional bias is
     handled upstream by the strategy's POC bias filter (BUY only above
     POC, SELL only below POC), so by the time a signal reaches Agent
     Zero the side is already settled.  Agent Zero's two jobs are:
       (a) sanity-check the entry — confirm setup quality on its own
           merits (structure, distances, confidence), approve or reject;
       (b) manage every live trade like a real day trader — protect
           profit, cut losers early, ride winners.  Allowed live actions:
           hold, tighten, close_early.

  3. ONE AGENT, ONE PERSONA.  The pipeline has exactly one AI persona —
     Agent Zero — a professional day trader who runs each trade end-to-
     end.  One system prompt, one Ollama client, one public method
     (`AgentZero.review`) whose behaviour switches on the input shape.

Public surface used by ai_pro.py:

  • get_agent_zero().review(signal_dict, symbol=..., df=...) -> dict
  • get_agent_zero().review(position_ctx)                    -> dict   [legacy]

No shims, no aliases, no second opinion — the pipeline cannot disagree
with itself.

Configuration via env:
  OLLAMA_URL      (default http://localhost:11434)
  OLLAMA_MODEL    (default qwen2.5:3b-instruct — CPU-friendly; override
                   with qwen2.5:7b-instruct or qwen2.5:14b-instruct on
                   GPU hosts, or llama3.2:3b on Snapdragon X / ARM CPUs)
  AGENT_TIMEOUT_S (default 60 — bumped from 30 to tolerate CPU inference)
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Optional

# HTTP to Ollama goes through the stdlib (urllib.request + json), so
# Agent Zero has zero external pip dependencies beyond what Python
# already ships with. This removed the old `requests` requirement that
# was tripping up fresh installs.
import json as _json
import urllib.request
import urllib.error


def _http_post_json(url: str, payload: dict, timeout: float) -> dict:
    """POST JSON, return parsed JSON dict. Raises on HTTP / transport error."""
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
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"non-JSON response from {url}: {exc}") from exc


def _http_get_json(url: str, timeout: float) -> dict:
    """GET, return parsed JSON dict. Raises on HTTP / transport error."""
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    try:
        return _json.loads(raw.decode("utf-8", errors="replace"))
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"non-JSON response from {url}: {exc}") from exc

log = logging.getLogger("ai_agent")


# ========================================================================== #
# Ollama HTTP client                                                          #
# ========================================================================== #

class OllamaClient:
    """Thin wrapper around Ollama's /api/chat endpoint with JSON mode."""

    def __init__(self,
                 url: Optional[str] = None,
                 model: Optional[str] = None,
                 timeout: Optional[float] = None) -> None:
        self.url = (url or os.getenv("OLLAMA_URL", "http://localhost:11434")).rstrip("/")
        # Default to qwen2.5:3b-instruct: small enough to run responsively
        # on CPU (including Snapdragon X / ARM64), strong enough to follow
        # the strict JSON-only system prompt. Override via OLLAMA_MODEL for
        # GPU hosts that can handle qwen2.5:7b/14b-instruct.
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")
        # 60s ceiling — CPU inference of a ~200-token JSON verdict on a 3B
        # model typically finishes in 5-15s, but we keep headroom so slow
        # first-token latency (model load) doesn't trip the review.
        self.timeout = float(timeout or os.getenv("AGENT_TIMEOUT_S", "60"))

    def chat(self,
             system: str,
             user: str,
             *,
             temperature: float = 0.0,
             expect_json: bool = True,
             max_tokens: int = 512) -> str:
        """
        Call /api/chat and return the raw assistant string. JSON mode is
        enabled when expect_json=True (Ollama >=0.1.32 supports format=json).
        Raises on transport error so callers can choose a safe fallback.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
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
# Raw market context (back-compat shim — POC gate now lives in the strategy) #
# ========================================================================== #
#
# Bias filtering used to be initiated here via daily_open / current_price
# fed to the LLM.  That responsibility has moved to the strategy layer
# (`ai_pro.generate_trade_signal` applies the POC bias gate).  This shim
# is kept only so older callers that still import `MarketContext`,
# `HTFBias`, `fetch_market_context`, or `compute_htf_bias` don't crash.

@dataclass
class MarketContext:
    """Back-compat shim. Bias decisions no longer flow through here."""
    daily_open: float = 0.0
    current_price: float = 0.0
    detail: str = "bias gate moved to strategy POC filter"

    # Back-compat aliases kept so older callers importing HTFBias still work.
    d1: str = "n/a"
    h4: str = "n/a"
    h1: str = "n/a"
    score: int = 0
    reason: str = ""


# Back-compat alias — older code imports `HTFBias`.
HTFBias = MarketContext


def fetch_market_context(symbol: str) -> MarketContext:
    """
    Back-compat shim.  Returns a `MarketContext` populated with today's
    daily open and current price purely for legacy callers that still
    log this; the agent no longer consumes these values for bias.
    """
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


# Back-compat alias — older code calls `compute_htf_bias`.
compute_htf_bias = fetch_market_context


# ========================================================================== #
# JSON extraction helpers                                                     #
# ========================================================================== #

_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _extract_json_dict(raw: str) -> Optional[dict]:
    if not raw:
        return None
    raw = raw.strip()
    # fast path — the whole string is JSON
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    # scan for first balanced object
    for m in _JSON_RE.finditer(raw):
        try:
            parsed = json.loads(m.group(0))
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
# Position context dataclass — the shape Agent Zero reads for a live trade   #
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
# Agent Zero — the ONE agent in the entire pipeline                           #
# ========================================================================== #
#
# Agent Zero is a professional day trader who runs every trade end-to-end:
# before entry he filters each strategy signal through the daily-bias gate,
# and once a trade is live he actively manages it — tightening the stop or
# closing early, never loosening.  One system prompt, one Ollama client, one
# public method (AgentZero.review).  The input he receives on his desk — a
# SIGNAL block for a fresh setup or a POSITION block for a live trade —
# determines the shape of his JSON reply.

AGENT_ZERO_SYSTEM_PROMPT = (
    "You are AGENT ZERO — a professional day trader specialising in GBPUSD, "
    "GBPJPY, EURUSD and EURJPY. These four pairs are your bread and butter: "
    "the cleanest London/NY sessions, where your edge is strongest. You "
    "also review setups on other symbols the strategy presents, but you "
    "know the core four best and trust them most.\n"
    "\n"
    "You run every trade end-to-end — from the moment a signal lands on "
    "your desk to the moment the position closes — so your thinking stays "
    "consistent across the trade's whole life. Your edge rests on two "
    "habits:\n"
    "\n"
    "1) ENTRY QUALITY CHECK. Directional bias is already handled upstream "
    "by the strategy's POC bias filter (BUY only above POC, SELL only "
    "below POC), so the side is settled before the signal reaches you. "
    "Your job at entry is to sanity-check the SETUP — does the named "
    "environment (CHoCH or Continuation) make sense at the level it cites, "
    "are the stop and target distances reasonable, is the strategy "
    "confidence plausible? Approve clean setups, reject only the broken "
    "ones. Do not re-apply directional bias.\n"
    "\n"
    "2) ACTIVE TRADE MANAGEMENT. Once a trade is live, you manage it like "
    "a real day trader to MAXIMISE winners and MINIMISE losers. You may "
    "TIGHTEN the stop (move it closer to price, never further) or ask to "
    "CLOSE EARLY when the structure that justified the trade has broken. "
    "Use peak unrealised profit, current ATR, fresh structure breaks, and "
    "trend-intactness clues from the POSITION block to decide. You never "
    "loosen risk and you cannot widen the TP. Allowed live actions: "
    "hold, tighten, close_early.\n"
    "\n"
    "You always reply with a single JSON object on one line — no prose, "
    "no markdown, no extra keys. The shape of the JSON depends on what "
    "the user has handed you:\n"
    "\n"
    "  • SIGNAL block (pre-entry review):\n"
    "      {\"approve\": <bool>, \"confidence\": <0.0-1.0>, "
    "\"reason\": \"<=20 words\"}\n"
    "\n"
    "  • POSITION block (live-trade review):\n"
    "      {\"action\": \"hold|tighten|close_early\", "
    "\"new_sl\": <float or null>, \"reason\": \"<=20 words\"}\n"
    "      If action is hold or close_early, new_sl must be null. "
    "If action is tighten, new_sl must be a valid numeric stop that is "
    "strictly tighter than the current stop and on the correct side of "
    "the current price.\n"
)


class AgentZero:
    """
    The single AI persona in the pipeline — a professional day trader who
    runs every trade end-to-end: he filters fresh signals by the daily-bias
    gate and actively manages live positions by tightening stops or closing
    early.  One system prompt, one Ollama client, one method.
    """

    name = "Agent Zero"

    def __init__(self, client: Optional[OllamaClient] = None) -> None:
        self.client = client or OllamaClient()

    def review(self,
               item: "dict | PositionContext",
               *,
               symbol: str = "",
               df: Any = None) -> dict:
        """
        Hand Agent Zero whatever is on his desk and he decides:

          • a strategy signal (dict) → pre-entry review; returns
            {"approve": bool, "confidence": float, "reason": str,
             "htf_detail": str}

          • a live position (PositionContext) → live-trade review; returns
            {"action": "hold"|"close"|"move_sl", "new_sl"?: float,
             "reason": str}

        For a signal, `symbol` (and optionally `df`) must also be supplied as
        keyword arguments — Agent Zero needs the symbol to fetch today's
        daily open.
        """
        if isinstance(item, PositionContext):
            return self._review_position(item)
        if isinstance(item, dict):
            return self._review_signal(symbol, item, df)
        raise TypeError(
            f"Agent Zero cannot review {type(item).__name__}; "
            f"expected a signal dict or a PositionContext"
        )

    # ------------------------------------------------------------------ #
    # Internal helpers — one private method per input shape.  The public #
    # surface is .review(...); these just format the user message and    #
    # parse the JSON reply for each shape.                               #
    # ------------------------------------------------------------------ #

    def _review_signal(self, symbol: str, signal: dict, df: Any) -> dict:
        if not symbol:
            raise ValueError("Agent Zero needs a symbol to review a signal")
        side  = str(signal.get("signal", "")).upper()
        entry = _coerce_float(signal.get("entry_price"))
        sl    = _coerce_float(signal.get("stop_loss"))
        tp    = _coerce_float(signal.get("take_profit"))
        risk  = abs(entry - sl) if (entry and sl) else 0.0
        rew   = abs(tp - entry) if (entry and tp) else 0.0
        rr    = (rew / risk) if risk > 0 else 0.0

        # Strategy has already gated by POC bias before this method is
        # called. We surface the POC value to the agent purely as audit
        # context — the agent's job is quality review, not bias.
        vp  = signal.get("volume_profile") or {}
        poc = _coerce_float(vp.get("poc"))

        user = (
            f"SIGNAL\n"
            f"  symbol: {symbol}\n"
            f"  side: {side}\n"
            f"  setup: {signal.get('signal_source','?')}\n"
            f"  entry: {entry:.5f}  sl: {sl:.5f}  tp: {tp:.5f}\n"
            f"  risk_reward: {rr:.2f}\n"
            f"  strategy_confidence: {signal.get('confidence', 0)}%\n"
            f"  reason: {str(signal.get('reason',''))[:200]}\n\n"
            f"CONTEXT (already POC-gated by strategy)\n"
            f"  poc: {poc:.5f}   entry: {entry:.5f}\n"
        )

        bias_detail = (
            f"POC={poc:.5f} entry={entry:.5f} side={side} (strategy-gated)"
        )

        try:
            raw = self.client.chat(AGENT_ZERO_SYSTEM_PROMPT, user,
                                   expect_json=True, max_tokens=160)
        except Exception as exc:
            log.warning("agent zero signal-review transport error: %s — default reject", exc)
            return {
                "approve": False,
                "reason": f"agent transport error ({exc!s}) — rejected conservatively",
                "confidence": 0.0,
                "htf_detail": bias_detail,
            }

        parsed = _extract_json_dict(raw) or {}
        approve = _coerce_bool(parsed.get("approve"), default=False)
        conf    = max(0.0, min(1.0, _coerce_float(parsed.get("confidence"), default=0.0)))
        reason  = str(parsed.get("reason", "")).strip()[:200] or "no reason"

        return {
            "approve": approve,
            "reason": reason,
            "confidence": conf,
            "htf_detail": bias_detail,
        }

    def _review_position(self, ctx: PositionContext) -> dict:
        """
        Live-trade review.  Agent Zero may TIGHTEN the stop or request an
        EARLY close — never loosen, never widen TP, never override the
        broker's hard SL.  Hard rails enforce these invariants in code,
        regardless of what the LLM proposes.
        """
        is_buy = ctx.side.upper() == "BUY"

        user = (
            f"POSITION\n"
            f"  symbol: {ctx.symbol}  side: {ctx.side}  ticket: {ctx.ticket}\n"
            f"  entry: {ctx.entry:.5f}   price: {ctx.cur_price:.5f}\n"
            f"  current_sl: {ctx.cur_sl:.5f}   current_tp: {ctx.cur_tp:.5f}\n"
            f"  unrealised_pts: {ctx.profit_pts:+.1f}   peak_pts: {ctx.peak_pts:+.1f}\n"
            f"  atr: {ctx.atr:.5f}   digits: {ctx.digits}\n"
            f"STRUCTURE\n"
            f"  trend_intact: {ctx.trend_intact}\n"
            f"  structure_broken: {ctx.structure_broken}\n"
            f"  fresh_structure: {ctx.fresh_structure}\n"
            f"NOTES\n  {ctx.notes}\n"
        )

        try:
            raw = self.client.chat(AGENT_ZERO_SYSTEM_PROMPT, user,
                                   expect_json=True, max_tokens=160)
        except Exception as exc:
            log.warning("agent zero position-review transport error: %s — hold (hard SL still active)", exc)
            return {"action": "hold", "reason": f"agent transport error ({exc!s})"}

        parsed = _extract_json_dict(raw) or {}
        action = str(parsed.get("action", "hold")).lower().strip()
        reason = str(parsed.get("reason", "")).strip()[:160] or "no reason"

        if action in ("close", "close_early", "exit"):
            return {"action": "close", "reason": f"agent zero: {reason}"}

        if action in ("tighten", "trail", "move_sl"):
            new_sl = parsed.get("new_sl")
            try:
                new_sl = round(float(new_sl), ctx.digits)
            except Exception:
                return {"action": "hold", "reason": "agent zero: invalid new_sl — hold"}

            # --- hard rails: wrong side of price?
            wrong_side = (is_buy and new_sl >= ctx.cur_price) or \
                         ((not is_buy) and new_sl <= ctx.cur_price)
            if wrong_side:
                return {"action": "hold",
                        "reason": f"agent zero: proposed SL {new_sl} is wrong side of price — hold"}

            # --- hard rails: only tighten, never loosen
            if ctx.cur_sl:
                not_tighter = (is_buy and new_sl <= ctx.cur_sl) or \
                              ((not is_buy) and new_sl >= ctx.cur_sl)
                if not_tighter:
                    return {"action": "hold",
                            "reason": f"agent zero: proposed SL not tighter than {ctx.cur_sl} — hold"}

            return {"action": "move_sl", "new_sl": new_sl,
                    "reason": f"agent zero: {reason}"}

        # Anything else (including "hold") → do nothing. Hard SL remains.
        return {"action": "hold", "reason": f"agent zero: {reason}"}


# ========================================================================== #
# Module-level singleton (cheap to construct, heavy to call)                  #
# ========================================================================== #

_agent_zero: Optional[AgentZero] = None


def get_agent_zero() -> AgentZero:
    """Return the one shared Agent Zero singleton — the only agent in the pipeline."""
    global _agent_zero
    if _agent_zero is None:
        _agent_zero = AgentZero()
    return _agent_zero


def agent_backend_enabled() -> bool:
    """
    True if Agent Zero is the active (and only) AI backend.

    Agent Zero is the sole brain in the pipeline — there is no alternate
    backend. The AI_BACKEND env var exists only as a kill-switch: set it
    to "off" / "none" / "disabled" to run strategy-only (no AI gate, and
    therefore no auto-trading). Any other value — or unset — keeps Agent
    Zero on.
    """
    val = os.getenv("AI_BACKEND", "agent").strip().lower()
    return val not in ("off", "none", "disabled", "0", "false")


def ollama_health() -> dict:
    """
    Best-effort probe of the Ollama server. Returns a small dict that the
    dashboard can render — never raises.

    {
      "reachable":    bool,       # TCP + HTTP reachable
      "model_loaded": bool,       # configured model shows up in /api/tags
      "url":          str,        # base URL we tried
      "model":        str,        # model name we tried
      "error":        str|None,   # last error message, if any
    }
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
        # Most common real-world case: Ollama isn't running, or not on this URL.
        reason = getattr(exc, "reason", exc)
        out["error"] = f"cannot reach Ollama at {client.url}: {reason}"
    except Exception as exc:
        out["error"] = str(exc)
    return out
