"""
ai_agent.py — Local Ollama agent replacing DeepSeek-R1 for entry review and
in-flight risk management.

Design principles:

  1. HARD SL ALWAYS FIRES (code-enforced).  The risk agent can only TIGHTEN
     a stop or request an EARLY close.  It cannot override the broker's
     hard stop — that authority has been removed.

  2. DAILY BIAS is the agent's filter and is initiated ONLY via the prompt.
     This module does not classify or gate on bias; it only fetches raw
     daily_open and current_price and hands them to the agent. The LLM
     applies the filter per its system prompt.

The module exposes two classes used by ai_pro.py:
  • EntryReviewAgent.review(symbol, signal, df)  -> dict
  • RiskReviewAgent.review(symbol, pos_ctx)      -> dict

Both classes lazily create a shared OllamaClient.  Configuration via env:
  OLLAMA_URL      (default http://localhost:11434)
  OLLAMA_MODEL    (default qwen2.5:14b-instruct)
  AGENT_TIMEOUT_S (default 30)
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import requests

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
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:14b-instruct")
        self.timeout = float(timeout or os.getenv("AGENT_TIMEOUT_S", "30"))

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
        resp = requests.post(f"{self.url}/api/chat", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return (data.get("message") or {}).get("content", "") or ""


# ========================================================================== #
# Raw market context (no code-level bias decision)                            #
# ========================================================================== #
#
# This module does NOT classify the market.  It only fetches the daily open
# and current price and hands them to the agent.  The daily-bias filter is
# initiated entirely in the prompt — the LLM does the classification.

@dataclass
class MarketContext:
    """Raw values passed to the agent. No decision, no label."""
    daily_open: float = 0.0
    current_price: float = 0.0
    detail: str = "no data"

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
    Fetch today's daily open and current price from MT5. That's it.
    No bias labelling, no gating, no decision.
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

    ctx.detail = f"DO={ctx.daily_open:.5f} px={ctx.current_price:.5f}"
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
# Entry review agent                                                          #
# ========================================================================== #

ENTRY_SYSTEM_PROMPT = (
    "You are a professional day trader who specialises in trading GBPUSD, "
    "GBPJPY, EURUSD and EURJPY.\n\n"
    "Your main goal is to MAXIMISE winners and MINIMISE losers through "
    "effective trade management. You can also gauge daily bias and use it to "
    "filter signals.\n\n"
    "Daily bias rule (price vs today's daily open):\n"
    "  • current_price > daily_open → BULLISH → approve BUY only.\n"
    "  • current_price < daily_open → BEARISH → approve SELL only.\n"
    "  • current_price = daily_open → NEUTRAL → reject.\n\n"
    "If the signal's side agrees with the daily bias, APPROVE; otherwise "
    "REJECT.\n\n"
    "Reply with a single JSON object on one line:\n"
    "  {\"approve\": <bool>, \"confidence\": <0.0-1.0>, \"reason\": \"<=20 words\"}"
)


class EntryReviewAgent:
    """LLM-driven pre-trade approval. The daily-bias filter lives in the prompt."""

    def __init__(self, client: Optional[OllamaClient] = None) -> None:
        self.client = client or OllamaClient()

    def review(self, symbol: str, signal: dict, df: Any) -> dict:
        """
        signal must include at least: signal (BUY/SELL), entry_price, stop_loss,
        take_profit, confidence, signal_source, reason.
        Returns: {"approve": bool, "reason": str, "confidence": float,
                  "htf_detail": str}
        """
        side  = str(signal.get("signal", "")).upper()
        entry = _coerce_float(signal.get("entry_price"))
        sl    = _coerce_float(signal.get("stop_loss"))
        tp    = _coerce_float(signal.get("take_profit"))
        risk  = abs(entry - sl) if (entry and sl) else 0.0
        rew   = abs(tp - entry) if (entry and tp) else 0.0
        rr    = (rew / risk) if risk > 0 else 0.0

        # Fetch raw daily open + current price. The prompt does the bias work.
        ctx = fetch_market_context(symbol)

        user = (
            f"SIGNAL\n"
            f"  symbol: {symbol}\n"
            f"  side: {side}\n"
            f"  setup: {signal.get('signal_source','?')}\n"
            f"  entry: {entry:.5f}  sl: {sl:.5f}  tp: {tp:.5f}\n"
            f"  risk_reward: {rr:.2f}\n"
            f"  strategy_confidence: {signal.get('confidence', 0)}%\n"
            f"  reason: {str(signal.get('reason',''))[:200]}\n\n"
            f"MARKET CONTEXT\n"
            f"  daily_open: {ctx.daily_open:.5f}   current_price: {ctx.current_price:.5f}\n"
        )

        try:
            raw = self.client.chat(ENTRY_SYSTEM_PROMPT, user,
                                    expect_json=True, max_tokens=160)
        except Exception as exc:
            log.warning("agent entry review transport error: %s — default reject", exc)
            return {
                "approve": False,
                "reason": f"agent transport error ({exc!s}) — rejected conservatively",
                "confidence": 0.0,
                "htf_detail": ctx.detail,
            }

        parsed = _extract_json_dict(raw) or {}
        approve = _coerce_bool(parsed.get("approve"), default=False)
        conf    = max(0.0, min(1.0, _coerce_float(parsed.get("confidence"), default=0.0)))
        reason  = str(parsed.get("reason", "")).strip()[:200] or "no reason"

        return {
            "approve": approve,
            "reason": reason,
            "confidence": conf,
            "htf_detail": ctx.detail,
        }


# ========================================================================== #
# Risk review agent                                                           #
# ========================================================================== #

RISK_SYSTEM_PROMPT = (
    "You are a professional day trader who specialises in trading GBPUSD, "
    "GBPJPY, EURUSD and EURJPY.\n\n"
    "Your main goal is to MAXIMISE winners and MINIMISE losers through "
    "effective trade management. You can also gauge daily bias and use it to "
    "filter signals.\n\n"
    "Reply with a single JSON object on one line:\n"
    "  {\"action\":\"hold|tighten|close_early\", \"new_sl\": <float or null>, "
    "\"reason\":\"<=20 words\"}"
)


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


class RiskReviewAgent:
    """LLM-in-the-loop risk manager that can only tighten or close early."""

    def __init__(self, client: Optional[OllamaClient] = None) -> None:
        self.client = client or OllamaClient()

    def review(self, ctx: PositionContext) -> dict:
        """
        Returns one of:
          {"action": "hold"}
          {"action": "close", "reason": str}           # close_early
          {"action": "move_sl", "new_sl": float, "reason": str}   # tighten only
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
            raw = self.client.chat(RISK_SYSTEM_PROMPT, user,
                                    expect_json=True, max_tokens=160)
        except Exception as exc:
            log.warning("agent risk review transport error: %s — hold (hard SL still active)", exc)
            return {"action": "hold", "reason": f"agent transport error ({exc!s})"}

        parsed = _extract_json_dict(raw) or {}
        action = str(parsed.get("action", "hold")).lower().strip()
        reason = str(parsed.get("reason", "")).strip()[:160] or "no reason"

        if action in ("close", "close_early", "exit"):
            return {"action": "close", "reason": f"agent: {reason}"}

        if action in ("tighten", "trail", "move_sl"):
            new_sl = parsed.get("new_sl")
            try:
                new_sl = round(float(new_sl), ctx.digits)
            except Exception:
                return {"action": "hold", "reason": "agent: invalid new_sl — hold"}

            # --- hard rails: wrong side of price?
            wrong_side = (is_buy and new_sl >= ctx.cur_price) or \
                         ((not is_buy) and new_sl <= ctx.cur_price)
            if wrong_side:
                return {"action": "hold",
                        "reason": f"agent: proposed SL {new_sl} is wrong side of price — hold"}

            # --- hard rails: only tighten, never loosen (and never beyond entry to worse)
            if ctx.cur_sl:
                not_tighter = (is_buy and new_sl <= ctx.cur_sl) or \
                              ((not is_buy) and new_sl >= ctx.cur_sl)
                if not_tighter:
                    return {"action": "hold",
                            "reason": f"agent: proposed SL not tighter than {ctx.cur_sl} — hold"}

            return {"action": "move_sl", "new_sl": new_sl,
                    "reason": f"agent: {reason}"}

        # Anything else (including "hold") → do nothing.  Hard SL remains.
        return {"action": "hold", "reason": f"agent: {reason}"}


# ========================================================================== #
# Module-level singletons (cheap to construct, heavy to call)                 #
# ========================================================================== #

_entry_agent: Optional[EntryReviewAgent] = None
_risk_agent:  Optional[RiskReviewAgent]  = None


def get_entry_agent() -> EntryReviewAgent:
    global _entry_agent
    if _entry_agent is None:
        _entry_agent = EntryReviewAgent()
    return _entry_agent


def get_risk_agent() -> RiskReviewAgent:
    global _risk_agent
    if _risk_agent is None:
        _risk_agent = RiskReviewAgent()
    return _risk_agent


def agent_backend_enabled() -> bool:
    """True if AI_BACKEND env var selects the Ollama agent."""
    return os.getenv("AI_BACKEND", "deepseek").strip().lower() in ("agent", "ollama")
