"""
╔════════════════════════════════════════════════════════════════════╗
║                      ⚡ FORTIS AI PRO ⚡                          ║
║   CHoCH + Daily Levels Strategy — AI-Enhanced Edition              ║
╚════════════════════════════════════════════════════════════════════╝

Merges the complete FORTIS AI Pro signal engine (CHoCH + Continuation on 4
environments, ATR-based SL/TP, partial-close/breakeven trail)
with the full AI stack from algo-v2:

  ► LocalLLM         — DeepSeek-R1-Distill-Qwen-1.5B via HuggingFace Transformers
  ► Thought Logger   — thread-safe in-memory AI reasoning log (last 120)
  ► AI Signal Review — DeepSeek validates every AI Pro signal before execution
  ► AI Risk Manager  — DeepSeek reviews open positions every N ticks
                       (trail harder / tighten / close if momentum gone)
  ► Trade Memory     — JSON persistence (last 100 trades, win/loss stats)
  ► Flask Dashboard  — REST API + embedded UI at http://localhost:5000

The 4 Environments
----------------------------------------------
ENV 1 │ CHoCH BUY  at PDL — failed Lower Low anchored at Previous Day Low
ENV 2 │ CHoCH SELL at PDH — failed Higher High anchored at Previous Day High
ENV 3 │ Continuation BUY  — broke above PDH, retesting it as support
ENV 4 │ Continuation SELL — broke below PDL, retesting it as resistance

All environments require level interaction for confirmation.
CHoCH signals have priority (confidence 85) over Continuation (65-75).
Auto-trade fires when DeepSeek approves (confidence pre-filtered to >= 75).

Usage:
    python ai_pro.py              # starts Flask on :5000, bot idle
    python ai_pro.py --run        # starts bot immediately on EURUSD

Requirements:
    pip install flask flask-cors MetaTrader5 numpy pandas transformers torch pyyaml
"""

from __future__ import annotations

# ============================================================ #
# SECTION 0 — MT5 CONFIG                                       #
# ============================================================ #

MT5_CONFIG: dict = {
    "timeout":  10_000,
    "portable": False,
    "path":     None,   # e.g. "C:/Program Files/MetaTrader 5/terminal64.exe"
    "login":    None,   # int account number
    "password": None,   # string
    "server":   None,   # string broker server name
}

# ============================================================ #
# CREDENTIALS FILE — managed via core/mt5_config.py           #
# ============================================================ #

def _load_saved_credentials() -> None:
    """Load MT5 credentials from core/mt5_credentials_config.json into MT5_CONFIG."""
    global MT5_CONFIG
    # Import here so the module is available after Section 2 sets up Path etc.
    try:
        from core.mt5_config import load_credentials
    except Exception as exc:
        log.warning("Could not import core.mt5_config: %s", exc)
        return
    creds = load_credentials()
    if not creds:
        return
    if creds.get("login"):
        MT5_CONFIG["login"]    = int(creds["login"])
    if creds.get("password"):
        MT5_CONFIG["password"] = creds["password"]
    if creds.get("server"):
        MT5_CONFIG["server"]   = creds["server"]
    if creds.get("path"):
        MT5_CONFIG["path"]     = creds["path"]
    log.info("MT5 credentials loaded from mt5_credentials.json (login=%s)",
             MT5_CONFIG.get("login"))

def _save_credentials(login, password, server, path) -> None:
    """Persist MT5 credentials to disk (encrypted) via core/mt5_config.py."""
    try:
        from core.mt5_config import save_credentials
        save_credentials(login, password or "", server or "", path or "")
    except Exception as exc:
        log.warning("Could not save credentials via core.mt5_config: %s", exc)

# ============================================================ #
# SECTION 1 — TRADING RULES (embedded YAML)                    #
# ============================================================ #

_RULES_YAML = """
# allowed_symbols: empty list = NO restriction (any symbol is tradable).
# The agent filters per signal by daily bias — no code-level pair whitelist.
allowed_symbols: []

sessions:
  always: [0, 24]

risk:
  max_risk_per_trade_pct:   1.0
  # Per-symbol cap on simultaneous open positions. 0 = UNLIMITED.
  # Default 5 is a soft brake so a misfiring strategy can't stack 100 trades
  # on one symbol before the broker's margin check catches up.
  max_open_positions:       5
  # Account-wide cap on simultaneous open positions across all symbols.
  # 0 = UNLIMITED. Provides a global ceiling on total exposure.
  max_open_positions_total: 20
  # Circuit breaker: halt new entries for the rest of today (UTC) once
  # realised P&L for the day drops below -N% of the start-of-day balance.
  # 0 = DISABLED. Existing positions keep being managed.
  max_daily_loss_pct:       3.0
  max_spread_points:        30

entry:
  min_confluence_signals: 1
  allow_counter_trend:    false
  allow_pyramiding:       true   # no direction/profit guard; agent bias is the sole filter

exit:
  trailing_stop_fraction:    0.50
  breakeven_trigger_points:  10   # move SL to entry once 10 pts (MT5 points) in profit
  trailing_stop_points:       0   # 0 = disabled — remaining position holds to TP limit order
                                   # Set to N (pips) to trail N pips behind current price

  partial_take_profit:
    enabled:          true
    trigger_fraction: 0.50
    close_fraction:   0.50
    sl_plus_points:    5

  conditions:
    - "Trail SL at 50% of current profit pips (locks in half gains, lets winners run)."
    - "Partial close at 50% of halfway to TP (secure 25% of full TP profit early)."
    - "Let price extend beyond TP if momentum continues; no breakeven hardlock."

ai:
  extra_instructions: |
    AI_Pro AI risk notes:
    1. CHoCH setups (ENV1/ENV2) have structural backing — trail generously.
    2. Continuation setups (ENV3/ENV4) — tighten faster on structure breaks.
    3. Move SL to BE once 10pts in profit.
    4. Trail 8pts behind price while in profit.
    5. Close 50% at halfway to TP, lock SL above entry after partial.
    6. Every 3 ticks Qwen reviews and may trail harder, tighten, or close.
"""

# ============================================================ #
# SECTION 2 — STANDARD IMPORTS                                 #
# ============================================================ #

import json
import logging
import math
import os
import re
import socket
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# Optional local Ollama agent backend (replaces DeepSeek-R1 when
# AI_BACKEND=agent/ollama). Import is tolerant so the module still loads
# if ai_agent.py / requests is missing.
try:
    import ai_agent as _ai_agent      # noqa: F401
    _AGENT_IMPORT_OK = True
except Exception as _agent_exc:       # pragma: no cover
    _ai_agent = None
    _AGENT_IMPORT_OK = False
    _agent_import_err = _agent_exc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("ai_pro")

_MEMORY_PATH = Path(__file__).resolve().parent / "ai_pro_trade_log.json"
_CREDS_PATH  = Path(__file__).resolve().parent / "mt5_credentials.json"

# ============================================================ #
# SECTION 3 — LOCAL LLM (DeepSeek-R1-Distill-Qwen-1.5B)        #
# ============================================================ #

class LocalLLM:
    """
    Singleton HuggingFace wrapper.  Loads lazily on first generate() call.
    Override model with AI_MODEL env variable.
    """
    _tokenizer  = None
    _model      = None
    _model_name: Optional[str] = None
    _device:     Optional[str] = None

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or os.getenv(
            "AI_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        )

    def _ensure_loaded(self) -> None:
        if (self.__class__._model is not None
                and self.__class__._model_name == self.model_name):
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        log.info("Loading LLM %s ...", self.model_name)
        cuda = torch.cuda.is_available()
        device = "cuda" if cuda else "cpu"
        dtype  = torch.float16 if cuda else torch.float32

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model     = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=dtype
        ).to(device)
        model.eval()

        self.__class__._tokenizer  = tokenizer
        self.__class__._model      = model
        self.__class__._model_name = self.model_name
        self.__class__._device     = device
        log.info("LLM loaded on %s", device)

    def generate(self, prompt: str,
                 max_new_tokens: int = 250,
                 temperature: float  = 0.0) -> str:
        self._ensure_loaded()
        import torch
        tokenizer = self.__class__._tokenizer
        model     = self.__class__._model
        device    = self.__class__._device

        conversation = [
            {
                "role":    "system",
                "content": (
                    "You are a professional forex trader. Your only objectives are "
                    "to MAXIMISE winning trades and MINIMISE losing positions. "
                    "You ALWAYS filter trades through higher-timeframe bias (D1, H4, "
                    "H1): never take a setup that fights the dominant HTF direction, "
                    "and prefer setups where every higher timeframe agrees with the "
                    "trade side. Be strict — when in doubt, reject the trade. Keep "
                    "any internal reasoning under 50 words. Your final output MUST "
                    "be a single JSON object on its own line after </think>. No "
                    "markdown, no backticks, no prose."
                )
            },
            {"role": "user", "content": prompt},
        ]
        encoded = tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids      = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][input_ids.shape[-1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ============================================================ #
# SECTION 4 — THOUGHT LOGGER                                   #
# ============================================================ #

_thoughts_lock: threading.Lock = threading.Lock()
_thoughts: deque                = deque(maxlen=120)

# ── Cross-process thought bridge ──────────────────────────────────────────
# The dashboard server (core/server.py) runs in a separate process from
# this bot. A shared JSONL file lets both processes see the same stream of
# thoughts. ai_pro.py appends; server.py tails.
import json as _json
import os as _os

_THOUGHT_LOG_PATH = _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)),
    "ai_thoughts.jsonl",
)
# Cap the file size so it doesn't grow unbounded — when it exceeds this we
# keep only the most recent entries on disk.
_THOUGHT_LOG_MAX_LINES = 2000
_thought_file_lock: threading.Lock = threading.Lock()


def _append_thought_to_file(entry: dict) -> None:
    """Append a thought entry to the shared JSONL file. Best-effort; never raises."""
    try:
        line = _json.dumps(entry, ensure_ascii=False, default=str)
        with _thought_file_lock:
            with open(_THOUGHT_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            # Occasional size-cap — rewrite with only the tail when too big
            try:
                if _os.path.getsize(_THOUGHT_LOG_PATH) > 2_000_000:  # ~2MB
                    with open(_THOUGHT_LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
                        lines = f.readlines()
                    if len(lines) > _THOUGHT_LOG_MAX_LINES:
                        lines = lines[-_THOUGHT_LOG_MAX_LINES:]
                        with open(_THOUGHT_LOG_PATH, "w", encoding="utf-8") as f:
                            f.writelines(lines)
            except Exception:
                pass
    except Exception:
        pass


def log_thought(
    source: str,
    symbol: str,
    stage: str,
    summary: str,
    detail: Optional[str]     = None,
    action: Optional[str]     = None,
    confidence: Optional[float] = None,
) -> None:
    try:
        entry = {
            "ts":         datetime.now(timezone.utc).isoformat(),
            "source":     source,
            "symbol":     symbol,
            "stage":      stage,
            "summary":    summary,
            "detail":     detail     or "",
            "action":     action     or "",
            "confidence": round(confidence, 2) if confidence is not None else None,
        }
        with _thoughts_lock:
            _thoughts.append(entry)
        # Also write to the shared file so the dashboard server process can
        # see bot thoughts. (The in-memory deque is only used by this process.)
        _append_thought_to_file(entry)
    except Exception:
        pass


def get_thoughts(since_ts: Optional[str] = None, limit: int = 60) -> list:
    with _thoughts_lock:
        items = list(_thoughts)
    if since_ts:
        try:
            # Only include thoughts with timestamp >= since_ts
            items = [t for t in items if str(t.get("ts", "")) >= since_ts]
        except Exception:
            # If comparison fails, return all items
            pass
    return items[-limit:]


def clear_thoughts() -> None:
    with _thoughts_lock:
        _thoughts.clear()


# ============================================================ #
# SECTION 5 — TRADING RULES                                    #
# ============================================================ #

class TradingRules:
    def __init__(self, raw: dict) -> None:
        self._raw = raw
        self.allowed_symbols: list = [
            s.upper() for s in (raw.get("allowed_symbols") or [])
        ]

        sessions_raw: dict = raw.get("sessions") or {}
        self.sessions: dict = {}
        for name, window in sessions_raw.items():
            if isinstance(window, (list, tuple)) and len(window) == 2:
                self.sessions[name] = (int(window[0]), int(window[1]))

        risk: dict = raw.get("risk") or {}
        self.max_risk_per_trade_pct: float         = float(risk.get("max_risk_per_trade_pct", 1.0))
        # 0 = unlimited for both caps (per-symbol and account-wide).
        self.max_open_positions: int               = int(risk.get("max_open_positions", 5))
        self.max_open_positions_total: int         = int(risk.get("max_open_positions_total", 20))
        # 0 = daily loss circuit breaker disabled.
        self.max_daily_loss_pct: float             = float(risk.get("max_daily_loss_pct", 0.0))
        self.max_spread_points: int                = int(risk.get("max_spread_points", 30))

        entry: dict = raw.get("entry") or {}
        self.min_confluence_signals: int = int(entry.get("min_confluence_signals", 1))
        self.allow_counter_trend: bool   = bool(entry.get("allow_counter_trend", False))
        self.allow_pyramiding: bool      = bool(entry.get("allow_pyramiding", False))

        exit_: dict = raw.get("exit") or {}
        self.breakeven_trigger_points: Optional[int]   = _to_int(exit_.get("breakeven_trigger_points"))
        self.trailing_stop_points: Optional[int]       = _to_int(exit_.get("trailing_stop_points"))
        self.exit_conditions: list = list(exit_.get("conditions") or [])

        partial: dict = exit_.get("partial_take_profit") or {}
        self.partial_tp_enabled: bool           = bool(partial.get("enabled", True))
        self.partial_tp_trigger_fraction: float = float(partial.get("trigger_fraction", 0.5))
        self.partial_tp_close_fraction: float   = float(partial.get("close_fraction", 0.5))
        self.partial_tp_sl_plus_points: int     = int(partial.get("sl_plus_points", 5))

        ai: dict = raw.get("ai") or {}
        self.extra_instructions: str = str(ai.get("extra_instructions", ""))

    def is_trading_session(self, utc_now: Optional[datetime] = None) -> bool:
        if not self.sessions:
            return True
        now  = utc_now or datetime.now(timezone.utc)
        hour = now.hour
        for start, end in self.sessions.values():
            if start < end:
                if start <= hour < end:
                    return True
            else:
                if hour >= start or hour < end:
                    return True
        return False

    def session_status(self, utc_now: Optional[datetime] = None) -> str:
        now    = utc_now or datetime.now(timezone.utc)
        active = self.is_trading_session(now)
        hour   = now.hour
        current = []
        for name, (s, e) in self.sessions.items():
            if s < e:
                if s <= hour < e:
                    current.append(name)
            else:
                if hour >= s or hour < e:
                    current.append(name)
        if active:
            return f"ACTIVE ({', '.join(current)}) — UTC {now.strftime('%H:%M')}"
        return f"CLOSED — UTC {now.strftime('%H:%M')}"


def _load_rules() -> TradingRules:
    import yaml
    raw = yaml.safe_load(_RULES_YAML) or {}
    return TradingRules(raw)


def _to_float(val: Any) -> Optional[float]:
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _to_int(val: Any) -> Optional[int]:
    try:
        return int(val) if val is not None else None
    except (TypeError, ValueError):
        return None


# ============================================================ #
# SECTION 6 — JSON HELPERS                                     #
# ============================================================ #

_THINK_RE   = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think>.*?$",       re.DOTALL | re.IGNORECASE)
_FENCE_RE   = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _clean_llm_text(text: str) -> str:
    """Strip reasoning-model artifacts: <think> blocks, markdown fences, role tags."""
    if not text:
        return ""
    # Drop closed <think>...</think> blocks
    text = _THINK_RE.sub("", text)
    # Drop unterminated <think>... (model ran out of tokens mid-thought)
    text = _THINK_OPEN.sub("", text)
    # Drop assistant/user role artifacts from chat templates
    text = re.sub(r"</?(assistant|user|system)>", "", text, flags=re.IGNORECASE)
    return text.strip()


def _try_json_loads(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None


def _scan_json_objects(text: str):
    """Yield every balanced {...} substring in text that parses as a JSON dict.

    Walks the text manually so we tolerate nested braces, quoted strings with
    escaped quotes, and prose around the JSON.
    """
    if not text:
        return
    search_from = 0
    n = len(text)
    while True:
        start = text.find("{", search_from)
        if start == -1:
            return
        depth, in_string, escape_next = 0, False, False
        end = -1
        for i in range(start, n):
            ch = text[i]
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end == -1:
            search_from = start + 1
            continue
        candidate = text[start:end + 1]
        obj = _try_json_loads(candidate)
        if isinstance(obj, dict):
            yield obj
        else:
            normalised = (candidate
                          .replace("“", '"').replace("”", '"')
                          .replace("‘", "'").replace("’", "'"))
            normalised = re.sub(r",(\s*[}\]])", r"\1", normalised)
            obj = _try_json_loads(normalised)
            if isinstance(obj, dict):
                yield obj
        search_from = end + 1


def _extract_json(text: str, want_keys: tuple = ()):
    """Extract a JSON object from an LLM response.

    Strategy (most-permissive wins):
      1. Search the RAW text first — DeepSeek-R1 sometimes emits the final
         JSON inside the closing portion of <think> when token budget is tight.
      2. If `want_keys` is supplied, prefer the first object that contains
         any of those keys (so we don't accidentally pick up a sub-object).
      3. Fall back to scanning the cleaned text (with <think> stripped).
      4. Try fenced code blocks explicitly.
    Returns None only if every candidate fails to parse.
    """
    if not text:
        return None

    def best(objs):
        objs = list(objs)
        if not objs:
            return None
        if want_keys:
            for o in objs:
                if any(k in o for k in want_keys):
                    return o
        return objs[0]

    # 1) raw text — catches JSON emitted before/inside an unterminated <think>
    obj = best(_scan_json_objects(text))
    if obj is not None:
        return obj

    # 2) cleaned text — strips <think> and role tags
    cleaned = _clean_llm_text(text)
    if cleaned and cleaned != text:
        # whole-blob fast path
        whole = _try_json_loads(cleaned)
        if isinstance(whole, dict):
            return whole
        obj = best(_scan_json_objects(cleaned))
        if obj is not None:
            return obj

    # 3) fenced code blocks
    for m in _FENCE_RE.finditer(text):
        inner = m.group(1).strip()
        cand = _try_json_loads(inner)
        if isinstance(cand, dict):
            return cand
        obj = best(_scan_json_objects(inner))
        if obj is not None:
            return obj

    return None


def _keyword_fallback_entry(text: str) -> Optional[dict]:
    """If JSON parsing fails, infer an approve/reject verdict from free text.

    Scans the FULL raw response (including <think> reasoning) so we can still
    extract intent when the model never emitted JSON. Always returns a verdict
    (defaults to approve) — never returns None — so the AI gate is never silent.
    """
    if not text:
        # Empty model output → trust the strategy gate
        return {"approve": True,
                "reason": "Fallback parse: empty response, trusting strategy",
                "confidence": 0.5}

    t = text.lower()

    # Strong reject signals
    reject_terms = (
        "reject", "do not take", "don't take", "do not approve",
        "do not enter", "don't enter", "skip this", "avoid this",
        "not approve", "not approved", "disapprove", "decline",
        "false signal", "weak setup", "invalid setup", "wait for",
        "no trade", "stand aside", '"approve": false', '"approve":false',
        "approve: false",
    )
    reject_score = sum(1 for w in reject_terms if w in t)

    # Strong approve signals
    approve_terms = (
        "approve", "approved", "take the trade", "take this trade",
        "take the setup", "take the entry", "valid setup", "valid entry",
        "go long", "go short", "enter long", "enter short", "good setup",
        "high probability", "accept", "execute the trade",
        '"approve": true', '"approve":true', "approve: true",
    )
    approve_score = sum(1 for w in approve_terms if w in t)

    # Pull a confidence-like number if present (look for "0.7" or "70%")
    conf = 0.6
    m = re.search(r"confidence[^0-9]{0,12}(0?\.\d+|\d{1,3})\s*%?", t)
    if m:
        try:
            v = float(m.group(1))
            conf = v / 100.0 if v > 1.0 else v
            conf = max(0.0, min(1.0, conf))
        except ValueError:
            pass

    if reject_score > approve_score:
        return {"approve": False,
                "reason": f"Fallback parse: model leaned reject (score {reject_score} vs {approve_score})",
                "confidence": conf}
    if approve_score > reject_score:
        return {"approve": True,
                "reason": f"Fallback parse: model leaned approve (score {approve_score} vs {reject_score})",
                "confidence": conf}
    # Tie or no clear signal — defer to the strategy gate (which already
    # passed its own confidence threshold to even reach the AI reviewer)
    return {"approve": True,
            "reason": "Fallback parse: no clear verdict, deferring to strategy",
            "confidence": 0.5}


def _keyword_fallback_risk(text: str) -> Optional[dict]:
    """Fallback parser for risk-manager responses (hold/trail/tighten/close).

    Scans the full raw response (reasoning included) for action keywords.
    Always returns a verdict (defaults to hold) so the manager is never silent.
    """
    if not text:
        return {"action": "hold", "new_sl": None,
                "reason": "Fallback parse: empty response, holding"}
    t = text.lower()
    # Priority order: close > tighten > trail > hold
    if any(w in t for w in ("recommend close", "close the position",
                             "close position", "exit now", "action: close",
                             "should close", "close it", "exit the trade",
                             '"action": "close"', '"action":"close"',
                             '"close"')):
        return {"action": "close", "new_sl": None,
                "reason": "Fallback parse: close"}
    if any(w in t for w in ("tighten", "action: tighten",
                             '"action": "tighten"', '"action":"tighten"',
                             '"tighten"')):
        return {"action": "tighten", "new_sl": None,
                "reason": "Fallback parse: tighten"}
    if any(w in t for w in ("trail", "trailing stop", "action: trail",
                             '"action": "trail"', '"action":"trail"',
                             '"trail"', "lock gains", "lock in gains")):
        return {"action": "trail", "new_sl": None,
                "reason": "Fallback parse: trail"}
    # Default to hold — safest action
    return {"action": "hold", "new_sl": None,
            "reason": "Fallback parse: defaulting to hold"}


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y", "1", "approve", "approved"}:
            return True
        if normalized in {"false", "no", "n", "0", "reject", "rejected"}:
            return False
    return default


def _normalize_confidence(value: Any, default: float = 0.0) -> float:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return default
    if conf > 1.0:
        conf /= 100.0
    return max(0.0, min(conf, 1.0))


# ============================================================ #
# SECTION 7 — TRADE MEMORY                                     #
# ============================================================ #

_memory_lock = threading.Lock()


def _load_memory() -> list:
    try:
        with _memory_lock:
            if _MEMORY_PATH.exists():
                return json.loads(_MEMORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return []


def _save_memory(records: list) -> None:
    with _memory_lock:
        _MEMORY_PATH.write_text(
            json.dumps(records, indent=2), encoding="utf-8"
        )


def _todays_pnl(records: list) -> float:
    today = datetime.now(timezone.utc).date().isoformat()
    return sum(r.get("pnl", 0) for r in records
               if str(r.get("ts", "")).startswith(today))


def _memory_summary(records: list) -> str:
    if not records:
        return "No history."
    recent    = records[-10:]
    wins      = sum(1 for r in recent if r.get("outcome") == "WIN")
    losses    = sum(1 for r in recent if r.get("outcome") == "LOSS")
    total_pnl = sum(r.get("pnl", 0) for r in recent)
    return f"{len(recent)} trades: {wins}W/{losses}L  pnl={total_pnl:+.4f}"


def record_outcome(symbol: str, direction: str, source: str,
                   entry: float, exit_price: float,
                   volume: float, pnl: float,
                   ai_reason: str = "") -> None:
    records = _load_memory()
    records.append({
        "ts":        datetime.now(timezone.utc).isoformat(),
        "symbol":    symbol,
        "direction": direction,
        "source":    source,
        "entry":     round(entry, 6),
        "exit":      round(exit_price, 6),
        "volume":    volume,
        "pnl":       round(pnl, 4),
        "outcome":   ("WIN" if pnl > 0 else "LOSS" if pnl < 0 else "BREAK_EVEN"),
        "ai_reason": ai_reason,
    })
    _save_memory(records[-100:])
    log_thought(
        "memory", symbol, "trade_closed",
        f"Closed {direction} [{source}]: "
        f"{'WIN' if pnl > 0 else 'LOSS' if pnl < 0 else 'BE'}  P&L {pnl:+.4f}",
        detail=ai_reason, action="close",
    )


# ============================================================ #
# SECTION 8 — AI_Pro CORE STRATEGY                         #
# (all AI Pro logic intact, AI calls woven in)              #
# ============================================================ #

class AI_Pro:
    """
    AI Pro — CHoCH + Daily Levels Strategy, AI-Enhanced Edition.

    Signal engine from the CHoCH + Daily Levels core.
    Each signal is reviewed by a local DeepSeek LLM before execution.
    Open positions are monitored by an AI risk manager every N ticks.
    All AI reasoning is persisted in the thought log (see /ai_thoughts).
    """

    CONFIDENCE_THRESHOLD = 0.40   # DeepSeek min confidence to approve a trade
    AI_REVIEW_TICKS      = 3      # DeepSeek reviews open positions every N ticks
    AI_REVIEW_MIN_SECS   = 180    # …and at least this many seconds since the last AI review per ticket (survives bot restarts)
    MONITOR_HEARTBEAT_SECS = 600  # how often each open ticket emits a "monitoring" heartbeat to ai_thoughts

    def __init__(
        self,
        atr_tolerance_multiplier: float = 1.5,
        lookback_candles: int           = 50,
        atr_period: int                 = 14,
        sl_atr_mult: float              = 2.5,
        tp_atr_mult: float              = 4.5,
        level_interaction_bars: int     = 10,
        partial_close_ratio: float      = 0.5,
        partial_close_rr: float         = 1.0,
        breakeven_buffer_pips: float    = 1.0,
        use_ai: bool                    = True,
    ) -> None:
        self._mt5_initialized = False

        # Strategy parameters
        self.atr_tolerance_multiplier = atr_tolerance_multiplier
        self.lookback_candles         = lookback_candles
        self.atr_period               = atr_period
        self.sl_atr_mult              = sl_atr_mult
        self.tp_atr_mult              = tp_atr_mult
        self.level_interaction_bars   = level_interaction_bars
        self.partial_close_ratio      = partial_close_ratio
        self.partial_close_rr         = partial_close_rr
        self.breakeven_buffer_pips    = breakeven_buffer_pips
        self.use_ai                   = use_ai

        self.previous_day_high: Optional[float] = None
        self.previous_day_low:  Optional[float] = None

        self._partial_closed_tickets: set  = set()
        self._filling_mode_cache: dict     = {}
        self._mt5_info_cache: dict         = {"connected": False, "error": "MT5 not initialized"}

        # AI components
        self._llm: Optional[LocalLLM]      = None
        self._ai_tick_counters: dict       = {}   # ticket -> int
        self._ai_breakeven_done: set       = set()
        self._ai_partial_done: set         = set()
        self._ai_peak_profit: dict         = {}   # ticket -> float pts
        self._ai_last_review_ts: dict      = {}   # ticket -> unix seconds (last AI risk review)
        self._ai_last_heartbeat_ts: dict   = {}   # ticket -> unix seconds (last "monitoring" log)

    # ------------------------------------------------------------------ #
    # LLM access                                                          #
    # ------------------------------------------------------------------ #

    def _get_llm(self) -> LocalLLM:
        if self._llm is None:
            self._llm = LocalLLM()
        return self._llm

    def mt5_snapshot(self) -> dict:
        return dict(self._mt5_info_cache)

    # ------------------------------------------------------------------ #
    # Filling mode detection                                              #
    # ------------------------------------------------------------------ #

    def _get_filling_mode(self, symbol: str) -> int:
        if symbol in self._filling_mode_cache:
            return self._filling_mode_cache[symbol]
        import MetaTrader5 as mt5
        tick = mt5.symbol_info_tick(symbol)
        info = mt5.symbol_info(symbol)
        if tick is None or info is None:
            return mt5.ORDER_FILLING_FOK

        pip        = 0.01 if info.digits in (2, 3) else 0.0001
        price      = tick.ask
        sl         = round(price - 20 * pip, info.digits)
        tp         = round(price + 40 * pip, info.digits)
        candidates = [
            (mt5.ORDER_FILLING_FOK,    "FOK"),
            (mt5.ORDER_FILLING_IOC,    "IOC"),
            (mt5.ORDER_FILLING_RETURN, "RETURN"),
            (mt5.ORDER_FILLING_BOC,    "BOC"),
        ]
        for mode, name in candidates:
            probe = {
                "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol,
                "volume": info.volume_min, "type": mt5.ORDER_TYPE_BUY,
                "price": price, "sl": sl, "tp": tp, "deviation": 20,
                "magic": 234000, "comment": "probe",
                "type_time": mt5.ORDER_TIME_GTC, "type_filling": mode,
            }
            result = mt5.order_check(probe)
            if result is not None and result.retcode == 0:
                log.info("[FILL MODE] %s: %s", symbol, name)
                self._filling_mode_cache[symbol] = mode
                return mode

        self._filling_mode_cache[symbol] = mt5.ORDER_FILLING_FOK
        return mt5.ORDER_FILLING_FOK

    # ------------------------------------------------------------------ #
    # Pair-specific SL multiplier                                         #
    # ------------------------------------------------------------------ #

    def get_sl_multiplier(self, symbol: str) -> float:
        """Return pair-specific SL multiplier (ATR multiplier).
        
        Pairs with higher volatility need wider SL:
        - EURUSD, GBPUSD, GBPJPY: 2.5×ATR (work well, lower volatility)
        - USDCAD, USDJPY: 3.5×ATR (higher volatility, tighter SL causes early exits)
        - AUDUSD: 2.5×ATR (default, standard volatility)
        """
        if symbol in ("USDCAD", "USDJPY"):
            return 3.5  # Higher volatility pairs need wider stops
        return 2.5     # Default for other pairs

    # ------------------------------------------------------------------ #
    # MT5 helpers                                                         #
    # ------------------------------------------------------------------ #

    def _ensure_mt5(self) -> bool:
        if not self._mt5_initialized:
            import MetaTrader5 as mt5
            if _mt5_initialize():
                self._mt5_initialized = True
                self._mt5_info_cache = _read_mt5_runtime_info(mt5)
            else:
                log.error("Failed to initialize MT5: %s", mt5.last_error())
        return self._mt5_initialized

    def _select_symbol(self, symbol: str) -> bool:
        import MetaTrader5 as mt5
        if not mt5.symbol_select(symbol, True):
            log.error("Could not select %s", symbol)
            return False
        return True

    # ------------------------------------------------------------------ #
    # ATR                                                                  #
    # ------------------------------------------------------------------ #

    def _calculate_atr(self, df: pd.DataFrame) -> float:
        high       = df["high"]
        low        = df["low"]
        prev_close = df["close"].shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_period).mean().iloc[-1]
        return float(atr) if not np.isnan(atr) else 0.0001

    def _close_position(self, position) -> dict:
        import MetaTrader5 as mt5
        symbol      = position.symbol
        ticket      = position.ticket
        volume      = position.volume
        is_buy      = position.type == mt5.ORDER_TYPE_BUY
        close_type  = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY
        tick        = mt5.symbol_info_tick(symbol)
        close_price = tick.bid if is_buy else tick.ask
        filling     = self._get_filling_mode(symbol)

        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       symbol,
            "volume":       volume,
            "type":         close_type,
            "position":     ticket,
            "price":        close_price,
            "deviation":    20,
            "magic":        234000,
            "comment":      "AI_Pro close",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": filling,
        }
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            record_outcome(
                symbol=symbol,
                direction="BUY" if is_buy else "SELL",
                source="strategy_exit",
                entry=float(position.price_open),
                exit_price=float(result.price),
                volume=volume,
                pnl=float(position.profit),
            )
            return {
                "success": True, "ticket": ticket,
                "message": f"Closed #{ticket} @ {result.price:.5f}"
            }
        comment = result.comment if result else "N/A"
        return {"success": False, "ticket": ticket,
                "message": f"Close failed: {comment}"}

    # ------------------------------------------------------------------ #
    # Partial close + breakeven trail (AI Pro native)                  #
    # ------------------------------------------------------------------ #

    def _modify_sl(self, position, new_sl: float) -> dict:
        import MetaTrader5 as mt5
        symbol_info = mt5.symbol_info(position.symbol)
        if symbol_info is None:
            return {"success": False, "message": "Symbol info unavailable"}
        new_sl  = round(new_sl, symbol_info.digits)
        request = {
            "action":   mt5.TRADE_ACTION_SLTP,
            "symbol":   position.symbol,
            "position": position.ticket,
            "sl":       new_sl,
            "tp":       position.tp,
        }
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            return {"success": True, "ticket": position.ticket,
                    "message": f"SL -> {new_sl:.5f} on #{position.ticket}"}
        comment = result.comment if result else "N/A"
        return {"success": False, "ticket": position.ticket,
                "message": f"SL modify failed: {comment}"}

    def _partial_close(self, position, close_ratio: float) -> dict:
        import MetaTrader5 as mt5
        symbol_info = mt5.symbol_info(position.symbol)
        if symbol_info is None:
            return {"success": False, "message": "Symbol info unavailable"}
        vol_step  = symbol_info.volume_step
        raw_vol   = position.volume * close_ratio
        close_vol = round(raw_vol - (raw_vol % vol_step), 10)
        close_vol = max(close_vol, vol_step)
        close_vol = min(close_vol, position.volume - vol_step)
        if close_vol <= 0:
            return {"success": False,
                    "message": "Position too small to split"}
        is_buy      = position.type == mt5.ORDER_TYPE_BUY
        close_type  = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY
        tick        = mt5.symbol_info_tick(position.symbol)
        close_price = tick.bid if is_buy else tick.ask
        filling     = self._get_filling_mode(position.symbol)
        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       position.symbol,
            "volume":       float(close_vol),
            "type":         close_type,
            "position":     position.ticket,
            "price":        close_price,
            "deviation":    20,
            "magic":        234000,
            "comment":      "AI_Pro partial",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": filling,
        }
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            return {
                "success": True, "ticket": position.ticket,
                "closed_vol": close_vol,
                "remain_vol": round(position.volume - close_vol, 10),
                "price": result.price,
                "message": (f"Partial {close_vol} of {position.volume}"
                            f" @ {result.price:.5f} #{position.ticket}")
            }
        comment = result.comment if result else "N/A"
        return {"success": False, "ticket": position.ticket,
                "message": f"Partial failed: {comment}"}

    def _was_partially_closed(self, ticket: int) -> bool:
        """Detect via MT5 deal history whether this position has had a partial close.

        Survives bot restarts and catches partials done outside of this process.
        A position is considered partialled if the sum of OUT deals on this ticket
        is > 0 and < the original opening IN deal volume.
        """
        import MetaTrader5 as mt5
        try:
            deals = mt5.history_deals_get(position=int(ticket))
        except Exception:
            return False
        if not deals:
            return False
        opened_vol = 0.0
        closed_vol = 0.0
        for d in deals:
            if d.entry == mt5.DEAL_ENTRY_IN:
                opened_vol += float(d.volume)
            elif d.entry in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_OUT_BY):
                closed_vol += float(d.volume)
        if opened_vol <= 0:
            return False
        # Partial: some volume closed, but not all
        return 0 < closed_vol < opened_vol

    def check_partial_close_and_breakeven(self, symbol: str) -> list:
        import MetaTrader5 as mt5
        if not self._ensure_mt5():
            return []
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return []
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return []

        # Correct pip size: JPY pairs have 2-3 decimal digits (1 pip = 0.01),
        # all others have 4-5 decimal digits (1 pip = 0.0001).
        pip    = 0.01 if symbol_info.digits in (2, 3) else 0.0001
        be_buf = self.breakeven_buffer_pips * pip
        results = []

        # Import MT5 history window — last 7 days is enough for live trades
        try:
            from datetime import datetime, timedelta
            _hist_from = datetime.now() - timedelta(days=7)
            _hist_to   = datetime.now() + timedelta(days=1)
            mt5.history_select(_hist_from, _hist_to)
        except Exception:
            pass

        for pos in positions:
            ticket = pos.ticket
            entry  = pos.price_open
            sl     = pos.sl
            is_buy = pos.type == mt5.ORDER_TYPE_BUY

            # Reconcile in-memory set with actual MT5 deal history
            if ticket not in self._partial_closed_tickets:
                if self._was_partially_closed(ticket):
                    self._partial_closed_tickets.add(ticket)
                    log_thought("partial_close", symbol, "detected_external",
                                f"#{ticket} partial detected via history — will move SL to BE",
                                action="reconcile")

            if ticket in self._partial_closed_tickets:
                be_price = round(
                    entry + be_buf if is_buy else entry - be_buf,
                    symbol_info.digits
                )
                sl_at_be = (sl >= be_price) if is_buy else (sl <= be_price)
                if not sl_at_be:
                    log_thought("partial_close", symbol, "breakeven",
                                f"Moving SL to BE {be_price:.5f} on #{ticket} "
                                f"(partial already taken)",
                                action="move_sl")
                    results.append(self._modify_sl(pos, be_price))
                continue

            if sl == 0.0:
                continue

            risk_dist = abs(entry - sl)
            trigger   = (entry + risk_dist * self.partial_close_rr if is_buy
                         else entry - risk_dist * self.partial_close_rr)
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                continue
            current   = tick.bid if is_buy else tick.ask
            triggered = (current >= trigger) if is_buy else (current <= trigger)
            if not triggered:
                continue

            log_thought("partial_close", symbol, "trigger",
                        f"1:1 hit #{ticket} — closing {self.partial_close_ratio*100:.0f}%",
                        detail=f"entry={entry:.5f} trigger={trigger:.5f} current={current:.5f}",
                        action="partial_close")

            pc = self._partial_close(pos, self.partial_close_ratio)
            results.append(pc)
            if not pc["success"]:
                continue

            self._partial_closed_tickets.add(ticket)

            updated    = mt5.positions_get(ticket=ticket)
            target_pos = updated[0] if updated else pos
            be_price   = round(
                entry + be_buf if is_buy else entry - be_buf,
                symbol_info.digits
            )
            log_thought("partial_close", symbol, "breakeven",
                        f"Moving SL to BE {be_price:.5f} on #{ticket}",
                        action="move_sl")
            be = self._modify_sl(target_pos, be_price)
            results.append(be)

        return results

    # ------------------------------------------------------------------ #
    # AI Risk Manager (algo-v2 style, adapted for AI Pro positions)       #
    # ------------------------------------------------------------------ #

    def _ai_risk_review_position(self, pos, symbol: str,
                                  point: float, digits: int,
                                  atr: float) -> Optional[dict]:
        """
        Reviews one open position.  When AI_BACKEND=agent/ollama this is
        delegated to the Ollama agent in ai_agent.py, which can only
        TIGHTEN the stop or request an EARLY close — the broker's hard SL
        remains authoritative.  On any agent failure we fall back to the
        DeepSeek path below.
        """
        import MetaTrader5 as mt5
        ticket     = int(pos.ticket)
        is_buy     = pos.type == mt5.ORDER_TYPE_BUY
        entry      = float(pos.price_open)
        cur_sl     = float(pos.sl or 0)
        cur_tp     = float(pos.tp or 0)
        direction  = "BUY" if is_buy else "SELL"
        tick       = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        cur_price  = float(tick.bid) if is_buy else float(tick.ask)
        profit_pts = ((cur_price - entry) / point if is_buy
                      else (entry - cur_price) / point)

        self._ai_peak_profit.setdefault(ticket, 0)
        self._ai_peak_profit[ticket] = max(
            self._ai_peak_profit[ticket], profit_pts
        )
        peak = self._ai_peak_profit[ticket]

        # Fetch M15 data for structure analysis
        df = self._fetch_m15(symbol)
        hh_ll_status = {"trend_intact": True, "reason": "N/A"}
        structure_status = {"structure_broken": False, "reason": "N/A"}

        if df is not None:
            hh_ll_status = self._detect_hh_ll_trend(df, "buy" if is_buy else "sell")
            structure_status = self._detect_structure_break(df, entry, "buy" if is_buy else "sell", atr)
            fresh_struct = self._detect_fresh_structure(df, "buy" if is_buy else "sell")
        else:
            fresh_struct = {"fresh_structure": False, "structure_type": None}

        # ---------- Single-path agent risk review ----------
        # The Ollama agent is the sole risk manager. It can only TIGHTEN a
        # stop or request an EARLY close — never loosen, never override the
        # broker's hard SL. On any agent failure we HOLD (broker hard SL
        # remains authoritative), we never guess with a fallback LLM.
        if not (_AGENT_IMPORT_OK and _ai_agent is not None):
            log_thought("ai_risk", symbol, "error",
                        f"#{ticket} agent module unavailable — holding "
                        f"(broker SL still active).",
                        action="hold")
            return None

        try:
            log_thought(
                "ai_risk", symbol, "review_start",
                f"Asking Ollama agent about #{ticket} {direction} "
                f"({profit_pts:+.0f}pts, peak {peak:.0f}pts)",
                detail=(f"entry={entry:.5f} price={cur_price:.5f} "
                        f"SL={cur_sl:.5f} TP={cur_tp:.5f}"),
            )
            ctx = _ai_agent.PositionContext(
                symbol=symbol, ticket=ticket, side=direction,
                entry=entry, cur_price=cur_price, cur_sl=cur_sl,
                cur_tp=cur_tp, profit_pts=profit_pts, peak_pts=peak,
                atr=atr, digits=digits,
                trend_intact=hh_ll_status.get("trend_intact"),
                structure_broken=structure_status.get("structure_broken"),
                fresh_structure=fresh_struct.get("fresh_structure"),
                notes=(f"trend_reason={hh_ll_status.get('reason','')} "
                       f"struct_reason={structure_status.get('reason','')}"),
            )
            verdict = _ai_agent.get_risk_agent().review(ctx)
            action  = str(verdict.get("action", "hold"))
            reason  = str(verdict.get("reason", ""))[:120]

            if action == "close":
                log_thought(
                    "ai_risk", symbol, "close",
                    f"#{ticket} — agent says CLOSE: {reason[:80]}",
                    action="close",
                )
                return {"action": "close", "ticket": ticket,
                        "reason": reason}

            if action == "move_sl":
                new_sl = verdict.get("new_sl")
                log_thought(
                    "ai_risk", symbol, "move_sl",
                    f"#{ticket} {direction} — agent tighten: "
                    f"SL -> {new_sl} | {reason[:60]}",
                    action="move_sl",
                )
                return {"action": "move_sl", "ticket": ticket,
                        "new_sl": new_sl, "reason": reason}

            # hold (hard broker SL still fires — agent has no override)
            log_thought(
                "ai_risk", symbol, "hold",
                f"#{ticket} agent holding — broker SL still active. {reason[:80]}",
                action="hold",
            )
            return None
        except Exception as exc:
            log_thought(
                "ai_risk", symbol, "error",
                f"#{ticket} agent error: {exc} — holding "
                f"(broker SL still active).",
                action="hold",
            )
            return None

    def run_ai_risk_manager(self, symbol: str) -> list:
        """
        Called each cycle BEFORE entry signal generation.
        Runs rule-based checks (BE, trail) AND DeepSeek review every N ticks.
        """
        import MetaTrader5 as mt5
        if not self._ensure_mt5():
            return []
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return []

        info = mt5.symbol_info(symbol)
        if info is None:
            return []

        point  = float(getattr(info, "point",  0.0) or 1e-5)
        digits = int(getattr(info, "digits",   5))
        df     = self._fetch_m15(symbol)
        atr    = self._calculate_atr(df) if df is not None else 0.0

        try:
            rules = _load_rules()
        except Exception:
            rules = None

        results = []

        for pos in positions:
            ticket = int(pos.ticket)
            is_buy    = pos.type == mt5.ORDER_TYPE_BUY
            direction = "BUY" if is_buy else "SELL"
            entry     = float(pos.price_open)
            cur_sl = float(pos.sl or 0)
            cur_tp = float(pos.tp or 0)
            tick_  = mt5.symbol_info_tick(symbol)
            if tick_ is None:
                continue
            price  = float(tick_.bid) if is_buy else float(tick_.ask)
            profit_pts = ((price - entry) / point if is_buy
                          else (entry - price) / point)

            # Correct pip size (must match check_partial_close_and_breakeven)
            pip_size = 0.01 if digits in (2, 3) else 0.0001

            # -- Rule-based: breakeven
            # Fires once when profit_pts >= breakeven_trigger_points.
            # Uses the same breakeven_buffer_pips as the partial-close path so
            # the SL always lands at a consistent level (entry ± 1 pip by default).
            if ticket not in self._ai_breakeven_done and rules:
                be_pts = int(getattr(rules, "breakeven_trigger_points", 0) or 0)
                if be_pts and profit_pts >= be_pts:
                    be_buf = self.breakeven_buffer_pips * pip_size
                    be_sl  = round(
                        entry + be_buf if is_buy else entry - be_buf,
                        digits
                    )
                    if _sl_is_better(be_sl, cur_sl, is_buy):
                        self._ai_breakeven_done.add(ticket)
                        log_thought("rule_risk", symbol, "breakeven",
                                    f"#{ticket} — SL to BE {be_sl:.5f} "
                                    f"(profit={profit_pts:.1f}pts >= trigger={be_pts}pts)",
                                    action="move_sl")
                        updated = mt5.positions_get(ticket=ticket)
                        p2 = updated[0] if updated else pos
                        results.append(self._modify_sl(p2, be_sl))

            # -- Rule-based: trailing stop
            # Uses pip units (not raw MT5 points) so "8 trailing" means 8 pips.
            # Only fires if trailing_stop_points > 0 in config.
            # Does NOT fire if disabled (0) — the TP limit order handles final exit.
            if rules:
                trail_pts = int(getattr(rules, "trailing_stop_points", 0) or 0)
                if trail_pts and profit_pts > 0:
                    trail_sl = round(
                        price - trail_pts * pip_size if is_buy
                        else price + trail_pts * pip_size,
                        digits
                    )
                    if _sl_is_better(trail_sl, cur_sl, is_buy):
                        log_thought("rule_risk", symbol, "trail",
                                    f"#{ticket} — trail SL {trail_sl:.5f} "
                                    f"({trail_pts} pips behind {price:.5f})",
                                    action="move_sl")
                        updated = mt5.positions_get(ticket=ticket)
                        p2 = updated[0] if updated else pos
                        results.append(self._modify_sl(p2, trail_sl))

            # -- Heartbeat: prove to the user that risk monitoring is alive.
            # One log line per ticket every MONITOR_HEARTBEAT_SECS so the AI
            # Log shows continuous oversight instead of going silent between
            # reviews. Keeps the message lightweight to avoid log spam.
            now_ts = time.time()
            last_hb = self._ai_last_heartbeat_ts.get(ticket, 0.0)
            if now_ts - last_hb >= self.MONITOR_HEARTBEAT_SECS:
                self._ai_last_heartbeat_ts[ticket] = now_ts
                log_thought(
                    "rule_risk", symbol, "monitor",
                    f"#{ticket} {direction} monitoring — "
                    f"{profit_pts:+.1f}pts (peak {self._ai_peak_profit.get(ticket, 0):.0f}), "
                    f"SL={cur_sl:.5f} TP={cur_tp:.5f}",
                )

            # -- AI review: tick-counter OR time-based, whichever fires first.
            # Time-based gate ensures reviews still happen across bot restarts
            # (the in-memory tick counter resets to 0 every time the bot boots,
            # so prior versions could go forever without reviewing on a flaky
            # process).
            if self.use_ai:
                cnt = self._ai_tick_counters.get(ticket, 0) + 1
                self._ai_tick_counters[ticket] = cnt
                last_review = self._ai_last_review_ts.get(ticket, 0.0)
                tick_due = cnt >= self.AI_REVIEW_TICKS
                time_due = (now_ts - last_review) >= self.AI_REVIEW_MIN_SECS
                if tick_due or (time_due and last_review > 0):
                    self._ai_tick_counters[ticket] = 0
                    self._ai_last_review_ts[ticket] = now_ts
                    updated = mt5.positions_get(ticket=ticket)
                    p2 = updated[0] if updated else pos
                    action = self._ai_risk_review_position(
                        p2, symbol, point, digits, atr
                    )
                    if action:
                        if action["action"] == "close":
                            r = self._close_position(p2)
                            results.append(r)
                        elif action["action"] == "move_sl":
                            updated2 = mt5.positions_get(ticket=ticket)
                            p3 = updated2[0] if updated2 else p2
                            r = self._modify_sl(p3, action["new_sl"])
                            results.append(r)
                elif last_review == 0.0:
                    # First time seeing this ticket — seed the timer so the
                    # time-based gate has a baseline.
                    self._ai_last_review_ts[ticket] = now_ts

        # Clean up state for closed positions
        active = {int(p.ticket) for p in (mt5.positions_get(symbol=symbol) or [])}
        for t in list(self._ai_tick_counters):
            if t not in active:
                del self._ai_tick_counters[t]
                self._ai_breakeven_done.discard(t)
                self._ai_partial_done.discard(t)
                self._ai_peak_profit.pop(t, None)
                self._ai_last_review_ts.pop(t, None)
                self._ai_last_heartbeat_ts.pop(t, None)

        return results

    # ------------------------------------------------------------------ #
    # Swing points                                                         #
    # ------------------------------------------------------------------ #

    def _get_swing_points(self, df: pd.DataFrame):
        swing_highs, swing_lows = [], []
        for i in range(1, len(df) - 1):
            if (df.iloc[i]["high"] > df.iloc[i-1]["high"] and
                    df.iloc[i]["high"] > df.iloc[i+1]["high"]):
                swing_highs.append({
                    "index": i,
                    "price": df.iloc[i]["high"],
                    "time":  df.iloc[i]["time"]
                })
            if (df.iloc[i]["low"] < df.iloc[i-1]["low"] and
                    df.iloc[i]["low"] < df.iloc[i+1]["low"]):
                swing_lows.append({
                    "index": i,
                    "price": df.iloc[i]["low"],
                    "time":  df.iloc[i]["time"]
                })
        return swing_highs, swing_lows

    # ------------------------------------------------------------------ #
    # Level interaction                                                     #
    # ------------------------------------------------------------------ #

    def _level_interacted(self, df: pd.DataFrame, level: float,
                           atr: float, bar_multiplier: int = 1) -> bool:
        zone    = atr * self.atr_tolerance_multiplier
        n_bars  = self.level_interaction_bars * bar_multiplier
        recent  = df.tail(n_bars)
        touched = ((recent["low"]  <= level + zone) &
                   (recent["high"] >= level - zone))
        return bool(touched.any())

    def _detect_hh_ll_trend(self, df: pd.DataFrame, direction: str) -> dict:
        """
        Detects Higher Highs/Higher Lows (BUY trend) or Lower Lows/Lower Highs (SELL trend).
        Returns trend strength and whether trend is intact.
        """
        if len(df) < 5:
            return {"trend_intact": True, "hh_ll_count": 0, "reason": "insufficient candles"}
        
        highs = df["high"].tail(5).values
        lows = df["low"].tail(5).values
        
        hh_count, ll_count = 0, 0
        lh_count, hl_count = 0, 0
        
        # Count HH/LL patterns in last 5 candles
        for i in range(1, len(highs)):
            if highs[i] > highs[i-1]:
                hh_count += 1
            if highs[i] < highs[i-1]:
                lh_count += 1
            if lows[i] < lows[i-1]:
                ll_count += 1
            if lows[i] > lows[i-1]:
                hl_count += 1
        
        if direction == "buy":
            # BUY trend: want HH and HL (higher highs, higher lows)
            trend_intact = hh_count >= 2 and hl_count >= 2
            reason = f"HH:{hh_count} HL:{hl_count}"
        else:
            # SELL trend: want LL and LH (lower lows, lower highs)
            trend_intact = ll_count >= 2 and lh_count >= 2
            reason = f"LL:{ll_count} LH:{lh_count}"
        
        return {
            "trend_intact": trend_intact,
            "hh_ll_count": max(hh_count + hl_count, ll_count + lh_count),
            "reason": reason
        }

    def _detect_structure_break(self, df: pd.DataFrame, entry_price: float,
                                direction: str, atr: float) -> dict:
        """
        Detects if price has broken through entry/support structure.
        Returns whether structure is broken.
        """
        if len(df) < 3:
            return {"structure_broken": False, "reason": "insufficient candles"}
        
        recent_lows = df["low"].tail(3).min()
        recent_highs = df["high"].tail(3).max()
        current_price = df.iloc[-1]["close"]
        
        structure_threshold = atr * 0.5
        
        if direction == "buy":
            # Structure broken if price closes below entry by significant amount
            structure_broken = (entry_price - current_price) > structure_threshold
            reason = f"entry:{entry_price:.5f} price:{current_price:.5f} threshold:{structure_threshold:.5f}"
        else:
            # Structure broken if price closes above entry by significant amount
            structure_broken = (current_price - entry_price) > structure_threshold
            reason = f"entry:{entry_price:.5f} price:{current_price:.5f} threshold:{structure_threshold:.5f}"
        
        return {
            "structure_broken": structure_broken,
            "reason": reason
        }

    def _detect_fresh_structure(self, df: pd.DataFrame, direction: str) -> dict:
        """
        Detects if price just formed a NEW swing point on the current bar.
        For BUY: checks if current high > previous highest high in last 5 bars
        For SELL: checks if current low < previous lowest low in last 5 bars
        Returns: {"fresh_structure": bool, "structure_type": "HH"|"LL"|None}
        """
        if len(df) < 5:
            return {"fresh_structure": False, "structure_type": None}
        
        current_high = df.iloc[-1]["high"]
        current_low = df.iloc[-1]["low"]
        
        # Get 5-candle highs/lows (excluding current bar)
        prev_highs = df.iloc[-6:-1]["high"]
        prev_lows = df.iloc[-6:-1]["low"]
        
        if direction == "buy":
            # Fresh HH = current high > any of the previous 5 candles' highs
            prev_max_high = prev_highs.max() if len(prev_highs) > 0 else 0
            fresh_hh = current_high > prev_max_high
            # Also check if making higher low (HL)
            prev_max_low = prev_lows.max() if len(prev_lows) > 0 else 0
            fresh_hl = current_low > prev_max_low
            
            structure_type = "HH" if fresh_hh else ("HL" if fresh_hl else None)
            fresh_structure = fresh_hh or fresh_hl
            
        else:  # SELL
            # Fresh LL = current low < any of the previous 5 candles' lows
            prev_min_low = prev_lows.min() if len(prev_lows) > 0 else float('inf')
            fresh_ll = current_low < prev_min_low
            # Also check if making lower high (LH)
            prev_min_high = prev_highs.min() if len(prev_highs) > 0 else float('inf')
            fresh_lh = current_high < prev_min_high
            
            structure_type = "LL" if fresh_ll else ("LH" if fresh_lh else None)
            fresh_structure = fresh_ll or fresh_lh
        
        return {
            "fresh_structure": fresh_structure,
            "structure_type": structure_type
        }

    # ------------------------------------------------------------------ #
    # Data fetching                                                         #
    # ------------------------------------------------------------------ #

    def get_previous_day_levels(self, symbol: str) -> Optional[dict]:
        import MetaTrader5 as mt5
        if not self._ensure_mt5():
            return None
        try:
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, 2)
            if rates is None or len(rates) < 2:
                return None
            prev = rates[0]
            self.previous_day_high = prev["high"]
            self.previous_day_low  = prev["low"]
            return {
                "date":  datetime.fromtimestamp(prev["time"]).date(),
                "high":  self.previous_day_high,
                "low":   self.previous_day_low,
                "range": self.previous_day_high - self.previous_day_low,
            }
        except Exception as e:
            log.error("Daily levels error: %s", e)
            return None

    def _fetch_m15(self, symbol: str) -> Optional[pd.DataFrame]:
        import MetaTrader5 as mt5
        if not self._ensure_mt5():
            return None
        rates = mt5.copy_rates_from_pos(
            symbol, mt5.TIMEFRAME_M15, 0, self.lookback_candles
        )
        if rates is None or len(rates) < max(self.atr_period + 2, 10):
            return None
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df

    # ------------------------------------------------------------------ #
    # CHoCH detection (ENV 1 & 2)                                          #
    # ------------------------------------------------------------------ #

    def detect_choch_on_m15(self, symbol: str,
                              df: pd.DataFrame = None) -> Optional[dict]:
        if df is None:
            df = self._fetch_m15(symbol)
        if df is None:
            return None

        swing_highs, swing_lows = self._get_swing_points(df)
        current_price = df.iloc[-1]["close"]

        base = {
            "choch_detected": False, "type": None,
            "reason": "Insufficient swing points",
            "current_price": current_price,
            "swing_highs": swing_highs, "swing_lows": swing_lows
        }
        if len(swing_highs) < 3 or len(swing_lows) < 3:
            return base

        h = swing_highs[-3:]
        l = swing_lows[-3:]

        bullish = (l[1]["price"] < l[0]["price"] and
                   l[2]["price"] > l[1]["price"])
        bearish = (h[1]["price"] > h[0]["price"] and
                   h[2]["price"] < h[1]["price"])

        if bullish and bearish:
            choch_type = ("bullish"
                          if swing_lows[-1]["index"] > swing_highs[-1]["index"]
                          else "bearish")
        elif bullish:
            choch_type = "bullish"
        elif bearish:
            choch_type = "bearish"
        else:
            choch_type = None

        reason = {
            "bullish": (f"Bullish CHoCH: failed LL — latest low "
                        f"{l[2]['price']:.5f} > prev {l[1]['price']:.5f}"),
            "bearish": (f"Bearish CHoCH: failed HH — latest high "
                        f"{h[2]['price']:.5f} < prev {h[1]['price']:.5f}"),
            None:      "No CHoCH — structure continuing",
        }[choch_type]

        return {
            "choch_detected": choch_type is not None,
            "type": choch_type,
            "reason": reason,
            "current_price": current_price,
            "swing_highs": swing_highs,
            "swing_lows":  swing_lows,
            "latest_high": swing_highs[-1]["price"] if swing_highs else 0,
            "latest_low":  swing_lows[-1]["price"]  if swing_lows  else 0,
        }

    # ------------------------------------------------------------------ #
    # Trend continuation (ENV 3 & 4)                                       #
    # ------------------------------------------------------------------ #

    def detect_trend_continuation(self, symbol: str,
                                   df: pd.DataFrame = None) -> Optional[dict]:
        if df is None:
            df = self._fetch_m15(symbol)
        if df is None:
            return None

        swing_highs, swing_lows = self._get_swing_points(df)
        current_price = df.iloc[-1]["close"]

        base = {
            "continuation_detected": False, "type": None,
            "reason": "Insufficient swing points",
            "current_price": current_price,
            "swing_highs": swing_highs, "swing_lows": swing_lows,
            "trend_strength": 0
        }
        if len(swing_highs) < 3 or len(swing_lows) < 3:
            return base

        h = swing_highs[-3:]
        l = swing_lows[-3:]
        hh = sum(1 for i in range(1, 3) if h[i]["price"] > h[i-1]["price"])
        hl = sum(1 for i in range(1, 3) if l[i]["price"] > l[i-1]["price"])
        ll = sum(1 for i in range(1, 3) if l[i]["price"] < l[i-1]["price"])
        lh = sum(1 for i in range(1, 3) if h[i]["price"] < h[i-1]["price"])

        bullish = hh >= 1 and hl >= 1
        bearish = ll >= 1 and lh >= 1

        if bullish and bearish:
            cont_type = ("bullish"
                         if swing_highs[-1]["index"] > swing_lows[-1]["index"]
                         else "bearish")
        elif bullish:
            cont_type = "bullish"
        elif bearish:
            cont_type = "bearish"
        else:
            cont_type = None

        strength = {
            "bullish": min(hh, hl),
            "bearish": min(ll, lh),
            None: 0
        }[cont_type]

        reason = {
            "bullish": f"Bullish: HH({hh}/2) HL({hl}/2) — uptrend",
            "bearish": f"Bearish: LL({ll}/2) LH({lh}/2) — downtrend",
            None:      "No clear continuation",
        }[cont_type]

        return {
            "continuation_detected": cont_type is not None,
            "type": cont_type, "reason": reason,
            "current_price": current_price,
            "swing_highs": swing_highs, "swing_lows": swing_lows,
            "trend_strength": strength
        }

    # ------------------------------------------------------------------ #
    # AI Signal Review (DeepSeek approves/rejects AI Pro signal)           #
    # ------------------------------------------------------------------ #

    def _ai_review_signal(self, symbol: str, signal: dict,
                           df: pd.DataFrame) -> dict:
        """
        Single-path entry review via the Ollama agent (ai_agent.py).
        The agent filters each raw signal by DAILY BIAS and returns:
          {"approve": bool, "reason": str, "confidence": float}

        There is NO fallback LLM. If the agent module is missing or the
        agent raises, we return a safe REJECT — never a default approve.
        The broker-side hard SL remains authoritative regardless.
        """
        direction = signal.get("signal", "")
        env       = signal.get("signal_source", "unknown")

        # Hard guard — agent module must be importable.
        if not (_AGENT_IMPORT_OK and _ai_agent is not None):
            log_thought(
                "ai_entry", symbol, "error",
                "Agent module unavailable — rejecting signal (no fallback LLM).",
                action="hold", confidence=0.0,
            )
            return {"approve": False,
                    "reason": "agent module unavailable — safe reject",
                    "confidence": 0.0}

        try:
            log_thought(
                "ai_entry", symbol, "review_start",
                f"Sending {direction} [{env}] to Ollama agent for review",
                detail=str(signal.get("reason", ""))[:120],
                action=direction.lower() if direction else None,
            )
            verdict    = _ai_agent.get_entry_agent().review(symbol, signal, df)
            approve    = bool(verdict.get("approve", False))
            reason     = str(verdict.get("reason", "")).strip() or "no reason"
            ai_conf    = float(verdict.get("confidence", 0.0) or 0.0)
            htf_detail = str(verdict.get("htf_detail", "?"))
            log_thought(
                "ai_entry", symbol, "verdict",
                f"Ollama agent {'APPROVED' if approve else 'REJECTED'}: {reason[:80]}",
                detail=(f"{direction} [{env}] conf={ai_conf:.2f} | "
                        f"htf={htf_detail}"),
                action=direction.lower() if approve else "hold",
                confidence=ai_conf,
            )
            return {"approve": approve, "reason": reason, "confidence": ai_conf}
        except Exception as exc:
            log_thought(
                "ai_entry", symbol, "error",
                f"Agent error: {exc} — rejecting signal (no fallback LLM).",
                action="hold", confidence=0.0,
            )
            return {"approve": False,
                    "reason": f"agent error: {exc!s} — safe reject",
                    "confidence": 0.0}

    # ------------------------------------------------------------------ #
    # HTF bias detection                                                   #
    # ------------------------------------------------------------------ #

    def _get_daily_trend_bias(self, symbol: str) -> str:
        """Determine daily timeframe trend direction."""
        try:
            import MetaTrader5 as mt5
            # Get last daily candle
            daily_data = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, 2)
            if daily_data is None or len(daily_data) < 1:
                return "neutral"
            
            last_daily = daily_data[-1]
            daily_close = last_daily["close"]
            daily_open  = last_daily["open"]
            
            # Simple trend: close above open = bullish, below = bearish
            if daily_close > daily_open:
                return "bullish"
            elif daily_close < daily_open:
                return "bearish"
            else:
                return "neutral"
        except Exception:
            return "neutral"

    # ------------------------------------------------------------------ #
    # Main signal generator                                                #
    # ------------------------------------------------------------------ #

    def generate_trade_signal(self, symbol: str) -> dict:
        daily_levels = self.get_previous_day_levels(symbol)
        if daily_levels is None:
            return self._neutral(symbol, "Could not get daily levels")

        df = self._fetch_m15(symbol)
        if df is None:
            return self._neutral(symbol, "Could not fetch M15 data")

        daily_trend = self._get_daily_trend_bias(symbol)
        atr           = self._calculate_atr(df)
        current_price = df.iloc[-1]["close"]
        zone          = atr * self.atr_tolerance_multiplier

        near_high_choch = self._level_interacted(df, daily_levels["high"], atr, 2)
        near_low_choch  = self._level_interacted(df, daily_levels["low"],  atr, 2)
        near_high       = self._level_interacted(df, daily_levels["high"], atr)
        near_low        = self._level_interacted(df, daily_levels["low"],  atr)

        choch_data = self.detect_choch_on_m15(symbol, df)
        cont_data  = self.detect_trend_continuation(symbol, df)

        signal = {
            "symbol":            symbol,
            "signal":            "neutral",
            "signal_source":     None,
            "reason":            "",
            "confidence":        0,
            "entry_price":       current_price,
            "stop_loss":         None,
            "take_profit":       None,
            "previous_day_high": daily_levels["high"],
            "previous_day_low":  daily_levels["low"],
            "atr":               round(atr, 6),
            "dynamic_zone_pips": round(zone / 0.0001, 1),
            "ai_approved":       None,
            "ai_reason":         None,
            "ai_confidence":     None,
        }

        def buy_levels():
            sl_mult = self.get_sl_multiplier(symbol)
            sl = round(current_price - atr * sl_mult, 5)
            tp = round(current_price + atr * self.tp_atr_mult, 5)
            return sl, tp

        def sell_levels():
            sl_mult = self.get_sl_multiplier(symbol)
            sl = round(current_price + atr * sl_mult, 5)
            tp = round(current_price - atr * self.tp_atr_mult, 5)
            return sl, tp

        triggered = []

        # ENV 1: CHoCH BUY at PDL
        failed_low_at_pdl = (
            choch_data is not None
            and len(choch_data.get("swing_lows", [])) >= 2
            and abs(choch_data["swing_lows"][-2]["price"] - daily_levels["low"])
                <= atr * self.atr_tolerance_multiplier * 2.0
        )
        if (choch_data and choch_data["choch_detected"]
                and choch_data["type"] == "bullish"
                and near_low_choch and failed_low_at_pdl
                and daily_trend != "bearish"):
            sl, tp = buy_levels()
            triggered.append((1, {
                "signal": "BUY", "signal_source": "CHoCH-BUY@PDL",
                "confidence": 85,
                "reason": (f"ENV1 — CHoCH BUY: failed LL at PDL "
                           f"{daily_levels['low']:.5f} ±{signal['dynamic_zone_pips']}p. "
                           f"{choch_data['reason']}"),
                "stop_loss": sl, "take_profit": tp
            }))

        # ENV 2: CHoCH SELL at PDH
        failed_high_at_pdh = (
            choch_data is not None
            and len(choch_data.get("swing_highs", [])) >= 2
            and abs(choch_data["swing_highs"][-2]["price"] - daily_levels["high"])
                <= atr * self.atr_tolerance_multiplier * 2.0
        )
        if (choch_data and choch_data["choch_detected"]
                and choch_data["type"] == "bearish"
                and near_high_choch and failed_high_at_pdh
                and daily_trend != "bullish"):
            sl, tp = sell_levels()
            triggered.append((1, {
                "signal": "SELL", "signal_source": "CHoCH-SELL@PDH",
                "confidence": 85,
                "reason": (f"ENV2 — CHoCH SELL: failed HH at PDH "
                           f"{daily_levels['high']:.5f} ±{signal['dynamic_zone_pips']}p. "
                           f"{choch_data['reason']}"),
                "stop_loss": sl, "take_profit": tp
            }))

        # ENV 3: Continuation BUY — broke above PDH
        broke_above_pdh = (
            df["close"].tail(self.level_interaction_bars) > daily_levels["high"]
        ).any()
        if (cont_data and cont_data["continuation_detected"]
                and cont_data["type"] == "bullish"
                and broke_above_pdh and near_high
                and daily_trend != "bearish"):
            strength   = cont_data.get("trend_strength", 1)
            confidence = 65 + strength * 10
            sl, tp     = buy_levels()
            triggered.append((2, {
                "signal": "BUY", "signal_source": "Continuation-BUY@PDH",
                "confidence": confidence,
                "reason": (f"ENV3 — Continuation BUY: broke above PDH "
                           f"{daily_levels['high']:.5f}, retesting as support. "
                           f"Strength {strength}/2. {cont_data['reason']}"),
                "stop_loss": sl, "take_profit": tp
            }))

        # ENV 4: Continuation SELL — broke below PDL
        broke_below_pdl = (
            df["close"].tail(self.level_interaction_bars) < daily_levels["low"]
        ).any()
        if (cont_data and cont_data["continuation_detected"]
                and cont_data["type"] == "bearish"
                and broke_below_pdl and near_low
                and daily_trend != "bullish"):
            strength   = cont_data.get("trend_strength", 1)
            confidence = 65 + strength * 10
            sl, tp     = sell_levels()
            triggered.append((2, {
                "signal": "SELL", "signal_source": "Continuation-SELL@PDL",
                "confidence": confidence,
                "reason": (f"ENV4 — Continuation SELL: broke below PDL "
                           f"{daily_levels['low']:.5f}, retesting as resistance. "
                           f"Strength {strength}/2. {cont_data['reason']}"),
                "stop_loss": sl, "take_profit": tp
            }))

        if triggered:
            triggered.sort(key=lambda x: x[0])
            signal.update(triggered[0][1])

            # AI review gate
            if self.use_ai:
                review = self._ai_review_signal(symbol, signal, df)
                signal["ai_approved"]   = review["approve"]
                signal["ai_reason"]     = review["reason"]
                signal["ai_confidence"] = review["confidence"]
            else:
                signal["ai_approved"]   = True
                signal["ai_reason"]     = "AI disabled"
                signal["ai_confidence"] = 1.0
        else:
            env1_ready = bool(
                choch_data and choch_data["choch_detected"]
                and choch_data["type"] == "bullish"
                and near_low_choch and failed_low_at_pdl
            )
            env2_ready = bool(
                choch_data and choch_data["choch_detected"]
                and choch_data["type"] == "bearish"
                and near_high_choch and failed_high_at_pdh
            )
            env3_ready = bool(
                cont_data and cont_data["continuation_detected"]
                and cont_data["type"] == "bullish"
                and broke_above_pdh and near_high
            )
            env4_ready = bool(
                cont_data and cont_data["continuation_detected"]
                and cont_data["type"] == "bearish"
                and broke_below_pdl and near_low
            )
            signal["reason"] = self._environment_summary(
                env1_ready, env2_ready, env3_ready, env4_ready
            )

        # Filter: Only return signals with sufficient confidence
        # This reduces noise and focuses on high-quality setups
        MIN_CONFIDENCE_THRESHOLD = 75  # Require at least 75% confidence
        if signal["signal"] != "neutral" and signal["confidence"] < MIN_CONFIDENCE_THRESHOLD:
            return self._neutral(symbol, f"Low confidence ({signal['confidence']}% < {MIN_CONFIDENCE_THRESHOLD}%)")

        log_thought(
            "strategy", symbol, "signal",
            f"Signal: {signal['signal']} [{signal.get('signal_source','—')}] "
            f"conf={signal['confidence']}% "
            f"AI={'AI verified' if signal.get('ai_approved') else '✗' if signal.get('ai_approved') is False else '—'}",
            detail=signal["reason"],
            action=signal["signal"].lower(),
            confidence=signal["confidence"] / 100 if signal["confidence"] else None,
        )

        return signal

    def _neutral(self, symbol: str, reason: str) -> dict:
        return {
            "symbol": symbol, "signal": "neutral",
            "signal_source": None, "reason": reason,
            "confidence": 0, "entry_price": None,
            "stop_loss": None, "take_profit": None,
            "atr": None,
            "ai_approved": None, "ai_reason": None, "ai_confidence": None,
        }

    @staticmethod
    def _environment_summary(
        env1: bool = False,
        env2: bool = False,
        env3: bool = False,
        env4: bool = False,
    ) -> str:
        return " | ".join([
            f"ENV1 {'active' if env1 else 'inactive'}",
            f"ENV2 {'active' if env2 else 'inactive'}",
            f"ENV3 {'active' if env3 else 'inactive'}",
            f"ENV4 {'active' if env4 else 'inactive'}",
        ])

    # ------------------------------------------------------------------ #
    # Trade execution                                                       #
    # ------------------------------------------------------------------ #

    def execute_trade(self, symbol: str, signal: dict,
                       lot_size: float = 0.50) -> dict:
        import MetaTrader5 as mt5
        if not self._ensure_mt5():
            return {"success": False, "message": "MT5 not initialized"}
        if signal["signal"] == "neutral":
            return {"success": False, "message": "Neutral — no trade"}
        if not signal.get("stop_loss") or not signal.get("take_profit"):
            return {"success": False, "message": "SL/TP missing"}
        if not self._select_symbol(symbol):
            return {"success": False, "message": f"Cannot select {symbol}"}

        account = mt5.account_info()
        if account is not None and not bool(getattr(account, "trade_allowed", False)):
            return {
                "success": False,
                "message": "MT5 account trading is disabled",
                "retcode": None,
                "comment": None,
                "last_error": mt5.last_error(),
            }

        info = mt5.symbol_info(symbol)
        if info is None:
            return {"success": False, "message": f"Symbol {symbol} not found"}

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return {"success": False, "message": "No tick data"}

        digits    = info.digits
        is_buy    = signal["signal"] == "BUY"
        order_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
        price      = tick.ask if is_buy else tick.bid
        sl         = round(signal["stop_loss"],   digits)

        # No direction guard — raw signals approved by the agent's daily-bias
        # filter execute regardless of any existing position's direction. The
        # agent (not code) is the sole gate on whether a new trade opens.
        tp         = round(signal["take_profit"], digits)
        filling    = self._get_filling_mode(symbol)

        # Build comment string, limited to MT5 max 31 chars
        source_short = signal.get('signal_source', '?')[:15]
        direction_short = signal['signal'][:3]
        comment_str = f"AP_{source_short[:10]}_{direction_short}"[:31]
        
        # Market order (spot entry only)
        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       symbol,
            "volume":       float(lot_size),
            "type":         order_type,
            "price":        price,
            "sl":           sl,
            "tp":           tp,
            "deviation":    20,
            "magic":        234000,
            "comment":      comment_str,
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": filling,
        }
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            log_thought(
                "execution", symbol, "order_placed",
                f"Market order #{result.order} {signal['signal']} "
                f"@ {result.price:.5f}  SL={sl:.5f}  TP={tp:.5f}",
                action=signal["signal"].lower(),
                confidence=signal.get("ai_confidence"),
            )
            return {
                "success":       True,
                "message":       "Trade executed",
                "order_ticket":  result.order,
                "volume":        result.volume,
                "price":         result.price,
                "signal":        signal["signal"],
                "signal_source": signal.get("signal_source"),
                "stop_loss":     sl,
                "take_profit":   tp,
            }

        comment = result.comment if result else "N/A"
        retcode = result.retcode if result else "N/A"
        last_error = mt5.last_error()
        log_thought("execution", symbol, "order_failed",
                    f"Order FAILED: {comment} (retcode {retcode}, last_error {last_error})",
                    action="hold")
        return {
            "success": False,
            "message": f"Order failed: {comment} (retcode {retcode})",
            "retcode": retcode,
            "comment": comment,
            "last_error": last_error,
        }

    # ------------------------------------------------------------------ #
    # Runner                                                               #
    # ------------------------------------------------------------------ #

    def run_strategy(self, symbol: str, auto_trade: bool = False,
                     lot_size: float = 0.50) -> dict:
        """
        Workflow: STRATEGY → AGENT → MARKET

          STRATEGY (rule engine — untouched):
            1. Partial close + breakeven management on existing positions
            2. Generate fresh M15 signal via AI_Pro rules

          AGENT (Ollama — gates + manages):
            3. Risk review of open positions (tighten SL / close early only;
               broker's hard SL remains authoritative)
            4. Entry review — agent filters the signal by DAILY BIAS
               (daily open vs current price) and approves or rejects

          MARKET (broker):
            5. Auto-execute approved signals via MT5 (no hard-coded symbol
               whitelist — any configured symbol is tradable)
        """
        log.info("=" * 65)
        log.info("AI Pro — %s", symbol)
        log.info("=" * 65)

        df = self._fetch_m15(symbol)

        exit_results = []

        # Step 1 — AI Pro partial close / breakeven
        partial_results = self.check_partial_close_and_breakeven(symbol)

        # Step 2 — AI risk manager
        ai_risk_results = self.run_ai_risk_manager(symbol)

        # Step 3+4 — Signal + AI review
        signal = self.generate_trade_signal(symbol)

        log.info("Signal      : %s", signal["signal"])
        log.info("Environment : %s", signal.get("signal_source") or "—")
        log.info("Confidence  : %s%%", signal["confidence"])
        log.info("AI approved : %s", signal.get("ai_approved"))
        log.info("AI reason   : %s", signal.get("ai_reason"))
        log.info("Reason      : %s", signal["reason"])

        trade_result = None
        if signal["signal"] != "neutral":
            risk   = abs(signal["entry_price"] - signal["stop_loss"])
            reward = abs(signal["take_profit"] - signal["entry_price"])
            rr     = reward / risk if risk > 0 else 0
            log.info("Entry: %.5f  SL: %.5f  TP: %.5f  RR: 1:%.2f",
                     signal["entry_price"], signal["stop_loss"],
                     signal["take_profit"], rr)

            # Step 6 — Execute
            if (auto_trade
                    and signal.get("ai_approved", False)):
                log.info("AUTO TRADE — executing")
                trade_result = self.execute_trade(symbol, signal, lot_size)
                if trade_result["success"]:
                    log.info("OK ticket=%d @ %.5f",
                             trade_result["order_ticket"],
                             trade_result["price"])
                else:
                    log.error("FAILED: %s", trade_result["message"])
            elif auto_trade and not signal.get("ai_approved", True):
                log.info("AUTO TRADE skipped — AI rejected signal")

        if signal["signal"] != "neutral" and trade_result is None:
            if not auto_trade:
                trade_result = {
                    "success": False,
                    "message": "AUTO TRADE skipped: auto trading disabled or dry run enabled",
                }
            elif not signal.get("ai_approved", True):
                trade_result = {
                    "success": False,
                    "message": "AUTO TRADE skipped: AI rejected signal",
                }

        return {
            "signal":          signal,
            "trade_result":    trade_result,
            "exit_results":    exit_results,
            "partial_results": partial_results,
            "ai_risk_results": ai_risk_results,
        }

    def shutdown(self) -> None:
        if self._mt5_initialized:
            import MetaTrader5 as mt5
            mt5.shutdown()
            self._mt5_initialized = False
            self._mt5_info_cache = {"connected": False, "error": "MT5 shutdown"}


# ============================================================ #
# SECTION 9 — HARD RULE CHECKS                                 #
# ============================================================ #

def _sl_is_better(new_sl: float, cur_sl: float, is_long: bool) -> bool:
    if cur_sl == 0.0:
        return True
    return new_sl > cur_sl if is_long else new_sl < cur_sl


def _check_hard_rules(rules, symbol: str, positions,
                       tick, info, records: list) -> Optional[str]:
    if rules.allowed_symbols and symbol.upper() not in rules.allowed_symbols:
        return f"Symbol {symbol} not in allowed list"
    if rules.max_spread_points and tick and info:
        point = float(getattr(info, "point", 0.0) or 0.0)
        if point:
            spread = (tick.ask - tick.bid) / point
            if spread > rules.max_spread_points:
                return f"Spread {spread:.1f}pts > max {rules.max_spread_points}"
    return None


# ---------------------------------------------------------------- #
# Daily loss circuit breaker                                        #
# ---------------------------------------------------------------- #
#
# We cache the start-of-UTC-day equity the first time we're asked on a
# given UTC date. Today's realised P&L is computed from MT5 closed deals
# only (open positions' floating P&L is NOT counted — we don't want a
# temporary drawdown on an open position to halt new entries).

_daily_pnl_lock: threading.Lock = threading.Lock()
_daily_pnl_state: dict          = {"date": None, "start_balance": None}


def _utc_day_start():
    """Return datetime at 00:00 UTC today."""
    now = datetime.now(timezone.utc)
    return datetime(now.year, now.month, now.day, tzinfo=timezone.utc)


def _today_realised_pnl(mt5) -> float:
    """Sum of closed-deal profit+swap+commission since UTC midnight."""
    try:
        deals = list(mt5.history_deals_get(_utc_day_start(),
                                            datetime.now(timezone.utc)) or [])
    except Exception:
        return 0.0
    pnl = 0.0
    for d in deals:
        pnl += float(getattr(d, "profit", 0.0) or 0.0)
        pnl += float(getattr(d, "swap", 0.0) or 0.0)
        pnl += float(getattr(d, "commission", 0.0) or 0.0)
    return pnl


def _daily_loss_breach(mt5, rules) -> Optional[str]:
    """
    If today's realised losses exceed max_daily_loss_pct of the
    start-of-day balance, return a human-readable message.  Otherwise
    return None.  Returns None immediately if the breaker is disabled
    (max_daily_loss_pct == 0).
    """
    limit_pct = float(getattr(rules, "max_daily_loss_pct", 0.0) or 0.0)
    if limit_pct <= 0:
        return None

    global _daily_pnl_state
    today = _utc_day_start().date()
    try:
        acc = mt5.account_info()
    except Exception:
        return None
    if acc is None:
        return None
    cur_balance = float(getattr(acc, "balance", 0.0) or 0.0)
    if cur_balance <= 0:
        return None

    with _daily_pnl_lock:
        if _daily_pnl_state["date"] != today:
            # First call on a new UTC day — anchor start-of-day balance
            # using balance MINUS any P&L already realised today (so it
            # reflects the actual day-open balance even if the bot started
            # mid-session).
            realised = _today_realised_pnl(mt5)
            _daily_pnl_state = {
                "date": today,
                "start_balance": cur_balance - realised,
            }
        start_balance = float(_daily_pnl_state["start_balance"] or 0.0)

    if start_balance <= 0:
        return None

    pnl = _today_realised_pnl(mt5)
    loss_pct = (-pnl / start_balance) * 100.0 if pnl < 0 else 0.0
    if loss_pct >= limit_pct:
        return (f"Daily loss {loss_pct:.2f}% ≥ limit {limit_pct:.2f}% "
                f"(P&L={pnl:+.2f}, start-bal={start_balance:.2f}) — "
                f"new entries paused until 00:00 UTC")
    return None


def _count_all_open_positions(mt5) -> int:
    try:
        return len(list(mt5.positions_get() or []))
    except Exception:
        return 0


# ============================================================ #
# SECTION 10 — BOT (multi-symbol trading loop)                 #
# ============================================================ #

class Bot:
    def __init__(
        self,
        symbols:   list  = None,
        volume:    float = 0.50,
        poll_secs: float = 300.0,
        auto_trade: bool = True,
        use_ai:    bool  = True,
        conn      = None,
        **strategy_kwargs,
    ) -> None:
        self.symbols    = [s.strip().upper() for s in (symbols or ["EURUSD"])]
        self.volume     = float(volume)
        self.poll_secs  = float(poll_secs)
        self.auto_trade = bool(auto_trade)
        self.use_ai     = bool(use_ai)
        self._conn      = conn  # MT5Connection object for health checks
        self.strategy_config = {
            "atr_tolerance_multiplier": float(strategy_kwargs.get("atr_tolerance_multiplier", 1.5)),
            "sl_atr_mult":              float(strategy_kwargs.get("sl_atr_mult", 2.5)),
            "tp_atr_mult":              float(strategy_kwargs.get("tp_atr_mult", 4.5)),
            "partial_close_rr":         float(strategy_kwargs.get("partial_close_rr", 1.0)),
            "breakeven_buffer_pips":    float(strategy_kwargs.get("breakeven_buffer_pips", 1.0)),
        }

        self._strategy  = AI_Pro(use_ai=use_ai, **strategy_kwargs)
        self._running   = False
        self._stop      = threading.Event()

        self._results:     dict = {}
        self._results_lock = threading.Lock()
        self._positions:   list = []
        self._positions_lock = threading.Lock()

    def run(self) -> None:
        log.info("Bot starting — symbols=%s  poll=%ss  auto_trade=%s  use_ai=%s",
                 self.symbols, self.poll_secs, self.auto_trade, self.use_ai)
        if not self._strategy._ensure_mt5():
            raise RuntimeError("MT5 failed to initialize")
        self._running = True
        try:
            self._loop()
        except KeyboardInterrupt:
            log.info("Interrupted")
        finally:
            self._running = False
            self._strategy.shutdown()
            log.info("Bot stopped")

    def stop(self) -> None:
        self._running = False
        self._stop.set()

    def is_running(self) -> bool:
        return self._running

    def latest_results(self) -> dict:
        with self._results_lock:
            return dict(self._results)

    def open_positions(self) -> list:
        with self._positions_lock:
            return list(self._positions)

    def config_snapshot(self) -> dict:
        return {
            "symbols":    list(self.symbols),
            "volume":     self.volume,
            "poll_secs":  self.poll_secs,
            "auto_trade": self.auto_trade,
            "use_ai":     self.use_ai,
            "strategy":   dict(self.strategy_config),
        }

    def update_config(self, updates: dict) -> dict:
        """Update bot configuration on the fly."""
        try:
            if "volume" in updates:
                self.volume = float(updates["volume"])
            if "poll_secs" in updates:
                self.poll_secs = float(updates["poll_secs"])

            if "auto_trade" in updates:
                self.auto_trade = bool(updates["auto_trade"])
            if "use_ai" in updates:
                self.use_ai = bool(updates["use_ai"])
                self._strategy.use_ai = bool(updates["use_ai"])
            if "symbols" in updates:
                syms = updates["symbols"]
                if isinstance(syms, str):
                    syms = [syms]
                self.symbols = [s.strip().upper() for s in syms]
            
            strategy = updates.get("strategy", {})
            if strategy:
                if "atr_tolerance_multiplier" in strategy:
                    val = float(strategy["atr_tolerance_multiplier"])
                    self.strategy_config["atr_tolerance_multiplier"] = val
                    self._strategy.atr_tolerance_multiplier = val
                if "sl_atr_mult" in strategy:
                    val = float(strategy["sl_atr_mult"])
                    self.strategy_config["sl_atr_mult"] = val
                    self._strategy.sl_atr_mult = val
                if "tp_atr_mult" in strategy:
                    val = float(strategy["tp_atr_mult"])
                    self.strategy_config["tp_atr_mult"] = val
                    self._strategy.tp_atr_mult = val
                if "partial_close_rr" in strategy:
                    val = float(strategy["partial_close_rr"])
                    self.strategy_config["partial_close_rr"] = val
                    self._strategy.partial_close_rr = val
                if "breakeven_buffer_pips" in strategy:
                    val = float(strategy["breakeven_buffer_pips"])
                    self.strategy_config["breakeven_buffer_pips"] = val
                    self._strategy.breakeven_buffer_pips = val
            
            return {"ok": True, "config": self.config_snapshot()}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def mt5_snapshot(self) -> dict:
        return self._strategy.mt5_snapshot()

    def _loop(self) -> None:
        while self._running:
            # Perform periodic connection health check if connection object available
            if self._conn is not None:
                if not self._conn.check_connection():
                    log.warning("Connection health check failed, attempting reconnection...")
                    if not self._conn.reconnect():
                        log.error("Reconnection failed, pausing bot")
                        self._running = False
                        break
            
            for sym in self.symbols:
                if not self._running:
                    break
                try:
                    self._tick(sym)
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    log.error("Tick error [%s]: %s", sym, exc)
            if self._running:
                self._stop.wait(timeout=self.poll_secs)
            self._stop.clear()

    def _tick(self, symbol: str) -> None:
        import MetaTrader5 as mt5

        try:
            rules = _load_rules()
        except Exception:
            rules = None

        tick      = mt5.symbol_info_tick(symbol)
        info      = mt5.symbol_info(symbol)
        positions = list(mt5.positions_get(symbol=symbol) or [])
        records   = _load_memory()

        self._refresh_positions(mt5)

        # Hard rule pre-check
        if rules:
            block = _check_hard_rules(rules, symbol, positions, tick, info, records)
            if block:
                blocked_result = {
                    "signal": self._strategy._neutral(
                        symbol,
                        self._strategy._environment_summary(),
                    ),
                    "trade_result": None,
                    "exit_results": [],
                    "partial_results": [],
                    "ai_risk_results": [],
                }
                log_thought("preflight", symbol, "blocked", block, action="hold")
                log.info("[%s] Blocked: %s", symbol, block)
                with self._results_lock:
                    self._results[symbol] = blocked_result
                return

            # ALWAYS manage existing positions first (partial close /
            # breakeven / agent risk review). This runs regardless of any
            # circuit breakers below — we never abandon open trades.
            if positions:
                try:
                    self._strategy.check_partial_close_and_breakeven(symbol)
                except Exception as exc:
                    log.error("[%s] partial/BE check failed: %s", symbol, exc)
                try:
                    self._strategy.run_ai_risk_manager(symbol)
                except Exception as exc:
                    log.error("[%s] AI risk manager failed: %s", symbol, exc)

            # ------------------------------------------------------------
            # Circuit breakers — block NEW entries only (managers above
            # keep running). Each check falls through to run_strategy with
            # auto_trade=False so the UI still sees the signal but no
            # order fires.
            # ------------------------------------------------------------
            auto_trade_effective = self.auto_trade
            pause_reason: Optional[str] = None

            # (a) Daily loss breaker
            loss_msg = _daily_loss_breach(mt5, rules)
            if loss_msg:
                pause_reason = loss_msg

            # (b) Per-symbol cap (0 = unlimited)
            if pause_reason is None:
                per_cap = int(getattr(rules, "max_open_positions", 0) or 0)
                if per_cap > 0 and len(positions) >= per_cap:
                    pause_reason = (f"Per-symbol cap reached "
                                    f"({len(positions)}/{per_cap}) — managing only")

            # (c) Account-wide cap (0 = unlimited)
            if pause_reason is None:
                total_cap = int(getattr(rules, "max_open_positions_total", 0) or 0)
                if total_cap > 0:
                    total_open = _count_all_open_positions(mt5)
                    if total_open >= total_cap:
                        pause_reason = (f"Account-wide cap reached "
                                        f"({total_open}/{total_cap}) — managing only")

            # (d) Ollama liveness — if the agent is the backend and Ollama
            #     is unreachable, pause new entries (agent individually
            #     rejects, but this gives us a clearer single log line and
            #     avoids burning 30s timeouts per symbol).
            if (pause_reason is None
                    and _AGENT_IMPORT_OK
                    and _ai_agent is not None
                    and _ai_agent.agent_backend_enabled()):
                try:
                    health = _ai_agent.ollama_health()
                    if not health.get("reachable"):
                        pause_reason = (f"Ollama unreachable "
                                        f"({health.get('error','?')[:80]}) — "
                                        f"new entries paused")
                except Exception:
                    pass  # best-effort; don't block on a probe error

            if pause_reason:
                auto_trade_effective = False
                log_thought("preflight", symbol, "paused",
                            pause_reason, action="hold")
                log.info("[%s] New entries paused: %s", symbol, pause_reason)

        result = self._strategy.run_strategy(
            symbol=symbol,
            auto_trade=auto_trade_effective if rules else self.auto_trade,
            lot_size=self.volume,
        )

        with self._results_lock:
            self._results[symbol] = result

    def _refresh_positions(self, mt5) -> None:
        all_pos = []
        for sym in self.symbols:
            for p in (mt5.positions_get(symbol=sym) or []):
                tick = mt5.symbol_info_tick(p.symbol)
                is_long = p.type == mt5.POSITION_TYPE_BUY
                all_pos.append({
                    "ticket":    int(p.ticket),
                    "symbol":    str(p.symbol),
                    "direction": "BUY" if is_long else "SELL",
                    "volume":    float(p.volume),
                    "entry":     round(float(p.price_open), 5),
                    "current":   round(float(tick.bid if is_long else tick.ask), 5) if tick else None,
                    "sl":        round(float(p.sl), 5) if p.sl else None,
                    "tp":        round(float(p.tp), 5) if p.tp else None,
                    "pnl":       round(float(p.profit), 4),
                    "open_time": int(p.time),
                })
        with self._positions_lock:
            self._positions = all_pos


# ============================================================ #
# SECTION 11 — FLASK DASHBOARD                                 #
# ============================================================ #

from flask import Flask, request, jsonify, Response, send_file
from flask_cors import CORS
import mimetypes
from core.mt5_connection import MT5Connection, MT5ConnectionError

app = Flask(__name__)
CORS(app)

# ── Global persistent MT5 connection (shared across Flask + Bot) ─────
_mt5_conn = None
_mt5_conn_lock = threading.RLock()

def _ensure_shared_mt5_connection():
    """Ensure the global MT5Connection is created and connected."""
    global _mt5_conn
    with _mt5_conn_lock:
        if _mt5_conn is None:
            _mt5_conn = MT5Connection(dict(MT5_CONFIG))
            if not _mt5_conn.connect():
                _mt5_conn = None
        return _mt5_conn

# ── Locate the core/ asset folder relative to this file ──────────────
_HERE       = Path(__file__).resolve().parent
_CORE_HTML  = _HERE / "index.html"
_CORE_CSS   = _HERE / "core" / "css"   / "dashboard.css"
_CORE_JS    = _HERE / "core" / "js"    / "dashboard.js"

# Pre-load the HTML at startup (fast inline serve; refreshes on process restart)
def _load_dashboard_html() -> str:
    if _CORE_HTML.exists():
        return _CORE_HTML.read_text(encoding="utf-8")
    return "<h1>dashboard not found — expected index.html</h1>"

_DASHBOARD_HTML: str = _load_dashboard_html()

_bot_lock:   threading.Lock          = threading.Lock()
_bot:        Optional[Bot]           = None
_bot_thread: Optional[threading.Thread] = None
_bot_last_error: Optional[str] = None
_last_bot_config: dict = {
    "symbols":    ["EURUSD", "GBPUSD", "EURJPY", "GBPJPY"],
    "volume":     0.50,
    "poll_secs":  180.0,

    "auto_trade": True,
    "use_ai":     True,
    "strategy": {
        "atr_tolerance_multiplier": 1.5,
        "sl_atr_mult":              2.5,   # matches AI_Pro default and documented 2.5×ATR
        "tp_atr_mult":              4.5,   # matches AI_Pro default and documented 4.5×ATR
        "partial_close_rr":         1.0,
        "breakeven_buffer_pips":    1.0,
    },
}


def _mt5_initialize() -> bool:
    """
    Initialize MT5 using MT5Connection.
    Uses the global MT5_CONFIG (populated from env vars + saved credentials).
    Returns True on success.
    """
    import os
    cfg = dict(MT5_CONFIG)

    # Allow env vars to override without editing this file
    if os.getenv("MT5_PATH"):     cfg["path"]     = os.environ["MT5_PATH"]
    if os.getenv("MT5_LOGIN"):    cfg["login"]    = os.environ["MT5_LOGIN"]
    if os.getenv("MT5_PASSWORD"): cfg["password"] = os.environ["MT5_PASSWORD"]
    if os.getenv("MT5_SERVER"):   cfg["server"]   = os.environ["MT5_SERVER"]

    conn = MT5Connection(cfg)
    return conn.connect()


def _read_mt5_runtime_info(mt5) -> dict:
    terminal = mt5.terminal_info()
    account = mt5.account_info()
    symbols = list(mt5.symbols_get() or [])
    visible_symbols = [str(s.name) for s in symbols if getattr(s, "visible", False)]
    return {
        "connected": True,
        "terminal_name": getattr(terminal, "name", None) if terminal else None,
        "terminal_company": getattr(terminal, "company", None) if terminal else None,
        "terminal_path": getattr(terminal, "path", None) if terminal else None,
        "login": getattr(account, "login", None) if account else None,
        "server": getattr(account, "server", None) if account else None,
        "account_name": getattr(account, "name", None) if account else None,
        "currency": getattr(account, "currency", None) if account else None,
        "trade_allowed": bool(getattr(account, "trade_allowed", False)) if account else False,
        "visible_symbols": visible_symbols,
        "symbols_total": len(symbols),
    }


def _mt5_snapshot(shutdown_when_done: bool = True) -> dict:
    """
    Connect, read terminal + account info, then disconnect.
    Returns a serialisable dict; 'connected' is False on any failure.
    """
    import os
    cfg = dict(MT5_CONFIG)
    if os.getenv("MT5_PATH"):     cfg["path"]     = os.environ["MT5_PATH"]
    if os.getenv("MT5_LOGIN"):    cfg["login"]    = os.environ["MT5_LOGIN"]
    if os.getenv("MT5_PASSWORD"): cfg["password"] = os.environ["MT5_PASSWORD"]
    if os.getenv("MT5_SERVER"):   cfg["server"]   = os.environ["MT5_SERVER"]

    try:
        conn = MT5Connection(cfg)
    except Exception as exc:
        return {"connected": False, "error": str(exc)}

    if not conn.connect():
        import MetaTrader5 as mt5
        err = mt5.last_error()
        return {"connected": False, "error": f"MT5 initialize failed: {err}"}

    try:
        return conn.runtime_info()
    finally:
        if shutdown_when_done:
            conn.disconnect()


def _read_mt5_positions(mt5) -> list:
    positions = []
    for p in (mt5.positions_get() or []):
        tick = mt5.symbol_info_tick(p.symbol)
        is_long = p.type == mt5.POSITION_TYPE_BUY
        positions.append({
            "ticket":    int(p.ticket),
            "symbol":    str(p.symbol),
            "direction": "BUY" if is_long else "SELL",
            "volume":    float(p.volume),
            "entry":     round(float(p.price_open), 5),
            "current":   round(float(tick.bid if is_long else tick.ask), 5) if tick else None,
            "sl":        round(float(p.sl), 5) if p.sl else None,
            "tp":        round(float(p.tp), 5) if p.tp else None,
            "pnl":       round(float(p.profit), 4),
            "open_time": int(p.time),
        })
    return positions


def _mt5_positions_snapshot(shutdown_when_done: bool = True) -> list:
    try:
        import MetaTrader5 as mt5
    except Exception:
        return []

    if not _mt5_initialize():
        return []

    try:
        return _read_mt5_positions(mt5)
    finally:
        if shutdown_when_done:
            try:
                mt5.shutdown()
            except Exception:
                pass


def _read_mt5_trade_history(mt5, days: int = 30, limit: int = 100) -> list:
    utc_now = datetime.now(timezone.utc)
    from_dt = utc_now - timedelta(days=max(1, int(days)))
    deals = list(mt5.history_deals_get(from_dt, utc_now) or [])

    try:
        exit_entries = {mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_OUT_BY, mt5.DEAL_ENTRY_INOUT}
        buy_type = mt5.DEAL_TYPE_BUY
    except AttributeError:
        exit_entries = {1, 3, 2}
        buy_type = 0

    grouped: dict = {}
    for d in deals:
        position_id = int(getattr(d, "position_id", 0) or 0)
        if not position_id:
            continue
        bucket = grouped.setdefault(position_id, {
            "position_id": position_id,
            "symbol": str(getattr(d, "symbol", "")),
            "entry_price": None,
            "exit_price": None,
            "exit_price_notional": 0.0,
            "exit_volume": 0.0,
            "direction": None,
            "volume": 0.0,
            "pnl": 0.0,
            "ts": None,
            "source": "",
            "initial_sl": None,
            "initial_tp": None,
            "entry_deals": [],
            "exit_deals": [],
        })
        if bucket["symbol"] == "" and getattr(d, "symbol", None):
            bucket["symbol"] = str(d.symbol)
        entry_flag = int(getattr(d, "entry", -1))
        
        # Collect all costs (commission, swap, fee) from all deals
        deal_cost = (float(getattr(d, "commission", 0.0) or 0.0) +
                     float(getattr(d, "swap", 0.0) or 0.0) +
                     float(getattr(d, "fee", 0.0) or 0.0))
        bucket["pnl"] += deal_cost
        
        if entry_flag in exit_entries:
            bucket["exit_deals"].append(d)
            exit_vol = abs(float(getattr(d, "volume", 0.0) or 0.0))
            bucket["pnl"] += float(getattr(d, "profit", 0.0) or 0.0)
            bucket["volume"] += exit_vol
            bucket["exit_volume"] += exit_vol
            bucket["exit_price_notional"] += exit_vol * float(getattr(d, "price", 0.0) or 0.0)
            bucket["ts"] = datetime.fromtimestamp(int(getattr(d, "time", 0)), timezone.utc).isoformat()
            bucket["source"] = str(getattr(d, "comment", "") or "")
        else:
            bucket["entry_deals"].append(d)
            if bucket["entry_price"] is None:
                bucket["entry_price"] = float(getattr(d, "price", 0.0) or 0.0)
                bucket["direction"] = "BUY" if int(getattr(d, "type", -1)) == buy_type else "SELL"

    trades = []
    for bucket in grouped.values():
        if not bucket["exit_deals"]:
            continue
        if bucket["exit_volume"] > 0:
            bucket["exit_price"] = bucket["exit_price_notional"] / bucket["exit_volume"]

        try:
            orders = list(mt5.history_orders_get(position=bucket["position_id"]) or [])
        except Exception:
            orders = []
        for order in orders:
            sl = float(getattr(order, "sl", 0.0) or 0.0)
            tp = float(getattr(order, "tp", 0.0) or 0.0)
            if bucket["initial_sl"] is None and sl:
                bucket["initial_sl"] = sl
            if bucket["initial_tp"] is None and tp:
                bucket["initial_tp"] = tp
            if bucket["initial_sl"] is not None and bucket["initial_tp"] is not None:
                break

        realized_rr = None
        entry_price = float(bucket["entry_price"] or 0.0)
        exit_price = float(bucket["exit_price"] or 0.0)
        initial_sl = float(bucket["initial_sl"] or 0.0) if bucket["initial_sl"] is not None else None
        if entry_price and exit_price and initial_sl not in (None, 0.0):
            risk = abs(entry_price - initial_sl)
            reward = abs(exit_price - entry_price)
            if risk > 0:
                realized_rr = reward / risk

        trades.append({
            "ts": bucket["ts"],
            "symbol": bucket["symbol"],
            "direction": bucket["direction"] or "BUY",
            "source": bucket["source"] or "mt5",
            "entry": round(entry_price, 5),
            "exit": round(exit_price, 5),
            "volume": round(float(bucket["volume"]), 2),
            "pnl": round(float(bucket["pnl"]), 4),
            "position_id": bucket["position_id"],
            "initial_sl": round(initial_sl, 5) if initial_sl not in (None, 0.0) else None,
            "initial_tp": round(float(bucket["initial_tp"] or 0.0), 5) if bucket["initial_tp"] not in (None, 0.0) else None,
            "rr": round(realized_rr, 2) if realized_rr is not None else None,
            "outcome": "WIN" if float(bucket["pnl"]) > 0 else "LOSS" if float(bucket["pnl"]) < 0 else "BE",
        })

    trades.sort(key=lambda x: x["ts"] or "", reverse=True)
    return trades[:limit]


def _mt5_trade_history_snapshot(days: int = 30, limit: int = 100, shutdown_when_done: bool = True) -> list:
    try:
        import MetaTrader5 as mt5
    except Exception:
        return []

    if not _mt5_initialize():
        return []

    try:
        return _read_mt5_trade_history(mt5, days=days, limit=limit)
    finally:
        if shutdown_when_done:
            try:
                mt5.shutdown()
            except Exception:
                pass


def _choose_http_port() -> int:
    host = "0.0.0.0"
    # core/server.py owns :5000 for the main dashboard.
    # Keep ai_pro.py's optional Flask dashboard off that port to avoid conflicts.
    preferred_port = int(os.getenv("APP_PORT", "5001"))
    fallback_port = 5002
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, preferred_port))
            return preferred_port
        except OSError:
            return fallback_port


# ---- REST API ----------------------------------------------- #

def _get_bot():
    with _bot_lock:
        return _bot, _bot_thread


def _run_bot_thread(bot: Bot) -> None:
    global _bot, _bot_thread, _bot_last_error
    try:
        _bot_last_error = None
        bot.run()
    except Exception as exc:
        _bot_last_error = str(exc)
        log.exception("Bot crashed: %s", exc)
    finally:
        with _bot_lock:
            _bot        = None
            _bot_thread = None


@app.route("/")
def index():
    return Response(_DASHBOARD_HTML, mimetype="text/html")


@app.route("/core/css/dashboard.css")
def serve_css():
    if _CORE_CSS.exists():
        return send_file(_CORE_CSS, mimetype="text/css")
    return Response("/* css not found */", mimetype="text/css", status=404)


@app.route("/core/js/dashboard.js")
def serve_js():
    if _CORE_JS.exists():
        return send_file(_CORE_JS, mimetype="application/javascript")
    return Response("/* js not found */", mimetype="application/javascript", status=404)


@app.route("/health")
def health():
    return jsonify({"ok": True, "version": "AI_Pro"})


@app.route("/bot/status")
def bot_status():
    bot, thread = _get_bot()
    running = bool(thread and thread.is_alive())
    mt5 = bot.mt5_snapshot() if bot and running else _mt5_snapshot(shutdown_when_done=True)
    open_positions = []
    if running:
        try:
            import MetaTrader5 as mt5_module
            open_positions = _read_mt5_positions(mt5_module)
        except Exception:
            open_positions = bot.open_positions() if bot else []
    else:
        open_positions = _mt5_positions_snapshot(shutdown_when_done=True)
    latest_results = bot.latest_results() if bot else {}
    all_decisions = {}
    config = bot.config_snapshot() if bot else dict(_last_bot_config)
    if (not bot) and mt5.get("connected") and mt5.get("visible_symbols") and not config.get("symbols"):
        config["symbols"] = list(mt5["visible_symbols"][:6])
    if bot:
        for sym in bot.symbols:
            result = latest_results.get(sym)
            signal = (result or {}).get("signal") or {
                "signal": "neutral",
                "ai_approved": None,
                "reason": "Waiting for first scan...",
                "stop_loss": None,
                "take_profit": None,
                "confidence": 0,
                "ai_reason": None,
            }
            all_decisions[sym] = {
                "action":     signal["signal"],
                "approve":    signal.get("ai_approved"),
                "reason":     signal["reason"],
                "sl":         signal.get("stop_loss"),
                "tp":         signal.get("take_profit"),
                "confidence": signal["confidence"] / 100,
                "ai_reason":  signal.get("ai_reason"),
            }
    # ---- Active AI backend + liveness summary ----
    # Agent is the default. We report which backend is currently selected
    # and, if it's the agent, whether the Ollama server is actually up so
    # the dashboard can warn the user before a trade gets reviewed.
    ai_backend = {
        "active":       "deepseek",      # overwritten below when agent is on
        "agent_import": bool(_AGENT_IMPORT_OK),
    }
    try:
        if _AGENT_IMPORT_OK and _ai_agent:
            if _ai_agent.agent_backend_enabled():
                ai_backend["active"] = "agent"
            health = _ai_agent.ollama_health()
            ai_backend.update({
                "ollama_reachable":   bool(health.get("reachable")),
                "ollama_model":       health.get("model"),
                "ollama_model_loaded": bool(health.get("model_loaded")),
                "ollama_url":         health.get("url"),
                "ollama_error":       health.get("error"),
            })
    except Exception as _exc:
        ai_backend["error"] = str(_exc)

    return jsonify({
        "running": running,
        "bot": {
            "symbols":    bot.symbols,
            "volume":     bot.volume,
            "poll_secs":  bot.poll_secs,

            "auto_trade": bot.auto_trade,
        } if bot else None,
        "config": config,
        "mt5": mt5,
        "bot_error": _bot_last_error,
        "all_decisions":  all_decisions,
        "open_positions": open_positions,
        "ai_backend": ai_backend,
    })


@app.route("/bot/ai_backend")
def bot_ai_backend():
    """
    Lightweight probe for the dashboard to show which AI backend is live and
    whether the Ollama server is actually reachable. Safe to poll.
    """
    out = {
        "active":       "deepseek",
        "agent_import": bool(_AGENT_IMPORT_OK),
    }
    try:
        if _AGENT_IMPORT_OK and _ai_agent:
            if _ai_agent.agent_backend_enabled():
                out["active"] = "agent"
            h = _ai_agent.ollama_health()
            out.update({
                "ollama_reachable":   bool(h.get("reachable")),
                "ollama_model":       h.get("model"),
                "ollama_model_loaded": bool(h.get("model_loaded")),
                "ollama_url":         h.get("url"),
                "ollama_error":       h.get("error"),
            })
    except Exception as exc:
        out["error"] = str(exc)
    return jsonify(out)


@app.route("/bot/ai_thoughts")
def bot_ai_thoughts():
    try:
        since = request.args.get("since")
        limit_str = request.args.get("limit", "60")
        try:
            limit = min(int(limit_str), 120)
        except (ValueError, TypeError):
            limit = 60  # Default to 60 if parsing fails
        return jsonify({"ok": True,
                        "thoughts": get_thoughts(since_ts=since, limit=limit)})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc), "thoughts": []}), 500


@app.route("/bot/thoughts/clear", methods=["POST"])
def clear_thoughts_api():
    clear_thoughts()
    return jsonify({"ok": True})


@app.route("/bot/start", methods=["POST"])
def bot_start():
    global _bot, _bot_thread, _last_bot_config, _bot_last_error
    bot, thread = _get_bot()
    if thread and thread.is_alive():
        return jsonify({"error": "Already running"}), 409

    data = request.get_json(silent=True) or {}

    raw_syms = data.get("symbols") or ["EURUSD"]
    if isinstance(raw_syms, str):
        raw_syms = [raw_syms]
    symbols = []
    for s in raw_syms:
        s = str(s).strip().upper()
        if not re.fullmatch(r"[A-Z0-9]{2,12}", s):
            return jsonify({"error": f"Invalid symbol '{s}'"}), 400
        symbols.append(s)

    try:
        volume    = float(data.get("volume", 0.50))
        poll_secs = float(data.get("poll_secs", 300.0))
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400


    auto_trade = bool(data.get("auto_trade", True))
    use_ai     = bool(data.get("use_ai",     True))
    strategy   = data.get("strategy", {}) or {}

    _last_bot_config = {
        "symbols":    list(symbols),
        "volume":     volume,
        "poll_secs":  poll_secs,

        "auto_trade": auto_trade,
        "use_ai":     use_ai,
        "strategy": {
            "atr_tolerance_multiplier": float(strategy.get("atr_tolerance_multiplier", 1.5)),
            "sl_atr_mult":              float(strategy.get("sl_atr_mult",              1.5)),
            "tp_atr_mult":              float(strategy.get("tp_atr_mult",              3.0)),
            "partial_close_rr":         float(strategy.get("partial_close_rr",         1.0)),
            "breakeven_buffer_pips":    float(strategy.get("breakeven_buffer_pips",    1.0)),
        },
    }

    mt5_check = _mt5_snapshot(shutdown_when_done=True)
    if not mt5_check.get("connected"):
        _bot_last_error = mt5_check.get("error") or "MT5 connection failed"
        return jsonify({"error": _bot_last_error}), 400

    clear_thoughts()

    new_bot = Bot(
        symbols=symbols,
        volume=volume,
        poll_secs=poll_secs,

        auto_trade=auto_trade,
        use_ai=use_ai,
        atr_tolerance_multiplier = float(strategy.get("atr_tolerance_multiplier", 1.5)),
        sl_atr_mult              = float(strategy.get("sl_atr_mult",              1.5)),
        tp_atr_mult              = float(strategy.get("tp_atr_mult",              3.0)),
        partial_close_rr         = float(strategy.get("partial_close_rr",         1.0)),
        breakeven_buffer_pips    = float(strategy.get("breakeven_buffer_pips",    1.0)),
    )

    t = threading.Thread(target=_run_bot_thread, args=(new_bot,), daemon=True)
    with _bot_lock:
        _bot        = new_bot
        _bot_thread = t
    t.start()
    return jsonify({"ok": True, "symbols": symbols,
                    "use_ai": use_ai})


@app.route("/bot/stop", methods=["POST"])
def bot_stop():
    bot, _ = _get_bot()
    if bot:
        bot.stop()
    return jsonify({"ok": True})


@app.route("/bot/update_config", methods=["POST"])
def bot_update_config():
    """Update running bot configuration without restarting."""
    bot, thread = _get_bot()
    if not (thread and thread.is_alive()):
        return jsonify({"error": "Bot not running"}), 409
    
    data = request.get_json(silent=True) or {}
    result = bot.update_config(data)
    if result.get("ok"):
        # Also update global last config for future restarts
        global _last_bot_config
        _last_bot_config.update(data)
        return jsonify(result)
    else:
        return jsonify(result), 400

@app.route("/bot/config/symbols", methods=["POST"])
def bot_config_symbols():
    """Toggle symbol on/off in bot configuration."""
    bot, thread = _get_bot()
    
    data = request.get_json(silent=True) or {}
    symbol = data.get('symbol')
    enabled = data.get('enabled', True)
    
    if not symbol:
        return jsonify({"success": False, "message": "Symbol not provided"}), 400
    
    # Get current symbols
    global _last_bot_config
    current_symbols = _last_bot_config.get("symbols", ["EURUSD", "GBPUSD", "EURJPY", "GBPJPY"])
    
    # Update symbols list
    if enabled and symbol not in current_symbols:
        current_symbols.append(symbol)
    elif not enabled and symbol in current_symbols:
        current_symbols.remove(symbol)
    
    # Ensure at least 1 symbol remains
    if len(current_symbols) == 0:
        return jsonify({"success": False, "message": "At least one symbol must be enabled"}), 400
    
    # Update config
    _last_bot_config["symbols"] = current_symbols
    
    # If bot is running, update it too
    if thread and thread.is_alive():
        result = bot.update_config({"symbols": current_symbols})
        return jsonify({"success": result.get("ok", False), "symbols": current_symbols})
    
    return jsonify({"success": True, "symbols": current_symbols})


@app.route("/bot/history")
def bot_history():
    bot, thread = _get_bot()
    running = bool(thread and thread.is_alive())
    trades = []
    account_balance = None
    
    if running:
        try:
            import MetaTrader5 as mt5
            trades = _read_mt5_trade_history(mt5, days=30, limit=100)
            acc_info = mt5.account_info()
            if acc_info:
                account_balance = round(float(acc_info.balance), 2)
        except Exception:
            trades = _load_memory()
    else:
        trades = _mt5_trade_history_snapshot(days=30, limit=100, shutdown_when_done=True)
        try:
            import MetaTrader5 as mt5
            if _mt5_initialize():
                acc_info = mt5.account_info()
                if acc_info:
                    account_balance = round(float(acc_info.balance), 2)
                mt5.shutdown()
        except Exception:
            pass
    
    return jsonify({"trades": trades, "account_balance": account_balance})


@app.route("/mt5/connect", methods=["POST"])
def mt5_connect():
    """Accept credentials from the dashboard, update MT5_CONFIG, and test the connection."""
    global MT5_CONFIG
    data = request.get_json(silent=True) or {}

    login    = data.get("login")
    password = data.get("password", "")
    server   = data.get("server", "")
    path     = data.get("path") or None

    if not login:
        return jsonify({"ok": False, "error": "Account number (login) is required"}), 400
    try:
        login = int(login)
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Login must be a numeric account number"}), 400

    # Update the live config so future _mt5_initialize() calls pick it up
    MT5_CONFIG["login"]    = login
    MT5_CONFIG["password"] = password or None
    MT5_CONFIG["server"]   = server   or None
    MT5_CONFIG["path"]     = path

    # Push to env vars as a fallback layer
    os.environ["MT5_LOGIN"] = str(login)
    if password: os.environ["MT5_PASSWORD"] = password
    if server:   os.environ["MT5_SERVER"]   = server
    if path:     os.environ["MT5_PATH"]      = path

    # Persist to disk for auto-reconnect on next startup
    _save_credentials(login, password or "", server or "", path or "")

    # Use MT5Connection for the actual test
    conn = MT5Connection(dict(MT5_CONFIG))
    if conn.connect():
        try:
            info = conn.runtime_info()
        finally:
            conn.disconnect()
        return jsonify({"ok": True, "mt5": info})
    else:
        import MetaTrader5 as mt5
        err = mt5.last_error()
        return jsonify({"ok": False, "error": f"MT5 connection failed: {err}"}), 400


@app.route("/mt5/credentials", methods=["GET"])
def mt5_credentials_get():
    """Return saved credentials (login + server only — never expose password)."""
    import json as _json
    if not _CREDS_PATH.exists():
        return jsonify({"ok": True, "saved": False})
    try:
        creds = _json.loads(_CREDS_PATH.read_text(encoding="utf-8"))
        return jsonify({
            "ok": True,
            "saved": True,
            "login":  creds.get("login"),
            "server": creds.get("server"),
            "path":   creds.get("path"),
            "has_password": bool(creds.get("password")),
        })
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/mt5/credentials", methods=["DELETE"])
def mt5_credentials_delete():
    """Remove saved credentials file."""
    try:
        if _CREDS_PATH.exists():
            _CREDS_PATH.unlink()
        return jsonify({"ok": True})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/mt5/status")
def mt5_status_api():
    """Return MT5 account status: connection, server, account, trading, equity, balance (using shared persistent connection)."""
    global _mt5_conn
    try:
        # Try to use shared connection first
        conn = _ensure_shared_mt5_connection()
        if not conn or not conn.is_connected():
            # If shared conn isn't available, create a temporary one to check status
            temp_conn = MT5Connection(dict(MT5_CONFIG))
            if temp_conn.connect():
                conn = temp_conn
            else:
                return jsonify({
                    "connected": False,
                    "server": None,
                    "login": None,
                    "account_name": None,
                    "trade_allowed": False,
                    "equity": None,
                    "balance": None,
                    "error": "MT5 not connected"
                })
        
        # Get account info from the working connection
        acct = conn.account_info()
        if not acct:
            return jsonify({
                "connected": False,
                "server": None,
                "login": None,
                "account_name": None,
                "trade_allowed": False,
                "equity": None,
                "balance": None,
                "error": "No account info"
            })
        
        result = {
            "connected": True,
            "server": acct.server,
            "login": acct.login,
            "account_name": getattr(acct, 'name', None),
            "trade_allowed": acct.trade_allowed,
            "equity": float(acct.equity) if acct.equity else None,
            "balance": float(acct.balance) if acct.balance else None,
        }
        
        return jsonify(result)
    except Exception as exc:
        log.error("MT5 status API error: %s", exc)
        return jsonify({
            "connected": False,
            "error": str(exc)
        })


@app.route("/ai/init", methods=["POST"])
def ai_init():
    """Probe the Ollama agent (the only supported backend)."""
    if not (_AGENT_IMPORT_OK and _ai_agent is not None):
        return jsonify({"ok": False,
                        "message": "Agent module unavailable — check ai_agent.py import"}), 500
    try:
        health = _ai_agent.ollama_health()
    except Exception as exc:
        return jsonify({"ok": False, "message": f"Agent probe error: {exc}"}), 500
    ok = bool(health.get("reachable") and health.get("model_loaded"))
    log.info("Ollama agent probe: reachable=%s model_loaded=%s url=%s model=%s",
             health.get("reachable"), health.get("model_loaded"),
             health.get("url"), health.get("model"))
    return jsonify({"ok": ok, "message": "Ollama agent probed", "health": health})


@app.route("/bot/signal/<symbol>")
def manual_signal(symbol: str):
    """Run a one-shot signal check without a running bot (for testing)."""
    strategy = AI_Pro()
    try:
        sig = strategy.generate_trade_signal(symbol.upper())
    finally:
        strategy.shutdown()
    return jsonify(sig)


# ============================================================ #
# SECTION 12 — ENTRY POINT                                     #
# ============================================================ #

if __name__ == "__main__":
    if "--run" in sys.argv:
        # Headless bot mode — no Flask (always live trading).
        # Volume defaults to 0.50 lot; override with BOT_AUTOSTART_VOLUME.
        bot = Bot(
            symbols=["EURUSD", "GBPUSD", "EURJPY", "GBPJPY"],
            volume=float(os.getenv("BOT_AUTOSTART_VOLUME", "0.50")),
            poll_secs=300,
            auto_trade=True,
            use_ai=True,
        )
        bot.run()
    else:
        # Load any previously saved MT5 credentials before Flask starts
        _load_saved_credentials()

        # Announce the active AI backend at startup so there's zero
        # ambiguity about which review path will fire for new trades.
        def _announce_backend():
            if _AGENT_IMPORT_OK and _ai_agent is not None:
                try:
                    h = _ai_agent.ollama_health()
                    log.info(
                        "AI backend: Ollama agent | url=%s | model=%s | "
                        "reachable=%s | model_loaded=%s",
                        h.get("url"), h.get("model"),
                        h.get("reachable"), h.get("model_loaded"),
                    )
                    if not h.get("reachable"):
                        log.warning(
                            "Ollama is NOT reachable at startup — new entries "
                            "will be paused until it comes back. Error: %s",
                            h.get("error"),
                        )
                except Exception as exc:
                    log.warning("Backend probe error at startup: %s", exc)
            else:
                log.error(
                    "AI backend: UNAVAILABLE — ai_agent.py failed to import. "
                    "No new trades will be approved until this is fixed."
                )
        _announce_backend()

        # Attempt MT5 auto-connect in the background so it's ready when the page loads
        def _auto_connect_mt5():
            if MT5_CONFIG.get("login"):
                log.info("Auto-connecting MT5 with saved credentials (login=%s) ...", MT5_CONFIG["login"])
                result = _mt5_snapshot(shutdown_when_done=True)
                if result.get("connected"):
                    log.info("MT5 auto-connect OK — server=%s", result.get("server"))
                else:
                    log.warning("MT5 auto-connect failed: %s", result.get("error"))
            else:
                log.info("No saved MT5 credentials — waiting for manual connect via dashboard.")

        threading.Thread(target=_auto_connect_mt5, daemon=True, name="mt5-autoconnect").start()

        # Optional startup test trade. DISABLED by default in production —
        # set STARTUP_TEST_TRADE=1 to enable (useful for staging to verify
        # MT5 plumbing). Never fires on a live account unless explicitly
        # opted in.
        def _test_trade_on_startup():
            import time
            time.sleep(2)  # Wait for Flask to be ready
            if os.getenv("STARTUP_TEST_TRADE", "0").strip() != "1":
                log.info("Startup test trade disabled (set STARTUP_TEST_TRADE=1 to enable).")
                return
            log.info(">>> Running startup test trade (0.001 lot EURUSD BUY)...")
            try:
                # Reload credentials before test (in case they were just saved)
                _load_saved_credentials()

                # Use or create shared connection for test trade
                global _mt5_conn
                if _mt5_conn is None:
                    _mt5_conn = MT5Connection(dict(MT5_CONFIG))
                    _mt5_conn.connect()

                if _mt5_conn and _mt5_conn.is_connected():
                    try:
                        result = _mt5_conn.place_test_trade(symbol="EURUSD", volume=0.001)
                        if result.get("ok"):
                            log.info("===== SUCCESS: Test trade placed! Ticket=%s =====", result.get("ticket"))
                            return
                        else:
                            log.warning("Test trade failed: %s", result.get("error"))
                    except Exception as exc:
                        log.warning("Test trade error: %s", exc)
                else:
                    log.warning("Could not connect to MT5 for test trade. Enter credentials via dashboard.")
            except Exception as exc:
                log.warning("Test trade not executed (waiting for MT5 connection): %s", exc)

        threading.Thread(target=_test_trade_on_startup, daemon=True, name="test-trade").start()

        # Auto-start bot after MT5 connects
        def _auto_start_bot():
            import time
            max_wait = 15
            elapsed = 0
            while elapsed < max_wait:
                time.sleep(1)
                elapsed += 1
                mt5_status = _mt5_snapshot(shutdown_when_done=True)
                if mt5_status.get("connected") and mt5_status.get("trade_allowed"):
                    log.info("MT5 connected ✓ — auto-starting bot polling...")
                    # Use shared global connection for the bot
                    shared_conn = _ensure_shared_mt5_connection()
                    # Auto-start volume — 0.50 lot default. Override via
                    # BOT_AUTOSTART_VOLUME env var if you want smaller size
                    # for staging/testing.
                    auto_bot = Bot(
                        symbols=["EURUSD", "GBPUSD", "EURJPY", "GBPJPY"],
                        volume=float(os.getenv("BOT_AUTOSTART_VOLUME", "0.50")),
                        poll_secs=180,
                        auto_trade=True,
                        use_ai=True,
                        conn=shared_conn,
                        atr_tolerance_multiplier=1.5,
                        sl_atr_mult=2.5,
                        tp_atr_mult=4.5,
                        partial_close_rr=1.0,
                        breakeven_buffer_pips=1.0,
                    )
                    t = threading.Thread(target=_run_bot_thread, args=(auto_bot,), daemon=True)
                    with _bot_lock:
                        global _bot, _bot_thread
                        _bot = auto_bot
                        _bot_thread = t
                    t.start()
                    log.info("===== BOT AUTO-STARTED — Scanning: EURUSD, GBPUSD, EURJPY, GBPJPY =====")
                    return
            log.info("Auto-start timeout — bot waiting for MT5 connection via dashboard")

        threading.Thread(target=_auto_start_bot, daemon=True, name="bot-autostart").start()

        print("=" * 55)
        chosen_port = _choose_http_port()
        if chosen_port in (5000, 5001):
            print(f"  AI Pro Dashboard -> http://localhost:{chosen_port}")
            print(f"  LAN: http://<your-ip>:{chosen_port}")
        else:
            print(f"  AI Pro Dashboard -> http://localhost:{chosen_port}")
            print(f"  LAN: http://<your-ip>:{chosen_port}")
        print("  API:  /bot/start  /bot/stop  /bot/status")
        print("  Test: /bot/signal/EURUSD")
        print("=" * 55)
        print("  ▶ AUTO-START: Connecting MT5, Running Test Trade, Starting Bot...")
        print("=" * 55)
        app.run(host="0.0.0.0", port=chosen_port, debug=False, threaded=True)
