"""
ai_pro.py
============
AI Pro — CHoCH + Daily Levels Strategy, AI-Enhanced Edition
===============================================================
Merges the complete AI Pro signal engine (CHoCH + Continuation on 4
environments, ATR-based SL/TP, RSI exit, partial-close/breakeven trail)
with the full AI stack from algo-v2:

  ► LocalLLM         — Qwen2.5-1.5B-Instruct via HuggingFace Transformers
  ► Thought Logger   — thread-safe in-memory AI reasoning log (last 120)
  ► AI Signal Review — Qwen validates every AI Pro signal before execution
  ► AI Risk Manager  — Qwen reviews open positions every N ticks
                       (trail harder / tighten / close if momentum gone)
  ► Trade Memory     — JSON persistence (last 100 trades, win/loss stats)
  ► Flask Dashboard  — REST API + embedded UI at http://localhost:5000

The 4 Environments
----------------------------------------------
ENV 1 │ CHoCH BUY  at PDL — failed Lower Low anchored at Previous Day Low
ENV 2 │ CHoCH SELL at PDH — failed Higher High anchored at Previous Day High
ENV 3 │ Continuation BUY  — broke above PDH, retesting it as support
ENV 4 │ Continuation SELL — broke below PDL, retesting it as resistance

All environments still require momentum alignment + level interaction.
CHoCH signals have priority (confidence 85) over Continuation (65-75).
Auto-trade only fires when confidence >= 70 AND Qwen approves.

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
# SECTION 1 — TRADING RULES (embedded YAML)                    #
# ============================================================ #

_RULES_YAML = """
allowed_symbols:
  - EURUSD
  - GBPUSD
  - XAUUSD
  - USDJPY
  - GBPJPY

sessions:
  always: [0, 24]

risk:
  max_risk_per_trade_pct:  1.0
  max_open_positions:      1
  min_rr_ratio:            1.5
  max_spread_points:       30
  daily_loss_limit_usd:   -200.0
  daily_profit_target_usd: 500.0

entry:
  min_confluence_signals: 1
  allow_counter_trend:    false
  allow_pyramiding:       false

exit:
  breakeven_trigger_points: 10
  trailing_stop_points:      8

  partial_take_profit:
    enabled:          true
    trigger_fraction: 0.50
    close_fraction:   0.50
    sl_plus_points:    5

  max_trade_duration_minutes: 240

  conditions:
    - "If price stalls before TP, AI may close early."
    - "Never let a winning trade reverse past breakeven once BE is set."
    - "CHoCH entries get wider trailing tolerance — structure-based moves need room."

ai:
  extra_instructions: |
    AI_Pro AI risk notes:
    1. CHoCH setups (ENV1/ENV2) have structural backing — trail generously.
    2. Continuation setups (ENV3/ENV4) are momentum-dependent — tighten faster.
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
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("ai_pro")

_MEMORY_PATH = Path(__file__).resolve().parent / "ai_pro_trade_log.json"

# ============================================================ #
# SECTION 3 — LOCAL LLM (Qwen2.5-1.5B-Instruct)               #
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
            "AI_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"
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
                    "You are a strict JSON generator for a professional forex "
                    "trading AI.  Output JSON only.  No explanation, no "
                    "markdown, no backticks."
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
    except Exception:
        pass


def get_thoughts(since_ts: Optional[str] = None, limit: int = 60) -> list:
    with _thoughts_lock:
        items = list(_thoughts)
    if since_ts:
        items = [t for t in items if t["ts"] > since_ts]
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
        self.max_open_positions: int               = int(risk.get("max_open_positions", 1))
        self.min_rr_ratio: float                   = float(risk.get("min_rr_ratio", 1.5))
        self.max_spread_points: int                = int(risk.get("max_spread_points", 30))
        self.daily_loss_limit_usd: Optional[float] = _to_float(risk.get("daily_loss_limit_usd"))
        self.daily_profit_target_usd: Optional[float] = _to_float(risk.get("daily_profit_target_usd"))

        entry: dict = raw.get("entry") or {}
        self.min_confluence_signals: int = int(entry.get("min_confluence_signals", 1))
        self.allow_counter_trend: bool   = bool(entry.get("allow_counter_trend", False))
        self.allow_pyramiding: bool      = bool(entry.get("allow_pyramiding", False))

        exit_: dict = raw.get("exit") or {}
        self.breakeven_trigger_points: Optional[int]   = _to_int(exit_.get("breakeven_trigger_points"))
        self.trailing_stop_points: Optional[int]       = _to_int(exit_.get("trailing_stop_points"))
        self.max_trade_duration_minutes: Optional[int] = _to_int(exit_.get("max_trade_duration_minutes"))
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

def _extract_json(text: str):
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    if start == -1:
        return None
    depth, in_string, escape_next = 0, False, False
    for i, ch in enumerate(text[start:], start):
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
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    nxt = text.find("{", i + 1)
                    return _extract_json(text[nxt:]) if nxt != -1 else None
    return None


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
    Each signal is reviewed by a local Qwen LLM before execution.
    Open positions are monitored by an AI risk manager every N ticks.
    All AI reasoning is persisted in the thought log (see /ai_thoughts).
    """

    CONFIDENCE_THRESHOLD = 0.40   # Qwen min confidence to approve a trade
    AI_REVIEW_TICKS      = 3      # Qwen reviews open positions every N ticks

    def __init__(
        self,
        atr_tolerance_multiplier: float = 1.5,
        lookback_candles: int           = 50,
        atr_period: int                 = 14,
        sl_atr_mult: float              = 1.5,
        tp_atr_mult: float              = 3.0,
        level_interaction_bars: int     = 10,
        rsi_period: int                 = 14,
        rsi_overbought: float           = 70.0,
        rsi_oversold: float             = 30.0,
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
        self.rsi_period               = rsi_period
        self.rsi_overbought           = rsi_overbought
        self.rsi_oversold             = rsi_oversold
        self.partial_close_ratio      = partial_close_ratio
        self.partial_close_rr         = partial_close_rr
        self.breakeven_buffer_pips    = breakeven_buffer_pips
        self.use_ai                   = use_ai

        self.previous_day_high: Optional[float] = None
        self.previous_day_low:  Optional[float] = None

        self._partial_closed_tickets: set  = set()
        self._filling_mode_cache: dict     = {}

        # AI components
        self._llm: Optional[LocalLLM]      = None
        self._ai_tick_counters: dict       = {}   # ticket -> int
        self._ai_breakeven_done: set       = set()
        self._ai_partial_done: set         = set()
        self._ai_peak_profit: dict         = {}   # ticket -> float pts

    # ------------------------------------------------------------------ #
    # LLM access                                                          #
    # ------------------------------------------------------------------ #

    def _get_llm(self) -> LocalLLM:
        if self._llm is None:
            self._llm = LocalLLM()
        return self._llm

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
    # MT5 helpers                                                         #
    # ------------------------------------------------------------------ #

    def _ensure_mt5(self) -> bool:
        if not self._mt5_initialized:
            import MetaTrader5 as mt5
            if mt5.initialize():
                self._mt5_initialized = True
            else:
                log.error("Failed to initialize MT5")
        return self._mt5_initialized

    def _select_symbol(self, symbol: str) -> bool:
        import MetaTrader5 as mt5
        if not mt5.symbol_select(symbol, True):
            log.error("Could not select %s", symbol)
            return False
        return True

    # ------------------------------------------------------------------ #
    # ATR & RSI                                                           #
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

    def _calculate_rsi(self, df: pd.DataFrame) -> float:
        closes = df["close"].copy()
        if len(closes) < self.rsi_period + 1:
            return 50.0
        delta    = closes.diff()
        gain     = delta.clip(lower=0)
        loss     = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / self.rsi_period, adjust=False).mean()
        rs       = avg_gain / avg_loss.replace(0, np.nan)
        rsi      = 100 - (100 / (1 + rs))
        latest   = rsi.iloc[-1]
        return float(latest) if not np.isnan(latest) else 50.0

    # ------------------------------------------------------------------ #
    # RSI Exit                                                            #
    # ------------------------------------------------------------------ #

    def check_rsi_exit(self, symbol: str,
                        df: pd.DataFrame = None) -> list:
        import MetaTrader5 as mt5
        if not self._ensure_mt5():
            return []
        if df is None:
            df = self._fetch_m15(symbol)
        if df is None:
            return []

        rsi = self._calculate_rsi(df)
        log_thought("rsi_exit", symbol, "scan",
                    f"RSI={rsi:.2f}  OB={self.rsi_overbought}  OS={self.rsi_oversold}")

        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return []

        results = []
        for pos in positions:
            should_exit = False
            if pos.type == mt5.ORDER_TYPE_BUY and rsi >= self.rsi_overbought:
                should_exit = True
                reason = f"RSI {rsi:.2f} >= OB {self.rsi_overbought}"
            elif pos.type == mt5.ORDER_TYPE_SELL and rsi <= self.rsi_oversold:
                should_exit = True
                reason = f"RSI {rsi:.2f} <= OS {self.rsi_oversold}"
            else:
                reason = ""

            if should_exit:
                log_thought("rsi_exit", symbol, "exit",
                            f"Closing #{pos.ticket}: {reason}",
                            action="close")
                result = self._close_position(pos)
                result["rsi"] = rsi
                result["exit_reason"] = reason
                results.append(result)
        return results

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
                source="rsi_exit",
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

        pip    = 0.0001 if symbol_info.digits == 5 else 0.001
        be_buf = self.breakeven_buffer_pips * pip
        results = []

        for pos in positions:
            ticket = pos.ticket
            entry  = pos.price_open
            sl     = pos.sl
            is_buy = pos.type == mt5.ORDER_TYPE_BUY

            if ticket in self._partial_closed_tickets:
                be_price = round(
                    entry + be_buf if is_buy else entry - be_buf,
                    symbol_info.digits
                )
                sl_at_be = (sl >= be_price) if is_buy else (sl <= be_price)
                if not sl_at_be:
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
        Calls Qwen to review one open position.
        Returns an action dict or None if hold.
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

        # Fetch last 5 M15 closes for context
        df = self._fetch_m15(symbol)
        last_closes = []
        if df is not None:
            last_closes = [round(float(c), 5)
                           for c in df["close"].tail(5).tolist()]

        prompt = f"""You are an AI risk manager for AI_Pro forex strategy.

POSITION:
  Symbol:        {symbol}
  Direction:     {direction}
  Entry:         {entry:.5f}
  Current price: {cur_price:.5f}
  Current SL:    {cur_sl:.5f}
  Current TP:    {cur_tp:.5f}
  Profit (pts):  {profit_pts:.1f}
  Peak profit:   {peak:.1f} pts
  ATR (M15):     {atr:.5f}
  Last 5 closes: {last_closes}

YOUR JOB: Decide what to do with the STOP LOSS to maximise profit.
Options: "hold" | "trail" | "tighten" | "close"
Rules:
- For BUY:  new_sl MUST be below current price {cur_price:.5f}
- For SELL: new_sl MUST be above current price {cur_price:.5f}
- new_sl must be BETTER than current SL {cur_sl:.5f}
- Only recommend "close" if momentum clearly gone
- If profit < 5 pts, prefer "hold"
- If price pulled back 60%+ from peak, "tighten" or "close"

Respond ONLY with JSON:
{{"action":"hold","new_sl":null,"reason":"short reason"}}"""

        log_thought("ai_risk", symbol, "review_start",
                    f"Asking Qwen about #{ticket} {direction} "
                    f"({profit_pts:+.0f}pts, peak {peak:.0f}pts)",
                    detail=f"entry={entry:.5f} price={cur_price:.5f} "
                           f"SL={cur_sl:.5f} TP={cur_tp:.5f}")
        try:
            raw    = self._get_llm().generate(prompt, max_new_tokens=100)
            parsed = _extract_json(raw)
            if not isinstance(parsed, dict):
                log_thought("ai_risk", symbol, "parse_fail",
                            f"#{ticket} — JSON parse failed, holding")
                return None

            ai_action = str(parsed.get("action", "hold")).lower()
            ai_sl     = parsed.get("new_sl")
            ai_reason = str(parsed.get("reason", ""))

            if ai_action == "close":
                log_thought("ai_risk", symbol, "close",
                            f"#{ticket} — AI says CLOSE: {ai_reason[:80]}",
                            action="close")
                return {"action": "close", "ticket": ticket,
                        "reason": f"AI: {ai_reason}"}

            if ai_action in ("trail", "tighten") and ai_sl is not None:
                try:
                    new_sl = round(float(ai_sl), digits)
                    wrong_side = (
                        (is_buy  and new_sl >= cur_price) or
                        (not is_buy and new_sl <= cur_price)
                    )
                    not_better = (
                        (is_buy  and new_sl <= cur_sl and cur_sl != 0) or
                        (not is_buy and new_sl >= cur_sl and cur_sl != 0)
                    )
                    if wrong_side or not_better:
                        return None
                    log_thought("ai_risk", symbol, "move_sl",
                                f"#{ticket} {direction} — AI {ai_action}: "
                                f"SL -> {new_sl:.5f} | {ai_reason[:60]}",
                                action="move_sl")
                    return {"action": "move_sl", "ticket": ticket,
                            "new_sl": new_sl,
                            "reason": f"AI {ai_action}: {ai_reason}"}
                except (TypeError, ValueError):
                    pass

            log_thought("ai_risk", symbol, "hold",
                        f"#{ticket} holding — {ai_reason[:80]}",
                        action="hold")
            return None

        except Exception as exc:
            log_thought("ai_risk", symbol, "error",
                        f"#{ticket} AI error: {exc}", action="hold")
            return None

    def run_ai_risk_manager(self, symbol: str) -> list:
        """
        Called each cycle BEFORE entry signal generation.
        Runs rule-based checks (BE, trail) AND Qwen review every N ticks.
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
            is_buy = pos.type == mt5.ORDER_TYPE_BUY
            entry  = float(pos.price_open)
            cur_sl = float(pos.sl or 0)
            cur_tp = float(pos.tp or 0)
            tick_  = mt5.symbol_info_tick(symbol)
            if tick_ is None:
                continue
            price  = float(tick_.bid) if is_buy else float(tick_.ask)
            profit_pts = ((price - entry) / point if is_buy
                          else (entry - price) / point)

            # -- Rule-based: max duration
            if rules and rules.max_trade_duration_minutes:
                open_time = float(getattr(pos, "time", time.time()))
                elapsed   = (time.time() - open_time) / 60.0
                if elapsed >= rules.max_trade_duration_minutes:
                    log_thought("rule_risk", symbol, "max_duration",
                                f"#{ticket} max duration hit ({elapsed:.0f}m)",
                                action="close")
                    r = self._close_position(pos)
                    results.append(r)
                    continue

            # -- Rule-based: breakeven
            if ticket not in self._ai_breakeven_done and rules:
                be_pts = int(getattr(rules, "breakeven_trigger_points", 0) or 0)
                if be_pts and profit_pts >= be_pts:
                    be_sl = round(
                        entry + be_pts * point * 0.1 if is_buy
                        else entry - be_pts * point * 0.1,
                        digits
                    )
                    if _sl_is_better(be_sl, cur_sl, is_buy):
                        self._ai_breakeven_done.add(ticket)
                        log_thought("rule_risk", symbol, "breakeven",
                                    f"#{ticket} — SL to BE {be_sl:.5f}",
                                    action="move_sl")
                        updated = mt5.positions_get(ticket=ticket)
                        p2 = updated[0] if updated else pos
                        results.append(self._modify_sl(p2, be_sl))

            # -- Rule-based: trailing stop
            if rules:
                trail_pts = int(getattr(rules, "trailing_stop_points", 0) or 0)
                if trail_pts and profit_pts > 0:
                    trail_sl = round(
                        price - trail_pts * point if is_buy
                        else price + trail_pts * point,
                        digits
                    )
                    if _sl_is_better(trail_sl, cur_sl, is_buy):
                        log_thought("rule_risk", symbol, "trail",
                                    f"#{ticket} — trail SL {trail_sl:.5f}",
                                    action="move_sl")
                        updated = mt5.positions_get(ticket=ticket)
                        p2 = updated[0] if updated else pos
                        results.append(self._modify_sl(p2, trail_sl))

            # -- AI review every N ticks
            if self.use_ai:
                cnt = self._ai_tick_counters.get(ticket, 0) + 1
                self._ai_tick_counters[ticket] = cnt
                if cnt >= self.AI_REVIEW_TICKS:
                    self._ai_tick_counters[ticket] = 0
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

        # Clean up state for closed positions
        active = {int(p.ticket) for p in (mt5.positions_get(symbol=symbol) or [])}
        for t in list(self._ai_tick_counters):
            if t not in active:
                del self._ai_tick_counters[t]
                self._ai_breakeven_done.discard(t)
                self._ai_partial_done.discard(t)
                self._ai_peak_profit.pop(t, None)

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
    # Level interaction & momentum                                         #
    # ------------------------------------------------------------------ #

    def _level_interacted(self, df: pd.DataFrame, level: float,
                           atr: float, bar_multiplier: int = 1) -> bool:
        zone    = atr * self.atr_tolerance_multiplier
        n_bars  = self.level_interaction_bars * bar_multiplier
        recent  = df.tail(n_bars)
        touched = ((recent["low"]  <= level + zone) &
                   (recent["high"] >= level - zone))
        return bool(touched.any())

    def _momentum_aligned(self, df: pd.DataFrame, direction: str) -> bool:
        closes = df["close"].tail(4).values
        if len(closes) < 4:
            return True
        if direction == "buy":
            return closes[-1] > min(closes[-4], closes[-3])
        return closes[-1] < max(closes[-4], closes[-3])

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
    # AI Signal Review (Qwen approves/rejects AI Pro signal)              #
    # ------------------------------------------------------------------ #

    def _ai_review_signal(self, symbol: str, signal: dict,
                           df: pd.DataFrame) -> dict:
        """
        Sends the AI Pro signal to Qwen for approval.
        Returns {"approve": bool, "reason": str, "confidence": float}.
        """
        records      = _load_memory()
        history      = _memory_summary(records)
        last_closes  = [round(float(c), 5)
                        for c in df["close"].tail(10).tolist()]
        env          = signal.get("signal_source", "unknown")
        direction    = signal["signal"]
        entry        = signal.get("entry_price", 0)
        sl           = signal.get("stop_loss",   0)
        tp           = signal.get("take_profit", 0)
        confidence   = signal.get("confidence",  0)
        reason_txt   = signal.get("reason",      "")
        pdh          = signal.get("previous_day_high", 0)
        pdl          = signal.get("previous_day_low",  0)
        rsi          = signal.get("rsi", 50)
        atr          = signal.get("atr", 0)

        prompt = f"""You are reviewing a AI_Pro trade signal.

SIGNAL:
  Symbol:      {symbol}
  Direction:   {direction}
  Environment: {env}
  Confidence:  {confidence}%
  Reason:      {reason_txt[:200]}

LEVELS:
  Entry:           {entry:.5f}
  Stop Loss:       {sl:.5f}
  Take Profit:     {tp:.5f}
  Prev Day High:   {pdh:.5f}
  Prev Day Low:    {pdl:.5f}
  ATR:             {atr}
  RSI:             {rsi}
  Last 10 closes:  {last_closes}

RECENT TRADE HISTORY: {history}

Should this AI Pro signal be taken?
Consider: is price genuinely near the key level? Is RSI extreme or neutral?
Does the last-close momentum match the direction?

Reply ONLY with JSON:
{{"approve":true,"reason":"brief reason","confidence":0.0}}"""

        log_thought("ai_entry", symbol, "review_start",
                    f"Sending {direction} [{env}] to Qwen for review",
                    detail=reason_txt[:120], action=direction.lower())
        try:
            raw    = self._get_llm().generate(prompt, max_new_tokens=120)
            parsed = _extract_json(raw)
            if not isinstance(parsed, dict):
                log_thought("ai_entry", symbol, "parse_fail",
                            "Qwen parse failed — defaulting approve",
                            action=direction.lower())
                return {"approve": True, "reason": "Parse failed — default approve",
                        "confidence": 0.6}
            approve    = bool(parsed.get("approve", True))
            ai_reason  = str(parsed.get("reason", ""))
            ai_conf    = float(parsed.get("confidence", 0.6))

            if approve and ai_conf < self.CONFIDENCE_THRESHOLD:
                approve   = False
                ai_reason = f"Confidence too low ({ai_conf:.0%})"

            log_thought(
                "ai_entry", symbol, "verdict",
                f"Qwen {'APPROVED' if approve else 'REJECTED'}: {ai_reason[:80]}",
                detail=f"{direction} [{env}] conf={ai_conf:.2f}",
                action=direction.lower() if approve else "hold",
                confidence=ai_conf,
            )
            return {"approve": approve, "reason": ai_reason,
                    "confidence": ai_conf}

        except Exception as exc:
            log_thought("ai_entry", symbol, "error",
                        f"Qwen error: {exc} — default approve")
            return {"approve": True, "reason": f"Qwen error — default approve",
                    "confidence": 0.6}

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

        atr           = self._calculate_atr(df)
        rsi           = self._calculate_rsi(df)
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
            "rsi":               round(rsi, 2),
            "dynamic_zone_pips": round(zone / 0.0001, 1),
            "ai_approved":       None,
            "ai_reason":         None,
            "ai_confidence":     None,
        }

        def buy_levels():
            sl = round(current_price - atr * self.sl_atr_mult, 5)
            tp = round(current_price + atr * self.tp_atr_mult, 5)
            return sl, tp

        def sell_levels():
            sl = round(current_price + atr * self.sl_atr_mult, 5)
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
                and self._momentum_aligned(df, "buy")):
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
                and self._momentum_aligned(df, "sell")):
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
                and self._momentum_aligned(df, "buy")):
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
                and self._momentum_aligned(df, "sell")):
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
            cr = choch_data.get("reason", "") if choch_data else ""
            co = cont_data.get("reason",  "") if cont_data  else ""
            signal["reason"] = (
                f"No environment active. "
                f"near_high={near_high}(choch={near_high_choch}) "
                f"near_low={near_low}(choch={near_low_choch}) | {cr} | {co}"
            )

        log_thought(
            "ai_pro_signal", symbol, "signal",
            f"Signal: {signal['signal']} [{signal.get('signal_source','—')}] "
            f"conf={signal['confidence']}% "
            f"AI={'✓' if signal.get('ai_approved') else '✗' if signal.get('ai_approved') is False else '—'}",
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
            "rsi": None, "atr": None,
            "ai_approved": None, "ai_reason": None, "ai_confidence": None,
        }

    # ------------------------------------------------------------------ #
    # Trade execution                                                       #
    # ------------------------------------------------------------------ #

    def execute_trade(self, symbol: str, signal: dict,
                       lot_size: float = 0.01) -> dict:
        import MetaTrader5 as mt5
        if not self._ensure_mt5():
            return {"success": False, "message": "MT5 not initialized"}
        if signal["signal"] == "neutral":
            return {"success": False, "message": "Neutral — no trade"}
        if not signal.get("stop_loss") or not signal.get("take_profit"):
            return {"success": False, "message": "SL/TP missing"}
        if not self._select_symbol(symbol):
            return {"success": False, "message": f"Cannot select {symbol}"}

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
        tp         = round(signal["take_profit"], digits)
        filling    = self._get_filling_mode(symbol)

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
            "comment":      (f"AI_Pro {signal.get('signal_source','?')} "
                             f"{signal['signal']}"),
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": filling,
        }
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            log_thought(
                "execution", symbol, "order_placed",
                f"Order placed #{result.order} {signal['signal']} "
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
        log_thought("execution", symbol, "order_failed",
                    f"Order FAILED: {comment} (retcode {retcode})",
                    action="hold")
        return {"success": False,
                "message": f"Order failed: {comment} (retcode {retcode})"}

    # ------------------------------------------------------------------ #
    # Runner                                                               #
    # ------------------------------------------------------------------ #

    def run_strategy(self, symbol: str, auto_trade: bool = False,
                     lot_size: float = 0.01) -> dict:
        """
        Full cycle:
          1. RSI exit scan
          2. Partial close + breakeven (AI Pro native)
          3. AI risk manager (rule-based + Qwen)
          4. AI Pro signal generation
          5. AI signal review (Qwen gate)
          6. Auto-execution if confidence >= 70 AND AI approved
        """
        log.info("=" * 65)
        log.info("AI Pro — %s", symbol)
        log.info("=" * 65)

        df = self._fetch_m15(symbol)

        # Step 1 — RSI exit
        exit_results = self.check_rsi_exit(symbol, df)

        # Step 2 — AI Pro partial close / breakeven
        partial_results = self.check_partial_close_and_breakeven(symbol)

        # Step 3 — AI risk manager
        ai_risk_results = self.run_ai_risk_manager(symbol)

        # Step 4+5 — Signal + AI review
        signal = self.generate_trade_signal(symbol)

        log.info("Signal      : %s", signal["signal"])
        log.info("Environment : %s", signal.get("signal_source") or "—")
        log.info("Confidence  : %s%%", signal["confidence"])
        log.info("AI approved : %s", signal.get("ai_approved"))
        log.info("AI reason   : %s", signal.get("ai_reason"))
        log.info("RSI         : %s", signal.get("rsi"))
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
                    and signal["confidence"] >= 70
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
            elif auto_trade:
                log.info("AUTO TRADE skipped — confidence %s%% < 70",
                         signal["confidence"])

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
    today_pnl = _todays_pnl(records)
    if (rules.daily_loss_limit_usd is not None
            and today_pnl <= rules.daily_loss_limit_usd):
        return f"Daily loss limit hit (${today_pnl:.2f})"
    if (rules.daily_profit_target_usd is not None
            and today_pnl >= rules.daily_profit_target_usd):
        return f"Daily profit target hit (${today_pnl:.2f})"
    return None


# ============================================================ #
# SECTION 10 — BOT (multi-symbol trading loop)                 #
# ============================================================ #

class Bot:
    def __init__(
        self,
        symbols:   list  = None,
        volume:    float = 0.01,
        poll_secs: float = 30.0,
        dry_run:   bool  = False,
        auto_trade: bool = True,
        use_ai:    bool  = True,
        **strategy_kwargs,
    ) -> None:
        self.symbols    = [s.strip().upper() for s in (symbols or ["EURUSD"])]
        self.volume     = float(volume)
        self.poll_secs  = float(poll_secs)
        self.dry_run    = bool(dry_run)
        self.auto_trade = bool(auto_trade)

        self._strategy  = AI_Pro(use_ai=use_ai, **strategy_kwargs)
        self._running   = False
        self._stop      = threading.Event()

        self._results:     dict = {}
        self._results_lock = threading.Lock()
        self._positions:   list = []
        self._positions_lock = threading.Lock()

    def run(self) -> None:
        log.info("Bot starting — symbols=%s  poll=%ss  dry_run=%s",
                 self.symbols, self.poll_secs, self.dry_run)
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

    def _loop(self) -> None:
        while self._running:
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
                log_thought("preflight", symbol, "blocked", block, action="hold")
                log.info("[%s] Blocked: %s", symbol, block)
                return

            max_pos = rules.max_open_positions
            if len(positions) >= max_pos and not self.dry_run:
                log.info("[%s] At max positions (%d)", symbol, max_pos)
                # Still run risk manager even when at max positions
                self._strategy.run_ai_risk_manager(symbol)
                return

        result = self._strategy.run_strategy(
            symbol=symbol,
            auto_trade=self.auto_trade and not self.dry_run,
            lot_size=self.volume,
        )

        if self.dry_run and result["signal"]["signal"] != "neutral":
            sig = result["signal"]
            log.warning("[DRY RUN] %s %s [%s] conf=%d%% AI=%s",
                        symbol, sig["signal"],
                        sig.get("signal_source", "?"),
                        sig["confidence"],
                        sig.get("ai_approved"))

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

from flask import Flask, request, jsonify, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

_bot_lock:   threading.Lock          = threading.Lock()
_bot:        Optional[Bot]           = None
_bot_thread: Optional[threading.Thread] = None

# ---- Embedded Dashboard HTML -------------------------------- #

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Pro</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@400;700;800&display=swap');

  :root {
    --bg:       #0b0d11;
    --surface:  #111318;
    --border:   #1e2230;
    --muted:    #3a3f52;
    --text:     #cdd5e0;
    --dim:      #6b7391;
    --accent:   #4fc3f7;
    --buy:      #26d97f;
    --sell:     #f55e6e;
    --warn:     #f5a623;
    --ai:       #bb86fc;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    min-height: 100vh;
  }

  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 28px;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
  }

  .logo {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 22px;
    letter-spacing: -0.5px;
    color: var(--accent);
  }
  .logo span { color: var(--ai); }

  .status-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--muted); display: inline-block;
    margin-right: 8px; transition: background 0.3s;
  }
  .status-dot.live { background: var(--buy);
    box-shadow: 0 0 8px var(--buy); }

  .controls { display: flex; gap: 10px; align-items: center; }

  button {
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--text);
    padding: 7px 16px;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    cursor: pointer;
    transition: all 0.15s;
  }
  button:hover { border-color: var(--accent); color: var(--accent); }
  button.primary {
    background: var(--accent);
    border-color: var(--accent);
    color: #000;
    font-weight: 600;
  }
  button.primary:hover { opacity: 0.85; }
  button.danger { border-color: var(--sell); color: var(--sell); }
  button.danger:hover { background: var(--sell); color: #fff; }

  main {
    display: grid;
    grid-template-columns: 300px 1fr;
    grid-template-rows: auto 1fr;
    gap: 1px;
    background: var(--border);
    height: calc(100vh - 61px);
  }

  .panel {
    background: var(--bg);
    padding: 20px;
    overflow-y: auto;
  }

  .panel-title {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 11px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--dim);
    margin-bottom: 16px;
  }

  /* Left sidebar — config */
  .config-panel { grid-row: 1 / 3; }

  label {
    display: block;
    color: var(--dim);
    font-size: 11px;
    margin-bottom: 4px;
    margin-top: 14px;
  }

  input, select {
    width: 100%;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 5px;
    color: var(--text);
    padding: 7px 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
  }

  input:focus, select:focus {
    outline: none;
    border-color: var(--accent);
  }

  .checkbox-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 14px;
  }
  .checkbox-row input { width: auto; }

  .divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 20px 0;
  }

  /* Signal card */
  .signal-panel { border-bottom: 1px solid var(--border); }

  .signal-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 14px;
  }

  .sig-field { display: flex; flex-direction: column; gap: 4px; }
  .sig-label { font-size: 10px; color: var(--dim); letter-spacing: 0.8px; text-transform: uppercase; }
  .sig-value { font-size: 14px; font-weight: 600; }

  .tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 700;
  }
  .tag.buy  { background: rgba(38,217,127,0.15); color: var(--buy); }
  .tag.sell { background: rgba(245, 94,110,0.15); color: var(--sell); }
  .tag.neutral { background: var(--surface); color: var(--dim); border: 1px solid var(--border); }
  .tag.approved { background: rgba(187,134,252,0.15); color: var(--ai); }
  .tag.rejected { background: rgba(245, 94,110,0.15); color: var(--sell); }

  /* Thoughts log */
  .thoughts-panel { position: relative; }

  #thoughts {
    display: flex;
    flex-direction: column;
    gap: 6px;
    max-height: calc(100% - 40px);
    overflow-y: auto;
  }

  .thought {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--muted);
    border-radius: 0 6px 6px 0;
    padding: 9px 12px;
    transition: border-color 0.2s;
    animation: slide-in 0.2s ease;
  }
  @keyframes slide-in {
    from { opacity: 0; transform: translateY(-4px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  .thought[data-source="ai_pro_signal"] { border-left-color: var(--accent); }
  .thought[data-source="ai_entry"]      { border-left-color: var(--ai); }
  .thought[data-source="ai_risk"]       { border-left-color: var(--warn); }
  .thought[data-source="rsi_exit"]      { border-left-color: var(--sell); }
  .thought[data-source="partial_close"] { border-left-color: var(--buy); }
  .thought[data-source="execution"]     { border-left-color: var(--buy); }
  .thought[data-source="rule_risk"]     { border-left-color: var(--warn); }
  .thought[data-source="memory"]        { border-left-color: var(--dim); }

  .thought-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
  }

  .thought-meta {
    display: flex;
    gap: 8px;
    align-items: center;
  }

  .thought-source {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: var(--accent);
  }

  .thought[data-source="ai_entry"] .thought-source,
  .thought[data-source="ai_risk"]  .thought-source { color: var(--ai); }

  .thought-stage {
    font-size: 10px;
    color: var(--dim);
  }

  .thought-ts {
    font-size: 10px;
    color: var(--muted);
  }

  .thought-summary { font-size: 12px; color: var(--text); }
  .thought-detail  { font-size: 11px; color: var(--dim); margin-top: 4px; }

  /* Positions table */
  .pos-table { width: 100%; border-collapse: collapse; }
  .pos-table th {
    text-align: left;
    font-size: 10px;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: var(--dim);
    padding: 0 8px 8px;
  }
  .pos-table td { padding: 8px; border-top: 1px solid var(--border); }
  .pos-table tr:hover td { background: var(--surface); }

  .pnl-pos { color: var(--buy); }
  .pnl-neg { color: var(--sell); }

  .empty { color: var(--muted); font-size: 12px; padding: 20px 0; }

  /* History */
  #history-list { display: flex; flex-direction: column; gap: 6px; }
  .hist-item {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px 14px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .hist-left { display: flex; flex-direction: column; gap: 3px; }
  .hist-sym { font-weight: 600; }
  .hist-sub { font-size: 11px; color: var(--dim); }
  .hist-pnl { font-weight: 700; font-size: 14px; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--muted); border-radius: 2px; }
</style>
</head>
<body>

<header>
  <div class="logo">AI<span> Pro</span></div>
  <div style="display:flex;align-items:center;gap:12px">
    <span class="status-dot" id="dot"></span>
    <span id="status-text" style="color:var(--dim)">idle</span>
  </div>
  <div class="controls">
    <button onclick="showTab('signals')">Signals</button>
    <button onclick="showTab('thoughts')">AI Thoughts</button>
    <button onclick="showTab('positions')">Positions</button>
    <button onclick="showTab('history')">History</button>
    <button id="start-btn" class="primary" onclick="startBot()">Start Bot</button>
    <button id="stop-btn" class="danger" style="display:none" onclick="stopBot()">Stop</button>
  </div>
</header>

<main>
  <!-- Config sidebar -->
  <div class="panel config-panel">
    <div class="panel-title">Configuration</div>

    <label>Symbols (comma-separated)</label>
    <input id="cfg-symbols" value="EURUSD" />

    <label>Lot Size</label>
    <input id="cfg-volume" type="number" value="0.01" step="0.01" />

    <label>Poll Interval (s)</label>
    <input id="cfg-poll" type="number" value="30" />

    <div class="checkbox-row">
      <input type="checkbox" id="cfg-dry" />
      <label style="margin:0">Dry Run (no real trades)</label>
    </div>
    <div class="checkbox-row">
      <input type="checkbox" id="cfg-ai" checked />
      <label style="margin:0">Enable AI Review (Qwen)</label>
    </div>
    <div class="checkbox-row">
      <input type="checkbox" id="cfg-auto" checked />
      <label style="margin:0">Auto Trade</label>
    </div>

    <hr class="divider">

    <div class="panel-title">Strategy Params</div>

    <label>ATR Tolerance Mult</label>
    <input id="cfg-atr-mult" type="number" value="1.5" step="0.1" />

    <label>SL ATR Mult</label>
    <input id="cfg-sl-mult" type="number" value="1.5" step="0.1" />

    <label>TP ATR Mult</label>
    <input id="cfg-tp-mult" type="number" value="3.0" step="0.1" />

    <label>RSI Overbought</label>
    <input id="cfg-rsi-ob" type="number" value="70" />

    <label>RSI Oversold</label>
    <input id="cfg-rsi-os" type="number" value="30" />

    <label>Partial Close RR</label>
    <input id="cfg-pc-rr" type="number" value="1.0" step="0.1" />

    <label>BE Buffer (pips)</label>
    <input id="cfg-be-buf" type="number" value="1.0" step="0.5" />
  </div>

  <!-- Right top: signals -->
  <div class="panel signal-panel" id="tab-signals">
    <div class="panel-title">Latest Signal</div>
    <div id="signal-cards">
      <div class="empty">Waiting for bot to start...</div>
    </div>
  </div>

  <!-- Right bottom: tabbed content -->
  <div class="panel thoughts-panel" id="tab-thoughts" style="display:none">
    <div class="panel-title" style="display:flex;justify-content:space-between">
      AI Thought Log
      <button onclick="clearThoughts()" style="padding:3px 10px;font-size:11px">Clear</button>
    </div>
    <div id="thoughts"></div>
  </div>

  <div class="panel" id="tab-positions" style="display:none">
    <div class="panel-title">Open Positions</div>
    <div id="positions-content">
      <div class="empty">No open positions.</div>
    </div>
  </div>

  <div class="panel" id="tab-history" style="display:none">
    <div class="panel-title">Trade History</div>
    <div id="history-list"><div class="empty">No trades recorded yet.</div></div>
  </div>
</main>

<script>
let since = null;
let pollTimer = null;
let activeTab = 'signals';

function showTab(tab) {
  ['signals','thoughts','positions','history'].forEach(t => {
    const el = document.getElementById('tab-' + t);
    if (el) el.style.display = (t === tab) ? 'block' : 'none';
  });
  activeTab = tab;
  if (tab === 'history') loadHistory();
}

async function startBot() {
  const symbols = document.getElementById('cfg-symbols').value
    .split(',').map(s => s.trim()).filter(Boolean);
  const body = {
    symbols:    symbols,
    volume:     parseFloat(document.getElementById('cfg-volume').value),
    poll_secs:  parseFloat(document.getElementById('cfg-poll').value),
    dry_run:    document.getElementById('cfg-dry').checked,
    use_ai:     document.getElementById('cfg-ai').checked,
    auto_trade: document.getElementById('cfg-auto').checked,
    strategy: {
      atr_tolerance_multiplier: parseFloat(document.getElementById('cfg-atr-mult').value),
      sl_atr_mult:              parseFloat(document.getElementById('cfg-sl-mult').value),
      tp_atr_mult:              parseFloat(document.getElementById('cfg-tp-mult').value),
      rsi_overbought:           parseFloat(document.getElementById('cfg-rsi-ob').value),
      rsi_oversold:             parseFloat(document.getElementById('cfg-rsi-os').value),
      partial_close_rr:         parseFloat(document.getElementById('cfg-pc-rr').value),
      breakeven_buffer_pips:    parseFloat(document.getElementById('cfg-be-buf').value),
    }
  };
  const r = await fetch('/bot/start', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body)
  });
  const data = await r.json();
  if (data.ok) {
    document.getElementById('start-btn').style.display = 'none';
    document.getElementById('stop-btn').style.display = 'inline-block';
    startPolling();
  } else {
    alert('Error: ' + (data.error || JSON.stringify(data)));
  }
}

async function stopBot() {
  await fetch('/bot/stop', { method: 'POST' });
  document.getElementById('start-btn').style.display = 'inline-block';
  document.getElementById('stop-btn').style.display = 'none';
  stopPolling();
}

function startPolling() {
  if (pollTimer) clearInterval(pollTimer);
  poll();
  pollTimer = setInterval(poll, 3000);
}

function stopPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = null;
}

async function poll() {
  try {
    const [statusRes, thoughtsRes] = await Promise.all([
      fetch('/bot/status'),
      fetch('/bot/ai_thoughts' + (since ? '?since=' + since : '')),
    ]);
    const status   = await statusRes.json();
    const thoughts = await thoughtsRes.json();

    updateStatus(status);
    if (thoughts.ok && thoughts.thoughts.length > 0) {
      appendThoughts(thoughts.thoughts);
      since = thoughts.thoughts[thoughts.thoughts.length - 1].ts;
    }
    if (activeTab === 'signals') updateSignals(status);
    if (activeTab === 'positions') updatePositions(status.open_positions || []);
  } catch(e) { console.error(e); }
}

function updateStatus(status) {
  const dot  = document.getElementById('dot');
  const txt  = document.getElementById('status-text');
  const live = status.running;
  dot.className = 'status-dot' + (live ? ' live' : '');
  txt.textContent = live ? 'LIVE' : 'idle';
  txt.style.color = live ? 'var(--buy)' : 'var(--dim)';
  if (!live) {
    document.getElementById('start-btn').style.display = 'inline-block';
    document.getElementById('stop-btn').style.display = 'none';
    stopPolling();
  }
}

function updateSignals(status) {
  const all = status.all_decisions || {};
  const container = document.getElementById('signal-cards');
  if (Object.keys(all).length === 0) {
    container.innerHTML = '<div class="empty">Waiting for first signal...</div>';
    return;
  }
  container.innerHTML = Object.entries(all).map(([sym, d]) => {
    const dir   = (d.action || 'neutral').toUpperCase();
    const cls   = dir === 'BUY' ? 'buy' : dir === 'SELL' ? 'sell' : 'neutral';
    const aiCls = d.approve ? 'approved' : 'rejected';
    const aiTxt = d.approve ? '✓ AI Approved' : d.approve === false ? '✗ AI Rejected' : '— No AI';
    return `
    <div style="margin-bottom:12px">
      <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:16px;margin-bottom:10px">${sym}</div>
      <div class="signal-card">
        <div class="sig-field">
          <span class="sig-label">Direction</span>
          <span class="sig-value"><span class="tag ${cls}">${dir}</span></span>
        </div>
        <div class="sig-field">
          <span class="sig-label">Confidence</span>
          <span class="sig-value" style="color:var(--accent)">${((d.confidence||0)*100).toFixed(0)}%</span>
        </div>
        <div class="sig-field">
          <span class="sig-label">AI Verdict</span>
          <span class="sig-value"><span class="tag ${aiCls}">${aiTxt}</span></span>
        </div>
        <div class="sig-field" style="grid-column:1/-1">
          <span class="sig-label">Reason</span>
          <span class="sig-value" style="font-size:11px;color:var(--dim);font-weight:400">${(d.reason||'—').substring(0,180)}</span>
        </div>
        ${d.sl ? `
        <div class="sig-field">
          <span class="sig-label">Entry</span>
          <span class="sig-value" style="font-size:12px">${(d.action||'') !== 'hold' ? '~market' : '—'}</span>
        </div>
        <div class="sig-field">
          <span class="sig-label">SL</span>
          <span class="sig-value" style="font-size:12px;color:var(--sell)">${d.sl}</span>
        </div>
        <div class="sig-field">
          <span class="sig-label">TP</span>
          <span class="sig-value" style="font-size:12px;color:var(--buy)">${d.tp}</span>
        </div>` : ''}
      </div>
    </div>`;
  }).join('');
}

function appendThoughts(items) {
  const container = document.getElementById('thoughts');
  items.forEach(t => {
    const div = document.createElement('div');
    div.className = 'thought';
    div.dataset.source = t.source;
    const ts = new Date(t.ts).toLocaleTimeString();
    const conf = t.confidence != null
      ? `<span style="color:var(--ai);margin-left:6px">${(t.confidence*100).toFixed(0)}%</span>`
      : '';
    div.innerHTML = `
      <div class="thought-header">
        <div class="thought-meta">
          <span class="thought-source">${t.source}</span>
          <span class="thought-stage">${t.stage}</span>
          ${conf}
        </div>
        <span class="thought-ts">${ts}</span>
      </div>
      <div class="thought-summary">${t.summary}</div>
      ${t.detail ? `<div class="thought-detail">${t.detail.substring(0,200)}</div>` : ''}
    `;
    container.insertBefore(div, container.firstChild);
  });
}

async function clearThoughts() {
  await fetch('/bot/thoughts/clear', { method: 'POST' });
  document.getElementById('thoughts').innerHTML = '';
  since = null;
}

function updatePositions(positions) {
  const el = document.getElementById('positions-content');
  if (!positions.length) {
    el.innerHTML = '<div class="empty">No open positions.</div>';
    return;
  }
  el.innerHTML = `<table class="pos-table">
    <thead><tr>
      <th>Ticket</th><th>Symbol</th><th>Dir</th>
      <th>Vol</th><th>Entry</th><th>Current</th>
      <th>SL</th><th>TP</th><th>P&L</th>
    </tr></thead>
    <tbody>
    ${positions.map(p => {
      const pnlCls = p.pnl > 0 ? 'pnl-pos' : p.pnl < 0 ? 'pnl-neg' : '';
      const dirCls = p.direction === 'BUY' ? 'buy' : 'sell';
      return `<tr>
        <td>${p.ticket}</td>
        <td><strong>${p.symbol}</strong></td>
        <td><span class="tag ${dirCls}">${p.direction}</span></td>
        <td>${p.volume}</td>
        <td>${p.entry}</td>
        <td>${p.current || '—'}</td>
        <td style="color:var(--sell)">${p.sl || '—'}</td>
        <td style="color:var(--buy)">${p.tp || '—'}</td>
        <td class="${pnlCls}">${p.pnl > 0 ? '+' : ''}${p.pnl}</td>
      </tr>`;
    }).join('')}
    </tbody>
  </table>`;
}

async function loadHistory() {
  const r = await fetch('/bot/history');
  const data = await r.json();
  const trades = (data.trades || []).slice().reverse();
  const el = document.getElementById('history-list');
  if (!trades.length) {
    el.innerHTML = '<div class="empty">No trades recorded yet.</div>';
    return;
  }
  el.innerHTML = trades.map(t => {
    const pnlCls = t.pnl > 0 ? 'pnl-pos' : t.pnl < 0 ? 'pnl-neg' : '';
    const ts = new Date(t.ts).toLocaleString();
    return `<div class="hist-item">
      <div class="hist-left">
        <span class="hist-sym">${t.symbol} <span class="tag ${t.direction==='BUY'?'buy':'sell'}" style="font-size:10px">${t.direction}</span></span>
        <span class="hist-sub">${t.source || ''} · ${ts}</span>
        <span class="hist-sub">Entry ${t.entry} → Exit ${t.exit} · Vol ${t.volume}</span>
      </div>
      <span class="hist-pnl ${pnlCls}">${t.pnl > 0 ? '+' : ''}${t.pnl}</span>
    </div>`;
  }).join('');
}

// Auto-check status on load
(async () => {
  try {
    const r = await fetch('/bot/status');
    const s = await r.json();
    if (s.running) {
      document.getElementById('start-btn').style.display = 'none';
      document.getElementById('stop-btn').style.display = 'inline-block';
      startPolling();
    }
  } catch(e) {}
})();
</script>
</body>
</html>"""


# ---- REST API ----------------------------------------------- #

def _get_bot():
    with _bot_lock:
        return _bot, _bot_thread


def _run_bot_thread(bot: Bot) -> None:
    global _bot, _bot_thread
    try:
        bot.run()
    except Exception as exc:
        log.exception("Bot crashed: %s", exc)
    finally:
        with _bot_lock:
            _bot        = None
            _bot_thread = None


@app.route("/")
def index():
    return Response(_DASHBOARD_HTML, mimetype="text/html")


@app.route("/health")
def health():
    return jsonify({"ok": True, "version": "AI_Pro"})


@app.route("/bot/status")
def bot_status():
    bot, thread = _get_bot()
    running = bool(thread and thread.is_alive())
    return jsonify({
        "running": running,
        "bot": {
            "symbols":    bot.symbols,
            "volume":     bot.volume,
            "poll_secs":  bot.poll_secs,
            "dry_run":    bot.dry_run,
            "auto_trade": bot.auto_trade,
        } if bot else None,
        "all_decisions":  {
            sym: {
                "action":     r["signal"]["signal"],
                "approve":    r["signal"].get("ai_approved"),
                "reason":     r["signal"]["reason"],
                "sl":         r["signal"].get("stop_loss"),
                "tp":         r["signal"].get("take_profit"),
                "confidence": r["signal"]["confidence"] / 100,
                "ai_reason":  r["signal"].get("ai_reason"),
            }
            for sym, r in (bot.latest_results().items() if bot else {})
        },
        "open_positions": bot.open_positions() if bot else [],
    })


@app.route("/bot/ai_thoughts")
def bot_ai_thoughts():
    try:
        since = request.args.get("since")
        limit = min(int(request.args.get("limit", 60)), 120)
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
    global _bot, _bot_thread
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
        volume    = float(data.get("volume", 0.01))
        poll_secs = float(data.get("poll_secs", 30.0))
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400

    dry_run    = bool(data.get("dry_run",    False))
    auto_trade = bool(data.get("auto_trade", True))
    use_ai     = bool(data.get("use_ai",     True))
    strategy   = data.get("strategy", {}) or {}

    clear_thoughts()

    new_bot = Bot(
        symbols=symbols,
        volume=volume,
        poll_secs=poll_secs,
        dry_run=dry_run,
        auto_trade=auto_trade,
        use_ai=use_ai,
        atr_tolerance_multiplier = float(strategy.get("atr_tolerance_multiplier", 1.5)),
        sl_atr_mult              = float(strategy.get("sl_atr_mult",              1.5)),
        tp_atr_mult              = float(strategy.get("tp_atr_mult",              3.0)),
        rsi_overbought           = float(strategy.get("rsi_overbought",           70.0)),
        rsi_oversold             = float(strategy.get("rsi_oversold",             30.0)),
        partial_close_rr         = float(strategy.get("partial_close_rr",         1.0)),
        breakeven_buffer_pips    = float(strategy.get("breakeven_buffer_pips",    1.0)),
    )

    t = threading.Thread(target=_run_bot_thread, args=(new_bot,), daemon=True)
    with _bot_lock:
        _bot        = new_bot
        _bot_thread = t
    t.start()
    return jsonify({"ok": True, "symbols": symbols, "dry_run": dry_run,
                    "use_ai": use_ai})


@app.route("/bot/stop", methods=["POST"])
def bot_stop():
    bot, _ = _get_bot()
    if bot:
        bot.stop()
    return jsonify({"ok": True})


@app.route("/bot/history")
def bot_history():
    return jsonify({"trades": _load_memory()})


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
        # Headless bot mode — no Flask
        bot = Bot(
            symbols=[["EURUSD"],["USDJPY"],["GBPUSD"],["AUDUSD"],["USDCAD"],["USDCHF"]],
            volume=0.01,
            poll_secs=30,
            dry_run=True,   # ← change to False for live trading
            auto_trade=True,
            use_ai=True,
        )
        bot.run()
    else:
        print("=" * 55)
        print("  AI Pro Dashboard → http://localhost:5000")
        print("  API:  /bot/start  /bot/stop  /bot/status")
        print("  Test: /bot/signal/EURUSD")
        print("=" * 55)
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
