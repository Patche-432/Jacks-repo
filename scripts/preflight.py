"""
preflight.py -- Validate the agent stack before the Fortis bot starts.

Run from the repo root:
    python scripts/preflight.py

What it checks (in order, fail-fast):
  1. Python version >= 3.10
  2. Required pip packages installed (Flask, flask-cors, numpy, pandas,
     MetaTrader5, requests, waitress). Pair bots use only stdlib, so no
     extra bot packages are needed.
  3. Local repo modules import cleanly (ai_agent, agent_learning_loop).
  4. Ollama HTTP reachable at OLLAMA_URL.
  5. Configured OLLAMA_MODEL is pulled.
  6. Orchestrator smoke test: round-trip synthetic bot verdicts through
     Agent 0 (Orchestrator) and confirm valid decision list returned.
  7. Pair-bot smoke test: run all 4 deterministic bots with a synthetic
     PositionContext and confirm each returns a valid verdict dict.
  8. Optional: MT5 import works (Windows-only -- warned, not failed, on
     non-Windows so this script is still useful in CI).

Exit codes:
  0  -- all checks passed
  1  -- one or more required checks failed (details printed)

The script never makes an actual broker call, never opens a position,
and never reads or writes credentials.
"""

from __future__ import annotations

import importlib
import os
import platform
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ---- ANSI helpers --------------------------------------------------------

def _supports_colour() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if platform.system() == "Windows":
        return True
    return sys.stdout.isatty()


_USE_COLOUR = _supports_colour()


def _c(code: str, msg: str) -> str:
    return f"\033[{code}m{msg}\033[0m" if _USE_COLOUR else msg


def green(s: str)  -> str: return _c("32", s)
def red(s: str)    -> str: return _c("31", s)
def yellow(s: str) -> str: return _c("33", s)
def cyan(s: str)   -> str: return _c("36", s)
def bold(s: str)   -> str: return _c("1",  s)


# ---- Result tracking -----------------------------------------------------

class CheckResult:
    __slots__ = ("name", "ok", "detail", "fatal")

    def __init__(self, name: str, ok: bool,
                 detail: str = "", fatal: bool = True):
        self.name   = name
        self.ok     = ok
        self.detail = detail
        self.fatal  = fatal


results: list[CheckResult] = []


def _record(name: str, ok: bool, detail: str = "",
            fatal: bool = True) -> bool:
    results.append(CheckResult(name, ok, detail, fatal))
    icon = green("v") if ok else (red("x") if fatal else yellow("!"))
    print(f"  {icon} {name}" + (f"  -- {detail}" if detail else ""))
    return ok


# ---- Individual checks ---------------------------------------------------

def check_python_version() -> bool:
    v = sys.version_info
    ok = v >= (3, 10)
    detail = f"{v.major}.{v.minor}.{v.micro}"
    if not ok:
        detail += " (need >= 3.10)"
    return _record("Python version", ok, detail)


REQUIRED_PACKAGES = [
    ("flask",         "Flask",      "dashboard server"),
    ("flask_cors",    "flask-cors", "dashboard CORS"),
    ("numpy",         "numpy",      "indicators / backtest math"),
    ("pandas",        "pandas",     "OHLC data handling"),
    ("requests",      "requests",   "HTTP helpers"),
    ("waitress",      "waitress",   "production WSGI server"),
]


def check_pip_packages() -> bool:
    all_ok = True
    for import_name, install_name, purpose in REQUIRED_PACKAGES:
        try:
            importlib.import_module(import_name)
            _record(f"pip: {install_name}", True, purpose)
        except ImportError:
            all_ok = False
            _record(f"pip: {install_name}", False,
                    f"missing -- install with `pip install {install_name}` ({purpose})")
    return all_ok


def check_repo_modules() -> bool:
    all_ok = True
    for module_name in ("ai_agent", "agent_learning_loop"):
        try:
            importlib.import_module(module_name)
            _record(f"import {module_name}", True)
        except Exception as exc:
            all_ok = False
            _record(f"import {module_name}", False,
                    f"{type(exc).__name__}: {exc}")
    return all_ok


def check_ollama_reachable() -> Optional[dict]:
    try:
        from ai_agent import ollama_health
    except Exception as exc:
        _record("Ollama: probe import", False,
                f"cannot import ai_agent.ollama_health: {exc}")
        return None

    health = ollama_health()
    url = health.get("url", "?")

    if not health.get("reachable"):
        _record("Ollama: reachable", False,
                f"cannot reach {url}: {health.get('error') or 'unknown error'}. "
                "Run scripts/setup_ollama.ps1 first.")
        return health

    _record("Ollama: reachable", True, url)
    return health


def check_ollama_model(health: dict) -> bool:
    if not health:
        return False
    if health.get("model_loaded"):
        return _record("Ollama: model pulled", True, health.get("model", "?"))
    return _record("Ollama: model pulled", False,
                   f"'{health.get('model','?')}' not found. "
                   f"Run: ollama pull {health.get('model','qwen2.5:14b-instruct')}")


def check_orchestrator_smoke_test() -> bool:
    """
    Send a synthetic set of pair-bot verdicts to the Orchestrator (Agent 0)
    and verify it returns a valid decision list. This is the only LLM call
    in the preflight -- it confirms the Ollama -> Orchestrator path works
    end-to-end.
    """
    try:
        from ai_agent import Orchestrator, OllamaClient, PortfolioSnapshot
    except Exception as exc:
        return _record("Orchestrator: smoke test", False,
                       f"import failed: {exc}")

    # Synthetic bot verdicts -- one hold and one move_sl.
    fake_verdicts = [
        {"symbol": "EURUSD", "ticket": 1001, "action": "hold",
         "new_sl": None,
         "reason": "EURUSD bot: hold (profit 0.50 ATR)",
         "agent": "EURUSDBot"},
        {"symbol": "GBPJPY", "ticket": 1002, "action": "move_sl",
         "new_sl": 190.500,
         "reason": "GBPJPY bot: profit 1.60 ATR -- trail SL 1.5 ATR",
         "agent": "GBPJPYBot"},
    ]
    fake_portfolio = PortfolioSnapshot(
        equity=10000.0, balance=10100.0, daily_pl=-100.0,
        open_positions_count=2, max_positions_total=4,
    )

    try:
        orch = Orchestrator(OllamaClient())
        decisions = orch.orchestrate(fake_verdicts, portfolio=fake_portfolio)
    except Exception as exc:
        return _record("Orchestrator: smoke test", False,
                       f"orchestrate() raised {type(exc).__name__}: {exc}")

    if not isinstance(decisions, list):
        return _record("Orchestrator: smoke test", False,
                       f"expected list, got {type(decisions).__name__}")

    # Each item must have at minimum a ticket and action.
    for item in decisions:
        if not isinstance(item, dict):
            return _record("Orchestrator: smoke test", False,
                           f"non-dict item in decisions: {item!r}")
        if "ticket" not in item or "action" not in item:
            return _record("Orchestrator: smoke test", False,
                           f"decision missing ticket/action: {item!r}")

    return _record("Orchestrator: smoke test", True,
                   f"{len(decisions)} decision(s) returned, all valid")


def check_pair_bot_smoke_test() -> bool:
    """
    Run all 4 deterministic pair bots with a synthetic PositionContext and
    verify each returns a valid verdict dict. No Ollama needed.
    """
    try:
        from ai_agent import (
            EURUSDBot, GBPUSDBot, GBPJPYBot, EURJPYBot, PositionContext,
        )
    except Exception as exc:
        return _record("Pair bots: smoke test", False,
                       f"import failed: {exc}")

    bots = [
        (EURUSDBot(),  "EURUSD", 5, 0.0010),
        (GBPUSDBot(),  "GBPUSD", 5, 0.0010),
        (GBPJPYBot(),  "GBPJPY", 3, 0.5000),
        (EURJPYBot(),  "EURJPY", 3, 0.3000),
    ]

    all_ok = True
    for bot, sym, digits, atr in bots:
        ctx = PositionContext(
            symbol=sym, ticket=9000, side="BUY",
            entry=1.10, cur_price=1.11, cur_sl=1.09,
            cur_tp=1.13, profit_pts=100.0, peak_pts=100.0,
            atr=atr, digits=digits,
        )
        try:
            v = bot.manage_position(ctx)
            if not isinstance(v, dict) or "action" not in v:
                raise ValueError(f"bad verdict shape: {v!r}")
            _record(f"Bot {sym}", True, f"action={v['action']}")
        except Exception as exc:
            _record(f"Bot {sym}", False,
                    f"{type(exc).__name__}: {exc}")
            all_ok = False

    return all_ok


def check_mt5_optional() -> bool:
    is_windows = platform.system() == "Windows"
    try:
        importlib.import_module("MetaTrader5")
        return _record("MT5: importable", True, "Windows broker bridge")
    except ImportError:
        if is_windows:
            return _record(
                "MT5: importable", False,
                "missing -- install with `pip install MetaTrader5` "
                "(also requires the MT5 desktop terminal installed and logged in)",
            )
        return _record(
            "MT5: importable (non-Windows)", True,
            "skipped -- not available on this OS",
            fatal=False,
        )


# ---- Driver --------------------------------------------------------------

def main() -> int:
    print()
    print(cyan("=" * 67))
    print(cyan(bold("  Fortis Multi-Agent Stack -- Preflight Check")))
    print(cyan("=" * 67))
    print()

    print(bold("Python & dependencies"))
    py_ok   = check_python_version()
    pip_ok  = check_pip_packages()
    repo_ok = check_repo_modules()

    print()
    print(bold("Ollama (required for Orchestrator / Agent 0)"))
    health   = check_ollama_reachable()
    model_ok = check_ollama_model(health) if health else False

    print()
    print(bold("Agent 0 -- Orchestrator (LLM)"))
    orch_ok = False
    if health and model_ok:
        orch_ok = check_orchestrator_smoke_test()
    else:
        _record("Orchestrator: smoke test", False,
                "skipped -- Ollama not ready", fatal=True)

    print()
    print(bold("Bots 1-4 -- Pair bots (deterministic, no LLM needed)"))
    bots_ok = check_pair_bot_smoke_test()

    print()
    print(bold("Broker (optional in CI)"))
    mt5_ok = check_mt5_optional()

    # ---- Summary ---------------------------------------------------------
    print()
    print(cyan("-" * 67))
    fatal_failures = [r for r in results if not r.ok and r.fatal]
    warnings       = [r for r in results if not r.ok and not r.fatal]

    if fatal_failures:
        print(red(bold(f"  x {len(fatal_failures)} check(s) failed:")))
        for r in fatal_failures:
            print(red(f"      - {r.name}: {r.detail}"))
        if warnings:
            print(yellow(f"  ! {len(warnings)} warning(s)"))
        print(cyan("-" * 67))
        print()
        print(red("  Bot is NOT production-ready. Fix the failures above and re-run."))
        print()
        return 1

    if warnings:
        print(yellow(bold(f"  ! {len(warnings)} warning(s) (non-fatal):")))
        for r in warnings:
            print(yellow(f"      - {r.name}: {r.detail}"))

    print(green(bold("  v All required checks passed -- agent stack is production-ready.")))
    print(cyan("-" * 67))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
