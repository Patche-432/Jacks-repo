# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

**Fortis** — a Windows-only multi-agent FX trading bot for MetaTrader 5. It trades 4 currency pairs (EURUSD, GBPUSD, GBPJPY, EURJPY) using a CHoCH + daily-levels strategy, with a 5-agent architecture: 4 deterministic pair bots + 1 LLM orchestrator (via Ollama). A Flask dashboard serves live state and controls.

## Commands

### Setup (first time)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
.\scripts\setup_ollama.ps1   # installs Ollama + pulls LLM model
python scripts\preflight.py  # validates the full stack
```

### Run
```powershell
python scripts\serve.py      # production (waitress on :5000)
python core\server.py        # dev server (Flask, localhost only)
python ai_pro.py --run       # headless, no dashboard
```

### Test
```powershell
pytest                       # all tests
pytest -v --durations=10     # verbose, slowest tests first
pytest tests/test_agent_bots.py  # single file
```

### Lint
```powershell
ruff check .                 # check (CI mode)
ruff check --fix .           # auto-fix
```

CI runs ruff + pytest on both Linux and Windows via `.github/workflows/ci.yml`.

## Architecture

### Agent Layers

```
Strategy (ai_pro.py)      → generates CHoCH/Continuation signals, applies POC bias gate
Pair Bots (ai_agent.py)   → deterministic position management per-symbol (no LLM)
Orchestrator (ai_agent.py)→ LLM (Ollama) portfolio-level veto/override across all 4 bots
Learning Loop             → auto-reloads backtest_insights.json to adjust per-pair confidence
Flask Dashboard           → REST + SSE; serves index.html for live monitoring and MT5 control
```

The key design choice: pair bots are **pure Python with no LLM calls**, so position management works even when Ollama is down. The orchestrator's fallback is approve-all.

### Key Files

| File | Lines | Role |
|------|-------|------|
| `ai_pro.py` | 4 445 | `Bot` class (main loop, signal gen); `AgentZeroBot` (subprocess entry, talks to Flask) |
| `ai_agent.py` | 877 | `PairBotBase` + 4 pair subclasses; `AgentZero` orchestrator; `OllamaClient` |
| `agent_learning_loop.py` | 354 | Mtime-watched auto-reload of backtest feature weights into per-pair confidence |
| `core/server.py` | 1 958 | Flask app, SSE `/stream`, MT5 connection wrapper with 12s health-monitor thread |
| `odl/backtest.py` | 1 720 | M15 strategy simulator → writes `backtest_insights.json` for the learning loop |

### Hard Rails (enforced in code, not config)

- **Pair bots**: can only tighten a stop or request early close. Cannot loosen SL, widen TP, or issue new entry orders.
- **Orchestrator**: can only VETO a verdict (suppress to hold) or OVERRIDE a hold to close. Cannot propose new SL values or tighten by itself.
- **POC bias gate**: upstream in `ai_pro.py`; misaligned signals never reach any agent.

### Thread Safety

- `_mt5_lock` serialises all calls into the MT5 C extension (not thread-safe).
- Learning loop reload is a dict swap at GIL level — no explicit lock needed.
- Flask handlers guard `mt5_conn` / `mt5_status` mutations with `_mt5_state_lock`.

### Pair-Bot Subclass Pattern

`EURUSDBot`, `GBPUSDBot`, `GBPJPYBot`, `EURJPYBot` each inherit `PairBotBase` and override only pair-specific constants (`ATR_TIGHTEN_THRESHOLD`, `JPY_PIP_MULTIPLE`, etc.). Prefer extending this pattern over adding switch statements in the base class.

### Testing on Linux (No MT5)

`tests/conftest.py` injects a minimal `MetaTrader5` stub so the full suite runs on Linux CI. Tests that require a real MT5 terminal are marked `@pytest.mark.requires_mt5` and only run in the Windows CI job.

## Environment Variables

Set in `.env` (see `.env.example`):

| Variable | Default | Notes |
|----------|---------|-------|
| `OLLAMA_URL` | `http://localhost:11434` | Orchestrator LLM |
| `OLLAMA_MODEL` | `qwen2.5:14b-instruct` | Also supports `qwen2.5:7b`, `qwen2.5:3b`, `phi4:14b`, `llama3.2:3b` |
| `AGENT_TIMEOUT_S` | `90` | Max LLM wait per call (14b on CPU needs 40-60s; 90s gives headroom) |
| `AI_BACKEND` | `agent` | Set to `off`/`none`/`disabled` to kill-switch agents |
| `FORTIS_HOST` | `127.0.0.1` | Dashboard bind address |
| `FORTIS_PORT` | `5000` | Dashboard port |

MT5 credentials are encrypted in `core/mt5_credentials_config.json` (gitignored) or via `MT5_LOGIN` / `MT5_PASSWORD` / `MT5_SERVER` / `MT5_PATH` env vars.

## Backtester → Live Pipeline

1. Run `odl/backtest.py` (or `odl/backtest_all_pairs.py`) to produce `backtest_insights.json`.
2. `AgentLearningLoop` checks file mtime on every poll cycle and hot-reloads it — no bot restart needed.
3. Weights from RandomForest feature importances (`hour_of_day`, `dist_to_poc_pips`, `risk_pips`) are applied to per-pair confidence, clamped to [60, 95].
