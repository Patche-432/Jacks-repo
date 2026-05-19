# Fortis — Multi-Agent FX Trading Bot

CHoCH + daily-levels strategy with a 5-agent architecture, MT5 broker bridge,
Flask dashboard, and a backtest-driven learning loop that adapts per-pair
signal confidence over time.

## Architecture

```
                     Strategy engine (ai_pro.py)
                     4 ENV signal types + POC bias gate
                              |
                              | signals (already bias-filtered)
                              v
          ┌───────────────────────────────────────┐
          │  4 Pair Bots  (deterministic Python)   │
          │  Bot 1: EURUSD  Bot 2: GBPUSD          │
          │  Bot 3: GBPJPY  Bot 4: EURJPY          │
          │  Each manages its own ATR / trail /     │
          │  structure rules  ── no Ollama needed  │
          └──────────────┬────────────────────────┘
                         | 4 verdicts (hold/move_sl/close)
                         v
          ┌───────────────────────────────────────┐
          │  Agent 0 -- Orchestrator  (LLM)        │
          │  Sees all 4 verdicts + portfolio state │
          │  Can VETO any verdict or OVERRIDE a    │
          │  hold to a close when portfolio at     │
          │  risk. Falls back to approve-all if    │
          │  Ollama is down -- bots keep working.  │
          └──────────────┬────────────────────────┘
                         | approved actions
                         v
                    MT5 broker (via terminal)


          ┌────────────────────┐    ┌──────────────────────┐
          │  Backtester        │───▶│  backtest_insights   │
          │  odl/backtest.py   │    │     .json            │
          └────────────────────┘    └──────────┬───────────┘
                                               │ auto-reload on mtime
                                               v
                                  ┌──────────────────────┐
                                  │  AgentLearningLoop   │
                                  │  adjusts confidence  │
                                  │  scores per pair     │
                                  └──────────────────────┘
```

### Why this shape

| Problem | Solution |
|---|---|
| One LLM trying to be a specialist on 4 pairs regresses to generic advice | Encode each pair's known behaviour deterministically in `PairBotBase` subclasses |
| Managing 4 positions was costing 4 Ollama calls per poll cycle | Pair bots are now pure Python — zero LLM calls for position management |
| Ollama outage stopped all position management | Pair bots are deterministic; only orchestrator needs Ollama. Ollama outage = approve-all fallback, bots still manage |
| No cross-pair portfolio awareness | Agent 0 (Orchestrator) sees all 4 verdicts and the portfolio snapshot each cycle |

## What's here

| Path | Purpose |
|---|---|
| `ai_pro.py` | Strategy engine + bot subprocess entry point |
| `ai_agent.py` | 4 pair bots (`EURUSDBot` / `GBPUSDBot` / `GBPJPYBot` / `EURJPYBot`) + `Orchestrator` (Agent 0) |
| `agent_learning_loop.py` | Loads backtest insights and adjusts per-signal confidence at runtime |
| `core/server.py` | Flask app: dashboard UI + REST/SSE endpoints |
| `core/mt5_connection.py` | MT5 wrapper (connect, status, place trades, history) |
| `odl/backtest.py` | Strategy backtester. Writes `backtest_insights.json` |
| `tests/` | Pytest suite (bots, orchestrator, learning loop, POC parity) |
| `scripts/setup_ollama.ps1` | One-shot Windows installer/verifier for Ollama + model |
| `scripts/preflight.py` | Pre-flight check (Python, packages, Ollama, orchestrator + bot smoke tests) |
| `scripts/serve.py` | Production WSGI launcher (waitress on Windows) |
| `index.html` | Dashboard UI |

## Requirements

- **Windows** (the `MetaTrader5` Python package is Windows-only)
- **Python 3.10+**
- **MetaTrader 5 desktop terminal** installed and logged into a broker account
- **Ollama** running locally — only needed for Agent 0 (Orchestrator). Default model: `qwen2.5:14b-instruct` (requires ~10 GB RAM)

## First-time setup

```powershell
# 1. Create + activate a virtualenv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install runtime deps (includes waitress production server)
pip install -r requirements.txt

# 3. Set up Ollama (idempotent -- re-runnable any time)
.\scripts\setup_ollama.ps1

# 4. Verify the whole stack
python scripts\preflight.py
```

Preflight checks Python version, pip packages, Ollama reachability, an
orchestrator LLM round-trip, and all 4 pair bots with synthetic data.

## Running

### Production (recommended)

```powershell
python scripts\serve.py
```

Starts the dashboard via waitress on `http://127.0.0.1:5000`. From there:

1. Connect MT5 (sidebar)
2. Hit **Start** — the bot subprocess launches and the dashboard polls live state

### Development

```powershell
python core\server.py
```

Flask dev server — fine for local hacking, logs a warning, not for production.

## Running the tests

```powershell
pytest
```

Discovers tests in `tests/` and `odl/`. Coverage includes:

- **`test_agent_bots.py`** — all 4 pair bots (per-pair ATR thresholds, JPY pip math, EURJPY trend-close rule, hard rails, orchestrator veto/override, kill-switch)
- **`test_agent_learning_loop.py`** — learning loop auto-reload, confidence adjustments, multi-pair persistence
- **`test_poc_filter_parity.py`** — live strategy and backtester POC filter produce identical results; constants can't drift
- **`odl/test_backtest_management.py`** — backtester simulation correctness (14 cases)

## Agent 0 — Orchestrator details

Each poll cycle, after all 4 pair bots have produced verdicts:

1. **Stage 1 (deterministic)** — hard portfolio rules: daily loss breach suppresses tightens, position cap enforced. The LLM cannot loosen these.
2. **Stage 2 (LLM)** — Agent 0 sees the surviving verdicts + a `PortfolioSnapshot` (equity, daily P&L, open count vs limit) and may:
   - **VETO** — suppress a verdict (turned to hold, no-op for executor)
   - **OVERRIDE** — turn a "hold" into a "close" when portfolio in trouble
   - **APPROVE** — pass through unchanged (default)

If Ollama is unreachable, Stage 2 is skipped and all Stage 1 verdicts are approved as-is. The broker's hard SL/TP remains authoritative regardless.

## How the learning loop works

Every backtest run writes per-pair feature importances to `backtest_insights.json`. The learning loop watches this file's mtime; when it changes, the next signal review picks up new weights without a bot restart.

Confidence is adjusted for three features:

| Feature | Effect |
|---|---|
| `hour_of_day` | Boost when in London/NY overlap hours {8–11, 13–16} UTC |
| `dist_to_poc_pips` | Boost when price is within 50 pips of the volume POC |
| `risk_pips` | Boost in 40–100 pip range; penalty below 40 pips |

Output is clamped to `[60, 95]`.

## Failure modes

| Failure | Impact | Recovery |
|---|---|---|
| Ollama down | Orchestrator falls back to approve-all; pair bots continue unaffected | Restart Ollama |
| ai_agent import fails | Pair bots unavailable; bot holds all positions | Check install |
| MT5 disconnects | Bot pauses trading; reconnects on next poll | MT5 terminal |

## Configuration

| Variable | Default | Effect |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server location |
| `OLLAMA_MODEL` | `qwen2.5:14b-instruct` | Orchestrator model (downgrade to 7b/3b on smaller boxes) |
| `AGENT_TIMEOUT_S` | `90` | Max wait per LLM call (14b on CPU needs 40-60s) |
| `AI_BACKEND` | `agent` | Set to `off` to disable orchestrator |
| `FORTIS_HOST` | `127.0.0.1` | Dashboard bind address |
| `FORTIS_PORT` | `5000` | Dashboard port |

## Documentation

- `OLLAMA_SETUP.md` — Ollama install, model picker, troubleshooting
- `AGENT_LEARNING_LOOP_README.md` — backtest → live confidence pipeline
- `requirements.md` — dependency notes

## License

Private repository.
