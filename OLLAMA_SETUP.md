# Ollama setup for the Fortis multi-agent bot

The agent layer (Agent Zero + the four pair agents in `ai_agent.py`) calls a
**local Ollama server** over HTTP. The Python code uses only `urllib` from
the stdlib, so there are no pip packages to install for the agent itself —
but you do need Ollama running on the host machine, with the configured
model pulled.

This document covers the one-time setup. After it's done, the agents work
unattended; the dashboard's "Start" button will refuse to launch the bot if
the agent layer is unreachable, so you'll get a clear error if it ever
breaks.

---

## Quick start (Windows, recommended path)

From the repo root in PowerShell:

```powershell
# 1. Install Ollama (one of these works)
winget install Ollama.Ollama
# OR download from https://ollama.com/download/windows

# 2. Pull the model + smoke-test (idempotent — safe to re-run)
.\scripts\setup_ollama.ps1

# 3. Verify the whole agent stack
python scripts\preflight.py
```

If the preflight ends with `✓ All required checks passed`, the agents are
ready.

---

## What the agents talk to

| Setting           | Default                  | Override via env var | Purpose                                              |
|-------------------|--------------------------|----------------------|------------------------------------------------------|
| Ollama URL        | `http://localhost:11434` | `OLLAMA_URL`         | Where the Ollama HTTP server is listening            |
| Model             | `qwen2.5:14b-instruct`   | `OLLAMA_MODEL`       | Which model the agents call                          |
| Per-call timeout  | 90 seconds               | `AGENT_TIMEOUT_S`    | Max wait for a single chat call                      |
| Agent kill-switch | `agent` (on)             | `AI_BACKEND`         | Set to `off` / `none` / `disabled` to bypass the LLM |

These are all read by `OllamaClient` in `ai_agent.py`. A copy of these is
also kept in `.env.example` for reference.

## Picking a model

The default is sized for a modern desktop CPU (Ryzen 7000-class) with 16 GB RAM.
Downgrade to 7b/3b on lighter hardware:

| Model                  | Memory   | Notes                                              |
|------------------------|----------|----------------------------------------------------|
| `qwen2.5:14b-instruct` | ~9 GB    | Default. Real multi-signal reasoning. Recommended on Ryzen 7000 + 16 GB RAM. |
| `phi4:14b`             | ~9 GB    | Alternative 14b — slightly stronger reasoning, occasionally drifts on JSON. |
| `qwen2.5:7b-instruct`  | ~6 GB    | Adequate rule-follower. Use on mini PCs / N100-class. |
| `qwen2.5:3b-instruct`  | ~3 GB    | Minimal. Weak reasoning but very fast.             |
| `llama3.2:3b`          | ~3 GB    | Alternative small model, decent on ARM.            |

Switch model:

```powershell
$env:OLLAMA_MODEL = "qwen2.5:14b-instruct"
ollama pull qwen2.5:14b-instruct
.\scripts\setup_ollama.ps1   # idempotent — re-validates with the new model
```

## How the dashboard uses Ollama

- `GET /api/ollama/health` — JSON probe of the local Ollama server. The
  dashboard can poll this without starting the bot. Returns:
  ```json
  {"reachable": true, "model_loaded": true, "url": "http://localhost:11434",
   "model": "qwen2.5:14b-instruct", "error": null}
  ```
- `POST /bot/start` — when "AI review" is enabled in the sidebar, the
  endpoint runs a preflight check before launching the bot subprocess. If
  Ollama is unreachable or the model isn't pulled, you get a `503` with an
  actionable error message instead of a silently broken bot.

## Failure modes (these are by design)

The agent layer **never blocks the bot from holding a position** if Ollama
goes down at runtime:

- **Signal review**: a transport error returns an *automatic reject* — no
  trade is opened. (Conservative: rather not trade than trade blindly.)
- **Position management**: a transport error returns an *automatic hold* —
  no position is touched. The broker's hard SL/TP remain authoritative.

So if Ollama crashes mid-session, the worst case is the bot stops opening
new trades and stops trailing stops; existing positions are still protected
by the broker-side stop. Restart Ollama and the bot picks up again on the
next poll cycle without intervention.

## Running Ollama as a service (so it survives reboots)

The Windows installer registers a per-user background service that
auto-starts on login. If you want it to run as a true machine service
(starts even if no user is logged in), use [NSSM](https://nssm.cc/):

```powershell
# Run once as admin
nssm install Ollama "C:\Program Files\Ollama\ollama.exe" "serve"
nssm set Ollama AppEnvironmentExtra OLLAMA_HOST=127.0.0.1:11434
nssm start Ollama
```

(NSSM is also useful for running `core\server.py` itself as a service —
covered separately in the deployment notes.)

## Troubleshooting

**"Ollama is not reachable at http://localhost:11434"**
- Open the Ollama desktop app once — that starts the service.
- Or run `ollama serve` in a separate terminal.

**"Ollama model 'qwen2.5:14b-instruct' is not pulled"**
- Run: `ollama pull qwen2.5:14b-instruct`
- Or run `.\scripts\setup_ollama.ps1` to do the same plus a smoke test.

**"agent transport error" appearing in `ai_thoughts.jsonl` while the bot is running**
- Check `GET /api/ollama/health` — Ollama probably restarted or the model
  was unloaded.
- The bot is safe — it's auto-rejecting new entries and holding positions
  until Ollama returns. No action needed unless you want to skip the LLM
  gate (set `AI_BACKEND=off` and restart the bot for strategy-only mode).

**Slow first response after a restart**
- Ollama lazy-loads models. The first chat call after a model isn't in
  memory can take 10–30 seconds for small models, 30–60s for 14b on CPU.
  The default `AGENT_TIMEOUT_S=90` accounts for this.
