# Agent Learning Loop

The agent learning loop closes the feedback loop between the **backtester**
and the **live bot**: every backtest run writes per-pair feature importances
to `backtest_insights.json`, and the live bot's signal generator reads them
to adjust per-signal confidence on the fly. No bot restart required.

This document describes the pipeline as it actually exists in the code.
For verification, see `tests/test_agent_learning_loop.py`.

---

## Pipeline (one cycle)

```
┌──────────────────────────┐
│ 1. odl/backtest.py       │
│    Runs strategy on M15  │
│    history; trains a     │
│    RandomForest on       │
│    decided trades        │
└──────────┬───────────────┘
           │ feature_importance dict
           ▼
┌──────────────────────────┐
│ 2. _save_insights_for_   │
│    pair(symbol, fi)      │
│    Atomic temp+rename;   │
│    merges with other     │
│    pairs' insights       │
└──────────┬───────────────┘
           │ writes
           ▼
┌──────────────────────────┐
│ 3. backtest_insights.json│
│    Repo-root file:       │
│    {symbol: {feature:    │
│              importance}}│
└──────────┬───────────────┘
           │ stat() + reload on mtime change
           ▼
┌──────────────────────────┐
│ 4. AgentLearningLoop     │
│    Singleton in          │
│    agent_learning_loop.py│
│    Auto-reloads each call│
└──────────┬───────────────┘
           │ apply_learned_weights(...)
           ▼
┌──────────────────────────┐
│ 5. ai_pro.py             │
│    generate_trade_signal │
│    Each ENV1..ENV4 path  │
│    calls the loop before │
│    publishing confidence │
└──────────────────────────┘
```

## What gets adjusted

The strategy generates four environment-specific signal types
(CHoCH BUY at PDL, CHoCH SELL at PDH, Continuation BUY at PDH,
Continuation SELL at PDL). Each starts with a base confidence:

| Environment            | Base | Where         |
|------------------------|------|---------------|
| ENV 1 — CHoCH BUY      | 85   | ai_pro.py:~1714 |
| ENV 2 — CHoCH SELL     | 85   | ai_pro.py:~1741 |
| ENV 3 — Continuation B | 75–85 | ai_pro.py:~1767 (`65 + strength*10`) |
| ENV 4 — Continuation S | 75–85 | ai_pro.py:~1793 (`65 + strength*10`) |

The learning loop then adjusts that base according to per-pair feature
weights from the most recent backtest, before the strategy adds the
`vp_score` (volume-profile score) and clamps to [60, 95].

## What features the loop actually uses

The backtester's RandomForest computes importances for ~14 features. The
live loop applies adjustments for three of them — the ones with a
defensible rationale and a stable interpretation across pairs:

| Feature              | When it boosts                                              | When it penalises                |
|----------------------|--------------------------------------------------------------|----------------------------------|
| `hour_of_day`        | importance > 0.15 AND current UTC hour ∈ {8–11, 13–16}        | (no penalty)                     |
| `dist_to_poc_pips`   | importance > 0.15 AND price within 50 pips of POC             | (no penalty)                     |
| `risk_pips`          | importance > 0.15 AND SL distance is 40–100 pips              | risk < 40 pips → −5 confidence   |

Importances below 0.15 are treated as noise — the strategy is not
adjusted for them.

## Auto-reload (the part that lets the bot pick up new backtests live)

The bot is a long-running subprocess. Without auto-reload, you'd have to
restart it every time a backtest produced new insights. To avoid this,
`AgentLearningLoop.apply_learned_weights()` does an `os.stat()` on
`backtest_insights.json` at the start of every call. If the mtime moved
forward since the last load, the loop re-reads the file before applying
adjustments.

The cost is one stat() per signal review — negligible compared to a
network round-trip to MT5 or an Ollama LLM call.

## Failure modes (graceful)

The learning loop is purely additive. If anything goes wrong, the strategy
falls back to its base confidence:

- **No `backtest_insights.json` yet** (fresh checkout, no backtest run)
  → returns `base_confidence` unchanged.
- **File contains corrupt JSON** → logs a warning, keeps last-known-good
  insights or empty if first load.
- **Pair not in insights** (e.g. ran backtest only for EURUSD, but signal
  fires on GBPJPY) → returns `base_confidence` unchanged.
- **Module import fails** (Python issue with `agent_learning_loop.py`) →
  the wrapper `_apply_learning_loop()` in ai_pro.py catches it and returns
  base_confidence; bot logs the import error to stderr at startup.

You'll never get a bot crash from a learning-loop bug. Worst case is the
bot reverts to non-adaptive behaviour.

## Multi-pair persistence

`_save_insights_for_pair("EURUSD", fi)` does a read-modify-write cycle:

1. Read existing `backtest_insights.json` (if any)
2. Update *only* the `"EURUSD"` key
3. Write to a temp file in the same directory
4. `os.replace()` the temp file over the real one (atomic on Linux/Win)

So running a backtest for EURUSD does not erase the insights from a
previous GBPJPY run. Each pair's weights persist independently.

## Verification

```powershell
# Run the unit + integration test suite
python -m pytest tests/test_agent_learning_loop.py -v
```

Tests cover:
- Returns base when no insights exist
- Boost when current hour is in the optimal window
- Boost from POC alignment
- Boost from optimal risk range, penalty when risk too tight
- Output clamping to [60, 95]
- Auto-reload on mtime change (the bot pickup mechanism)
- Auto-clear when the file is deleted
- Backtester writes correct shape
- Backtester preserves other pairs on update
- ai_pro.py imports cleanly with the wiring in place

## Inspection

Programmatically:

```python
from agent_learning_loop import get_learning_loop
loop = get_learning_loop()
print(loop.status())
# {
#   "insights_file": ".../backtest_insights.json",
#   "file_exists": True,
#   "last_mtime": "2026-05-04T03:54:32.108910",
#   "pairs": ["EURUSD", "GBPJPY"],
#   "feature_count": {"EURUSD": 4, "GBPJPY": 5}
# }
```
