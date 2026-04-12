# AI_Pro Backtester — User Guide

## Quick Start

### Demo Mode (No MT5 Required)
```bash
python backtest.py --demo
```
✅ Generates 200 bars of synthetic trading data and shows a sample.

---

### Live Backtest (Requires MT5)
```bash
# Basic backtest
python backtest.py --symbol EURUSD --days 30

# With trade export
python backtest.py --symbol GBPUSD --days 7 --export

# Full analysis
python backtest.py --symbol USDJPY --days 14 --export --plot
```

---

## What Gets Measured

After a backtest completes, you'll see:

### Performance Metrics
- **Trades:** Total count, wins, losses, win rate
- **P&L:** Total profit/loss, average per trade
- **Win/Loss Pips:** Average winning trade vs losing trade
- **Profit Factor:** How much more winners made than losers lost (>1.5 = good)
- **R:R Achieved:** Actual risk:reward ratio vs expected

### Key Analysis
**Component Score Correlations**
- Shows which signal components predict winners
- Example: "structure_strength: ↑ +0.35" = higher structure score = more wins
- Use this to tune `_calculate_signal_confidence()` weights

**Performance by Signal Quality**
```
  weak   (53 trades):  38% WR, $-42.50 avg, 0.82 RR
  fair   (41 trades):  46% WR, $-15.20 avg, 1.05 RR
  good   (28 trades):  61% WR, $+28.30 avg, 1.42 RR
  strong (12 trades):  75% WR, $+85.10 avg, 2.10 RR
```
→ Ideally, stronger signals = higher win rate + better R:R

---

## Interpreting Results

### Good Signs
✅ Win rate increases with signal quality (weak < fair < good < strong)
✅ Profit factor > 1.5
✅ Strong signals have 60%+ win rate
✅ Average R:R achieved is close to expected (1.5-2.0)

### Red Flags
❌ All signals lose or barely break even
❌ No difference between weak and strong signals
❌ Average loss > average win (kills profitability)
❌ R:R achieved << expected (SL/TP issues)

---

## Next Steps

### 1. Run baseline backtest
```bash
python backtest.py --symbol EURUSD --days 30 --export
```
Save results in `backtest_EURUSD_*.json`

### 2. Analyze component correlations
Look for which components correlate with wins:
- Structure, level, momentum should all be positive
- If a component is negative or ~0, it's not helping

### 3. Fine-tune weights
In `ai_pro.py`, `_calculate_signal_confidence()` has these weights:
```python
weights = {
    "structure": 0.35,      # ← Adjust these
    "level": 0.25,
    "momentum": 0.20,
    "spread_volatility": 0.10,
    "environment_fit": 0.10,
}
```

Example: If level interaction has highest correlation, increase its weight to 0.30.

### 4. Re-backtest and compare
```bash
python backtest.py --symbol EURUSD --days 30 --export
```
Did win rates improve? Did R:R achieve get closer to target?

---

## Export & Analysis

### Trade Log Format
```json
[
  {
    "entry_time": "2026-04-11T10:30:00+00:00",
    "exit_time": "2026-04-11T11:45:00+00:00",
    "signal": "BUY",
    "source": "CHoCH-BUY@PDL",
    "entry": 1.08523,
    "exit": 1.08645,
    "sl": 1.08412,
    "tp": 1.08734,
    "pips": 12.2,
    "pnl": 122.00,
    "outcome": "WIN",
    "rr_achieved": 1.42,
    "signal_quality": "good",
    "components": {
      "structure_strength": 0.8,
      "level_interaction": 0.7,
      "momentum_quality": 0.75,
      ...
    }
  },
  ...
]
```

Use this to:
- Deep-dive into winning vs losing trades
- See which component combinations predict winners
- Validate SL/TP placement logic

---

## Troubleshooting

### "Error: AI_Pro module not found"
→ Make sure `ai_pro.py` is in the **same folder** as `backtest.py`

### "MT5 initialization failed"
→ Ensure:
1. MT5 terminal is running
2. You've configured login/password via the dashboard (/bot/status)
3. Account allows trading (`Trade allowed = true`)
4. Try demo mode first: `python backtest.py --demo`

### "Not enough data for EURUSD"
→ Wait for 50+ minutes of M15 data to accumulate
→ You can backtest with fewer bars by adjusting `lookback_bars` parameter

---

## Performance Expectations

**Baseline** (before tuning):
- Win rate: 40-50% (vs 50% random)
- P&L: Break-even to slightly negative
- Component correlation: Some positive, some negative

**After improvements** (all 5 tasks completed):
- Win rate: 50-65%
- P&L: Positive overall, better on strong signals
- Structure score correlation: +0.3 to +0.5
- Level score correlation: +0.2 to +0.4
- Momentum score correlation: +0.2 to +0.3

---

## Advanced

### Custom Backtest Window
```bash
# Run 100-day backtest (slower)
python backtest.py --symbol EURUSD --days 100

# Run on multiple symbols
for symbol in EURUSD GBPUSD USDJPY; do
  python backtest.py --symbol $symbol --days 30 --export
done
```

### Adjusting Risk Per Trade
In `AI_ProBacktester.__init__()`:
```python
risk_per_trade_pips=50.0,  # Adjust SL distance (default 50 pips)
max_trade_duration_bars=96,  # Close if not exited in 24h (96 × 15min)
```

---

## Questions?

Check the weekly results snapshots in `backtest_*.json` files. The component scores + outcomes will tell you:
1. Which components are predictive
2. What to optimize next
3. Whether recent changes helped or hurt

Start with structure detection, then level interaction, then momentum. Most edge comes from better definitions, not more rules.
