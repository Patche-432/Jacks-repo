import sys
sys.path.insert(0, '.')
from backtest import AI_ProBacktester
from datetime import datetime
from collections import Counter

bt = AI_ProBacktester()
df = bt.fetch_data('EURUSD', days=7)

print(f'\n📊 M15 DATA RANGE:')
print(f'   First bar:  {df.iloc[0]["time"]}')
print(f'   Last bar:   {df.iloc[-1]["time"]}')
print(f'   Total bars: {len(df)}')

# Count bars by date
bar_dates = df['time'].dt.date
date_counts = Counter(bar_dates)
print(f'\n   Bars by date:')
for date in sorted(date_counts.keys()):
    print(f'      {date}: {date_counts[date]} bars')

print(f'\n🎯 GENERATING SIGNALS:')
signals = bt.generate_signals('EURUSD', df)

# Count signals by date
sig_dates = [sig.ts.date() for sig in signals]
sig_counts = Counter(sig_dates)

print(f'   Total signals: {len(signals)}')
print(f'\n   Signals by date:')
for date in sorted(sig_counts.keys()):
    pct = (sig_counts[date] / len(signals) * 100)
    print(f'      {date}: {sig_counts[date]} signals ({pct:.1f}%)')

if signals:
    print(f'\n   Signal distribution:')
    print(f'      First signal: {signals[0].ts}')
    print(f'      Last signal:  {signals[-1].ts}')
