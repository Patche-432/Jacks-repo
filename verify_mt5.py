import sys
sys.path.insert(0, '.')
from ai_pro import _mt5_initialize
from datetime import datetime
import MetaTrader5 as mt5_lib

if _mt5_initialize():
    print('✓ MT5 Connected')
    acct_info = mt5_lib.account_info()
    print(f'  Account: {acct_info.login}')
    print(f'  Balance: {acct_info.balance:.2f} {acct_info.currency}')
    
    # Fetch some real data
    rates = mt5_lib.copy_rates_from_pos('EURUSD', mt5_lib.TIMEFRAME_M15, 0, 5)
    if rates is not None:
        print(f'\n✓ M15 Data retrieved: {len(rates)} candles')
        for i, rate in enumerate(rates[-3:]):
            ts = datetime.fromtimestamp(rate['time'])
            print(f'  {i}: {ts} | O:{rate["open"]:.5f} H:{rate["high"]:.5f} L:{rate["low"]:.5f} C:{rate["close"]:.5f}')
    else:
        print('✗ Failed to fetch M15 data')
else:
    print('✗ MT5 Connection failed')
