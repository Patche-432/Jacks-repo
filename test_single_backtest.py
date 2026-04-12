#!/usr/bin/env python3
"""
Simple test of single pair backtest
"""
import sys
sys.path.insert(0, '.')

from backtest import AI_ProBacktester
import traceback

symbol = "EURUSD"
days = 7  # Test with 7 days like the multi-pair test

try:
    print(f"Testing backtest for {symbol} ({days} days)...")
    bt = AI_ProBacktester()
    results = bt.run(symbol, days=days)
    
    if "error" in results:
        print(f"ERROR: {results['error']}")
    else:
        print(f"SUCCESS: {results['total_trades']} trades, {results['win_rate']:.1%} WR")
        print(f"P&L: ${results['total_pnl']:+.2f}")
    
except Exception as e:
    print(f"Exception: {e}")
    traceback.print_exc()
