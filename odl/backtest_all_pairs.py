#!/usr/bin/env python3
"""
Backtest all trading pairs and generate results table
"""
import sys
sys.path.insert(0, '.')

from odl.backtest import AI_ProBacktester
import pandas as pd

# Trading pairs from configuration
PAIRS = ["EURUSD", "GBPUSD", "EURJPY", "GBPJPY"]
DAYS = 7

def run_all_backtests(days=7):
    """Run backtest for all pairs and return results table."""
    results_list = []
    
    for symbol in PAIRS:
        print(f"\n{'='*70}")
        print(f"[BACKTESTING] {symbol}")
        print(f"{'='*70}")
        
        bt = AI_ProBacktester()
        results = bt.run(symbol, days=days)
        
        if "error" in results:
            print(f"[ERROR] {results['error']}")
            results_list.append({
                "Symbol": symbol,
                "Trades": 0,
                "Win Rate": "ERROR",
                "P&L": "ERROR",
                "Avg P&L/Trade": "ERROR",
                "Profit Factor": "ERROR",
                "Avg R:R": "ERROR",
                "Status": "Failed"
            })
        else:
            print(f"[COMPLETE] {results['total_trades']} trades")
            results_list.append({
                "Symbol": symbol,
                "Trades": results["total_trades"],
                "Wins": results["wins"],
                "Losses": results["losses"],
                "Win Rate": f"{results['win_rate']:.1%}",
                "P&L": f"${results['total_pnl']:+,.2f}",
                "Avg P&L/Trade": f"${results['avg_pnl_per_trade']:+.2f}",
                "Avg Win": f"+{results['avg_win_pips']:.0f}p",
                "Avg Loss": f"{results['avg_loss_pips']:.0f}p",
                "Profit Factor": f"{results['profit_factor']:.2f}x",
                "Avg R:R": f"{results['avg_rr_achieved']:.2f}",
                "Period": results["period"]
            })
    
    return results_list

def print_results_table(results_list):
    """Print results in a formatted table."""
    df = pd.DataFrame(results_list)
    
    print("\n\n" + "="*140)
    print(f"{'BACKTEST RESULTS TABLE':^140}")
    print("="*140)
    print(df.to_string(index=False))
    print("="*140)
    
    # Summary statistics
    valid_results = [r for r in results_list if r["Status"] != "Failed" if "Status" in r]
    
    if valid_results:
        print(f"\n[SUMMARY] ACROSS ALL PAIRS:")
        print(f"   Total pairs tested: {len(PAIRS)}")
        print(f"   Successful: {len(valid_results)}")
    
    return df

if __name__ == "__main__":
    print(f"\nStarting backtest for {len(PAIRS)} pairs ({DAYS} days each)\n")
    
    results = run_all_backtests(days=DAYS)
    df = print_results_table(results)
    
    # Export to CSV
    csv_file = f"backtest_results_{DAYS}days_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(csv_file, index=False)
    print(f"\nResults exported to: {csv_file}")
