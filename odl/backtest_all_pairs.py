#!/usr/bin/env python3
"""
Backtest all trading pairs and generate results table.
"""
import sys
from pathlib import Path

# Allow both `python backtest_all_pairs.py` and `python -m odl.backtest_all_pairs`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from odl.backtest import AgentZeroBacktester
import pandas as pd

# Trading pairs from configuration
PAIRS = ["EURUSD", "GBPUSD", "EURJPY", "GBPJPY"]
DAYS = 7


def run_all_backtests(days: int = 7):
    """Run backtest for all pairs and return a list of result dicts."""
    results_list = []

    for symbol in PAIRS:
        print(f"\n{'='*70}")
        print(f"[BACKTESTING] {symbol}")
        print(f"{'='*70}")

        bt = AgentZeroBacktester()
        results = bt.run(symbol, days=days)

        if "error" in results:
            print(f"[ERROR] {results['error']}")
            results_list.append({
                "Symbol":        symbol,
                "Trades":        0,
                "Wins":          "-",
                "Losses":        "-",
                "Win Rate":      "ERROR",
                "P&L":           "ERROR",
                "Avg P&L/Trade": "ERROR",
                "Avg Win":       "ERROR",
                "Avg Loss":      "ERROR",
                "Profit Factor": "ERROR",
                "Avg R:R":       "ERROR",
                "Period":        results.get("period", "-"),
                "Status":        "Failed",
            })
        else:
            print(f"[COMPLETE] {results['total_trades']} trades")
            results_list.append({
                "Symbol":        symbol,
                "Trades":        results["total_trades"],
                "Wins":          results["wins"],
                "Losses":        results["losses"],
                "Win Rate":      f"{results['win_rate']:.1%}",
                "P&L":           f"${results['total_pnl']:+,.2f}",
                "Avg P&L/Trade": f"${results['avg_pnl_per_trade']:+.2f}",
                "Avg Win":       f"+{results['avg_win_pips']:.0f}p",
                "Avg Loss":      f"{results['avg_loss_pips']:.0f}p",
                "Profit Factor": f"{results['profit_factor']:.2f}x",
                "Avg R:R":       f"{results['avg_rr_achieved']:.2f}",
                "Period":        results["period"],
                "Status":        "OK",
            })

    return results_list


def print_results_table(results_list):
    """Print results in a formatted table and return the DataFrame."""
    df = pd.DataFrame(results_list)

    print("\n\n" + "=" * 140)
    print(f"{'BACKTEST RESULTS TABLE':^140}")
    print("=" * 140)
    print(df.to_string(index=False))
    print("=" * 140)

    valid = [r for r in results_list if r.get("Status") == "OK"]
    failed = [r for r in results_list if r.get("Status") == "Failed"]

    print(f"\n[SUMMARY] ACROSS ALL PAIRS:")
    print(f"   Total pairs tested : {len(PAIRS)}")
    print(f"   Successful         : {len(valid)}")
    print(f"   Failed             : {len(failed)}")

    if valid:
        total_trades = sum(r["Trades"] for r in valid)
        print(f"   Total trades       : {total_trades}")

    return df


if __name__ == "__main__":
    print(f"\nStarting backtest for {len(PAIRS)} pairs ({DAYS} days each)\n")

    results = run_all_backtests(days=DAYS)
    df = print_results_table(results)

    csv_file = (
        f"backtest_results_{DAYS}days_"
        f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    df.to_csv(csv_file, index=False)
    print(f"\nResults exported to: {csv_file}")
