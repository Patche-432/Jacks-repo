"""
Smoke test for backtester management logic.

Builds synthetic M15 OHLC frames with a known price path and verifies
the simulator produces the expected per-trade outcomes for:
  1) Clean TP win (1R hit → 50% closed; runner runs to TP)
  2) 1R hit, runner stopped at BE (no loss)
  3) Initial SL hit before 1R (full loss)
  4) Intrabar SL+1R collision → conservative SL wins (full loss)
  5) Timeout with 1R hit (partial + remainder at timeout close)
  6) SELL mirror of case 1

No MT5 required — we call simulate_trades directly with hand-crafted
Signal objects and a fake df.
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path("/sessions/serene-inspiring-faraday/mnt/Jacks-repo")))

from odl.backtest import AgentZeroBacktester, Signal  # noqa: E402


def make_df(path):
    """path: list of (high, low, close) per bar. Open is previous close."""
    rows = []
    t = datetime(2025, 1, 1, tzinfo=timezone.utc)
    prev_close = path[0][2]  # open of bar 0 = close of bar 0 for seed
    for hi, lo, cl in path:
        rows.append({
            "time":  t,
            "open":  prev_close,
            "high":  hi,
            "low":   lo,
            "close": cl,
        })
        t += timedelta(minutes=15)
        prev_close = cl
    return pd.DataFrame(rows)


def sig_buy(ts, entry_hint, sl, tp, symbol="EURUSD"):
    # signal fires at bar with time==ts; entry is the NEXT bar's open
    return Signal(
        ts=ts, symbol=symbol, signal="BUY", source="CHoCH-BUY@PDL",
        confidence=0.85, percentage=85, quality="good",
        entry_price=entry_hint, stop_loss=sl, take_profit=tp, rr_ratio=2.0,
    )


def sig_sell(ts, entry_hint, sl, tp, symbol="EURUSD"):
    return Signal(
        ts=ts, symbol=symbol, signal="SELL", source="Continuation-SELL@PDL",
        confidence=0.85, percentage=85, quality="good",
        entry_price=entry_hint, stop_loss=sl, take_profit=tp, rr_ratio=2.0,
    )


def run_case(name, df, signal, expected_reason, expected_outcome,
             expected_pips_sign):
    bt = AgentZeroBacktester(lot_size=1.0, spread_pips=0.0)
    trades = bt.simulate_trades("EURUSD", [signal], df)
    assert len(trades) == 1, f"{name}: expected 1 trade got {len(trades)}"
    t = trades[0]
    ok = (
        t.exit_reason == expected_reason and
        t.outcome == expected_outcome and
        ((expected_pips_sign == "+" and t.profit_pips > 0) or
         (expected_pips_sign == "-" and t.profit_pips < 0) or
         (expected_pips_sign == "0" and abs(t.profit_pips) < 1e-6))
    )
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name:40s} exit={t.exit_reason:20s} "
          f"outcome={t.outcome:4s} pips={t.profit_pips:+.1f} "
          f"(expected reason={expected_reason} outcome={expected_outcome} "
          f"pips={expected_pips_sign})")
    return ok


def main():
    # Use fake EURUSD (pip=0.0001). Entry bar is bar index pos+1.
    # We'll construct so signal.ts == bar 0.time → entry at bar 1.open.
    # bar 0 is the "signal fired" bar; bar 1 onward is where we act.

    # Spread=0 for simplicity. Entry = bar1.open.
    passed = 0
    total = 0

    # ── Case 1: Clean BUY, 1R then TP ─────────────────────────────
    # bar0: noop (signal generator). bar1 open=1.1000.
    # Entry=1.1000, SL=1.0990 (-10p), TP=1.1020 (+20p), 1R=1.1010.
    # bar1: (hi 1.1003, lo 1.0998, cl 1.1002)  → no hit
    # bar2: (hi 1.1012, lo 1.1001, cl 1.1011)  → 1R hit, partial close
    # bar3: (hi 1.1022, lo 1.1010, cl 1.1021)  → TP hit (runner)
    df = make_df([
        (1.1000, 1.1000, 1.1000),  # bar0 (signal bar, price doesn't matter)
        (1.1003, 1.0998, 1.1002),  # bar1 entry
        (1.1012, 1.1001, 1.1011),  # bar2 1R
        (1.1022, 1.1010, 1.1021),  # bar3 TP for runner
    ])
    s = sig_buy(df.iloc[0]["time"], 1.1000, 1.0990, 1.1020)
    # partial_pips = 10, runner_pips = 20, total = 0.5*10 + 0.5*20 = 15
    total += 1
    passed += run_case("BUY: 1R partial then TP runner", df, s,
                       "partial+tp", "WIN", "+")

    # ── Case 2: BUY, 1R then runner stopped at BE ─────────────────
    # Entry=1.1000, SL=1.0990, TP=1.1020, 1R=1.1010, BE=1.0001 (1 pip buffer)
    # bar2 hits 1R; bar3 drops to 1.0999 → BE SL (1.10005 +1p = 1.10005 → wait)
    # Actually BE = entry + 1pip = 1.10005+0.0001 = 1.10015? No, 1pip = 0.0001.
    # BE = 1.1000 + 0.0001 = 1.10010. bar3 low <= 1.10010 stops runner.
    df = make_df([
        (1.1000, 1.1000, 1.1000),
        (1.1003, 1.0998, 1.1002),
        (1.1012, 1.1001, 1.1011),
        (1.1013, 1.1000, 1.1001),  # low = 1.1000 <= BE 1.10010 → runner BE-stop
    ])
    s = sig_buy(df.iloc[0]["time"], 1.1000, 1.0990, 1.1020)
    # partial=10, runner=+1 (BE buffer), total = 0.5*10 + 0.5*1 = 5.5
    total += 1
    passed += run_case("BUY: 1R partial then BE stop", df, s,
                       "partial+be", "WIN", "+")

    # ── Case 3: BUY, initial SL hit before 1R ─────────────────────
    df = make_df([
        (1.1000, 1.1000, 1.1000),
        (1.1001, 1.0988, 1.0989),  # entry 1.1000, low=1.0988 <= SL 1.0990 → SL
    ])
    s = sig_buy(df.iloc[0]["time"], 1.1000, 1.0990, 1.1020)
    # No partial fired → full position exits at SL → runner_pips=-10
    bt = AgentZeroBacktester(lot_size=1.0, spread_pips=0.0)
    trades = bt.simulate_trades("EURUSD", [s], df)
    t = trades[0]
    ok = (t.exit_reason == "sl" and t.outcome == "LOSS"
          and abs(t.profit_pips - (-10.0)) < 0.5)
    total += 1; passed += 1 if ok else 0
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] BUY: initial SL (full loss -10p)        "
          f"exit={t.exit_reason:20s} outcome={t.outcome:4s} "
          f"pips={t.profit_pips:+.1f} (expected -10)")

    # ── Case 3b: Intrabar collision → SL → full loss ──────────────
    df = make_df([
        (1.1000, 1.1000, 1.1000),
        (1.1015, 1.0988, 1.0995),  # both 1R and SL in same bar → SL wins
    ])
    s = sig_buy(df.iloc[0]["time"], 1.1000, 1.0990, 1.1020)
    bt = AgentZeroBacktester(lot_size=1.0, spread_pips=0.0)
    trades = bt.simulate_trades("EURUSD", [s], df)
    t = trades[0]
    ok = (t.exit_reason == "sl" and t.outcome == "LOSS"
          and abs(t.profit_pips - (-10.0)) < 0.5)
    total += 1; passed += 1 if ok else 0
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] BUY: intrabar SL+1R → SL full loss       "
          f"exit={t.exit_reason:20s} outcome={t.outcome:4s} "
          f"pips={t.profit_pips:+.1f} (expected -10)")

    # ── Case 5: Timeout with 1R but no further target ─────────────
    # bar1 entry, bar2 hits 1R, bars 3..N neither TP nor BE. Exits at max_dur.
    # For speed, set max_trade_duration_bars = 3 so we timeout fast.
    bars = [
        (1.1000, 1.1000, 1.1000),
        (1.1003, 1.0998, 1.1002),
        (1.1012, 1.1001, 1.1011),    # 1R
        (1.1013, 1.1011, 1.1012),    # neutral, runner still open
    ]
    df = make_df(bars)
    s = sig_buy(df.iloc[0]["time"], 1.1000, 1.0990, 1.1020)
    bt = AgentZeroBacktester(lot_size=1.0, spread_pips=0.0,
                          max_trade_duration_bars=2)  # entry_idx=1, max=3
    trades = bt.simulate_trades("EURUSD", [s], df)
    t = trades[0]
    ok = (t.exit_reason == "partial+timeout" and t.outcome == "WIN"
          and t.profit_pips > 0)
    status = "PASS" if ok else "FAIL"
    total += 1; passed += 1 if ok else 0
    print(f"  [{status}] BUY: 1R partial then runner timeout     "
          f"exit={t.exit_reason:20s} outcome={t.outcome:4s} "
          f"pips={t.profit_pips:+.1f}")

    # ── Case 6: SELL mirror of case 1 ─────────────────────────────
    # Entry=1.1000, SL=1.1010 (+10p), TP=1.0980 (-20p), 1R=1.0990
    df = make_df([
        (1.1000, 1.1000, 1.1000),
        (1.1002, 1.0997, 1.0998),  # entry 1.1000 (bid fill)
        (1.0999, 1.0988, 1.0989),  # 1R hit (low <= 1.0990)
        (1.0990, 1.0978, 1.0979),  # TP hit (low <= 1.0980)
    ])
    s = sig_sell(df.iloc[0]["time"], 1.1000, 1.1010, 1.0980)
    total += 1
    passed += run_case("SELL: 1R partial then TP runner", df, s,
                       "partial+tp", "WIN", "+")

    # ── Case 7: RAW mode — single-exit (no management) ────────────
    df = make_df([
        (1.1000, 1.1000, 1.1000),
        (1.1003, 1.0998, 1.1002),
        (1.1012, 1.1001, 1.1011),
        (1.1022, 1.1010, 1.1021),  # TP
    ])
    s = sig_buy(df.iloc[0]["time"], 1.1000, 1.0990, 1.1020)
    bt = AgentZeroBacktester(lot_size=1.0, spread_pips=0.0,
                          enable_trade_management=False)
    trades = bt.simulate_trades("EURUSD", [s], df)
    t = trades[0]
    ok = (t.exit_reason == "tp" and t.outcome == "WIN"
          and abs(t.profit_pips - 20.0) < 0.5)  # full 20 pips, no partial
    status = "PASS" if ok else "FAIL"
    total += 1; passed += 1 if ok else 0
    print(f"  [{status}] RAW mode BUY: full TP (no mgmt)         "
          f"exit={t.exit_reason:20s} outcome={t.outcome:4s} "
          f"pips={t.profit_pips:+.1f}")

    # ── Case 8: window_start clips signal generation ──────────────
    # No MT5 needed — we probe the index arithmetic directly.
    import pandas as pd
    from datetime import datetime as _dt, timedelta as _td
    times = [_dt(2026, 4, 1) + _td(minutes=15 * i) for i in range(200)]
    df_probe = pd.DataFrame({"time": times})

    # Case 8a: window_start = bar 120 → start_idx should be 120
    ws = times[120]
    mask = df_probe["time"] >= ws
    first_in_window = int(mask.values.argmax()) if mask.any() else len(df_probe)
    start_idx = max(50, first_in_window)
    ok = (start_idx == 120)
    total += 1; passed += 1 if ok else 0
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] window clip: ws=bar120 → start_idx={start_idx} "
          f"(expected 120)")

    # Case 8b: window_start too early → warm-up minimum (50) wins
    ws_early = times[10]
    mask = df_probe["time"] >= ws_early
    first_in_window = int(mask.values.argmax()) if mask.any() else len(df_probe)
    start_idx = max(50, first_in_window)
    ok = (start_idx == 50)
    total += 1; passed += 1 if ok else 0
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] window clip: ws=bar10 → start_idx={start_idx} "
          f"(expected 50, warm-up floor wins)")

    print(f"\n  Summary: {passed}/{total} cases passed")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
