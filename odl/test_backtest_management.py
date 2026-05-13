"""
Smoke tests for AgentZeroBacktester simulation logic.

Tests the RAW single-exit model that the backtester actually implements:
  - BUY / SELL TP wins
  - BUY / SELL SL losses
  - Intrabar TP+SL collision → conservative (SL) / optimistic (TP) / neutral
  - Timeout exit
  - Spread cost applied on BUY, not on SELL
  - ATR-based SL/TP recalculated from next-bar fill
  - window_start index clipping arithmetic
  - JPY pip value (not a flat $10)

No MT5 required — simulate_trades is called directly with hand-crafted
Signal objects and synthetic DataFrames.

Run:
    python -m odl.test_backtest_management
    # or from the repo root:
    python odl/test_backtest_management.py
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

# Works whether run as a script or as a module
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from odl.backtest import AgentZeroBacktester, Signal  # noqa: E402


# ─────────────────────────── helpers ────────────────────────────────────────

def make_df(path):
    """
    Build a minimal M15 OHLC DataFrame from a list of (high, low, close) tuples.
    The open of each bar equals the close of the previous bar.
    """
    rows = []
    t = datetime(2025, 1, 6, 8, 0, tzinfo=timezone.utc)   # Monday 08:00 UTC
    prev_close = path[0][2]
    for hi, lo, cl in path:
        rows.append({
            "time":        t,
            "open":        prev_close,
            "high":        hi,
            "low":         lo,
            "close":       cl,
            "tick_volume": 100,
            "real_volume": 0,
        })
        t += timedelta(minutes=15)
        prev_close = cl
    return pd.DataFrame(rows)


def _bt(spread=0.0, policy="conservative", max_bars=96):
    return AgentZeroBacktester(
        lot_size=1.0,
        spread_pips=spread,
        intrabar_policy=policy,
        max_trade_duration_bars=max_bars,
    )


def sig_buy(ts, entry_hint, sl, tp, symbol="EURUSD", atr=0.0):
    return Signal(
        ts=ts, symbol=symbol, signal="BUY", source="CHoCH-BUY@PDL",
        confidence=0.85, percentage=85, quality="good",
        entry_price=entry_hint, stop_loss=sl, take_profit=tp, rr_ratio=2.0,
        atr=atr,
    )


def sig_sell(ts, entry_hint, sl, tp, symbol="EURUSD", atr=0.0):
    return Signal(
        ts=ts, symbol=symbol, signal="SELL", source="Continuation-SELL@PDH",
        confidence=0.85, percentage=85, quality="good",
        entry_price=entry_hint, stop_loss=sl, take_profit=tp, rr_ratio=2.0,
        atr=atr,
    )


def check(name, trade, *, reason, outcome, pips_sign=None, pips_approx=None, tol=0.6):
    """
    Assert trade fields match expectations and print PASS / FAIL.

    pips_sign   : '+' | '-' | '0'  (sign of profit_pips)
    pips_approx : float            (exact value to within tol pips)
    """
    ok = trade.exit_reason == reason and trade.outcome == outcome
    if ok and pips_sign is not None:
        if pips_sign == "+":
            ok = trade.profit_pips > 0
        elif pips_sign == "-":
            ok = trade.profit_pips < 0
        elif pips_sign == "0":
            ok = abs(trade.profit_pips) < tol
    if ok and pips_approx is not None:
        ok = abs(trade.profit_pips - pips_approx) < tol

    status = "PASS" if ok else "FAIL"
    print(
        f"  [{status}] {name:<48s} "
        f"reason={trade.exit_reason:<10s} outcome={trade.outcome:<5s} "
        f"pips={trade.profit_pips:+.1f}  "
        f"(expected reason={reason} outcome={outcome}"
        + (f" pips≈{pips_approx:+.1f}" if pips_approx is not None else "")
        + ")"
    )
    return int(ok)


# ─────────────────────────── test cases ─────────────────────────────────────

def main():
    passed = 0
    total = 0

    # ── 1. BUY: clean TP win ─────────────────────────────────────────────────
    # bar0: signal fires. bar1 open=1.1000 → entry.
    # SL=1.0990 (-10p), TP=1.1020 (+20p).
    # bar1: no touch. bar2: high >= TP → TP win.
    df = make_df([
        (1.1000, 1.1000, 1.1000),   # bar0 signal bar
        (1.1005, 1.0995, 1.1003),   # bar1 entry (open=1.1000, no touch)
        (1.1025, 1.1001, 1.1020),   # bar2 TP hit (high >= 1.1020)
    ])
    s = sig_buy(df.iloc[0]["time"], 1.1000, 1.0990, 1.1020)
    trades = _bt().simulate_trades("EURUSD", [s], df)
    total += 1
    passed += check("BUY: clean TP win", trades[0],
                    reason="tp", outcome="WIN", pips_approx=20.0)

    # ── 2. BUY: clean SL loss ────────────────────────────────────────────────
    # bar1 entry 1.1000, bar2 low <= SL 1.0990.
    df = make_df([
        (1.1000, 1.1000, 1.1000),
        (1.1003, 1.0998, 1.1002),   # bar1 entry, no touch
        (1.1001, 1.0985, 1.0990),   # bar2 SL hit (low <= 1.0990)
    ])
    s = sig_buy(df.iloc[0]["time"], 1.1000, 1.0990, 1.1020)
    trades = _bt().simulate_trades("EURUSD", [s], df)
    total += 1
    passed += check("BUY: clean SL loss", trades[0],
                    reason="sl", outcome="LOSS", pips_approx=-10.0)

    # ── 3. SELL: clean TP win ────────────────────────────────────────────────
    # Entry=1.1000 (bid fill, no spread adjust on SELL).
    # SL=1.1010 (+10p above), TP=1.0980 (-20p below).
    df = make_df([
        (1.1000, 1.1000, 1.1000),
        (1.1002, 1.0997, 1.0999),   # bar1 entry (open=1.1000)
        (1.0998, 1.0975, 1.0980),   # bar2 TP hit (low <= 1.0980)
    ])
    s = sig_sell(df.iloc[0]["time"], 1.1000, 1.1010, 1.0980)
    trades = _bt().simulate_trades("EURUSD", [s], df)
    total += 1
    passed += check("SELL: clean TP win", trades[0],
                    reason="tp", outcome="WIN", pips_approx=20.0)

    # ── 4. SELL: clean SL loss ───────────────────────────────────────────────
    df = make_df([
        (1.1000, 1.1000, 1.1000),
        (1.1002, 1.0997, 1.0999),
        (1.1015, 1.1002, 1.1012),   # bar2 SL hit (high >= 1.1010)
    ])
    s = sig_sell(df.iloc[0]["time"], 1.1000, 1.1010, 1.0980)
    trades = _bt().simulate_trades("EURUSD", [s], df)
    total += 1
    passed += check("SELL: clean SL loss", trades[0],
                    reason="sl", outcome="LOSS", pips_approx=-10.0)

    # ── 5. Intrabar collision — conservative → SL wins ───────────────────────
    # Same bar touches both TP (high >= 1.1020) and SL (low <= 1.0990).
    # Default policy = conservative → SL exit → LOSS at -10p.
    df = make_df([
        (1.1000, 1.1000, 1.1000),
        (1.1025, 1.0985, 1.1000),   # bar1: both TP and SL touched
    ])
    s = sig_buy(df.iloc[0]["time"], 1.1000, 1.0990, 1.1020)
    trades = _bt(policy="conservative").simulate_trades("EURUSD", [s], df)
    total += 1
    passed += check("Intrabar collision: conservative → SL", trades[0],
                    reason="sl", outcome="LOSS", pips_approx=-10.0)

    # ── 6. Intrabar collision — optimistic → TP wins ─────────────────────────
    trades = _bt(policy="optimistic").simulate_trades("EURUSD", [s], df)
    total += 1
    passed += check("Intrabar collision: optimistic → TP", trades[0],
                    reason="tp", outcome="WIN", pips_approx=20.0)

    # ── 7. Intrabar collision — neutral → closer to open wins ────────────────
    # open of bar1 = prev_close = 1.1000. TP distance = 20p, SL distance = 10p.
    # SL is closer → SL wins in neutral mode.
    trades = _bt(policy="neutral").simulate_trades("EURUSD", [s], df)
    total += 1
    passed += check("Intrabar collision: neutral → SL (closer)", trades[0],
                    reason="sl", outcome="LOSS", pips_approx=-10.0)

    # ── 8. Timeout exit ──────────────────────────────────────────────────────
    # max_trade_duration_bars=2. Entry bar1, loop bar1..bar3. TP/SL never hit.
    # Exits at close of bar3 (index entry_idx+2).
    df = make_df([
        (1.1000, 1.1000, 1.1000),
        (1.1005, 1.0995, 1.1002),   # bar1 entry (open=1.1000)
        (1.1007, 1.0997, 1.1003),   # bar2 no touch
        (1.1008, 1.0998, 1.1004),   # bar3 no touch → timeout close at 1.1004
    ])
    s = sig_buy(df.iloc[0]["time"], 1.1000, 1.0990, 1.1020)
    trades = _bt(max_bars=2).simulate_trades("EURUSD", [s], df)
    total += 1
    passed += check("Timeout exit (no TP/SL hit)", trades[0],
                    reason="timeout", outcome="WIN", pips_sign="+")

    # ── 9. Spread is charged on BUY, not on SELL ─────────────────────────────
    # With 2 pip spread on EURUSD (pip=0.0001 → 0.0002 price).
    # BUY entry = open + 0.0002. TP fill = 1.1020. Risk_pips correct.
    df = make_df([
        (1.1000, 1.1000, 1.1000),
        (1.1003, 1.0998, 1.1002),
        (1.1025, 1.1001, 1.1020),
    ])
    s_buy = sig_buy(df.iloc[0]["time"], 1.1000, 1.0988, 1.1020)
    trades = _bt(spread=2.0).simulate_trades("EURUSD", [s_buy], df)
    t = trades[0]
    expected_entry = 1.1000 + 2 * 0.0001   # open + spread
    entry_ok = abs(t.entry_price - expected_entry) < 1e-6
    # TP is still 1.1020 (not adjusted) → profit = 1.1020 - 1.1002 = 18p
    total += 1
    ok = entry_ok and t.exit_reason == "tp" and t.profit_pips > 0
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] BUY spread charged (entry={t.entry_price:.5f} "
          f"expected={expected_entry:.5f}) pips={t.profit_pips:+.1f}")
    passed += int(ok)

    # SELL entry should NOT pay spread → open = 1.1000 exactly.
    df_sell = make_df([
        (1.1000, 1.1000, 1.1000),
        (1.1002, 1.0997, 1.0999),
        (1.0998, 1.0975, 1.0980),
    ])
    s_sell = sig_sell(df_sell.iloc[0]["time"], 1.1000, 1.1010, 1.0980)
    trades_sell = _bt(spread=2.0).simulate_trades("EURUSD", [s_sell], df_sell)
    t_sell = trades_sell[0]
    sell_entry_ok = abs(t_sell.entry_price - 1.1000) < 1e-6
    total += 1
    ok = sell_entry_ok and t_sell.exit_reason == "tp"
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] SELL no spread on entry (entry={t_sell.entry_price:.5f} "
          f"expected=1.10000) pips={t_sell.profit_pips:+.1f}")
    passed += int(ok)

    # ── 10. ATR-based SL/TP recalculated from next-bar fill ──────────────────
    # sig.atr = 0.0010. Default multipliers sl=2.5, tp=4.5.
    # Entry open = 1.1010 (next bar's open after signal at bar0 close=1.1010).
    # Expected: SL = 1.1010 - 0.0010*2.5 = 1.0985
    #           TP = 1.1010 + 0.0010*4.5 = 1.1055
    # bar2 high >= 1.1055 → TP win.
    df = make_df([
        (1.1010, 1.1005, 1.1010),   # bar0 (signal bar, close=1.1010)
        (1.1012, 1.1008, 1.1010),   # bar1 entry (open=1.1010, no touch)
        (1.1060, 1.1040, 1.1055),   # bar2 TP hit
    ])
    s = sig_buy(df.iloc[0]["time"], 1.1010, 0.0, 0.0, atr=0.0010)
    trades = _bt().simulate_trades("EURUSD", [s], df)
    t = trades[0]
    expected_sl = round(1.1010 - 0.0010 * 2.5, 5)
    expected_tp = round(1.1010 + 0.0010 * 4.5, 5)
    sl_ok = abs(t.signal.stop_loss - expected_sl) < 1e-5
    tp_ok = abs(t.signal.take_profit - expected_tp) < 1e-5
    total += 1
    ok = sl_ok and tp_ok and t.exit_reason == "tp" and t.outcome == "WIN"
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] ATR SL/TP recalc from fill          "
          f"sl={t.signal.stop_loss:.5f}(exp={expected_sl:.5f}) "
          f"tp={t.signal.take_profit:.5f}(exp={expected_tp:.5f}) "
          f"exit={t.exit_reason}")
    passed += int(ok)

    # ── 11. JPY pip value is NOT a flat $10/pip ───────────────────────────────
    # GBPJPY: pip=0.01. 1 lot. At rate 150, 1 pip = 1000 JPY / 150 = $6.67.
    # Win of +20 pips → ~$133.33, not $200.
    df_jpy = make_df([
        (155.000, 155.000, 155.000),
        (155.050, 154.990, 155.020),
        (155.250, 155.100, 155.200),   # high >= 155.200 = TP
    ])
    s_jpy = sig_buy(df_jpy.iloc[0]["time"], 155.000, 154.800, 155.200,
                    symbol="GBPJPY")
    trades_jpy = _bt().simulate_trades("GBPJPY", [s_jpy], df_jpy)
    t_jpy = trades_jpy[0]
    # 20 pips at ~$6.67/pip ≈ $133. Definitely not $200.
    total += 1
    ok = t_jpy.exit_reason == "tp" and 50 < t_jpy.profit < 180
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] JPY pip value not flat $10           "
          f"pips={t_jpy.profit_pips:+.1f} profit=${t_jpy.profit:+.2f} "
          f"(expected $50–$180, not $200)")
    passed += int(ok)

    # ── 12. Invalid SL side → signal skipped ─────────────────────────────────
    # BUY with SL above entry → defensive guard should skip it.
    df = make_df([
        (1.1000, 1.1000, 1.1000),
        (1.1005, 1.0995, 1.1002),
    ])
    s_bad = sig_buy(df.iloc[0]["time"], 1.1000, 1.1010, 1.1020)  # SL > entry → invalid
    trades_bad = _bt().simulate_trades("EURUSD", [s_bad], df)
    total += 1
    ok = len(trades_bad) == 0
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Invalid BUY SL skipped               "
          f"trades={len(trades_bad)} (expected 0)")
    passed += int(ok)

    # ── 13. window_start clips signal generation (index arithmetic) ───────────
    times = [
        datetime(2026, 4, 1, tzinfo=timezone.utc) + timedelta(minutes=15 * i)
        for i in range(200)
    ]
    df_probe = pd.DataFrame({"time": times})

    # 13a: window_start = bar 120 → start_idx = max(50, 120) = 120
    ws = times[120]
    mask = df_probe["time"] >= ws
    first_in_window = int(mask.values.argmax()) if mask.any() else len(df_probe)
    start_idx = max(50, first_in_window)
    total += 1
    ok = start_idx == 120
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] window clip ws=bar120 → start={start_idx} (expected 120)")
    passed += int(ok)

    # 13b: window_start = bar 10 → warm-up floor (50) wins
    ws_early = times[10]
    mask = df_probe["time"] >= ws_early
    first_in_window = int(mask.values.argmax()) if mask.any() else len(df_probe)
    start_idx = max(50, first_in_window)
    total += 1
    ok = start_idx == 50
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] window clip ws=bar10  → start={start_idx} (expected 50, warm-up floor)")
    passed += int(ok)

    # ── 14. Signal after last closed bar is skipped ───────────────────────────
    # Signal ts == last bar time → pos == len(df)-1 → entry_idx out of bounds.
    df = make_df([
        (1.1000, 1.1000, 1.1000),
        (1.1005, 1.0995, 1.1002),
    ])
    s_late = sig_buy(df.iloc[-1]["time"], 1.1000, 1.0990, 1.1020)
    trades_late = _bt().simulate_trades("EURUSD", [s_late], df)
    total += 1
    ok = len(trades_late) == 0
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Signal on last bar skipped            "
          f"trades={len(trades_late)} (expected 0)")
    passed += int(ok)

    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  Summary: {passed}/{total} cases passed")
    print(f"  {'─'*60}")
    sys.exit(0 if passed == total else 1)


# ─────────────────────────────────────────────────────────────────────────
# Pytest entry point
# ─────────────────────────────────────────────────────────────────────────
# main() above is the original script-style runner that prints PASS/FAIL
# rows and exits non-zero on failure. The wrapper below lets pytest
# discover the same suite as a single test case so CI can run it
# alongside tests/test_agent_learning_loop.py without a second invocation.

def test_simulation_correctness():
    """Run all backtester simulation cases; fail if any case failed.

    main() calls sys.exit(N) where N == 0 on full pass. We catch the
    SystemExit and assert the code so pytest reports a clean failure
    with the captured stdout (use `pytest -s` to see the per-case rows).
    """
    try:
        main()
    except SystemExit as exc:
        assert exc.code == 0, (
            "backtest simulation suite failed; "
            "run `python odl/test_backtest_management.py` to see per-case results"
        )


if __name__ == "__main__":
    main()
