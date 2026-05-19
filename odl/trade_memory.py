# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  trade_memory.py — SQLite-backed cumulative backtest memory              ║
# ║                                                                          ║
# ║  Stores every backtest run's trade-level results, feature importances,   ║
# ║  and tuned strategy params.  The backtest engine writes here after each  ║
# ║  run; the dashboard reads it via /api/backtest/memory; the live bot      ║
# ║  reads tuned_params via agent_learning_loop.py.                          ║
# ║                                                                          ║
# ║  Tables                                                                  ║
# ║  ──────                                                                  ║
# ║  backtest_runs   — one row per (symbol, run), summary statistics         ║
# ║  trades          — one row per trade in that run                         ║
# ║  run_importances — one row per feature per run,  FK → backtest_runs      ║
# ║  run_tuned       — one row per param per run,  FK → backtest_runs        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

from __future__ import annotations

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

log = logging.getLogger(__name__)

# ── Schema ───────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS backtest_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,
    run_at          TEXT    NOT NULL,           -- ISO-8601 UTC
    requested_days  INTEGER,
    total_trades    INTEGER DEFAULT 0,
    wins            INTEGER DEFAULT 0,
    losses          INTEGER DEFAULT 0,
    win_rate        REAL,
    total_pnl       REAL    DEFAULT 0,
    avg_pnl         REAL    DEFAULT 0,
    avg_win_pips    REAL,
    avg_loss_pips   REAL,
    profit_factor   REAL,
    avg_rr          REAL,
    avg_duration_m  REAL
);

CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL REFERENCES backtest_runs(id),
    symbol          TEXT    NOT NULL,
    signal          TEXT,                      -- BUY | SELL
    source          TEXT,                      -- signal source / environment
    quality         TEXT,
    confidence      REAL,
    entry_time      TEXT,
    exit_time       TEXT,
    entry_price     REAL,
    exit_price      REAL,
    sl              REAL,
    tp              REAL,
    atr             REAL,
    poc             REAL,
    risk_pips       REAL,
    profit_pips     REAL,
    profit_usd      REAL,
    rr_achieved     REAL,
    exit_reason     TEXT,                      -- tp | sl | timeout
    outcome         TEXT,                      -- WIN | LOSS | BE
    duration_m      INTEGER,
    component_scores_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_trades_run    ON trades(run_id);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_entry  ON trades(entry_time);

CREATE TABLE IF NOT EXISTS run_importances (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL REFERENCES backtest_runs(id),
    symbol      TEXT    NOT NULL,
    feature     TEXT    NOT NULL,
    importance  REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS run_tuned (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id  INTEGER NOT NULL REFERENCES backtest_runs(id),
    symbol  TEXT    NOT NULL,
    param   TEXT    NOT NULL,
    value   REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tuned_run    ON run_tuned(run_id);
CREATE INDEX IF NOT EXISTS idx_tuned_symbol ON run_tuned(symbol);
"""

# Default DB path — same directory as this file's package root
_DEFAULT_DB = Path(__file__).parent.parent / "trade_memory.db"


class TradeMemory:
    """Persistent, append-only store for backtest runs and trade-level data."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path = str(db_path or _DEFAULT_DB)
        self._init_schema()

    # ── Internal helpers ─────────────────────────────────────────────────────

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _ping(conn: sqlite3.Connection) -> bool:
        try:
            conn.execute("SELECT 1")
            return True
        except Exception:
            return False

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    # ── Write ────────────────────────────────────────────────────────────────

    def save_run(
        self,
        symbol: str,
        results: Dict[str, Any],
        trades: list,
        feature_importance: Optional[Dict[str, Any]] = None,
        tuned_params: Optional[Dict[str, float]] = None,
    ) -> int:
        """Persist one backtest run.  Returns the new run_id."""

        def _f(key: str, default: Any = None) -> Any:
            v = results.get(key)
            if v is None:
                return default
            try:
                return float(v)
            except (TypeError, ValueError):
                return default

        symbol = symbol.upper()
        run_at = datetime.now(timezone.utc).isoformat()
        total = len([t for t in trades if getattr(t, "outcome", None) in ("WIN", "LOSS", "BE")])
        wins   = len([t for t in trades if getattr(t, "outcome", None) == "WIN"])
        losses = len([t for t in trades if getattr(t, "outcome", None) == "LOSS"])
        wr     = (wins / (wins + losses)) if (wins + losses) > 0 else 0.0
        pip_size = 0.01 if "JPY" in symbol else 0.0001

        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO backtest_runs
                    (symbol, run_at, requested_days,
                     total_trades, wins, losses, win_rate,
                     total_pnl, avg_pnl, avg_win_pips, avg_loss_pips,
                     profit_factor, avg_rr, avg_duration_m)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    symbol, run_at, results.get("requested_days"),
                    total, wins, losses, round(wr, 4),
                    _f("total_pnl", 0.0),
                    _f("avg_pnl", 0.0),
                    _f("avg_win_pips", 0.0),
                    abs(float(_f("avg_loss_pips", 0.0) or 0.0)),
                    _f("profit_factor", 0.0),
                    _f("avg_rr_achieved", 0.0),
                    _f("avg_duration_minutes", 0.0),
                ),
            )
            run_id = cur.lastrowid

            # ── Per-trade rows ───────────────────────────────────────────────
            for t in trades:
                sig = getattr(t, "signal", None)
                if sig is None:
                    continue
                outcome = getattr(t, "outcome", "")
                if outcome not in ("WIN", "LOSS", "BE"):
                    continue

                entry_price = float(getattr(t, "entry_price", 0.0) or 0.0)
                sl_price    = float(getattr(sig, "stop_loss",   0.0) or 0.0)
                risk_pips   = (abs(entry_price - sl_price) / pip_size) if pip_size else 0.0
                cs          = getattr(sig, "component_scores", None) or {}

                conn.execute(
                    """
                    INSERT INTO trades
                        (run_id, symbol, signal, source, quality, confidence,
                         entry_time, exit_time, entry_price, exit_price,
                         sl, tp, atr, poc, risk_pips,
                         profit_pips, profit_usd, rr_achieved,
                         exit_reason, outcome, duration_m,
                         component_scores_json)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        run_id,
                        symbol,
                        str(getattr(sig, "signal", "") or ""),
                        str(getattr(sig, "source", "") or ""),
                        str(getattr(sig, "quality", "") or ""),
                        float(getattr(sig, "confidence", 0.0) or 0.0),
                        t.entry_time.isoformat() if t.entry_time else None,
                        t.exit_time.isoformat()  if t.exit_time  else None,
                        entry_price,
                        float(getattr(t,   "exit_price",  0.0) or 0.0),
                        sl_price,
                        float(getattr(sig, "take_profit", 0.0) or 0.0),
                        float(getattr(sig, "atr",         0.0) or 0.0),
                        float(getattr(sig, "poc",         0.0) or 0.0),
                        risk_pips,
                        float(getattr(t, "profit_pips", 0.0) or 0.0),
                        float(getattr(t, "profit",      0.0) or 0.0),
                        float(getattr(t, "rr_achieved", 0.0) or 0.0),
                        str(getattr(t, "exit_reason", "") or ""),
                        outcome,
                        int(getattr(t, "duration_minutes", 0) or 0),
                        json.dumps(cs) if cs else None,
                    ),
                )

            # ── Feature importances ──────────────────────────────────────────
            if feature_importance and "importances" in feature_importance:
                for entry in feature_importance["importances"]:
                    if not isinstance(entry, dict):
                        continue
                    fname = entry.get("feature")
                    score = entry.get("importance")
                    if fname and score is not None:
                        conn.execute(
                            "INSERT INTO run_importances (run_id, symbol, feature, importance) VALUES (?,?,?,?)",
                            (run_id, symbol, fname, float(score)),
                        )

            # ── Tuned params ─────────────────────────────────────────────────
            if tuned_params:
                for param, value in tuned_params.items():
                    conn.execute(
                        "INSERT INTO run_tuned (run_id, symbol, param, value) VALUES (?,?,?,?)",
                        (run_id, symbol, param, float(value)),
                    )

        log.info(
            "trade_memory: saved run #%d for %s – %d trades, %d importances, %d tuned params",
            run_id, symbol, total,
            len(feature_importance.get("importances", []) if feature_importance else []),
            len(tuned_params or {}),
        )
        return run_id

    # ── Aggregate ────────────────────────────────────────────────────────────

    def aggregate_insights(
        self,
        symbol: str,
        min_runs: int = 1,
        max_age_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Aggregate importances and tuned_params across every stored run for
        `symbol`, returning a dict in exactly the shape that
        `agent_learning_loop.AgentLearningLoop` expects:

            {
              "importances":  {feature: weighted_avg_importance, ...},
              "tuned_params": {param:   latest_value, ...},
              "trade_count":  total_decided_trades_across_all_runs,
              "win_rate":     overall_win_rate,
              "backtest_at":  most_recent_run_timestamp
            }

        Strategy
        --------
        • importances   — weighted average across runs, weight = run.total_trades.
                          Weighting by trade count means a 200-trade run shapes
                          the signal more than a 5-trade run.  Features with a
                          weighted average below 0.01 are dropped (noise).
        • tuned_params  — most recent run that HAS tuned_params.  Thin runs
                          (e.g. EURJPY with only 4 trades) never produce
                          tuned_params, so we look back to find the last run
                          that did. Falls back to the plain latest run if no
                          run ever produced tuned params.
        • win_rate      — derived from the SUM of wins / SUM of decided trades
                          across all runs, not an average-of-averages.

        Returns an empty dict when fewer than `min_runs` runs exist for `symbol`
        so the caller can decide whether to fall back to the single-run values.
        """
        age_clause = ""
        age_params: List[Any] = [symbol]
        if max_age_days is not None:
            from datetime import timedelta
            cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat()
            age_clause = " AND run_at >= ?"
            age_params.append(cutoff)

        with self._conn() as conn:
            # ── How many qualifying runs? ────────────────────────────────
            row = conn.execute(
                f"SELECT COUNT(*) AS n FROM backtest_runs WHERE symbol=?{age_clause}",
                age_params,
            ).fetchone()
            n_runs = row["n"] if row else 0
            if n_runs < min_runs:
                log.debug(
                    "aggregate_insights(%s): only %d runs, need %d — returning empty",
                    symbol, n_runs, min_runs,
                )
                return {}

            # ── Weighted-average importances ─────────────────────────────
            rows = conn.execute(
                f"""
                SELECT ri.feature,
                       SUM(ri.importance * br.total_trades) AS weighted_sum,
                       SUM(br.total_trades)                 AS weight_total
                FROM   run_importances ri
                JOIN   backtest_runs br ON br.id = ri.run_id
                WHERE  br.symbol=?{age_clause}
                  AND  br.total_trades > 0
                GROUP BY ri.feature
                """,
                age_params,
            ).fetchall()
            importances: Dict[str, float] = {}
            for r in rows:
                wt = r["weight_total"] or 0.0
                if wt > 0:
                    avg = r["weighted_sum"] / wt
                    if avg >= 0.01:   # drop noise below 1 %
                        importances[r["feature"]] = round(float(avg), 4)

            # ── Latest tuned_params ──────────────────────────────────────
            # Use the most recent run that HAS tuned_params.
            # Thin runs (e.g. EURJPY ~4 trades) never produce
            # tuned_params, so taking the plain latest run returns
            # nothing. We look further back for the last good run.
            latest_run = conn.execute(
                f"""
                SELECT br.id
                FROM   backtest_runs br
                WHERE  br.symbol=?{age_clause}
                  AND  EXISTS (
                        SELECT 1 FROM run_tuned rt
                        WHERE  rt.run_id = br.id
                       )
                ORDER  BY br.id DESC
                LIMIT  1
                """,
                age_params,
            ).fetchone()
            if not latest_run:
                latest_run = conn.execute(
                    f"""
                    SELECT id FROM backtest_runs
                    WHERE  symbol=?{age_clause}
                    ORDER  BY id DESC LIMIT 1
                    """,
                    age_params,
                ).fetchone()
            tuned: Dict[str, float] = {}
            if latest_run:
                t_rows = conn.execute(
                    "SELECT param, value FROM run_tuned WHERE run_id=?",
                    (latest_run["id"],),
                ).fetchall()
                tuned = {r["param"]: round(float(r["value"]), 4) for r in t_rows}

            # ── Aggregate win-rate and trade count ───────────────────────
            agg = conn.execute(
                f"""
                SELECT SUM(wins)         AS total_wins,
                       SUM(losses)       AS total_losses,
                       SUM(total_trades) AS total_trades,
                       MAX(run_at)       AS latest_run_at
                FROM   backtest_runs
                WHERE  symbol=?{age_clause}
                """,
                age_params,
            ).fetchone()

        total_wins   = int(agg["total_wins"]   or 0)
        total_losses = int(agg["total_losses"] or 0)
        total_trades = int(agg["total_trades"] or 0)
        decided      = total_wins + total_losses
        win_rate     = (total_wins / decided) if decided > 0 else 0.0
        latest_at    = agg["latest_run_at"] or datetime.now(timezone.utc).isoformat()

        out: Dict[str, Any] = {}
        if importances:
            out["importances"]  = importances
        if tuned:
            out["tuned_params"] = tuned
        if total_trades > 0:
            out["trade_count"] = total_trades
            out["win_rate"]    = round(win_rate, 4)
        out["backtest_at"] = latest_at
        out["n_runs"]      = n_runs       # informational; live loop ignores it

        return out

    # ── Query helpers (diagnostics / export) ────────────────────────────────

    def query_trades(
        self,
        symbol: Optional[str] = None,
        outcome: Optional[str] = None,
        min_run_id: Optional[int] = None,
        limit: int = 5000,
    ) -> List[Dict[str, Any]]:
        """
        Return trade rows as plain dicts.  Filters are ANDed.

        Typical uses:
            mem.query_trades(symbol="EURUSD")           # all EURUSD trades
            mem.query_trades(outcome="WIN")             # all winners
            mem.query_trades(symbol="GBPJPY", min_run_id=5)
        """
        clauses: List[str] = []
        params: List[Any] = []
        if symbol:
            clauses.append("symbol=?"); params.append(symbol.upper())
        if outcome:
            clauses.append("outcome=?"); params.append(outcome.upper())
        if min_run_id is not None:
            clauses.append("run_id>=?"); params.append(int(min_run_id))
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(int(limit))

        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM trades {where} ORDER BY entry_time DESC LIMIT ?",
                params,
            ).fetchall()
        return [dict(r) for r in rows]

    def query_runs(
        self,
        symbol: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """Return run summary rows as plain dicts."""
        where = "WHERE symbol=?" if symbol else ""
        params: List[Any] = ([symbol.upper()] if symbol else []) + [int(limit)]
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM backtest_runs {where} ORDER BY id DESC LIMIT ?",
                params,
            ).fetchall()
        return [dict(r) for r in rows]

    def get_cumulative_stats(self, symbol: str) -> Dict[str, Any]:
        """
        High-level cumulative stats for the memory panel dashboard widget.

        Returns dict with keys:
            n_runs, total_trades, wins, losses, win_rate,
            total_pnl, avg_pnl, earliest_run, latest_run,
            tuned_params  (from most-recent run that has them, or {})
        """
        symbol = symbol.upper()
        with self._conn() as conn:
            runs = conn.execute(
                """
                SELECT COUNT(*)          AS n_runs,
                       SUM(total_trades) AS total_trades,
                       SUM(wins)         AS wins,
                       SUM(losses)       AS losses,
                       SUM(total_pnl)    AS total_pnl,
                       AVG(avg_pnl)      AS avg_pnl,
                       MIN(run_at)       AS earliest,
                       MAX(run_at)       AS latest
                FROM   backtest_runs
                WHERE  symbol=?
                """,
                (symbol,),
            ).fetchone()

            decided = int(runs["wins"] or 0) + int(runs["losses"] or 0)
            wr = (int(runs["wins"] or 0) / decided) if decided else 0.0

            # Latest run WITH tuned params
            trow = conn.execute(
                """
                SELECT br.id FROM backtest_runs br
                WHERE  br.symbol=?
                  AND  EXISTS (SELECT 1 FROM run_tuned rt WHERE rt.run_id = br.id)
                ORDER  BY br.id DESC LIMIT 1
                """,
                (symbol,),
            ).fetchone()
            tuned: Dict[str, float] = {}
            if trow:
                t_rows = conn.execute(
                    "SELECT param, value FROM run_tuned WHERE run_id=?",
                    (trow["id"],),
                ).fetchall()
                tuned = {r["param"]: round(float(r["value"]), 4) for r in t_rows}

            avg_win_rate_row = conn.execute(
                """
                SELECT AVG(win_rate)       AS avg_win_rate,
                       AVG(profit_factor)  AS avg_pf,
                       AVG(avg_rr)         AS avg_rr
                FROM   backtest_runs
                WHERE  symbol=?
                """,
                (symbol,),
            ).fetchone()

        return {
            "n_runs":       int(runs["n_runs"]       or 0),
            "total_trades": int(runs["total_trades"] or 0),
            "wins":         int(runs["wins"]         or 0),
            "losses":       int(runs["losses"]       or 0),
            "win_rate":     round(wr, 4),
            "total_pnl":    round(float(runs["total_pnl"] or 0), 2),
            "avg_pnl":      round(float(runs["avg_pnl"]   or 0), 2),
            "earliest_run": runs["earliest"],
            "latest_run":   runs["latest"],
            "tuned_params": tuned,
            "avg_win_rate": round(float((avg_win_rate_row["avg_win_rate"] or 0)), 4),
            "avg_pf":       round(float((avg_win_rate_row["avg_pf"]       or 0)), 2),
            "avg_rr":       round(float((avg_win_rate_row["avg_rr"]       or 0)), 2),
        }

    def export_trades_csv(self, symbol: str, filepath: str) -> int:
        """Export all trades for `symbol` to a CSV file. Returns row count."""
        import csv
        rows = self.query_trades(symbol=symbol, limit=100_000)
        if not rows:
            return 0
        with open(filepath, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return len(rows)

    def __enter__(self) -> "TradeMemory":
        return self

    def __exit__(self, *_: Any) -> None:
        pass  # connections are per-operation; nothing to close here


# ── Module-level singleton ───────────────────────────────────────────────────

_memory_instance: Optional[TradeMemory] = None


def get_memory(db_path: Optional[Path] = None) -> TradeMemory:
    """Return (or lazily create) the module-level TradeMemory singleton."""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = TradeMemory(db_path)
    return _memory_instance
