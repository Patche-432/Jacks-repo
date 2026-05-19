"""
agent_memory.py — Persistent memory bank for Agent Zero (Orchestrator).

PURPOSE
=======
The Orchestrator is stateless by design — each orchestrate() call gets a
fresh portfolio snapshot + bot verdicts and returns decisions with no
recollection of previous cycles. That's fine for individual trade
management, but it means the LLM can't:

  • Learn that "override_close on GBPJPY during Asian session" tends to
    cost money because the move continues anyway.
  • Notice that the portfolio daily-loss limit gets breached most on
    Tuesdays at 09:00 UTC and tighten its risk stance proactively.
  • Remember what it decided last cycle and why, so it doesn't flip-flop.

This module provides a SQLite-backed memory bank that:

  1. RECORDS every orchestrator decision (ticket, symbol, action taken,
     PnL outcome when the trade eventually closes, hour, session).

  2. SUMMARISES patterns into a compact "memory context" string that
     is prepended to each orchestrator prompt — small enough to fit in
     a 3-7B model's context window (<400 tokens).

  3. EXPOSES an update hook called by ai_pro.py when a position closes,
     so the memory contains real outcome labels (WIN / LOSS / BE).

DB location:  <repo_root>/agent_memory.db

Schema
------
  decisions
      id, ts, symbol, ticket, action_bot, action_taken, new_sl,
      bot_reason, orch_reason, session, hour_utc, dow

  outcomes
      id, decision_id FK, closed_at, outcome (WIN/LOSS/BE),
      profit_pips, profit_usd, duration_m

  pattern_cache
      id, generated_at, context_text   (latest summarised memory — 1 row)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

log = logging.getLogger("agent_memory")

# ── Schema ───────────────────────────────────────────────────────────────────
_DDL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS decisions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT    NOT NULL,           -- ISO-8601 UTC
    symbol          TEXT    NOT NULL,
    ticket          INTEGER NOT NULL,
    action_bot      TEXT    NOT NULL,           -- what the pair bot proposed
    action_taken    TEXT    NOT NULL,           -- what was actually executed
    new_sl          REAL,
    bot_reason      TEXT,
    orch_reason     TEXT,                       -- orchestrator's stated reason
    session         TEXT,                       -- London / NewYork / Asian / Overlap
    hour_utc        INTEGER,
    dow             INTEGER,                    -- 0=Mon..6=Sun
    trend_intact    INTEGER,                    -- 1=HH/HL intact, 0=broken, NULL=unknown
    structure_broken INTEGER                    -- 1=structure broken, 0=ok, NULL=unknown
);

CREATE INDEX IF NOT EXISTS idx_dec_symbol ON decisions(symbol);
CREATE INDEX IF NOT EXISTS idx_dec_ts     ON decisions(ts);
CREATE INDEX IF NOT EXISTS idx_dec_ticket ON decisions(ticket);

CREATE TABLE IF NOT EXISTS outcomes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id INTEGER REFERENCES decisions(id),
    ticket      INTEGER NOT NULL,
    closed_at   TEXT    NOT NULL,
    outcome     TEXT    NOT NULL,           -- WIN / LOSS / BE
    profit_pips REAL,
    profit_usd  REAL,
    duration_m  INTEGER
);

CREATE INDEX IF NOT EXISTS idx_out_ticket ON outcomes(ticket);

CREATE TABLE IF NOT EXISTS pattern_cache (
    id              INTEGER PRIMARY KEY CHECK (id = 1),
    generated_at    TEXT    NOT NULL,
    context_text    TEXT    NOT NULL
);
"""

# ── Session helper ────────────────────────────────────────────────────────────
def _session(hour_utc: int) -> str:
    if 7 <= hour_utc < 9:   return "London-open"
    if 9 <= hour_utc < 13:  return "London"
    if 13 <= hour_utc < 16: return "Overlap"
    if 16 <= hour_utc < 21: return "NewYork"
    return "Asian"


# ── Thread-local connections ──────────────────────────────────────────────────
_local = threading.local()


class AgentMemory:
    """
    Lightweight SQLite memory bank for the Agent Zero orchestrator.

    Typical usage
    -------------
    mem = get_memory()                              # module-level singleton

    # After orchestrate() returns, record each decision:
    mem.record_decisions(verdicts_in, decisions_out)

    # When a trade closes (called from ai_pro.py):
    mem.record_outcome(ticket=12345, outcome="WIN",
                       profit_pips=42.0, profit_usd=280.0, duration_m=95)

    # Build the context string for the next orchestrator prompt:
    ctx = mem.build_context()   # ≤400 tokens
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        if db_path is None:
            db_path = Path(__file__).resolve().parent / "agent_memory.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ── Internal ─────────────────────────────────────────────────────────────

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = getattr(_local, "agent_mem_conn", None)
        if conn is None or not self._ping(conn):
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=10,
                check_same_thread=False,
            )
            conn.row_factory = sqlite3.Row
            _local.agent_mem_conn = conn
        yield conn

    @staticmethod
    def _ping(conn: sqlite3.Connection) -> bool:
        try:
            conn.execute("SELECT 1")
            return True
        except Exception:
            return False

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(_DDL)
            # Migrate existing DBs that predate the price-action columns
            for col, typedef in (
                ("trend_intact",     "INTEGER"),
                ("structure_broken", "INTEGER"),
            ):
                try:
                    conn.execute(f"ALTER TABLE decisions ADD COLUMN {col} {typedef}")
                    conn.commit()
                except Exception:
                    pass  # column already exists

    # ── Write ─────────────────────────────────────────────────────────────────

    def record_decisions(
        self,
        verdicts_in: List[Dict[str, Any]],
        decisions_out: List[Dict[str, Any]],
    ) -> None:
        """
        Record what the bots proposed vs what the orchestrator executed.

        verdicts_in  : list from pair bots (action_bot, ticket, symbol…)
        decisions_out: list returned by Orchestrator.orchestrate()
        """
        now = datetime.now(timezone.utc)
        hour = now.hour
        dow  = now.weekday()
        sess = _session(hour)
        ts   = now.isoformat()

        # Index the outputs by ticket for O(1) lookup
        out_by_ticket: Dict[int, Dict] = {}
        for d in decisions_out:
            t = d.get("ticket")
            if t is not None:
                out_by_ticket[int(t)] = d

        rows = []
        for v in verdicts_in:
            ticket = v.get("ticket")
            if ticket is None:
                continue
            ticket = int(ticket)
            d = out_by_ticket.get(ticket, v)  # fallback: treated as approved

            ti = v.get("trend_intact")
            sb = v.get("structure_broken")
            rows.append((
                ts,
                str(v.get("symbol", "?")),
                ticket,
                str(v.get("action", "?")),
                str(d.get("action", v.get("action", "?"))),
                d.get("new_sl"),
                str(v.get("reason", ""))[:300],
                str(d.get("orchestrator_reason", ""))[:300],
                sess,
                hour,
                dow,
                (1 if ti else 0) if ti is not None else None,
                (1 if sb else 0) if sb is not None else None,
            ))

        if not rows:
            return

        with self._conn() as conn:
            conn.executemany(
                """INSERT INTO decisions
                   (ts, symbol, ticket, action_bot, action_taken,
                    new_sl, bot_reason, orch_reason, session, hour_utc, dow,
                    trend_intact, structure_broken)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )
            conn.commit()
        log.debug("agent_memory: recorded %d decisions", len(rows))

    def record_outcome(
        self,
        ticket: int,
        outcome: str,
        profit_pips: float = 0.0,
        profit_usd: float = 0.0,
        duration_m: int = 0,
    ) -> None:
        """
        Label the decision for this ticket with its actual trade outcome.
        Called from ai_pro.py whenever a position closes.

        outcome: "WIN" | "LOSS" | "BE"
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        outcome = outcome.upper().strip()

        with self._conn() as conn:
            # Find the most recent decision for this ticket
            row = conn.execute(
                "SELECT id FROM decisions WHERE ticket=? ORDER BY id DESC LIMIT 1",
                (int(ticket),),
            ).fetchone()
            if row is None:
                # No decision recorded (possible if agent_memory was added
                # after the trade opened). Insert a stub.
                conn.execute(
                    """INSERT INTO decisions
                       (ts, symbol, ticket, action_bot, action_taken,
                        session, hour_utc, dow)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (now_iso, "?", int(ticket), "?", "?", "?", 0, 0),
                )
                conn.commit()
                row = conn.execute(
                    "SELECT id FROM decisions WHERE ticket=? ORDER BY id DESC LIMIT 1",
                    (int(ticket),),
                ).fetchone()

            conn.execute(
                """INSERT INTO outcomes
                   (decision_id, ticket, closed_at, outcome,
                    profit_pips, profit_usd, duration_m)
                   VALUES (?,?,?,?,?,?,?)""",
                (row["id"], int(ticket), now_iso, outcome,
                 float(profit_pips), float(profit_usd), int(duration_m)),
            )
            conn.commit()
        log.debug("agent_memory: outcome %s for ticket #%d", outcome, ticket)

    # ── Read / summarise ──────────────────────────────────────────────────────

    def build_context(self, max_age_days: int = 14) -> str:
        """
        Build a compact memory context string for the orchestrator prompt.

        Covers the last `max_age_days` days of decisions with outcomes.
        Returns a plain-text block of ≤400 tokens that summarises:
          • overall override / veto win rates
          • per-symbol patterns (which pairs benefit from overrides)
          • session / hour patterns (when overrides help vs hurt)
          • recent decisions (last 5, for recency bias avoidance)

        Returns empty string when there's not enough data to say anything
        useful (< 10 labelled outcomes).
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat()

        with self._conn() as conn:
            # ── 1. Overall override / veto stats ─────────────────────────
            override_rows = conn.execute(
                """
                SELECT d.action_taken, o.outcome, COUNT(*) AS n
                FROM decisions d
                JOIN outcomes o ON o.decision_id = d.id
                WHERE d.ts >= ?
                  AND d.action_taken IN ('close','hold')
                  AND d.action_bot   != d.action_taken
                GROUP BY d.action_taken, o.outcome
                """,
                (cutoff,),
            ).fetchall()

            # ── 2. Per-symbol outcome for overrides ──────────────────────
            sym_rows = conn.execute(
                """
                SELECT d.symbol,
                       d.action_taken,
                       o.outcome,
                       COUNT(*) AS n,
                       AVG(o.profit_pips) AS avg_pips
                FROM decisions d
                JOIN outcomes o ON o.decision_id = d.id
                WHERE d.ts >= ?
                  AND d.action_taken IN ('close','hold')
                  AND d.action_bot   != d.action_taken
                GROUP BY d.symbol, d.action_taken, o.outcome
                ORDER BY d.symbol, d.action_taken, o.outcome
                """,
                (cutoff,),
            ).fetchall()

            # ── 3. Session / hour patterns for overrides ─────────────────
            sess_rows = conn.execute(
                """
                SELECT d.session, o.outcome, COUNT(*) AS n,
                       AVG(o.profit_pips) AS avg_pips
                FROM decisions d
                JOIN outcomes o ON o.decision_id = d.id
                WHERE d.ts >= ?
                  AND d.action_taken = 'close'
                  AND d.action_bot   = 'hold'
                GROUP BY d.session, o.outcome
                ORDER BY d.session, o.outcome
                """,
                (cutoff,),
            ).fetchall()

            # ── 4. Price-action conditional win rates ────────────────────
            # "When trend intact at review time, did the trade win?"
            price_action_rows = conn.execute(
                """
                SELECT d.symbol,
                       d.trend_intact,
                       d.structure_broken,
                       o.outcome,
                       COUNT(*) AS n,
                       AVG(o.profit_pips) AS avg_pips
                FROM decisions d
                JOIN outcomes o ON o.decision_id = d.id
                WHERE d.ts >= ?
                  AND d.trend_intact    IS NOT NULL
                  AND d.structure_broken IS NOT NULL
                  AND o.outcome IN ('WIN','LOSS')
                GROUP BY d.symbol, d.trend_intact, d.structure_broken, o.outcome
                ORDER BY d.symbol, d.trend_intact DESC, d.structure_broken
                """,
                (cutoff,),
            ).fetchall()

            # ── 5. Recent decisions (last 5 with outcomes) ───────────────
            recent_rows = conn.execute(
                """
                SELECT d.ts, d.symbol, d.action_bot, d.action_taken,
                       d.orch_reason, o.outcome, o.profit_pips
                FROM decisions d
                LEFT JOIN outcomes o ON o.decision_id = d.id
                WHERE d.ts >= ?
                ORDER BY d.id DESC
                LIMIT 5
                """,
                (cutoff,),
            ).fetchall()

            # ── 6. Total labelled decisions ──────────────────────────────
            total_labelled = conn.execute(
                "SELECT COUNT(*) FROM outcomes o JOIN decisions d ON d.id=o.decision_id WHERE d.ts>=?",
                (cutoff,),
            ).fetchone()[0]

        if total_labelled < 5:
            return ""   # not enough data to say anything meaningful

        parts: List[str] = []

        # ── Override summary ─────────────────────────────────────────────
        ov_win = ov_loss = veto_win = veto_loss = 0
        for r in override_rows:
            taken, outcome, n = r["action_taken"], r["outcome"], r["n"]
            if taken == "close":
                if outcome == "WIN":   ov_win   += n
                elif outcome == "LOSS": ov_loss  += n
            elif taken == "hold":
                if outcome == "WIN":   veto_win  += n
                elif outcome == "LOSS": veto_loss += n

        ov_total   = ov_win + ov_loss
        veto_total = veto_win + veto_loss
        lines: List[str] = []
        if ov_total > 0:
            ov_wr = ov_win / ov_total
            lines.append(
                f"override_close outcomes (last {max_age_days}d): "
                f"{ov_win}W/{ov_loss}L of {ov_total} ({ov_wr:.0%} WR) — "
                + ("OVERRIDES TEND TO WIN" if ov_wr >= 0.55
                   else "OVERRIDES TEND TO LOSE" if ov_wr < 0.45
                   else "mixed")
            )
        if veto_total > 0:
            veto_wr = veto_win / veto_total
            lines.append(
                f"veto outcomes: {veto_win}W/{veto_loss}L of {veto_total} "
                f"({veto_wr:.0%} WR) — "
                + ("VETOES TEND TO WIN" if veto_wr >= 0.55
                   else "VETOES TEND TO LOSE" if veto_wr < 0.45
                   else "mixed")
            )
        if lines:
            parts.append("OVERRIDE/VETO TRACK RECORD:\n" + "\n".join(lines))

        # ── Per-symbol override patterns ─────────────────────────────────
        sym_agg: Dict[str, Dict] = {}
        for r in sym_rows:
            key = (r["symbol"], r["action_taken"])
            if key not in sym_agg:
                sym_agg[key] = {"win": 0, "loss": 0, "pips": []}
            if r["outcome"] == "WIN":
                sym_agg[key]["win"] += r["n"]
            elif r["outcome"] == "LOSS":
                sym_agg[key]["loss"] += r["n"]
            if r["avg_pips"] is not None:
                sym_agg[key]["pips"].append(float(r["avg_pips"]) * r["n"])

        sym_lines: List[str] = []
        for (sym, action), d in sorted(sym_agg.items()):
            total = d["win"] + d["loss"]
            if total < 2:
                continue
            wr = d["win"] / total
            avg_p = sum(d["pips"]) / total if d["pips"] else 0.0
            sym_lines.append(
                f"  {sym} {action}: {d['win']}W/{d['loss']}L "
                f"({wr:.0%} WR, avg {avg_p:+.1f}p)"
            )
        if sym_lines:
            parts.append("PER-SYMBOL:\n" + "\n".join(sym_lines))

        # ── Price-action conditional win rates ───────────────────────────
        # Aggregate: by symbol + (trend_intact, structure_broken) state
        pa_agg: Dict[str, Dict] = {}
        for r in price_action_rows:
            key = (r["symbol"], int(r["trend_intact"] or 0), int(r["structure_broken"] or 0))
            if key not in pa_agg:
                pa_agg[key] = {"win": 0, "loss": 0, "pips": 0.0}
            if r["outcome"] == "WIN":
                pa_agg[key]["win"] += r["n"]
                pa_agg[key]["pips"] += (r["avg_pips"] or 0.0) * r["n"]
            elif r["outcome"] == "LOSS":
                pa_agg[key]["loss"] += r["n"]

        pa_lines: List[str] = []
        for (sym, ti, sb), d in sorted(pa_agg.items()):
            total = d["win"] + d["loss"]
            if total < 2:
                continue
            wr = d["win"] / total
            avg_p = d["pips"] / total
            state = (
                "trend=OK struct=OK"  if ti == 1 and sb == 0 else
                "trend=OK struct=BRK" if ti == 1 and sb == 1 else
                "trend=BRK struct=OK" if ti == 0 and sb == 0 else
                "trend=BRK struct=BRK"
            )
            pa_lines.append(
                f"  {sym} [{state}]: {d['win']}W/{d['loss']}L "
                f"({wr:.0%} WR, avg {avg_p:+.1f}p)"
            )
        if pa_lines:
            parts.append(
                "PRICE ACTION AT REVIEW TIME (live data):\n"
                + "\n".join(pa_lines)
                + "\n  Use these to calibrate: if trend=OK struct=OK WR is high → "
                "APPROVE holds. If trend=BRK → consider OVERRIDE."
            )

        # ── Session patterns ─────────────────────────────────────────────
        sess_agg: Dict[str, Dict] = {}
        for r in sess_rows:
            s = r["session"]
            if s not in sess_agg:
                sess_agg[s] = {"win": 0, "loss": 0, "pips": []}
            if r["outcome"] == "WIN":
                sess_agg[s]["win"] += r["n"]
            elif r["outcome"] == "LOSS":
                sess_agg[s]["loss"] += r["n"]
            if r["avg_pips"] is not None:
                sess_agg[s]["pips"].append(float(r["avg_pips"]) * r["n"])

        sess_lines: List[str] = []
        for sess, d in sorted(sess_agg.items()):
            total = d["win"] + d["loss"]
            if total < 2:
                continue
            wr = d["win"] / total
            avg_p = sum(d["pips"]) / total if d["pips"] else 0.0
            sess_lines.append(
                f"  {sess}: override_close {d['win']}W/{d['loss']}L "
                f"({wr:.0%}, avg {avg_p:+.1f}p)"
            )
        if sess_lines:
            parts.append("SESSION PATTERNS (override_close):\n" + "\n".join(sess_lines))

        # ── Recent decisions ─────────────────────────────────────────────
        rec_lines: List[str] = []
        for r in recent_rows:
            ts_short = r["ts"][:16] if r["ts"] else "?"
            outcome_str = r["outcome"] or "open"
            pips_str = f"{r['profit_pips']:+.1f}p" if r["profit_pips"] is not None else ""
            changed = r["action_bot"] != r["action_taken"]
            rec_lines.append(
                f"  {ts_short} {r['symbol']:>6} "
                f"bot={r['action_bot']} -> {r['action_taken']}"
                + (" *" if changed else "")
                + f" | {outcome_str}{' '+pips_str if pips_str else ''}"
                + (f" | {r['orch_reason'][:60]}" if r["orch_reason"] else "")
            )
        if rec_lines:
            parts.append(
                "LAST 5 DECISIONS (* = orchestrator changed):\n"
                + "\n".join(rec_lines)
            )

        if not parts:
            return ""

        return (
            "=== AGENT ZERO MEMORY ===\n"
            + "\n\n".join(parts)
            + "\n=========================\n"
        )

    def stats(self) -> Dict[str, Any]:
        """Quick diagnostic summary for the dashboard / health endpoint."""
        with self._conn() as conn:
            total_d = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
            total_o = conn.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]
            latest  = conn.execute(
                "SELECT ts FROM decisions ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return {
            "decisions_stored": total_d,
            "outcomes_labelled": total_o,
            "latest_decision": latest[0] if latest else None,
            "db_path": str(self.db_path),
        }

    def get_today_summary(self) -> str:
        """Return today's closed trade results as a compact string for the LLM.

        Example output: "Today: 2W 1L 0BE"
        Returns empty string when no outcomes recorded today.
        """
        today = datetime.now(timezone.utc).date().isoformat()
        cutoff = today + "T00:00:00+00:00"
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT o.outcome, COUNT(*) AS n
                FROM outcomes o
                JOIN decisions d ON d.id = o.decision_id
                WHERE d.ts >= ?
                GROUP BY o.outcome
                """,
                (cutoff,),
            ).fetchall()
        if not rows:
            return ""
        counts = {r["outcome"]: r["n"] for r in rows}
        w  = counts.get("WIN",  0)
        lo = counts.get("LOSS", 0)
        be = counts.get("BE",   0)
        return f"Today: {w}W {lo}L {be}BE"


# ── Module-level singleton ────────────────────────────────────────────────────
_memory: Optional[AgentMemory] = None
_memory_lock = threading.Lock()


def get_memory(db_path: Optional[Path] = None) -> AgentMemory:
    """Return (or lazily create) the process-level AgentMemory singleton."""
    global _memory
    with _memory_lock:
        if _memory is None:
            _memory = AgentMemory(db_path=db_path)
    return _memory
