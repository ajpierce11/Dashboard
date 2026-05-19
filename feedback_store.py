"""
feedback_store.py
Single-file SQLite store for in-app feedback, bug reports, and
auto-captured errors.

Why SQLite, not JSON Lines:
    The Feedback Inbox tab needs filtering (by status, type, date) and
    "mark resolved" updates. SQL handles both in two-line queries, while
    JSONL would need full-file rewrites for status changes. SQLite ships
    with Python — no new dependency, no server process.

Database location:
    feedback/feedback.db — at the project root, deliberately OUTSIDE the
    Library folder so the document indexer never picks it up. The
    feedback/ directory is gitignored.

Schema is created on first call to init_db() and is idempotent — safe
to call on every app startup.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

# Resolved relative to this file so the path is stable across CML and
# local development. Project root is wherever feedback_store.py lives.
_DB_DIR  = Path(__file__).resolve().parent / "feedback"
_DB_PATH = _DB_DIR / "feedback.db"

# SQLite connections aren't thread-safe by default in Python; serialise
# writes with a single lock. Streamlit reruns can fire from multiple
# threads, so this matters even for a single user.
_LOCK = threading.Lock()


def _connect() -> sqlite3.Connection:
    _DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """
    Create the feedback table if it doesn't exist. Safe to call on
    every app startup — uses CREATE TABLE IF NOT EXISTS.
    """
    with _LOCK, _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp    TEXT    NOT NULL,
                user         TEXT    NOT NULL,
                type         TEXT    NOT NULL,
                category     TEXT,
                title        TEXT    NOT NULL,
                body         TEXT,
                context      TEXT,
                traceback    TEXT,
                status       TEXT    NOT NULL DEFAULT 'open',
                resolved_at  TEXT,
                resolved_by  TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_feedback_status_ts
            ON feedback (status, timestamp DESC)
        """)
        conn.commit()


def submit(
    *,
    user: str,
    type: str,
    title: str,
    body: str = "",
    category: str | None = None,
    context: dict | None = None,
    traceback: str | None = None,
) -> int:
    """
    Persist one feedback entry. Returns the new row id.

    `type` is one of: "feedback", "bug", "question", "error".
    `context` is an arbitrary dict (which tab, current selections, etc.)
    serialised to JSON. `traceback` is only set for auto-captured errors.

    Failures are swallowed and -1 returned — feedback logging must never
    crash the app it's reporting on.
    """
    try:
        ctx_json = json.dumps(context, default=str) if context else None
        with _LOCK, _connect() as conn:
            cur = conn.execute("""
                INSERT INTO feedback
                    (timestamp, user, type, category, title, body, context, traceback)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(timespec="seconds"),
                user or "(unknown)",
                type,
                category,
                title.strip()[:500],
                (body or "").strip(),
                ctx_json,
                traceback,
            ))
            conn.commit()
            return cur.lastrowid or -1
    except Exception:
        return -1


def list_entries(
    *,
    status: str | None = None,
    type: str | None = None,
    limit: int = 500,
) -> list[dict]:
    """
    Return entries matching the filters, newest first. status/type of
    None or 'all' means no filter on that field.
    """
    try:
        clauses: list[str] = []
        params:  list = []
        if status and status != "all":
            clauses.append("status = ?")
            params.append(status)
        if type and type != "all":
            clauses.append("type = ?")
            params.append(type)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(int(limit))
        with _LOCK, _connect() as conn:
            rows = conn.execute(f"""
                SELECT * FROM feedback
                {where}
                ORDER BY datetime(timestamp) DESC
                LIMIT ?
            """, params).fetchall()
            return [dict(r) for r in rows]
    except Exception:
        return []


def mark_resolved(entry_id: int, by: str) -> bool:
    try:
        with _LOCK, _connect() as conn:
            conn.execute("""
                UPDATE feedback
                SET status = 'resolved',
                    resolved_at = ?,
                    resolved_by = ?
                WHERE id = ?
            """, (datetime.now().isoformat(timespec="seconds"), by, entry_id))
            conn.commit()
            return True
    except Exception:
        return False


def reopen(entry_id: int) -> bool:
    try:
        with _LOCK, _connect() as conn:
            conn.execute("""
                UPDATE feedback
                SET status = 'open',
                    resolved_at = NULL,
                    resolved_by = NULL
                WHERE id = ?
            """, (entry_id,))
            conn.commit()
            return True
    except Exception:
        return False


def counts_by_status() -> dict[str, int]:
    """Quick stats for the inbox header — 'X open, Y resolved'."""
    try:
        with _LOCK, _connect() as conn:
            rows = conn.execute("""
                SELECT status, COUNT(*) AS n FROM feedback GROUP BY status
            """).fetchall()
            return {r["status"]: r["n"] for r in rows}
    except Exception:
        return {}
