"""SQLite-backed queue for scheduled Brews.

A Brew is one user-submitted topic earmarked for *delayed* delivery —
the user taps 🍵 Brew, confirms with ☕ +3h or 🌅 8am, and the row sits
here until a PTB JobQueue worker picks it up at its ``reveal_at`` time,
runs the pipeline, renders the Pillow card, and sends it back.

Design choices:
- Single file ``brews.db`` next to the bot's other state. WAL journal
  mode so the bot can write while the worker reads.
- No ORM. Raw sqlite3 with a tiny context-manager. Schema is fixed-
  shape; migrations would be ALTER TABLE in this module.
- Status state machine: pending → brewing → delivered | failed.
  Workers transition pending → brewing atomically (UPDATE...WHERE
  status='pending') so two concurrent pollers can't double-process the
  same brew.
- ``failure_reason`` captures one-line errors for ops debugging. Not
  shown to users — they see a generic "couldn't finish this brew"
  message.
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

DB_PATH = Path(__file__).parent / "brews.db"


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    """Short-lived connection with WAL + row factory. Caller is expected
    to call execute/commit on the yielded object; we close on exit."""
    conn = sqlite3.connect(DB_PATH, isolation_level=None, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    """Create schema if missing. Idempotent — safe to call on every bot
    boot. Called from build_app() at startup."""
    with _conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS brews (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id         INTEGER NOT NULL,
                topic           TEXT NOT NULL,
                mode            TEXT NOT NULL DEFAULT 'default',
                extra           TEXT,                     -- JSON: lucid_prior etc.
                reveal_at       REAL NOT NULL,            -- unix ts
                status          TEXT NOT NULL DEFAULT 'pending',
                created_at      REAL NOT NULL,
                message_id      INTEGER,                  -- delivered photo's message_id
                result_path     TEXT,                     -- path to rendered PNG
                failure_reason  TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_brews_status_reveal
                ON brews(status, reveal_at);

            CREATE INDEX IF NOT EXISTS idx_brews_chat_created
                ON brews(chat_id, created_at);
            """
        )


def schedule(
    *,
    chat_id: int,
    topic: str,
    mode: str = "default",
    reveal_at: float,
    extra: dict | None = None,
) -> int:
    """Insert a pending brew, return its row id."""
    extra_json = json.dumps(extra) if extra else None
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO brews "
            "(chat_id, topic, mode, extra, reveal_at, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, 'pending', ?)",
            (chat_id, topic, mode, extra_json, reveal_at, time.time()),
        )
        return int(cur.lastrowid)


def due_brews(limit: int = 10) -> list[dict]:
    """Brews whose reveal_at <= now and status='pending', oldest first."""
    now = time.time()
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM brews "
            "WHERE status='pending' AND reveal_at <= ? "
            "ORDER BY reveal_at LIMIT ?",
            (now, limit),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def claim(brew_id: int) -> bool:
    """Atomically transition pending → brewing. Returns True if claimed
    (so the worker should proceed), False if some other process got it
    first (so the worker should skip)."""
    with _conn() as c:
        cur = c.execute(
            "UPDATE brews SET status='brewing' "
            "WHERE id=? AND status='pending'",
            (brew_id,),
        )
        return cur.rowcount > 0


def mark_delivered(brew_id: int, message_id: int, result_path: str) -> None:
    with _conn() as c:
        c.execute(
            "UPDATE brews SET status='delivered', message_id=?, result_path=? "
            "WHERE id=?",
            (message_id, str(result_path), brew_id),
        )


def mark_failed(brew_id: int, reason: str) -> None:
    with _conn() as c:
        c.execute(
            "UPDATE brews SET status='failed', failure_reason=? WHERE id=?",
            (reason[:500], brew_id),
        )


def cancel(brew_id: int, chat_id: int) -> bool:
    """User-initiated cancel — only works on still-pending brews and only
    from the same chat that scheduled them. Returns True on success."""
    with _conn() as c:
        cur = c.execute(
            "UPDATE brews SET status='failed', failure_reason='cancelled by user' "
            "WHERE id=? AND chat_id=? AND status='pending'",
            (brew_id, chat_id),
        )
        return cur.rowcount > 0


def pending_for_chat(chat_id: int, limit: int = 20) -> list[dict]:
    """User's still-pending brews, soonest reveal first. For /brews."""
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM brews "
            "WHERE chat_id=? AND status='pending' "
            "ORDER BY reveal_at LIMIT ?",
            (chat_id, limit),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def history_for_chat(chat_id: int, limit: int = 5) -> list[dict]:
    """Most recent N brews for a chat regardless of status. For the
    Brew history / status display."""
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM brews "
            "WHERE chat_id=? "
            "ORDER BY created_at DESC LIMIT ?",
            (chat_id, limit),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    if d.get("extra"):
        try:
            d["extra"] = json.loads(d["extra"])
        except json.JSONDecodeError:
            d["extra"] = None
    return d
