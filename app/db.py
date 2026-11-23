"""SQLite storage. Plain sqlite3 with parameterised queries only."""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    message TEXT NOT NULL,
    category TEXT NOT NULL,
    category_confidence REAL NOT NULL,
    priority TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    draft_status TEXT NOT NULL DEFAULT 'pending',
    draft_text TEXT,
    draft_faq_ids TEXT NOT NULL DEFAULT '[]',
    draft_grounded INTEGER,
    draft_source TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tickets_category ON tickets(category);
"""


@contextmanager
def connect():
    Path(config.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


def _row(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["draft_faq_ids"] = json.loads(d["draft_faq_ids"])
    d["draft_grounded"] = None if d["draft_grounded"] is None else bool(d["draft_grounded"])
    return d


def create_ticket(name: str, email: str, message: str, category: str, confidence: float, priority: str) -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO tickets (name, email, message, category, category_confidence, priority, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (name, email, message, category, confidence, priority, datetime.now(timezone.utc).isoformat()),
        )
        return cur.lastrowid


def get_ticket(ticket_id: int) -> Optional[dict]:
    with connect() as conn:
        r = conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
    return _row(r) if r else None


def list_tickets(category: Optional[str] = None, status: Optional[str] = None) -> list[dict]:
    sql, args = "SELECT * FROM tickets", []
    clauses = []
    if category:
        clauses.append("category = ?")
        args.append(category)
    if status:
        clauses.append("status = ?")
        args.append(status)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY id DESC"
    with connect() as conn:
        return [_row(r) for r in conn.execute(sql, args).fetchall()]


def update_status(ticket_id: int, status: str) -> bool:
    with connect() as conn:
        cur = conn.execute("UPDATE tickets SET status = ? WHERE id = ?", (status, ticket_id))
        return cur.rowcount > 0


def save_draft(ticket_id: int, text: Optional[str], faq_ids: list[str], grounded: bool, source: str, ok: bool) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE tickets SET draft_status = ?, draft_text = ?, draft_faq_ids = ?, draft_grounded = ?, "
            "draft_source = ? WHERE id = ?",
            ("ready" if ok else "failed", text, json.dumps(faq_ids), int(grounded), source, ticket_id),
        )
