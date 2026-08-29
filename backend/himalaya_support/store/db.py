from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SupportStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    locale TEXT,
                    channel TEXT,
                    created_at TEXT,
                    updated_at TEXT
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT,
                    role TEXT,
                    content TEXT,
                    meta TEXT,
                    created_at TEXT
                );
                CREATE TABLE IF NOT EXISTS tickets (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT,
                    user_id TEXT,
                    subject TEXT,
                    description TEXT,
                    category TEXT,
                    priority TEXT,
                    status TEXT,
                    created_at TEXT,
                    updated_at TEXT
                );
                """
            )
            self._ensure_column(conn, "conversations", "channel", "TEXT")

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, decl: str) -> None:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        names = {row["name"] for row in rows}
        if column not in names:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

    def get_or_create_conversation(
        self,
        conversation_id: str | None,
        user_id: str | None,
        locale: str,
        channel: str = "chat",
    ) -> str:
        kind = (channel or "chat").strip().lower() or "chat"
        if conversation_id:
            with self._connect() as conn:
                row = conn.execute("SELECT id FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
                if row:
                    return conversation_id
        new_id = conversation_id or uuid.uuid4().hex
        stamp = _now()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversations (id, user_id, locale, channel, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (new_id, user_id, locale, kind, stamp, stamp),
            )
        return new_id

    def list_conversations(self, channel: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        sql = """
            SELECT c.id, c.user_id, c.locale, COALESCE(c.channel, 'chat') AS channel,
                   c.created_at, c.updated_at,
                   (SELECT content FROM messages WHERE conversation_id = c.id ORDER BY created_at DESC LIMIT 1) AS preview,
                   (SELECT COUNT(*) FROM messages WHERE conversation_id = c.id) AS message_count
            FROM conversations c
        """
        params: list[Any] = []
        if channel:
            sql += " WHERE COALESCE(c.channel, 'chat') = ?"
            params.append(channel)
        sql += " ORDER BY c.updated_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def list_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, role, content, meta, created_at FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
                (conversation_id,),
            ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            try:
                item["meta"] = json.loads(item.get("meta") or "{}")
            except json.JSONDecodeError:
                item["meta"] = {}
            out.append(item)
        return out

    def add_message(self, conversation_id: str, role: str, content: str, meta: dict[str, Any] | None = None) -> str:
        message_id = uuid.uuid4().hex
        stamp = _now()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages (id, conversation_id, role, content, meta, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (message_id, conversation_id, role, content, json.dumps(meta or {}, ensure_ascii=False), stamp),
            )
            conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (stamp, conversation_id))
        return message_id

    def recent_messages(self, conversation_id: str, limit: int = 12) -> list[dict[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY created_at DESC LIMIT ?",
                (conversation_id, limit),
            ).fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]

    def create_ticket(self, payload: dict[str, Any]) -> dict[str, Any]:
        ticket_id = "TCK-" + uuid.uuid4().hex[:8].upper()
        stamp = _now()
        record = {
            "id": ticket_id,
            "conversation_id": payload.get("conversation_id"),
            "user_id": payload.get("user_id"),
            "subject": payload.get("subject") or "Support request",
            "description": payload.get("description") or "",
            "category": payload.get("category") or "other",
            "priority": payload.get("priority") or "normal",
            "status": "open",
            "created_at": stamp,
            "updated_at": stamp,
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tickets
                (id, conversation_id, user_id, subject, description, category, priority, status, created_at, updated_at)
                VALUES (:id, :conversation_id, :user_id, :subject, :description, :category, :priority, :status, :created_at, :updated_at)
                """,
                record,
            )
        return record

    def get_ticket(self, ticket_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
        return dict(row) if row else None

    def list_tickets(self, user_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            if user_id:
                rows = conn.execute(
                    "SELECT * FROM tickets WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                    (user_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM tickets ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(row) for row in rows]

    def update_ticket(self, ticket_id: str, **fields: Any) -> dict[str, Any] | None:
        current = self.get_ticket(ticket_id)
        if not current:
            return None
        allowed = {"subject", "description", "category", "priority", "status"}
        updates = {key: value for key, value in fields.items() if key in allowed and value is not None}
        if not updates:
            return current
        updates["updated_at"] = _now()
        assignments = ", ".join(f"{key} = :{key}" for key in updates)
        updates["id"] = ticket_id
        with self._connect() as conn:
            conn.execute(f"UPDATE tickets SET {assignments} WHERE id = :id", updates)
        return self.get_ticket(ticket_id)
