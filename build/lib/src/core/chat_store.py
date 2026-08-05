"""Persistent chat session + message store.

Sessions are user-facing conversations on the Chat page. Each session
groups an ordered list of user / assistant messages and survives app
restarts. The pair of tables (`_chat_sessions`, `_chat_messages`) lives
in the same SQLite database as `_query_log` and is created on first use.

sensitivity_tier: 3 (stores raw user prompts and assistant answers)
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from src.core.sqlite.engine import DatabaseEngine

logger = logging.getLogger(__name__)

DEFAULT_TITLE = "New chat"
_TITLE_MAX_LEN = 60
_PREVIEW_MAX_LEN = 80


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


class ChatStore:
    """SQLite-backed store for chat sessions and messages.

    sensitivity_tier: 3
    """

    def __init__(self, db_engine: DatabaseEngine) -> None:
        self._db = db_engine
        self._ensure_tables()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _ensure_tables(self) -> None:
        """Create the chat tables if they don't exist.

        sensitivity_tier: N/A
        """
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS _chat_sessions (
                id            VARCHAR PRIMARY KEY,
                title         VARCHAR NOT NULL DEFAULT 'New chat',
                created_at    TEXT NOT NULL,
                updated_at    TEXT NOT NULL,
                message_count INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS _chat_messages (
                id           VARCHAR PRIMARY KEY,
                session_id   VARCHAR NOT NULL,
                role         VARCHAR NOT NULL,
                content      TEXT NOT NULL,
                timestamp    TEXT NOT NULL,
                parts_json   TEXT,
                sources_json TEXT,
                latency_ms   DOUBLE,
                model        VARCHAR,
                thinking     TEXT,
                query_id     VARCHAR,
                FOREIGN KEY (session_id) REFERENCES _chat_sessions(id)
            )
            """
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_messages_session "
            "ON _chat_messages(session_id, timestamp)"
        )

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    def create_session(self, title: str | None = None) -> str:
        """Insert a new session row and return its uuid.

        sensitivity_tier: 2 (stores the title only)
        """
        session_id = str(uuid.uuid4())
        now = _now_iso()
        self._db.execute(
            "INSERT INTO _chat_sessions "
            "(id, title, created_at, updated_at, message_count) "
            "VALUES (?, ?, ?, ?, 0)",
            [session_id, title or DEFAULT_TITLE, now, now],
        )
        return session_id

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return session summaries ordered by most-recently-updated first.

        sensitivity_tier: 3 (preview contains user text)
        """
        rows = self._db.query(
            """
            SELECT
                s.id,
                s.title,
                s.created_at,
                s.updated_at,
                s.message_count,
                (
                    SELECT m.content
                    FROM _chat_messages m
                    WHERE m.session_id = s.id AND m.role = 'user'
                    ORDER BY m.timestamp DESC
                    LIMIT 1
                ) AS preview
            FROM _chat_sessions s
            ORDER BY s.updated_at DESC
            LIMIT ?
            """,
            [limit],
        )
        for row in rows:
            preview = row.get("preview")
            row["preview"] = (
                _truncate(preview, _PREVIEW_MAX_LEN) if preview else None
            )
        return rows

    def load_session(self, session_id: str) -> list[dict[str, Any]]:
        """Return all messages for *session_id* in chronological order.

        Each row matches the Rust ``ChatMessage`` DTO shape; the JSON
        side-tables (`parts`, `sources`) are deserialized back into
        lists. Returns an empty list if the session does not exist.

        sensitivity_tier: 3
        """
        rows = self._db.query(
            "SELECT role, content, timestamp, parts_json, sources_json, "
            "latency_ms, model, thinking "
            "FROM _chat_messages WHERE session_id = ? "
            "ORDER BY timestamp ASC",
            [session_id],
        )
        out: list[dict[str, Any]] = []
        for row in rows:
            out.append(
                {
                    "role": row["role"],
                    "content": row["content"],
                    "timestamp": row["timestamp"],
                    "parts": _json_loads_list(row.get("parts_json")),
                    "sources": _json_loads_list(row.get("sources_json")),
                    "latency_ms": row.get("latency_ms"),
                    "model": row.get("model"),
                    "thinking": row.get("thinking"),
                }
            )
        return out

    def delete_session(self, session_id: str) -> None:
        """Delete a session and all of its messages.

        sensitivity_tier: 1 (deletion metadata only)
        """
        self._db.execute(
            "DELETE FROM _chat_messages WHERE session_id = ?",
            [session_id],
        )
        self._db.execute(
            "DELETE FROM _chat_sessions WHERE id = ?",
            [session_id],
        )

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        parts: list[Any] | None = None,
        sources: list[Any] | None = None,
        latency_ms: float | None = None,
        model: str | None = None,
        thinking: str | None = None,
        query_id: str | None = None,
    ) -> str:
        """Append a message to *session_id*; returns the new message id.

        Also bumps the session's ``updated_at`` and ``message_count``,
        and replaces the default title with the first user message.

        sensitivity_tier: 3
        """
        if role not in {"user", "assistant"}:
            raise ValueError(f"invalid role: {role!r}")

        message_id = str(uuid.uuid4())
        now = _now_iso()
        # Single transaction: the message INSERT, the title UPDATE and the
        # message_count bump must land together or not at all — a crash
        # between auto-committed statements would desync the session row
        # from the actual messages.
        with self._db.transaction():
            self._db.execute(
                "INSERT INTO _chat_messages "
                "(id, session_id, role, content, timestamp, parts_json, "
                "sources_json, latency_ms, model, thinking, query_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    message_id,
                    session_id,
                    role,
                    content,
                    now,
                    json.dumps(parts) if parts else None,
                    json.dumps(sources) if sources else None,
                    latency_ms,
                    model,
                    thinking,
                    query_id,
                ],
            )

            if role == "user":
                self._maybe_set_title(session_id, content)

            self._db.execute(
                "UPDATE _chat_sessions "
                "SET updated_at = ?, message_count = message_count + 1 "
                "WHERE id = ?",
                [now, session_id],
            )
        return message_id

    def _maybe_set_title(self, session_id: str, first_user_text: str) -> None:
        """Replace the default 'New chat' title with the first user message.

        sensitivity_tier: 2
        """
        rows = self._db.query(
            "SELECT title FROM _chat_sessions WHERE id = ?",
            [session_id],
        )
        if not rows:
            return
        if rows[0].get("title") != DEFAULT_TITLE:
            return
        self._db.execute(
            "UPDATE _chat_sessions SET title = ? WHERE id = ?",
            [_truncate(first_user_text, _TITLE_MAX_LEN), session_id],
        )


def _json_loads_list(raw: Any) -> list[Any]:
    """Decode a JSON-list column, returning ``[]`` for NULL or malformed
    rows. The Rust DTO declares ``parts: Vec<serde_json::Value>`` (not
    optional), so emitting ``null`` here breaks ``serde_json::from_str``
    on the Tauri side — see fix/chat-session-load-null-parts. Sources
    is ``Option<Vec<..>>`` on the Rust side, so an empty list is also
    fine there.
    """
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return []
    if isinstance(value, list):
        return value
    return []
