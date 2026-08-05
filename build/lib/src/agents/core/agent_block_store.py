"""Per-agent block table — gates LLM access when a local eval failed.

In Arandu every prompt runs locally, but the eval gate is still
useful: when a user changes their Ollama model (or its config) and an
agent's eval suite no longer passes, that agent's gateway calls fail
closed instead of producing degraded output.

The block is per-agent so one bad eval doesn't disable the whole
product. Re-running the eval and seeing it pass clears the row — the
eval CLI deletes the block on a passing run.

sensitivity_tier: 1
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from src.core.sqlite.engine import connect_with_pragmas

logger = logging.getLogger(__name__)


DEFAULT_DB_PATH = (
    Path.home() / ".arandu" / "data" / "arandu.sqlite3"
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_blocked (
    agent_id   TEXT PRIMARY KEY,
    reason     TEXT NOT NULL,
    blocked_at TEXT NOT NULL
)
"""


@dataclass(frozen=True)
class AgentBlock:
    """One persisted block row.

    sensitivity_tier: 1
    """

    agent_id: str
    reason: str
    blocked_at: str


class AgentBlockStore:
    """SQLite-backed per-agent block table.

    Always opens in autocommit mode — every call site opens a fresh
    connection that exits immediately after returning.

    sensitivity_tier: 1
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or DEFAULT_DB_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = connect_with_pragmas(
            self._path, check_same_thread=False,
        )
        self._conn.execute(_SCHEMA)

    def block(self, agent_id: str, *, reason: str) -> None:
        """Mark ``agent_id`` as blocked at the gateway.

        sensitivity_tier: 1
        """
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO agent_blocked "
                "(agent_id, reason, blocked_at) VALUES (?, ?, ?)",
                (agent_id, reason, datetime.now(tz=UTC).isoformat()),
            )

    def unblock(self, agent_id: str) -> None:
        """Drop the block row for ``agent_id``, if present.

        sensitivity_tier: 1
        """
        with self._lock:
            self._conn.execute(
                "DELETE FROM agent_blocked WHERE agent_id = ?",
                (agent_id,),
            )

    def clear(self) -> None:
        """Wipe every block — used when the user disables local-only mode.

        sensitivity_tier: 1
        """
        with self._lock:
            self._conn.execute("DELETE FROM agent_blocked")

    def get_block(self, agent_id: str) -> str | None:
        """Return the block reason for ``agent_id`` if blocked.

        ``None`` when the agent is not blocked. Hot-path read used by
        :func:`chat_via_firewalls`.

        sensitivity_tier: 1
        """
        with self._lock:
            cur = self._conn.execute(
                "SELECT reason FROM agent_blocked WHERE agent_id = ?",
                (agent_id,),
            )
            row = cur.fetchone()
            return row[0] if row else None

    def list_blocked(self) -> list[AgentBlock]:
        """Return every blocked agent (for the Agents page banner).

        sensitivity_tier: 1
        """
        with self._lock:
            cur = self._conn.execute(
                "SELECT agent_id, reason, blocked_at FROM agent_blocked "
                "ORDER BY blocked_at DESC",
            )
            return [
                AgentBlock(
                    agent_id=row[0],
                    reason=row[1],
                    blocked_at=row[2],
                )
                for row in cur.fetchall()
            ]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


_default_store: AgentBlockStore | None = None
_default_lock = threading.Lock()


def default_agent_block_store() -> AgentBlockStore:
    """Return the process-wide :class:`AgentBlockStore`.

    sensitivity_tier: 1
    """
    global _default_store
    if _default_store is None:
        with _default_lock:
            if _default_store is None:
                _default_store = AgentBlockStore()
    return _default_store


def reset_agent_block_store_for_tests(
    *, path: Path | None = None,
) -> AgentBlockStore:
    """Re-create the process-wide store — test isolation only.

    sensitivity_tier: 1
    """
    global _default_store
    with _default_lock:
        if _default_store is not None:
            _default_store.close()
        _default_store = AgentBlockStore(path=path)
    return _default_store


__all__ = [
    "AgentBlock",
    "AgentBlockStore",
    "DEFAULT_DB_PATH",
    "default_agent_block_store",
    "reset_agent_block_store_for_tests",
]
