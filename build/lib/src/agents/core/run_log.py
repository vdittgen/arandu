"""Per-agent input/output run log.

Records every ``SBAgent.run`` / ``SBDeepAgent.run`` invocation so the
Agents page can surface the most recent inputs and outputs for one
agent — both for diagnosing odd behaviour and for picking eval cases
from real traffic. Keeps the newest ``MAX_PER_AGENT`` entries per
``agent_id`` and discards older rows on insert.

The log is *not* an audit chain — see
``src/agents/core/audit.py`` for the tamper-evident decision trail.
This module stores raw prompts and serialized outputs so the user can
actually read them. Local persistence only; nothing here ever leaves
the device. The DB file should be treated as sensitive.

sensitivity_tier: varies (records raw prompts and outputs across tiers)
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.sqlite.engine import connect_with_pragmas

logger = logging.getLogger(__name__)

MAX_PER_AGENT = 1000

DEFAULT_DB_PATH = (
    Path.home() / ".arandu" / "data" / "arandu.sqlite3"
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_run_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    input TEXT,
    output TEXT,
    duration_ms REAL,
    route TEXT,
    status TEXT NOT NULL,
    error TEXT
)
"""

_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_agent_run_log_agent_id "
    "ON agent_run_log(agent_id, id DESC)"
)


@dataclass(frozen=True)
class AgentRunLogEntry:
    """One row of the run log; serialized to JSON for the UI.

    ``output`` is a JSON string when the agent produced a structured
    output, otherwise ``None``. ``input`` is the raw prompt string
    ``SBAgent.build_prompt`` returned.

    sensitivity_tier: varies
    """

    id: int
    agent_id: str
    ts: str
    input: str | None
    output: str | None
    duration_ms: float | None
    route: str | None
    status: str
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "ts": self.ts,
            "input": self.input,
            "output": self.output,
            "duration_ms": self.duration_ms,
            "route": self.route,
            "status": self.status,
            "error": self.error,
        }


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _stringify_output(output: Any) -> str | None:
    """Best-effort projection of an agent output to a JSON string.

    Tries pydantic's ``model_dump_json`` first (the common case for
    SBAgent outputs), then ``model_dump`` + ``json.dumps``, then
    plain ``json.dumps``, falling back to ``str()``.

    sensitivity_tier: varies
    """
    if output is None:
        return None
    if hasattr(output, "model_dump_json"):
        try:
            return output.model_dump_json()
        except Exception:  # noqa: BLE001
            logger.debug("model_dump_json failed", exc_info=True)
    if hasattr(output, "model_dump"):
        try:
            return json.dumps(output.model_dump(), default=str)
        except Exception:  # noqa: BLE001
            logger.debug("model_dump failed", exc_info=True)
    try:
        return json.dumps(output, default=str)
    except Exception:  # noqa: BLE001
        return str(output)


class AgentRunLog:
    """Thread-safe SQLite-backed run log.

    One instance per process holds the connection open in autocommit
    mode. The lock serialises writes (insert + trim run as two
    statements); reads use the same lock for a brief moment. Cheap
    enough that the hot path of every agent call can afford it.

    Subprocess-style callers (e.g. the CLI dispatched by Tauri) open
    their own instance via :func:`default_run_log`; OS-level append
    semantics keep individual writes safe even across processes.

    sensitivity_tier: varies
    """

    def __init__(
        self,
        path: Path | None = None,
        *,
        max_per_agent: int = MAX_PER_AGENT,
    ) -> None:
        self._path = path or DEFAULT_DB_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._max = max_per_agent
        self._conn = connect_with_pragmas(
            self._path, check_same_thread=False,
        )
        self._lock = threading.Lock()
        self._ensure_schema()

    @property
    def path(self) -> Path:
        return self._path

    def _ensure_schema(self) -> None:
        with self._lock:
            self._conn.execute(_SCHEMA)
            self._conn.execute(_INDEX)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def record(
        self,
        *,
        agent_id: str,
        input: str | None,
        output: Any | None,
        duration_ms: float | None,
        route: str | None,
        status: str,
        error: str | None = None,
    ) -> None:
        """Append one entry; trim to the most-recent ``max_per_agent``.

        Every parameter except ``agent_id`` and ``status`` is optional
        — a failed run still gets a row so the user can see *that* it
        failed even when there's no output to record.

        sensitivity_tier: varies
        """
        if not agent_id:
            return
        output_str = _stringify_output(output)
        with self._lock:
            try:
                self._conn.execute(
                    """
                    INSERT INTO agent_run_log (
                        agent_id, ts, input, output, duration_ms,
                        route, status, error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        agent_id, _now_iso(), input, output_str,
                        duration_ms, route, status, error,
                    ),
                )
                # Trim — keep newest ``max_per_agent`` for this id.
                self._conn.execute(
                    """
                    DELETE FROM agent_run_log
                    WHERE agent_id = ? AND id NOT IN (
                        SELECT id FROM agent_run_log
                        WHERE agent_id = ?
                        ORDER BY id DESC
                        LIMIT ?
                    )
                    """,
                    (agent_id, agent_id, self._max),
                )
            except sqlite3.Error:
                logger.exception(
                    "agent_run_log insert failed for %s", agent_id,
                )

    def recent(
        self,
        agent_id: str,
        *,
        limit: int = 100,
    ) -> list[AgentRunLogEntry]:
        """Most-recent ``limit`` entries for ``agent_id``, newest first.

        ``limit`` is clamped to ``[1, MAX_PER_AGENT]``.

        sensitivity_tier: varies
        """
        clamped = max(1, min(int(limit), self._max))
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT id, agent_id, ts, input, output, duration_ms,
                       route, status, error
                FROM agent_run_log
                WHERE agent_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (agent_id, clamped),
            )
            rows = cur.fetchall()
        return [
            AgentRunLogEntry(
                id=r[0], agent_id=r[1], ts=r[2], input=r[3], output=r[4],
                duration_ms=r[5], route=r[6], status=r[7], error=r[8],
            )
            for r in rows
        ]

    def count(self, agent_id: str) -> int:
        """Return the number of rows currently stored for ``agent_id``.

        sensitivity_tier: 1
        """
        with self._lock:
            cur = self._conn.execute(
                "SELECT COUNT(*) FROM agent_run_log WHERE agent_id = ?",
                (agent_id,),
            )
            row = cur.fetchone()
        return int(row[0]) if row else 0


_default_log: AgentRunLog | None = None
_default_lock = threading.Lock()


def default_run_log() -> AgentRunLog:
    """Return the process-wide default run log.

    Honours ``ARANDU_AGENT_RUN_LOG_PATH`` so tests can point at a
    throw-away DB without touching the user's data dir.

    sensitivity_tier: varies
    """
    global _default_log
    if _default_log is None:
        with _default_lock:
            if _default_log is None:
                override = os.environ.get("ARANDU_AGENT_RUN_LOG_PATH")
                path = Path(override) if override else DEFAULT_DB_PATH
                _default_log = AgentRunLog(path=path)
    return _default_log


def reset_default_run_log_for_tests() -> None:
    """Drop the cached default log.

    Tests that swap ``ARANDU_AGENT_RUN_LOG_PATH`` mid-run should call
    this to avoid leaking a connection pointed at the old path.

    sensitivity_tier: N/A
    """
    global _default_log
    with _default_lock:
        if _default_log is not None:
            try:
                _default_log.close()
            except Exception:  # noqa: BLE001
                logger.debug("close on stale run log failed", exc_info=True)
        _default_log = None


__all__ = [
    "DEFAULT_DB_PATH",
    "MAX_PER_AGENT",
    "AgentRunLog",
    "AgentRunLogEntry",
    "default_run_log",
    "reset_default_run_log_for_tests",
]
