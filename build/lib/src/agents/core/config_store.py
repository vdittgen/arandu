"""SQLite-backed agent config overrides.

Default agent configuration lives in code (manifest + system-prompt
constants). When a user edits a sub-agent in the Agents page, the patch
is persisted here. The store is read-through: ``resolve(agent_id)`` returns
the merged ``AgentConfig`` (default ⊕ override).

Editability is field-level:

- ``editable=True`` agents accept any patch (system prompt, tools,
  skills, model route, model override).
- ``editable=False`` agents (brain, firewall.injection, firewall.egress)
  accept ONLY ``model_override`` patches — the prompt, tools and routing
  contract for these agents must be changed via PR.

sensitivity_tier: 1 (prompts and tool choices, no user data)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.sqlite.engine import connect_with_pragmas

logger = logging.getLogger(__name__)

# When set, :func:`current_model_override` returns this proposed value
# for the matching agent instead of reading the saved row. Used to
# evaluate a candidate model BEFORE persisting the change so a failing
# eval can reject the override before any user-facing run uses it.
# Keyed by agent_id so concurrent proposals for different agents don't
# clobber each other. A ``None`` value explicitly means "no proposal".
_PROPOSED_OVERRIDE: ContextVar[dict[str, str | None]] = ContextVar(
    "arandu.proposed_model_override", default={},
)

DEFAULT_DB_PATH = Path.home() / ".arandu" / "data" / "arandu.sqlite3"


@dataclass(frozen=True)
class AgentConfig:
    """Resolved configuration for one agent.

    sensitivity_tier: 1
    """

    agent_id: str
    system_prompt: str
    model_route: str  # "remote" | "local" | "inherit"
    model_override: str | None
    enabled_tools: tuple[str, ...]
    enabled_skills: tuple[str, ...]
    editable: bool
    version: int = 1


@dataclass
class _Override:
    """In-memory representation of a row in ``agent_configs``.

    sensitivity_tier: 1
    """

    agent_id: str
    system_prompt: str | None = None
    model_route: str | None = None
    model_override: str | None = None
    enabled_tools: tuple[str, ...] | None = None
    enabled_skills: tuple[str, ...] | None = None
    updated_at: str = ""
    version: int = 1


class AgentConfigStoreError(Exception):
    """Raised on illegal config operations.

    sensitivity_tier: 1
    """


class AgentConfigStore:
    """Read/merge/persist editable agent overrides.

    The store accepts any DB-API connection that supports
    ``execute(sql, params)`` and returns row tuples. We use the project's
    ``DatabaseEngine`` in production; tests inject an ``sqlite3``
    connection directly.

    sensitivity_tier: 1
    """

    def __init__(self, conn: Any) -> None:
        self._conn = conn
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    DDL = """
    CREATE TABLE IF NOT EXISTS agent_configs (
        agent_id        TEXT PRIMARY KEY,
        system_prompt   TEXT,
        model_route     TEXT,
        model_override  TEXT,
        enabled_tools   TEXT,
        enabled_skills  TEXT,
        updated_at      TEXT NOT NULL,
        version         INTEGER NOT NULL DEFAULT 1
    )
    """

    def initialize(self) -> None:
        """Create the underlying table if it does not exist.

        sensitivity_tier: 1
        """
        with self._lock:
            self._conn.execute(self.DDL)

    # ------------------------------------------------------------------
    # Resolve
    # ------------------------------------------------------------------

    def resolve(
        self,
        agent_id: str,
        *,
        default: AgentConfig,
    ) -> AgentConfig:
        """Return ``default`` merged with any persisted override.

        Non-null fields in the override win. ``default.editable`` is
        preserved unconditionally — users can't promote a locked agent.

        sensitivity_tier: 1
        """
        ov = self._read(agent_id)
        if ov is None:
            return default
        return AgentConfig(
            agent_id=default.agent_id,
            system_prompt=ov.system_prompt or default.system_prompt,
            model_route=ov.model_route or default.model_route,
            model_override=(
                ov.model_override
                if ov.model_override is not None
                else default.model_override
            ),
            enabled_tools=(
                ov.enabled_tools
                if ov.enabled_tools is not None
                else default.enabled_tools
            ),
            enabled_skills=(
                ov.enabled_skills
                if ov.enabled_skills is not None
                else default.enabled_skills
            ),
            editable=default.editable,
            version=max(default.version, ov.version),
        )

    # ------------------------------------------------------------------
    # Update / reset
    # ------------------------------------------------------------------

    # Subset of fields that locked agents (brain, firewalls) accept.
    # The user is allowed to pick the LLM (model_override) and the
    # endpoint it runs on (model_route) because route and override
    # are a single coupled choice — a `local`-only id like
    # `llama3.1:8b` doesn't work on the `remote` endpoint, so locking
    # one without the other produces unusable states. The Model Picker
    # validates the (route, model_id) pair against the live catalog
    # before persisting. Prompt / tools stay under PR control.
    _LOCKED_AGENT_ALLOWED_KEYS: frozenset[str] = frozenset({
        "model_override",
        "model_route",
    })

    def update(
        self,
        agent_id: str,
        *,
        default: AgentConfig,
        patch: dict[str, Any],
    ) -> AgentConfig:
        """Apply ``patch`` and return the new merged config.

        For ``editable=True`` agents, any of the standard config fields
        may be patched. For ``editable=False`` agents (brain, firewalls)
        only ``model_override`` and ``model_route`` are accepted — the
        user can pick which LLM serves the agent and which endpoint it
        runs on, but the prompt/tools contract stays under PR control.

        Raises ``AgentConfigStoreError`` if a forbidden field is patched.

        sensitivity_tier: 1
        """
        allowed_keys = {
            "system_prompt",
            "model_route",
            "model_override",
            "enabled_tools",
            "enabled_skills",
        }
        bad = set(patch).difference(allowed_keys)
        if bad:
            msg = f"Unknown config fields: {sorted(bad)}"
            raise AgentConfigStoreError(msg)
        if not default.editable:
            forbidden = set(patch).difference(self._LOCKED_AGENT_ALLOWED_KEYS)
            if forbidden:
                msg = (
                    f"Agent {agent_id!r} is locked; only "
                    f"{sorted(self._LOCKED_AGENT_ALLOWED_KEYS)} may be "
                    f"patched (got {sorted(forbidden)})"
                )
                raise AgentConfigStoreError(msg)

        current = self._read(agent_id) or _Override(agent_id=agent_id)
        for key, val in patch.items():
            if key in {"enabled_tools", "enabled_skills"} and val is not None:
                val = tuple(val)
            setattr(current, key, val)
        current.updated_at = datetime.now(tz=timezone.utc).isoformat()
        current.version += 1
        self._write(current)
        return self.resolve(agent_id, default=default)

    def reset(self, agent_id: str, *, default: AgentConfig) -> AgentConfig:
        """Drop the override row and return the default config.

        sensitivity_tier: 1
        """
        if not default.editable:
            msg = f"Agent {agent_id!r} is not editable"
            raise AgentConfigStoreError(msg)
        with self._lock:
            self._conn.execute(
                "DELETE FROM agent_configs WHERE agent_id = ?",
                (agent_id,),
            )
        return default

    # ------------------------------------------------------------------
    # Private I/O
    # ------------------------------------------------------------------

    def _read(self, agent_id: str) -> _Override | None:
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT system_prompt, model_route, model_override,
                       enabled_tools, enabled_skills, updated_at, version
                FROM agent_configs WHERE agent_id = ?
                """,
                (agent_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return _Override(
            agent_id=agent_id,
            system_prompt=row[0],
            model_route=row[1],
            model_override=row[2],
            enabled_tools=tuple(json.loads(row[3])) if row[3] else None,
            enabled_skills=tuple(json.loads(row[4])) if row[4] else None,
            updated_at=row[5] or "",
            version=int(row[6] or 1),
        )

    def _write(self, ov: _Override) -> None:
        params = (
            ov.agent_id,
            ov.system_prompt,
            ov.model_route,
            ov.model_override,
            json.dumps(list(ov.enabled_tools)) if ov.enabled_tools else None,
            json.dumps(list(ov.enabled_skills)) if ov.enabled_skills else None,
            ov.updated_at,
            ov.version,
        )
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO agent_configs (
                    agent_id, system_prompt, model_route, model_override,
                    enabled_tools, enabled_skills, updated_at, version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(agent_id) DO UPDATE SET
                    system_prompt = excluded.system_prompt,
                    model_route = excluded.model_route,
                    model_override = excluded.model_override,
                    enabled_tools = excluded.enabled_tools,
                    enabled_skills = excluded.enabled_skills,
                    updated_at = excluded.updated_at,
                    version = excluded.version
                """,
                params,
            )


@contextmanager
def proposed_model_override(agent_id: str, model: str) -> Iterator[None]:
    """Bind a candidate ``model`` for ``agent_id`` to the current context.

    While the scope is open, :func:`current_model_override` returns
    ``model`` for that agent — so any agent constructed inside the
    block runs with the candidate override instead of the saved row.
    The block is the canonical way to evaluate a proposed model
    change before persisting it.

    sensitivity_tier: 1
    """
    current = _PROPOSED_OVERRIDE.get()
    updated = {**current, agent_id: model}
    token = _PROPOSED_OVERRIDE.set(updated)
    try:
        yield
    finally:
        _PROPOSED_OVERRIDE.reset(token)


def current_model_override(agent_id: str) -> str | None:
    """Resolve the ``model_override`` to use for ``agent_id`` right now.

    A proposed-override bound by :func:`proposed_model_override` takes
    precedence so eval runs can target a candidate model. Otherwise the
    persisted row from SQLite is used.

    Best-effort: returns ``None`` if the DB doesn't exist, the table
    isn't yet created, or the row has no override. Never raises — this
    is called on every agent construction, so it must degrade silently
    when the store is unavailable (e.g. in unit tests with no on-disk
    DB).

    sensitivity_tier: 1
    """
    proposed = _PROPOSED_OVERRIDE.get().get(agent_id)
    if proposed is not None:
        return proposed
    if not DEFAULT_DB_PATH.exists():
        return None
    conn = None
    try:
        conn = connect_with_pragmas(DEFAULT_DB_PATH)
        cur = conn.execute(
            "SELECT model_override FROM agent_configs WHERE agent_id = ?",
            (agent_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        value = row[0]
        return str(value) if value else None
    except sqlite3.Error:
        return None
    finally:
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error:
                pass


__all__ = [
    "AgentConfig",
    "AgentConfigStore",
    "AgentConfigStoreError",
    "DEFAULT_DB_PATH",
    "current_model_override",
    "proposed_model_override",
]
