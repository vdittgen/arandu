"""SQLite-backed store for user-authored agents.

Each row in ``user_agents`` is one editable agent the user created
from the Agents page. Tools the agent may use (MCP action tools,
``recall_context`` into Brain, registered skills) are persisted as
JSON arrays so the registration step can rebuild the agent definition
on every process start.

sensitivity_tier: 1
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.sqlite.engine import connect_with_pragmas

DEFAULT_DB_PATH: Path = (
    Path.home() / ".arandu" / "data" / "arandu.sqlite3"
)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS user_agents (
    agent_id            TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    description         TEXT NOT NULL DEFAULT '',
    system_prompt       TEXT NOT NULL,
    model_route         TEXT NOT NULL DEFAULT 'inherit',
    model_override      TEXT,
    enabled_skills_json TEXT NOT NULL DEFAULT '[]',
    enabled_mcp_tools_json TEXT NOT NULL DEFAULT '[]',
    brain_access        INTEGER NOT NULL DEFAULT 1,
    max_sensitivity_tier INTEGER NOT NULL DEFAULT 2,
    schedule_cron       TEXT,
    schedule_enabled    INTEGER NOT NULL DEFAULT 0,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL,
    version             INTEGER NOT NULL DEFAULT 1,
    pre_ai_system_prompt TEXT,
    pre_ai_description   TEXT,
    pattern             TEXT NOT NULL DEFAULT 'single',
    subagents_json      TEXT NOT NULL DEFAULT '[]'
)
"""

# Columns added after the initial table creation. Existing databases
# need ALTER TABLE on startup; new databases get them from the DDL
# above. Each entry is ``(column_name, sqlite_type)`` — the migration
# loop runs ``ALTER TABLE user_agents ADD COLUMN`` for any column
# missing from ``PRAGMA table_info``.
_LATE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("pre_ai_system_prompt", "TEXT"),
    ("pre_ai_description", "TEXT"),
    ("pattern", "TEXT NOT NULL DEFAULT 'single'"),
    ("subagents_json", "TEXT NOT NULL DEFAULT '[]'"),
    # Retained for back-compat; the column is no longer read after the
    # unify-tools migration converts any non-empty value into the
    # appropriate ``connector_id:list_*`` entry in
    # ``enabled_mcp_tools_json``. See ``_migrate_message_sources``.
    ("message_sources_json", "TEXT NOT NULL DEFAULT '[]'"),
    ("delivery_tools_json", "TEXT NOT NULL DEFAULT '[]'"),
    # Per-tool static args merged into the call before the LLM digest
    # text is placed. Shape: ``{tool_id: {arg_name: value}}``. Lets a
    # multi-required-field tool like ``whatsapp:send_message`` receive
    # its routing slot (``to``) without forcing the LLM to invent it.
    ("delivery_targets_json", "TEXT NOT NULL DEFAULT '{}'"),
)


@dataclass(frozen=True)
class UserAgentRow:
    """One row of the ``user_agents`` table.

    ``pre_ai_system_prompt`` and ``pre_ai_description`` hold the
    just-prior values captured the last time the prompt engineer
    applied a rewrite. They are ``None`` when no rewrite is pending
    revert. The UI uses their presence to surface a "Revert
    prompt-engineer edits" button.

    sensitivity_tier: 1
    """

    agent_id: str
    name: str
    description: str
    system_prompt: str
    model_route: str
    model_override: str | None
    enabled_skills: tuple[str, ...]
    enabled_mcp_tools: tuple[str, ...]
    brain_access: bool
    max_sensitivity_tier: int
    schedule_cron: str | None
    schedule_enabled: bool
    created_at: str
    updated_at: str
    version: int = 1
    pre_ai_system_prompt: str | None = None
    pre_ai_description: str | None = None
    # ``"single"`` (default) or ``"orchestrator"``. Orchestrator rows
    # delegate to the agents named in :attr:`subagents` via the
    # ``SBOrchestrator`` base class.
    pattern: str = "single"
    subagents: tuple[str, ...] = ()
    # Action-typed MCP tools the runner invokes as a post-batch
    # delivery hook. Independent of ``enabled_mcp_tools``: a tool may
    # be a delivery target without also being LLM-callable, and vice
    # versa. The LLM never sees delivery tools during per-item runs.
    delivery_tools: tuple[str, ...] = ()
    # Per-tool static args merged in before the LLM digest text fills
    # the first remaining required string field. Shape:
    # ``{tool_id: {arg_name: value}}``. Empty {} means "let the
    # coercion heuristic pick the field for summary_text".
    delivery_targets: Mapping[str, Mapping[str, Any]] = field(
        default_factory=dict,
    )

    def to_dict(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "model_route": self.model_route,
            "model_override": self.model_override,
            "enabled_skills": list(self.enabled_skills),
            "enabled_mcp_tools": list(self.enabled_mcp_tools),
            "brain_access": self.brain_access,
            "max_sensitivity_tier": self.max_sensitivity_tier,
            "schedule_cron": self.schedule_cron,
            "schedule_enabled": self.schedule_enabled,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version,
            "pre_ai_system_prompt": self.pre_ai_system_prompt,
            "pre_ai_description": self.pre_ai_description,
            "pattern": self.pattern,
            "subagents": list(self.subagents),
            "delivery_tools": list(self.delivery_tools),
            "delivery_targets": {
                k: dict(v) for k, v in self.delivery_targets.items()
            },
        }


@dataclass
class UserAgentUpsert:
    """Mutable input for create / update calls.

    sensitivity_tier: 1
    """

    name: str
    description: str
    system_prompt: str
    model_route: str = "inherit"
    model_override: str | None = None
    enabled_skills: tuple[str, ...] = ()
    enabled_mcp_tools: tuple[str, ...] = ()
    brain_access: bool = True
    max_sensitivity_tier: int = 2
    schedule_cron: str | None = None
    schedule_enabled: bool = False
    pattern: str = "single"
    subagents: tuple[str, ...] = ()
    delivery_tools: tuple[str, ...] = ()
    delivery_targets: Mapping[str, Mapping[str, Any]] = field(
        default_factory=dict,
    )


def _targets_to_jsonable(
    targets: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {k: dict(v) for k, v in targets.items()}


def _parse_targets(raw: str | None) -> dict[str, dict[str, Any]]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for k, v in data.items():
        if isinstance(k, str) and isinstance(v, dict):
            out[k] = {ak: av for ak, av in v.items() if isinstance(ak, str)}
    return out


_SLUG_RE = re.compile(r"[^a-z0-9_]+")


def make_agent_id(name: str) -> str:
    """Derive a stable ``user.<slug>`` id from a free-text name.

    sensitivity_tier: 1
    """
    slug = _SLUG_RE.sub("_", name.strip().lower()).strip("_") or "agent"
    return f"user.{slug}"


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


class UserAgentStore:
    """Read / write helper for ``user_agents``.

    Autocommit mode mirrors :class:`EvalRunStore` — the CLI handlers
    are short-lived subprocess invocations and must persist their
    write before exiting.

    sensitivity_tier: 1
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or DEFAULT_DB_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = connect_with_pragmas(self._path)
        self._conn.execute(_SCHEMA)
        self._run_migrations()

    def _run_migrations(self) -> None:
        """Add columns that were introduced after the initial DDL.

        Idempotent — checks ``PRAGMA table_info`` and only runs an
        ``ALTER TABLE`` when a column is missing. SQLite allows
        adding NULL-able columns in place.

        sensitivity_tier: 1
        """
        cur = self._conn.execute("PRAGMA table_info(user_agents)")
        existing = {row[1] for row in cur.fetchall()}
        for column, sql_type in _LATE_COLUMNS:
            if column not in existing:
                self._conn.execute(
                    f"ALTER TABLE user_agents ADD COLUMN {column} {sql_type}",
                )
        self._migrate_message_sources_to_data_tools()

    def _migrate_message_sources_to_data_tools(self) -> None:
        """Convert legacy ``message_sources_json`` rows into data-tool ids.

        Old rows stored a list of bare connector ids
        (``["apple-mail"]``) that the runner translated to a hardcoded
        source table. We now treat the connector's data tool
        (``apple-mail:list_emails``) as the source binding so the same
        picker that selects callable / delivery tools also selects
        sources. For each legacy row, look up the connector's data
        tool in the catalog, add ``connector_id:tool_name`` to
        ``enabled_mcp_tools_json``, and clear ``message_sources_json``.

        Idempotent: a row whose ``message_sources_json`` is already
        ``'[]'`` is skipped. Catalog lookup failures (e.g. a connector
        that no longer exists in the catalog) leave the legacy row
        alone but still clear it so we don't loop on the next open.

        sensitivity_tier: 1
        """
        cur = self._conn.execute(
            "SELECT agent_id, enabled_mcp_tools_json, message_sources_json "
            "FROM user_agents WHERE message_sources_json IS NOT NULL "
            "AND message_sources_json != '[]' AND message_sources_json != ''",
        )
        legacy_rows = cur.fetchall()
        if not legacy_rows:
            return
        try:
            from src.extensions.connectors.catalog import ConnectorCatalog
        except Exception:  # noqa: BLE001 -- catalog missing in some test setups
            return
        catalog = ConnectorCatalog()
        for agent_id, tools_json, sources_json in legacy_rows:
            try:
                sources = json.loads(sources_json or "[]")
                tools = set(json.loads(tools_json or "[]"))
            except (TypeError, json.JSONDecodeError):
                sources, tools = [], set()
            for connector_id in sources:
                template = catalog.get(connector_id)
                if template is None:
                    continue
                data_tool = next(
                    (
                        t for t in template.tools
                        if t.tool_type == "data" and t.target_table
                    ),
                    None,
                )
                if data_tool is not None:
                    tools.add(f"{connector_id}:{data_tool.tool_name}")
            self._conn.execute(
                "UPDATE user_agents SET enabled_mcp_tools_json = ?, "
                "message_sources_json = '[]' WHERE agent_id = ?",
                (json.dumps(sorted(tools)), agent_id),
            )

    def close(self) -> None:
        self._conn.close()

    def list_all(self) -> list[UserAgentRow]:
        cur = self._conn.execute(
            """
            SELECT agent_id, name, description, system_prompt,
                   model_route, model_override, enabled_skills_json,
                   enabled_mcp_tools_json, brain_access,
                   max_sensitivity_tier, schedule_cron,
                   schedule_enabled, created_at, updated_at, version,
                   pre_ai_system_prompt, pre_ai_description,
                   pattern, subagents_json, delivery_tools_json,
                   delivery_targets_json
            FROM user_agents
            ORDER BY created_at ASC
            """,
        )
        return [_row_to_agent(r) for r in cur.fetchall()]

    def get(self, agent_id: str) -> UserAgentRow | None:
        cur = self._conn.execute(
            """
            SELECT agent_id, name, description, system_prompt,
                   model_route, model_override, enabled_skills_json,
                   enabled_mcp_tools_json, brain_access,
                   max_sensitivity_tier, schedule_cron,
                   schedule_enabled, created_at, updated_at, version,
                   pre_ai_system_prompt, pre_ai_description,
                   pattern, subagents_json, delivery_tools_json,
                   delivery_targets_json
            FROM user_agents
            WHERE agent_id = ?
            """,
            (agent_id,),
        )
        row = cur.fetchone()
        return _row_to_agent(row) if row is not None else None

    def insert(self, upsert: UserAgentUpsert) -> UserAgentRow:
        agent_id = make_agent_id(upsert.name)
        # Collision-safe — append a numeric suffix if the slug exists.
        base = agent_id
        n = 2
        while self.get(agent_id) is not None:
            agent_id = f"{base}_{n}"
            n += 1
        now = _now_iso()
        self._conn.execute(
            """
            INSERT INTO user_agents (
                agent_id, name, description, system_prompt,
                model_route, model_override, enabled_skills_json,
                enabled_mcp_tools_json, brain_access,
                max_sensitivity_tier, schedule_cron,
                schedule_enabled, created_at, updated_at, version,
                pattern, subagents_json, delivery_tools_json,
                delivery_targets_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)
            """,
            (
                agent_id,
                upsert.name,
                upsert.description,
                upsert.system_prompt,
                upsert.model_route,
                upsert.model_override,
                json.dumps(list(upsert.enabled_skills)),
                json.dumps(list(upsert.enabled_mcp_tools)),
                int(bool(upsert.brain_access)),
                int(upsert.max_sensitivity_tier),
                upsert.schedule_cron,
                int(bool(upsert.schedule_enabled)),
                now,
                now,
                upsert.pattern,
                json.dumps(list(upsert.subagents)),
                json.dumps(list(upsert.delivery_tools)),
                json.dumps(_targets_to_jsonable(upsert.delivery_targets)),
            ),
        )
        row = self.get(agent_id)
        assert row is not None  # noqa: S101
        return row

    def update(self, agent_id: str, upsert: UserAgentUpsert) -> UserAgentRow:
        existing = self.get(agent_id)
        if existing is None:
            msg = f"unknown user agent: {agent_id!r}"
            raise KeyError(msg)
        now = _now_iso()
        self._conn.execute(
            """
            UPDATE user_agents SET
                name = ?, description = ?, system_prompt = ?,
                model_route = ?, model_override = ?,
                enabled_skills_json = ?, enabled_mcp_tools_json = ?,
                brain_access = ?, max_sensitivity_tier = ?,
                schedule_cron = ?, schedule_enabled = ?,
                pattern = ?, subagents_json = ?,
                delivery_tools_json = ?,
                delivery_targets_json = ?,
                updated_at = ?, version = version + 1
            WHERE agent_id = ?
            """,
            (
                upsert.name,
                upsert.description,
                upsert.system_prompt,
                upsert.model_route,
                upsert.model_override,
                json.dumps(list(upsert.enabled_skills)),
                json.dumps(list(upsert.enabled_mcp_tools)),
                int(bool(upsert.brain_access)),
                int(upsert.max_sensitivity_tier),
                upsert.schedule_cron,
                int(bool(upsert.schedule_enabled)),
                upsert.pattern,
                json.dumps(list(upsert.subagents)),
                json.dumps(list(upsert.delivery_tools)),
                json.dumps(_targets_to_jsonable(upsert.delivery_targets)),
                now,
                agent_id,
            ),
        )
        row = self.get(agent_id)
        assert row is not None  # noqa: S101
        return row

    def set_schedule(
        self,
        agent_id: str,
        *,
        cron: str | None,
        enabled: bool,
    ) -> UserAgentRow:
        """Persist the schedule cron + enabled flag.

        Source / callable / delivery tool selection lives in
        ``enabled_mcp_tools`` and ``delivery_tools``; the "Edit
        schedule" modal now only owns cron + enabled.

        sensitivity_tier: 1
        """
        if self.get(agent_id) is None:
            msg = f"unknown user agent: {agent_id!r}"
            raise KeyError(msg)
        self._conn.execute(
            """
            UPDATE user_agents
            SET schedule_cron = ?, schedule_enabled = ?, updated_at = ?
            WHERE agent_id = ?
            """,
            (cron, int(bool(enabled)), _now_iso(), agent_id),
        )
        row = self.get(agent_id)
        assert row is not None  # noqa: S101
        return row

    def snapshot_pre_ai_edit(
        self,
        agent_id: str,
        *,
        prev_system_prompt: str,
        prev_description: str,
    ) -> None:
        """Save the just-prior prompt + description before an AI rewrite.

        Overwrites any existing snapshot — the slot only ever holds
        the most recent pre-AI state. Manual edits between AI applies
        do not reach this method; they leave the slot intact.

        sensitivity_tier: 1
        """
        if self.get(agent_id) is None:
            msg = f"unknown user agent: {agent_id!r}"
            raise KeyError(msg)
        self._conn.execute(
            """
            UPDATE user_agents
            SET pre_ai_system_prompt = ?, pre_ai_description = ?
            WHERE agent_id = ?
            """,
            (prev_system_prompt, prev_description, agent_id),
        )

    def clear_pre_ai_snapshot(self, agent_id: str) -> None:
        """Drop the pre-AI snapshot without changing the live fields.

        sensitivity_tier: 1
        """
        self._conn.execute(
            """
            UPDATE user_agents
            SET pre_ai_system_prompt = NULL, pre_ai_description = NULL
            WHERE agent_id = ?
            """,
            (agent_id,),
        )

    def revert_pre_ai_snapshot(self, agent_id: str) -> UserAgentRow:
        """Restore the pre-AI snapshot and clear the slot atomically.

        Raises :class:`LookupError` when no snapshot is on file for
        the agent — the caller should surface a friendly "nothing to
        revert" message in that case.

        sensitivity_tier: 1
        """
        existing = self.get(agent_id)
        if existing is None:
            msg = f"unknown user agent: {agent_id!r}"
            raise KeyError(msg)
        if (
            existing.pre_ai_system_prompt is None
            and existing.pre_ai_description is None
        ):
            msg = f"no pre-AI snapshot on file for {agent_id!r}"
            raise LookupError(msg)
        restored_prompt = (
            existing.pre_ai_system_prompt
            if existing.pre_ai_system_prompt is not None
            else existing.system_prompt
        )
        restored_description = (
            existing.pre_ai_description
            if existing.pre_ai_description is not None
            else existing.description
        )
        self._conn.execute(
            """
            UPDATE user_agents SET
                system_prompt = ?, description = ?,
                pre_ai_system_prompt = NULL, pre_ai_description = NULL,
                updated_at = ?, version = version + 1
            WHERE agent_id = ?
            """,
            (restored_prompt, restored_description, _now_iso(), agent_id),
        )
        row = self.get(agent_id)
        assert row is not None  # noqa: S101
        return row

    def delete(self, agent_id: str) -> bool:
        cur = self._conn.execute(
            "DELETE FROM user_agents WHERE agent_id = ?", (agent_id,),
        )
        return cur.rowcount > 0


def _row_to_agent(row: tuple) -> UserAgentRow:
    return UserAgentRow(
        agent_id=row[0],
        name=row[1],
        description=row[2] or "",
        system_prompt=row[3],
        model_route=row[4] or "inherit",
        model_override=row[5],
        enabled_skills=tuple(json.loads(row[6] or "[]")),
        enabled_mcp_tools=tuple(json.loads(row[7] or "[]")),
        brain_access=bool(row[8]),
        max_sensitivity_tier=int(row[9] or 2),
        schedule_cron=row[10],
        schedule_enabled=bool(row[11]),
        created_at=row[12],
        updated_at=row[13],
        version=int(row[14] or 1),
        pre_ai_system_prompt=row[15],
        pre_ai_description=row[16],
        pattern=row[17] or "single",
        subagents=tuple(json.loads(row[18] or "[]")),
        delivery_tools=tuple(json.loads(row[19] or "[]")),
        delivery_targets=_parse_targets(row[20]),
    )


__all__ = [
    "DEFAULT_DB_PATH",
    "UserAgentRow",
    "UserAgentStore",
    "UserAgentUpsert",
    "make_agent_id",
]
