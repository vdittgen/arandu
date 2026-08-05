"""Agent execution context — the sandboxed API surface.

Provides the methods an agent can call during execution.  Every
method goes through :class:`SensitivityGuard` for access control.

sensitivity_tier: varies (depends on data accessed)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.agent_runtime.models import AgentManifest
from src.agent_runtime.sensitivity_guard import SensitivityGuard
from src.agents.firewall.egress_firewall import Lane
from src.models.llm_gateway import GatewayBlocked, chat_via_firewalls
from src.models.llm_provider import (
    LLMProvider,
    create_provider_from_settings,
)

logger = logging.getLogger(__name__)


class AgentAccessDeniedError(Exception):
    """Raised when an agent tries to access unauthorized data.

    sensitivity_tier: N/A
    """


class AgentContext:
    """Sandboxed execution context for an agent.

    Provides:
    - ``query(sql)`` — read from approved tables with tier filtering
    - ``ask_llm(prompt)`` — local Ollama only, if manifest allows
    - ``write(table, data)`` — write to ``ext_{agent_id}_*`` tables only
    - ``call_skill(skill_id)`` — call registered stateless skills
    - ``get_user_preference(key)`` — non-sensitive settings
    - ``log(message)`` — agent-specific log file

    sensitivity_tier: varies
    """

    def __init__(
        self,
        agent_id: str,
        manifest: AgentManifest,
        db_engine: Any,
        guard: SensitivityGuard,
        skills: dict[str, Any] | None = None,
        ollama_host: str = "http://localhost:11434",
        settings: dict[str, Any] | None = None,
        llm_provider: LLMProvider | None = None,
    ) -> None:
        self._agent_id = agent_id
        self._manifest = manifest
        self._db = db_engine
        self._guard = guard
        self._skills = skills or {}
        self._ollama_host = ollama_host
        self._settings = settings or {}
        if llm_provider is not None:
            self._provider = llm_provider
        else:
            self._provider = create_provider_from_settings(
                background=True,
            )
        self._llm_calls = 0
        self._rows_written = 0
        self._tables_written: set[str] = set()
        self._log_path = (
            Path.home()
            / ".arandu"
            / "data"
            / "agents"
            / agent_id
            / "logs"
            / f"{datetime.now(tz=timezone.utc).strftime('%Y%m%d')}.log"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, sql: str) -> list[dict[str, Any]]:
        """Execute a read-only SQL query against DuckDB.

        Flow:
        1. Guard validates tables/fields.
        2. Guard injects tier filter.
        3. Engine executes filtered SQL.
        4. Guard logs the access decision.

        Raises:
            AgentAccessDeniedError: If query accesses unauthorized data.

        sensitivity_tier: varies (capped by manifest.max_sensitivity_tier)
        """
        decision = self._guard.check_query(sql)
        self._guard.log_access(decision, {"action": "query", "sql": sql})

        if not decision.allowed:
            msg = f"Access denied: {decision.reason}"
            raise AgentAccessDeniedError(msg)

        max_tier = min(decision.effective_tier, self._manifest.max_sensitivity_tier)
        filtered_sql = self._guard.inject_tier_filter(sql, max_tier)

        return self._db.query(filtered_sql)

    def ask_llm(
        self,
        prompt: str,
        context_data: str = "",
        model: str | None = None,
    ) -> str:
        """Send a prompt to the LLM via :func:`chat_via_firewalls`.

        Only allowed if ``manifest.can_use_llm`` is True. The gateway
        owns provider selection: even if the agent context was given a
        remote-pointed :class:`LLMProvider` at construction time, a
        Tier 3 prompt under the balanced policy still lands on local
        Ollama because the gateway rebuilds the provider per call from
        the egress firewall's decision.

        Raises:
            AgentAccessDeniedError: If LLM access is not permitted or
                a firewall blocked the call.

        sensitivity_tier: varies (depends on prompt content)
        """
        if not self._manifest.can_use_llm:
            msg = "LLM access not permitted by agent manifest"
            raise AgentAccessDeniedError(msg)

        full_prompt = prompt
        if context_data:
            full_prompt = f"{prompt}\n\nContext:\n{context_data}"
        messages = [{"role": "user", "content": full_prompt}]

        try:
            resp = chat_via_firewalls(
                messages,
                agent_id=self._agent_id,
                lane=Lane.BACKGROUND,
                agent_max_tier=self._manifest.max_sensitivity_tier,
                model_override=model,
            )
        except GatewayBlocked as exc:
            raise AgentAccessDeniedError(str(exc)) from exc
        except Exception:
            logger.warning(
                "LLM call failed for agent %s",
                self._agent_id,
                exc_info=True,
            )
            return ""

        self._llm_calls += 1
        return resp.content

    def write(self, table: str, data: list[dict[str, Any]]) -> int:
        """Write rows to an agent-owned DuckDB table.

        Table must match pattern ``ext_{agent_id}_*`` and be declared
        in ``manifest.write_tables``.  Auto-creates table on first write.

        Returns:
            Number of rows written.

        Raises:
            AgentAccessDeniedError: If table name is not valid.

        sensitivity_tier: 1 (agent output data)
        """
        if not self._guard.validate_write_table(table):
            msg = f"Write denied: table '{table}' not permitted"
            raise AgentAccessDeniedError(msg)

        if not data:
            return 0

        self._ensure_table(table, data[0])

        columns = list(data[0].keys())
        placeholders = ", ".join(["?"] * len(columns))
        col_list = ", ".join(columns)
        insert_sql = f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})"

        rows_inserted = 0
        for row in data:
            values = [row.get(c) for c in columns]
            self._db.execute(insert_sql, values)
            rows_inserted += 1

        self._rows_written += rows_inserted
        self._tables_written.add(table)

        self._guard.log_access(
            self._guard.check_table_access(table),
            {"action": "write", "table": table, "rows": rows_inserted},
        )

        return rows_inserted

    def call_skill(self, skill_id: str, **kwargs: Any) -> Any:
        """Invoke a registered stateless skill.

        Raises:
            KeyError: If skill_id is not found.

        sensitivity_tier: 1
        """
        from src.agent_runtime.skills import Skill

        skill = self._skills.get(skill_id)
        if skill is None:
            msg = f"Skill '{skill_id}' not found"
            raise KeyError(msg)

        if isinstance(skill, Skill):
            return skill.execute_fn(**kwargs)
        return skill(**kwargs)

    def get_user_preference(self, key: str) -> Any:
        """Read a non-sensitive user preference/setting.

        Only returns tier-1 settings. Returns None for unknown keys.

        sensitivity_tier: 1
        """
        return self._settings.get(key)

    def log(self, message: str, level: str = "info") -> None:
        """Write to the agent's log file.

        Logs stored at ``~/.arandu/data/agents/{agent_id}/logs/``.

        sensitivity_tier: 1
        """
        ts = datetime.now(tz=timezone.utc).isoformat()
        entry = f"[{ts}] [{level.upper()}] {message}\n"
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(entry)
        except OSError:
            logger.warning(
                "Failed to write log for agent %s",
                self._agent_id,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Internal state accessors (for AgentRunner)
    # ------------------------------------------------------------------

    @property
    def llm_calls(self) -> int:
        """Number of LLM calls made during this execution."""
        return self._llm_calls

    @property
    def rows_written(self) -> int:
        """Total rows written during this execution."""
        return self._rows_written

    @property
    def tables_written(self) -> tuple[str, ...]:
        """Tables written to during this execution."""
        return tuple(self._tables_written)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_table(self, table: str, sample_row: dict[str, Any]) -> None:
        """Create the table if it doesn't exist, inferring schema from data.

        sensitivity_tier: 1
        """
        columns: list[str] = []
        for col, val in sample_row.items():
            if isinstance(val, int):
                columns.append(f"{col} BIGINT")
            elif isinstance(val, float):
                columns.append(f"{col} DOUBLE")
            elif isinstance(val, bool):
                columns.append(f"{col} BOOLEAN")
            else:
                columns.append(f"{col} VARCHAR")

        columns.append("sensitivity_tier INTEGER NOT NULL DEFAULT 1")
        columns.append("created_at TEXT NOT NULL DEFAULT current_timestamp")

        ddl = (
            f"CREATE TABLE IF NOT EXISTS {table} ("
            + ", ".join(columns)
            + ")"
        )
        self._db.execute(ddl)
