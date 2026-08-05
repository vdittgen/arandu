"""Mount user-authored agents into the global registry.

Reads every row of :class:`UserAgentStore` and builds an
``AgentDefinition`` whose factory produces a single-call ``SBAgent``
with optional ``recall_context`` (Brain access), MCP action tools,
and registered skills attached as pydantic-ai tools.

sensitivity_tier: 1
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic_ai import RunContext
else:
    try:
        from pydantic_ai import RunContext
    except ImportError:  # pragma: no cover
        RunContext = Any  # type: ignore[assignment,misc]

from src.agents.core.agent_base import SBAgent, SBOrchestrator
from src.agents.core.config_store import AgentConfig
from src.agents.core.output_types import BrainResponse
from src.agents.core.registry import (
    AgentDefinition,
    get_agent,
    register_agent,
    unregister_agent,
)
from src.agents.core.scheduler import Tier
from src.agents.user_agents.store import UserAgentRow, UserAgentStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-row agent class
# ---------------------------------------------------------------------------


def _resolve_user_agent_prompt(row: UserAgentRow) -> str:
    """Return the prompt to use at runtime: agent_configs override, else baseline.

    ``cmd_agents_update`` writes textarea edits to ``agent_configs``;
    ``cmd_agents_user_apply_prompt_edit`` / the AI-rewrite flow writes
    to both ``user_agents`` (baseline) and ``agent_configs``. So
    looking up ``agent_configs`` first and falling back to the row's
    ``system_prompt`` yields the live edited value either way. When
    the override row is missing or its ``system_prompt`` is blank, we
    use the baseline.

    Failures are swallowed and fall back to the baseline — agent
    construction must never abort on a transient SQLite issue.

    sensitivity_tier: 1
    """
    try:
        from src.agents.core.config_store import DEFAULT_DB_PATH
        from src.core.sqlite.engine import connect_with_pragmas

        if not DEFAULT_DB_PATH.exists():
            return row.system_prompt
        conn = connect_with_pragmas(DEFAULT_DB_PATH)
        try:
            cur = conn.execute(
                "SELECT system_prompt FROM agent_configs "
                "WHERE agent_id = ?",
                (row.agent_id,),
            )
            result = cur.fetchone()
        finally:
            conn.close()
        if result is None:
            return row.system_prompt
        override = result[0]
        if isinstance(override, str) and override.strip():
            return override
        return row.system_prompt
    except Exception:  # noqa: BLE001
        return row.system_prompt


def _append_skill_menu(
    prompt: str,
    enabled_skills: tuple[str, ...],
) -> str:
    """Append L1 skill menu to a user agent's system prompt.

    sensitivity_tier: 1
    """
    if not enabled_skills:
        return prompt
    try:
        from src.agent_runtime.skill_loader import build_skill_menu
        menu = build_skill_menu(only_ids=enabled_skills)
        if menu:
            return f"{prompt}\n\n{menu}"
    except Exception:  # noqa: BLE001
        pass
    return prompt


class _UserAgent(SBAgent[str, BrainResponse]):
    """A user-authored agent. Returns a :class:`BrainResponse`.

    The class is instantiated once per row by the factory closure
    below. ``register_tools`` wires whichever optional capabilities
    the row enabled.

    sensitivity_tier: 1
    """

    output_type = BrainResponse
    tier = Tier.INTERACTIVE

    def __init__(
        self,
        row: UserAgentRow,
        *,
        query_engine: Any | None = None,
    ) -> None:
        super().__init__()
        self._row = row
        self._query_engine = query_engine
        self._last_sources: list[dict[str, Any]] = []
        # SBAgent reads ``agent_id`` / ``system_prompt`` as class
        # attributes; bind them per instance. The effective prompt
        # is resolved with ``agent_configs`` taking precedence over
        # the ``user_agents`` baseline so a textarea-save in the UI
        # (which writes only to ``agent_configs``) is honored at
        # runtime. The ``user_agents`` row is the "saved" baseline
        # that ``cmd_agents_reset`` restores when the override is
        # cleared.
        effective_prompt = _resolve_user_agent_prompt(row)
        effective_prompt = _append_skill_menu(
            effective_prompt, row.enabled_skills,
        )
        type(self).agent_id  # noqa: B018
        self.__class__ = type(  # dynamic subclass for per-row metadata
            f"_UserAgent_{row.agent_id.replace('.', '_')}",
            (_UserAgent,),
            {
                "agent_id": row.agent_id,
                "system_prompt": effective_prompt,
            },
        )

    def register_tools(self, pa_agent: Any) -> None:
        _register_user_capability_tools(
            pa_agent,
            row=self._row,
            query_engine=self._query_engine,
            sources_box=self._last_sources_box(),
        )
        super().register_tools(pa_agent)

    def _last_sources_box(self) -> list[dict[str, Any]]:
        """Return the mutable list the ``recall_context`` tool updates.

        Wrapping in a method keeps the closure tied to the instance —
        the helper writes into ``self._last_sources`` via this list.

        sensitivity_tier: 1
        """
        return self._last_sources


# ---------------------------------------------------------------------------
# Orchestrator variant
# ---------------------------------------------------------------------------


class _UserOrchestrator(SBOrchestrator[str, BrainResponse]):
    """A user-authored orchestrator that delegates to other agents.

    Wraps :class:`SBOrchestrator` so one pydantic-ai delegation tool is
    attached per id in ``subagents``. The same recall/skill/MCP tools
    available to :class:`_UserAgent` are also attached so the LLM can
    mix direct tool use with delegation.

    sensitivity_tier: 1
    """

    output_type = BrainResponse
    tier = Tier.INTERACTIVE

    def __init__(
        self,
        row: UserAgentRow,
        *,
        query_engine: Any | None = None,
    ) -> None:
        super().__init__()
        self._row = row
        self._query_engine = query_engine
        self._last_sources: list[dict[str, Any]] = []
        effective_prompt = _append_skill_menu(
            row.system_prompt, row.enabled_skills,
        )
        self.__class__ = type(
            f"_UserOrchestrator_{row.agent_id.replace('.', '_')}",
            (_UserOrchestrator,),
            {
                "agent_id": row.agent_id,
                "system_prompt": effective_prompt,
                "subagents": tuple(row.subagents),
            },
        )

    def register_tools(self, pa_agent: Any) -> None:
        # Delegation tools first (one per sub-agent id).
        super().register_tools(pa_agent)
        # Then the same capabilities a single-pattern user agent has,
        # so the LLM can mix direct tool calls with delegation.
        _register_user_capability_tools(
            pa_agent,
            row=self._row,
            query_engine=self._query_engine,
            sources_box=self._last_sources,
        )


# ---------------------------------------------------------------------------
# Shared capability wiring
# ---------------------------------------------------------------------------


def _register_user_capability_tools(
    pa_agent: Any,
    *,
    row: UserAgentRow,
    query_engine: Any | None,
    sources_box: list[dict[str, Any]],
) -> None:
    """Attach recall_context / load_skill / run_mcp_tool based on row flags.

    Shared by :class:`_UserAgent` and :class:`_UserOrchestrator` so the
    set of tools a user agent gets is identical regardless of pattern.

    sensitivity_tier: 1
    """
    if row.brain_access and query_engine is not None:
        engine = query_engine

        async def recall_context(_ctx: RunContext[None], query: str) -> str:
            """Fetch personal context via the Brain query engine."""
            from src.agents.brain.context import (
                build_context_summary,
                format_context,
                truncate_context,
            )
            qctx = engine.query(
                query,
                max_sensitivity_tier=row.max_sensitivity_tier,
            )
            text, sources = format_context(qctx)
            sources_box.clear()
            sources_box.extend(sources)
            _ = build_context_summary(qctx)
            return truncate_context(text) or "No personal context."

        pa_agent.tool(recall_context)

    # Skills are injected into the system prompt via _append_skill_menu.
    # User agents with enabled_skills also get load_skill / load_skill_resource
    # tools so the LLM can read full instructions on demand.
    if row.enabled_skills:
        enabled_ids = set(row.enabled_skills)

        async def load_skill(
            _ctx: RunContext[None],
            skill_id: str,
        ) -> str:
            """Load full instructions for an enabled skill."""
            if skill_id not in enabled_ids:
                return f"Skill '{skill_id}' is not enabled for this agent."
            from src.agent_runtime.skill_loader import SkillLoader
            loader = SkillLoader()
            doc = loader.load(skill_id)
            if doc is None:
                return f"Skill '{skill_id}' not found."
            return doc.instructions

        async def load_skill_resource(
            _ctx: RunContext[None],
            skill_id: str,
            path: str,
        ) -> str:
            """Read a resource file bundled with an enabled skill."""
            if skill_id not in enabled_ids:
                return f"Skill '{skill_id}' is not enabled for this agent."
            from src.agent_runtime.skill_loader import SkillLoader
            loader = SkillLoader()
            content = loader.load_resource(skill_id, path)
            if content is None:
                return f"Resource '{path}' not found in skill '{skill_id}'."
            return content

        pa_agent.tool(load_skill)
        pa_agent.tool(load_skill_resource)

    callable_tool_specs = _callable_mcp_tool_specs(row)
    if callable_tool_specs:
        tool_specs = callable_tool_specs

        async def run_mcp_tool(
            _ctx: RunContext[None],
            tool_id: str,
            arguments: dict[str, Any] | None = None,
        ) -> str:
            """Invoke one of this agent's enabled MCP action tools.

            Data tools (sources) and delivery tools are intentionally
            excluded — sources are pulled by the runner each tick, and
            delivery tools are dispatched by the post-batch hook only
            so the LLM cannot send an unfinished thought mid-loop.
            """
            if tool_id not in tool_specs:
                return f"tool {tool_id!r} not enabled for this agent"
            return _run_mcp_action(tool_id, arguments or {})

        pa_agent.tool(run_mcp_tool)


def _callable_mcp_tool_specs(row: UserAgentRow) -> tuple[str, ...]:
    """Return the subset of ``enabled_mcp_tools`` the LLM may invoke.

    The LLM-facing ``run_mcp_tool`` dispatcher only accepts catalog
    ``action`` tools that are NOT in ``row.delivery_tools``. Data
    tools never reach the LLM (they're consumed by the runner each
    tick), and delivery tools never reach the LLM during per-item
    runs (so the at-least-once delivery contract stays the runner's
    responsibility, not the model's discretion).

    sensitivity_tier: 1
    """
    if not row.enabled_mcp_tools:
        return ()
    try:
        from src.extensions.connectors.catalog import ConnectorCatalog
    except Exception:  # noqa: BLE001
        # No catalog → assume every entry is a callable action tool;
        # mirrors the pre-unification behavior so test setups without
        # the catalog still wire tools through.
        return tuple(
            t for t in row.enabled_mcp_tools
            if t not in row.delivery_tools
        )
    catalog = ConnectorCatalog()
    delivery = set(row.delivery_tools)
    out: list[str] = []
    for tool_id in row.enabled_mcp_tools:
        if tool_id in delivery:
            continue
        if ":" not in tool_id:
            continue
        connector_id, tool_name = tool_id.split(":", 1)
        template = catalog.get(connector_id)
        if template is None:
            continue
        for tool in template.tools:
            if tool.tool_name == tool_name and tool.tool_type == "action":
                out.append(tool_id)
                break
    return tuple(out)


def _run_mcp_action(tool_id: str, arguments: dict[str, Any]) -> str:
    """Dispatch ``connector_id:tool_name`` through the MCP client.

    The connector + tool id format mirrors the picker in the Agents
    page modal. Errors are returned as strings so the LLM can react
    rather than crashing the run.

    sensitivity_tier: 2
    """
    if ":" not in tool_id:
        return f"invalid tool id: {tool_id!r} (expected connector:tool)"
    connector_id, tool_name = tool_id.split(":", 1)
    try:
        from src.extensions.mcp.client import MCPClient
    except ImportError:
        return "mcp client unavailable"
    try:
        client = MCPClient(connector_id=connector_id)
        result = client.call_tool(tool_name, arguments)
        return str(result)
    except Exception as exc:  # noqa: BLE001
        logger.warning("MCP call failed for %s: %s", tool_id, exc)
        return f"mcp call failed: {exc}"


# ---------------------------------------------------------------------------
# Registry hooks
# ---------------------------------------------------------------------------


def _definition_for(
    row: UserAgentRow,
    *,
    query_engine: Any | None,
) -> AgentDefinition:
    default = AgentConfig(
        agent_id=row.agent_id,
        system_prompt=row.system_prompt,
        model_route=row.model_route,
        model_override=row.model_override,
        enabled_tools=tuple(row.enabled_mcp_tools),
        enabled_skills=tuple(row.enabled_skills),
        editable=True,
    )
    callable_specs = _callable_mcp_tool_specs(row)
    available_tools: tuple[str, ...] = ()
    if row.brain_access:
        available_tools += ("recall_context",)
    if row.enabled_skills:
        available_tools += ("load_skill", "load_skill_resource")
    if callable_specs:
        available_tools += ("run_mcp_tool",)
    if row.delivery_tools:
        # Surface delivery tools so the Agents page can render the
        # post-batch hook membership without a second registry lookup.
        available_tools += tuple(f"deliver:{t}" for t in row.delivery_tools)
    if row.pattern == "orchestrator":
        # Surface each delegation target so the Agents page can render
        # "calls: foo, bar, baz" without a second registry lookup.
        available_tools += tuple(f"delegate:{sub}" for sub in row.subagents)

    if row.pattern == "orchestrator":
        def _factory() -> Any:
            return _UserOrchestrator(row, query_engine=query_engine)
    else:
        def _factory() -> Any:
            return _UserAgent(row, query_engine=query_engine)

    return AgentDefinition(
        agent_id=row.agent_id,
        name=row.name,
        description=row.description,
        category="user",
        parent_agent="brain",
        tier=Tier.INTERACTIVE,
        max_sensitivity_tier=row.max_sensitivity_tier,
        editable=True,
        default_config=default,
        available_tools=available_tools,
        available_skills=tuple(row.enabled_skills),
        output_schema="BrainResponse",
        pattern=row.pattern,
        factory=_factory,
        tags=("user",),
        subagents=tuple(row.subagents),
    )


def register_user_agents(*, query_engine: Any | None = None) -> None:
    """Pull every user-authored agent from SQLite and register it.

    Idempotent — re-registering the same id is a no-op. Used at
    process start (from :func:`bootstrap_agents`) and after a create
    / update / delete from the IPC layer so the in-process registry
    matches the SQLite source of truth.

    sensitivity_tier: 1
    """
    try:
        store = UserAgentStore()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not open user_agents store: %s", exc)
        return
    try:
        for row in store.list_all():
            if get_agent(row.agent_id) is not None:
                continue
            try:
                register_agent(_definition_for(row, query_engine=query_engine))
            except ValueError:
                # Already registered by a concurrent call — fine.
                continue
    finally:
        store.close()


def register_one_user_agent(
    row: UserAgentRow,
    *,
    query_engine: Any | None = None,
) -> None:
    """Mount (or replace) a single user-authored agent.

    sensitivity_tier: 1
    """
    unregister_agent(row.agent_id)
    register_agent(_definition_for(row, query_engine=query_engine))


def unregister_user_agent(agent_id: str) -> bool:
    """Drop a user-authored agent from the registry.

    Refuses non-``user.*`` agent ids so the IPC layer can't be tricked
    into pulling a shipped agent out of the registry.

    sensitivity_tier: 1
    """
    if not agent_id.startswith("user."):
        return False
    return unregister_agent(agent_id)


__all__ = [
    "register_one_user_agent",
    "register_user_agents",
    "unregister_user_agent",
]
