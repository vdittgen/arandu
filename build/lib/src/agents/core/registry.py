"""Registry of all known agents in the running process.

Used by the Agents page IPC commands (``list_pydantic_agents``,
``get_agent_config``) and by the orchestrator + deep-agent base classes
when they need to look up a sub-agent by id.

The registry holds *definitions*, not running instances. Construction
is up to the caller (``BrainAgent``, ``AgentRunner``, IPC handler).

sensitivity_tier: 1
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace

from src.agents.core.config_store import AgentConfig
from src.agents.core.model_tiers import tier_model_for
from src.agents.core.scheduler import Tier

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentDefinition:
    """Everything the registry knows about one agent.

    sensitivity_tier: 1
    """

    agent_id: str
    name: str
    description: str
    category: str
    parent_agent: str | None
    tier: Tier
    max_sensitivity_tier: int
    editable: bool
    default_config: AgentConfig
    available_tools: tuple[str, ...]
    available_skills: tuple[str, ...]
    output_schema: str  # class name in src.agents.core.output_types
    pattern: str  # "single" | "orchestrator" | "deep"
    factory: Callable[..., object] | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)
    # Sub-agent ids this agent delegates to. Only meaningful when
    # ``pattern in {"orchestrator", "deep"}``. Mirrors the runtime
    # ``SBOrchestrator.subagents`` / ``SBDeepAgent.allowed_subagents``
    # tuple so the UI can render the delegation graph without
    # instantiating the factory.
    subagents: tuple[str, ...] = field(default_factory=tuple)


class _AgentRegistry:
    """Process-wide collection of ``AgentDefinition``s.

    sensitivity_tier: 1
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._agents: dict[str, AgentDefinition] = {}

    def register(self, definition: AgentDefinition) -> None:
        with self._lock:
            if definition.agent_id in self._agents:
                msg = f"Agent already registered: {definition.agent_id!r}"
                raise ValueError(msg)
            self._agents[definition.agent_id] = definition

    def get(self, agent_id: str) -> AgentDefinition | None:
        with self._lock:
            return self._agents.get(agent_id)

    def all(self) -> tuple[AgentDefinition, ...]:
        with self._lock:
            return tuple(sorted(
                self._agents.values(),
                key=lambda d: (d.parent_agent or "", d.agent_id),
            ))

    def children_of(self, parent_id: str | None) -> tuple[AgentDefinition, ...]:
        with self._lock:
            return tuple(
                d for d in self._agents.values()
                if d.parent_agent == parent_id
            )

    def unregister(self, agent_id: str) -> bool:
        with self._lock:
            return self._agents.pop(agent_id, None) is not None

    def reset_for_tests(self) -> None:
        with self._lock:
            self._agents.clear()


_registry = _AgentRegistry()


def register_agent(definition: AgentDefinition) -> None:
    """Add ``definition`` to the global registry.

    Before storing, apply the cost-tier default from
    :mod:`src.agents.core.model_tiers` when the registrar didn't pick
    a ``model_override`` explicitly. The persisted user override (set
    via the Agents page) still wins at resolve time — this only sets
    the PR-default for agents that haven't been touched in the UI.

    sensitivity_tier: 1
    """
    cfg = definition.default_config
    if cfg.model_override is None:
        tier_model = tier_model_for(definition.agent_id)
        if tier_model is not None:
            definition = replace(
                definition,
                default_config=replace(cfg, model_override=tier_model),
            )
    _registry.register(definition)


def get_agent(agent_id: str) -> AgentDefinition | None:
    """Look up an agent definition by id.

    sensitivity_tier: 1
    """
    return _registry.get(agent_id)


def all_agents() -> tuple[AgentDefinition, ...]:
    """Return every registered agent, sorted by parent then id.

    sensitivity_tier: 1
    """
    return _registry.all()


def children_of(parent_id: str | None) -> tuple[AgentDefinition, ...]:
    """Return direct children of ``parent_id`` (or top-level if None).

    sensitivity_tier: 1
    """
    return _registry.children_of(parent_id)


def unregister_agent(agent_id: str) -> bool:
    """Remove ``agent_id`` from the global registry.

    Returns ``True`` if the agent was registered and got removed,
    ``False`` if it wasn't present. Used by the user-agents code path
    to drop an agent when the user deletes it from the UI.

    sensitivity_tier: 1
    """
    return _registry.unregister(agent_id)


def reset_registry_for_tests() -> None:
    """Drop all registrations — for test isolation.

    sensitivity_tier: 1
    """
    _registry.reset_for_tests()


def filter_tools_for_agent(
    definition: AgentDefinition,
    enabled_tools: Iterable[str],
) -> tuple[str, ...]:
    """Restrict ``enabled_tools`` to the agent's allowlist.

    Returns the intersection of ``enabled_tools`` and
    ``definition.available_tools``. Unknown tool ids are dropped silently
    (Agents page validates before write; this guards against stale rows).

    sensitivity_tier: 1
    """
    allowed = set(definition.available_tools)
    return tuple(t for t in enabled_tools if t in allowed)


__all__ = [
    "AgentDefinition",
    "all_agents",
    "children_of",
    "filter_tools_for_agent",
    "get_agent",
    "register_agent",
    "reset_registry_for_tests",
    "unregister_agent",
]
