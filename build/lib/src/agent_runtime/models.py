"""Agent system data models.

Frozen dataclasses for agent manifests, results, and status.
Used by AgentRunner, AgentContext, and SensitivityGuard.

sensitivity_tier: 1 (metadata only, no user data)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TriggerMode(str, Enum):
    """When an agent should be triggered.

    sensitivity_tier: 1
    """

    SCHEDULED = "scheduled"
    ON_DATA_CHANGE = "on_data_change"
    MANUAL = "manual"
    ON_QUERY = "on_query"


@dataclass(frozen=True)
class TablePermission:
    """A single table access permission declared in the manifest.

    sensitivity_tier: 1
    """

    table: str
    max_tier: int
    columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentManifest:
    """Declarative manifest for a Arandu agent.

    Loaded from manifest.yaml. Defines permissions, triggers,
    resource limits, and metadata.

    sensitivity_tier: 1
    """

    id: str
    name: str
    version: str
    description: str
    author: str

    # Permissions
    tables: tuple[TablePermission, ...] = ()
    max_sensitivity_tier: int = 1
    can_use_llm: bool = False
    write_tables: tuple[str, ...] = ()
    skills: tuple[str, ...] = ()

    # Triggers
    triggers: tuple[TriggerMode, ...] = (TriggerMode.MANUAL,)
    schedule: str | None = None

    # Resource limits
    memory_mb: int = 256
    timeout_seconds: int = 60

    # Classification
    category: str = "general"
    builtin: bool = False


@dataclass(frozen=True)
class AgentResult:
    """Result returned by an agent after execution.

    sensitivity_tier: varies (depends on agent output)
    """

    agent_id: str
    status: str
    output: str = ""
    tables_written: tuple[str, ...] = ()
    rows_written: int = 0
    llm_calls: int = 0
    duration_ms: float = 0.0
    error: str | None = None


@dataclass(frozen=True)
class AgentStatus:
    """Runtime status of a registered agent for the frontend.

    sensitivity_tier: 1
    """

    agent_id: str
    name: str
    description: str
    category: str
    status: str
    builtin: bool
    triggers: tuple[str, ...] = ()
    max_sensitivity_tier: int = 1
    last_run_at: str | None = None
    last_result: str | None = None
    error: str | None = None
