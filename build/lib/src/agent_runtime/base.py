"""Base class for Arandu agents.

All agents (built-in and third-party) must subclass BrainAgent
and implement the run() method.

sensitivity_tier: N/A (abstract interface)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agent_runtime.context import AgentContext
    from src.agent_runtime.models import AgentManifest, AgentResult


class BrainAgent(ABC):
    """Abstract base for all Arandu agents.

    Every agent must:
    1. Define a ``manifest`` class attribute (:class:`AgentManifest`).
    2. Implement ``run(context)`` returning :class:`AgentResult`.

    sensitivity_tier: N/A (interface definition)
    """

    manifest: AgentManifest

    @abstractmethod
    def run(self, context: AgentContext) -> AgentResult:
        """Execute the agent's logic within the sandbox.

        Args:
            context: Sandboxed API surface for data access, LLM, writes.

        Returns:
            AgentResult with output data and execution metadata.

        sensitivity_tier: varies (depends on agent implementation)
        """
        ...
