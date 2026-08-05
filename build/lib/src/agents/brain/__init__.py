"""Brain agent v2 — Pydantic AI orchestrator.

The Brain runs as an :class:`SBOrchestrator` whose tools are the
data-recall, web-search, MCP action-proposal, notification-preference,
and (Phase 3+) delegated sub-agent calls. It returns a structured
:class:`BrainResponse` and routes through the firewalls + scheduler
like every other agent.

sensitivity_tier: 3 (processes all raw user data)
"""

from src.agents.brain.v2 import (
    BrainAgentV2,
    BrainDepsV2,
    StreamChunk,
    bootstrap_agents,
)

__all__ = [
    "BrainAgentV2",
    "BrainDepsV2",
    "StreamChunk",
    "bootstrap_agents",
]
