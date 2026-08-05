"""Task proposer sub-agent.

sensitivity_tier: 2
"""

from __future__ import annotations

from src.agents.task_proposer.agent import (
    DEFAULT_SYSTEM_PROMPT,
    TaskProposerAgent,
    TaskProposerDeps,
    register_task_proposer_agent,
)

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "TaskProposerAgent",
    "TaskProposerDeps",
    "register_task_proposer_agent",
]
