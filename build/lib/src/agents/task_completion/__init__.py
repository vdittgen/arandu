"""Task completion detector sub-agent.

sensitivity_tier: 2
"""

from __future__ import annotations

from src.agents.task_completion.agent import (
    DEFAULT_SYSTEM_PROMPT,
    TaskCompletionAgent,
    TaskCompletionDeps,
    register_task_completion_agent,
)

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "TaskCompletionAgent",
    "TaskCompletionDeps",
    "register_task_completion_agent",
]
