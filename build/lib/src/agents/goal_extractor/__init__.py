"""Goal extractor sub-agent.

Mines user-level goals (with horizon and *why*) from recent messages,
notes, learned facts, and chat history. Distinct from topics, which
are per-contact situations.

sensitivity_tier: 2
"""

from __future__ import annotations

from src.agents.goal_extractor.agent import (
    DEFAULT_SYSTEM_PROMPT,
    GoalExtractorAgent,
    GoalExtractorDeps,
    register_goal_extractor_agent,
)

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "GoalExtractorAgent",
    "GoalExtractorDeps",
    "register_goal_extractor_agent",
]
