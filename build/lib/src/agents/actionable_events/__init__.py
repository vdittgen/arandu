"""Actionable-events detector as an SBAgent.

Migration target: the actionable-event step inside
``src/agents/proactive_intelligence.py``. The legacy orchestrator
keeps the calendar scan + birthday detection. This module provides
the LLM step that picks which events need user action and rates them.

sensitivity_tier: 2
"""

from src.agents.actionable_events.agent import (
    DEFAULT_SYSTEM_PROMPT,
    ActionableEventsAgent,
    ActionableEventsDeps,
    register_actionable_events_agent,
)

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "ActionableEventsAgent",
    "ActionableEventsDeps",
    "register_actionable_events_agent",
]
