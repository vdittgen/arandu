"""Contact-context summarizer as an SBAgent.

Migration target: the ``evaluate_contact_contexts`` step inside
``src/agents/proactive_intelligence.py``. The legacy orchestrator
keeps its DB scans (message + contact joins, 7-day windows, birthday
detection) and persistence. This module provides the LLM step that
turns the per-contact aggregates into a structured situation summary.

sensitivity_tier: 2
"""

from src.agents.contact_context.agent import (
    DEFAULT_SYSTEM_PROMPT,
    ContactContextAgent,
    ContactContextDeps,
    register_contact_context_agent,
)

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "ContactContextAgent",
    "ContactContextDeps",
    "register_contact_context_agent",
]
