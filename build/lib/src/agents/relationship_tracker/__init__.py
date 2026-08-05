"""Relationship tracker as an SBAgent.

Migration target: ``src/extensions/builtin/relationship_tracker/
agent.py``. The legacy built-in keeps contact-scanning and the
``ext_relationship_tracker_nudges`` table; this module supplies the
LLM authoring primitive that writes the warm reach-out text.

sensitivity_tier: 2
"""

from src.agents.relationship_tracker.agent import (
    DEFAULT_SYSTEM_PROMPT,
    RelationshipTrackerAgent,
    register_relationship_tracker_agent,
)

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "RelationshipTrackerAgent",
    "register_relationship_tracker_agent",
]
