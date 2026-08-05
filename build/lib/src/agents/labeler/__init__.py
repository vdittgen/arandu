"""Emotional labeler as an SBAgent.

Migration target: ``src/models/labeler.py``. The legacy
``EmotionalLabeler`` class keeps working; this module adds a new
:class:`LabelerAgent` returning a typed :class:`EmotionalLabel`.

sensitivity_tier: 3 (sees free-text content; output is Tier 3)
"""

from src.agents.labeler.agent import (
    DEFAULT_SYSTEM_PROMPT,
    LabelerAgent,
    register_labeler_agent,
)

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "LabelerAgent",
    "register_labeler_agent",
]
