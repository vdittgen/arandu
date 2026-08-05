"""Sensitivity classifier as an SBAgent.

Migration target: ``src/models/sensitivity_classifier.py``. The legacy
``SensitivityClassifier`` class keeps working; this module adds a new
:class:`SensitivityAgent` that other agents (Brain v2's delegation
tools, future evaluators) can call through the Phase 1 base class.

sensitivity_tier: N/A (classifier itself stores no user data)
"""

from src.agents.sensitivity.agent import (
    DEFAULT_SYSTEM_PROMPT,
    SensitivityAgent,
    register_sensitivity_agent,
)

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "SensitivityAgent",
    "register_sensitivity_agent",
]
