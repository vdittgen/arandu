"""Model generator as an SBAgent.

Migration target: ``src/extensions/ingestion/model_generator.py``.
Legacy keeps the rule-based SQL templating; this agent powers the
optional LLM enhancement that produces a complete SQLMesh model.

sensitivity_tier: 1
"""

from src.agents.model_generator.agent import (
    DEFAULT_SYSTEM_PROMPT,
    ModelGeneratorAgent,
    ModelGeneratorDeps,
    register_model_generator_agent,
)

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "ModelGeneratorAgent",
    "ModelGeneratorDeps",
    "register_model_generator_agent",
]
