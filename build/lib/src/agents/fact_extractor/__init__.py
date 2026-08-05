"""Fact extractor — SBAgent + DB persistence.

The :class:`FactExtractorAgent` is the LLM classification primitive;
:class:`FactLearner` (re-exported from :mod:`.persistence`) owns the
``_learned_facts`` table lifecycle (persist, confirm, dismiss,
supersede). Both names live here so callers can keep doing
``from src.agents.fact_extractor import FactLearner`` after the
Phase E relocation from ``src/agents/fact_learner.py``.

sensitivity_tier: 3
"""

from src.agents.fact_extractor.agent import (
    DEFAULT_SYSTEM_PROMPT,
    FactExtractorAgent,
    register_fact_extractor_agent,
)
from src.agents.fact_extractor.persistence import (
    FactLearner,
    LearnedFact,
)

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "FactExtractorAgent",
    "FactLearner",
    "LearnedFact",
    "register_fact_extractor_agent",
]
