"""Pydantic AI fact extractor.

Takes a conversation block (a string) and returns a
:class:`LearnedFactBatch` of structured fact drafts. The legacy
``FactLearner`` keeps the DB layer — ``ext_learned_facts`` persistence,
upsert-on-conflict, confirm/dismiss/supersede lifecycle — and uses
this agent for the LLM step.

Pydantic-ai enforces the closed-set ``category`` literal and the
[1, 3] ``sensitivity_tier`` range so the legacy code's manual
validation can be retired once callers swap over.

sensitivity_tier: 3
"""

from __future__ import annotations

import logging

from src.agents.core.agent_base import SBAgent
from src.agents.core.output_types import LearnedFactBatch
from src.agents.core.scheduler import Tier
from src.models.prompt_cache import PROMPTS_DIR, FrozenPromptTemplate

logger = logging.getLogger(__name__)

_TEMPLATE = FrozenPromptTemplate(PROMPTS_DIR / "fact_extractor_v1.txt")
DEFAULT_SYSTEM_PROMPT = _TEMPLATE.prefix


class FactExtractorAgent(SBAgent[str, LearnedFactBatch]):
    """Extract structured personal facts from a conversation block.

    Deps is the raw conversation text. The output is a typed batch
    that the legacy ``FactLearner`` can promote to full
    :class:`LearnedFact` rows by attaching id, source pointers, and
    timestamps.

    sensitivity_tier: 3
    """

    agent_id = "fact_extractor"
    output_type = LearnedFactBatch
    tier = Tier.BACKGROUND
    system_prompt = DEFAULT_SYSTEM_PROMPT

    def extract(self, conversation: str) -> LearnedFactBatch | None:
        """Convenience wrapper returning the batch or None on failure.

        Empty conversations short-circuit without invoking the LLM.

        sensitivity_tier: 3
        """
        if not conversation or not conversation.strip():
            return LearnedFactBatch(facts=[])
        record = self.run(conversation)
        if record.output is None or record.error is not None:
            return None
        return record.output


def register_fact_extractor_agent() -> None:
    """Register the fact extractor in the global agent registry.

    sensitivity_tier: 1
    """
    from src.agents.core.config_store import AgentConfig
    from src.agents.core.registry import (
        AgentDefinition,
        get_agent,
        register_agent,
    )

    if get_agent("fact_extractor") is not None:
        return

    default = AgentConfig(
        agent_id="fact_extractor",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        model_route="inherit",
        model_override=None,
        enabled_tools=(),
        enabled_skills=(),
        editable=False,
    )
    register_agent(AgentDefinition(
        agent_id="fact_extractor",
        name="Fact Extractor",
        description=(
            "Extracts atomic personal facts from a conversation block. "
            "Categorizes each fact and assigns a sensitivity tier."
        ),
        category="extractor",
        parent_agent="brain",
        tier=Tier.BACKGROUND,
        max_sensitivity_tier=3,
        editable=False,
        default_config=default,
        available_tools=(),
        available_skills=(),
        output_schema="LearnedFactBatch",
        pattern="single",
        factory=FactExtractorAgent,
        tags=("extractor",),
    ))


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "FactExtractorAgent",
    "register_fact_extractor_agent",
]
