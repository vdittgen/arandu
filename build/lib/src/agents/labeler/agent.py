"""Pydantic AI emotional labeler.

Single-message version of the legacy :class:`EmotionalLabeler`. Returns
:class:`EmotionalLabel` directly — pydantic-ai enforces the literals
(primary_emotion, domain) and the float range on intensity so the
brittle ``_validate_label`` helper from the legacy code is no longer
needed.

Batch labelling (the legacy code does up to 10 at once for throughput)
is intentionally not part of this agent. Callers that need batches
either run multiple agents in parallel through the scheduler or use a
specialised future agent (Phase 3b's ``LabelerBatchAgent``).

sensitivity_tier: 3
"""

from __future__ import annotations

import logging

from src.agents.core.agent_base import SBAgent
from src.agents.core.output_types import EmotionalLabel
from src.agents.core.scheduler import Tier
from src.models.prompt_cache import PROMPTS_DIR, FrozenPromptTemplate

logger = logging.getLogger(__name__)

_TEMPLATE = FrozenPromptTemplate(PROMPTS_DIR / "labeler_agent_v1.txt")
DEFAULT_SYSTEM_PROMPT = _TEMPLATE.prefix


class LabelerAgent(SBAgent[str, EmotionalLabel]):
    """Classify a single text into structured emotional dimensions.

    Deps is a raw string — the text to label. Pydantic-ai enforces the
    schema, including the closed-set primary_emotion + domain literals
    and the [0.0, 1.0] intensity range.

    sensitivity_tier: 3
    """

    agent_id = "labeler"
    output_type = EmotionalLabel
    tier = Tier.BACKGROUND
    system_prompt = DEFAULT_SYSTEM_PROMPT

    def label(self, text: str) -> EmotionalLabel | None:
        """Convenience wrapper returning the label or None on failure.

        Mirrors the shape of the legacy ``EmotionalLabeler.label_text``
        so callers can swap to this agent with a minimal diff.

        sensitivity_tier: 3
        """
        if not text:
            return None
        record = self.run(text)
        if record.output is None or record.error is not None:
            return None
        return record.output


def register_labeler_agent() -> None:
    """Register the labeler agent in the global agent registry.

    sensitivity_tier: 1
    """
    from src.agents.core.config_store import AgentConfig
    from src.agents.core.registry import (
        AgentDefinition,
        get_agent,
        register_agent,
    )

    if get_agent("labeler") is not None:
        return

    default = AgentConfig(
        agent_id="labeler",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        model_route="inherit",
        model_override=None,
        enabled_tools=(),
        enabled_skills=(),
        editable=False,
    )
    register_agent(AgentDefinition(
        agent_id="labeler",
        name="Emotional Labeler",
        description=(
            "Classifies a single text into structured emotional "
            "dimensions (emotion, intensity, feelings, desires, domain)."
        ),
        category="classifier",
        parent_agent="brain",
        tier=Tier.BACKGROUND,
        max_sensitivity_tier=3,
        editable=False,
        default_config=default,
        available_tools=(),
        available_skills=(),
        output_schema="EmotionalLabel",
        pattern="single",
        factory=LabelerAgent,
        tags=("classifier",),
    ))


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "LabelerAgent",
    "register_labeler_agent",
]
