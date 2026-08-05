"""Pydantic AI sensitivity classifier.

Replaces the LLM call inside :class:`SensitivityClassifier` with a
structured ``SBAgent`` returning :class:`SensitivityVerdict`. The agent
is **non-editable** in the Agents page; behaviour is locked to the
shipped code and only changes via PR.

sensitivity_tier: N/A
"""

from __future__ import annotations

import logging

from src.agents.core.agent_base import SBAgent
from src.agents.core.output_types import SensitivityVerdict
from src.agents.core.scheduler import Tier
from src.models.prompt_cache import PROMPTS_DIR, FrozenPromptTemplate

logger = logging.getLogger(__name__)

# Frozen template loaded from src/models/prompts/. Edits to the .txt
# file invalidate the provider's prompt cache; the golden test in
# tests/unit/models/prompts/test_golden_prompts.py catches drift.
_TEMPLATE = FrozenPromptTemplate(
    PROMPTS_DIR / "sensitivity_classifier_v1.txt",
)
DEFAULT_SYSTEM_PROMPT = _TEMPLATE.prefix

# Fail-safe tier when the agent or the underlying model is unavailable.
FAIL_SAFE_TIER: int = 3


class SensitivityAgent(SBAgent[str, SensitivityVerdict]):
    """Classify a single text into a sensitivity tier (1-3).

    The deps shape is a raw string — the text to classify. This matches
    the ``SBOrchestrator`` delegation contract, so Brain can call us
    as a tool by passing the text directly.

    sensitivity_tier: N/A
    """

    agent_id = "sensitivity"
    output_type = SensitivityVerdict
    tier = Tier.BACKGROUND
    system_prompt = DEFAULT_SYSTEM_PROMPT

    def classify_tier(self, text: str) -> int:
        """Convenience wrapper returning just the integer tier.

        Falls back to :data:`FAIL_SAFE_TIER` on any error so callers
        (e.g., :class:`SensitivityClassifier`) can swap us in without
        special-casing the exception path.

        sensitivity_tier: N/A
        """
        if not text:
            return 1
        record = self.run(text)
        if record.output is None or record.error is not None:
            return FAIL_SAFE_TIER
        return int(record.output.tier)


# ---------------------------------------------------------------------------
# Registry hook
# ---------------------------------------------------------------------------


def register_sensitivity_agent() -> None:
    """Register the sensitivity agent in the global agent registry.

    Idempotent. Marked ``editable=False``; behaviour is locked to the
    shipped code.

    sensitivity_tier: 1
    """
    from src.agents.core.config_store import AgentConfig
    from src.agents.core.registry import (
        AgentDefinition,
        get_agent,
        register_agent,
    )

    if get_agent("sensitivity") is not None:
        return

    default = AgentConfig(
        agent_id="sensitivity",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        model_route="inherit",
        model_override=None,
        enabled_tools=(),
        enabled_skills=(),
        editable=False,
    )
    register_agent(AgentDefinition(
        agent_id="sensitivity",
        name="Sensitivity Classifier",
        description=(
            "Classifies a piece of text into sensitivity tier 1-3 so "
            "the firewall and egress router can decide where it may go."
        ),
        category="classifier",
        parent_agent="brain",
        tier=Tier.BACKGROUND,
        max_sensitivity_tier=3,
        editable=False,
        default_config=default,
        available_tools=(),
        available_skills=(),
        output_schema="SensitivityVerdict",
        pattern="single",
        factory=SensitivityAgent,
        tags=("classifier",),
    ))


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "FAIL_SAFE_TIER",
    "SensitivityAgent",
    "register_sensitivity_agent",
]
