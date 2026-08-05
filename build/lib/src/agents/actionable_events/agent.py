"""Pydantic AI actionable-events detector.

Given a batch of upcoming calendar events, decide which need the user
to take action (prepare, bring something, RSVP, send birthday wishes)
and rate importance 1-10. Returns :class:`ActionableEventBatch`; the
orchestrator joins event metadata and persists.

sensitivity_tier: 2
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from src.agents.core.agent_base import SBAgent
from src.agents.core.output_types import ActionableEventBatch
from src.agents.core.scheduler import Tier
from src.models.prompt_cache import PROMPTS_DIR, FrozenPromptTemplate

logger = logging.getLogger(__name__)

_TEMPLATE = FrozenPromptTemplate(
    PROMPTS_DIR / "actionable_events_v1.txt",
)
DEFAULT_SYSTEM_PROMPT = _TEMPLATE.prefix


@dataclass(frozen=True)
class ActionableEventsDeps:
    """Typed input bundle for :class:`ActionableEventsAgent`.

    sensitivity_tier: 2
    """

    events: tuple[dict[str, Any], ...] = field(default_factory=tuple)


class ActionableEventsAgent(
    SBAgent[ActionableEventsDeps | str, ActionableEventBatch],
):
    """Pick the calendar events that need user action.

    sensitivity_tier: 2
    """

    agent_id = "actionable_events"
    output_type = ActionableEventBatch
    tier = Tier.PROACTIVE
    system_prompt = DEFAULT_SYSTEM_PROMPT

    def build_prompt(
        self, deps: ActionableEventsDeps | str,
    ) -> str:
        """Render deps into a JSON user message.

        sensitivity_tier: 2
        """
        if isinstance(deps, str):
            return deps
        return (
            "Upcoming events (JSON array):\n"
            f"{json.dumps(list(deps.events))}\n\n"
            "Return only events needing action, in importance order."
        )

    def detect(
        self,
        *,
        events: list[dict[str, Any]],
    ) -> ActionableEventBatch | None:
        """Convenience entrypoint mirroring the legacy call shape.

        sensitivity_tier: 2
        """
        if not events:
            return ActionableEventBatch(events=[])
        deps = ActionableEventsDeps(events=tuple(events))
        record = self.run(deps)
        if record.output is None or record.error is not None:
            return None
        return record.output


def register_actionable_events_agent() -> None:
    """Register the actionable-events agent in the global registry.

    sensitivity_tier: 1
    """
    from src.agents.core.config_store import AgentConfig
    from src.agents.core.registry import (
        AgentDefinition,
        get_agent,
        register_agent,
    )

    if get_agent("actionable_events") is not None:
        return

    default = AgentConfig(
        agent_id="actionable_events",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        model_route="inherit",
        model_override=None,
        enabled_tools=(),
        enabled_skills=(),
        editable=False,
    )
    register_agent(AgentDefinition(
        agent_id="actionable_events",
        name="Actionable Events Detector",
        description=(
            "Picks the upcoming calendar events that need the user "
            "to take action, with a one-sentence action and importance."
        ),
        category="evaluator",
        parent_agent="brain",
        tier=Tier.PROACTIVE,
        max_sensitivity_tier=2,
        editable=False,
        default_config=default,
        available_tools=(),
        available_skills=(),
        output_schema="ActionableEventBatch",
        pattern="single",
        factory=ActionableEventsAgent,
        tags=("evaluator", "batch"),
    ))


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "ActionableEventsAgent",
    "ActionableEventsDeps",
    "register_actionable_events_agent",
]
