"""Pydantic AI contact-context summarizer.

Given per-contact activity aggregates plus active important topics,
produces a structured situation summary per important contact:
short free-text active_context, relevant life domains, and a 0-3
priority score. The orchestrator decides which contacts to feed in
and how to combine the drafts with persisted contact metadata.

sensitivity_tier: 2
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from src.agents.core.agent_base import SBAgent
from src.agents.core.output_types import ContactContextBatch
from src.agents.core.scheduler import Tier

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """\
Build context summaries for the user's important contacts. Use the \
recent message activity and the contact's active important topics.

For each contact with a meaningful ongoing situation, return a \
ContactContextDraft with:
- ``contact_id``: the contact identifier from the input
- ``active_context``: a 1-2 sentence summary of the ongoing situation
- ``context_domains``: relevant life domains drawn from \
{health, work, family, social, personal, finance}
- ``context_priority``: 0-3 (0 = no context, 1 = low, 2 = medium, \
3 = urgent or health/crisis)

Prefer higher priority for:
- contacts with critical active topics (importance >= 7)
- health, finance, or crisis-coded situations
- explicit time pressure in recent messages

Skip contacts with no meaningful ongoing situation — return an empty \
``contexts`` list if none qualify.\
"""


@dataclass(frozen=True)
class ContactContextDeps:
    """Typed input bundle for :class:`ContactContextAgent`.

    ``contacts`` is a list of aggregate dicts as produced by the legacy
    orchestrator: id, name, message counts, last_message preview,
    upcoming events, etc.

    sensitivity_tier: 2
    """

    contacts: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    topics: dict[str, Any] = field(default_factory=dict)


class ContactContextAgent(
    SBAgent[ContactContextDeps | str, ContactContextBatch],
):
    """Summarize per-contact situations into structured drafts.

    sensitivity_tier: 2
    """

    agent_id = "contact_context"
    output_type = ContactContextBatch
    tier = Tier.PROACTIVE
    system_prompt = DEFAULT_SYSTEM_PROMPT

    def build_prompt(
        self, deps: ContactContextDeps | str,
    ) -> str:
        """Render deps into a JSON-laden user message.

        sensitivity_tier: 2
        """
        if isinstance(deps, str):
            return deps
        return (
            "Active important topics per contact (JSON):\n"
            f"{json.dumps(deps.topics, sort_keys=True)}\n\n"
            "Contact aggregates (JSON array):\n"
            f"{json.dumps(list(deps.contacts))}\n\n"
            "Return only contacts with a meaningful ongoing situation, "
            "in priority order."
        )

    def summarize(
        self,
        *,
        contacts: list[dict[str, Any]],
        topics: dict[str, Any] | None = None,
    ) -> ContactContextBatch | None:
        """Convenience entrypoint mirroring the legacy call shape.

        sensitivity_tier: 2
        """
        if not contacts:
            return ContactContextBatch(contexts=[])
        deps = ContactContextDeps(
            contacts=tuple(contacts), topics=topics or {},
        )
        record = self.run(deps)
        if record.output is None or record.error is not None:
            return None
        return record.output


def register_contact_context_agent() -> None:
    """Register the contact-context agent in the global agent registry.

    sensitivity_tier: 1
    """
    from src.agents.core.config_store import AgentConfig
    from src.agents.core.registry import (
        AgentDefinition,
        get_agent,
        register_agent,
    )

    if get_agent("contact_context") is not None:
        return

    default = AgentConfig(
        agent_id="contact_context",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        model_route="inherit",
        model_override=None,
        enabled_tools=(),
        enabled_skills=(),
        editable=False,
    )
    register_agent(AgentDefinition(
        agent_id="contact_context",
        name="Contact Context Summarizer",
        description=(
            "Summarizes per-contact situations into structured "
            "active_context + domains + priority drafts for the "
            "proactive panel."
        ),
        category="evaluator",
        parent_agent="brain",
        tier=Tier.PROACTIVE,
        max_sensitivity_tier=2,
        editable=False,
        default_config=default,
        available_tools=(),
        available_skills=(),
        output_schema="ContactContextBatch",
        pattern="single",
        factory=ContactContextAgent,
        tags=("evaluator", "batch"),
    ))


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "ContactContextAgent",
    "ContactContextDeps",
    "register_contact_context_agent",
]
