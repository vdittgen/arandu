"""Pydantic AI relationship nudge author.

Given a contact name plus context about why we're nudging (last
contact date, shared topics, relationship type), writes a brief warm
reach-out suggestion. Returns :class:`RelationshipNudge`.

sensitivity_tier: 2
"""

from __future__ import annotations

import logging

from src.agents.core.agent_base import SBAgent
from src.agents.core.output_types import RelationshipNudge
from src.agents.core.scheduler import Tier

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """\
You author a single short reminder to reach out to one of the user's \
contacts. Return a RelationshipNudge matching the schema.

Rules:
- ``contact_name``: the name from the supplied context.
- ``nudge``: a one-sentence warm suggestion, addressed to the user, \
not the contact. Reference the relationship or last interaction when \
possible. No greetings or sign-offs.
- ``suggested_topic``: an optional one-phrase opener the user could \
mention. Leave null if nothing obvious comes to mind.
- Never fabricate dates or details that aren't in the supplied \
context. If context is sparse, keep the nudge generic but warm.\
"""


class RelationshipTrackerAgent(SBAgent[str, RelationshipNudge]):
    """Write a single relationship-nudge note for a contact.

    sensitivity_tier: 2
    """

    agent_id = "relationship_tracker"
    output_type = RelationshipNudge
    tier = Tier.BACKGROUND
    system_prompt = DEFAULT_SYSTEM_PROMPT

    def author(self, context_block: str) -> RelationshipNudge | None:
        """Convenience wrapper returning the nudge or None on failure.

        sensitivity_tier: 2
        """
        if not context_block or not context_block.strip():
            return None
        record = self.run(context_block)
        if record.output is None or record.error is not None:
            return None
        return record.output


def register_relationship_tracker_agent() -> None:
    """Register the relationship tracker in the registry.

    sensitivity_tier: 1
    """
    from src.agents.core.config_store import AgentConfig
    from src.agents.core.registry import (
        AgentDefinition,
        get_agent,
        register_agent,
    )

    if get_agent("relationship_tracker") is not None:
        return

    default = AgentConfig(
        agent_id="relationship_tracker",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        model_route="inherit",
        model_override=None,
        enabled_tools=(),
        enabled_skills=(),
        editable=False,
    )
    register_agent(AgentDefinition(
        agent_id="relationship_tracker",
        name="Relationship Tracker",
        description=(
            "Writes a single warm reach-out nudge for a contact the "
            "user hasn't interacted with recently. Invoked by the "
            "built-in relationship-tracker scheduled job."
        ),
        category="author",
        parent_agent="brain",
        tier=Tier.BACKGROUND,
        max_sensitivity_tier=2,
        editable=False,
        default_config=default,
        available_tools=(),
        available_skills=(),
        output_schema="RelationshipNudge",
        pattern="single",
        factory=RelationshipTrackerAgent,
        tags=("author", "indirect", "builtin"),
    ))


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "RelationshipTrackerAgent",
    "register_relationship_tracker_agent",
]
