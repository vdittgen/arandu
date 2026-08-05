"""Pydantic AI per-contact topic extractor.

Given a contact name and a block of recent messages, return up to
five ongoing topics with importance + status. The pipeline retains
ownership of the message-batching and caching layer; this agent is
the LLM primitive it wraps.

sensitivity_tier: 2
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.agents.core.agent_base import SBAgent
from src.agents.core.output_types import TopicBatch
from src.agents.core.scheduler import Tier

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """\
You analyse a chat between the user and ONE specific contact. \
Extract the main ongoing topics or situations from THIS conversation \
only. Return a TopicBatch matching the schema.

Rules:
- A topic is a SPECIFIC situation, not a generic category.
  Good: "hiring a psychologist for the clinic", \
"father's cancer treatment", "house renovation in Garopaba", \
"planning birthday party for Maria".
  Bad: "work", "health", "personal" (too generic).
- Each Topic needs:
  * ``topic`` (short name)
  * ``description`` (one sentence)
  * ``importance`` 1-10 (10 = life-critical)
  * ``status`` one of "active", "resolved", "stale"
  * ``category`` one of "personal" | "life" | "work":
    - work = career, projects, income
    - life = family, health, relationships, big life chapters
    - personal = self-care, hobbies, identity, growth
- Max 5 topics. Only include meaningful ongoing situations.
- If messages are just casual chat with no clear topic, return an \
empty ``topics`` list.\
"""

# Truncate the messages block at this many characters so prompt cost
# stays bounded; the pipeline already trims, this is defence in depth.
_MAX_BLOCK_CHARS = 8000


@dataclass(frozen=True)
class TopicExtractorDeps:
    """Typed input bundle for :class:`TopicExtractorAgent`.

    sensitivity_tier: 2
    """

    contact_name: str
    messages_block: str = ""


class TopicExtractorAgent(
    SBAgent[TopicExtractorDeps | str, TopicBatch],
):
    """Extract ongoing topics from one contact's chat history.

    sensitivity_tier: 2
    """

    agent_id = "topic_extractor"
    output_type = TopicBatch
    tier = Tier.BACKGROUND
    system_prompt = DEFAULT_SYSTEM_PROMPT

    def build_prompt(
        self, deps: TopicExtractorDeps | str,
    ) -> str:
        """Render deps into the contact + messages prompt body.

        sensitivity_tier: 2
        """
        if isinstance(deps, str):
            return deps
        block = deps.messages_block or ""
        if len(block) > _MAX_BLOCK_CHARS:
            block = block[:_MAX_BLOCK_CHARS] + "\n... [truncated]"
        return (
            f"Contact: {deps.contact_name}\n\n"
            f"Recent messages (newest first):\n{block}\n\n"
            "Return up to 5 Topic entries describing ongoing "
            "situations in this conversation."
        )

    def extract(
        self,
        *,
        contact_name: str,
        messages_block: str,
    ) -> TopicBatch | None:
        """Convenience entrypoint mirroring the legacy call shape.

        sensitivity_tier: 2
        """
        if not messages_block or not messages_block.strip():
            return TopicBatch(topics=[])
        deps = TopicExtractorDeps(
            contact_name=contact_name or "(unknown)",
            messages_block=messages_block,
        )
        record = self.run(deps)
        if record.output is None or record.error is not None:
            return None
        # Defensive cap matching the system prompt.
        if len(record.output.topics) > 5:
            return TopicBatch(topics=record.output.topics[:5])
        return record.output


def register_topic_extractor_agent() -> None:
    """Register the topic extractor in the global agent registry.

    sensitivity_tier: 1
    """
    from src.agents.core.config_store import AgentConfig
    from src.agents.core.registry import (
        AgentDefinition,
        get_agent,
        register_agent,
    )

    if get_agent("topic_extractor") is not None:
        return

    default = AgentConfig(
        agent_id="topic_extractor",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        model_route="inherit",
        model_override=None,
        enabled_tools=(),
        enabled_skills=(),
        editable=False,
    )
    register_agent(AgentDefinition(
        agent_id="topic_extractor",
        name="Topic Extractor",
        description=(
            "Extracts ongoing topics from one contact's chat history. "
            "Runs inside the SQLMesh pipeline; not directly delegated "
            "by Brain."
        ),
        category="extractor",
        # Pipeline-only; UI groups under Brain for visibility.
        parent_agent="brain",
        tier=Tier.BACKGROUND,
        max_sensitivity_tier=2,
        editable=False,
        default_config=default,
        available_tools=(),
        available_skills=(),
        output_schema="TopicBatch",
        pattern="single",
        factory=TopicExtractorAgent,
        tags=("extractor", "indirect"),
    ))


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "TopicExtractorAgent",
    "TopicExtractorDeps",
    "register_topic_extractor_agent",
]
