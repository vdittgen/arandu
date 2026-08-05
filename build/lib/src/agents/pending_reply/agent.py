"""Pydantic AI pending-reply detector.

Given a batch of recent messages (and optional active topics), decide
which ones need the user's reply, score importance 1-10, and tag a
domain. Returns :class:`PendingReplyBatch`; the orchestrator joins in
contact, source, timestamp, sensitivity tier, and persists the result.

sensitivity_tier: 2
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from src.agents.core.agent_base import SBAgent
from src.agents.core.output_types import PendingReplyBatch
from src.agents.core.scheduler import Tier

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """\
You decide which of the listed messages need the user's reply. Return \
a PendingReplyBatch matching the schema.

Include:
- direct questions, requests, decisions, deadlines
- ongoing situations (work, health, family, finance) where the user \
owes a response
- anything explicitly asking the user something

Skip:
- group-wide chatter without a direct address to the user
- bots, automated alerts, system notifications
- acknowledgements like "ok", "thanks", lone stickers / emojis
- promos, marketing, newsletters

For every kept message return a PendingReplyDraft with:
- ``needs_reply``: true (we omit no-reply candidates)
- ``importance``: 1-10, base 5; raise to 8+ for time-sensitive or \
emotionally weighty asks
- ``domain``: one of personal, work, family, social, health
- ``reason``: a short phrase grounded in the message content (≤ 12 \
words; never echo prompt text or a generic platitude)\
"""


@dataclass(frozen=True)
class PendingReplyDeps:
    """Typed input bundle for :class:`PendingReplyAgent`.

    sensitivity_tier: 2
    """

    messages: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    topics: dict[str, Any] = field(default_factory=dict)


class PendingReplyAgent(
    SBAgent[PendingReplyDeps | str, PendingReplyBatch],
):
    """Classify which messages in a batch need the user's reply.

    sensitivity_tier: 2
    """

    agent_id = "pending_reply"
    output_type = PendingReplyBatch
    tier = Tier.PROACTIVE
    system_prompt = DEFAULT_SYSTEM_PROMPT

    def build_prompt(
        self, deps: PendingReplyDeps | str,
    ) -> str:
        """Render deps into a JSON-laden user message.

        sensitivity_tier: 2
        """
        if isinstance(deps, str):
            return deps
        return (
            "Active important topics per contact (JSON):\n"
            f"{json.dumps(deps.topics, sort_keys=True)}\n\n"
            "Recent messages (JSON array):\n"
            f"{json.dumps(list(deps.messages))}\n\n"
            "Return only the items needing the user's reply, in "
            "importance order."
        )

    def detect(
        self,
        *,
        messages: list[dict[str, Any]],
        topics: dict[str, Any] | None = None,
    ) -> PendingReplyBatch | None:
        """Convenience entrypoint mirroring the legacy call shape.

        sensitivity_tier: 2
        """
        if not messages:
            return PendingReplyBatch(replies=[])
        deps = PendingReplyDeps(
            messages=tuple(messages), topics=topics or {},
        )
        record = self.run(deps)
        if record.output is None or record.error is not None:
            return None
        # Drop any draft the LLM included with needs_reply=False.
        filtered = [r for r in record.output.replies if r.needs_reply]
        if len(filtered) == len(record.output.replies):
            return record.output
        return PendingReplyBatch(replies=filtered)


def register_pending_reply_agent() -> None:
    """Register the pending-reply agent in the global agent registry.

    sensitivity_tier: 1
    """
    from src.agents.core.config_store import AgentConfig
    from src.agents.core.registry import (
        AgentDefinition,
        get_agent,
        register_agent,
    )

    if get_agent("pending_reply") is not None:
        return

    default = AgentConfig(
        agent_id="pending_reply",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        model_route="inherit",
        model_override=None,
        enabled_tools=(),
        enabled_skills=(),
        editable=False,
    )
    register_agent(AgentDefinition(
        agent_id="pending_reply",
        name="Pending Reply Detector",
        description=(
            "Decides which recent messages need the user's reply "
            "and scores importance + domain."
        ),
        category="evaluator",
        parent_agent="brain",
        tier=Tier.PROACTIVE,
        max_sensitivity_tier=2,
        editable=False,
        default_config=default,
        available_tools=(),
        available_skills=(),
        output_schema="PendingReplyBatch",
        pattern="single",
        factory=PendingReplyAgent,
        tags=("evaluator", "batch"),
    ))


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "PendingReplyAgent",
    "PendingReplyDeps",
    "register_pending_reply_agent",
]
