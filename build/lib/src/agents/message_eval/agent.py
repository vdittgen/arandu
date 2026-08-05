"""Pydantic AI topic-driven message evaluator.

Given a batch of new messages plus the user's active important topics,
today's events, and already-flagged ids, picks which messages warrant
a notification and at what importance. Pydantic-ai enforces the
notification-type literal and the [1, 10] importance range, so the
legacy ``_validate_notification`` helper can be retired when callers
swap over.

sensitivity_tier: 3
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from src.agents.core.agent_base import SBAgent
from src.agents.core.output_types import MessageNotificationBatch
from src.agents.core.scheduler import Tier
from src.models.prompt_cache import PROMPTS_DIR, FrozenPromptTemplate

logger = logging.getLogger(__name__)

_TEMPLATE = FrozenPromptTemplate(PROMPTS_DIR / "message_eval_v1.txt")
DEFAULT_SYSTEM_PROMPT = _TEMPLATE.prefix

# Soft cap when the agent's response includes more drafts than the
# system prompt allowed.
MAX_NOTIFICATIONS = 3


@dataclass(frozen=True)
class MessageEvalDeps:
    """Typed input bundle for :class:`MessageEvaluatorAgent`.

    sensitivity_tier: 3
    """

    messages: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    topics: dict[str, Any] = field(default_factory=dict)
    today_events: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    existing_pending_ids: tuple[str, ...] = field(default_factory=tuple)


class MessageEvaluatorAgent(
    SBAgent[MessageEvalDeps | str, MessageNotificationBatch],
):
    """Pick the topic-relevant notifications from a batch of messages.

    Deps shape: prefer :class:`MessageEvalDeps`, but a raw string is
    also accepted for orchestrator-driven delegation.

    sensitivity_tier: 3
    """

    agent_id = "message_evaluator"
    output_type = MessageNotificationBatch
    tier = Tier.PROACTIVE
    system_prompt = DEFAULT_SYSTEM_PROMPT

    def build_prompt(
        self, deps: MessageEvalDeps | str,
    ) -> str:
        """Render deps into a JSON-laden user message.

        sensitivity_tier: 3
        """
        if isinstance(deps, str):
            return deps
        return (
            "Active important topics per contact (JSON):\n"
            f"{json.dumps(deps.topics, sort_keys=True)}\n\n"
            "Today's upcoming events (JSON array):\n"
            f"{json.dumps(list(deps.today_events))}\n\n"
            "Messages already flagged (JSON array of ids):\n"
            f"{json.dumps(list(deps.existing_pending_ids))}\n\n"
            "New messages to evaluate (JSON array):\n"
            f"{json.dumps(list(deps.messages))}\n\n"
            "Return a MessageNotificationBatch with up to "
            f"{MAX_NOTIFICATIONS} drafts, in importance order."
        )

    def evaluate(
        self,
        *,
        messages: list[dict[str, Any]],
        topics: dict[str, Any] | None = None,
        today_events: list[dict[str, Any]] | None = None,
        existing_pending_ids: list[str] | None = None,
    ) -> MessageNotificationBatch | None:
        """Convenience entrypoint mirroring the legacy call shape.

        sensitivity_tier: 3
        """
        if not messages:
            return MessageNotificationBatch(notifications=[])
        deps = MessageEvalDeps(
            messages=tuple(messages),
            topics=topics or {},
            today_events=tuple(today_events or ()),
            existing_pending_ids=tuple(existing_pending_ids or ()),
        )
        record = self.run(deps)
        if record.output is None or record.error is not None:
            return None
        # Defensive cap — system prompt asks for <=3 but agents
        # sometimes return more.
        if len(record.output.notifications) > MAX_NOTIFICATIONS:
            return MessageNotificationBatch(
                notifications=record.output.notifications[
                    :MAX_NOTIFICATIONS
                ],
            )
        return record.output


def register_message_evaluator_agent() -> None:
    """Register the message evaluator in the global agent registry.

    sensitivity_tier: 1
    """
    from src.agents.core.config_store import AgentConfig
    from src.agents.core.registry import (
        AgentDefinition,
        get_agent,
        register_agent,
    )

    if get_agent("message_evaluator") is not None:
        return

    default = AgentConfig(
        agent_id="message_evaluator",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        model_route="inherit",
        model_override=None,
        enabled_tools=(),
        enabled_skills=(),
        editable=False,
    )
    register_agent(AgentDefinition(
        agent_id="message_evaluator",
        name="Message Evaluator",
        description=(
            "Picks the topic-relevant messages worth notifying the "
            "user about, given active important topics and today's "
            "schedule."
        ),
        category="evaluator",
        parent_agent="brain",
        tier=Tier.PROACTIVE,
        max_sensitivity_tier=3,
        editable=False,
        default_config=default,
        available_tools=(),
        available_skills=(),
        output_schema="MessageNotificationBatch",
        pattern="single",
        factory=MessageEvaluatorAgent,
        tags=("evaluator", "batch"),
    ))


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "MAX_NOTIFICATIONS",
    "MessageEvalDeps",
    "MessageEvaluatorAgent",
    "register_message_evaluator_agent",
]
