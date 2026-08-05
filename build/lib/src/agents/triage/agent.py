"""Pydantic AI message triage.

Takes a batch of messages and returns a :class:`TriageBatch` of
keep/drop verdicts. Each verdict carries a short reason and three
boolean flags (``is_promo``, ``is_automated``, ``is_ack_only``) so
downstream code can audit why something was dropped.

The legacy ``MessageTriager`` persists verdicts to ``_triage_log``
keyed by ``message_id``. Phase 3a leaves that cache where it is; the
agent here is the pure classification primitive that the cache wraps.

sensitivity_tier: 3
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.agents.core.agent_base import SBAgent
from src.agents.core.output_types import TriageBatch
from src.agents.core.scheduler import Tier
from src.models.prompt_cache import PROMPTS_DIR, FrozenPromptTemplate

logger = logging.getLogger(__name__)

_TEMPLATE = FrozenPromptTemplate(PROMPTS_DIR / "triage_v1.txt")
DEFAULT_SYSTEM_PROMPT = _TEMPLATE.prefix

# Truncation limit per message to keep prompts bounded.
_CONTENT_TRUNCATE = 240


@dataclass(frozen=True)
class TriageMessage:
    """One candidate message for triage.

    sensitivity_tier: 2
    """

    message_id: str
    content: str
    sender_name: str = ""
    source: str = ""


@dataclass(frozen=True)
class TriageDeps:
    """Typed input batch for :class:`TriageAgent`.

    sensitivity_tier: 2
    """

    messages: tuple[TriageMessage, ...] = field(default_factory=tuple)


def _format_messages(messages: tuple[TriageMessage, ...]) -> str:
    """Format a batch into the numbered block the LLM expects.

    sensitivity_tier: 3
    """
    lines: list[str] = []
    for idx, msg in enumerate(messages, start=1):
        content = (msg.content or "").strip().replace("\n", " ")
        if len(content) > _CONTENT_TRUNCATE:
            content = content[:_CONTENT_TRUNCATE].rstrip() + "…"
        sender = msg.sender_name or "(unknown)"
        source = msg.source or ""
        lines.append(
            f"[{idx}] id={msg.message_id} "
            f"from={sender} source={source}\n    {content}",
        )
    return "\n".join(lines) if lines else "(no messages)"


class TriageAgent(SBAgent[TriageDeps | tuple[TriageMessage, ...] | str, TriageBatch]):
    """Classify a batch of messages as keep/drop.

    Accepts three deps shapes for caller convenience:

    - :class:`TriageDeps` — typed batch.
    - A tuple of :class:`TriageMessage`.
    - A raw string — taken as a single message body. Useful when an
      orchestrator delegates to us with a plain text prompt; the LLM
      will return a single-element batch.

    sensitivity_tier: 3
    """

    agent_id = "triage"
    output_type = TriageBatch
    tier = Tier.BACKGROUND
    system_prompt = DEFAULT_SYSTEM_PROMPT

    def build_prompt(
        self,
        deps: TriageDeps | tuple[TriageMessage, ...] | str,
    ) -> str:
        """Project deps into the numbered-block user message.

        sensitivity_tier: 3
        """
        if isinstance(deps, TriageDeps):
            messages = deps.messages
        elif isinstance(deps, tuple):
            messages = deps  # type: ignore[assignment]
        elif isinstance(deps, str):
            messages = (
                TriageMessage(message_id="msg_1", content=deps),
            )
        else:
            messages = ()
        return (
            "Messages to triage:\n\n"
            f"{_format_messages(messages)}\n\n"
            "Return one TriageDecision per message, preserving order."
        )

    def triage(
        self,
        messages: list[TriageMessage] | tuple[TriageMessage, ...],
    ) -> TriageBatch | None:
        """Convenience helper returning the structured batch or None.

        sensitivity_tier: 3
        """
        if not messages:
            return TriageBatch(decisions=[])
        record = self.run(TriageDeps(messages=tuple(messages)))
        if record.output is None or record.error is not None:
            return None
        return record.output


def register_triage_agent() -> None:
    """Register the triage agent in the global agent registry.

    sensitivity_tier: 1
    """
    from src.agents.core.config_store import AgentConfig
    from src.agents.core.registry import (
        AgentDefinition,
        get_agent,
        register_agent,
    )

    if get_agent("triage") is not None:
        return

    default = AgentConfig(
        agent_id="triage",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        model_route="inherit",
        model_override=None,
        enabled_tools=(),
        enabled_skills=(),
        editable=False,
    )
    register_agent(AgentDefinition(
        agent_id="triage",
        name="Message Triage",
        description=(
            "Decides whether each message in a batch is worth the "
            "user's attention. Filters promos, automated alerts, and "
            "ack-only chatter before the expensive downstream steps."
        ),
        category="classifier",
        parent_agent="brain",
        tier=Tier.BACKGROUND,
        max_sensitivity_tier=3,
        editable=False,
        default_config=default,
        available_tools=(),
        available_skills=(),
        output_schema="TriageBatch",
        pattern="single",
        factory=TriageAgent,
        tags=("classifier", "batch"),
    ))


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "TriageAgent",
    "TriageDeps",
    "TriageMessage",
    "register_triage_agent",
]
