"""Pydantic AI goal extractor.

Mines user-level goals from a mixed batch of evidence: recent
messages, notes, learned facts, and chat history. Each goal has a
category (personal | life | work), a horizon, and a *why* — distinct
from topics (per-contact, situational) and tasks (single units of
work).

Used both from the proactive 2-hour cycle (full re-mine) and from
chat via ``task_curator`` ("mine my goals from the last month").

sensitivity_tier: 2
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from src.agents.core.agent_base import SBAgent
from src.agents.core.output_types import GoalBatch
from src.agents.core.scheduler import Tier

logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = """\
You mine the user's GOALS from a mixed batch of personal evidence \
(recent messages, notes, learned facts, chat history). Return a \
GoalBatch matching the schema.

A goal is the user's own ongoing commitment with a horizon and a \
*why*. It is NOT:
- a single task ("send the proposal Friday" → task, not goal)
- a per-contact topic ("hiring a psychologist for the clinic" → \
topic, not goal)
- a passing wish or a vague platitude

Good goals (examples):
- "Staff the clinic by end of Q3" (work, short)
- "Run a half marathon this year" (personal, medium)
- "Stabilise dad's treatment plan" (life, medium)
- "Ship v1 of the Arandu extension marketplace" (work, medium)

For each goal return a GoalDraft with:
- ``title`` short imperative phrasing (≤ 80 chars)
- ``description`` one sentence of context (≤ 200 chars)
- ``category`` one of "personal" | "life" | "work"
  * personal = self-care, hobbies, identity, growth
  * life = family, health, relationships, big life chapters
  * work = career, projects, income
- ``horizon`` one of "short" (≤ 3 months) | "medium" (3-12 months) | \
"long" (12+ months)
- ``target_date`` ISO date if the evidence implies one, else null
- ``importance`` 1-10 (10 = mission-critical)
- ``why`` one short sentence the user could read back and recognise \
("so I can finish my PhD"). NEVER invent a *why* — leave it as an \
empty string if the evidence doesn't support one.
- ``source_kind`` one of "message" | "note" | "fact" | "chat" | \
"topic"
- ``source_ref`` the evidence id you read it from (message id, note \
id, fact id, chat session id, or topic id)
- ``linked_topic_hint`` a topic name from the supplied list that \
this goal subsumes, or null. Pick at most one.

The supplied topics are evidence too: when a topic clearly reflects \
an ongoing commitment of the user's own (high importance, recurring, \
user-driven — not just something a contact mentioned), you may mine \
a goal from it directly with ``source_kind`` "topic" and the topic's \
id as ``source_ref`` (also set ``linked_topic_hint`` to its name).

Rules:
- Emit at most 8 goals. Quality over quantity.
- If you see the same goal twice in different evidence, emit it once \
and pick the strongest source.
- If the evidence is too thin to support any goal, return an empty \
list. Do not pad.\
"""

# Soft per-section budgets so the prompt stays bounded without ever
# slicing in the middle of a JSON record. The goal extractor only
# needs the strongest recent signal — packing 200 messages is wasteful
# and historically caused the body to be char-truncated mid-object,
# which corrupted the JSON and made the LLM bail with an empty batch.
_MAX_MESSAGES_CHARS = 8000
_MAX_NOTES_CHARS = 3000
_MAX_FACTS_CHARS = 2000
_MAX_TOPICS_CHARS = 2000
_MAX_CHAT_CHARS = 3000


def _pack_records(rows: list[dict[str, Any]], max_chars: int) -> str:
    """Serialize ``rows`` as a JSON array, truncating by record count.

    Never splits an individual record. Records are kept in input
    order; once the next one would push the array past ``max_chars``,
    iteration stops. Returns a syntactically valid JSON array.
    """
    if not rows:
        return "[]"
    kept: list[dict[str, Any]] = []
    for row in rows:
        candidate = json.dumps(kept + [row], ensure_ascii=False)
        if len(candidate) > max_chars and kept:
            break
        kept.append(row)
    return json.dumps(kept, ensure_ascii=False)


def _pack_chat(excerpts: list[str], max_chars: int) -> str:
    """Join chat excerpts within ``max_chars``, dropping oldest first."""
    if not excerpts:
        return ""
    out: list[str] = []
    total = 0
    for chunk in reversed(excerpts):
        added = len(chunk) + 5  # account for the "\n---\n" joiner
        if out and total + added > max_chars:
            break
        out.append(chunk)
        total += added
    return "\n---\n".join(reversed(out))


@dataclass(frozen=True)
class GoalExtractorDeps:
    """Typed input bundle for :class:`GoalExtractorAgent`.

    sensitivity_tier: 2
    """

    messages: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    notes: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    facts: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    chat_excerpts: tuple[str, ...] = field(default_factory=tuple)
    known_topics: tuple[dict[str, Any], ...] = field(default_factory=tuple)


class GoalExtractorAgent(SBAgent[GoalExtractorDeps | str, GoalBatch]):
    """Extract goals from mixed user evidence.

    sensitivity_tier: 2
    """

    agent_id = "goal_extractor"
    output_type = GoalBatch
    tier = Tier.PROACTIVE
    system_prompt = DEFAULT_SYSTEM_PROMPT

    def build_prompt(
        self, deps: GoalExtractorDeps | str,
    ) -> str:
        """sensitivity_tier: 2"""
        if isinstance(deps, str):
            return deps
        messages_json = _pack_records(
            list(deps.messages), _MAX_MESSAGES_CHARS,
        )
        notes_json = _pack_records(list(deps.notes), _MAX_NOTES_CHARS)
        facts_json = _pack_records(list(deps.facts), _MAX_FACTS_CHARS)
        topics_json = _pack_records(
            list(deps.known_topics), _MAX_TOPICS_CHARS,
        )
        chat_block = _pack_chat(list(deps.chat_excerpts), _MAX_CHAT_CHARS)
        return (
            "Recent messages (JSON):\n"
            f"{messages_json}\n\n"
            "Notes (JSON):\n"
            f"{notes_json}\n\n"
            "Learned facts (JSON):\n"
            f"{facts_json}\n\n"
            "Chat excerpts:\n"
            f"{chat_block}\n\n"
            "Known topics (JSON) — evidence and link targets:\n"
            f"{topics_json}\n\n"
            "Return up to 8 GoalDraft entries grounded in this evidence."
        )

    def extract(
        self,
        *,
        messages: list[dict[str, Any]] | None = None,
        notes: list[dict[str, Any]] | None = None,
        facts: list[dict[str, Any]] | None = None,
        chat_excerpts: list[str] | None = None,
        known_topics: list[dict[str, Any]] | None = None,
    ) -> GoalBatch | None:
        """Convenience entrypoint used by the curator.

        Returns None on LLM failure; an empty batch when the evidence
        legitimately yields nothing.

        sensitivity_tier: 2
        """
        if not any([messages, notes, facts, chat_excerpts, known_topics]):
            return GoalBatch(goals=[])
        deps = GoalExtractorDeps(
            messages=tuple(messages or []),
            notes=tuple(notes or []),
            facts=tuple(facts or []),
            chat_excerpts=tuple(chat_excerpts or []),
            known_topics=tuple(known_topics or []),
        )
        record = self.run(deps)
        if record.output is None or record.error is not None:
            return None
        # Defensive cap matching the system prompt.
        if len(record.output.goals) > 8:
            return GoalBatch(goals=record.output.goals[:8])
        return record.output


def register_goal_extractor_agent() -> None:
    """Register the goal extractor in the global agent registry.

    sensitivity_tier: 1
    """
    from src.agents.core.config_store import AgentConfig
    from src.agents.core.registry import (
        AgentDefinition,
        get_agent,
        register_agent,
    )

    if get_agent("goal_extractor") is not None:
        return

    default = AgentConfig(
        agent_id="goal_extractor",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        model_route="inherit",
        model_override=None,
        enabled_tools=(),
        enabled_skills=(),
        editable=False,
    )
    register_agent(AgentDefinition(
        agent_id="goal_extractor",
        name="Goal Extractor",
        description=(
            "Mines user-level goals (with horizon and why) from "
            "messages, notes, learned facts, and chat history. "
            "Populates the _goals aggregation table."
        ),
        category="extractor",
        parent_agent="brain",
        tier=Tier.PROACTIVE,
        max_sensitivity_tier=2,
        editable=False,
        default_config=default,
        available_tools=(),
        available_skills=(),
        output_schema="GoalBatch",
        pattern="single",
        factory=GoalExtractorAgent,
        tags=("extractor", "batch"),
    ))


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "GoalExtractorAgent",
    "GoalExtractorDeps",
    "register_goal_extractor_agent",
]
