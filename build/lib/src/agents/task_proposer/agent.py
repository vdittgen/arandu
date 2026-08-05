"""Pydantic AI task proposer.

Reads a batch of recent messages plus active topics and goals, and
emits proposed *tasks* — explicit work the user must do. Critically,
it distinguishes tasks from pending replies:

- "Can you send me the deck by Friday?" → task (send the deck).
- "What did you think of the deck?" → pending reply, NOT a task.

When the task plausibly serves a known goal, ``parent_goal_hint`` is
populated so the curator can attach it via ``_tasks.goal_id``.

sensitivity_tier: 2
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from src.agents.core.agent_base import SBAgent
from src.agents.core.output_types import TaskProposalBatch
from src.agents.core.scheduler import Tier

logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = """\
You read a batch of recent messages and propose TASKS for the user. \
Return a TaskProposalBatch matching the schema.

A task is a concrete unit of work the user must do. It is NOT:
- a pending reply to a question (that's handled by another agent — \
skip "what do you think?", "did you see this?", etc.)
- a goal (the long-running commitment) — tasks may *serve* a goal \
but are not goals themselves
- automated/notification messages, group chatter, marketing

Good tasks (examples):
- "Send Maria the proposal draft" (from "can you send me the proposal \
by Friday?")
- "Book flight to São Paulo for the offsite" (from "we'll need flights \
for the May offsite")
- "Pick up dad's prescription" (from "the pharmacy called, your dad's \
meds are ready")

For every proposed task return a TaskProposalDraft with:
- ``title`` short imperative (≤ 80 chars). Start with a verb.
- ``notes`` optional ≤ 200 chars of context grounded in the message
- ``category`` one of "personal" | "life" | "work"
- ``importance`` 1-10 (base 5; raise to 8+ for explicit deadlines or \
high-impact asks)
- ``due_at`` ISO timestamp if the message implies one, else null
- ``source_message_ids`` ids of the messages this task is grounded in
- ``parent_topic_hint`` the topic from the supplied list this task \
relates to, or null
- ``parent_goal_hint`` the goal title from the supplied list this \
task plausibly serves, or null. Pick at most one.
- ``reason`` ≤ 12 words, grounded in the message — explain why this \
is a task

Rules:
- Skip anything that's purely a question/reply.
- Don't invent dates — only fill ``due_at`` when the message names one.
- Emit at most 6 tasks per batch.
- If nothing in the batch deserves a task, return an empty list.\
"""


@dataclass(frozen=True)
class TaskProposerDeps:
    """Typed input bundle for :class:`TaskProposerAgent`.

    sensitivity_tier: 2
    """

    messages: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    topics: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    goals: tuple[dict[str, Any], ...] = field(default_factory=tuple)


class TaskProposerAgent(
    SBAgent[TaskProposerDeps | str, TaskProposalBatch],
):
    """Propose tasks from a batch of messages.

    sensitivity_tier: 2
    """

    agent_id = "task_proposer"
    output_type = TaskProposalBatch
    tier = Tier.PROACTIVE
    system_prompt = DEFAULT_SYSTEM_PROMPT

    def build_prompt(self, deps: TaskProposerDeps | str) -> str:
        """sensitivity_tier: 2"""
        if isinstance(deps, str):
            return deps
        return (
            "Active topics (JSON):\n"
            f"{json.dumps(list(deps.topics))}\n\n"
            "Active goals (JSON):\n"
            f"{json.dumps(list(deps.goals))}\n\n"
            "Recent messages (JSON):\n"
            f"{json.dumps(list(deps.messages))}\n\n"
            "Return up to 6 TaskProposalDraft entries — tasks only, "
            "never replies."
        )

    def propose(
        self,
        *,
        messages: list[dict[str, Any]],
        topics: list[dict[str, Any]] | None = None,
        goals: list[dict[str, Any]] | None = None,
    ) -> TaskProposalBatch | None:
        """sensitivity_tier: 2"""
        if not messages:
            return TaskProposalBatch(tasks=[])
        deps = TaskProposerDeps(
            messages=tuple(messages),
            topics=tuple(topics or []),
            goals=tuple(goals or []),
        )
        record = self.run(deps)
        if record.output is None or record.error is not None:
            return None
        if len(record.output.tasks) > 6:
            return TaskProposalBatch(tasks=record.output.tasks[:6])
        return record.output


def register_task_proposer_agent() -> None:
    """sensitivity_tier: 1"""
    from src.agents.core.config_store import AgentConfig
    from src.agents.core.registry import (
        AgentDefinition,
        get_agent,
        register_agent,
    )

    if get_agent("task_proposer") is not None:
        return

    default = AgentConfig(
        agent_id="task_proposer",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        model_route="inherit",
        model_override=None,
        enabled_tools=(),
        enabled_skills=(),
        editable=False,
    )
    register_agent(AgentDefinition(
        agent_id="task_proposer",
        name="Task Proposer",
        description=(
            "Proposes tasks (real work to do) from messages and active "
            "topics/goals. Distinct from the pending-reply detector."
        ),
        category="evaluator",
        parent_agent="brain",
        tier=Tier.PROACTIVE,
        max_sensitivity_tier=2,
        editable=False,
        default_config=default,
        available_tools=(),
        available_skills=(),
        output_schema="TaskProposalBatch",
        pattern="single",
        factory=TaskProposerAgent,
        tags=("evaluator", "batch"),
    ))


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "TaskProposerAgent",
    "TaskProposerDeps",
    "register_task_proposer_agent",
]
