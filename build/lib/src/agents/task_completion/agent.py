"""Pydantic AI task completion detector.

Given open tasks and a batch of new evidence (messages, events),
decides which tasks the evidence resolves and how confident the
verdict is. High-confidence verdicts (>= 0.7) close the task and
attach an evidence id; lower-confidence verdicts land in
``_task_completion_candidates`` for the user to review.

sensitivity_tier: 2
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from src.agents.core.agent_base import SBAgent
from src.agents.core.output_types import TaskCompletionBatch
from src.agents.core.scheduler import Tier

logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = """\
You decide which OPEN TASKS have been completed based on a batch of \
new evidence. Return a TaskCompletionBatch matching the schema.

For every task you believe is now done, emit one TaskCompletionDraft:
- ``task_id`` from the supplied open-tasks list
- ``evidence_message_id`` the single message id that proves it
- ``evidence_summary`` ≤ 140 chars naming what the evidence says
- ``confidence`` 0.0-1.0:
  * 0.9+ = the evidence is explicit and unambiguous ("done, here it is")
  * 0.7-0.9 = the evidence is strong but indirect (the user's message \
acknowledges shipping, the deliverable was discussed in past tense)
  * < 0.7 = there's a plausible signal but you're not sure — emit it \
anyway, the caller will queue it for user review.

Rules:
- Only emit verdicts for tasks IN THE SUPPLIED LIST.
- Never invent task ids.
- A single message can complete multiple tasks; emit one Draft per task.
- If no evidence resolves any open task, return an empty list.
- Don't double-count: if the same task is "completed" in two messages, \
pick the more explicit one.\
"""


@dataclass(frozen=True)
class TaskCompletionDeps:
    """Typed input bundle for :class:`TaskCompletionAgent`.

    sensitivity_tier: 2
    """

    open_tasks: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    evidence: tuple[dict[str, Any], ...] = field(default_factory=tuple)


class TaskCompletionAgent(
    SBAgent[TaskCompletionDeps | str, TaskCompletionBatch],
):
    """Detect task completions from new evidence.

    sensitivity_tier: 2
    """

    agent_id = "task_completion"
    output_type = TaskCompletionBatch
    tier = Tier.PROACTIVE
    system_prompt = DEFAULT_SYSTEM_PROMPT

    def build_prompt(self, deps: TaskCompletionDeps | str) -> str:
        """sensitivity_tier: 2"""
        if isinstance(deps, str):
            return deps
        return (
            "Open tasks (JSON):\n"
            f"{json.dumps(list(deps.open_tasks))}\n\n"
            "New evidence (JSON):\n"
            f"{json.dumps(list(deps.evidence))}\n\n"
            "Return TaskCompletionDraft entries — one per task you "
            "think is now done."
        )

    def detect(
        self,
        *,
        open_tasks: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
    ) -> TaskCompletionBatch | None:
        """sensitivity_tier: 2"""
        if not open_tasks or not evidence:
            return TaskCompletionBatch(completions=[])
        deps = TaskCompletionDeps(
            open_tasks=tuple(open_tasks),
            evidence=tuple(evidence),
        )
        record = self.run(deps)
        if record.output is None or record.error is not None:
            return None
        return record.output


def register_task_completion_agent() -> None:
    """sensitivity_tier: 1"""
    from src.agents.core.config_store import AgentConfig
    from src.agents.core.registry import (
        AgentDefinition,
        get_agent,
        register_agent,
    )

    if get_agent("task_completion") is not None:
        return

    default = AgentConfig(
        agent_id="task_completion",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        model_route="inherit",
        model_override=None,
        enabled_tools=(),
        enabled_skills=(),
        editable=False,
    )
    register_agent(AgentDefinition(
        agent_id="task_completion",
        name="Task Completion Detector",
        description=(
            "Reads open tasks + new evidence and decides which tasks "
            "the evidence resolves, with a confidence score."
        ),
        category="evaluator",
        parent_agent="brain",
        tier=Tier.PROACTIVE,
        max_sensitivity_tier=2,
        editable=False,
        default_config=default,
        available_tools=(),
        available_skills=(),
        output_schema="TaskCompletionBatch",
        pattern="single",
        factory=TaskCompletionAgent,
        tags=("evaluator", "batch"),
    ))


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "TaskCompletionAgent",
    "TaskCompletionDeps",
    "register_task_completion_agent",
]
