"""Pydantic AI task-curator orchestrator.

Routes user chat requests about tasks, goals, schedules, and habits
to the right sub-agent. The Brain delegates to this orchestrator
when a turn is about "what should I do / what are my goals / replan
my day / give me one habit per goal".

The heavy lifting (CRUD, dedup, completion side-effects) lives in
:class:`src.agents.tasks.curator.TaskCurator`. This module is only
the LLM-facing surface: prompt + sub-agent delegations.

sensitivity_tier: 2
"""

from __future__ import annotations

import logging

from src.agents.core.agent_base import SBOrchestrator
from src.agents.core.output_types import BrainResponse
from src.agents.core.scheduler import Tier

logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = """\
You are the user's task curator. You don't answer questions in prose \
unless asked — instead you call the right sub-agent for the job:

- ``delegate_goal_extractor`` — mine goals from the user's sources \
("what are my work goals right now?", "find any new goals from this \
week").
- ``delegate_task_proposer`` — propose tasks from a batch of messages \
or under a named topic/goal ("create tasks for the clinic hiring \
project", "extract tasks from last week's messages").
- ``delegate_task_completion`` — check whether new evidence closes any \
open tasks ("did I finish anything from yesterday?").
- ``delegate_daily_scheduler`` — produce a plan for one day ("what's \
my day look like tomorrow?", "give me a focused schedule for today").
- ``delegate_habit_suggester`` — propose atomic habits anchored to \
the user's goals ("suggest one habit per goal", "what daily practices \
would help my work goals?").

After calling a sub-agent, return a BrainResponse whose ``answer`` \
narrates the result for the user in 2-4 sentences. Never invent \
goals, tasks, habits, or schedule slots — only refer to ones the \
sub-agents returned. If a sub-agent returns nothing useful, say so \
plainly.\
"""


class TaskCuratorAgent(SBOrchestrator[str, BrainResponse]):
    """Routes task/goal/schedule/habit requests to the right sub-agent.

    sensitivity_tier: 2
    """

    agent_id = "task_curator"
    output_type = BrainResponse
    tier = Tier.INTERACTIVE
    system_prompt = DEFAULT_SYSTEM_PROMPT
    subagents: tuple[str, ...] = (
        "goal_extractor",
        "task_proposer",
        "task_completion",
        "daily_scheduler",
        "habit_suggester",
    )


def register_task_curator_agent() -> None:
    """Register the task curator in the global registry.

    sensitivity_tier: 1
    """
    from src.agents.core.config_store import AgentConfig
    from src.agents.core.registry import (
        AgentDefinition,
        get_agent,
        register_agent,
    )

    if get_agent("task_curator") is not None:
        return

    default = AgentConfig(
        agent_id="task_curator",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        model_route="inherit",
        model_override=None,
        enabled_tools=(),
        enabled_skills=(),
        editable=False,
    )
    register_agent(AgentDefinition(
        agent_id="task_curator",
        name="Task Curator",
        description=(
            "Orchestrates the goal/task/habit/schedule sub-agents. "
            "Brain delegates here when a turn is about planning work."
        ),
        category="orchestrator",
        parent_agent="brain",
        tier=Tier.INTERACTIVE,
        max_sensitivity_tier=2,
        editable=False,
        default_config=default,
        available_tools=(),
        available_skills=(),
        output_schema="BrainResponse",
        pattern="orchestrator",
        factory=TaskCuratorAgent,
        tags=("orchestrator", "tasks"),
        subagents=TaskCuratorAgent.subagents,
    ))


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "TaskCuratorAgent",
    "register_task_curator_agent",
]
