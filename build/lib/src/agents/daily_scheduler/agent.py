"""Pydantic AI daily scheduler.

Packs open tasks + active habits into a single-day plan around the
user's fixed calendar events. Balances time across the three
categories and weights high-importance goals.

sensitivity_tier: 2
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from src.agents.core.agent_base import SBAgent
from src.agents.core.output_types import DailySchedule
from src.agents.core.scheduler import Tier

logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = """\
You build the user's plan for ONE day. Return a DailySchedule \
matching the schema.

Inputs:
- ``schedule_date`` the date you're planning (use it verbatim in the \
output).
- A list of FIXED calendar events you cannot move.
- A list of OPEN tasks with importance, due dates, and goal anchors.
- A list of ACTIVE habits, each anchored to a goal.
- ``working_hours`` the user's available block (defaults 08:00-19:00 \
local).
- ``category_mix`` the user's target balance across personal/life/work \
(defaults work:60% / personal:25% / life:15% on a weekday).

Rules:
1. Place each calendar event as a fixed ``event`` slot (do not move \
or omit them).
2. Fit task slots into the gaps. Sort by: overdue first → due today \
→ explicit deadline → high importance.
3. Place habit slots in their ``preferred_window`` (morning / midday \
/ evening) when free.
4. Slot length defaults: task 30-60 min, habit 10-25 min. Use \
shorter slots when the day is busy.
5. Try to keep the placed slots' minute-totals close to ``category_mix``.
6. Anything you can't fit goes to ``unscheduled_overflow`` by task id.
7. ``rationale`` ≤ 220 chars — explain the day's shape in plain English \
("Morning is for deep work on the proposal; afternoon is meetings; \
30 min reserved for clinic hiring research").

For each ScheduleSlot:
- ``kind`` one of "event" | "task" | "habit"
- ``ref_id`` the calendar event id, task id, or habit id
- ``title`` short title
- ``start`` / ``end`` ISO datetimes
- ``why`` one short reason for placement (≤ 80 chars)
- ``category`` and ``goal_id`` when known (events may have neither)

Fill ``category_balance`` as ``{"personal": minutes, "life": minutes, \
"work": minutes}``.\
"""


@dataclass(frozen=True)
class DailySchedulerDeps:
    """Typed input bundle for :class:`DailySchedulerAgent`.

    sensitivity_tier: 2
    """

    schedule_date: str
    events: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    tasks: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    habits: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    goals: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    working_hours: str = "08:00-19:00"
    category_mix: dict[str, int] = field(
        default_factory=lambda: {
            "work": 60, "personal": 25, "life": 15,
        },
    )


class DailySchedulerAgent(
    SBAgent[DailySchedulerDeps | str, DailySchedule],
):
    """Pack the user's day around fixed events.

    sensitivity_tier: 2
    """

    agent_id = "daily_scheduler"
    output_type = DailySchedule
    tier = Tier.BACKGROUND
    system_prompt = DEFAULT_SYSTEM_PROMPT

    def build_prompt(self, deps: DailySchedulerDeps | str) -> str:
        """sensitivity_tier: 2"""
        if isinstance(deps, str):
            return deps
        return (
            f"schedule_date: {deps.schedule_date}\n"
            f"working_hours: {deps.working_hours}\n"
            f"category_mix: {json.dumps(deps.category_mix)}\n\n"
            "Fixed events (JSON):\n"
            f"{json.dumps(list(deps.events))}\n\n"
            "Open tasks (JSON):\n"
            f"{json.dumps(list(deps.tasks))}\n\n"
            "Active habits (JSON):\n"
            f"{json.dumps(list(deps.habits))}\n\n"
            "Active goals (JSON, for context):\n"
            f"{json.dumps(list(deps.goals))}\n\n"
            "Return a DailySchedule for the requested date."
        )

    def plan(
        self,
        *,
        schedule_date: str,
        events: list[dict[str, Any]] | None = None,
        tasks: list[dict[str, Any]] | None = None,
        habits: list[dict[str, Any]] | None = None,
        goals: list[dict[str, Any]] | None = None,
        working_hours: str = "08:00-19:00",
        category_mix: dict[str, int] | None = None,
    ) -> DailySchedule | None:
        """sensitivity_tier: 2"""
        deps = DailySchedulerDeps(
            schedule_date=schedule_date,
            events=tuple(events or []),
            tasks=tuple(tasks or []),
            habits=tuple(habits or []),
            goals=tuple(goals or []),
            working_hours=working_hours,
            category_mix=category_mix or {
                "work": 60, "personal": 25, "life": 15,
            },
        )
        record = self.run(deps)
        if record.output is None or record.error is not None:
            return None
        return record.output


def register_daily_scheduler_agent() -> None:
    """sensitivity_tier: 1"""
    from src.agents.core.config_store import AgentConfig
    from src.agents.core.registry import (
        AgentDefinition,
        get_agent,
        register_agent,
    )

    if get_agent("daily_scheduler") is not None:
        return

    default = AgentConfig(
        agent_id="daily_scheduler",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        model_route="inherit",
        model_override=None,
        enabled_tools=(),
        enabled_skills=(),
        editable=False,
    )
    register_agent(AgentDefinition(
        agent_id="daily_scheduler",
        name="Daily Scheduler",
        description=(
            "Packs open tasks + active habits into a single-day plan "
            "around fixed calendar events. Balances categories and "
            "goal importance."
        ),
        category="planner",
        parent_agent="brain",
        tier=Tier.BACKGROUND,
        max_sensitivity_tier=2,
        editable=False,
        default_config=default,
        available_tools=(),
        available_skills=(),
        output_schema="DailySchedule",
        pattern="single",
        factory=DailySchedulerAgent,
        tags=("planner", "daily"),
    ))


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "DailySchedulerAgent",
    "DailySchedulerDeps",
    "register_daily_scheduler_agent",
]
