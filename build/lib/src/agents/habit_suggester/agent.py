"""Pydantic AI atomic-habits planner.

For each active goal, reads the goal itself (for the *why*) and the
topics rolled up under it (for situational context), and suggests
small recurring practices anchored to a specific ``goal_id``. The
prompt refuses to emit goal-less habits — that's the atomic-habits
coupling enforced at the schema layer (``_habits.goal_id NOT NULL``).

sensitivity_tier: 1
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from src.agents.core.agent_base import SBAgent
from src.agents.core.output_types import HabitBatch
from src.agents.core.scheduler import Tier

logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = """\
You suggest atomic habits for the user. Return a HabitBatch matching \
the schema.

Inputs:
- A list of ACTIVE goals (each with id, title, category, why).
- A list of TOPICS, each linked to a goal id, with the current \
situational context (e.g. for the goal "staff the clinic", a linked \
topic might be "hiring a psychologist").

Rules:
1. EVERY habit must be anchored to one goal from the supplied list \
via ``goal_id``. Never invent goal ids. If you can't tie a habit to \
a real goal, do not emit it.
2. Aim for 3-6 habits total, spread across the categories of the \
goals you receive.
3. Each habit should be small enough to do most days without ceremony \
("read 1 page", "30 squats", "1 résumé skim"). Atomic, not heroic.
4. Use the linked topics to make habits CONCRETE. Don't suggest "work \
on the clinic" — suggest "10 min/day reading hiring résumés" when \
that's the linked topic.
5. Cadence: prefer "daily"; use "specific_days" when the habit only \
makes sense some days; use "weekly" sparingly.
6. ``preferred_window`` reflects when the habit fits best ("morning" \
for energy/health, "evening" for reflection, "midday" for breaks).
7. ``why`` quotes the goal's why back at the user verbatim ("so I \
can finish my PhD"). Keep ≤ 120 chars.
8. ``reason`` ≤ 12 words — explain why this habit moves the goal.\
"""


@dataclass(frozen=True)
class HabitSuggesterDeps:
    """Typed input bundle for :class:`HabitSuggesterAgent`.

    sensitivity_tier: 2
    """

    goals: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    linked_topics: tuple[dict[str, Any], ...] = field(default_factory=tuple)


class HabitSuggesterAgent(SBAgent[HabitSuggesterDeps | str, HabitBatch]):
    """Suggest goal-anchored atomic habits.

    sensitivity_tier: 1
    """

    agent_id = "habit_suggester"
    output_type = HabitBatch
    tier = Tier.BACKGROUND
    system_prompt = DEFAULT_SYSTEM_PROMPT

    def build_prompt(self, deps: HabitSuggesterDeps | str) -> str:
        """sensitivity_tier: 2"""
        if isinstance(deps, str):
            return deps
        return (
            "Active goals (JSON):\n"
            f"{json.dumps(list(deps.goals))}\n\n"
            "Topics linked to those goals (JSON):\n"
            f"{json.dumps(list(deps.linked_topics))}\n\n"
            "Return 3-6 HabitDraft entries, each anchored to a real "
            "goal_id from the list above."
        )

    def suggest(
        self,
        *,
        goals: list[dict[str, Any]],
        linked_topics: list[dict[str, Any]] | None = None,
    ) -> HabitBatch | None:
        """sensitivity_tier: 2"""
        if not goals:
            return HabitBatch(habits=[])
        deps = HabitSuggesterDeps(
            goals=tuple(goals),
            linked_topics=tuple(linked_topics or []),
        )
        record = self.run(deps)
        if record.output is None or record.error is not None:
            return None
        # Defensive: drop any habit whose goal_id isn't in the input.
        valid_goal_ids = {str(g.get("id", "")) for g in goals}
        filtered = [
            h for h in record.output.habits
            if h.goal_id in valid_goal_ids
        ]
        return HabitBatch(habits=filtered)


def register_habit_suggester_agent() -> None:
    """sensitivity_tier: 1"""
    from src.agents.core.config_store import AgentConfig
    from src.agents.core.registry import (
        AgentDefinition,
        get_agent,
        register_agent,
    )

    if get_agent("habit_suggester") is not None:
        return

    default = AgentConfig(
        agent_id="habit_suggester",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        model_route="inherit",
        model_override=None,
        enabled_tools=(),
        enabled_skills=(),
        editable=False,
    )
    register_agent(AgentDefinition(
        agent_id="habit_suggester",
        name="Habit Suggester",
        description=(
            "Suggests atomic habits, each anchored to one of the "
            "user's active goals. Reads goals (for the why) plus "
            "topics linked to those goals (for situational context)."
        ),
        category="planner",
        parent_agent="brain",
        tier=Tier.BACKGROUND,
        max_sensitivity_tier=2,
        editable=False,
        default_config=default,
        available_tools=(),
        available_skills=(),
        output_schema="HabitBatch",
        pattern="single",
        factory=HabitSuggesterAgent,
        tags=("planner", "habits"),
    ))


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "HabitSuggesterAgent",
    "HabitSuggesterDeps",
    "register_habit_suggester_agent",
]
