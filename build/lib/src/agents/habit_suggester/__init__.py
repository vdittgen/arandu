"""Habit suggester (atomic-habits style).

sensitivity_tier: 1
"""

from __future__ import annotations

from src.agents.habit_suggester.agent import (
    DEFAULT_SYSTEM_PROMPT,
    HabitSuggesterAgent,
    HabitSuggesterDeps,
    register_habit_suggester_agent,
)

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "HabitSuggesterAgent",
    "HabitSuggesterDeps",
    "register_habit_suggester_agent",
]
