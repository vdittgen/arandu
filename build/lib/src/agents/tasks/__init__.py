"""Tasks / Goals / Habits / Schedule curator surface.

Public entry points are :class:`TaskCurator` (the orchestrator that
owns CRUD + agent delegation) and the data dataclasses re-exported
for the CLI handlers.

sensitivity_tier: 2
"""

from __future__ import annotations

from src.agents.tasks.curator import TaskCurator
from src.agents.tasks.models import (
    DEFAULT_PROJECTS,
    Goal,
    Habit,
    Project,
    ScheduleSlotRecord,
    Task,
)

__all__ = [
    "DEFAULT_PROJECTS",
    "Goal",
    "Habit",
    "Project",
    "ScheduleSlotRecord",
    "Task",
    "TaskCurator",
]
