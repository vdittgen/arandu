"""Frozen dataclasses for the tasks/goals/habits surface.

Mirror the Rust DTOs in ``src-tauri/src/commands/types.rs`` exactly —
the CLI handlers ``asdict()`` these and the Tauri layer deserialises
them into the Rust structs. Keep field names and types in lock-step
when editing either side (see "Type sources of truth" in CLAUDE.md).

sensitivity_tier: 2
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Default projects seeded on first curator boot — one per category so
# every manual task lands somewhere sensible without forcing the user
# to pick a project up front.
DEFAULT_PROJECTS: tuple[tuple[str, str], ...] = (
    ("Personal", "personal"),
    ("Life", "life"),
    ("Work", "work"),
)


@dataclass(frozen=True)
class Goal:
    """A user-level goal aggregated by the Brain or entered manually.

    sensitivity_tier: 2
    """

    id: str
    title: str
    description: str = ""
    category: str = "personal"  # personal | life | work
    horizon: str = "medium"     # short | medium | long
    target_date: str | None = None
    status: str = "active"  # active | paused | achieved | abandoned | archived
    importance: int = 5
    why: str = ""
    source: str = "user"        # user | brain
    source_ref: str | None = None
    created_at: str = ""
    updated_at: str = ""
    last_confirmed_at: str | None = None
    sensitivity_tier: int = 2


@dataclass(frozen=True)
class Project:
    """A grouping of tasks. Optionally rolls up under a goal or topic.

    sensitivity_tier: 2
    """

    id: str
    name: str
    category: str = "personal"
    topic_id: str | None = None
    goal_id: str | None = None
    status: str = "active"      # active | archived
    color: str | None = None
    created_at: str = ""
    updated_at: str = ""
    sensitivity_tier: int = 2


@dataclass(frozen=True)
class Task:
    """A single tracked unit of work.

    Subtask iff ``parent_task_id`` is not None.

    sensitivity_tier: 2
    """

    id: str
    title: str
    project_id: str | None = None
    parent_task_id: str | None = None
    goal_id: str | None = None
    notes: str = ""
    status: str = "todo"        # todo | in_progress | done | cancelled
    importance: int = 5
    due_at: str | None = None
    scheduled_for: str | None = None
    source: str = "user"        # user | brain | message | event
    source_ref: str | None = None
    completion_note: str | None = None
    completion_evidence_id: str | None = None
    completed_at: str | None = None
    created_at: str = ""
    updated_at: str = ""
    sensitivity_tier: int = 2


@dataclass(frozen=True)
class Habit:
    """A recurring practice anchored to a goal (atomic-habits style).

    sensitivity_tier: 1
    """

    id: str
    title: str
    goal_id: str                # required — no goal-less habits
    cadence: str = "daily"      # daily | weekly | specific_days
    days_of_week: tuple[str, ...] = ()
    preferred_window: str = "any"
    why: str = ""
    source: str = "user"
    status: str = "active"      # active | paused
    created_at: str = ""
    sensitivity_tier: int = 1


@dataclass(frozen=True)
class ScheduleSlotRecord:
    """One slot inside a persisted day plan.

    sensitivity_tier: 2
    """

    kind: str                   # event | task | habit
    ref_id: str
    title: str
    start: str
    end: str
    why: str = ""
    category: str | None = None
    goal_id: str | None = None


@dataclass(frozen=True)
class DailyScheduleRecord:
    """Persisted daily plan returned by the curator.

    sensitivity_tier: 2
    """

    schedule_date: str
    slots: list[ScheduleSlotRecord] = field(default_factory=list)
    unscheduled_overflow: list[str] = field(default_factory=list)
    rationale: str = ""
    category_balance: dict[str, int] = field(default_factory=dict)
    generated_at: str = ""
    sensitivity_tier: int = 2
