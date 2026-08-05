"""SQLite persistence for goals, projects, tasks, habits, schedules.

Agent-owned runtime tables (``_`` prefix), same convention as
``_pending_replies`` / ``_topics``. Created in this module — never in
``src/core/sqlite/migrations.py`` (that file is reserved for
connector-introduced raw tables).

Adds two additive columns to ``_topics``:
- ``category`` — set by the extended ``TopicExtractorAgent``.
- ``linked_goal_id`` — FK to ``_goals.id``, set by ``TaskCurator``
  when reconciling ``GoalDraft.linked_topic_hint`` against existing
  topics. The goal↔topic relationship is canonical here; ``_goals``
  has no back-link so there's no FK to keep in sync.

sensitivity_tier: 2
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import asdict
from typing import Any

from src.agents.tasks.models import (
    DEFAULT_PROJECTS,
    DailyScheduleRecord,
    Goal,
    Habit,
    Project,
    ScheduleSlotRecord,
    Task,
)
from src.core.db_helpers import (
    ensure_tables,
    get_table_columns,
    make_hash_id,
    utc_now_iso,
)

logger = logging.getLogger(__name__)


_DDL: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS _goals (
        id                  VARCHAR PRIMARY KEY,
        title               VARCHAR NOT NULL,
        description         TEXT DEFAULT '',
        category            VARCHAR NOT NULL DEFAULT 'personal',
        horizon             VARCHAR NOT NULL DEFAULT 'medium',
        target_date         TEXT,
        status              VARCHAR NOT NULL DEFAULT 'active',
        importance          INTEGER NOT NULL DEFAULT 5,
        why                 TEXT DEFAULT '',
        source              VARCHAR NOT NULL DEFAULT 'user',
        source_ref          TEXT,
        created_at          TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at          TEXT DEFAULT CURRENT_TIMESTAMP,
        last_confirmed_at   TEXT,
        sensitivity_tier    INTEGER DEFAULT 2
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_goals_active
    ON _goals (status, category, importance)
    """,
    """
    CREATE TABLE IF NOT EXISTS _projects (
        id                  VARCHAR PRIMARY KEY,
        name                VARCHAR NOT NULL,
        category            VARCHAR NOT NULL DEFAULT 'personal',
        topic_id            VARCHAR,
        goal_id             VARCHAR,
        status              VARCHAR NOT NULL DEFAULT 'active',
        color               VARCHAR,
        created_at          TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at          TEXT DEFAULT CURRENT_TIMESTAMP,
        sensitivity_tier    INTEGER DEFAULT 2
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_projects_active
    ON _projects (status, category)
    """,
    """
    CREATE TABLE IF NOT EXISTS _tasks (
        id                      VARCHAR PRIMARY KEY,
        title                   VARCHAR NOT NULL,
        project_id              VARCHAR,
        parent_task_id          VARCHAR,
        goal_id                 VARCHAR,
        notes                   TEXT DEFAULT '',
        status                  VARCHAR NOT NULL DEFAULT 'todo',
        importance              INTEGER NOT NULL DEFAULT 5,
        due_at                  TEXT,
        scheduled_for           TEXT,
        source                  VARCHAR NOT NULL DEFAULT 'user',
        source_ref              TEXT,
        completion_note         TEXT,
        completion_evidence_id  VARCHAR,
        completed_at            TEXT,
        created_at              TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at              TEXT DEFAULT CURRENT_TIMESTAMP,
        sensitivity_tier        INTEGER DEFAULT 2
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_tasks_active
    ON _tasks (status, due_at)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_tasks_project
    ON _tasks (project_id, status)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_tasks_goal
    ON _tasks (goal_id, status)
    """,
    """
    CREATE TABLE IF NOT EXISTS _habits (
        id                  VARCHAR PRIMARY KEY,
        title               VARCHAR NOT NULL,
        goal_id             VARCHAR NOT NULL,
        cadence             VARCHAR NOT NULL DEFAULT 'daily',
        days_of_week        TEXT DEFAULT '[]',
        preferred_window    VARCHAR NOT NULL DEFAULT 'any',
        why                 TEXT DEFAULT '',
        source              VARCHAR NOT NULL DEFAULT 'user',
        status              VARCHAR NOT NULL DEFAULT 'active',
        created_at          TEXT DEFAULT CURRENT_TIMESTAMP,
        sensitivity_tier    INTEGER DEFAULT 1
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_habits_active
    ON _habits (status, goal_id)
    """,
    """
    CREATE TABLE IF NOT EXISTS _schedule_suggestions (
        schedule_date       TEXT PRIMARY KEY,
        items_json          TEXT NOT NULL,
        rationale           TEXT DEFAULT '',
        category_balance    TEXT DEFAULT '{}',
        unscheduled_json    TEXT DEFAULT '[]',
        generated_at        TEXT DEFAULT CURRENT_TIMESTAMP,
        sensitivity_tier    INTEGER DEFAULT 2
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS _task_completion_candidates (
        id                  VARCHAR PRIMARY KEY,
        task_id             VARCHAR NOT NULL,
        evidence_message_id VARCHAR NOT NULL,
        evidence_summary    TEXT NOT NULL,
        confidence          REAL NOT NULL,
        status              VARCHAR NOT NULL DEFAULT 'pending',
        detected_at         TEXT DEFAULT CURRENT_TIMESTAMP,
        sensitivity_tier    INTEGER DEFAULT 2
    )
    """,
]


# Additive column additions on the existing ``_topics`` table.
_TOPICS_COLUMN_ADDITIONS: tuple[tuple[str, str], ...] = (
    ("category", "VARCHAR DEFAULT NULL"),
    ("linked_goal_id", "VARCHAR DEFAULT NULL"),
)


def ensure_schema(db: Any) -> None:
    """Create the goals/projects/tasks/habits tables and extend _topics.

    Idempotent. Safe to call on every curator instantiation.

    sensitivity_tier: 1
    """
    ensure_tables(db, _DDL)

    # _topics may not exist yet on a brand-new install (it's created
    # lazily by message_eval/persistence.py on first message arrival).
    # Skip the column additions in that case — the next message_eval
    # boot will materialise it without the new columns, and the next
    # curator call will retry the ALTER and succeed.
    existing = get_table_columns(db, "_topics")
    if not existing:
        return
    for column, sql_type in _TOPICS_COLUMN_ADDITIONS:
        if column in existing:
            continue
        try:
            db.execute(
                f"ALTER TABLE _topics ADD COLUMN {column} {sql_type}",
            )
        except Exception:  # noqa: BLE001
            logger.debug(
                "ALTER TABLE _topics ADD COLUMN %s failed",
                column,
                exc_info=True,
            )


def seed_default_projects(db: Any) -> None:
    """Seed the three default projects on first boot.

    Idempotent — looks up by ``id`` (deterministic hash) and skips
    rows that already exist.

    sensitivity_tier: 1
    """
    now = utc_now_iso()
    for name, category in DEFAULT_PROJECTS:
        pid = make_hash_id("project", "default", name.lower())
        try:
            db.execute(
                """INSERT OR IGNORE INTO _projects
                   (id, name, category, status, created_at, updated_at,
                    sensitivity_tier)
                   VALUES (?, ?, ?, 'active', ?, ?, 2)""",
                [pid, name, category, now, now],
            )
        except Exception:  # noqa: BLE001
            logger.debug(
                "Default project seed failed for %s",
                name,
                exc_info=True,
            )


# ---------------------------------------------------------------------------
# Row → dataclass mappers
# ---------------------------------------------------------------------------


def _goal_from_row(row: dict[str, Any]) -> Goal:
    """sensitivity_tier: 2"""
    return Goal(
        id=str(row["id"]),
        title=str(row.get("title", "")),
        description=str(row.get("description") or ""),
        category=str(row.get("category", "personal")),
        horizon=str(row.get("horizon", "medium")),
        target_date=row.get("target_date"),
        status=str(row.get("status", "active")),
        importance=int(row.get("importance", 5)),
        why=str(row.get("why") or ""),
        source=str(row.get("source", "user")),
        source_ref=row.get("source_ref"),
        created_at=str(row.get("created_at") or ""),
        updated_at=str(row.get("updated_at") or ""),
        last_confirmed_at=row.get("last_confirmed_at"),
        sensitivity_tier=int(row.get("sensitivity_tier", 2)),
    )


def _project_from_row(row: dict[str, Any]) -> Project:
    """sensitivity_tier: 2"""
    return Project(
        id=str(row["id"]),
        name=str(row.get("name", "")),
        category=str(row.get("category", "personal")),
        topic_id=row.get("topic_id"),
        goal_id=row.get("goal_id"),
        status=str(row.get("status", "active")),
        color=row.get("color"),
        created_at=str(row.get("created_at") or ""),
        updated_at=str(row.get("updated_at") or ""),
        sensitivity_tier=int(row.get("sensitivity_tier", 2)),
    )


def _task_from_row(row: dict[str, Any]) -> Task:
    """sensitivity_tier: 2"""
    return Task(
        id=str(row["id"]),
        title=str(row.get("title", "")),
        project_id=row.get("project_id"),
        parent_task_id=row.get("parent_task_id"),
        goal_id=row.get("goal_id"),
        notes=str(row.get("notes") or ""),
        status=str(row.get("status", "todo")),
        importance=int(row.get("importance", 5)),
        due_at=row.get("due_at"),
        scheduled_for=row.get("scheduled_for"),
        source=str(row.get("source", "user")),
        source_ref=row.get("source_ref"),
        completion_note=row.get("completion_note"),
        completion_evidence_id=row.get("completion_evidence_id"),
        completed_at=row.get("completed_at"),
        created_at=str(row.get("created_at") or ""),
        updated_at=str(row.get("updated_at") or ""),
        sensitivity_tier=int(row.get("sensitivity_tier", 2)),
    )


def _habit_from_row(row: dict[str, Any]) -> Habit:
    """sensitivity_tier: 1"""
    raw_days = row.get("days_of_week") or "[]"
    try:
        days = tuple(json.loads(raw_days))
    except (json.JSONDecodeError, TypeError):
        days = ()
    return Habit(
        id=str(row["id"]),
        title=str(row.get("title", "")),
        goal_id=str(row.get("goal_id", "")),
        cadence=str(row.get("cadence", "daily")),
        days_of_week=tuple(str(d) for d in days),
        preferred_window=str(row.get("preferred_window", "any")),
        why=str(row.get("why") or ""),
        source=str(row.get("source", "user")),
        status=str(row.get("status", "active")),
        created_at=str(row.get("created_at") or ""),
        sensitivity_tier=int(row.get("sensitivity_tier", 1)),
    )


def _schedule_from_row(row: dict[str, Any]) -> DailyScheduleRecord:
    """sensitivity_tier: 2"""
    items_raw = row.get("items_json") or "[]"
    overflow_raw = row.get("unscheduled_json") or "[]"
    balance_raw = row.get("category_balance") or "{}"
    try:
        items = json.loads(items_raw)
    except (json.JSONDecodeError, TypeError):
        items = []
    try:
        overflow = json.loads(overflow_raw)
    except (json.JSONDecodeError, TypeError):
        overflow = []
    try:
        balance = json.loads(balance_raw)
    except (json.JSONDecodeError, TypeError):
        balance = {}
    slots = []
    for entry in items if isinstance(items, list) else []:
        if not isinstance(entry, dict):
            continue
        slots.append(ScheduleSlotRecord(
            kind=str(entry.get("kind", "task")),
            ref_id=str(entry.get("ref_id", "")),
            title=str(entry.get("title", "")),
            start=str(entry.get("start", "")),
            end=str(entry.get("end", "")),
            why=str(entry.get("why") or ""),
            category=entry.get("category"),
            goal_id=entry.get("goal_id"),
        ))
    overflow_list = overflow if isinstance(overflow, list) else []
    return DailyScheduleRecord(
        schedule_date=str(row.get("schedule_date", "")),
        slots=slots,
        unscheduled_overflow=[str(x) for x in overflow_list],
        rationale=str(row.get("rationale") or ""),
        category_balance={
            str(k): int(v) for k, v in (
                balance.items() if isinstance(balance, dict) else ()
            )
        },
        generated_at=str(row.get("generated_at") or ""),
        sensitivity_tier=int(row.get("sensitivity_tier", 2)),
    )


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------


def list_goals(
    db: Any,
    *,
    status: str | None = "active",
    category: str | None = None,
) -> list[Goal]:
    """List goals, optionally filtered by status and category.

    sensitivity_tier: 2
    """
    where: list[str] = []
    params: list[Any] = []
    if status:
        where.append("status = ?")
        params.append(status)
    if category:
        where.append("category = ?")
        params.append(category)
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    rows = db.query(
        f"SELECT * FROM _goals {clause} "  # noqa: S608
        "ORDER BY importance DESC, created_at DESC",
        params,
    )
    return [_goal_from_row(r) for r in rows]


def list_projects(
    db: Any,
    *,
    status: str | None = "active",
    category: str | None = None,
) -> list[Project]:
    """sensitivity_tier: 2"""
    where: list[str] = []
    params: list[Any] = []
    if status:
        where.append("status = ?")
        params.append(status)
    if category:
        where.append("category = ?")
        params.append(category)
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    rows = db.query(
        f"SELECT * FROM _projects {clause} "  # noqa: S608
        "ORDER BY category, name",
        params,
    )
    return [_project_from_row(r) for r in rows]


def list_tasks(
    db: Any,
    *,
    status: str | None = None,
    project_id: str | None = None,
    goal_id: str | None = None,
    parent_task_id: str | None = None,
    limit: int = 500,
) -> list[Task]:
    """sensitivity_tier: 2"""
    where: list[str] = []
    params: list[Any] = []
    if status:
        where.append("status = ?")
        params.append(status)
    if project_id:
        where.append("project_id = ?")
        params.append(project_id)
    if goal_id:
        where.append("goal_id = ?")
        params.append(goal_id)
    if parent_task_id is not None:
        where.append("parent_task_id = ?")
        params.append(parent_task_id)
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    rows = db.query(
        f"SELECT * FROM _tasks {clause} "  # noqa: S608
        "ORDER BY "
        "CASE status WHEN 'todo' THEN 0 WHEN 'in_progress' THEN 1 "
        "WHEN 'done' THEN 2 ELSE 3 END, "
        "due_at IS NULL, due_at ASC, importance DESC "
        f"LIMIT {int(limit)}",
        params,
    )
    return [_task_from_row(r) for r in rows]


def list_habits(
    db: Any,
    *,
    status: str | None = "active",
    goal_id: str | None = None,
) -> list[Habit]:
    """sensitivity_tier: 1"""
    where: list[str] = []
    params: list[Any] = []
    if status:
        where.append("status = ?")
        params.append(status)
    if goal_id:
        where.append("goal_id = ?")
        params.append(goal_id)
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    rows = db.query(
        f"SELECT * FROM _habits {clause} "  # noqa: S608
        "ORDER BY goal_id, title",
        params,
    )
    return [_habit_from_row(r) for r in rows]


def get_goal(db: Any, goal_id: str) -> Goal | None:
    """sensitivity_tier: 2"""
    rows = db.query("SELECT * FROM _goals WHERE id = ?", [goal_id])
    return _goal_from_row(rows[0]) if rows else None


def get_task(db: Any, task_id: str) -> Task | None:
    """sensitivity_tier: 2"""
    rows = db.query("SELECT * FROM _tasks WHERE id = ?", [task_id])
    return _task_from_row(rows[0]) if rows else None


def get_project(db: Any, project_id: str) -> Project | None:
    """sensitivity_tier: 2"""
    rows = db.query(
        "SELECT * FROM _projects WHERE id = ?", [project_id],
    )
    return _project_from_row(rows[0]) if rows else None


def get_daily_schedule(
    db: Any, schedule_date: str,
) -> DailyScheduleRecord | None:
    """sensitivity_tier: 2"""
    rows = db.query(
        "SELECT * FROM _schedule_suggestions WHERE schedule_date = ?",
        [schedule_date],
    )
    return _schedule_from_row(rows[0]) if rows else None


def list_topics_for_goal(
    db: Any, goal_id: str,
) -> list[dict[str, Any]]:
    """List topics rolled up under a goal via ``_topics.linked_goal_id``.

    Used by the Goals page and by ``habit_suggester`` (which only
    reads topics tagged to a goal — never free-floating ones).

    sensitivity_tier: 2
    """
    try:
        return db.query(
            "SELECT id, contact_name, topic, description, "
            "importance, status, category "
            "FROM _topics WHERE linked_goal_id = ? "
            "ORDER BY importance DESC",
            [goal_id],
        )
    except Exception:  # noqa: BLE001
        return []


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------


def insert_goal(db: Any, goal: Goal) -> None:
    """Insert a goal row. Idempotent on ``id``.

    sensitivity_tier: 2
    """
    db.execute(
        """INSERT OR REPLACE INTO _goals
           (id, title, description, category, horizon, target_date,
            status, importance, why, source, source_ref,
            created_at, updated_at, last_confirmed_at, sensitivity_tier)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            goal.id, goal.title, goal.description, goal.category,
            goal.horizon, goal.target_date, goal.status, goal.importance,
            goal.why, goal.source, goal.source_ref,
            goal.created_at or utc_now_iso(),
            goal.updated_at or utc_now_iso(),
            goal.last_confirmed_at, goal.sensitivity_tier,
        ],
    )


def update_goal_fields(db: Any, goal_id: str, **fields: Any) -> None:
    """Partial update of mutable goal fields.

    sensitivity_tier: 2
    """
    allowed = {
        "title", "description", "category", "horizon", "target_date",
        "status", "importance", "why", "last_confirmed_at",
    }
    keys = [k for k in fields if k in allowed]
    if not keys:
        return
    sets = ", ".join(f"{k} = ?" for k in keys)
    params = [fields[k] for k in keys] + [utc_now_iso(), goal_id]
    db.execute(
        f"UPDATE _goals SET {sets}, updated_at = ? WHERE id = ?",  # noqa: S608
        params,
    )


def insert_project(db: Any, project: Project) -> None:
    """sensitivity_tier: 2"""
    db.execute(
        """INSERT OR REPLACE INTO _projects
           (id, name, category, topic_id, goal_id, status, color,
            created_at, updated_at, sensitivity_tier)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            project.id, project.name, project.category, project.topic_id,
            project.goal_id, project.status, project.color,
            project.created_at or utc_now_iso(),
            project.updated_at or utc_now_iso(),
            project.sensitivity_tier,
        ],
    )


def insert_task(db: Any, task: Task) -> None:
    """sensitivity_tier: 2"""
    db.execute(
        """INSERT OR REPLACE INTO _tasks
           (id, title, project_id, parent_task_id, goal_id, notes,
            status, importance, due_at, scheduled_for, source,
            source_ref, completion_note, completion_evidence_id,
            completed_at, created_at, updated_at, sensitivity_tier)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            task.id, task.title, task.project_id, task.parent_task_id,
            task.goal_id, task.notes, task.status, task.importance,
            task.due_at, task.scheduled_for, task.source, task.source_ref,
            task.completion_note, task.completion_evidence_id,
            task.completed_at,
            task.created_at or utc_now_iso(),
            task.updated_at or utc_now_iso(),
            task.sensitivity_tier,
        ],
    )


def update_task_fields(db: Any, task_id: str, **fields: Any) -> None:
    """Partial update of mutable task fields.

    sensitivity_tier: 2
    """
    allowed = {
        "title", "notes", "status", "importance", "due_at",
        "scheduled_for", "project_id", "goal_id", "parent_task_id",
        "completion_note", "completion_evidence_id", "completed_at",
    }
    keys = [k for k in fields if k in allowed]
    if not keys:
        return
    sets = ", ".join(f"{k} = ?" for k in keys)
    params = [fields[k] for k in keys] + [utc_now_iso(), task_id]
    db.execute(
        f"UPDATE _tasks SET {sets}, updated_at = ? WHERE id = ?",  # noqa: S608
        params,
    )


def delete_task(db: Any, task_id: str) -> None:
    """sensitivity_tier: 2"""
    db.execute("DELETE FROM _tasks WHERE id = ?", [task_id])


def insert_habit(db: Any, habit: Habit) -> None:
    """sensitivity_tier: 1"""
    db.execute(
        """INSERT OR REPLACE INTO _habits
           (id, title, goal_id, cadence, days_of_week, preferred_window,
            why, source, status, created_at, sensitivity_tier)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            habit.id, habit.title, habit.goal_id, habit.cadence,
            json.dumps(list(habit.days_of_week)),
            habit.preferred_window, habit.why, habit.source,
            habit.status,
            habit.created_at or utc_now_iso(),
            habit.sensitivity_tier,
        ],
    )


def update_habit_status(db: Any, habit_id: str, status: str) -> None:
    """sensitivity_tier: 1"""
    db.execute(
        "UPDATE _habits SET status = ? WHERE id = ?",
        [status, habit_id],
    )


def delete_habit(db: Any, habit_id: str) -> None:
    """sensitivity_tier: 1"""
    db.execute("DELETE FROM _habits WHERE id = ?", [habit_id])


def upsert_schedule(
    db: Any, record: DailyScheduleRecord,
) -> None:
    """sensitivity_tier: 2"""
    db.execute(
        """INSERT OR REPLACE INTO _schedule_suggestions
           (schedule_date, items_json, rationale, category_balance,
            unscheduled_json, generated_at, sensitivity_tier)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        [
            record.schedule_date,
            json.dumps([asdict(s) for s in record.slots]),
            record.rationale,
            json.dumps(record.category_balance),
            json.dumps(record.unscheduled_overflow),
            record.generated_at or utc_now_iso(),
            record.sensitivity_tier,
        ],
    )


def set_topic_linked_goal(
    db: Any, topic_id: str, goal_id: str | None,
) -> None:
    """Set or clear ``_topics.linked_goal_id``.

    sensitivity_tier: 1
    """
    try:
        db.execute(
            "UPDATE _topics SET linked_goal_id = ? WHERE id = ?",
            [goal_id, topic_id],
        )
    except Exception:  # noqa: BLE001
        logger.debug(
            "Topic linked_goal_id update failed for %s",
            topic_id,
            exc_info=True,
        )


def find_topic_id_by_name(
    db: Any, topic_hint: str,
) -> str | None:
    """Best-effort lookup of a ``_topics.id`` by name fragment.

    Used by the curator to resolve ``GoalDraft.linked_topic_hint`` into
    a canonical FK. Case-insensitive substring match — returns the
    highest-importance active topic that matches, or ``None``.

    sensitivity_tier: 2
    """
    hint = (topic_hint or "").strip().lower()
    if len(hint) < 3:
        return None
    try:
        rows = db.query(
            "SELECT id FROM _topics "
            "WHERE LOWER(topic) LIKE ? AND status = 'active' "
            "ORDER BY importance DESC LIMIT 1",
            [f"%{hint}%"],
        )
    except Exception:  # noqa: BLE001
        return None
    return str(rows[0]["id"]) if rows else None


def dedup_goal_id(
    db: Any, title: str, category: str,
) -> str | None:
    """Return an existing goal id matching ``(title, category)`` if any.

    Active goals only — closed goals don't block a fresh entry with
    the same title (e.g. "lose 5kg" repeated next year).

    sensitivity_tier: 2
    """
    if not title:
        return None
    try:
        rows = db.query(
            "SELECT id FROM _goals "
            "WHERE LOWER(title) = ? AND category = ? "
            "AND status = 'active' LIMIT 1",
            [title.strip().lower(), category],
        )
    except Exception:  # noqa: BLE001
        return None
    return str(rows[0]["id"]) if rows else None


def find_open_task_titles(
    db: Any, limit: int = 200,
) -> dict[str, str]:
    """Map ``lower(title) → task_id`` for dedup in the proposer.

    Only considers ``todo`` / ``in_progress`` tasks.

    sensitivity_tier: 2
    """
    try:
        rows = db.query(
            "SELECT id, title FROM _tasks "
            "WHERE status IN ('todo', 'in_progress') "
            f"LIMIT {int(limit)}",
        )
    except Exception:  # noqa: BLE001
        return {}
    return {
        str(r["title"]).strip().lower(): str(r["id"])
        for r in rows
        if r.get("title")
    }


def open_tasks_for_completion(
    db: Any, limit: int = 50,
) -> list[Task]:
    """Open tasks the completion detector should consider.

    sensitivity_tier: 2
    """
    rows = db.query(
        "SELECT * FROM _tasks "
        "WHERE status IN ('todo', 'in_progress') "
        "ORDER BY created_at DESC "
        f"LIMIT {int(limit)}",
    )
    return [_task_from_row(r) for r in rows]


def record_completion_candidate(
    db: Any,
    *,
    task_id: str,
    evidence_message_id: str,
    evidence_summary: str,
    confidence: float,
) -> None:
    """Persist a low-confidence completion verdict for user review.

    sensitivity_tier: 2
    """
    cid = make_hash_id(
        "task_completion",
        task_id,
        evidence_message_id,
    )
    try:
        db.execute(
            """INSERT OR IGNORE INTO _task_completion_candidates
               (id, task_id, evidence_message_id, evidence_summary,
                confidence, status, detected_at, sensitivity_tier)
               VALUES (?, ?, ?, ?, ?, 'pending', ?, 2)""",
            [
                cid, task_id, evidence_message_id,
                evidence_summary, confidence, utc_now_iso(),
            ],
        )
    except Exception:  # noqa: BLE001
        logger.debug(
            "Completion candidate insert failed", exc_info=True,
        )


def goal_titles_for_prompt(
    db: Any, max_goals: int = 30,
) -> list[dict[str, Any]]:
    """Compact projection of active goals for LLM prompts.

    sensitivity_tier: 2
    """
    rows = db.query(
        "SELECT id, title, category, importance, why "
        "FROM _goals WHERE status = 'active' "
        "ORDER BY importance DESC "
        f"LIMIT {int(max_goals)}",
    )
    return [dict(r) for r in rows]


def topics_for_prompt(
    db: Any, *, with_category: bool = True, limit: int = 50,
) -> list[dict[str, Any]]:
    """Compact projection of active topics for LLM prompts.

    sensitivity_tier: 2
    """
    try:
        cols = "id, contact_name, topic, description, importance, category"
        if not with_category:
            cols = "id, contact_name, topic, description, importance"
        rows = db.query(
            f"SELECT {cols} FROM _topics "
            "WHERE status = 'active' AND importance >= 5 "
            "ORDER BY importance DESC "
            f"LIMIT {int(limit)}",
        )
        return [dict(r) for r in rows]
    except Exception:  # noqa: BLE001
        return []


def iter_dicts(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Coerce a query result into a list of plain dicts.

    sensitivity_tier: 1
    """
    return [dict(r) for r in rows]


def get_actionable_tasks_today(
    db: Any,
    today_iso: str,
) -> list[dict[str, Any]]:
    """Return open tasks that need attention today, with goal metadata.

    Includes: overdue tasks, tasks due today, tasks scheduled today,
    and tasks with no due date (backlog) sorted by importance.

    sensitivity_tier: 2
    """
    rows = db.query(
        "SELECT t.id, t.title, t.goal_id, t.importance, t.due_at, "
        "  t.scheduled_for, t.status, t.notes, t.source, "
        "  g.title AS goal_title, g.category "
        "FROM _tasks t "
        "LEFT JOIN _goals g ON t.goal_id = g.id "
        "WHERE t.status IN ('todo', 'in_progress') "
        "ORDER BY "
        "  CASE "
        "    WHEN DATE(t.due_at) < DATE(?) THEN 0 "
        "    WHEN DATE(t.due_at) = DATE(?) THEN 1 "
        "    WHEN DATE(t.scheduled_for) = DATE(?) THEN 2 "
        "    ELSE 3 END, "
        "  t.importance DESC, t.due_at ASC "
        "LIMIT 50",
        [today_iso, today_iso, today_iso],
    )
    return [dict(r) for r in rows]


def get_habits_today(
    db: Any,
    today_iso: str,
) -> list[dict[str, Any]]:
    """Return active habits due today with goal metadata.

    Matches daily habits and habits whose days_of_week includes
    today's weekday name.

    sensitivity_tier: 1
    """
    from datetime import date as _date

    target = _date.fromisoformat(today_iso)
    weekday = target.strftime("%A").lower()

    rows = db.query(
        "SELECT h.id, h.title, h.goal_id, h.cadence, "
        "  h.preferred_window, h.days_of_week, "
        "  g.title AS goal_title, g.category "
        "FROM _habits h "
        "LEFT JOIN _goals g ON h.goal_id = g.id "
        "WHERE h.status = 'active' "
        "ORDER BY "
        "  CASE h.preferred_window "
        "    WHEN 'morning' THEN 0 WHEN 'midday' THEN 1 "
        "    WHEN 'evening' THEN 2 ELSE 3 END, "
        "  h.title",
        [],
    )
    results: list[dict[str, Any]] = []
    for r in rows:
        row = dict(r)
        cadence = row.get("cadence", "daily")
        if cadence == "daily":
            results.append(row)
        elif cadence == "specific_days":
            dow_raw = row.get("days_of_week") or "[]"
            if isinstance(dow_raw, str):
                import json
                try:
                    days = json.loads(dow_raw)
                except (json.JSONDecodeError, TypeError):
                    days = []
            else:
                days = list(dow_raw)
            if weekday in [d.lower() for d in days]:
                results.append(row)
        elif cadence == "weekly":
            results.append(row)
    return results
