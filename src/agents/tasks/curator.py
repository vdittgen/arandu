"""TaskCurator — owns CRUD + agent orchestration for tasks/goals/habits.

The pydantic-ai ``TaskCuratorAgent`` is the LLM-facing surface used
when Brain delegates a chat turn here. ``TaskCurator`` is the
Python-facing surface used by the CLI, the post-sync hook, and the
2-hour proactive cycle. It owns the database side-effects so the
sub-agents can stay stateless single-purpose pieces.

sensitivity_tier: 2
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import date as date_cls
from typing import Any

from src.agents.tasks.decay import (
    ARCHIVE_AFTER_DAYS,
    FADE_AFTER_DAYS,
    goal_decay,
    should_archive,
)
from src.agents.tasks.models import (
    DailyScheduleRecord,
    Goal,
    Habit,
    Project,
    ScheduleSlotRecord,
    Task,
)
from src.agents.tasks.persistence import (
    dedup_goal_id,
    delete_habit,
    delete_task,
    ensure_schema,
    find_open_task_titles,
    find_topic_id_by_name,
    get_daily_schedule,
    get_goal,
    get_project,
    get_task,
    goal_titles_for_prompt,
    insert_goal,
    insert_habit,
    insert_project,
    insert_task,
    list_goals,
    list_habits,
    list_projects,
    list_tasks,
    list_topics_for_goal,
    open_tasks_for_completion,
    record_completion_candidate,
    seed_default_projects,
    set_topic_linked_goal,
    topics_for_prompt,
    update_goal_fields,
    update_habit_status,
    update_task_fields,
    upsert_schedule,
)
from src.core.db_helpers import make_hash_id, utc_ago_iso, utc_now_iso

logger = logging.getLogger(__name__)


# Confidence floor above which task_completion auto-closes a task.
_AUTO_COMPLETE_CONFIDENCE = 0.7


class TaskCurator:
    """Curator surface for goals/tasks/habits/schedule.

    Stateless except for the database handle. Each public method
    handles one verb so CLI handlers stay one-liners.

    sensitivity_tier: 2
    """

    def __init__(self, db_engine: Any) -> None:
        self._db = db_engine
        ensure_schema(self._db)
        seed_default_projects(self._db)

    # ------------------------------------------------------------------
    # Goal CRUD
    # ------------------------------------------------------------------

    def create_goal(
        self,
        *,
        title: str,
        category: str,
        description: str = "",
        horizon: str = "medium",
        target_date: str | None = None,
        importance: int = 5,
        why: str = "",
        source: str = "user",
        source_ref: str | None = None,
    ) -> Goal:
        """Create or fetch a goal. Dedups on (title, category).

        sensitivity_tier: 2
        """
        existing_id = dedup_goal_id(self._db, title, category)
        if existing_id:
            existing = get_goal(self._db, existing_id)
            if existing is not None:
                return existing
        now = utc_now_iso()
        goal = Goal(
            id=make_hash_id("goal", category, title.strip().lower(), now),
            title=title.strip(),
            description=description,
            category=category,
            horizon=horizon,
            target_date=target_date,
            status="active",
            importance=max(1, min(10, int(importance))),
            why=why,
            source=source,
            source_ref=source_ref,
            created_at=now,
            updated_at=now,
            last_confirmed_at=now,
        )
        insert_goal(self._db, goal)
        return goal

    def update_goal(self, goal_id: str, **fields: Any) -> Goal | None:
        """sensitivity_tier: 2"""
        update_goal_fields(self._db, goal_id, **fields)
        return get_goal(self._db, goal_id)

    def list_goals(
        self,
        *,
        status: str | None = "active",
        category: str | None = None,
    ) -> list[Goal]:
        """sensitivity_tier: 2"""
        return list_goals(self._db, status=status, category=category)

    def get_goal(self, goal_id: str) -> Goal | None:
        """sensitivity_tier: 2"""
        return get_goal(self._db, goal_id)

    def archive_goal(self, goal_id: str) -> None:
        """sensitivity_tier: 2"""
        update_goal_fields(self._db, goal_id, status="archived")

    def decay_goals(
        self,
        *,
        fade_after_days: int = FADE_AFTER_DAYS,
        archive_after_days: int = ARCHIVE_AFTER_DAYS,
    ) -> list[Goal]:
        """Auto-archive brain-mined goals whose evidence has gone stale.

        A goal the miner stops re-confirming keeps an increasingly old
        ``last_confirmed_at``; once it crosses the archive threshold it
        is moved to ``status='archived'``. Reversible — the row keeps
        all its data and is restored by setting ``status`` back to
        ``active`` (the archived list is the review queue). Only active
        brain-mined goals are considered; user goals never decay.

        Returns the goals archived this pass.

        sensitivity_tier: 2
        """
        archived: list[Goal] = []
        for goal in list_goals(self._db, status="active"):
            if goal.source != "brain":
                continue
            decay = goal_decay(
                source=goal.source,
                last_confirmed_at=goal.last_confirmed_at,
                created_at=goal.created_at,
                fade_after_days=fade_after_days,
                archive_after_days=archive_after_days,
            )
            if should_archive(decay):
                update_goal_fields(self._db, goal.id, status="archived")
                archived.append(goal)
        if archived:
            logger.info(
                "Goal decay archived %d stale brain goal(s)", len(archived),
            )
        return archived

    # ------------------------------------------------------------------
    # Project CRUD
    # ------------------------------------------------------------------

    def create_project(
        self,
        *,
        name: str,
        category: str = "personal",
        goal_id: str | None = None,
        topic_id: str | None = None,
        color: str | None = None,
    ) -> Project:
        """sensitivity_tier: 2"""
        if goal_id and category == "personal":
            goal = get_goal(self._db, goal_id)
            if goal is not None:
                category = goal.category
        now = utc_now_iso()
        project = Project(
            id=make_hash_id("project", name.strip().lower(), now),
            name=name.strip(),
            category=category,
            topic_id=topic_id,
            goal_id=goal_id,
            status="active",
            color=color,
            created_at=now,
            updated_at=now,
        )
        insert_project(self._db, project)
        return project

    def list_projects(
        self,
        *,
        status: str | None = "active",
        category: str | None = None,
    ) -> list[Project]:
        """sensitivity_tier: 2"""
        return list_projects(self._db, status=status, category=category)

    def archive_project(self, project_id: str) -> None:
        """sensitivity_tier: 2"""
        self._db.execute(
            "UPDATE _projects SET status = 'archived', updated_at = ? "
            "WHERE id = ?",
            [utc_now_iso(), project_id],
        )

    # ------------------------------------------------------------------
    # Task CRUD
    # ------------------------------------------------------------------

    def create_task(
        self,
        *,
        title: str,
        project_id: str | None = None,
        parent_task_id: str | None = None,
        goal_id: str | None = None,
        notes: str = "",
        importance: int = 5,
        due_at: str | None = None,
        source: str = "user",
        source_ref: str | None = None,
    ) -> Task:
        """sensitivity_tier: 2"""
        # Inherit goal_id from parent project when not set explicitly.
        if goal_id is None and project_id:
            project = get_project(self._db, project_id)
            if project is not None:
                goal_id = project.goal_id
        now = utc_now_iso()
        task = Task(
            id=make_hash_id("task", title.strip().lower(), now),
            title=title.strip(),
            project_id=project_id,
            parent_task_id=parent_task_id,
            goal_id=goal_id,
            notes=notes,
            status="todo",
            importance=max(1, min(10, int(importance))),
            due_at=due_at,
            source=source,
            source_ref=source_ref,
            created_at=now,
            updated_at=now,
        )
        insert_task(self._db, task)
        return task

    def update_task(self, task_id: str, **fields: Any) -> Task | None:
        """sensitivity_tier: 2"""
        update_task_fields(self._db, task_id, **fields)
        return get_task(self._db, task_id)

    def toggle_task_done(
        self,
        task_id: str,
        *,
        completion_note: str | None = None,
        completion_evidence_id: str | None = None,
    ) -> Task | None:
        """Mark a task done (or back to todo if currently done).

        sensitivity_tier: 2
        """
        task = get_task(self._db, task_id)
        if task is None:
            return None
        if task.status == "done":
            update_task_fields(
                self._db, task_id,
                status="todo",
                completed_at=None,
                completion_note=None,
                completion_evidence_id=None,
            )
        else:
            update_task_fields(
                self._db, task_id,
                status="done",
                completed_at=utc_now_iso(),
                completion_note=completion_note,
                completion_evidence_id=completion_evidence_id,
            )
        return get_task(self._db, task_id)

    def delete_task(self, task_id: str) -> None:
        """sensitivity_tier: 2"""
        delete_task(self._db, task_id)

    def list_tasks(
        self,
        *,
        status: str | None = None,
        project_id: str | None = None,
        goal_id: str | None = None,
        parent_task_id: str | None = None,
    ) -> list[Task]:
        """sensitivity_tier: 2"""
        return list_tasks(
            self._db,
            status=status,
            project_id=project_id,
            goal_id=goal_id,
            parent_task_id=parent_task_id,
        )

    # ------------------------------------------------------------------
    # Habit CRUD
    # ------------------------------------------------------------------

    def create_habit(
        self,
        *,
        title: str,
        goal_id: str,
        cadence: str = "daily",
        days_of_week: tuple[str, ...] = (),
        preferred_window: str = "any",
        why: str = "",
        source: str = "user",
    ) -> Habit:
        """sensitivity_tier: 1"""
        if not goal_id:
            raise ValueError("habit must be anchored to a goal_id")
        now = utc_now_iso()
        habit = Habit(
            id=make_hash_id("habit", goal_id, title.strip().lower(), now),
            title=title.strip(),
            goal_id=goal_id,
            cadence=cadence,
            days_of_week=tuple(days_of_week),
            preferred_window=preferred_window,
            why=why,
            source=source,
            status="active",
            created_at=now,
        )
        insert_habit(self._db, habit)
        return habit

    def list_habits(
        self,
        *,
        status: str | None = "active",
        goal_id: str | None = None,
    ) -> list[Habit]:
        """sensitivity_tier: 1"""
        return list_habits(self._db, status=status, goal_id=goal_id)

    def toggle_habit(self, habit_id: str) -> None:
        """Flip habit active/paused.

        sensitivity_tier: 1
        """
        rows = self._db.query(
            "SELECT status FROM _habits WHERE id = ?", [habit_id],
        )
        if not rows:
            return
        new_status = "paused" if rows[0]["status"] == "active" else "active"
        update_habit_status(self._db, habit_id, new_status)

    def delete_habit(self, habit_id: str) -> None:
        """sensitivity_tier: 1"""
        delete_habit(self._db, habit_id)

    # ------------------------------------------------------------------
    # Proactive flows — invoked by post-sync and the 2-hour cycle
    # ------------------------------------------------------------------

    def mine_goals(
        self,
        *,
        message_limit: int = 200,
        note_limit: int = 50,
        fact_limit: int = 50,
        chat_limit: int = 20,
    ) -> list[Goal] | None:
        """Run goal_extractor over recent evidence; upsert new goals.

        Existing active goals matching ``(title, category)`` get
        ``last_confirmed_at`` bumped instead of duplicated. Returns the
        list of *new* goals inserted this pass, an empty list when the
        evidence legitimately yields nothing, or ``None`` when the
        extractor itself failed (model error, provider down) — callers
        must not treat a failed run as "no goals found".

        sensitivity_tier: 2
        """
        try:
            messages = self._fetch_recent_messages(message_limit)
            notes = self._fetch_recent_notes(note_limit)
            facts = self._fetch_recent_facts(fact_limit)
            chat_excerpts = self._fetch_recent_chat_excerpts(chat_limit)
            known_topics = topics_for_prompt(self._db)
        except Exception:  # noqa: BLE001
            logger.warning("Goal mining evidence fetch failed", exc_info=True)
            return []

        from src.agents.goal_extractor.agent import GoalExtractorAgent

        try:
            batch = GoalExtractorAgent().extract(
                messages=messages,
                notes=notes,
                facts=facts,
                chat_excerpts=chat_excerpts,
                known_topics=known_topics,
            )
        except Exception:  # noqa: BLE001
            logger.warning("GoalExtractorAgent failed", exc_info=True)
            return None
        if batch is None:
            logger.warning("Goal extraction did not complete (model error)")
            return None

        created: list[Goal] = []
        for draft in batch.goals:
            existing_id = dedup_goal_id(
                self._db, draft.title, draft.category,
            )
            if existing_id:
                update_goal_fields(
                    self._db, existing_id,
                    last_confirmed_at=utc_now_iso(),
                )
                # Reconcile linked topic even on dedup.
                if draft.linked_topic_hint:
                    topic_id = find_topic_id_by_name(
                        self._db, draft.linked_topic_hint,
                    )
                    if topic_id:
                        set_topic_linked_goal(
                            self._db, topic_id, existing_id,
                        )
                continue

            goal = self.create_goal(
                title=draft.title,
                category=draft.category,
                description=draft.description,
                horizon=draft.horizon,
                target_date=draft.target_date,
                importance=draft.importance,
                why=draft.why,
                source="brain",
                source_ref=f"{draft.source_kind}:{draft.source_ref}",
            )
            created.append(goal)
            if draft.linked_topic_hint:
                topic_id = find_topic_id_by_name(
                    self._db, draft.linked_topic_hint,
                )
                if topic_id:
                    set_topic_linked_goal(self._db, topic_id, goal.id)

        # Decay runs on the mining cadence: goals the miner just
        # re-confirmed above had their timestamp bumped, so only goals
        # whose evidence has genuinely dried up cross the threshold.
        # Non-fatal — a decay failure must not sink a good mining pass.
        try:
            self.decay_goals()
        except Exception:  # noqa: BLE001
            logger.warning("Goal decay pass failed", exc_info=True)

        return created

    def propose_from_messages(
        self, message_batch: list[dict[str, Any]],
    ) -> list[Task]:
        """Run task_proposer over a batch; insert new (deduped) tasks.

        sensitivity_tier: 2
        """
        if not message_batch:
            return []
        topics = topics_for_prompt(self._db)
        goals = goal_titles_for_prompt(self._db)
        from src.agents.task_proposer.agent import TaskProposerAgent

        try:
            batch = TaskProposerAgent().propose(
                messages=message_batch,
                topics=topics,
                goals=goals,
            )
        except Exception:  # noqa: BLE001
            logger.warning("TaskProposerAgent failed", exc_info=True)
            return []
        if batch is None or not batch.tasks:
            return []

        open_titles = find_open_task_titles(self._db)
        goal_lookup = {g["title"].lower(): g["id"] for g in goals}
        # Default project per category (the seeded ones).
        default_projects = {
            p.category: p.id
            for p in list_projects(self._db)
            if p.name in ("Personal", "Life", "Work")
        }

        created: list[Task] = []
        for draft in batch.tasks:
            key = draft.title.strip().lower()
            if key in open_titles:
                continue
            goal_id: str | None = None
            if draft.parent_goal_hint:
                goal_id = goal_lookup.get(draft.parent_goal_hint.lower())
            task = self.create_task(
                title=draft.title,
                project_id=default_projects.get(draft.category),
                goal_id=goal_id,
                notes=draft.notes,
                importance=draft.importance,
                due_at=draft.due_at,
                source="brain",
                source_ref=";".join(draft.source_message_ids),
            )
            created.append(task)
        return created

    def detect_completions(
        self, evidence_batch: list[dict[str, Any]],
    ) -> list[Task]:
        """Run task_completion over new evidence; close confident matches.

        Lower-confidence verdicts go to ``_task_completion_candidates``.
        Returns tasks auto-closed this pass.

        sensitivity_tier: 2
        """
        if not evidence_batch:
            return []
        open_tasks = open_tasks_for_completion(self._db)
        if not open_tasks:
            return []
        from src.agents.task_completion.agent import TaskCompletionAgent

        task_dicts = [asdict(t) for t in open_tasks]
        try:
            batch = TaskCompletionAgent().detect(
                open_tasks=task_dicts,
                evidence=evidence_batch,
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "TaskCompletionAgent failed", exc_info=True,
            )
            return []
        if batch is None or not batch.completions:
            return []

        valid_ids = {t.id for t in open_tasks}
        closed: list[Task] = []
        for draft in batch.completions:
            if draft.task_id not in valid_ids:
                continue
            if draft.confidence >= _AUTO_COMPLETE_CONFIDENCE:
                self.toggle_task_done(
                    draft.task_id,
                    completion_note=draft.evidence_summary,
                    completion_evidence_id=draft.evidence_message_id,
                )
                t = get_task(self._db, draft.task_id)
                if t is not None:
                    closed.append(t)
            else:
                record_completion_candidate(
                    self._db,
                    task_id=draft.task_id,
                    evidence_message_id=draft.evidence_message_id,
                    evidence_summary=draft.evidence_summary,
                    confidence=draft.confidence,
                )
        return closed

    def regenerate_habits(self) -> list[Habit]:
        """Run habit_suggester; upsert brain-sourced habits.

        Reads `_goals` + topics linked to those goals. User-added
        habits (``source='user'``) are never overwritten.

        sensitivity_tier: 1
        """
        goals = goal_titles_for_prompt(self._db)
        if not goals:
            return []
        # Gather topics linked to any active goal.
        linked_topics: list[dict[str, Any]] = []
        for g in goals:
            rows = list_topics_for_goal(self._db, str(g["id"]))
            for r in rows:
                row = dict(r)
                row["goal_id"] = g["id"]
                linked_topics.append(row)

        from src.agents.habit_suggester.agent import HabitSuggesterAgent

        try:
            batch = HabitSuggesterAgent().suggest(
                goals=goals, linked_topics=linked_topics,
            )
        except Exception:  # noqa: BLE001
            logger.warning("HabitSuggesterAgent failed", exc_info=True)
            return []
        if batch is None or not batch.habits:
            return []

        # Wipe existing brain-sourced habits before re-inserting — user
        # habits are preserved.
        self._db.execute(
            "DELETE FROM _habits WHERE source = 'brain'",
        )
        created: list[Habit] = []
        for draft in batch.habits:
            habit = self.create_habit(
                title=draft.title,
                goal_id=draft.goal_id,
                cadence=draft.cadence,
                days_of_week=tuple(draft.days_of_week),
                preferred_window=draft.preferred_window,
                why=draft.why,
                source="brain",
            )
            created.append(habit)
        return created

    def regenerate_daily_schedule(
        self,
        *,
        schedule_date: str | None = None,
        working_hours: str = "08:00-19:00",
        category_mix: dict[str, int] | None = None,
    ) -> DailyScheduleRecord | None:
        """Build today's plan via ``daily_scheduler`` and persist it.

        sensitivity_tier: 2
        """
        target = schedule_date or date_cls.today().isoformat()
        events = self._fetch_events_for_date(target)
        tasks = [
            asdict(t) for t in self.list_tasks(status="todo")
            if not t.parent_task_id
        ]
        habits = [
            asdict(h) for h in self.list_habits(status="active")
        ]
        goals = goal_titles_for_prompt(self._db)
        from src.agents.daily_scheduler.agent import DailySchedulerAgent

        try:
            schedule = DailySchedulerAgent().plan(
                schedule_date=target,
                events=events,
                tasks=tasks,
                habits=habits,
                goals=goals,
                working_hours=working_hours,
                category_mix=category_mix or {
                    "work": 60, "personal": 25, "life": 15,
                },
            )
        except Exception:  # noqa: BLE001
            logger.warning("DailySchedulerAgent failed", exc_info=True)
            return None
        if schedule is None:
            return None

        record = DailyScheduleRecord(
            schedule_date=target,
            slots=[
                ScheduleSlotRecord(
                    kind=s.kind,
                    ref_id=s.ref_id,
                    title=s.title,
                    start=s.start,
                    end=s.end,
                    why=s.why,
                    category=s.category,
                    goal_id=s.goal_id,
                )
                for s in schedule.slots
            ],
            unscheduled_overflow=list(schedule.unscheduled_overflow),
            rationale=schedule.rationale,
            category_balance=dict(schedule.category_balance),
            generated_at=utc_now_iso(),
        )
        upsert_schedule(self._db, record)
        return record

    def get_daily_schedule(
        self, schedule_date: str | None = None,
    ) -> DailyScheduleRecord | None:
        """sensitivity_tier: 2"""
        target = schedule_date or date_cls.today().isoformat()
        return get_daily_schedule(self._db, target)

    # ------------------------------------------------------------------
    # Evidence fetchers — keep DB SQL out of the agents themselves
    # ------------------------------------------------------------------

    def _fetch_recent_messages(
        self, limit: int,
    ) -> list[dict[str, Any]]:
        """Recent messages worth mining, newest first.

        Filters the raw stream down to signal before it ever reaches a
        prompt budget (unfiltered, the newest rows are dominated by
        group chatter and bridge media stubs):

        - drops bridge placeholder stubs (``[audio]``, ``[image]``,
          ``[messageContextInfo]`` …) — structural artifacts of the
          WhatsApp bridge, not content;
        - keeps direct messages and anything the user authored;
        - keeps group messages only from groups the user actively
          participates in (sent at least one message in the past
          month) — other groups are spectator noise.

        sensitivity_tier: 3
        """
        try:
            rows = self._db.query(
                "SELECT id, source, sender, content, timestamp "
                "FROM raw_messages "
                "WHERE timestamp >= ? "
                "  AND TRIM(content) != '' "
                "  AND content NOT LIKE '[%]' "
                "  AND ( "
                "    COALESCE(is_group, 0) = 0 "
                "    OR COALESCE(is_from_me, 0) = 1 "
                "    OR chat_name IN ( "
                "      SELECT DISTINCT chat_name FROM raw_messages "
                "      WHERE COALESCE(is_group, 0) = 1 "
                "        AND COALESCE(is_from_me, 0) = 1 "
                "        AND chat_name IS NOT NULL "
                "        AND timestamp >= ? "
                "    ) "
                "  ) "
                "ORDER BY timestamp DESC "
                f"LIMIT {int(limit)}",
                [utc_ago_iso(days=30), utc_ago_iso(days=30)],
            )
        except Exception:  # noqa: BLE001
            return []
        return [dict(r) for r in rows]

    def _fetch_recent_chat_excerpts(self, limit: int) -> list[str]:
        """Recent questions the user asked the Brain, oldest first.

        What the user asks about is direct evidence of what they care
        about; ``_query_log`` records every ``ask``/``ask-stream``.

        sensitivity_tier: 3
        """
        try:
            # asked_at is CURRENT_TIMESTAMP (space-separated UTC), so
            # SQLite's own datetime() is the format-matched comparison
            # here — unlike the ISO-T raw_* columns above.
            rows = self._db.query(
                "SELECT id, question FROM _query_log "
                "WHERE asked_at >= datetime('now', '-30 days') "
                "ORDER BY asked_at DESC "
                f"LIMIT {int(limit)}",
            )
        except Exception:  # noqa: BLE001
            return []
        return [
            f"[{r['id']}] user asked: {r['question']}"
            for r in reversed(list(rows))
        ]

    def _fetch_recent_notes(self, limit: int) -> list[dict[str, Any]]:
        """sensitivity_tier: 2"""
        try:
            rows = self._db.query(
                "SELECT id, title, content "
                "FROM raw_notes "
                "WHERE updated_at >= datetime('now', '-90 days') "
                "ORDER BY updated_at DESC "
                f"LIMIT {int(limit)}",
            )
        except Exception:  # noqa: BLE001
            return []
        return [dict(r) for r in rows]

    def _fetch_recent_facts(self, limit: int) -> list[dict[str, Any]]:
        """sensitivity_tier: 2"""
        try:
            rows = self._db.query(
                "SELECT id, category, subject, predicate, content "
                "FROM _learned_facts "
                "WHERE dismissed_at IS NULL "
                "ORDER BY extracted_at DESC "
                f"LIMIT {int(limit)}",
            )
        except Exception:  # noqa: BLE001
            return []
        return [dict(r) for r in rows]

    def _fetch_events_for_date(
        self, target: str,
    ) -> list[dict[str, Any]]:
        """Return the user's own committed events on ``target``.

        Routed through :func:`personal_events_for_date` so the scheduler
        and the daily brief share the same definition of "events the
        user actually owns" — subscribed calendars and team-awareness
        entries belong to dedicated awareness panels, not the plan.

        sensitivity_tier: 2
        """
        from src.core.calendar_filters import personal_events_for_date

        return personal_events_for_date(self._db, target)
