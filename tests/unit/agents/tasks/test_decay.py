"""Goal decay — pure logic + curator archive pass.

Issue #41: ``last_confirmed_at`` was written by the miner but never read
to fade/archive stale goals. These cover the pure decay verdict and the
curator's reversible auto-archive of stale brain-mined goals.

sensitivity_tier: N/A — test infrastructure
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from src.agents.core.output_types import GoalBatch, GoalDraft
from src.agents.tasks import TaskCurator
from src.agents.tasks.decay import (
    ARCHIVE_AFTER_DAYS,
    FADE_AFTER_DAYS,
    MAX_PENALTY,
    goal_decay,
    should_archive,
)
from src.agents.tasks.persistence import update_goal_fields
from src.core.sqlite.engine import DatabaseEngine

_NOW = datetime(2026, 7, 23, 12, 0, tzinfo=timezone.utc)


def _ago(days: int) -> str:
    return (_NOW - timedelta(days=days)).isoformat()


# ================================================================
# Pure decay verdict
# ================================================================


class TestGoalDecay:
    def test_fresh_when_recently_confirmed(self) -> None:
        d = goal_decay(
            source="brain", last_confirmed_at=_ago(1),
            created_at=_ago(40), now=_NOW,
        )
        assert d.state == "fresh"
        assert d.penalty == 0

    def test_fading_past_fade_threshold(self) -> None:
        d = goal_decay(
            source="brain", last_confirmed_at=_ago(FADE_AFTER_DAYS + 1),
            created_at=_ago(60), now=_NOW,
        )
        assert d.state == "fading"
        assert 0 < d.penalty <= MAX_PENALTY

    def test_stale_past_archive_threshold(self) -> None:
        d = goal_decay(
            source="brain", last_confirmed_at=_ago(ARCHIVE_AFTER_DAYS + 5),
            created_at=_ago(90), now=_NOW,
        )
        assert d.state == "stale"
        assert d.penalty == MAX_PENALTY
        assert should_archive(d)

    def test_user_goals_never_decay(self) -> None:
        d = goal_decay(
            source="user", last_confirmed_at=_ago(365),
            created_at=_ago(400), now=_NOW,
        )
        assert d.state == "fresh"
        assert d.penalty == 0
        assert not should_archive(d)

    def test_falls_back_to_created_at_when_never_confirmed(self) -> None:
        d = goal_decay(
            source="brain", last_confirmed_at=None,
            created_at=_ago(ARCHIVE_AFTER_DAYS + 1), now=_NOW,
        )
        assert d.state == "stale"

    def test_unparseable_anchor_is_fresh(self) -> None:
        d = goal_decay(
            source="brain", last_confirmed_at="not-a-date",
            created_at=None, now=_NOW,
        )
        assert d.state == "fresh"
        assert d.penalty == 0

    def test_penalty_ramps_monotonically(self) -> None:
        p_fade = goal_decay(
            source="brain", last_confirmed_at=_ago(FADE_AFTER_DAYS),
            created_at=None, now=_NOW,
        ).penalty
        p_mid = goal_decay(
            source="brain",
            last_confirmed_at=_ago((FADE_AFTER_DAYS + ARCHIVE_AFTER_DAYS) // 2),
            created_at=None, now=_NOW,
        ).penalty
        p_stale = goal_decay(
            source="brain", last_confirmed_at=_ago(ARCHIVE_AFTER_DAYS),
            created_at=None, now=_NOW,
        ).penalty
        assert p_fade <= p_mid <= p_stale == MAX_PENALTY


# ================================================================
# Curator archive pass (reversible)
# ================================================================


@pytest.fixture()
def db(tmp_path: Path) -> DatabaseEngine:
    return DatabaseEngine(tmp_path / "test_decay.db")


@pytest.fixture()
def curator(db: DatabaseEngine) -> TaskCurator:
    return TaskCurator(db_engine=db)


class TestDecayGoalsPass:
    def test_archives_stale_brain_goal(self, curator, db) -> None:
        g = curator.create_goal(
            title="Learn Rust", category="work", source="brain",
        )
        # Backdate its evidence far past the archive threshold.
        old = (datetime.now(timezone.utc)
               - timedelta(days=ARCHIVE_AFTER_DAYS + 5)).isoformat()
        update_goal_fields(db, g.id, last_confirmed_at=old)

        archived = curator.decay_goals()

        assert [a.id for a in archived] == [g.id]
        assert curator.get_goal(g.id).status == "archived"
        # It's out of the default active list…
        assert g.id not in {x.id for x in curator.list_goals(status="active")}
        # …but present in the archived review queue (reversible).
        assert g.id in {x.id for x in curator.list_goals(status="archived")}

    def test_does_not_archive_fresh_brain_goal(self, curator) -> None:
        g = curator.create_goal(
            title="Ship v2", category="work", source="brain",
        )
        # create_goal sets last_confirmed_at=now, so it's fresh.
        assert curator.decay_goals() == []
        assert curator.get_goal(g.id).status == "active"

    def test_never_archives_user_goal(self, curator, db) -> None:
        g = curator.create_goal(
            title="Run a marathon", category="personal", source="user",
        )
        old = (datetime.now(timezone.utc)
               - timedelta(days=ARCHIVE_AFTER_DAYS + 50)).isoformat()
        update_goal_fields(db, g.id, last_confirmed_at=old)

        assert curator.decay_goals() == []
        assert curator.get_goal(g.id).status == "active"

    def test_restore_is_possible(self, curator, db) -> None:
        """Archiving is reversible — the review queue can restore."""
        g = curator.create_goal(
            title="Read more", category="personal", source="brain",
        )
        old = (datetime.now(timezone.utc)
               - timedelta(days=ARCHIVE_AFTER_DAYS + 5)).isoformat()
        update_goal_fields(db, g.id, last_confirmed_at=old)
        curator.decay_goals()
        assert curator.get_goal(g.id).status == "archived"

        curator.update_goal(g.id, status="active")
        assert curator.get_goal(g.id).status == "active"

    def test_returned_snapshot_reflects_archived_status(
        self, curator, db,
    ) -> None:
        """The reported goal's status matches what was persisted."""
        g = curator.create_goal(
            title="Stale", category="work", source="brain",
        )
        old = (datetime.now(timezone.utc)
               - timedelta(days=ARCHIVE_AFTER_DAYS + 1)).isoformat()
        update_goal_fields(db, g.id, last_confirmed_at=old)
        [archived] = curator.decay_goals()
        assert archived.status == "archived"


class TestMineGoalsRunsDecay:
    """The decay pass is wired into the mining cadence, and a goal
    re-confirmed in the same pass is spared."""

    def test_mining_archives_stale_but_spares_reconfirmed(
        self, curator, db, monkeypatch,
    ) -> None:
        # One goal the upcoming mining pass will re-confirm…
        keep = curator.create_goal(
            title="Ship v1", category="work", source="brain",
        )
        # …and one it won't mention, already past the archive line.
        drop = curator.create_goal(
            title="Learn cello", category="work", source="brain",
        )
        old = (datetime.now(timezone.utc)
               - timedelta(days=ARCHIVE_AFTER_DAYS + 5)).isoformat()
        # Backdate BOTH so only the reconfirm (not recency) saves `keep`.
        update_goal_fields(db, keep.id, last_confirmed_at=old)
        update_goal_fields(db, drop.id, last_confirmed_at=old)

        drafts = GoalBatch(goals=[
            GoalDraft(
                title="Ship v1",  # dedups onto `keep` -> reconfirms it
                description="",
                category="work",
                horizon="short",
                importance=8,
                why="",
                source_kind="message",
                source_ref="m1",
                linked_topic_hint=None,
            ),
        ])
        monkeypatch.setattr(
            "src.agents.goal_extractor.agent.GoalExtractorAgent",
            MagicMock(
                return_value=MagicMock(
                    extract=MagicMock(return_value=drafts),
                ),
            ),
        )
        monkeypatch.setattr(
            curator, "_fetch_recent_messages", lambda n: [{"id": "m1"}],
        )
        monkeypatch.setattr(curator, "_fetch_recent_notes", lambda n: [])
        monkeypatch.setattr(curator, "_fetch_recent_facts", lambda n: [])

        curator.mine_goals()

        # Reconfirmed goal survived; unmentioned stale goal was archived.
        assert curator.get_goal(keep.id).status == "active"
        assert curator.get_goal(drop.id).status == "archived"
