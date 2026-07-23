"""Meeting prep briefs (#40).

Composes a "before your 2pm with X, here's where things stand" pack for
an upcoming meeting from already-cached data — the enriched-events mart,
cached ``_contact_contexts``, and ``int_contact_topics`` — with no LLM,
so it's safe on page load.

sensitivity_tier: N/A — test infrastructure
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest
from src.agents.proactive import ProactiveIntelligence
from src.agents.proactive.persistence import _within_window
from src.core.sqlite.engine import DatabaseEngine

_NOW = datetime(2026, 7, 23, 12, 0)


@pytest.fixture()
def db(tmp_path: Path) -> DatabaseEngine:
    engine = DatabaseEngine(tmp_path / "test_prep.db")
    yield engine
    engine.close()


def _seed_event(
    db: DatabaseEngine, *, start: datetime, attendees: str,
    title: str = "Sync", location: str = "Room 4", eid: str = "e1",
) -> None:
    db.execute(
        """CREATE TABLE IF NOT EXISTS int_events_enriched (
            id VARCHAR, title VARCHAR, start_time TEXT, location VARCHAR,
            known_attendee_names VARCHAR
        )""",
    )
    db.execute(
        "INSERT INTO int_events_enriched "
        "(id, title, start_time, location, known_attendee_names) "
        "VALUES (?, ?, ?, ?, ?)",
        [eid, title, start.isoformat(), location, attendees],
    )


def _seed_contact_context(
    pi: ProactiveIntelligence, db: DatabaseEngine, *,
    name: str, situation: str, domains: str = '["work"]',
    preview: str = "sent the draft",
) -> None:
    db.execute(
        "INSERT INTO _contact_contexts "
        "(contact_id, contact_name, active_context, context_domains, "
        " last_message_at, last_message_preview) VALUES (?, ?, ?, ?, ?, ?)",
        [name.lower(), name, situation, domains, "2026-07-22T09:00", preview],
    )


def _seed_topics(db: DatabaseEngine, *, name: str) -> None:
    db.execute(
        """CREATE TABLE IF NOT EXISTS int_contact_topics (
            contact_name VARCHAR, topic VARCHAR, description TEXT,
            importance INTEGER, status VARCHAR
        )""",
    )
    db.execute(
        "INSERT INTO int_contact_topics VALUES (?, ?, ?, ?, ?)",
        [name, "Q3 budget", "waiting on your sign-off", 8, "active"],
    )
    db.execute(
        "INSERT INTO int_contact_topics VALUES (?, ?, ?, ?, ?)",
        [name, "resolved thing", "done", 3, "resolved"],
    )


# ================================================================
# _within_window (pure)
# ================================================================


class TestWithinWindow:
    def test_event_in_window(self) -> None:
        assert _within_window(
            (_NOW + timedelta(hours=2)).isoformat(), _NOW, 24.0,
        )

    def test_event_past_is_excluded(self) -> None:
        assert not _within_window(
            (_NOW - timedelta(hours=1)).isoformat(), _NOW, 24.0,
        )

    def test_event_beyond_window_excluded(self) -> None:
        assert not _within_window(
            (_NOW + timedelta(hours=30)).isoformat(), _NOW, 24.0,
        )

    def test_unparseable_is_excluded(self) -> None:
        assert not _within_window("not-a-date", _NOW, 24.0)

    def test_tolerates_space_separated(self) -> None:
        assert _within_window("2026-07-23 14:00:00", _NOW, 24.0)


# ================================================================
# get_meeting_prep_briefs
# ================================================================


class TestMeetingPrepBriefs:
    def test_composes_brief_for_upcoming_meeting(self, db) -> None:
        pi = ProactiveIntelligence(db_engine=db)
        _seed_event(
            db, start=_NOW + timedelta(hours=2), attendees="Maria Silva",
        )
        _seed_contact_context(
            pi, db, name="Maria Silva",
            situation="mid-negotiation on the clinic lease",
        )
        _seed_topics(db, name="Maria Silva")

        briefs = pi.get_meeting_prep_briefs(within_hours=24, now=_NOW)

        assert len(briefs) == 1
        b = briefs[0]
        assert b.title == "Sync"
        assert b.location == "Room 4"
        assert len(b.attendees) == 1
        a = b.attendees[0]
        assert a.contact_name == "Maria Silva"
        assert a.situation == "mid-negotiation on the clinic lease"
        assert a.domains == ["work"]
        assert a.last_message_preview == "sent the draft"
        # Only the active open loop, not the resolved one.
        assert [loop["topic"] for loop in a.open_loops] == ["Q3 budget"]

    def test_multiple_attendees_split(self, db) -> None:
        pi = ProactiveIntelligence(db_engine=db)
        _seed_event(
            db, start=_NOW + timedelta(hours=1),
            attendees="Maria Silva, Bob Jones",
        )
        _seed_contact_context(pi, db, name="Maria Silva", situation="s1")

        briefs = pi.get_meeting_prep_briefs(within_hours=24, now=_NOW)
        names = {a.contact_name for a in briefs[0].attendees}
        # Both attendees appear; the one without a cached context still
        # shows (name only), so nothing known is dropped.
        assert names == {"Maria Silva", "Bob Jones"}

    def test_past_meeting_produces_no_brief(self, db) -> None:
        pi = ProactiveIntelligence(db_engine=db)
        _seed_event(
            db, start=_NOW - timedelta(hours=2), attendees="Maria Silva",
        )
        _seed_contact_context(pi, db, name="Maria Silva", situation="s")
        assert pi.get_meeting_prep_briefs(within_hours=24, now=_NOW) == []

    def test_no_mart_returns_empty(self, db) -> None:
        """Absent enriched-events mart (pipeline never ran) → no briefs."""
        pi = ProactiveIntelligence(db_engine=db)
        assert pi.get_meeting_prep_briefs(within_hours=24, now=_NOW) == []

    def test_briefs_sorted_soonest_first(self, db) -> None:
        pi = ProactiveIntelligence(db_engine=db)
        _seed_event(
            db, start=_NOW + timedelta(hours=5),
            attendees="Maria Silva", eid="late", title="Late",
        )
        _seed_event(
            db, start=_NOW + timedelta(hours=1),
            attendees="Maria Silva", eid="soon", title="Soon",
        )
        _seed_contact_context(pi, db, name="Maria Silva", situation="s")

        briefs = pi.get_meeting_prep_briefs(within_hours=24, now=_NOW)
        assert [b.title for b in briefs] == ["Soon", "Late"]
