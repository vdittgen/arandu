"""Unit tests for intermediate pipeline models.

Each test creates a temporary SQLite database, loads raw data, materializes the
staging tables, then runs intermediate model SQL to verify enrichment,
aggregation, and data quality.

``int_personal_enriched`` now joins ``int_labeled_messages.domain`` to set
``message_category``; the tests seed that table directly with deterministic
domains so we don't need an LLM during testing.

``int_events_enriched`` is a Python intermediate driven by
:class:`EventCategorizerAgent`; the tests monkey-patch the agent's
``categorize`` method so categorisation is deterministic and offline.
"""

from __future__ import annotations

import re
import typing as t
from pathlib import Path

import pytest
from src.core.sqlite.engine import DatabaseEngine
from src.core.sqlite.schemas import create_all_tables

from tests.fixtures.sample_data import load_all_fixtures

PIPELINE_DIR = Path(__file__).resolve().parents[3] / "src" / "pipeline"
STAGING_DIR = PIPELINE_DIR / "staging"
INTERMEDIATE_DIR = PIPELINE_DIR / "intermediate"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_model_sql(model_path: Path) -> str:
    """Read the SQL SELECT/WITH statement from a pipeline model file."""
    text = model_path.read_text()
    match = re.search(r"(?m)^(SELECT|WITH)\b", text)
    if match:
        return text[match.start() :].strip()
    msg = f"Could not find SELECT/WITH in {model_path}"
    raise ValueError(msg)


def _materialize_staging(engine: DatabaseEngine) -> None:
    """Materialize all staging models as tables."""
    for name in [
        "stg_messages", "stg_calendar_events", "stg_notes",
        "stg_health_metrics", "stg_contacts",
        "stg_emails", "stg_reminders",
    ]:
        sql = _read_model_sql(STAGING_DIR / f"{name}.sql")
        engine.execute(f"CREATE TABLE {name} AS {sql}")


# Map of fixture message_id → domain that the seeded labeller table
# should report. Mirrors the verdicts the LabelerAgent would produce
# against the fixture data, but is deterministic and offline.
_FIXTURE_MESSAGE_DOMAINS: dict[str, str] = {
    "msg-001": "work",      # boss@company.com — planning session
    "msg-002": "work",      # alice@company.com via Slack — PR review
    "msg-003": "health",    # doctor@healthclinic.com — follow-up
    "msg-004": "personal",  # mom via iMessage
    "msg-005": "personal",  # bank statement
    "msg-006": "work",      # bob@company.com via Slack — deploy
    "msg-007": "personal",  # newsletter
    "msg-008": "social",    # carlos via iMessage — concert invite
    "msg-009": "work",      # recruiter
    "msg-010": "personal",  # mom via iMessage — birthday
}


def _seed_labeled_messages(engine: DatabaseEngine) -> None:
    """Populate ``int_labeled_messages`` with deterministic domains.

    The real intermediate is itself an LLM-driven Python model; we
    bypass it here so message-category tests stay hermetic.
    """
    engine.execute("DROP TABLE IF EXISTS int_labeled_messages")
    engine.execute("""
        CREATE TABLE int_labeled_messages (
            message_id      TEXT PRIMARY KEY,
            primary_emotion TEXT,
            intensity       REAL,
            feelings_json   TEXT,
            desires_json    TEXT,
            actors_json     TEXT,
            environment     TEXT,
            domain          TEXT,
            sensitivity_tier INTEGER
        )
    """)
    rows = engine.query("SELECT id FROM stg_messages")
    for row in rows:
        message_id = str(row["id"])
        domain = _FIXTURE_MESSAGE_DOMAINS.get(message_id, "personal")
        engine.execute(
            "INSERT INTO int_labeled_messages "
            "(message_id, primary_emotion, intensity, feelings_json, "
            "desires_json, actors_json, environment, domain, "
            "sensitivity_tier) "
            "VALUES (?, 'trust', 0.5, '[]', '[]', '[]', '', ?, 3)",
            [message_id, domain],
        )


def _run_intermediate(engine: DatabaseEngine, model_name: str) -> list[dict]:
    """Run an intermediate model SQL and return all rows."""
    sql = _read_model_sql(INTERMEDIATE_DIR / f"{model_name}.sql")
    return engine.query(sql)


@pytest.fixture()
def pipeline_db(tmp_path: Path) -> DatabaseEngine:
    """SQLite with raw data + materialized staging tables."""
    engine = DatabaseEngine(db_path=tmp_path / "test_intermediate.sqlite3")
    create_all_tables(engine)
    load_all_fixtures(engine)
    _materialize_staging(engine)
    yield engine
    engine.close()


@pytest.fixture()
def labeled_db(pipeline_db: DatabaseEngine) -> DatabaseEngine:
    """Staging + seeded ``int_labeled_messages`` for message tests."""
    _seed_labeled_messages(pipeline_db)
    return pipeline_db


# ---------------------------------------------------------------------------
# int_personal_enriched
# ---------------------------------------------------------------------------


class TestIntPersonalEnriched:
    def test_row_count_matches_messages(
        self, labeled_db: DatabaseEngine,
    ) -> None:
        """Should have exactly one row per message (LEFT JOIN preserves all)."""
        rows = _run_intermediate(labeled_db, "int_personal_enriched")
        raw_count = labeled_db.query(
            "SELECT COUNT(*) AS cnt FROM raw_messages",
        )[0]["cnt"]
        assert len(rows) == raw_count

    def test_expected_columns(
        self, labeled_db: DatabaseEngine,
    ) -> None:
        rows = _run_intermediate(labeled_db, "int_personal_enriched")
        expected = {
            "id", "source", "sender", "recipient", "content",
            "timestamp", "sensitivity_tier", "message_length",
            "contact_name", "relationship", "message_category",
            "_loaded_at",
        }
        assert set(rows[0].keys()) == expected

    def test_category_values(
        self, labeled_db: DatabaseEngine,
    ) -> None:
        """message_category must be one of the defined categories."""
        rows = _run_intermediate(labeled_db, "int_personal_enriched")
        valid = {"personal", "work", "health"}
        for row in rows:
            assert row["message_category"] in valid

    def test_work_labelled_messages_routed_to_work(
        self, labeled_db: DatabaseEngine,
    ) -> None:
        """Messages the labeller tags ``work`` should land in work."""
        rows = _run_intermediate(labeled_db, "int_personal_enriched")
        by_id = {r["id"]: r for r in rows}
        for message_id, domain in _FIXTURE_MESSAGE_DOMAINS.items():
            if domain != "work":
                continue
            assert message_id in by_id, message_id
            assert by_id[message_id]["message_category"] == "work"

    def test_personal_labelled_messages_routed_to_personal(
        self, labeled_db: DatabaseEngine,
    ) -> None:
        """Personal-domain messages flow into the personal bucket."""
        rows = _run_intermediate(labeled_db, "int_personal_enriched")
        mom_msgs = [r for r in rows if r["sender"] == "mom"]
        assert mom_msgs, "fixture should have at least one mom message"
        for row in mom_msgs:
            assert row["message_category"] == "personal"

    def test_social_labelled_messages_collapse_to_personal(
        self, labeled_db: DatabaseEngine,
    ) -> None:
        """``social`` from the labeller maps to ``personal`` here.

        The ``mart_personal`` mart unions message rows with social
        events, so collapsing social → personal at this layer keeps
        the message stream consistent with the audit vocabulary
        (``personal``/``work``/``health``).
        """
        rows = _run_intermediate(labeled_db, "int_personal_enriched")
        carlos_msgs = [
            r for r in rows if r["sender"] == "best_friend_carlos"
        ]
        assert carlos_msgs, "fixture should include carlos messages"
        for row in carlos_msgs:
            assert row["message_category"] == "personal"

    def test_contact_name_resolved(
        self, labeled_db: DatabaseEngine,
    ) -> None:
        """Messages from known contacts should have contact_name filled."""
        rows = _run_intermediate(labeled_db, "int_personal_enriched")
        alice_msgs = [
            r for r in rows if r["sender"] == "alice@company.com"
        ]
        assert len(alice_msgs) > 0
        assert alice_msgs[0]["contact_name"] == "Alice Kim"

    def test_health_labelled_messages_routed_to_health(
        self, labeled_db: DatabaseEngine,
    ) -> None:
        """Health-domain labels land in the health bucket."""
        rows = _run_intermediate(labeled_db, "int_personal_enriched")
        health = [
            r for r in rows
            if _FIXTURE_MESSAGE_DOMAINS.get(r["id"]) == "health"
        ]
        assert health, "fixture should include at least one health-labelled message"
        for row in health:
            assert row["message_category"] == "health"

    def test_sensitivity_tier_values(
        self, labeled_db: DatabaseEngine,
    ) -> None:
        rows = _run_intermediate(labeled_db, "int_personal_enriched")
        for row in rows:
            assert row["sensitivity_tier"] in (1, 2, 3)

    def test_id_uniqueness(
        self, labeled_db: DatabaseEngine,
    ) -> None:
        rows = _run_intermediate(labeled_db, "int_personal_enriched")
        ids = [r["id"] for r in rows]
        assert len(ids) == len(set(ids))

    def test_unlabelled_messages_default_to_personal(
        self, labeled_db: DatabaseEngine,
    ) -> None:
        """Messages without a labeller verdict fall back to ``personal``.

        Simulates the dominant real-world case: int_labeled_messages
        marks most older messages as ``unlabeled`` with no domain
        signal, and the COALESCE/CASE chain must not leak ``other``.
        """
        labeled_db.execute("DELETE FROM int_labeled_messages")
        rows = _run_intermediate(labeled_db, "int_personal_enriched")
        for row in rows:
            assert row["message_category"] == "personal"


# ---------------------------------------------------------------------------
# int_events_enriched (Python intermediate, agent-driven)
# ---------------------------------------------------------------------------


# Title fragment → category — matches the fixture event titles. The
# stub keeps the keyword shape of the legacy SQL so the existing
# assertions still describe a meaningful contract, but the contract
# now sits behind the agent boundary instead of inside the model.
_STUB_RULES = (
    ("therapy", "health"),
    ("doctor", "health"),
    ("dentist", "health"),
    ("flight", "travel"),
    ("concert", "social"),
    ("dinner", "social"),
    ("lunch", "social"),
    ("party", "social"),
    ("stand-up", "meeting"),
    ("standup", "meeting"),
    ("planning", "meeting"),
    ("review", "meeting"),
    ("1-on-1", "meeting"),
)


def _stub_categorize(*, title: str, **_kwargs: t.Any) -> t.Any:
    """Deterministic stand-in for the agent's ``categorize`` method.

    Falls back to ``meeting`` for ambiguous calendar entries, matching
    the production fallback used when the LLM is unavailable.
    """
    from src.agents.core.output_types import EventCategoryDecision
    lower = (title or "").lower()
    for needle, category in _STUB_RULES:
        if needle in lower:
            return EventCategoryDecision(
                category=category, reason="stub",
            )
    return EventCategoryDecision(category="meeting", reason="stub")


def _run_events_intermediate(
    engine: DatabaseEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict]:
    """Run the events intermediate with the agent stubbed out."""
    from src.agents.event_categorizer import agent as agent_module
    from src.pipeline.intermediate import int_events_enriched as model

    monkeypatch.setattr(
        agent_module.EventCategorizerAgent,
        "categorize",
        lambda self, **kwargs: _stub_categorize(**kwargs),
    )
    return model.execute(engine)


class TestIntEventsEnriched:
    def test_row_count_matches_events(
        self, pipeline_db: DatabaseEngine,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        rows = _run_events_intermediate(pipeline_db, monkeypatch)
        raw_count = pipeline_db.query(
            "SELECT COUNT(*) AS cnt FROM raw_calendar_events"
        )[0]["cnt"]
        assert len(rows) == raw_count

    def test_expected_columns(
        self, pipeline_db: DatabaseEngine,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        rows = _run_events_intermediate(pipeline_db, monkeypatch)
        expected = {
            "id", "title", "description", "start_time", "end_time",
            "location", "attendees", "attendees_count", "duration_minutes",
            "sensitivity_tier",
            "calendar_name", "calendar_owner_email",
            "is_shared_calendar", "is_subscribed_calendar",
            "self_response_status", "event_origin",
            "known_attendee_names",
            "attendee_relationships", "event_category", "is_recurring",
            "_loaded_at",
        }
        assert set(rows[0].keys()) == expected

    def test_category_values(
        self, pipeline_db: DatabaseEngine,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        rows = _run_events_intermediate(pipeline_db, monkeypatch)
        valid = {"meeting", "social", "health", "travel", "other"}
        for row in rows:
            assert row["event_category"] in valid

    def test_therapy_classified_as_health(
        self, pipeline_db: DatabaseEngine,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        rows = _run_events_intermediate(pipeline_db, monkeypatch)
        therapy = next(r for r in rows if r["id"] == "cal-004")
        assert therapy["event_category"] == "health"

    def test_standup_classified_as_meeting(
        self, pipeline_db: DatabaseEngine,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        rows = _run_events_intermediate(pipeline_db, monkeypatch)
        standup = next(r for r in rows if r["id"] == "cal-005")
        assert standup["event_category"] == "meeting"

    def test_concert_classified_as_social(
        self, pipeline_db: DatabaseEngine,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        rows = _run_events_intermediate(pipeline_db, monkeypatch)
        concert = next(r for r in rows if r["id"] == "cal-006")
        assert concert["event_category"] == "social"

    def test_flight_classified_as_travel(
        self, pipeline_db: DatabaseEngine,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        rows = _run_events_intermediate(pipeline_db, monkeypatch)
        flight = next(r for r in rows if r["id"] == "cal-008")
        assert flight["event_category"] == "travel"

    def test_recurring_flag_on_standup(
        self, pipeline_db: DatabaseEngine,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        rows = _run_events_intermediate(pipeline_db, monkeypatch)
        standup = next(r for r in rows if r["id"] == "cal-005")
        assert standup["is_recurring"] == 1

    def test_recurring_flag_off_for_concert(
        self, pipeline_db: DatabaseEngine,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        rows = _run_events_intermediate(pipeline_db, monkeypatch)
        concert = next(r for r in rows if r["id"] == "cal-006")
        assert concert["is_recurring"] == 0

    def test_known_attendees_resolved(
        self, pipeline_db: DatabaseEngine,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        rows = _run_events_intermediate(pipeline_db, monkeypatch)
        dinner = next(r for r in rows if r["id"] == "cal-009")
        assert dinner["known_attendee_names"] is not None
        assert "Mom" in dinner["known_attendee_names"]

    def test_sensitivity_tier_values(
        self, pipeline_db: DatabaseEngine,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        rows = _run_events_intermediate(pipeline_db, monkeypatch)
        for row in rows:
            assert int(row["sensitivity_tier"]) in (1, 2, 3)

    def test_id_uniqueness(
        self, pipeline_db: DatabaseEngine,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        rows = _run_events_intermediate(pipeline_db, monkeypatch)
        ids = [r["id"] for r in rows]
        assert len(ids) == len(set(ids))

    def test_falls_back_to_meeting_when_agent_returns_none(
        self, pipeline_db: DatabaseEngine,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Agent failure must not abort the run — emit fallback rows."""
        from src.agents.event_categorizer import agent as agent_module
        from src.pipeline.intermediate import int_events_enriched as model

        monkeypatch.setattr(
            agent_module.EventCategorizerAgent,
            "categorize",
            lambda self, **kwargs: None,
        )
        rows = model.execute(pipeline_db)
        assert rows, "should still emit rows when LLM unavailable"
        for row in rows:
            assert row["event_category"] == "meeting"

    def test_cache_short_circuits_second_run(
        self, pipeline_db: DatabaseEngine,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A second run with identical events must not call the agent."""
        from src.agents.event_categorizer import agent as agent_module
        from src.pipeline.intermediate import int_events_enriched as model

        calls = {"count": 0}

        def _counting(*, title: str, **kwargs: t.Any) -> t.Any:
            calls["count"] += 1
            return _stub_categorize(title=title, **kwargs)

        monkeypatch.setattr(
            agent_module.EventCategorizerAgent,
            "categorize",
            lambda self, **kwargs: _counting(**kwargs),
        )
        model.execute(pipeline_db)
        first_run = calls["count"]
        assert first_run > 0
        model.execute(pipeline_db)
        assert calls["count"] == first_run, (
            "second run should be served entirely from cache"
        )


# ---------------------------------------------------------------------------
# int_daily_summary
# ---------------------------------------------------------------------------


class TestIntDailySummary:
    def test_has_rows(self, pipeline_db: DatabaseEngine) -> None:
        """Should produce at least one summary row."""
        rows = _run_intermediate(pipeline_db, "int_daily_summary")
        assert len(rows) > 0

    def test_expected_columns(self, pipeline_db: DatabaseEngine) -> None:
        rows = _run_intermediate(pipeline_db, "int_daily_summary")
        expected = {
            "summary_date", "message_count", "avg_message_length",
            "event_count", "total_meeting_hours", "notes_created",
            "email_count", "reminder_count", "overdue_reminders",
            "latest_heart_rate", "latest_steps", "latest_sleep_hours",
            "latest_weight_kg", "sensitivity_tier", "_loaded_at",
        }
        assert set(rows[0].keys()) == expected

    def test_unique_dates(self, pipeline_db: DatabaseEngine) -> None:
        """Each date should appear exactly once."""
        rows = _run_intermediate(pipeline_db, "int_daily_summary")
        dates = [r["summary_date"] for r in rows]
        assert len(dates) == len(set(dates))

    def test_counts_non_negative(self, pipeline_db: DatabaseEngine) -> None:
        """Counts and hours must be >= 0."""
        rows = _run_intermediate(pipeline_db, "int_daily_summary")
        for row in rows:
            assert row["message_count"] >= 0
            assert row["event_count"] >= 0
            assert row["notes_created"] >= 0
            assert row["total_meeting_hours"] >= 0
            assert row["email_count"] >= 0
            assert row["reminder_count"] >= 0
            assert row["overdue_reminders"] >= 0

    def test_june_2_has_messages(self, pipeline_db: DatabaseEngine) -> None:
        """June 2, 2025 has 2 messages in fixture data."""
        rows = _run_intermediate(pipeline_db, "int_daily_summary")
        # SQLite DATE() returns strings like '2025-06-02'
        june2 = next(
            (r for r in rows if r["summary_date"] == "2025-06-02"),
            None,
        )
        assert june2 is not None
        assert june2["message_count"] == 2

    def test_health_metrics_on_june_1(self, pipeline_db: DatabaseEngine) -> None:
        """June 1 has heart_rate=72 and steps=9854 in fixtures."""
        rows = _run_intermediate(pipeline_db, "int_daily_summary")
        june1 = next(
            (r for r in rows if r["summary_date"] == "2025-06-01"),
            None,
        )
        assert june1 is not None
        assert june1["latest_heart_rate"] == 72.0
        assert june1["latest_steps"] == 9854.0

    def test_all_tier_3(self, pipeline_db: DatabaseEngine) -> None:
        """Daily summary includes health data, so sensitivity_tier should be 3."""
        rows = _run_intermediate(pipeline_db, "int_daily_summary")
        for row in rows:
            assert row["sensitivity_tier"] == 3


class TestIntEventsEnrichedCircuitBreaker:
    """A failing provider must not re-fire the whole backlog every run.

    Failures are deliberately not cached so a transient glitch retries.
    Without a breaker that turns a *persistent* failure into one paid
    LLM call per uncached event, on every run, indefinitely — observed
    in production as batches of exactly 70 identical 404s.
    """

    @staticmethod
    def _run_with_failing_agent(
        engine: DatabaseEngine,
        monkeypatch: pytest.MonkeyPatch,
        *,
        fail_all: bool = True,
    ) -> tuple[list[dict], int]:
        """Run the model with a categorizer that raises. Returns rows + calls."""
        from src.agents.event_categorizer import agent as agent_module
        from src.pipeline.intermediate import int_events_enriched as model

        calls = {"n": 0}

        def _boom(self: t.Any, **_kwargs: t.Any) -> t.Any:
            calls["n"] += 1
            msg = "model 'arandu-pro' not found"
            raise RuntimeError(msg)

        monkeypatch.setattr(
            agent_module.EventCategorizerAgent, "categorize", _boom,
        )
        rows = model.execute(engine)
        return rows, calls["n"]

    def test_breaker_caps_calls_when_the_provider_is_down(
        self, pipeline_db: DatabaseEngine,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from src.pipeline.intermediate import int_events_enriched as model

        rows, calls = self._run_with_failing_agent(pipeline_db, monkeypatch)
        # Every event still gets a row — the marts must not go hungry.
        assert rows
        # ...but the LLM was called only until the breaker tripped.
        assert calls == model._CONSECUTIVE_FAILURE_LIMIT, (
            f"expected the breaker to stop after "
            f"{model._CONSECUTIVE_FAILURE_LIMIT} consecutive failures, "
            f"but the agent was called {calls} times"
        )

    def test_every_row_falls_back_when_breaker_trips(
        self, pipeline_db: DatabaseEngine,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from src.pipeline.intermediate import int_events_enriched as model

        rows, _ = self._run_with_failing_agent(pipeline_db, monkeypatch)
        assert all(
            r["event_category"] == model._FALLBACK_CATEGORY for r in rows
        )

    def test_failures_are_still_not_cached(
        self, pipeline_db: DatabaseEngine,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A recovered provider must re-categorise, not inherit fallbacks."""
        from src.pipeline.intermediate import int_events_enriched as model

        self._run_with_failing_agent(pipeline_db, monkeypatch)
        rows = _run_events_intermediate(pipeline_db, monkeypatch)
        assert any(
            r["event_category"] != model._FALLBACK_CATEGORY for r in rows
        ), "after recovery the events should be categorised, not stuck"

    def test_isolated_failure_does_not_trip_the_breaker(
        self, pipeline_db: DatabaseEngine,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """One bad event is the event's problem, not the provider's."""
        from src.agents.core.output_types import EventCategoryDecision
        from src.agents.event_categorizer import agent as agent_module
        from src.pipeline.intermediate import int_events_enriched as model

        calls = {"n": 0}

        def _flaky(self: t.Any, **kwargs: t.Any) -> t.Any:
            calls["n"] += 1
            if calls["n"] == 1:  # first event fails, rest succeed
                msg = "transient"
                raise RuntimeError(msg)
            return EventCategoryDecision(category="meeting", reason="ok")

        monkeypatch.setattr(
            agent_module.EventCategorizerAgent, "categorize", _flaky,
        )
        rows = model.execute(pipeline_db)
        assert calls["n"] > model._CONSECUTIVE_FAILURE_LIMIT, (
            "a single failure surrounded by successes must not trip the "
            "breaker — consecutive failures are the signal"
        )
        assert rows
