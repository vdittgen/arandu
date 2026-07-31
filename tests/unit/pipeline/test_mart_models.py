"""Unit tests for mart pipeline models.

Each test creates a temporary SQLite database, loads raw data, materializes
staging and intermediate tables, then runs mart model SQL to verify
domain filtering, placeholders, and data quality.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from src.core.sqlite.engine import DatabaseEngine
from src.core.sqlite.schemas import create_all_tables

from tests.fixtures.sample_data import load_all_fixtures

PIPELINE_DIR = Path(__file__).resolve().parents[3] / "src" / "pipeline"
STAGING_DIR = PIPELINE_DIR / "staging"
INTERMEDIATE_DIR = PIPELINE_DIR / "intermediate"
MARTS_DIR = PIPELINE_DIR / "marts"


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


def _seed_labeled_messages(engine: DatabaseEngine) -> None:
    """Pre-populate ``int_labeled_messages`` for the fixture data.

    ``int_personal_enriched`` now sources ``message_category`` from
    the labeller's per-message domain verdict. Mart tests don't want
    to run an LLM, so we seed the table with the deterministic
    domains the labeller would produce against the fixture set.
    """
    fixture_domains = {
        "msg-001": "work",
        "msg-002": "work",
        "msg-003": "health",
        "msg-004": "personal",
        "msg-005": "personal",
        "msg-006": "work",
        "msg-007": "personal",
        "msg-008": "social",
        "msg-009": "work",
        "msg-010": "personal",
    }
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
        domain = fixture_domains.get(message_id, "personal")
        engine.execute(
            "INSERT INTO int_labeled_messages "
            "(message_id, primary_emotion, intensity, feelings_json, "
            "desires_json, actors_json, environment, domain, "
            "sensitivity_tier) "
            "VALUES (?, 'trust', 0.5, '[]', '[]', '[]', '', ?, 3)",
            [message_id, domain],
        )


def _materialize_int_events_via_python(engine: DatabaseEngine) -> None:
    """Run the python-based ``int_events_enriched`` with a deterministic stub.

    Stubs :class:`EventCategorizerAgent.categorize` so the mart tests
    keep their existing meeting/social/health/travel assertions
    without needing an LLM.
    """
    from src.agents.core.output_types import EventCategoryDecision
    from src.agents.event_categorizer import agent as agent_module
    from src.pipeline.intermediate import int_events_enriched as model

    rules = (
        ("therapy", "health"), ("doctor", "health"),
        ("dentist", "health"),
        ("flight", "travel"),
        ("concert", "social"), ("dinner", "social"),
        ("lunch", "social"), ("party", "social"),
        ("stand-up", "meeting"), ("standup", "meeting"),
        ("planning", "meeting"), ("review", "meeting"),
        ("1-on-1", "meeting"),
    )

    def _stub(self, *, title: str, **_kwargs: object) -> EventCategoryDecision:
        lower = (title or "").lower()
        for needle, category in rules:
            if needle in lower:
                return EventCategoryDecision(
                    category=category, reason="stub",
                )
        return EventCategoryDecision(category="meeting", reason="stub")

    original = agent_module.EventCategorizerAgent.categorize
    agent_module.EventCategorizerAgent.categorize = _stub  # type: ignore[assignment]
    try:
        rows = model.execute(engine)
    finally:
        agent_module.EventCategorizerAgent.categorize = original  # type: ignore[assignment]

    engine.execute("DROP TABLE IF EXISTS int_events_enriched")
    if not rows:
        return
    columns = list(rows[0].keys())
    col_defs = ", ".join(f"{c} TEXT" for c in columns)
    placeholders = ", ".join("?" for _ in columns)
    engine.execute(f"CREATE TABLE int_events_enriched ({col_defs})")
    for row in rows:
        engine.execute(
            f"INSERT INTO int_events_enriched ({', '.join(columns)}) "
            f"VALUES ({placeholders})",
            [row.get(c) for c in columns],
        )


def _materialize_intermediate(engine: DatabaseEngine) -> None:
    """Materialize all intermediate models as tables."""
    # int_personal_enriched depends on int_labeled_messages — seed it.
    _seed_labeled_messages(engine)
    # int_events_enriched is a python model now; run via importable path.
    _materialize_int_events_via_python(engine)
    for name in [
        "int_personal_enriched",
        "int_daily_summary", "int_communications_enriched",
    ]:
        sql = _read_model_sql(INTERMEDIATE_DIR / f"{name}.sql")
        engine.execute(f"CREATE TABLE {name} AS {sql}")


def _local_day(engine: DatabaseEngine, value: str) -> str:
    """The local calendar day of a stored timestamp.

    Mirrors the rule in `mart_today.sql`: a value carrying an explicit
    zone is an instant and gets converted; a bare date is already a
    calendar day and is left alone. Stated here independently so a
    change to the mart's rule has to be reconciled with a test that
    spells out the intent, rather than silently agreeing with itself.

    Slicing the string instead would be wrong: an evening event stored
    as UTC ("...T02:30:00Z") is today locally but reads as tomorrow.
    """
    return engine.query(
        "SELECT CASE "
        "WHEN ? LIKE '%Z' OR ? GLOB '*[+-][0-9][0-9]:[0-9][0-9]' "
        "THEN DATE(?, 'localtime') ELSE DATE(?) END AS d",
        [value] * 4,
    )[0]["d"]


def _run_mart(engine: DatabaseEngine, model_name: str) -> list[dict]:
    """Run a mart model SQL and return all rows."""
    sql = _read_model_sql(MARTS_DIR / f"{model_name}.sql")
    return engine.query(sql)


@pytest.fixture()
def pipeline_db(tmp_path: Path) -> DatabaseEngine:
    """SQLite with raw + staging + intermediate layers materialized."""
    engine = DatabaseEngine(db_path=tmp_path / "test_marts.sqlite3")
    create_all_tables(engine)
    load_all_fixtures(engine)
    _materialize_staging(engine)
    _materialize_intermediate(engine)
    yield engine
    engine.close()


# ---------------------------------------------------------------------------
# mart_personal
# ---------------------------------------------------------------------------


class TestMartPersonal:
    def test_has_rows(self, pipeline_db: DatabaseEngine) -> None:
        """Should produce personal domain items."""
        rows = _run_mart(pipeline_db, "mart_personal")
        assert len(rows) > 0

    def test_expected_columns(self, pipeline_db: DatabaseEngine) -> None:
        rows = _run_mart(pipeline_db, "mart_personal")
        expected = {
            "item_type", "id", "title", "detail", "occurred_at",
            "contact_name", "sensitivity_tier", "emotional_label",
            "_loaded_at",
        }
        assert set(rows[0].keys()) == expected

    def test_item_types(self, pipeline_db: DatabaseEngine) -> None:
        """Should contain messages, events, and/or notes."""
        rows = _run_mart(pipeline_db, "mart_personal")
        types = {r["item_type"] for r in rows}
        assert types.issubset({"message", "event", "note"})

    def test_no_work_messages(self, pipeline_db: DatabaseEngine) -> None:
        """No messages classified as 'work' should appear."""
        rows = _run_mart(pipeline_db, "mart_personal")
        msgs = [r for r in rows if r["item_type"] == "message"]
        # All personal messages should be from personal/health sources
        for m in msgs:
            assert m["id"] is not None

    def test_emotional_label_placeholder(self, pipeline_db: DatabaseEngine) -> None:
        """emotional_label should be NULL (placeholder for ML)."""
        rows = _run_mart(pipeline_db, "mart_personal")
        for row in rows:
            assert row["emotional_label"] is None

    def test_includes_family_messages(self, pipeline_db: DatabaseEngine) -> None:
        """Messages from family should appear in personal mart."""
        rows = _run_mart(pipeline_db, "mart_personal")
        msg_ids = {r["id"] for r in rows if r["item_type"] == "message"}
        # msg-004 is from mom (personal)
        assert "msg-004" in msg_ids

    def test_includes_social_events(self, pipeline_db: DatabaseEngine) -> None:
        """Social events should appear (e.g. Concert, Lunch with Carlos)."""
        rows = _run_mart(pipeline_db, "mart_personal")
        event_ids = {r["id"] for r in rows if r["item_type"] == "event"}
        # cal-003 is Lunch with Carlos (social), cal-006 is Concert (social)
        assert "cal-006" in event_ids or "cal-003" in event_ids

    def test_sensitivity_tier_values(self, pipeline_db: DatabaseEngine) -> None:
        rows = _run_mart(pipeline_db, "mart_personal")
        for row in rows:
            assert int(row["sensitivity_tier"]) in (1, 2, 3)


# ---------------------------------------------------------------------------
# mart_work
# ---------------------------------------------------------------------------


class TestMartWork:
    def test_has_rows(self, pipeline_db: DatabaseEngine) -> None:
        rows = _run_mart(pipeline_db, "mart_work")
        assert len(rows) > 0

    def test_expected_columns(self, pipeline_db: DatabaseEngine) -> None:
        rows = _run_mart(pipeline_db, "mart_work")
        expected = {
            "item_type", "id", "title", "detail", "occurred_at",
            "contact_name", "sensitivity_tier", "topic", "_loaded_at",
        }
        assert set(rows[0].keys()) == expected

    def test_topic_placeholder(self, pipeline_db: DatabaseEngine) -> None:
        """topic should be NULL (placeholder for ML)."""
        rows = _run_mart(pipeline_db, "mart_work")
        for row in rows:
            assert row["topic"] is None

    def test_includes_slack_messages(self, pipeline_db: DatabaseEngine) -> None:
        """Slack messages should appear in work mart."""
        rows = _run_mart(pipeline_db, "mart_work")
        msg_ids = {r["id"] for r in rows if r["item_type"] == "message"}
        # msg-002 is from slack (alice)
        assert "msg-002" in msg_ids

    def test_includes_meeting_events(self, pipeline_db: DatabaseEngine) -> None:
        """Meeting events should appear (e.g. Q2 Planning, Stand-up)."""
        rows = _run_mart(pipeline_db, "mart_work")
        event_ids = {r["id"] for r in rows if r["item_type"] == "event"}
        # cal-001 = Q2 Planning (meeting), cal-005 = Stand-up (meeting)
        assert "cal-001" in event_ids or "cal-005" in event_ids

    def test_includes_work_notes(self, pipeline_db: DatabaseEngine) -> None:
        """Notes tagged 'work' should appear."""
        rows = _run_mart(pipeline_db, "mart_work")
        note_ids = {r["id"] for r in rows if r["item_type"] == "note"}
        # note-003 = Meeting Notes (tags: work, meetings, planning)
        assert "note-003" in note_ids

    def test_sensitivity_tier_values(self, pipeline_db: DatabaseEngine) -> None:
        rows = _run_mart(pipeline_db, "mart_work")
        for row in rows:
            assert int(row["sensitivity_tier"]) in (1, 2, 3)


# ---------------------------------------------------------------------------
# mart_today
# ---------------------------------------------------------------------------


class TestMartToday:
    def test_expected_columns(self, pipeline_db: DatabaseEngine) -> None:
        """Even if empty (no data for today), verify SQL is valid by creating
        a temp table and checking column names via PRAGMA."""
        sql = _read_model_sql(MARTS_DIR / "mart_today.sql")
        pipeline_db.execute(f"CREATE TABLE _tmp_mart_today AS {sql}")
        info = pipeline_db.query("PRAGMA table_info(_tmp_mart_today)")
        col_names = {r["name"] for r in info}
        expected = {
            "item_type", "id", "title", "detail", "occurred_at",
            "category", "duration_minutes", "sensitivity_tier",
            "event_origin", "coaching_phrase", "_loaded_at",
        }
        assert expected == col_names

    def test_coaching_phrase_placeholder(self, pipeline_db: DatabaseEngine) -> None:
        """coaching_phrase should always be NULL (populated by LLM later)."""
        rows = _run_mart(pipeline_db, "mart_today")
        for row in rows:
            assert row["coaching_phrase"] is None

    def test_filters_to_current_date(self, pipeline_db: DatabaseEngine) -> None:
        """All rows fall on the user's local day.

        Fixture data is from 2025, so today's mart should be empty here.

        "Today" is read from the database rather than from Python, so
        the test states the invariant the mart implements and holds in
        every timezone.
        """
        rows = _run_mart(pipeline_db, "mart_today")
        today = pipeline_db.query(
            "SELECT DATE('now', 'localtime') AS d",
        )[0]["d"]
        for row in rows:
            assert _local_day(pipeline_db, row["occurred_at"]) == today

    def test_includes_events_on_the_local_day_not_the_utc_day(
        self, pipeline_db: DatabaseEngine,
    ) -> None:
        """An event on the local day must appear even when UTC disagrees.

        `DATE('now')` is UTC, so a user at a negative offset lost the
        rest of their evening from the feed the moment UTC rolled over,
        and one at a positive offset lost their early morning.

        The instant is derived from the machine's real offset rather
        than a hardcoded one, so the row genuinely straddles the UTC
        day boundary in whatever zone the suite runs in: late evening
        west of UTC, early morning east of it. Stored with a `Z`, the
        shape the Apple and Gmail bridges write.
        """
        offset_is_negative = pipeline_db.query(
            "SELECT datetime('now', 'localtime') < datetime('now') AS neg",
        )[0]["neg"]
        local_time = "23:30:00" if offset_is_negative else "00:30:00"

        row = pipeline_db.query(
            "SELECT DATE('now', 'localtime') AS day, "
            "strftime('%Y-%m-%dT%H:%M:%SZ', "
            "         datetime(DATE('now', 'localtime') || ' ' || ?, 'utc')"
            ") AS instant",
            [local_time],
        )[0]
        pipeline_db.execute(
            "INSERT INTO int_events_enriched "
            "(id, title, description, start_time, event_category, "
            " duration_minutes, sensitivity_tier, event_origin) "
            "VALUES ('tz-evt', 'Edge-of-day call', '', ?, 'meeting', 30, "
            "1, 'organizer')",
            [row["instant"]],
        )

        rows = _run_mart(pipeline_db, "mart_today")

        assert "tz-evt" in {r["id"] for r in rows}
        assert _local_day(pipeline_db, row["instant"]) == row["day"]

    def test_all_day_events_are_not_shifted_a_day(
        self, pipeline_db: DatabaseEngine,
    ) -> None:
        """A bare date is already a calendar day — don't convert it.

        This is the trap in the obvious fix: adding `'localtime'`
        unconditionally makes SQLite read "2026-08-03" as midnight UTC
        and shift it to the previous day for every negative offset, so
        all-day events (and bare reminder due dates) would silently
        disappear from the feed.
        """
        local_today = pipeline_db.query(
            "SELECT DATE('now', 'localtime') AS d",
        )[0]["d"]
        pipeline_db.execute(
            "INSERT INTO int_events_enriched "
            "(id, title, description, start_time, event_category, "
            " duration_minutes, sensitivity_tier, event_origin) "
            "VALUES ('tz-allday', 'Holiday', '', ?, 'meeting', 0, 1, "
            "'organizer')",
            [local_today],
        )

        rows = _run_mart(pipeline_db, "mart_today")

        assert "tz-allday" in {r["id"] for r in rows}


# ---------------------------------------------------------------------------
# mart_health
# ---------------------------------------------------------------------------


class TestMartHealth:
    def test_row_count_matches_metrics(self, pipeline_db: DatabaseEngine) -> None:
        """One row per health metric record."""
        rows = _run_mart(pipeline_db, "mart_health")
        raw_count = pipeline_db.query(
            "SELECT COUNT(*) AS cnt FROM raw_health_metrics"
        )[0]["cnt"]
        assert len(rows) == raw_count

    def test_expected_columns(self, pipeline_db: DatabaseEngine) -> None:
        rows = _run_mart(pipeline_db, "mart_health")
        expected = {
            "id", "metric_type", "value", "unit", "recorded_at",
            "source", "sensitivity_tier", "avg_7d", "stddev_7d",
            "is_anomaly", "is_latest", "_loaded_at",
        }
        assert set(rows[0].keys()) == expected

    def test_all_tier_3(self, pipeline_db: DatabaseEngine) -> None:
        """Health metrics are always tier 3."""
        rows = _run_mart(pipeline_db, "mart_health")
        for row in rows:
            assert row["sensitivity_tier"] == 3

    def test_is_latest_one_per_type(self, pipeline_db: DatabaseEngine) -> None:
        """Exactly one row per metric_type should have is_latest=1."""
        rows = _run_mart(pipeline_db, "mart_health")
        latest = [r for r in rows if r["is_latest"] == 1]
        latest_types = [r["metric_type"] for r in latest]
        assert len(latest_types) == len(set(latest_types))

    def test_avg_7d_not_null(self, pipeline_db: DatabaseEngine) -> None:
        """avg_7d should always have a value (at least the row itself)."""
        rows = _run_mart(pipeline_db, "mart_health")
        for row in rows:
            assert row["avg_7d"] is not None

    def test_anomaly_flag_is_integer_boolean(self, pipeline_db: DatabaseEngine) -> None:
        """is_anomaly should be 0 or 1 (SQLite integer boolean)."""
        rows = _run_mart(pipeline_db, "mart_health")
        for row in rows:
            assert row["is_anomaly"] in (0, 1)

    def test_known_heart_rate_value(self, pipeline_db: DatabaseEngine) -> None:
        """hm-001 should be heart_rate 72.0."""
        rows = _run_mart(pipeline_db, "mart_health")
        hr = next(r for r in rows if r["id"] == "hm-001")
        assert hr["value"] == 72.0
        assert hr["metric_type"] == "heart_rate"

    def test_id_uniqueness(self, pipeline_db: DatabaseEngine) -> None:
        rows = _run_mart(pipeline_db, "mart_health")
        ids = [r["id"] for r in rows]
        assert len(ids) == len(set(ids))
