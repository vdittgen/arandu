"""Tests for ProactiveIntelligence.

Mocks LLMProvider to avoid requiring a running Ollama instance.
Uses a real temp DuckDB for table operations.

sensitivity_tier: N/A — test infrastructure
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from src.agents.proactive import (
    ActionableEvent,
    ContactContext,
    PendingReply,
    ProactiveIntelligence,
    ProactiveResult,
)
from src.agents.proactive.persistence import _topic_boost_importance
from src.core.db_helpers import make_hash_id, safe_str
from src.core.llm_helpers import parse_llm_json_array
from src.core.sqlite.engine import DatabaseEngine

# ================================================================
# Fixtures
# ================================================================


@pytest.fixture()
def tmp_db(tmp_path: Path) -> DatabaseEngine:
    """Fresh DuckDB engine backed by a temp file."""
    db_path = tmp_path / "test_proactive.duckdb"
    engine = DatabaseEngine(db_path=db_path)
    yield engine
    engine.close()


@pytest.fixture()
def stub_pending_reply(monkeypatch):
    """Monkey-patch ``PendingReplyAgent.detect`` with an empty default."""
    from src.agents.core.output_types import PendingReplyBatch
    fake = MagicMock(return_value=PendingReplyBatch(replies=[]))

    def _bound(self, *, messages, topics=None):  # noqa: ARG001
        result = fake(messages=messages, topics=topics)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(
        "src.agents.pending_reply.agent.PendingReplyAgent.detect", _bound,
    )
    return fake


@pytest.fixture()
def stub_contact_context(monkeypatch):
    """Monkey-patch ``ContactContextAgent.summarize`` with an empty default."""
    from src.agents.core.output_types import ContactContextBatch
    fake = MagicMock(return_value=ContactContextBatch(contexts=[]))

    def _bound(self, *, contacts, topics=None):  # noqa: ARG001
        result = fake(contacts=contacts, topics=topics)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(
        "src.agents.contact_context.agent.ContactContextAgent.summarize",
        _bound,
    )
    return fake


@pytest.fixture()
def stub_actionable_events(monkeypatch):
    """Monkey-patch ``ActionableEventsAgent.detect`` with an empty default."""
    from src.agents.core.output_types import ActionableEventBatch
    fake = MagicMock(return_value=ActionableEventBatch(events=[]))

    def _bound(self, *, events):  # noqa: ARG001
        result = fake(events=events)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(
        "src.agents.actionable_events.agent.ActionableEventsAgent.detect",
        _bound,
    )
    return fake


@pytest.fixture()
def proactive(
    tmp_db: DatabaseEngine,
    stub_pending_reply,  # noqa: ARG001 (auto-applied)
    stub_contact_context,  # noqa: ARG001
    stub_actionable_events,  # noqa: ARG001
) -> ProactiveIntelligence:
    """ProactiveIntelligence with stubbed SBAgents."""
    return ProactiveIntelligence(db_engine=tmp_db)


@pytest.fixture()
def proactive_readonly(
    tmp_db: DatabaseEngine,
) -> ProactiveIntelligence:
    """ProactiveIntelligence without SBAgent stubs (persistence-only)."""
    return ProactiveIntelligence(db_engine=tmp_db)


def _seed_messages(db: DatabaseEngine) -> None:
    """Create raw_messages table with test data."""
    db.execute("""
        CREATE TABLE IF NOT EXISTS raw_messages (
            id VARCHAR PRIMARY KEY,
            source VARCHAR,
            sender VARCHAR,
            sender_name VARCHAR,
            content TEXT,
            timestamp TEXT,
            is_from_me INTEGER,
            chat_name VARCHAR,
            is_group INTEGER,
            sensitivity_tier INTEGER DEFAULT 2
        )
    """)
    # Unreplied messages — Father has 3 msgs to pass activity filters
    db.execute("""
        INSERT INTO raw_messages
        (id, source, sender, sender_name, content, timestamp,
         is_from_me, chat_name, is_group)
        VALUES
        ('msg1', 'whatsapp', '5548@s.whatsapp.net', 'Father',
         'How did the doctor appointment go?',
         datetime('now', '-1 hour'),
         0, 'Father', 0),
        ('msg1b', 'whatsapp', '5548@s.whatsapp.net', 'Father',
         'Also, your mother is worried.',
         datetime('now', '-3 hours'),
         0, 'Father', 0),
        ('msg1c', 'whatsapp', '5548@s.whatsapp.net', 'Father',
         'Call me when you can.',
         datetime('now', '-5 hours'),
         0, 'Father', 0),
        ('msg2', 'whatsapp', '5549@s.whatsapp.net', 'Friend',
         'ok thanks',
         datetime('now', '-2 hours'),
         0, 'Friend', 0)
    """)


def _seed_contacts(db: DatabaseEngine) -> None:
    """Create raw_contacts table with test data."""
    db.execute("""
        CREATE TABLE IF NOT EXISTS raw_contacts (
            id VARCHAR PRIMARY KEY,
            name VARCHAR,
            phone VARCHAR,
            email VARCHAR,
            birthday VARCHAR,
            sensitivity_tier INTEGER DEFAULT 2
        )
    """)
    db.execute("""
        INSERT INTO raw_contacts (id, name, phone, email, birthday)
        VALUES
        ('c1', 'Father', '+5548999', 'father@email.com', NULL),
        ('c2', 'Prince', '+5548888', NULL, NULL)
    """)


def _seed_calendar(db: DatabaseEngine) -> None:
    """Create raw_calendar_events table with test data."""
    db.execute("""
        CREATE TABLE IF NOT EXISTS raw_calendar_events (
            id VARCHAR PRIMARY KEY,
            title VARCHAR,
            start_date TEXT,
            end_date TEXT,
            location VARCHAR,
            attendees VARCHAR,
            sensitivity_tier INTEGER DEFAULT 2
        )
    """)
    db.execute("""
        INSERT INTO raw_calendar_events
        (id, title, start_date, end_date, location)
        VALUES
        ('ev1', 'Team Meeting',
         datetime('now', '+1 day'),
         datetime('now', '+1 day', '+1 hour'),
         'Conference Room A')
    """)


# ================================================================
# Test: Helper functions
# ================================================================


class TestHelpers:
    """Tests for module-level helper functions."""

    def test_make_id_deterministic(self) -> None:
        """Same input produces same ID."""
        id1 = make_hash_id("a", "b", "c")
        id2 = make_hash_id("a", "b", "c")
        assert id1 == id2
        assert len(id1) == 16

    def test_make_id_different_inputs(self) -> None:
        """Different inputs produce different IDs."""
        id1 = make_hash_id("a", "b")
        id2 = make_hash_id("a", "c")
        assert id1 != id2

    def test_safe_str_truncation(self) -> None:
        """Long strings are truncated."""
        assert safe_str("a" * 300, 100) == "a" * 100

    def test_safe_str_none(self) -> None:
        """None returns empty string."""
        assert safe_str(None) == ""

    def test_parse_llm_json_valid(self) -> None:
        """Valid JSON array is parsed."""
        result = parse_llm_json_array('[{"a": 1}]')
        assert result == [{"a": 1}]

    def test_parse_llm_json_with_markdown_fences(self) -> None:
        """JSON with markdown code fences is parsed."""
        raw = '```json\n[{"a": 1}]\n```'
        result = parse_llm_json_array(raw)
        assert result == [{"a": 1}]

    def test_parse_llm_json_empty(self) -> None:
        """Empty or invalid returns empty list."""
        assert parse_llm_json_array("no json here") == []

    def test_parse_llm_json_with_prefix_text(self) -> None:
        """JSON array with leading text is extracted."""
        raw = 'Here are the results:\n[{"a": 1}]'
        result = parse_llm_json_array(raw)
        assert result == [{"a": 1}]


# ================================================================
# Test: Table creation
# ================================================================


class TestTableCreation:
    """Tables are created on init."""

    def test_creates_pending_replies(
        self, proactive: ProactiveIntelligence, tmp_db: DatabaseEngine,
    ) -> None:
        """_pending_replies table exists after init."""
        rows = tmp_db.query("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {r["name"] for r in rows}
        assert "_pending_replies" in tables

    def test_creates_contact_contexts(
        self, proactive: ProactiveIntelligence, tmp_db: DatabaseEngine,
    ) -> None:
        """_contact_contexts table exists after init."""
        rows = tmp_db.query("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {r["name"] for r in rows}
        assert "_contact_contexts" in tables

    def test_creates_actionable_events(
        self, proactive: ProactiveIntelligence, tmp_db: DatabaseEngine,
    ) -> None:
        """_actionable_events table exists after init."""
        rows = tmp_db.query("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {r["name"] for r in rows}
        assert "_actionable_events" in tables


# ================================================================
# Test: Read-only accessors
# ================================================================


class TestReadOnlyAccessors:
    """Read-only methods work without LLM."""

    def test_get_pending_replies_empty(
        self, proactive_readonly: ProactiveIntelligence,
    ) -> None:
        """Returns empty list when no data."""
        result = proactive_readonly.get_pending_replies()
        assert result == []

    def test_get_contact_contexts_empty(
        self, proactive_readonly: ProactiveIntelligence,
    ) -> None:
        """Returns empty list when no data."""
        result = proactive_readonly.get_contact_contexts()
        assert result == []

    def test_get_actionable_events_empty(
        self, proactive_readonly: ProactiveIntelligence,
    ) -> None:
        """Returns empty list when no data."""
        result = proactive_readonly.get_actionable_events()
        assert result == []

    def test_get_pending_replies_returns_stored(
        self,
        proactive_readonly: ProactiveIntelligence,
        tmp_db: DatabaseEngine,
    ) -> None:
        """Returns stored pending replies."""
        tmp_db.execute("""
            INSERT INTO _pending_replies
            (id, message_id, source, contact_name, domain,
             preview, importance, reason, message_at, detected_at)
            VALUES
            ('r1', 'msg1', 'whatsapp', 'Father', 'family',
             'How did the appointment go?', 9,
             'Asking about health update',
             '2026-03-01T10:00:00Z', '2026-03-01T12:00:00Z')
        """)
        result = proactive_readonly.get_pending_replies()
        assert len(result) == 1
        assert result[0].contact_name == "Father"
        assert result[0].importance == 9

    def test_dismissed_replies_excluded(
        self,
        proactive_readonly: ProactiveIntelligence,
        tmp_db: DatabaseEngine,
    ) -> None:
        """Dismissed replies are not returned."""
        tmp_db.execute("""
            INSERT INTO _pending_replies
            (id, message_id, source, contact_name, domain,
             preview, importance, reason, message_at, detected_at,
             dismissed_at)
            VALUES
            ('r2', 'msg2', 'whatsapp', 'Friend', 'social',
             'ok thanks', 3, 'Low priority',
             '2026-03-01T09:00:00Z', '2026-03-01T12:00:00Z',
             '2026-03-01T13:00:00Z')
        """)
        result = proactive_readonly.get_pending_replies()
        assert len(result) == 0

    def test_get_contact_contexts_returns_stored(
        self,
        proactive_readonly: ProactiveIntelligence,
        tmp_db: DatabaseEngine,
    ) -> None:
        """Returns stored contact contexts."""
        tmp_db.execute("""
            INSERT INTO _contact_contexts
            (contact_id, contact_name, total_messages, messages_7d,
             active_context, context_domains, context_priority,
             updated_at)
            VALUES
            ('c1', 'Father', 50, 12,
             'Father has health issues, discussing new medication',
             '["health", "family"]', 3, '2026-03-01T12:00:00Z')
        """)
        result = proactive_readonly.get_contact_contexts()
        assert len(result) == 1
        assert result[0].contact_name == "Father"
        assert result[0].context_priority == 3
        assert "health" in result[0].context_domains

    def test_get_actionable_events_returns_stored(
        self,
        proactive_readonly: ProactiveIntelligence,
        tmp_db: DatabaseEngine,
    ) -> None:
        """Returns stored actionable events."""
        tmp_db.execute("""
            INSERT INTO _actionable_events
            (id, event_id, event_type, title, event_date,
             contact_name, action_needed, importance, detected_at)
            VALUES
            ('ae1', 'c2', 'birthday', 'Prince Birthday',
             '2026-03-02', 'Prince',
             'Tomorrow is Prince birthday! Send wishes', 9,
             '2026-03-01T12:00:00Z')
        """)
        result = proactive_readonly.get_actionable_events()
        assert len(result) == 1
        assert result[0].event_type == "birthday"
        assert result[0].importance == 9


# ================================================================
# Test: Dismiss actions
# ================================================================


class TestDismissActions:
    """Dismiss methods update the database."""

    def test_dismiss_pending_reply(
        self,
        proactive_readonly: ProactiveIntelligence,
        tmp_db: DatabaseEngine,
    ) -> None:
        """Dismissing a reply sets dismissed_at."""
        tmp_db.execute("""
            INSERT INTO _pending_replies
            (id, message_id, source, contact_name, domain,
             preview, importance, reason, message_at, detected_at)
            VALUES
            ('r1', 'msg1', 'whatsapp', 'Father', 'family',
             'Test', 9, 'Test', '2026-03-01T10:00:00Z',
             '2026-03-01T12:00:00Z')
        """)
        proactive_readonly.dismiss_pending_reply("r1")
        result = proactive_readonly.get_pending_replies()
        assert len(result) == 0

    def test_store_pending_replies_preserves_dismissed_at(
        self,
        proactive_readonly: ProactiveIntelligence,
        tmp_db: DatabaseEngine,
    ) -> None:
        """A user dismiss must survive a re-evaluation of the same message.

        Regression: ``_store_pending_replies`` used to DELETE+INSERT each
        row, which silently cleared the user's ``dismissed_at`` on the
        next proactive cycle whenever the LLM still classified the
        message as needing a reply (e.g. when the auto-sweep couldn't
        detect the user's reply because of a chat_name mismatch).
        """
        tmp_db.execute("""
            INSERT INTO _pending_replies
            (id, message_id, source, contact_name, domain,
             preview, importance, reason, message_at, detected_at)
            VALUES
            ('r1', 'msg1', 'whatsapp', 'Sandra', 'family',
             'Old preview', 8, 'Old reason',
             '2026-05-24T18:05:00Z', '2026-05-24T19:00:00Z')
        """)
        proactive_readonly.dismiss_pending_reply("r1")

        dismissed_before = tmp_db.query(
            "SELECT dismissed_at FROM _pending_replies WHERE id = ?",
            ["r1"],
        )
        assert dismissed_before and dismissed_before[0]["dismissed_at"]

        refreshed = PendingReply(
            id="r1",
            message_id="msg1",
            source="whatsapp",
            contact_name="Sandra",
            domain="family",
            preview="Refreshed preview",
            importance=10,
            reason="Refreshed reason",
            message_at="2026-05-24T18:05:00Z",
            detected_at="2026-05-26T15:00:00Z",
        )
        proactive_readonly._store_pending_replies([refreshed])

        rows = tmp_db.query(
            """SELECT dismissed_at, preview, importance, reason
               FROM _pending_replies WHERE id = ?""",
            ["r1"],
        )
        assert rows, "row should still exist after _store_pending_replies"
        assert rows[0]["dismissed_at"], (
            "dismissed_at must be preserved across re-evaluation"
        )
        # Metadata should refresh in place.
        assert rows[0]["preview"] == "Refreshed preview"
        assert int(rows[0]["importance"]) == 10
        assert rows[0]["reason"] == "Refreshed reason"
        # And the dismissed row stays out of the active list.
        assert proactive_readonly.get_pending_replies() == []

    def test_dismiss_actionable_event(
        self,
        proactive_readonly: ProactiveIntelligence,
        tmp_db: DatabaseEngine,
    ) -> None:
        """Dismissing an event sets dismissed_at."""
        tmp_db.execute("""
            INSERT INTO _actionable_events
            (id, event_id, event_type, title, event_date,
             action_needed, importance, detected_at)
            VALUES
            ('ae1', 'ev1', 'meeting', 'Team Meeting',
             '2026-03-02', 'Prepare slides', 7,
             '2026-03-01T12:00:00Z')
        """)
        proactive_readonly.dismiss_actionable_event("ae1")
        result = proactive_readonly.get_actionable_events()
        assert len(result) == 0


# ================================================================
# Test: SQL pre-filtering
# ================================================================


class TestSQLPrefiltering:
    """SQL pre-filter logic for messages."""

    def test_prefilter_finds_unreplied_messages(
        self,
        proactive: ProactiveIntelligence,
        tmp_db: DatabaseEngine,
    ) -> None:
        """Pre-filter finds messages without a reply."""
        _seed_messages(tmp_db)
        candidates = proactive._sql_prefilter_messages()
        assert len(candidates) >= 1
        ids = {c["id"] for c in candidates}
        assert "msg1" in ids

    def test_prefilter_skips_replied_messages(
        self,
        proactive: ProactiveIntelligence,
        tmp_db: DatabaseEngine,
    ) -> None:
        """Pre-filter excludes messages with a later reply."""
        _seed_messages(tmp_db)
        # Add a reply to msg1's chat
        tmp_db.execute("""
            INSERT INTO raw_messages
            (id, source, sender, content, timestamp,
             is_from_me, chat_name, is_group)
            VALUES
            ('reply1', 'whatsapp', 'me', 'It went well',
             datetime('now', '-30 minutes'),
             1, 'Father', 0)
        """)
        candidates = proactive._sql_prefilter_messages()
        ids = {c["id"] for c in candidates}
        assert "msg1" not in ids

    def test_prefilter_finds_unread_emails(
        self,
        proactive: ProactiveIntelligence,
        tmp_db: DatabaseEngine,
    ) -> None:
        """Pre-filter includes unread emails."""
        tmp_db.execute("""
            CREATE TABLE IF NOT EXISTS raw_emails (
                id VARCHAR PRIMARY KEY,
                source VARCHAR DEFAULT 'gmail',
                from_address VARCHAR,
                subject VARCHAR,
                body_preview TEXT,
                date TEXT,
                is_read INTEGER DEFAULT 0,
                sensitivity_tier INTEGER DEFAULT 2
            )
        """)
        tmp_db.execute("""
            INSERT INTO raw_emails
            (id, from_address, subject, body_preview, date, is_read)
            VALUES
            ('email1', 'boss@company.com', 'Project Update',
             'Please review the attached document',
             datetime('now', '-3 hours'), 0)
        """)
        candidates = proactive._sql_prefilter_messages()
        types = {c.get("_type") for c in candidates}
        assert "email" in types


# ================================================================
# Test: LLM evaluation
# ================================================================


class TestLLMEvaluation:
    """LLM evaluation with mock provider."""

    def test_evaluate_messages_with_agent(
        self,
        proactive: ProactiveIntelligence,
        tmp_db: DatabaseEngine,
        stub_pending_reply,
    ) -> None:
        """Agent is called once per sender (batch size 1)."""
        from src.agents.core.output_types import (
            PendingReplyBatch,
            PendingReplyDraft,
        )

        _seed_messages(tmp_db)

        # Father call → needs reply; Friend call → no reply
        def side(*, messages, topics):  # noqa: ARG001
            ids = {m["message_id"] for m in messages}
            if "msg1" in ids:
                return PendingReplyBatch(replies=[
                    PendingReplyDraft(
                        message_id="msg1",
                        needs_reply=True,
                        importance=9,
                        domain="family",
                        reason="Father asking about doctor appointment",
                    ),
                ])
            return PendingReplyBatch(replies=[])

        stub_pending_reply.side_effect = side

        replies = proactive.evaluate_pending_replies()
        assert len(replies) == 1
        assert replies[0].contact_name == "Father"
        assert replies[0].importance == 9
        assert replies[0].domain == "family"
        assert stub_pending_reply.call_count == 2

    def test_evaluate_messages_stores_results(
        self,
        proactive: ProactiveIntelligence,
        tmp_db: DatabaseEngine,
        stub_pending_reply,
    ) -> None:
        """Results are stored in _pending_replies table."""
        from src.agents.core.output_types import (
            PendingReplyBatch,
            PendingReplyDraft,
        )

        _seed_messages(tmp_db)
        stub_pending_reply.return_value = PendingReplyBatch(replies=[
            PendingReplyDraft(
                message_id="msg1",
                needs_reply=True,
                importance=8,
                domain="family",
                reason="Health question",
            ),
        ])
        proactive.evaluate_pending_replies()
        # Read back from table
        stored = proactive.get_pending_replies()
        assert len(stored) >= 1
        assert all(s.source == "whatsapp" for s in stored)

    def test_evaluate_contacts_with_agent(
        self,
        proactive: ProactiveIntelligence,
        tmp_db: DatabaseEngine,
        stub_contact_context,
    ) -> None:
        """Contact context evaluation calls the SBAgent."""
        from src.agents.core.output_types import (
            ContactContextBatch,
            ContactContextDraft,
        )

        _seed_contacts(tmp_db)
        _seed_messages(tmp_db)
        stub_contact_context.return_value = ContactContextBatch(contexts=[
            ContactContextDraft(
                contact_id="c1",
                active_context="Father has health issues",
                context_domains=["health", "family"],
                context_priority=3,
            ),
        ])
        proactive.evaluate_contact_contexts()
        stub_contact_context.assert_called_once()

    def test_evaluate_events_with_calendar(
        self,
        proactive: ProactiveIntelligence,
        tmp_db: DatabaseEngine,
        stub_actionable_events,
    ) -> None:
        """Event evaluation includes calendar events."""
        from src.agents.core.output_types import (
            ActionableEventBatch,
            ActionableEventDraft,
        )

        _seed_calendar(tmp_db)
        stub_actionable_events.return_value = ActionableEventBatch(events=[
            ActionableEventDraft(
                event_id="ev1",
                action_needed="Prepare agenda for meeting",
                importance=7,
            ),
        ])
        events = proactive.evaluate_actionable_events()
        assert len(events) >= 1
        meeting_events = [
            e for e in events if e.event_type == "meeting"
        ]
        assert len(meeting_events) == 1
        assert meeting_events[0].title == "Team Meeting"

    def test_agent_failure_returns_empty(
        self,
        proactive: ProactiveIntelligence,
        tmp_db: DatabaseEngine,
        stub_pending_reply,
    ) -> None:
        """Agent failure returns empty list, doesn't raise."""
        _seed_messages(tmp_db)
        stub_pending_reply.side_effect = Exception("agent down")
        replies = proactive.evaluate_pending_replies()
        assert replies == []

    def test_cold_mart_bypass_falls_back_to_message_count(
        self,
        proactive: ProactiveIntelligence,
        tmp_db: DatabaseEngine,
        stub_pending_reply,
    ) -> None:
        """Topic contacts exist but match no sender (cold/stale mart).

        The topic-driven filter must not starve the eval to zero —
        it falls back to count-based sender selection so a fresh
        install with unanswered messages still surfaces pending
        replies on the first eval.
        """
        _seed_messages(tmp_db)
        candidates = proactive._sql_prefilter_messages()
        assert candidates, "fixture should produce unreplied messages"

        cold_topic_contacts = {
            "someone unrelated": {
                "name": "Someone Unrelated",
                "importance": 9,
                "topics": [],
                "messages_7d": 0,
                "notification_priority": 0,
                "top_topic": "",
            },
        }
        evaluated = proactive._llm_evaluate_messages(
            candidates, cold_topic_contacts,
        )
        # Without the bypass, _prioritize_senders returns [] and the
        # agent is never called.
        assert stub_pending_reply.call_count >= 1
        # The stub returns no drafts by default, so no results — but
        # the senders WERE evaluated, which is the starvation fix.
        assert evaluated == []

    def test_topic_filter_still_narrows_when_mart_is_warm(
        self,
        proactive: ProactiveIntelligence,
        tmp_db: DatabaseEngine,
        stub_pending_reply,
    ) -> None:
        """Warm mart: only topic-matching senders are evaluated."""
        from src.agents.core.output_types import (
            PendingReplyBatch,
            PendingReplyDraft,
        )

        _seed_messages(tmp_db)
        stub_pending_reply.return_value = PendingReplyBatch(replies=[
            PendingReplyDraft(
                message_id="msg1",
                needs_reply=True,
                importance=8,
                domain="family",
                reason="x",
            ),
        ])
        warm_topic_contacts = {
            "father": {
                "name": "Father",
                "importance": 9,
                "topics": [],
                "messages_7d": 0,
                "notification_priority": 0,
                "top_topic": "",
            },
        }
        candidates = proactive._sql_prefilter_messages()
        proactive._llm_evaluate_messages(candidates, warm_topic_contacts)
        # Only Father matches — the other sender is filtered out.
        assert stub_pending_reply.call_count == 1

    def test_get_status_reports_freshness(
        self,
        proactive: ProactiveIntelligence,
    ) -> None:
        """get_status exposes last-evaluated-at for the dashboard."""
        status = proactive.get_status()
        assert status["last_evaluated_at"] is None
        assert status["pending_replies"] == 0

        proactive._mark_evaluated()
        status = proactive.get_status()
        assert status["last_evaluated_at"] is not None

    def test_get_status_falls_back_to_fingerprint_on_upgrade(
        self,
        proactive: ProactiveIntelligence,
    ) -> None:
        """An install with a fingerprint but no marker isn't "never run"."""
        proactive._store_fingerprint("fp-1")
        assert proactive.get_status()["last_evaluated_at"] is not None

    def test_unchanged_data_still_refreshes_freshness(
        self,
        proactive: ProactiveIntelligence,
    ) -> None:
        """An idle-but-healthy loop must not read as starved.

        ``evaluate_all`` short-circuits when the data fingerprint is
        unchanged, so sourcing freshness from that row's ``updated_at``
        froze the timestamp whenever no new mail arrived — flagging the
        dashboard amber on a loop that was running perfectly. A
        freshness signal that false-alarms trains the user to ignore the
        one indicator meant to reveal real starvation.
        """
        stale = "2020-01-01T00:00:00+00:00"
        proactive._store_fingerprint(proactive._compute_data_fingerprint())
        proactive._db.execute(
            "UPDATE _proactive_state SET updated_at = ? "
            "WHERE key = 'fingerprint'",
            [stale],
        )
        assert proactive.get_status()["last_evaluated_at"] == stale

        # Data is unchanged, so this cycle short-circuits — but it did
        # run to completion, and freshness must say so.
        proactive.evaluate_all()

        refreshed = proactive.get_status()["last_evaluated_at"]
        assert refreshed is not None
        assert refreshed != stale

    def test_failed_pillar_leaves_freshness_stale(
        self,
        proactive: ProactiveIntelligence,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A cycle that failed really did produce nothing — flag it.

        The counterpart to the test above: freshness must not be so
        eager that it hides genuine starvation.
        """
        _seed_messages(proactive._db)

        def _boom(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("provider down")

        monkeypatch.setattr(proactive, "_llm_evaluate_messages", _boom)
        proactive.evaluate_all()

        assert proactive.get_status()["last_evaluated_at"] is None


# ================================================================
# Test: Birthday detection
# ================================================================


class TestBirthdayDetection:
    """Birthday detection from raw_contacts."""

    def test_detects_birthday_today(
        self,
        proactive: ProactiveIntelligence,
        tmp_db: DatabaseEngine,
    ) -> None:
        """Detects a birthday that is today."""
        tmp_db.execute("""
            CREATE TABLE IF NOT EXISTS raw_contacts (
                id VARCHAR PRIMARY KEY,
                name VARCHAR,
                phone VARCHAR,
                email VARCHAR,
                birthday VARCHAR,
                sensitivity_tier INTEGER DEFAULT 2
            )
        """)
        # Fixed clock, and a birthday keyed to it. This used to derive
        # the birthday from `datetime.now()` (local) while the SQL used
        # `date('now')` (UTC) and ignored the `now` argument entirely —
        # so the test failed for anyone west of UTC between local and
        # UTC midnight, on code they hadn't touched.
        now = "2026-03-01T00:00:00Z"
        tmp_db.execute(
            "INSERT INTO raw_contacts (id, name, birthday) "
            "VALUES ('b1', 'Birthday Person', '1990-03-01')",
        )

        birthdays = proactive._detect_birthdays(now)

        assert len(birthdays) >= 1
        assert birthdays[0].event_type == "birthday"
        assert "Birthday Person" in birthdays[0].title

    def test_birthday_window_follows_the_supplied_clock(
        self,
        proactive: ProactiveIntelligence,
        tmp_db: DatabaseEngine,
    ) -> None:
        """`now` drives the window — it is not decoration.

        The SQL previously hardcoded `date('now')`, so this argument was
        accepted and discarded. Anchoring on it is what makes the
        three-day lookahead testable at all, and it puts the choice of
        clock at the call site.
        """
        tmp_db.execute("""
            CREATE TABLE IF NOT EXISTS raw_contacts (
                id VARCHAR PRIMARY KEY,
                name VARCHAR,
                phone VARCHAR,
                email VARCHAR,
                birthday VARCHAR,
                sensitivity_tier INTEGER DEFAULT 2
            )
        """)
        tmp_db.execute(
            "INSERT INTO raw_contacts (id, name, birthday) "
            "VALUES ('b2', 'June Person', '1990-06-15')",
        )

        # Two days out: inside the three-day window.
        assert proactive._detect_birthdays("2026-06-13T12:00:00Z")
        # Two weeks out: outside it.
        assert not proactive._detect_birthdays("2026-06-01T12:00:00Z")

    def test_no_birthday_no_contacts(
        self,
        proactive: ProactiveIntelligence,
        tmp_db: DatabaseEngine,
    ) -> None:
        """No birthdays when raw_contacts doesn't exist."""
        birthdays = proactive._detect_birthdays(
            "2026-03-01T00:00:00Z",
        )
        assert birthdays == []


# ================================================================
# Test: evaluate_all
# ================================================================


class TestEvaluateAll:
    """Full evaluation cycle."""

    def test_evaluate_all_returns_result(
        self,
        proactive: ProactiveIntelligence,
        tmp_db: DatabaseEngine,

    ) -> None:
        """evaluate_all returns ProactiveResult."""

        result = proactive.evaluate_all()
        assert isinstance(result, ProactiveResult)
        assert result.evaluated_at != ""

    def test_evaluate_all_handles_partial_failure(
        self,
        proactive: ProactiveIntelligence,
        tmp_db: DatabaseEngine,
        stub_pending_reply,
    ) -> None:
        """evaluate_all continues even if one pillar fails."""
        _seed_messages(tmp_db)
        # PendingReplyAgent fails; the other two pillars still run.
        stub_pending_reply.side_effect = Exception(
            "First pillar failed",
        )
        result = proactive.evaluate_all()
        assert isinstance(result, ProactiveResult)

    def test_failed_goal_mining_does_not_store_fingerprint(
        self,
        proactive: ProactiveIntelligence,
        tmp_db: DatabaseEngine,
        monkeypatch,
    ) -> None:
        """A failed cycle must not mark the data state as evaluated —
        otherwise (e.g. with the provider down) later cycles skip
        re-evaluation until unrelated new data changes the hash, and
        goals are never re-mined."""
        _seed_messages(tmp_db)
        calls: list[int] = []

        def mine_goals_flaky(self, **kwargs):  # noqa: ANN001, ANN003
            calls.append(1)
            # First cycle: extractor failure (None). Then: empty result.
            return None if len(calls) == 1 else []

        monkeypatch.setattr(
            "src.agents.tasks.TaskCurator.mine_goals", mine_goals_flaky,
        )
        proactive.evaluate_all()
        assert proactive._get_stored_fingerprint() is None

        # Mining recovers → the SAME data state is re-evaluated and the
        # fingerprint is stored this time.
        proactive.evaluate_all()
        assert proactive._get_stored_fingerprint() is not None

    def test_failed_goal_mining_does_not_starve_task_proposing(
        self,
        proactive: ProactiveIntelligence,
        tmp_db: DatabaseEngine,
        monkeypatch,
    ) -> None:
        """A transient goal-mining failure (provider blip, firewall
        false-positive on one evidence batch) must not block the rest
        of the pillar — observed live: one rejected goal prompt starved
        task proposing for the whole cycle."""
        from src.core.db_helpers import utc_ago_iso

        _seed_messages(tmp_db)
        # The pillar's window predicate compares ISO-T strings (the
        # production format); seed one in-window row in that format.
        tmp_db.execute(
            "INSERT INTO raw_messages (id, source, sender, content, "
            "timestamp) VALUES ('fresh1', 'whatsapp', 's', "
            "'send the contract tomorrow', ?)",
            [utc_ago_iso(minutes=30)],
        )
        called = {"propose": False}
        monkeypatch.setattr(
            "src.agents.tasks.TaskCurator.mine_goals",
            lambda self, **kwargs: None,
        )

        def fake_propose(self, msgs):  # noqa: ANN001
            called["propose"] = True
            return []

        monkeypatch.setattr(
            "src.agents.tasks.TaskCurator.propose_from_messages",
            fake_propose,
        )
        proactive.evaluate_all()
        # The proposer still ran, and the failure still blocked the
        # fingerprint so the cycle retries.
        assert called["propose"] is True
        assert proactive._get_stored_fingerprint() is None


# ================================================================
# Test: Data class serialization
# ================================================================


class TestDataClasses:
    """Frozen dataclass validation."""

    def test_pending_reply_creation(self) -> None:
        """PendingReply can be created."""
        pr = PendingReply(
            id="test",
            message_id="msg1",
            source="whatsapp",
            contact_name="Father",
            domain="family",
            preview="Test message",
            importance=9,
            reason="Health question",
            message_at="2026-03-01T10:00:00Z",
        )
        assert pr.sensitivity_tier == 2

    def test_contact_context_creation(self) -> None:
        """ContactContext can be created."""
        cc = ContactContext(
            contact_id="c1",
            contact_name="Father",
            active_context="Health issues",
            context_domains=["health"],
            context_priority=3,
        )
        assert cc.context_priority == 3

    def test_actionable_event_creation(self) -> None:
        """ActionableEvent can be created."""
        ae = ActionableEvent(
            id="ae1",
            event_id="ev1",
            event_type="birthday",
            title="Prince Birthday",
            event_date="2026-03-02",
            contact_name="Prince",
            action_needed="Send wishes",
            importance=9,
        )
        assert ae.event_type == "birthday"


# ================================================================
# Topic-awareness tests
# ================================================================


def _seed_mart_contact_summary(db: DatabaseEngine) -> None:
    """Create mart_contact_summary with topic data."""
    import json as _json

    db.execute("""
        CREATE TABLE IF NOT EXISTS mart_contact_summary (
            contact_name VARCHAR,
            top_topic VARCHAR,
            max_topic_importance INTEGER,
            active_topics_json TEXT,
            notification_priority INTEGER,
            messages_7d INTEGER
        )
    """)
    db.execute(
        "INSERT INTO mart_contact_summary VALUES "
        "(?, ?, ?, ?, ?, ?)",
        [
            "Father",
            "Health appointment",
            9,
            _json.dumps([
                {"topic": "Health appointment"},
            ]),
            90,
            15,
        ],
    )
    db.execute(
        "INSERT INTO mart_contact_summary VALUES "
        "(?, ?, ?, ?, ?, ?)",
        [
            "Friend",
            "Weekend plans",
            5,
            _json.dumps([
                {"topic": "Weekend plans"},
            ]),
            40,
            3,
        ],
    )


class TestTopicBoostImportance:
    """Tests for _topic_boost_importance helper."""

    def test_high_importance_boost(self) -> None:
        """Contact with importance >= 7 gets +2."""
        tc = {
            "father": {
                "name": "Father",
                "importance": 9,
            },
        }
        result = _topic_boost_importance(5, "Father", tc)
        assert result == 7

    def test_medium_importance_boost(self) -> None:
        """Contact with importance 5-6 gets +1."""
        tc = {
            "friend": {
                "name": "Friend",
                "importance": 5,
            },
        }
        result = _topic_boost_importance(5, "Friend", tc)
        assert result == 6

    def test_no_boost_unknown_contact(self) -> None:
        """Unknown contact gets no boost."""
        tc = {
            "father": {
                "name": "Father",
                "importance": 9,
            },
        }
        result = _topic_boost_importance(5, "Stranger", tc)
        assert result == 5

    def test_capped_at_10(self) -> None:
        """Boosted importance never exceeds 10."""
        tc = {
            "father": {
                "name": "Father",
                "importance": 9,
            },
        }
        result = _topic_boost_importance(9, "Father", tc)
        assert result == 10


class TestTopicAwarePendingReplies:
    """Tests for topic-boosted pending reply importance."""

    def test_reply_importance_boosted(
        self,
        tmp_db: DatabaseEngine,
        stub_pending_reply,
    ) -> None:
        """Replies from topic contacts get boosted."""
        from src.agents.core.output_types import (
            PendingReplyBatch,
            PendingReplyDraft,
        )

        _seed_messages(tmp_db)
        _seed_mart_contact_summary(tmp_db)

        stub_pending_reply.return_value = PendingReplyBatch(replies=[
            PendingReplyDraft(
                message_id="msg1",
                needs_reply=True,
                importance=6,
                domain="health",
                reason="Health question",
            ),
        ])

        pi = ProactiveIntelligence(db_engine=tmp_db)
        results = pi.evaluate_pending_replies()

        # Father has importance=9 in topics → +2 boost
        father_reply = next(
            (r for r in results if r.contact_name == "Father"),
            None,
        )
        assert father_reply is not None
        assert father_reply.importance == 8  # 6 + 2

    def test_agent_input_includes_topics(
        self,
        tmp_db: DatabaseEngine,
        stub_pending_reply,
    ) -> None:
        """Agent input includes topic context for relevant sender."""
        _seed_messages(tmp_db)
        _seed_mart_contact_summary(tmp_db)

        pi = ProactiveIntelligence(db_engine=tmp_db)
        pi.evaluate_pending_replies()

        # Father's call should have topics in its kwargs.
        father_call = next(
            (
                c for c in stub_pending_reply.call_args_list
                if any(
                    "Father" in str(m.get("sender", ""))
                    for m in c.kwargs.get("messages", [])
                )
            ),
            None,
        )
        assert father_call is not None
        topics = father_call.kwargs.get("topics") or {}
        topic_blob = " ".join(
            str(v) for v in topics.values()
        )
        assert "Health appointment" in topic_blob


class TestTopicAwareContactContexts:
    """Tests for topic-boosted contact context priority."""

    def test_context_priority_boosted(
        self,
        tmp_db: DatabaseEngine,
        stub_contact_context,
    ) -> None:
        """Contacts with high topic importance get boosted."""
        from src.agents.core.output_types import (
            ContactContextBatch,
            ContactContextDraft,
        )

        _seed_messages(tmp_db)
        _seed_contacts(tmp_db)
        _seed_mart_contact_summary(tmp_db)

        stub_contact_context.return_value = ContactContextBatch(contexts=[
            ContactContextDraft(
                contact_id="c1",
                active_context="Health issues",
                context_domains=["health"],
                context_priority=1,
            ),
        ])

        pi = ProactiveIntelligence(db_engine=tmp_db)
        results = pi.evaluate_contact_contexts()

        father_ctx = next(
            (c for c in results if c.contact_name == "Father"),
            None,
        )
        assert father_ctx is not None
        # Father has importance=9 → boosted to at least 8 (1-10 scale)
        assert father_ctx.context_priority >= 8

    def test_medium_topic_importance_boosts_to_6(
        self,
        tmp_db: DatabaseEngine,
        stub_contact_context,
    ) -> None:
        """Contacts with medium topic importance (5-6) get boosted to 6."""
        from src.agents.core.output_types import (
            ContactContextBatch,
            ContactContextDraft,
        )

        _seed_messages(tmp_db)
        _seed_contacts(tmp_db)

        # Create mart_contact_summary with importance=5
        tmp_db.execute("""
            CREATE TABLE IF NOT EXISTS mart_contact_summary (
                contact_name TEXT, top_topic TEXT,
                max_topic_importance INTEGER, active_topics_json TEXT,
                notification_priority INTEGER, messages_7d INTEGER
            )
        """)
        tmp_db.execute("""
            INSERT INTO mart_contact_summary VALUES
            ('Father', 'Work project', 5, '[]', 5, 3)
        """)

        stub_contact_context.return_value = ContactContextBatch(contexts=[
            ContactContextDraft(
                contact_id="c1",
                active_context="Work project updates",
                context_domains=["work"],
                context_priority=3,
            ),
        ])

        pi = ProactiveIntelligence(db_engine=tmp_db)
        results = pi.evaluate_contact_contexts()

        father_ctx = next(
            (c for c in results if c.contact_name == "Father"),
            None,
        )
        assert father_ctx is not None
        # importance=5 → boosted to at least 6
        assert father_ctx.context_priority >= 6

    def test_contact_agent_input_includes_topics(
        self,
        tmp_db: DatabaseEngine,
        stub_contact_context,
    ) -> None:
        """Contact context agent receives topic context."""
        _seed_messages(tmp_db)
        _seed_contacts(tmp_db)
        _seed_mart_contact_summary(tmp_db)

        pi = ProactiveIntelligence(db_engine=tmp_db)
        pi.evaluate_contact_contexts()

        call_args = stub_contact_context.call_args
        contacts = call_args.kwargs["contacts"]
        topics = call_args.kwargs.get("topics") or {}
        contact_names = {c.get("name") for c in contacts}
        topic_blob = " ".join(str(v) for v in topics.values())
        assert "Father" in contact_names
        assert "Health appointment" in topic_blob


class TestSenderNotification:
    """Tests for per-sender notification formatting."""

    def test_formats_sender_with_reason(self) -> None:
        """Includes sender name and LLM reason."""
        from src.core.cli import _format_sender_notification

        msg = _format_sender_notification(
            sender_name="Alice",
            actionable=[{
                "message_id": "msg1",
                "importance": 8,
                "reason": "Asking about meeting",
                "domain": "work",
            }],
            raw_candidates=[{
                "id": "msg1",
                "content": "Did you see the meeting invite?",
            }],
        )
        assert "Alice" in msg
        assert "Asking about meeting" in msg
        assert "work" in msg

    def test_includes_message_preview(self) -> None:
        """Shows original message preview for context."""
        from src.core.cli import _format_sender_notification

        msg = _format_sender_notification(
            sender_name="Bob",
            actionable=[{
                "message_id": "msg2",
                "importance": 9,
                "reason": "Urgent health question",
                "domain": "family",
            }],
            raw_candidates=[{
                "id": "msg2",
                "content": "How did the doctor appointment go?",
            }],
        )
        assert "doctor appointment" in msg

    def test_empty_actionable_returns_empty(self) -> None:
        """Returns empty string when no actionable items."""
        from src.core.cli import _format_sender_notification

        msg = _format_sender_notification(
            sender_name="Carol",
            actionable=[],
            raw_candidates=[],
        )
        assert msg == "🧠 Carol"

    def test_caps_at_three_items(self) -> None:
        """Only includes up to 3 actionable items per sender."""
        from src.core.cli import _format_sender_notification

        items = [
            {"message_id": f"m{i}", "importance": 7,
             "reason": f"Reason {i}", "domain": "work"}
            for i in range(5)
        ]
        msg = _format_sender_notification(
            sender_name="Dave",
            actionable=items,
            raw_candidates=[],
        )
        assert "Reason 0" in msg
        assert "Reason 2" in msg
        assert "Reason 3" not in msg


class TestTopicDigest:
    """Tests for ProactiveIntelligence.build_topic_digest()."""

    def test_detects_new_topics(
        self,
        proactive: ProactiveIntelligence,
        tmp_db: DatabaseEngine,
    ) -> None:
        """New important topics appear as 'new' in digest."""
        # Seed int_contact_topics
        tmp_db.execute("""
            CREATE TABLE IF NOT EXISTS int_contact_topics (
                contact_name TEXT,
                topic TEXT,
                description TEXT,
                importance INTEGER,
                status TEXT,
                sensitivity_tier INTEGER DEFAULT 3
            )
        """)
        tmp_db.execute("""
            INSERT INTO int_contact_topics
            (contact_name, topic, description, importance, status)
            VALUES
            ('Alice', 'hiring psychologist', 'Looking for clinic psych', 8, 'active'),
            ('Bob', 'casual chat', 'Just chatting', 3, 'active')
        """)

        digest = proactive.build_topic_digest()

        # Alice's topic is new + important → included
        new_entries = [e for e in digest if e.change_type == "new"]
        assert len(new_entries) == 1
        assert new_entries[0].contact_name == "Alice"
        assert new_entries[0].topic == "hiring psychologist"
        # Bob's topic is too low importance → excluded
        assert not any(e.contact_name == "Bob" for e in digest)

    def test_detects_promoted_topics(
        self,
        proactive: ProactiveIntelligence,
        tmp_db: DatabaseEngine,
    ) -> None:
        """Topics with importance increase >= 2 are promoted."""
        # Store a previous snapshot
        import json

        tmp_db.execute(
            "INSERT OR REPLACE INTO _proactive_state "
            "(key, value, updated_at) VALUES (?, ?, datetime('now'))",
            [
                "topic_snapshot",
                json.dumps({
                    "Alice:project X": {
                        "importance": 5,
                        "status": "active",
                        "description": "Working on project X",
                        "contact_name": "Alice",
                        "topic": "project X",
                    },
                }),
            ],
        )

        # Current topics: importance went up
        tmp_db.execute("""
            CREATE TABLE IF NOT EXISTS int_contact_topics (
                contact_name TEXT,
                topic TEXT,
                description TEXT,
                importance INTEGER,
                status TEXT,
                sensitivity_tier INTEGER DEFAULT 3
            )
        """)
        tmp_db.execute("""
            INSERT INTO int_contact_topics
            (contact_name, topic, description, importance, status)
            VALUES ('Alice', 'project X', 'Now critical deadline', 8, 'active')
        """)

        digest = proactive.build_topic_digest()

        promoted = [e for e in digest if e.change_type == "promoted"]
        assert len(promoted) == 1
        assert promoted[0].previous_importance == 5
        assert promoted[0].importance == 8

    def test_detects_resolved_topics(
        self,
        proactive: ProactiveIntelligence,
        tmp_db: DatabaseEngine,
    ) -> None:
        """Topics that went from active to resolved are detected."""
        import json

        tmp_db.execute(
            "INSERT OR REPLACE INTO _proactive_state "
            "(key, value, updated_at) VALUES (?, ?, datetime('now'))",
            [
                "topic_snapshot",
                json.dumps({
                    "Bob:health issue": {
                        "importance": 7,
                        "status": "active",
                        "description": "Ongoing health concern",
                        "contact_name": "Bob",
                        "topic": "health issue",
                    },
                }),
            ],
        )

        tmp_db.execute("""
            CREATE TABLE IF NOT EXISTS int_contact_topics (
                contact_name TEXT,
                topic TEXT,
                description TEXT,
                importance INTEGER,
                status TEXT,
                sensitivity_tier INTEGER DEFAULT 3
            )
        """)
        tmp_db.execute("""
            INSERT INTO int_contact_topics
            (contact_name, topic, description, importance, status)
            VALUES ('Bob', 'health issue', 'Resolved now', 7, 'resolved')
        """)

        digest = proactive.build_topic_digest()

        resolved = [e for e in digest if e.change_type == "resolved"]
        assert len(resolved) == 1
        assert resolved[0].contact_name == "Bob"
        assert resolved[0].status == "resolved"

    def test_detects_disappeared_topics(
        self,
        proactive: ProactiveIntelligence,
        tmp_db: DatabaseEngine,
    ) -> None:
        """Topics that were important but disappeared are resolved."""
        import json

        tmp_db.execute(
            "INSERT OR REPLACE INTO _proactive_state "
            "(key, value, updated_at) VALUES (?, ?, datetime('now'))",
            [
                "topic_snapshot",
                json.dumps({
                    "Carol:job search": {
                        "importance": 8,
                        "status": "active",
                        "description": "Looking for jobs",
                        "contact_name": "Carol",
                        "topic": "job search",
                    },
                }),
            ],
        )

        # No int_contact_topics table → empty current
        digest = proactive.build_topic_digest()

        resolved = [e for e in digest if e.change_type == "resolved"]
        assert len(resolved) == 1
        assert resolved[0].contact_name == "Carol"
        assert resolved[0].importance == 0

    def test_falls_back_to_cache_table(
        self,
        proactive: ProactiveIntelligence,
        tmp_db: DatabaseEngine,
    ) -> None:
        """Uses _contact_topics_cache when int_contact_topics is absent."""
        import json

        tmp_db.execute("""
            CREATE TABLE IF NOT EXISTS _contact_topics_cache (
                contact_name TEXT PRIMARY KEY,
                fingerprint TEXT NOT NULL,
                topics_json TEXT NOT NULL,
                extracted_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        tmp_db.execute(
            "INSERT INTO _contact_topics_cache "
            "(contact_name, fingerprint, topics_json) VALUES (?, ?, ?)",
            [
                "Dave",
                "abc123",
                json.dumps([
                    {
                        "topic": "wedding planning",
                        "description": "Getting married next month",
                        "importance": 9,
                        "status": "active",
                    },
                ]),
            ],
        )

        digest = proactive.build_topic_digest()

        assert len(digest) == 1
        assert digest[0].contact_name == "Dave"
        assert digest[0].topic == "wedding planning"
        assert digest[0].change_type == "new"

    def test_snapshot_persists(
        self,
        proactive: ProactiveIntelligence,
        tmp_db: DatabaseEngine,
    ) -> None:
        """After build_topic_digest, snapshot is stored for next run."""
        import json

        tmp_db.execute("""
            CREATE TABLE IF NOT EXISTS int_contact_topics (
                contact_name TEXT,
                topic TEXT,
                description TEXT,
                importance INTEGER,
                status TEXT,
                sensitivity_tier INTEGER DEFAULT 3
            )
        """)
        tmp_db.execute("""
            INSERT INTO int_contact_topics
            (contact_name, topic, description, importance, status)
            VALUES ('Eve', 'project deadline', 'Due next week', 7, 'active')
        """)

        proactive.build_topic_digest()

        # Verify snapshot was stored
        rows = tmp_db.query(
            "SELECT value FROM _proactive_state WHERE key = 'topic_snapshot'"
        )
        assert len(rows) == 1
        snapshot = json.loads(rows[0]["value"])
        assert "Eve:project deadline" in snapshot

    def test_updated_topics_require_importance_threshold(
        self,
        proactive: ProactiveIntelligence,
        tmp_db: DatabaseEngine,
    ) -> None:
        """Active topics below importance threshold are excluded."""
        import json

        tmp_db.execute(
            "INSERT OR REPLACE INTO _proactive_state "
            "(key, value, updated_at) VALUES (?, ?, datetime('now'))",
            [
                "topic_snapshot",
                json.dumps({
                    "Zoe:casual plans": {
                        "importance": 3,
                        "status": "active",
                        "description": "Maybe meeting",
                        "contact_name": "Zoe",
                        "topic": "casual plans",
                    },
                }),
            ],
        )

        tmp_db.execute("""
            CREATE TABLE IF NOT EXISTS int_contact_topics (
                contact_name TEXT,
                topic TEXT,
                description TEXT,
                importance INTEGER,
                status TEXT,
                sensitivity_tier INTEGER DEFAULT 3
            )
        """)
        tmp_db.execute("""
            INSERT INTO int_contact_topics
            (contact_name, topic, description, importance, status)
            VALUES ('Zoe', 'casual plans', 'Maybe meeting', 4, 'active')
        """)

        digest = proactive.build_topic_digest()

        # importance 4 < threshold 5 → not included as "updated"
        assert not any(
            e.contact_name == "Zoe" and e.change_type == "updated"
            for e in digest
        )


# ================================================================
# Test: sweep_resolved_pending_replies
# ================================================================


def _seed_raw_emails_table(db: DatabaseEngine) -> None:
    """Create raw_emails matching the migrations schema."""
    db.execute("""
        CREATE TABLE IF NOT EXISTS raw_emails (
            id              TEXT PRIMARY KEY,
            source          TEXT NOT NULL DEFAULT 'unknown',
            message_id      TEXT,
            subject         TEXT,
            from_address    TEXT,
            to_addresses    TEXT,
            date            TEXT,
            body_preview    TEXT,
            is_read         INTEGER DEFAULT 0,
            folder          TEXT,
            labels          TEXT,
            sensitivity_tier INTEGER NOT NULL DEFAULT 2,
            created_at      TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)


class TestSweepResolvedPendingReplies:
    """``sweep_resolved_pending_replies`` cheaply clears resolved loops."""

    def test_email_extract_addr_with_brackets(self) -> None:
        from src.agents.proactive.persistence import _extract_email_addr
        assert _extract_email_addr(
            "Elmara Dittgen <elmara@example.com>",
        ) == "elmara@example.com"

    def test_email_extract_addr_bare(self) -> None:
        from src.agents.proactive.persistence import _extract_email_addr
        assert _extract_email_addr("elmara@example.com") == (
            "elmara@example.com"
        )

    def test_email_extract_addr_invalid(self) -> None:
        from src.agents.proactive.persistence import _extract_email_addr
        assert _extract_email_addr("just a name") == ""
        assert _extract_email_addr("") == ""

    def test_sweep_whatsapp_dismisses_replied(
        self,
        proactive_readonly: ProactiveIntelligence,
        tmp_db: DatabaseEngine,
    ) -> None:
        """A pending reply is dismissed when an outbound msg exists."""
        _seed_messages(tmp_db)
        tmp_db.execute("""
            INSERT INTO _pending_replies
            (id, message_id, source, contact_name, domain,
             preview, importance, reason, message_at, detected_at)
            VALUES
            ('pr-father', 'msg1', 'whatsapp', 'Father', 'family',
             'preview', 9, 'reason',
             datetime('now', '-1 hour'),
             datetime('now', '-50 minutes'))
        """)
        tmp_db.execute("""
            INSERT INTO raw_messages
            (id, source, sender, content, timestamp,
             is_from_me, chat_name, is_group)
            VALUES
            ('reply-from-me', 'whatsapp', 'me', 'I replied',
             datetime('now', '-10 minutes'),
             1, 'Father', 0)
        """)

        dismissed = proactive_readonly.sweep_resolved_pending_replies()

        assert dismissed == 1
        assert proactive_readonly.get_pending_replies() == []

    def test_sweep_whatsapp_keeps_unreplied(
        self,
        proactive_readonly: ProactiveIntelligence,
        tmp_db: DatabaseEngine,
    ) -> None:
        """Pending replies without an outbound message stay."""
        _seed_messages(tmp_db)
        tmp_db.execute("""
            INSERT INTO _pending_replies
            (id, message_id, source, contact_name, domain,
             preview, importance, reason, message_at, detected_at)
            VALUES
            ('pr-father', 'msg1', 'whatsapp', 'Father', 'family',
             'preview', 9, 'reason',
             datetime('now', '-1 hour'),
             datetime('now', '-50 minutes'))
        """)

        dismissed = proactive_readonly.sweep_resolved_pending_replies()

        assert dismissed == 0
        assert len(proactive_readonly.get_pending_replies()) == 1

    def test_sweep_email_dismisses_via_sent_folder(
        self,
        proactive_readonly: ProactiveIntelligence,
        tmp_db: DatabaseEngine,
    ) -> None:
        """Email loop closes when a Sent row exists to the same address."""
        _seed_raw_emails_table(tmp_db)
        tmp_db.execute("""
            INSERT INTO raw_emails
            (id, source, from_address, to_addresses, date, is_read, folder)
            VALUES
            ('email-in', 'apple_mail',
             'Elmara Dittgen <elmara@example.com>',
             '["me@example.com"]',
             datetime('now', '-3 hours'),
             0, 'INBOX'),
            ('email-out', 'apple_mail',
             'me@example.com',
             '["elmara@example.com"]',
             datetime('now', '-30 minutes'),
             1, 'Sent')
        """)
        tmp_db.execute("""
            INSERT INTO _pending_replies
            (id, message_id, source, contact_name, domain,
             preview, importance, reason, message_at, detected_at)
            VALUES
            ('pr-elmara', 'email-in', 'gmail', 'Elmara', 'personal',
             'Asked about watering plants', 7, 'direct question',
             datetime('now', '-3 hours'),
             datetime('now', '-2 hours'))
        """)

        dismissed = proactive_readonly.sweep_resolved_pending_replies()

        assert dismissed == 1
        assert proactive_readonly.get_pending_replies() == []

    def test_sweep_email_dismisses_when_read(
        self,
        proactive_readonly: ProactiveIntelligence,
        tmp_db: DatabaseEngine,
    ) -> None:
        """An email marked is_read=1 in Mail.app dismisses the loop."""
        _seed_raw_emails_table(tmp_db)
        tmp_db.execute("""
            INSERT INTO raw_emails
            (id, source, from_address, to_addresses, date, is_read, folder)
            VALUES
            ('email-in', 'apple_mail',
             'Elmara <elmara@example.com>', '["me@example.com"]',
             datetime('now', '-3 hours'), 1, 'INBOX')
        """)
        tmp_db.execute("""
            INSERT INTO _pending_replies
            (id, message_id, source, contact_name, domain,
             preview, importance, reason, message_at, detected_at)
            VALUES
            ('pr-elmara', 'email-in', 'gmail', 'Elmara', 'personal',
             'preview', 7, 'reason',
             datetime('now', '-3 hours'),
             datetime('now', '-2 hours'))
        """)

        dismissed = proactive_readonly.sweep_resolved_pending_replies()

        assert dismissed == 1

    def test_sweep_email_keeps_when_sent_to_other_address(
        self,
        proactive_readonly: ProactiveIntelligence,
        tmp_db: DatabaseEngine,
    ) -> None:
        """A Sent email to a different correspondent does not dismiss."""
        _seed_raw_emails_table(tmp_db)
        tmp_db.execute("""
            INSERT INTO raw_emails
            (id, source, from_address, to_addresses, date, is_read, folder)
            VALUES
            ('email-in', 'apple_mail',
             'Elmara <elmara@example.com>', '["me@example.com"]',
             datetime('now', '-3 hours'), 0, 'INBOX'),
            ('email-out', 'apple_mail',
             'me@example.com', '["other@example.com"]',
             datetime('now', '-30 minutes'), 1, 'Sent')
        """)
        tmp_db.execute("""
            INSERT INTO _pending_replies
            (id, message_id, source, contact_name, domain,
             preview, importance, reason, message_at, detected_at)
            VALUES
            ('pr-elmara', 'email-in', 'gmail', 'Elmara', 'personal',
             'preview', 7, 'reason',
             datetime('now', '-3 hours'),
             datetime('now', '-2 hours'))
        """)

        dismissed = proactive_readonly.sweep_resolved_pending_replies()

        assert dismissed == 0
        assert len(proactive_readonly.get_pending_replies()) == 1

    def test_get_pending_replies_runs_sweep(
        self,
        proactive_readonly: ProactiveIntelligence,
        tmp_db: DatabaseEngine,
    ) -> None:
        """``get_pending_replies`` invokes the sweep transparently."""
        _seed_messages(tmp_db)
        tmp_db.execute("""
            INSERT INTO _pending_replies
            (id, message_id, source, contact_name, domain,
             preview, importance, reason, message_at, detected_at)
            VALUES
            ('pr-father', 'msg1', 'whatsapp', 'Father', 'family',
             'preview', 9, 'reason',
             datetime('now', '-1 hour'),
             datetime('now', '-50 minutes'))
        """)
        tmp_db.execute("""
            INSERT INTO raw_messages
            (id, source, sender, content, timestamp,
             is_from_me, chat_name, is_group)
            VALUES
            ('reply-from-me', 'whatsapp', 'me', 'I replied',
             datetime('now', '-10 minutes'),
             1, 'Father', 0)
        """)

        result = proactive_readonly.get_pending_replies()

        assert result == []

    def test_sweep_is_idempotent(
        self,
        proactive_readonly: ProactiveIntelligence,
        tmp_db: DatabaseEngine,
    ) -> None:
        """Second sweep call dismisses nothing further."""
        _seed_messages(tmp_db)
        tmp_db.execute("""
            INSERT INTO _pending_replies
            (id, message_id, source, contact_name, domain,
             preview, importance, reason, message_at, detected_at)
            VALUES
            ('pr-father', 'msg1', 'whatsapp', 'Father', 'family',
             'preview', 9, 'reason',
             datetime('now', '-1 hour'),
             datetime('now', '-50 minutes'))
        """)
        tmp_db.execute("""
            INSERT INTO raw_messages
            (id, source, sender, content, timestamp,
             is_from_me, chat_name, is_group)
            VALUES
            ('reply-from-me', 'whatsapp', 'me', 'I replied',
             datetime('now', '-10 minutes'),
             1, 'Father', 0)
        """)

        first = proactive_readonly.sweep_resolved_pending_replies()
        second = proactive_readonly.sweep_resolved_pending_replies()

        assert first == 1
        assert second == 0
