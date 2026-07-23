"""Proactive intelligence evaluation engine.

Scans raw data to identify:
1. Messages needing the user's reply (across WhatsApp, email, etc.)
2. Important people with ongoing contexts (health, work, family)
3. Calendar events and birthdays needing action
4. Topic digest — important topics with updates, promotions, resolutions

Runs periodically in the background (every ~2 hours).  Dashboard reads
stored results — no LLM call on page load.

sensitivity_tier: 3 (processes personal messages and contacts via LLM)
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.core.db_helpers import (
    ensure_tables,
    get_table_columns,
    make_hash_id,
    safe_str,
    table_exists,
    utc_ago_iso,
    utc_now_iso,
)
from src.core.sqlite.engine import DatabaseEngine
from src.core.topic_loader import (
    get_topic_contacts_for_prompt,
    load_topic_contacts,
)

_DEFAULT_PROACTIVE_WINDOW_HOURS = 48.0


def _proactive_window_modifier() -> str:
    """Return the SQLite modifier for the proactive evaluation window.

    Reads ``proactive_window_hours`` from settings.json; falls back to
    the 48-hour default when the setting is missing or invalid.  Values
    are emitted in whole units (hours when integral, otherwise minutes)
    so SQLite's ``datetime`` modifier parses them.

    sensitivity_tier: 1
    """
    try:
        from src.models.llm_provider import load_llm_settings
    except Exception:  # noqa: BLE001
        hours = _DEFAULT_PROACTIVE_WINDOW_HOURS
    else:
        raw = load_llm_settings().get("proactive_window_hours")
        try:
            hours = float(raw) if raw is not None else (
                _DEFAULT_PROACTIVE_WINDOW_HOURS
            )
        except (TypeError, ValueError):
            hours = _DEFAULT_PROACTIVE_WINDOW_HOURS
        if hours <= 0:
            hours = _DEFAULT_PROACTIVE_WINDOW_HOURS
    if float(hours).is_integer():
        return f"-{int(hours)} hours"
    return f"-{max(1, int(round(hours * 60)))} minutes"

logger = logging.getLogger(__name__)


def _extract_email_addr(raw: str) -> str:
    """Extract the bare email from an ``RFC 5322``-style ``Name <addr>``.

    Returns the substring inside angle brackets when present, otherwise
    the trimmed input if it looks like an email, otherwise an empty
    string. Used by ``sweep_resolved_pending_replies`` to match Sent
    rows by correspondent.

    sensitivity_tier: 1
    """
    s = (raw or "").strip()
    if not s:
        return ""
    lt = s.rfind("<")
    gt = s.rfind(">")
    if 0 <= lt < gt:
        candidate = s[lt + 1:gt].strip()
    else:
        candidate = s
    if "@" not in candidate:
        return ""
    return candidate


def _to_local_naive(dt: datetime) -> datetime:
    """Normalize to a naive *local* datetime for comparison.

    An aware value (the mart stores UTC ``…Z`` start times) is
    converted to the machine's local zone before dropping tzinfo; a
    naive value is assumed already-local and passed through. This keeps
    the window check correct against a naive-local ``now`` — comparing
    a UTC wall-clock directly would shift every event by the local UTC
    offset.

    sensitivity_tier: 1
    """
    if dt.tzinfo is not None:
        return dt.astimezone().replace(tzinfo=None)
    return dt


def _within_window(start_time: str, now: datetime, within_hours: float) -> bool:
    """True if ``start_time`` falls in ``(now, now + within_hours]``.

    Tolerant of ISO strings with or without a ``T`` / timezone; returns
    False on anything unparseable so a bad row never surfaces a brief.

    sensitivity_tier: 1
    """
    if not start_time:
        return False
    raw = start_time.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        try:
            dt = datetime.fromisoformat(raw[:19].replace("T", " "))
        except ValueError:
            return False
    dt = _to_local_naive(dt)
    ref = _to_local_naive(now)
    delta_h = (dt - ref).total_seconds() / 3600.0
    return 0 < delta_h <= within_hours


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------


@dataclass(frozen=True)
class PendingReply:
    """A message identified as needing the user's reply.

    sensitivity_tier: 2
    """

    id: str
    message_id: str
    source: str
    contact_name: str
    domain: str
    preview: str
    importance: int
    reason: str
    message_at: str
    detected_at: str = ""
    sensitivity_tier: int = 2


@dataclass(frozen=True)
class ContactContext:
    """Aggregated context for an important contact.

    sensitivity_tier: 3
    """

    contact_id: str
    contact_name: str
    phone: str | None = None
    email: str | None = None
    total_messages: int = 0
    messages_7d: int = 0
    last_message_at: str | None = None
    last_message_preview: str | None = None
    total_events: int = 0
    next_event_at: str | None = None
    next_event_title: str | None = None
    active_context: str | None = None
    context_domains: list[str] = field(default_factory=list)
    context_priority: int = 0
    birthday: str | None = None
    has_upcoming_birthday: bool = False
    updated_at: str = ""


@dataclass(frozen=True)
class ActionableEvent:
    """A calendar event or birthday needing user action.

    sensitivity_tier: 2
    """

    id: str
    event_id: str
    event_type: str
    title: str
    event_date: str
    contact_name: str | None = None
    action_needed: str = ""
    importance: int = 5
    detected_at: str = ""
    sensitivity_tier: int = 2


@dataclass(frozen=True)
class MeetingPrepAttendee:
    """One known attendee's standing going into a meeting.

    Composed from already-cached data (``_contact_contexts`` +
    ``int_contact_topics``) — no LLM.

    sensitivity_tier: 3
    """

    contact_name: str
    situation: str | None = None
    domains: list[str] = field(default_factory=list)
    last_message_at: str | None = None
    last_message_preview: str | None = None
    open_loops: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class MeetingPrepBrief:
    """A "before your 2pm with X, here's where things stand" pack.

    Deterministically composed for an upcoming calendar event with
    known attendees from the enriched-events mart + cached contact
    contexts + per-contact topics. No LLM runs on read, so it's cheap
    enough for page load.

    sensitivity_tier: 3
    """

    event_id: str
    title: str
    start_time: str
    location: str | None = None
    attendees: list[MeetingPrepAttendee] = field(default_factory=list)
    generated_at: str = ""


@dataclass(frozen=True)
class TopicDigestEntry:
    """A single entry in the topic digest notification.

    Represents a topic that is active/promoted/resolved with context
    about what changed since the last evaluation.

    sensitivity_tier: 3
    """

    contact_name: str
    topic: str
    description: str
    importance: int
    status: str  # "active", "resolved", "stale"
    change_type: str  # "updated", "promoted", "resolved", "new"
    previous_importance: int | None = None


@dataclass(frozen=True)
class ProactiveResult:
    """Combined result of all four evaluation pillars.

    sensitivity_tier: 3
    """

    pending_replies: list[PendingReply] = field(default_factory=list)
    contact_contexts: list[ContactContext] = field(default_factory=list)
    actionable_events: list[ActionableEvent] = field(default_factory=list)
    topic_digest: list[TopicDigestEntry] = field(default_factory=list)
    evaluated_at: str = ""


# ------------------------------------------------------------------
# ProactiveIntelligence
#
# LLM evaluation is delegated to three pydantic-ai SBAgents:
# ``PendingReplyAgent``, ``ContactContextAgent``, and
# ``ActionableEventsAgent``. The orchestrator below owns DB scans,
# per-sender batching, topic-boost post-processing, the on_sender_result
# streaming callback, and the birthday + topic-digest pillars.
# ------------------------------------------------------------------




def _topic_boost_importance(
    importance: int,
    contact_name: str,
    topic_contacts: dict[str, dict],
) -> int:
    """Boost importance score based on contact's topic importance.

    +2 for contacts with max_topic_importance >= 7 (critical topics).
    +1 for contacts with max_topic_importance >= 5 (active topics).
    Capped at 10.

    sensitivity_tier: 1
    """
    name_lower = contact_name.lower()
    for tc_name, tc_data in topic_contacts.items():
        if tc_name in name_lower or name_lower in tc_name:
            tc_imp = tc_data.get("importance", 0)
            if tc_imp >= 7:
                return min(10, importance + 2)
            if tc_imp >= 5:
                return min(10, importance + 1)
            break
    return importance


class ProactiveIntelligence:
    """Proactive intelligence evaluation engine.

    Three pillars:
    1. Pending replies — messages needing response
    2. Contact context — per-person situation tracking
    3. Actionable events — calendar items needing action

    sensitivity_tier: 3
    """

    def __init__(
        self,
        db_engine: DatabaseEngine,
    ) -> None:
        self._db = db_engine
        self._ensure_tables()

    # ----------------------------------------------------------
    # Table setup
    # ----------------------------------------------------------

    def _ensure_tables(self) -> None:
        """Create internal tables if they don't exist.

        sensitivity_tier: 1
        """
        ensure_tables(self._db, [
            """
            CREATE TABLE IF NOT EXISTS _pending_replies (
                id              VARCHAR PRIMARY KEY,
                message_id      VARCHAR NOT NULL,
                source          VARCHAR NOT NULL,
                contact_name    VARCHAR NOT NULL,
                domain          VARCHAR NOT NULL,
                preview         TEXT,
                importance      INTEGER DEFAULT 5,
                reason          TEXT,
                message_at      TEXT NOT NULL,
                detected_at     TEXT DEFAULT CURRENT_TIMESTAMP,
                dismissed_at    TEXT,
                notified_at     TEXT,
                sensitivity_tier INTEGER DEFAULT 2
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS _contact_contexts (
                contact_id          VARCHAR PRIMARY KEY,
                contact_name        VARCHAR NOT NULL,
                phone               VARCHAR,
                email               VARCHAR,
                total_messages      INTEGER DEFAULT 0,
                messages_7d         INTEGER DEFAULT 0,
                last_message_at     TEXT,
                last_message_preview TEXT,
                total_events        INTEGER DEFAULT 0,
                next_event_at       TEXT,
                next_event_title    VARCHAR,
                active_context      TEXT,
                context_domains     VARCHAR DEFAULT '[]',
                context_priority    INTEGER DEFAULT 0,
                birthday            VARCHAR,
                has_upcoming_birthday INTEGER DEFAULT 0,
                updated_at          TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS _actionable_events (
                id              VARCHAR PRIMARY KEY,
                event_id        VARCHAR NOT NULL,
                event_type      VARCHAR NOT NULL,
                title           VARCHAR NOT NULL,
                event_date      TEXT NOT NULL,
                contact_name    VARCHAR,
                action_needed   TEXT,
                importance      INTEGER DEFAULT 5,
                detected_at     TEXT DEFAULT CURRENT_TIMESTAMP,
                dismissed_at    TEXT,
                notified_at     TEXT,
                sensitivity_tier INTEGER DEFAULT 2
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS _proactive_state (
                key         TEXT PRIMARY KEY,
                value       TEXT NOT NULL,
                updated_at  TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """,
        ])

    # ----------------------------------------------------------
    # Data fingerprint (skip cycle when nothing changed)
    # ----------------------------------------------------------

    def _compute_data_fingerprint(self) -> str:
        """Build a lightweight hash of current data state.

        Hashes max IDs and row counts from message/email/event tables
        so we can detect whether a new evaluation is needed.

        sensitivity_tier: 1
        """
        parts: list[str] = []
        for table, col in [
            ("raw_messages", "id"),
            ("raw_emails", "id"),
            ("raw_calendar_events", "id"),
        ]:
            try:
                rows = self._db.query(
                    f"SELECT MAX({col}) AS mx, COUNT(*) AS cnt "
                    f"FROM {table}",  # noqa: S608
                )
                if rows:
                    parts.append(f"{table}:{rows[0]['mx']}:{rows[0]['cnt']}")
            except Exception:  # noqa: BLE001
                parts.append(f"{table}:err")

        return make_hash_id(*parts)

    def _get_stored_fingerprint(self) -> str | None:
        """Read last fingerprint from _proactive_state.

        sensitivity_tier: 1
        """
        try:
            rows = self._db.query(
                "SELECT value FROM _proactive_state WHERE key = 'fingerprint'"
            )
            return rows[0]["value"] if rows else None
        except Exception:  # noqa: BLE001
            return None

    def _store_fingerprint(self, fp: str) -> None:
        """Save fingerprint to _proactive_state.

        sensitivity_tier: 1
        """
        try:
            self._db.execute(
                "INSERT OR REPLACE INTO _proactive_state (key, value, updated_at) "
                "VALUES ('fingerprint', ?, ?)",
                [fp, utc_now_iso()],
            )
        except Exception:  # noqa: BLE001
            logger.debug("Could not store fingerprint", exc_info=True)

    # ----------------------------------------------------------
    # Main entry
    # ----------------------------------------------------------

    def evaluate_all(
        self,
        on_sender_result: SenderCallback | None = None,
    ) -> ProactiveResult:
        """Run all three evaluation pillars.

        Skips LLM evaluation when data hasn't changed since last run.
        If ``on_sender_result`` is provided, streams per-sender
        notifications as they are evaluated.

        sensitivity_tier: 3
        """
        now = utc_now_iso()

        # Skip if data hasn't changed since last evaluation
        current_fp = self._compute_data_fingerprint()
        stored_fp = self._get_stored_fingerprint()
        if current_fp == stored_fp:
            logger.info(
                "Proactive eval skipped — data unchanged (fp=%s)", current_fp,
            )
            return self._load_cached_result()

        # Clean stale entries first
        self._clean_stale_entries()

        replies: list[PendingReply] = []
        contexts: list[ContactContext] = []
        events: list[ActionableEvent] = []
        digest: list[TopicDigestEntry] = []
        pillar_failed = False

        try:
            replies = self.evaluate_pending_replies(
                on_sender_result=on_sender_result,
            )
        except Exception:
            pillar_failed = True
            logger.warning("Pending replies evaluation failed", exc_info=True)

        try:
            contexts = self.evaluate_contact_contexts()
        except Exception:
            pillar_failed = True
            logger.warning("Contact context evaluation failed", exc_info=True)

        try:
            events = self.evaluate_actionable_events()
        except Exception:
            pillar_failed = True
            logger.warning("Actionable events evaluation failed", exc_info=True)

        try:
            digest = self.build_topic_digest()
        except Exception:
            pillar_failed = True
            logger.warning("Topic digest build failed", exc_info=True)

        # Pillar 5: tasks + goals via the task_curator. Non-fatal —
        # the curator owns its own dedup so re-runs don't pile up.
        try:
            self._run_task_curator_pillar()
        except Exception:
            pillar_failed = True
            logger.warning(
                "Task curator pillar failed", exc_info=True,
            )

        # Save the fingerprint only when every pillar completed.
        # Storing it after a failed cycle (e.g. provider down) would
        # mark this data state as "evaluated" and skip re-evaluation
        # until unrelated new data happens to change the fingerprint.
        if pillar_failed:
            logger.info(
                "Fingerprint not stored — a pillar failed; the cycle "
                "will re-run on the next tick",
            )
        else:
            self._store_fingerprint(current_fp)

        return ProactiveResult(
            pending_replies=replies,
            contact_contexts=contexts,
            actionable_events=events,
            topic_digest=digest,
            evaluated_at=now,
        )

    def _run_task_curator_pillar(self) -> None:
        """Drive the task curator on every proactive cycle.

        Order matters: mine goals first (so the proposer sees the
        latest goal hints), then propose tasks from recent messages,
        then check for completions, then refresh habit suggestions.
        The daily schedule is regenerated by the daily 6am cron, not
        here — replanning every 2 hours would churn the timeline.

        sensitivity_tier: 3
        """
        from src.agents.tasks import TaskCurator

        curator = TaskCurator(db_engine=self._db)
        # Run every step even when an earlier one fails — a transient
        # goal-mining failure (provider blip, firewall false-positive
        # on one evidence batch) must not starve task proposing for
        # the whole cycle. Failure still surfaces to evaluate_all at
        # the end so the fingerprint is not stored and the cycle
        # retries.
        mining_failed = curator.mine_goals() is None

        try:
            # ISO-T column: bind a Python cutoff. SQLite's
            # datetime('now', ...) compares as a space-separated string
            # and 'T' > ' ' admits the whole UTC day.
            rows = self._db.query(
                "SELECT id, source, sender, content, timestamp "
                "FROM raw_messages "
                "WHERE timestamp >= ? "
                "ORDER BY timestamp DESC LIMIT 80",
                [utc_ago_iso(hours=6)],
            )
        except Exception:  # noqa: BLE001
            rows = []
        msgs = [dict(r) for r in rows]
        if msgs:
            curator.propose_from_messages(msgs)
            curator.detect_completions(msgs)

        curator.regenerate_habits()

        if mining_failed:
            # Raised last: tasks/completions/habits above already ran.
            raise RuntimeError("goal mining failed (model error)")

    def _load_cached_result(self) -> ProactiveResult:
        """Load previously stored results when data hasn't changed.

        Topic digest is always rebuilt (cheap DB diff, no LLM) so that
        the daily notification dedup — not the fingerprint cache —
        controls delivery frequency.

        sensitivity_tier: 2
        """
        replies = self.get_pending_replies()
        contexts = self.get_contact_contexts()
        events = self.get_actionable_events()

        digest: list[TopicDigestEntry] = []
        try:
            digest = self.build_topic_digest()
        except Exception:
            logger.warning("Topic digest build failed (cached path)", exc_info=True)

        return ProactiveResult(
            pending_replies=replies,
            contact_contexts=contexts,
            actionable_events=events,
            topic_digest=digest,
            evaluated_at=utc_now_iso(),
        )

    # ----------------------------------------------------------
    # Pillar 1: Pending replies
    # ----------------------------------------------------------

    def evaluate_pending_replies(
        self,
        on_sender_result: SenderCallback | None = None,
    ) -> list[PendingReply]:
        """Scan messages for unanswered ones and evaluate via LLM.

        Stage 1: SQL pre-filter (zero LLM cost)
        Stage 2: Load topic contacts for context
        Stage 3: Per-sender LLM evaluation (1 call per sender)
        Stage 4: Topic-boost importance scores
        Stage 5: Store results

        If ``on_sender_result`` is provided, fires after each sender's
        LLM call so notifications can stream without waiting for all.

        sensitivity_tier: 2
        """
        candidates = self._sql_prefilter_messages()
        candidates = self._filter_self_closed(candidates)
        if not candidates:
            logger.info("No unanswered messages found")
            # Clear all active pending replies — user has replied to everything
            self._clear_resolved_pending_replies(keep_ids=set())
            return []

        # Stage 2: load topic contacts for LLM context + boosting
        topic_contacts = load_topic_contacts(self._db)

        # Stage 3: per-sender LLM evaluation with streaming callback
        evaluated = self._llm_evaluate_messages(
            candidates, topic_contacts,
            on_sender_result=on_sender_result,
        )
        if not evaluated:
            return []

        # Stage 4+5: build results with topic-boosted importance
        now = utc_now_iso()
        results: list[PendingReply] = []

        for item in evaluated:
            msg_id = str(item.get("message_id", ""))
            if not msg_id:
                continue
            original = next(
                (c for c in candidates if str(c.get("id", "")) == msg_id),
                None,
            )
            if original is None:
                continue

            reply_id = make_hash_id("reply", msg_id)
            contact_name = str(
                original.get("sender_name")
                or original.get("from_address")
                or original.get("sender", "Unknown")
            )

            # Topic-boost: increase importance for topic contacts
            importance = int(item.get("importance", 5))
            importance = _topic_boost_importance(
                importance, contact_name, topic_contacts,
            )

            pr = PendingReply(
                id=reply_id,
                message_id=msg_id,
                source=str(original.get("source", "unknown")),
                contact_name=contact_name,
                domain=str(item.get("domain", "personal")),
                preview=safe_str(
                    original.get("content")
                    or original.get("body_preview")
                    or original.get("subject", ""),
                    200,
                ),
                importance=importance,
                reason=str(item.get("reason", "")),
                message_at=str(original.get("timestamp")
                               or original.get("date", "")),
                detected_at=now,
            )
            results.append(pr)

        self._store_pending_replies(results)
        return results

    def _sql_prefilter_messages(self) -> list[dict[str, Any]]:
        """SQL pre-filter: unreplied messages from last 48h.

        sensitivity_tier: 2
        """
        candidates: list[dict[str, Any]] = []

        # WhatsApp / other messages with is_from_me tracking
        try:
            existing_cols = get_table_columns(self._db,"raw_messages")
            if "is_from_me" in existing_cols:
                has_sender_name = "sender_name" in existing_cols
                has_chat_name = "chat_name" in existing_cols
                has_is_group = "is_group" in existing_cols

                sender_col = "sender_name" if has_sender_name else "sender"
                chat_col = (
                    ", chat_name" if has_chat_name else ""
                )
                chat_where = (
                    "AND chat_name IS NOT NULL"
                    if has_chat_name
                    else ""
                )
                # Filter out groups where user doesn't actively participate
                group_filter = (
                    """AND (
                        m.is_group = 0
                        OR m.chat_name IN (
                            SELECT chat_name FROM raw_messages
                            WHERE is_from_me = 1 AND is_group = 1
                            GROUP BY chat_name
                            HAVING COUNT(*) >= 3
                        )
                    )"""
                    if has_is_group and has_chat_name
                    else (
                        "AND m.is_group = 0"
                        if has_is_group else ""
                    )
                )

                rows = self._db.query(
                    f"""
                    SELECT id, source, sender, {sender_col} as display_name,
                           content, timestamp {chat_col}
                    FROM raw_messages m
                    WHERE m.is_from_me = false
                      AND m.timestamp > datetime('now', ?)
                      {chat_where}
                      {group_filter}
                      AND NOT EXISTS (
                          SELECT 1 FROM raw_messages reply
                          WHERE reply.is_from_me = true
                            AND reply.timestamp > m.timestamp
                            {"AND reply.chat_name = m.chat_name"
                             if has_chat_name else ""}
                      )
                    ORDER BY m.timestamp DESC
                    LIMIT 50
                    """,
                    [_proactive_window_modifier()],
                )
                for r in rows:
                    r["_type"] = "message"
                    # Use display_name as sender_name
                    r["sender_name"] = r.get("display_name") or r.get("sender")
                candidates.extend(rows)
        except Exception:
            logger.debug("Message pre-filter failed", exc_info=True)

        # Emails (unread from last 48h)
        try:
            if table_exists(self._db,"raw_emails"):
                rows = self._db.query(
                    """
                    SELECT id, 'gmail' as source, from_address,
                           subject, body_preview, date as timestamp
                    FROM raw_emails
                    WHERE is_read = false
                      AND date > datetime('now', ?)
                    ORDER BY date DESC
                    LIMIT 30
                    """,
                    [_proactive_window_modifier()],
                )
                for r in rows:
                    r["_type"] = "email"
                    r["sender_name"] = r.get("from_address", "Unknown")
                    r["content"] = (
                        f"Subject: {r.get('subject', '')}\n"
                        f"{r.get('body_preview', '')}"
                    )
                candidates.extend(rows)
        except Exception:
            logger.debug("Email pre-filter failed", exc_info=True)

        return candidates

    @staticmethod
    def _filter_self_closed(
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Remove candidates where the sender self-closed.

        If a sender's most recent message in a chat is short
        (<=20 chars) and not a question, it likely indicates
        acknowledgment ("Ok", "Marcado", thumbs-up). Skip that
        sender's entire conversation in that chat.

        sensitivity_tier: 2
        """
        if not candidates:
            return candidates

        # Group by (sender, chat_name)
        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for c in candidates:
            key = (
                c.get("sender", ""),
                c.get("chat_name", ""),
            )
            groups.setdefault(key, []).append(c)

        filtered: list[dict[str, Any]] = []
        for (sender, chat), msgs in groups.items():
            if len(msgs) < 2:
                filtered.extend(msgs)
                continue
            # Find the most recent message
            latest = max(
                msgs, key=lambda m: m.get("timestamp", ""),
            )
            text = str(latest.get("content", "")).strip()
            if len(text) <= 20 and "?" not in text:
                logger.info(
                    "Self-closed: %s sent '%s' after question",
                    sender, text,
                )
                continue
            filtered.extend(msgs)

        return filtered

    _MAX_SENDERS = 8  # cap to fit within subprocess timeout

    _SENDERS_PER_BATCH = 1  # 2b model needs focused single-sender eval

    # Callback type: (sender_name, llm_results, candidates) → None
    SenderCallback = Callable[
        [str, list[dict[str, Any]], list[dict[str, Any]]],
        None,
    ]

    def _llm_evaluate_messages(
        self,
        candidates: list[dict[str, Any]],
        topic_contacts: dict[str, dict] | None = None,
        on_sender_result: SenderCallback | None = None,
    ) -> list[dict[str, Any]]:
        """Evaluate messages with one LLM call per sender.

        A 2B model needs focused single-sender prompts
        for quality evaluation results.

        If ``on_sender_result`` is provided, it fires after each
        sender's LLM call with (sender_name, llm_items, raw_candidates)
        so callers can stream notifications without waiting for all
        senders to complete.

        sensitivity_tier: 2
        """
        tc = topic_contacts or {}
        topics_for_prompt = get_topic_contacts_for_prompt(tc)

        # Group messages by sender
        by_sender: dict[str, list[dict[str, Any]]] = {}
        for c in candidates:
            sender = str(
                c.get("sender_name") or c.get("sender", "Unknown"),
            )
            by_sender.setdefault(sender, []).append(c)

        # Topic-driven sender selection: keep only senders with topics
        if tc:
            prioritized = self._prioritize_senders(by_sender, tc)
        else:
            # Cold start: no topics yet, pick top senders by msg count
            prioritized = sorted(
                by_sender.keys(),
                key=lambda s: len(by_sender[s]),
                reverse=True,
            )[:self._MAX_SENDERS]

        logger.info(
            "Proactive eval: %d/%d senders selected (topic-driven=%s)",
            len(prioritized), len(by_sender), bool(tc),
        )

        if not prioritized:
            return []

        all_results: list[dict[str, Any]] = []

        # Process in batches of _SENDERS_PER_BATCH
        for batch_start in range(
            0, len(prioritized), self._SENDERS_PER_BATCH,
        ):
            batch_senders = prioritized[
                batch_start:batch_start + self._SENDERS_PER_BATCH
            ]

            batch_messages: list[dict[str, Any]] = []
            relevant_topics: list[dict[str, Any]] = []
            seen_topics: set[str] = set()

            for sender in batch_senders:
                msgs = by_sender[sender]
                for c in msgs[:3]:
                    content = safe_str(c.get("content", ""), 100)
                    # Skip WhatsApp protocol metadata (no real content).
                    if content.startswith("[") and content.endswith("]"):
                        continue
                    batch_messages.append({
                        "message_id": str(c.get("id", "")),
                        "source": str(c.get("source", "")),
                        "sender": sender,
                        "content": content,
                        "timestamp": str(c.get("timestamp", "")),
                    })
                sender_lower = sender.lower()
                for t in topics_for_prompt:
                    topic_id = t.get("topic", "")
                    if topic_id not in seen_topics and (
                        t.get("contact", "").lower() in sender_lower
                        or sender_lower in t.get("contact", "").lower()
                    ):
                        relevant_topics.append(t)
                        seen_topics.add(topic_id)

            if not batch_messages:
                logger.info(
                    "Sender [%s]: skipped (all messages are metadata)",
                    batch_senders[0],
                )
                continue

            sender_name = batch_senders[0]
            logger.info(
                "Sender [%s]: evaluating (%d/%d, %d msgs)…",
                sender_name,
                batch_start + 1,
                len(prioritized),
                len(batch_messages),
            )
            try:
                from src.agents.pending_reply.agent import (
                    PendingReplyAgent,
                )

                topics_map = {
                    t.get("topic", ""): t for t in relevant_topics
                }
                batch_obj = PendingReplyAgent().detect(
                    messages=batch_messages,
                    topics=topics_map,
                )
                if batch_obj is None:
                    raise RuntimeError("PendingReplyAgent returned None")
                # PendingReplyAgent already filters to needs_reply=True;
                # project drafts back into the legacy dict shape so the
                # downstream join + persistence code keeps working.
                batch_results = [
                    {
                        "message_id": draft.message_id,
                        "needs_reply": draft.needs_reply,
                        "importance": draft.importance,
                        "domain": draft.domain,
                        "reason": draft.reason,
                    }
                    for draft in batch_obj.replies
                ]
                all_results.extend(batch_results)

                logger.info(
                    "Sender [%s]: %d msgs need reply "
                    "(importances: %s)",
                    sender_name,
                    len(batch_results),
                    [r["importance"] for r in batch_results],
                )

                # Stream per-sender notification
                if on_sender_result and batch_results:
                    try:
                        on_sender_result(
                            sender_name,
                            batch_results,
                            by_sender[sender_name],
                        )
                    except Exception:
                        logger.debug(
                            "on_sender_result callback failed",
                            exc_info=True,
                        )
            except Exception:
                logger.warning(
                    "Sender [%s]: PendingReplyAgent eval failed",
                    sender_name,
                    exc_info=True,
                )

        return all_results

    @staticmethod
    def _prioritize_senders(
        by_sender: dict[str, list[dict[str, Any]]],
        topic_contacts: dict[str, dict],
    ) -> list[str]:
        """Select and sort senders by topic importance, capped.

        Only keeps senders that fuzzy-match a topic contact.
        Sorted by max topic importance (highest first).

        sensitivity_tier: 1
        """
        scored: list[tuple[str, int]] = []
        for sender in by_sender:
            sender_lower = sender.lower()
            # Fuzzy match against topic contact names
            best_imp = 0
            for tc_key, tc_data in topic_contacts.items():
                if tc_key in sender_lower or sender_lower in tc_key:
                    best_imp = max(
                        best_imp, tc_data.get("importance", 0),
                    )
            if best_imp > 0:
                scored.append((sender, best_imp))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            s for s, _ in scored[:ProactiveIntelligence._MAX_SENDERS]
        ]

    def sweep_resolved_pending_replies(self) -> int:
        """Dismiss pending replies that the user has already answered.

        Cheap SQL-only pass — no LLM, runs on every dashboard load.
        Two reply signals:

        * WhatsApp: a later outbound message exists in the same chat
          (mirrors the NOT EXISTS pre-filter at the start of this
          file used by the proactive pipeline).
        * Email: the inbound is now read, OR a Sent-folder row exists
          addressed to the same correspondent after the inbound date.

        Returns the number of pending replies marked dismissed.

        sensitivity_tier: 2
        """
        to_dismiss: list[str] = []

        # WhatsApp branch — only run if raw_messages exists and exposes
        # the columns the join relies on.
        if table_exists(self._db, "raw_messages"):
            msg_cols = get_table_columns(self._db, "raw_messages")
            if "is_from_me" in msg_cols and "chat_name" in msg_cols:
                try:
                    rows = self._db.query(
                        """
                        SELECT pr.id AS pr_id
                        FROM _pending_replies pr
                        JOIN raw_messages m ON pr.message_id = m.id
                        WHERE pr.dismissed_at IS NULL
                          AND pr.source = 'whatsapp'
                          AND EXISTS (
                              SELECT 1 FROM raw_messages reply
                              WHERE reply.is_from_me = true
                                AND reply.timestamp > m.timestamp
                                AND reply.chat_name = m.chat_name
                          )
                        """,
                    )
                    to_dismiss.extend(str(r["pr_id"]) for r in rows)
                except Exception:  # noqa: BLE001
                    logger.debug(
                        "sweep: whatsapp branch failed", exc_info=True,
                    )

        # Email branch — uses ingested Sent-folder rows (apple_mail
        # bridge writes folder='Sent' for sent messages) plus the
        # is_read fallback that the proactive pipeline already uses.
        if table_exists(self._db, "raw_emails"):
            try:
                rows = self._db.query(
                    """
                    SELECT pr.id AS pr_id,
                           m.from_address AS from_address,
                           m.date         AS m_date,
                           m.is_read      AS is_read
                    FROM _pending_replies pr
                    JOIN raw_emails m ON pr.message_id = m.id
                    WHERE pr.dismissed_at IS NULL
                      AND pr.source = 'gmail'
                    """,
                )
            except Exception:  # noqa: BLE001
                logger.debug(
                    "sweep: email candidate query failed",
                    exc_info=True,
                )
                rows = []

            for r in rows:
                if r.get("is_read"):
                    to_dismiss.append(str(r["pr_id"]))
                    continue
                addr = _extract_email_addr(r.get("from_address") or "")
                if not addr:
                    continue
                try:
                    hit = self._db.query(
                        """
                        SELECT 1 FROM raw_emails sent
                        WHERE LOWER(COALESCE(sent.folder, '')) LIKE '%sent%'
                          AND sent.date > ?
                          AND LOWER(COALESCE(sent.to_addresses, ''))
                              LIKE ?
                        LIMIT 1
                        """,
                        [str(r.get("m_date") or ""), f"%{addr.lower()}%"],
                    )
                except Exception:  # noqa: BLE001
                    logger.debug(
                        "sweep: sent lookup failed for %s",
                        r.get("pr_id"), exc_info=True,
                    )
                    continue
                if hit:
                    to_dismiss.append(str(r["pr_id"]))

        if not to_dismiss:
            return 0

        placeholders = ",".join("?" for _ in to_dismiss)
        self._db.execute(
            f"UPDATE _pending_replies "
            f"SET dismissed_at = CURRENT_TIMESTAMP "
            f"WHERE id IN ({placeholders})",
            list(to_dismiss),
        )
        return len(to_dismiss)

    def _clear_resolved_pending_replies(
        self,
        keep_ids: set[str],
    ) -> None:
        """Delete non-dismissed pending replies that are no longer detected.

        When the SQL pre-filter no longer finds a message (because the
        user has replied), the old ``_pending_replies`` entry must be
        removed.  This method deletes all active (non-dismissed) entries
        whose IDs are NOT in *keep_ids*.

        sensitivity_tier: 1
        """
        if keep_ids:
            placeholders = ",".join("?" for _ in keep_ids)
            self._db.execute(
                f"DELETE FROM _pending_replies "
                f"WHERE dismissed_at IS NULL AND id NOT IN ({placeholders})",
                list(keep_ids),
            )
        else:
            # No pending replies at all — clear everything active
            self._db.execute(
                "DELETE FROM _pending_replies WHERE dismissed_at IS NULL",
            )

    def _store_pending_replies(self, replies: list[PendingReply]) -> None:
        """Upsert pending replies into DuckDB.

        Also removes entries that are no longer detected (user replied).
        For rows that already exist, refresh metadata in place so the
        user's ``dismissed_at`` / ``notified_at`` state is preserved
        across proactive cycles — an explicit dismiss must never be
        wiped by a subsequent re-evaluation of the same message.

        sensitivity_tier: 2
        """
        # Remove entries no longer in results (user has replied)
        current_ids = {r.id for r in replies}
        self._clear_resolved_pending_replies(keep_ids=current_ids)

        for r in replies:
            existing = self._db.query(
                "SELECT 1 FROM _pending_replies WHERE id = ?", [r.id],
            )
            if existing:
                self._db.execute(
                    """UPDATE _pending_replies
                       SET message_id=?, source=?, contact_name=?,
                           domain=?, preview=?, importance=?, reason=?,
                           message_at=?, detected_at=?,
                           sensitivity_tier=?
                       WHERE id = ?""",
                    [
                        r.message_id, r.source, r.contact_name,
                        r.domain, r.preview, r.importance, r.reason,
                        r.message_at, r.detected_at,
                        r.sensitivity_tier, r.id,
                    ],
                )
            else:
                self._db.execute(
                    """INSERT INTO _pending_replies
                       (id, message_id, source, contact_name, domain,
                        preview, importance, reason, message_at,
                        detected_at, sensitivity_tier)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        r.id, r.message_id, r.source, r.contact_name,
                        r.domain, r.preview, r.importance, r.reason,
                        r.message_at, r.detected_at, r.sensitivity_tier,
                    ],
                )

    # ----------------------------------------------------------
    # Pillar 2: Contact context
    # ----------------------------------------------------------

    def evaluate_contact_contexts(self) -> list[ContactContext]:
        """Build per-contact context from recent activity.

        Stage 1: SQL aggregation (zero LLM cost)
        Stage 2: Load topic contacts + merge into activity
        Stage 3: LLM context building (1 call, topic-aware)
        Stage 4: Topic-boost context_priority
        Stage 5: Store results sorted by topic importance

        sensitivity_tier: 3
        """
        contact_activity = self._aggregate_contact_activity()
        if not contact_activity:
            logger.info("No contact activity to evaluate")
            return []

        # Stage 2: merge topic data into contact activity
        topic_contacts = load_topic_contacts(self._db)
        for activity in contact_activity:
            name = str(activity.get("contact_name", "")).lower()
            for tc_name, tc_data in topic_contacts.items():
                if tc_name in name or name in tc_name:
                    activity["max_topic_importance"] = tc_data.get(
                        "importance", 0,
                    )
                    activity["top_topic"] = tc_data.get("top_topic", "")
                    activity["notification_priority"] = tc_data.get(
                        "notification_priority", 0,
                    )
                    break

        # Stage 3: LLM evaluation with topic context
        llm_contexts = self._llm_evaluate_contacts(
            contact_activity, topic_contacts,
        )

        # Stage 4+5: build results with topic-boosted priority
        now = utc_now_iso()
        results: list[ContactContext] = []

        for activity in contact_activity:
            cid = str(activity.get("contact_id", ""))
            llm_data = next(
                (c for c in llm_contexts
                 if str(c.get("contact_id", "")) == cid),
                {},
            )

            # Topic-boost context_priority (same 1-10 scale as other pillars)
            priority = int(llm_data.get("context_priority", 0))
            tc_imp = activity.get("max_topic_importance", 0)
            if tc_imp >= 7:
                priority = max(priority, 8)
            elif tc_imp >= 5:
                priority = max(priority, 6)

            ctx = ContactContext(
                contact_id=cid,
                contact_name=str(activity.get("contact_name", "Unknown")),
                phone=activity.get("phone"),
                email=activity.get("email"),
                total_messages=int(activity.get("total_messages", 0)),
                messages_7d=int(activity.get("messages_7d", 0)),
                last_message_at=activity.get("last_message_at"),
                last_message_preview=safe_str(
                    activity.get("last_message_preview"), 200,
                ) or None,
                total_events=int(activity.get("total_events", 0)),
                next_event_at=activity.get("next_event_at"),
                next_event_title=activity.get("next_event_title"),
                active_context=llm_data.get("active_context"),
                context_domains=llm_data.get("context_domains", []),
                context_priority=priority,
                birthday=activity.get("birthday"),
                has_upcoming_birthday=bool(
                    activity.get("has_upcoming_birthday", False),
                ),
                updated_at=now,
            )
            results.append(ctx)

        # Sort by topic notification_priority, then messages_7d
        results.sort(
            key=lambda c: (
                c.context_priority,
                c.messages_7d,
            ),
            reverse=True,
        )

        self._store_contact_contexts(results)
        return results

    def _aggregate_contact_activity(self) -> list[dict[str, Any]]:
        """Aggregate message/event activity per contact via SQL.

        sensitivity_tier: 2
        """
        contacts: dict[str, dict[str, Any]] = {}

        # Source 1: raw_contacts for names, phones, birthdays
        try:
            if table_exists(self._db,"raw_contacts"):
                rows = self._db.query("""
                    SELECT id, name, phone, email, birthday
                    FROM raw_contacts
                    WHERE name IS NOT NULL AND name != ''
                    LIMIT 200
                """)
                for r in rows:
                    cid = str(r["id"])
                    contacts[cid] = {
                        "contact_id": cid,
                        "contact_name": str(r["name"]),
                        "phone": r.get("phone"),
                        "email": r.get("email"),
                        "birthday": r.get("birthday"),
                        "total_messages": 0,
                        "messages_7d": 0,
                        "last_message_at": None,
                        "last_message_preview": None,
                        "total_events": 0,
                        "next_event_at": None,
                        "next_event_title": None,
                        "has_upcoming_birthday": False,
                    }
        except Exception:
            logger.debug("Contact load failed", exc_info=True)

        # Source 2: message activity by sender_name
        try:
            existing_cols = get_table_columns(self._db,"raw_messages")
            has_sender_name = "sender_name" in existing_cols
            name_col = "sender_name" if has_sender_name else "sender"

            rows = self._db.query(f"""
                SELECT
                    {name_col} as contact_name,
                    COUNT(*) as total_messages,
                    SUM(CASE WHEN timestamp > datetime('now', '-7 days')
                        THEN 1 ELSE 0 END) as messages_7d,
                    MAX(timestamp) as last_message_at,
                    (SELECT content FROM raw_messages sub
                     WHERE sub.{name_col} = raw_messages.{name_col}
                     ORDER BY sub.timestamp DESC LIMIT 1
                    ) as last_message_preview
                FROM raw_messages
                WHERE {name_col} IS NOT NULL
                  AND {name_col} != 'me'
                  AND {name_col} != ''
                  AND {name_col} != 'Unknown'
                GROUP BY {name_col}
                HAVING COUNT(*) >= 3
                ORDER BY messages_7d DESC, total_messages DESC
                LIMIT 30
            """)

            for r in rows:
                name = str(r["contact_name"])
                # Try to match with existing contact by name
                matched_cid = None
                for cid, c in contacts.items():
                    if c["contact_name"].lower() == name.lower():
                        matched_cid = cid
                        break

                if matched_cid:
                    contacts[matched_cid].update({
                        "total_messages": int(r["total_messages"]),
                        "messages_7d": int(r["messages_7d"]),
                        "last_message_at": str(r["last_message_at"])
                        if r["last_message_at"] else None,
                        "last_message_preview": safe_str(
                            r["last_message_preview"], 200,
                        ),
                    })
                else:
                    # Create a synthetic contact entry
                    synth_id = make_hash_id("synth", name)
                    contacts[synth_id] = {
                        "contact_id": synth_id,
                        "contact_name": name,
                        "phone": None,
                        "email": None,
                        "birthday": None,
                        "total_messages": int(r["total_messages"]),
                        "messages_7d": int(r["messages_7d"]),
                        "last_message_at": str(r["last_message_at"])
                        if r["last_message_at"] else None,
                        "last_message_preview": safe_str(
                            r["last_message_preview"], 200,
                        ),
                        "total_events": 0,
                        "next_event_at": None,
                        "next_event_title": None,
                        "has_upcoming_birthday": False,
                    }
        except Exception:
            logger.debug("Message activity aggregation failed", exc_info=True)

        # Source 3: upcoming birthdays (next 7 days)
        try:
            if table_exists(self._db,"raw_contacts"):
                rows = self._db.query("""
                    SELECT id, name, birthday
                    FROM raw_contacts
                    WHERE birthday IS NOT NULL
                      AND date(birthday) IS NOT NULL
                      AND (
                        CAST(strftime('%m', date(birthday)) AS INTEGER)
                            = CAST(strftime('%m', date('now')) AS INTEGER)
                        AND CAST(strftime('%d', date(birthday)) AS INTEGER)
                            BETWEEN CAST(strftime('%d', date('now')) AS INTEGER)
                            AND CAST(strftime('%d', date('now')) AS INTEGER) + 7
                      ) OR (
                        -- Handle month boundary
                        CAST(strftime('%m', date(birthday)) AS INTEGER)
                            = CAST(strftime('%m', date('now', '+7 days')) AS INTEGER)
                        AND CAST(strftime('%d', date(birthday)) AS INTEGER)
                            <= CAST(strftime('%d', date('now', '+7 days')) AS INTEGER)
                        AND CAST(strftime('%m', date('now')) AS INTEGER)
                            != CAST(strftime('%m', date('now', '+7 days')) AS INTEGER)
                      )
                """)
                for r in rows:
                    cid = str(r["id"])
                    if cid in contacts:
                        contacts[cid]["has_upcoming_birthday"] = True
        except Exception:
            logger.debug("Birthday check failed", exc_info=True)

        # Filter to contacts with meaningful activity
        active = [
            c for c in contacts.values()
            if (
                c.get("messages_7d", 0) >= 2
                or c.get("has_upcoming_birthday")
                or c.get("total_messages", 0) >= 5
            )
        ]
        # Sort by recent activity
        active.sort(
            key=lambda c: (
                c.get("messages_7d", 0),
                c.get("total_messages", 0),
            ),
            reverse=True,
        )
        return active[:20]

    _CONTACTS_PER_BATCH = 10  # keep prompts small for reliable output

    def _llm_evaluate_contacts(
        self,
        contact_activity: list[dict[str, Any]],
        topic_contacts: dict[str, dict] | None = None,
    ) -> list[dict[str, Any]]:
        """Evaluate contacts in batched LLM calls.

        Batches 10 contacts per call (2 calls for 20 contacts).

        sensitivity_tier: 3
        """
        if not contact_activity:
            return []

        tc = topic_contacts or {}
        topics_for_prompt = get_topic_contacts_for_prompt(tc)

        # Cap at 20 contacts
        capped = contact_activity[:20]
        all_results: list[dict[str, Any]] = []

        for batch_start in range(
            0, len(capped), self._CONTACTS_PER_BATCH,
        ):
            chunk = capped[
                batch_start:batch_start + self._CONTACTS_PER_BATCH
            ]

            batch = []
            for c in chunk:
                entry: dict[str, Any] = {
                    "contact_id": c.get("contact_id", ""),
                    "name": c.get("contact_name", "Unknown"),
                    "total_messages": c.get("total_messages", 0),
                    "messages_7d": c.get("messages_7d", 0),
                    "last_message_preview": safe_str(
                        c.get("last_message_preview"), 150,
                    ),
                    "has_upcoming_birthday": c.get(
                        "has_upcoming_birthday", False,
                    ),
                }
                if c.get("top_topic"):
                    entry["top_topic"] = c["top_topic"]
                    entry["topic_importance"] = c.get(
                        "max_topic_importance", 0,
                    )
                batch.append(entry)

            # Only include topics relevant to this chunk
            chunk_names = {
                str(c.get("contact_name", "")).lower()
                for c in chunk
            }
            relevant_topics = [
                t for t in topics_for_prompt
                if any(
                    t.get("contact", "").lower() in n
                    or n in t.get("contact", "").lower()
                    for n in chunk_names
                )
            ]

            batch_names = [c.get("name", "?") for c in batch]
            logger.info(
                "Contacts [%s]: evaluating…",
                ", ".join(batch_names),
            )
            try:
                from src.agents.contact_context.agent import (
                    ContactContextAgent,
                )

                topics_map = {
                    t.get("topic", ""): t for t in relevant_topics
                }
                batch_obj = ContactContextAgent().summarize(
                    contacts=batch,
                    topics=topics_map,
                )
                if batch_obj is None:
                    raise RuntimeError(
                        "ContactContextAgent returned None",
                    )
                # Project ContactContextDraft back into the legacy dict
                # shape the surrounding orchestration expects. The
                # downstream rescale (0-3 → 0-10 via topic boost) stays
                # in evaluate_contact_contexts.
                parsed = [
                    {
                        "contact_id": draft.contact_id,
                        "active_context": draft.active_context,
                        "context_domains": list(draft.context_domains),
                        "context_priority": draft.context_priority,
                    }
                    for draft in batch_obj.contexts
                ]
                all_results.extend(parsed)
                logger.info(
                    "Contacts batch %s: %d results",
                    batch_names, len(parsed),
                )
            except Exception:
                logger.warning(
                    "Contacts batch %s: ContactContextAgent eval failed",
                    batch_names,
                    exc_info=True,
                )

        return all_results

    def _store_contact_contexts(
        self, contexts: list[ContactContext],
    ) -> None:
        """Upsert contact contexts into DuckDB.

        sensitivity_tier: 3
        """
        for c in contexts:
            self._db.execute(
                "DELETE FROM _contact_contexts WHERE contact_id = ?",
                [c.contact_id],
            )
            domains_json = json.dumps(c.context_domains)
            self._db.execute(
                """INSERT INTO _contact_contexts
                   (contact_id, contact_name, phone, email,
                    total_messages, messages_7d, last_message_at,
                    last_message_preview, total_events, next_event_at,
                    next_event_title, active_context, context_domains,
                    context_priority, birthday, has_upcoming_birthday,
                    updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    c.contact_id, c.contact_name, c.phone, c.email,
                    c.total_messages, c.messages_7d, c.last_message_at,
                    c.last_message_preview, c.total_events, c.next_event_at,
                    c.next_event_title, c.active_context, domains_json,
                    c.context_priority, c.birthday, c.has_upcoming_birthday,
                    c.updated_at,
                ],
            )

    # ----------------------------------------------------------
    # Pillar 3: Actionable events
    # ----------------------------------------------------------

    def evaluate_actionable_events(self) -> list[ActionableEvent]:
        """Scan calendar events and birthdays for actionable items.

        Birthdays: pure SQL (no LLM needed).
        Events: batched LLM evaluation.

        sensitivity_tier: 2
        """
        results: list[ActionableEvent] = []
        now = utc_now_iso()

        # Birthday events (pure SQL — no LLM)
        try:
            results.extend(self._detect_birthdays(now))
        except Exception:
            logger.debug("Birthday detection failed", exc_info=True)

        # Calendar events needing action
        try:
            upcoming = self._sql_prefilter_events()
            if upcoming:
                llm_results = self._llm_evaluate_events(upcoming)
                for item in llm_results:
                    eid = str(item.get("event_id", ""))
                    original = next(
                        (e for e in upcoming
                         if str(e.get("id", "")) == eid),
                        None,
                    )
                    if original is None:
                        continue
                    ae = ActionableEvent(
                        id=make_hash_id("event", eid),
                        event_id=eid,
                        event_type="meeting",
                        title=str(original.get("title", "")),
                        event_date=str(
                            original.get("start_date")
                            or original.get("start_time", ""),
                        ),
                        contact_name=None,
                        action_needed=str(
                            item.get("action_needed", ""),
                        ),
                        importance=int(item.get("importance", 5)),
                        detected_at=now,
                    )
                    results.append(ae)
        except Exception:
            logger.debug("Event evaluation failed", exc_info=True)

        self._store_actionable_events(results)
        return results

    def _detect_birthdays(self, now: str) -> list[ActionableEvent]:
        """Detect upcoming birthdays from raw_contacts (pure SQL).

        sensitivity_tier: 1
        """
        if not table_exists(self._db,"raw_contacts"):
            return []

        rows = self._db.query("""
            SELECT id, name, birthday
            FROM raw_contacts
            WHERE birthday IS NOT NULL
              AND date(birthday) IS NOT NULL
              AND name IS NOT NULL AND name != ''
              AND (
                (CAST(strftime('%m', date(birthday)) AS INTEGER)
                     = CAST(strftime('%m', date('now')) AS INTEGER)
                 AND CAST(strftime('%d', date(birthday)) AS INTEGER)
                     BETWEEN CAST(strftime('%d', date('now')) AS INTEGER)
                     AND CAST(strftime('%d', date('now')) AS INTEGER) + 3)
                OR (
                  CAST(strftime('%m', date(birthday)) AS INTEGER)
                      = CAST(strftime('%m', date('now', '+3 days')) AS INTEGER)
                  AND CAST(strftime('%d', date(birthday)) AS INTEGER)
                      <= CAST(strftime('%d', date('now', '+3 days')) AS INTEGER)
                  AND CAST(strftime('%m', date('now')) AS INTEGER)
                      != CAST(strftime('%m', date('now', '+3 days')) AS INTEGER)
                )
              )
        """)

        results: list[ActionableEvent] = []
        for r in rows:
            name = str(r["name"])
            birthday_str = str(r["birthday"])
            # Calculate days until birthday
            try:
                bday = datetime.strptime(birthday_str, "%Y-%m-%d")
                today = datetime.now()
                this_year_bday = bday.replace(year=today.year)
                if this_year_bday.date() < today.date():
                    this_year_bday = bday.replace(year=today.year + 1)
                days_until = (this_year_bday.date() - today.date()).days
                if days_until == 0:
                    action = f"Today is {name}'s birthday! Send birthday wishes"
                    importance = 9
                elif days_until == 1:
                    action = (
                        f"Tomorrow is {name}'s birthday. "
                        "Consider sending birthday wishes"
                    )
                    importance = 8
                else:
                    action = (
                        f"{name}'s birthday is in {days_until} days. "
                        "Plan birthday wishes"
                    )
                    importance = 6
            except (ValueError, TypeError):
                action = f"Send birthday wishes to {name}"
                importance = 7

            ae = ActionableEvent(
                id=make_hash_id("birthday", str(r["id"])),
                event_id=str(r["id"]),
                event_type="birthday",
                title=f"{name}'s Birthday",
                event_date=birthday_str,
                contact_name=name,
                action_needed=action,
                importance=importance,
                detected_at=now,
            )
            results.append(ae)

        return results

    def _sql_prefilter_events(self) -> list[dict[str, Any]]:
        """Get upcoming calendar events (next 3 days).

        sensitivity_tier: 2
        """
        if not table_exists(self._db,"raw_calendar_events"):
            return []

        try:
            return self._db.query("""
                SELECT id, title, start_date, end_date, location, attendees
                FROM raw_calendar_events
                WHERE start_date > CURRENT_TIMESTAMP
                  AND start_date < datetime('now', '+3 days')
                ORDER BY start_date ASC
                LIMIT 20
            """)
        except Exception:
            logger.debug("Event pre-filter failed", exc_info=True)
            return []

    def _llm_evaluate_events(
        self, events: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Evaluate which events need action via :class:`ActionableEventsAgent`.

        Returns the legacy dict shape ``[{event_id, action_needed,
        importance}, ...]`` so the surrounding orchestration code can
        join in title/event_date/event_type from the original event
        records.

        sensitivity_tier: 2
        """
        from src.agents.actionable_events.agent import ActionableEventsAgent

        batch = [
            {
                "event_id": str(e.get("id", "")),
                "title": safe_str(e.get("title"), 100),
                "start_date": str(e.get("start_date", "")),
                "location": safe_str(e.get("location"), 100),
                "attendees": safe_str(e.get("attendees"), 200),
            }
            for e in events
        ]
        try:
            result = ActionableEventsAgent().detect(events=batch)
        except Exception:  # noqa: BLE001
            logger.warning(
                "ActionableEventsAgent failed", exc_info=True,
            )
            return []
        if result is None:
            return []
        return [
            {
                "event_id": draft.event_id,
                "action_needed": draft.action_needed,
                "importance": draft.importance,
            }
            for draft in result.events
        ]

    def _store_actionable_events(
        self, events: list[ActionableEvent],
    ) -> None:
        """Upsert actionable events into DuckDB.

        sensitivity_tier: 2
        """
        for e in events:
            self._db.execute(
                "DELETE FROM _actionable_events WHERE id = ?", [e.id],
            )
            self._db.execute(
                """INSERT INTO _actionable_events
                   (id, event_id, event_type, title, event_date,
                    contact_name, action_needed, importance, detected_at,
                    sensitivity_tier)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    e.id, e.event_id, e.event_type, e.title,
                    e.event_date, e.contact_name, e.action_needed,
                    e.importance, e.detected_at, e.sensitivity_tier,
                ],
            )

    # ----------------------------------------------------------
    # Pillar 4: Topic digest
    # ----------------------------------------------------------

    def build_topic_digest(self) -> list[TopicDigestEntry]:
        """Build a topic-centric digest by comparing current vs previous topics.

        Reads the current topics from int_contact_topics (or the
        ``_contact_topics_cache`` table), compares against a stored
        snapshot in ``_proactive_state``, and detects:
        - Topics with new message activity (updated)
        - Newly promoted topics (importance crossed threshold)
        - Topics being resolved/downgraded

        sensitivity_tier: 3
        """
        current_topics = self._load_current_topics()
        previous_topics = self._load_topic_snapshot()

        if not current_topics and not previous_topics:
            logger.info("Topic digest: no current or previous topics")
            return []

        digest = self._diff_topics(current_topics, previous_topics)
        self._store_topic_snapshot(current_topics)

        logger.info(
            "Topic digest: %d entries (%d current, %d previous)",
            len(digest), len(current_topics), len(previous_topics),
        )
        return digest

    def _load_current_topics(self) -> list[dict[str, Any]]:
        """Load current topics from int_contact_topics or cache.

        Falls back to ``_contact_topics_cache`` when the pipeline
        model table isn't available.

        sensitivity_tier: 3
        """
        # Try int_contact_topics first (pipeline output)
        try:
            if table_exists(self._db, "int_contact_topics"):
                rows = self._db.query("""
                    SELECT contact_name, topic, description,
                           importance, status
                    FROM int_contact_topics
                    WHERE status IN ('active', 'resolved', 'stale')
                    ORDER BY importance DESC
                """)
                if rows:
                    return rows
        except Exception:
            logger.debug(
                "int_contact_topics read failed", exc_info=True,
            )

        # Fallback: read from cache table
        try:
            if table_exists(self._db, "_contact_topics_cache"):
                rows = self._db.query("""
                    SELECT contact_name, topics_json
                    FROM _contact_topics_cache
                """)
                result: list[dict[str, Any]] = []
                for r in rows:
                    try:
                        topics = json.loads(r["topics_json"])
                    except (json.JSONDecodeError, TypeError):
                        continue
                    for t in topics:
                        if isinstance(t, dict) and t.get("topic"):
                            result.append({
                                "contact_name": r["contact_name"],
                                "topic": t["topic"],
                                "description": t.get(
                                    "description", "",
                                ),
                                "importance": t.get("importance", 5),
                                "status": t.get("status", "active"),
                            })
                return result
        except Exception:
            logger.debug(
                "Topic cache read failed", exc_info=True,
            )

        return []

    def _load_topic_snapshot(self) -> dict[str, dict[str, Any]]:
        """Load previous topic snapshot from ``_proactive_state``.

        Returns a dict keyed by ``contact_name:topic`` with the
        previous importance and status.

        sensitivity_tier: 1
        """
        try:
            rows = self._db.query(
                "SELECT value FROM _proactive_state "
                "WHERE key = 'topic_snapshot'"
            )
            if rows:
                data = json.loads(rows[0]["value"])
                if isinstance(data, dict):
                    return data
        except Exception:
            logger.debug(
                "Topic snapshot load failed", exc_info=True,
            )
        return {}

    def _store_topic_snapshot(
        self, topics: list[dict[str, Any]],
    ) -> None:
        """Store current topics as snapshot for next comparison.

        Keyed by ``contact_name:topic`` for efficient diff lookup.

        sensitivity_tier: 1
        """
        snapshot: dict[str, dict[str, Any]] = {}
        for t in topics:
            key = f"{t.get('contact_name', '')}:{t.get('topic', '')}"
            snapshot[key] = {
                "importance": t.get("importance", 5),
                "status": t.get("status", "active"),
                "description": t.get("description", ""),
                "contact_name": t.get("contact_name", ""),
                "topic": t.get("topic", ""),
            }

        try:
            self._db.execute(
                "INSERT OR REPLACE INTO _proactive_state "
                "(key, value, updated_at) VALUES (?, ?, ?)",
                [
                    "topic_snapshot",
                    json.dumps(snapshot, default=str),
                    utc_now_iso(),
                ],
            )
        except Exception:
            logger.debug(
                "Could not store topic snapshot", exc_info=True,
            )

    def _diff_topics(
        self,
        current: list[dict[str, Any]],
        previous: dict[str, dict[str, Any]],
    ) -> list[TopicDigestEntry]:
        """Compare current topics against previous snapshot.

        Produces digest entries for:
        - ``new``: topic didn't exist before and is important (>= 5)
        - ``promoted``: importance increased significantly (>= +2)
        - ``updated``: active important topic with ongoing activity
        - ``resolved``: was active, now resolved or stale

        sensitivity_tier: 3
        """
        entries: list[TopicDigestEntry] = []
        seen_keys: set[str] = set()

        importance_threshold = 5

        for t in current:
            contact = t.get("contact_name", "")
            topic = t.get("topic", "")
            key = f"{contact}:{topic}"
            seen_keys.add(key)

            importance = int(t.get("importance", 5))
            status = str(t.get("status", "active"))
            description = str(t.get("description", ""))

            prev = previous.get(key)

            if prev is None:
                # New topic — only include if important enough
                if importance >= importance_threshold:
                    entries.append(TopicDigestEntry(
                        contact_name=contact,
                        topic=topic,
                        description=description,
                        importance=importance,
                        status=status,
                        change_type="new",
                        previous_importance=None,
                    ))
                continue

            prev_importance = int(prev.get("importance", 5))
            prev_status = str(prev.get("status", "active"))

            # Promoted: importance went up by >= 2
            if importance >= prev_importance + 2:
                entries.append(TopicDigestEntry(
                    contact_name=contact,
                    topic=topic,
                    description=description,
                    importance=importance,
                    status=status,
                    change_type="promoted",
                    previous_importance=prev_importance,
                ))
            # Resolved: was active, now resolved/stale
            elif (
                prev_status == "active"
                and status in ("resolved", "stale")
            ):
                entries.append(TopicDigestEntry(
                    contact_name=contact,
                    topic=topic,
                    description=description,
                    importance=importance,
                    status=status,
                    change_type="resolved",
                    previous_importance=prev_importance,
                ))
            # Updated: active important topic still ongoing
            elif (
                status == "active"
                and importance >= importance_threshold
            ):
                entries.append(TopicDigestEntry(
                    contact_name=contact,
                    topic=topic,
                    description=description,
                    importance=importance,
                    status=status,
                    change_type="updated",
                    previous_importance=prev_importance,
                ))

        # Topics that disappeared entirely (were important, now gone)
        for key, prev in previous.items():
            if key not in seen_keys:
                prev_importance = int(prev.get("importance", 5))
                if prev_importance >= importance_threshold:
                    entries.append(TopicDigestEntry(
                        contact_name=prev.get("contact_name", ""),
                        topic=prev.get("topic", ""),
                        description=prev.get("description", ""),
                        importance=0,
                        status="resolved",
                        change_type="resolved",
                        previous_importance=prev_importance,
                    ))

        # Sort: promoted/new first, then by importance
        change_order = {
            "promoted": 0, "new": 1, "resolved": 2, "updated": 3,
        }
        entries.sort(
            key=lambda e: (
                change_order.get(e.change_type, 9),
                -e.importance,
            ),
        )

        return entries

    # ----------------------------------------------------------
    # Read-only accessors (for Dashboard — no LLM)
    # ----------------------------------------------------------

    def get_pending_replies(self, limit: int = 20) -> list[PendingReply]:
        """Return active (non-dismissed) pending replies.

        sensitivity_tier: 2
        """
        try:
            self.sweep_resolved_pending_replies()
        except Exception:  # noqa: BLE001
            logger.debug(
                "sweep_resolved_pending_replies failed", exc_info=True,
            )
        rows = self._db.query(f"""
            SELECT id, message_id, source, contact_name, domain,
                   preview, importance, reason, message_at, detected_at,
                   sensitivity_tier
            FROM _pending_replies
            WHERE dismissed_at IS NULL
            ORDER BY importance DESC, message_at DESC
            LIMIT {int(limit)}
        """)
        return [
            PendingReply(
                id=str(r["id"]),
                message_id=str(r["message_id"]),
                source=str(r["source"]),
                contact_name=str(r["contact_name"]),
                domain=str(r["domain"]),
                preview=str(r.get("preview", "")),
                importance=int(r.get("importance", 5)),
                reason=str(r.get("reason", "")),
                message_at=str(r.get("message_at", "")),
                detected_at=str(r.get("detected_at", "")),
                sensitivity_tier=int(r.get("sensitivity_tier", 2)),
            )
            for r in rows
        ]

    def get_contact_contexts(
        self, limit: int = 20,
    ) -> list[ContactContext]:
        """Return contact contexts ordered by priority.

        sensitivity_tier: 3
        """
        rows = self._db.query(f"""
            SELECT contact_id, contact_name, phone, email,
                   total_messages, messages_7d, last_message_at,
                   last_message_preview, total_events, next_event_at,
                   next_event_title, active_context, context_domains,
                   context_priority, birthday, has_upcoming_birthday,
                   updated_at
            FROM _contact_contexts
            ORDER BY context_priority DESC, messages_7d DESC
            LIMIT {int(limit)}
        """)
        results: list[ContactContext] = []
        for r in rows:
            domains_raw = r.get("context_domains", "[]")
            try:
                domains = json.loads(str(domains_raw))
            except (json.JSONDecodeError, TypeError):
                domains = []

            results.append(ContactContext(
                contact_id=str(r["contact_id"]),
                contact_name=str(r["contact_name"]),
                phone=r.get("phone"),
                email=r.get("email"),
                total_messages=int(r.get("total_messages", 0)),
                messages_7d=int(r.get("messages_7d", 0)),
                last_message_at=str(r["last_message_at"])
                if r.get("last_message_at") else None,
                last_message_preview=str(r["last_message_preview"])
                if r.get("last_message_preview") else None,
                total_events=int(r.get("total_events", 0)),
                next_event_at=str(r["next_event_at"])
                if r.get("next_event_at") else None,
                next_event_title=str(r["next_event_title"])
                if r.get("next_event_title") else None,
                active_context=str(r["active_context"])
                if r.get("active_context") else None,
                context_domains=domains,
                context_priority=int(r.get("context_priority", 0)),
                birthday=str(r["birthday"]) if r.get("birthday") else None,
                has_upcoming_birthday=bool(
                    r.get("has_upcoming_birthday", False),
                ),
                updated_at=str(r.get("updated_at", "")),
            ))
        return results

    def get_actionable_events(
        self, limit: int = 20,
    ) -> list[ActionableEvent]:
        """Return active (non-dismissed) actionable events.

        sensitivity_tier: 2
        """
        rows = self._db.query(f"""
            SELECT id, event_id, event_type, title, event_date,
                   contact_name, action_needed, importance, detected_at,
                   sensitivity_tier
            FROM _actionable_events
            WHERE dismissed_at IS NULL
            ORDER BY importance DESC, event_date ASC
            LIMIT {int(limit)}
        """)
        return [
            ActionableEvent(
                id=str(r["id"]),
                event_id=str(r["event_id"]),
                event_type=str(r["event_type"]),
                title=str(r["title"]),
                event_date=str(r["event_date"]),
                contact_name=str(r.get("contact_name"))
                if r.get("contact_name") else None,
                action_needed=str(r.get("action_needed", "")),
                importance=int(r.get("importance", 5)),
                detected_at=str(r.get("detected_at", "")),
                sensitivity_tier=int(r.get("sensitivity_tier", 2)),
            )
            for r in rows
        ]

    # ----------------------------------------------------------
    # Meeting prep briefs (deterministic, no LLM — safe on page load)
    # ----------------------------------------------------------

    def get_meeting_prep_briefs(
        self,
        *,
        within_hours: float = 24.0,
        limit: int = 10,
        now: datetime | None = None,
    ) -> list[MeetingPrepBrief]:
        """Compose prep packs for upcoming meetings with known attendees.

        For each calendar event starting within ``within_hours`` that
        has at least one known attendee, gather every attendee's
        situation summary + active domains (from the cached
        ``_contact_contexts``), their open loops (from
        ``int_contact_topics``), and their last message — a "here's
        where things stand" card. Pure joins over already-computed
        data: no LLM, so it's cheap enough to build on read.

        Returns briefs soonest-first. Empty when the enriched-events
        mart is absent (pipeline hasn't run) or nothing is upcoming.

        sensitivity_tier: 3
        """
        if not table_exists(self._db, "int_events_enriched"):
            return []

        now = now or datetime.now()
        try:
            rows = self._db.query("""
                SELECT id, title, start_time, location,
                       known_attendee_names
                FROM int_events_enriched
                WHERE known_attendee_names IS NOT NULL
                  AND TRIM(known_attendee_names) != ''
                  AND start_time IS NOT NULL
                ORDER BY start_time ASC
            """)
        except Exception:  # noqa: BLE001
            logger.debug("meeting-prep event scan failed", exc_info=True)
            return []

        upcoming = [
            r for r in rows
            if _within_window(str(r.get("start_time") or ""), now, within_hours)
        ]
        if not upcoming:
            return []

        contexts = {
            c.contact_name.strip().lower(): c
            for c in self.get_contact_contexts(limit=200)
        }
        loops_by_contact = self._open_loops_by_contact()

        briefs: list[MeetingPrepBrief] = []
        for r in upcoming[: max(0, limit)]:
            names = [
                n.strip()
                for n in str(r.get("known_attendee_names") or "").split(",")
                if n.strip()
            ]
            attendees: list[MeetingPrepAttendee] = []
            seen: set[str] = set()
            for name in names:
                key = name.lower()
                if key in seen:
                    continue
                seen.add(key)
                ctx = contexts.get(key)
                attendees.append(MeetingPrepAttendee(
                    contact_name=(ctx.contact_name if ctx else name),
                    situation=ctx.active_context if ctx else None,
                    domains=list(ctx.context_domains) if ctx else [],
                    last_message_at=ctx.last_message_at if ctx else None,
                    last_message_preview=(
                        ctx.last_message_preview if ctx else None
                    ),
                    open_loops=loops_by_contact.get(key, []),
                ))
            # A brief where no attendee carries any situation, open loop,
            # or last message is a contextless name echo (e.g. a solo
            # calendar hold) — skip it rather than clutter the surface.
            if not any(
                a.situation or a.open_loops or a.last_message_preview
                for a in attendees
            ):
                continue
            briefs.append(MeetingPrepBrief(
                event_id=str(r.get("id") or ""),
                title=str(r.get("title") or "Untitled event"),
                start_time=str(r.get("start_time") or ""),
                location=(
                    str(r["location"]) if r.get("location") else None
                ),
                attendees=attendees,
                generated_at=utc_now_iso(),
            ))
        return briefs

    def _open_loops_by_contact(
        self, per_contact: int = 3,
    ) -> dict[str, list[dict[str, Any]]]:
        """Active per-contact topics, keyed by lower-cased contact name.

        The "open loops" going into a meeting. Reads the
        ``int_contact_topics`` mart; returns ``{}`` when absent.

        sensitivity_tier: 3
        """
        if not table_exists(self._db, "int_contact_topics"):
            return {}
        try:
            rows = self._db.query("""
                SELECT contact_name, topic, description, importance
                FROM int_contact_topics
                WHERE status = 'active'
                ORDER BY importance DESC
            """)
        except Exception:  # noqa: BLE001
            return {}
        out: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            key = str(r.get("contact_name") or "").strip().lower()
            if not key:
                continue
            bucket = out.setdefault(key, [])
            if len(bucket) >= per_contact:
                continue
            bucket.append({
                "topic": str(r.get("topic") or ""),
                "description": safe_str(r.get("description"), 200),
                "importance": int(r.get("importance") or 0),
            })
        return out

    # ----------------------------------------------------------
    # User actions
    # ----------------------------------------------------------

    def dismiss_pending_reply(self, reply_id: str) -> None:
        """Mark a pending reply as dismissed.

        sensitivity_tier: 1
        """
        self._db.execute(
            "UPDATE _pending_replies SET dismissed_at = CURRENT_TIMESTAMP "
            "WHERE id = ?",
            [reply_id],
        )

    def dismiss_actionable_event(self, event_id: str) -> None:
        """Mark an actionable event as dismissed.

        sensitivity_tier: 1
        """
        self._db.execute(
            "UPDATE _actionable_events SET dismissed_at = CURRENT_TIMESTAMP "
            "WHERE id = ?",
            [event_id],
        )

    # ----------------------------------------------------------
    # Cleanup
    # ----------------------------------------------------------

    def _clean_stale_entries(self) -> None:
        """Remove old entries to prevent table bloat.

        sensitivity_tier: 1
        """
        # Pending replies older than 48h
        self._db.execute(
            "DELETE FROM _pending_replies "
            "WHERE detected_at < datetime('now', '-48 hours')",
        )
        # Actionable events in the past
        self._db.execute(
            "DELETE FROM _actionable_events "
            "WHERE event_date < datetime('now', '-1 day') "
            "AND event_type != 'birthday'",
        )
        # Old dismissed birthdays
        self._db.execute(
            "DELETE FROM _actionable_events "
            "WHERE dismissed_at IS NOT NULL "
            "AND dismissed_at < datetime('now', '-7 days')",
        )

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------

