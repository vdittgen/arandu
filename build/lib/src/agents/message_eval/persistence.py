"""Topic-driven message evaluation for proactive notifications.

Evaluates newly ingested messages against important topics from
mart_contact_summary to identify:
1. Topic actions — user needs to act on an important topic
2. Topic enrichment — new info that enriches ongoing important conversations
3. Conversation digest — periodic 2h summary of general conversations

Only messages related to contacts with active high-importance topics
trigger real-time notifications. Everything else goes to the digest.

Triggered after each sync that produces new message/email rows.
Uses batched LLM calls (1 per sync cycle) for efficiency.

sensitivity_tier: 3 (processes personal messages through LLM)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from src.agents.triage.persistence import MessageTriager
from src.core.db_helpers import (
    ensure_tables,
    get_table_columns,
    make_hash_id,
    safe_str,
    table_exists,
    utc_now_iso,
)
from src.core.sqlite.engine import DatabaseEngine
from src.core.topic_loader import (
    get_topic_contacts_for_prompt,
    load_pending_reply_ids,
    load_today_events,
    load_topic_contacts,
    sync_topics_table_from_cache,
)

logger = logging.getLogger(__name__)


_DEFAULT_EVAL_WINDOW_HOURS = 4.0


def _load_window_hours(key: str, default: float) -> float:
    """Read a window-size override from settings.json.

    Falls back to ``default`` when the setting is missing or invalid.
    Negative values are clamped to ``default``.

    sensitivity_tier: 1
    """
    try:
        from src.models.llm_provider import load_llm_settings
    except Exception:  # noqa: BLE001
        return default
    raw = load_llm_settings().get(key)
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _eval_window_hours() -> float:
    """sensitivity_tier: 1"""
    return _load_window_hours("eval_window_hours", _DEFAULT_EVAL_WINDOW_HOURS)


def _sqlite_now_minus_hours(hours: float) -> str:
    """Format the ``datetime('now', '-Xh')`` argument for SQLite.

    SQLite's modifiers accept whole units only, so fractional hours are
    converted to whole minutes.  ``2.0`` → ``'-2 hours'``, ``0.5`` →
    ``'-30 minutes'``.

    sensitivity_tier: 1
    """
    if hours >= 1 and float(hours).is_integer():
        return f"-{int(hours)} hours"
    return f"-{max(1, int(round(hours * 60)))} minutes"


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------


@dataclass(frozen=True)
class MessageNotification:
    """A notification generated from real-time message evaluation.

    sensitivity_tier: 2
    """

    id: str
    message_ids: list[str] = field(default_factory=list)
    notification_type: str = ""  # "topic_action" or "topic_enrichment"
    importance: int = 0
    domain: str = "general"
    summary: str = ""
    contacts: list[str] = field(default_factory=list)
    related_context: str = ""
    created_at: str = ""


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

_MESSAGE_TABLES = frozenset({"raw_messages"})
_EMAIL_TABLES = frozenset({"raw_emails"})
_ALL_MESSAGE_TABLES = _MESSAGE_TABLES | _EMAIL_TABLES

# Connectors that produce messages/emails
MESSAGE_CONNECTORS: frozenset[str] = frozenset({
    "whatsapp", "apple-messages", "apple-mail",
})

# Map connector to target tables
_CONNECTOR_TABLES: dict[str, list[str]] = {
    "whatsapp": ["raw_messages"],
    "apple-messages": ["raw_messages"],
    "apple-mail": ["raw_emails"],
}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------



# ------------------------------------------------------------------
# MessageEvaluator
#
# LLM evaluation is delegated to :class:`MessageEvaluatorAgent`
# (pydantic-ai). This module retains DB scans, topic loading, today's
# events lookup, and the ``_evaluated_messages`` /
# ``_message_notifications`` / ``_topics`` table writes.
# ------------------------------------------------------------------


class MessageEvaluator:
    """Real-time post-sync evaluation of newly ingested messages.

    SQL prefilter → context building → optional pre-scoring →
    batched LLM evaluation → store results.

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
            CREATE TABLE IF NOT EXISTS _evaluated_messages (
                message_id      VARCHAR PRIMARY KEY,
                source_table    VARCHAR NOT NULL,
                connector_id    VARCHAR NOT NULL,
                evaluated_at    TEXT DEFAULT CURRENT_TIMESTAMP,
                notification_sent INTEGER DEFAULT 0,
                notification_type VARCHAR,
                importance      INTEGER DEFAULT 0
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS _message_notifications (
                id              VARCHAR PRIMARY KEY,
                message_ids     VARCHAR NOT NULL,
                notification_type VARCHAR NOT NULL,
                importance      INTEGER NOT NULL,
                domain          VARCHAR NOT NULL,
                summary         TEXT NOT NULL,
                contacts        VARCHAR DEFAULT '[]',
                related_context TEXT,
                created_at      TEXT DEFAULT CURRENT_TIMESTAMP,
                notified_at     TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS _topics (
                id               VARCHAR PRIMARY KEY,
                contact_name     VARCHAR NOT NULL,
                topic            VARCHAR NOT NULL,
                description      TEXT,
                importance       INTEGER DEFAULT 5,
                status           VARCHAR DEFAULT 'active',
                source           VARCHAR DEFAULT 'evaluator',
                first_seen       TEXT,
                last_seen        TEXT,
                message_count    INTEGER DEFAULT 1,
                sensitivity_tier INTEGER DEFAULT 3,
                category         VARCHAR,
                linked_goal_id   VARCHAR
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_topics_active
            ON _topics (status, importance)
            """,
        ])
        # Hydrate _topics from the pipeline's _contact_topics_cache so
        # the evaluator never has to bootstrap topics from raw messages.
        try:
            sync_topics_table_from_cache(self._db)
        except Exception:  # noqa: BLE001
            logger.debug(
                "Initial _topics hydration skipped", exc_info=True,
            )

    # ----------------------------------------------------------
    # Topic decay
    # ----------------------------------------------------------

    def _decay_stale_topics(self) -> None:
        """Mark old topics as stale/resolved based on last_seen age.

        Called at most once per hour (guarded by _proactive_state).

        sensitivity_tier: 1
        """
        try:
            self._db.execute("""
                UPDATE _topics
                SET status = 'stale'
                WHERE status = 'active'
                  AND last_seen < datetime('now', '-14 days')
            """)
            self._db.execute("""
                UPDATE _topics
                SET status = 'resolved'
                WHERE status = 'stale'
                  AND last_seen < datetime('now', '-30 days')
            """)
        except Exception:
            logger.debug("Topic decay failed", exc_info=True)

    # ----------------------------------------------------------
    # Main entry
    # ----------------------------------------------------------

    def evaluate_new_messages(
        self,
        connector_id: str,
        target_table: str,
    ) -> list[MessageNotification]:
        """Evaluate newly ingested messages for a connector/table.

        Pipeline: SQL prefilter → AI triage (keep/drop) → topic-aware
        LLM evaluation.  Returns notifications with importance >= 7.

        sensitivity_tier: 3
        """
        # Decay stale topics periodically
        self._decay_stale_topics()

        # Step 1: get unevaluated messages (SQL prefilter: recency,
        # not-from-me, active groups, dedup against _evaluated_messages)
        candidates = self._get_unevaluated_messages(target_table)
        if not candidates:
            logger.info(
                "MessageEvaluator: no new messages in %s",
                target_table,
            )
            return []

        logger.info(
            "MessageEvaluator: %d new messages in %s",
            len(candidates), target_table,
        )

        # Step 2: AI triage (replaces the old keyword/regex scoring).
        # Already-pending messages are excluded so we never re-notify.
        existing_pending = load_pending_reply_ids(self._db)
        candidates = [
            c for c in candidates
            if str(c.get("id", "")) not in existing_pending
        ]
        if not candidates:
            return []

        triager = MessageTriager(self._db)
        decisions = triager.triage(candidates)
        kept = [
            c for c, d in zip(candidates, decisions, strict=False)
            if d.keep
        ]
        logger.info(
            "MessageEvaluator: %d/%d candidates passed triage",
            len(kept), len(candidates),
        )

        if not kept:
            self._mark_evaluated(candidates, connector_id, target_table)
            return []

        # Cap at 20 to bound LLM prompt size.
        kept = kept[:20]

        # Step 3: build evaluation context and run topic-aware LLM eval
        context = self._build_evaluation_context()
        llm_results = self._llm_evaluate(kept, context)

        # Step 4: store results and mark ALL candidates as evaluated
        notifications = self._store_results(
            llm_results, candidates, connector_id, target_table,
        )

        return [n for n in notifications if n.importance >= 7]

    # ----------------------------------------------------------
    # Step 1: Get unevaluated messages
    # ----------------------------------------------------------

    def _get_unevaluated_messages(
        self,
        target_table: str,
    ) -> list[dict[str, Any]]:
        """Get messages not yet evaluated from target table.

        sensitivity_tier: 2
        """
        if target_table not in _ALL_MESSAGE_TABLES:
            return []

        if target_table in _MESSAGE_TABLES:
            return self._get_unevaluated_chat_messages()
        return self._get_unevaluated_emails()

    def _get_unevaluated_chat_messages(self) -> list[dict[str, Any]]:
        """Get unevaluated chat messages from raw_messages.

        sensitivity_tier: 2
        """
        try:
            cols = get_table_columns(self._db,"raw_messages")
            has_sender_name = "sender_name" in cols
            has_chat_name = "chat_name" in cols
            has_is_group = "is_group" in cols

            sender_col = "sender_name" if has_sender_name else "sender"
            extra_cols = ""
            if has_chat_name:
                extra_cols += ", m.chat_name"
            if has_is_group:
                extra_cols += ", m.is_group"

            # Filter: not from me, recent (4h), not already evaluated.
            # Skip group messages from groups where the user hasn't
            # posted in the last 7 days — avoids wasting LLM calls
            # on read-only groups.
            is_group_order = (
                "m.is_group ASC NULLS FIRST, "
                if has_is_group else ""
            )

            active_group_filter = ""
            if has_is_group and has_chat_name:
                active_group_filter = """
                  AND (
                    m.is_group = 0
                    OR m.chat_name IN (
                      SELECT DISTINCT chat_name
                      FROM raw_messages
                      WHERE is_from_me = 1
                        AND chat_name IS NOT NULL
                        AND timestamp > datetime('now', '-7 days')
                    )
                  )
                """

            window_modifier = _sqlite_now_minus_hours(_eval_window_hours())
            rows = self._db.query(
                f"""
                SELECT m.id, m.source, m.sender,
                       m.{sender_col} as sender_name,
                       m.content, m.timestamp
                       {extra_cols}
                FROM raw_messages m
                LEFT JOIN _evaluated_messages e
                    ON e.message_id = CAST(m.id AS VARCHAR)
                WHERE e.message_id IS NULL
                  AND m.is_from_me = false
                  AND m.timestamp > datetime('now', ?)
                  {active_group_filter}
                ORDER BY {is_group_order}m.timestamp DESC
                LIMIT 50
                """,
                [window_modifier],
            )
            for r in rows:
                r["_table"] = "raw_messages"
            return rows
        except Exception:
            logger.debug(
                "Chat message pre-filter failed",
                exc_info=True,
            )
            return []

    def _get_unevaluated_emails(self) -> list[dict[str, Any]]:
        """Get unevaluated emails from raw_emails.

        sensitivity_tier: 2
        """
        try:
            if not table_exists(self._db,"raw_emails"):
                return []

            # No is_read filter — triage drops newsletters and promo
            # regardless of read state.  Recency stays in SQL.
            window_modifier = _sqlite_now_minus_hours(_eval_window_hours())
            rows = self._db.query(
                """
                SELECT e.id, 'email' as source,
                       e.from_address as sender_name,
                       e.subject,
                       e.body_preview as content,
                       e.date as timestamp
                FROM raw_emails e
                LEFT JOIN _evaluated_messages ev
                    ON ev.message_id = CAST(e.id AS VARCHAR)
                WHERE ev.message_id IS NULL
                  AND e.date > datetime('now', ?)
                ORDER BY e.date DESC
                LIMIT 20
                """,
                [window_modifier],
            )
            for r in rows:
                r["_table"] = "raw_emails"
            return rows
        except Exception:
            logger.debug(
                "Email pre-filter failed",
                exc_info=True,
            )
            return []

    # ----------------------------------------------------------
    # Step 2: Build evaluation context
    # ----------------------------------------------------------

    def _build_evaluation_context(self) -> dict[str, Any]:
        """Gather topic-driven context for the downstream LLM call.

        Triage already removed trash, so this only needs topic and
        calendar context.  Pending-ID filtering is applied upstream
        in :meth:`evaluate_new_messages`.

        sensitivity_tier: 2
        """
        return {
            "topic_contacts": load_topic_contacts(self._db),
            "today_events": load_today_events(self._db),
            "existing_pending_ids": load_pending_reply_ids(self._db),
        }

    # ----------------------------------------------------------
    # Step 3: Batch LLM evaluation
    # ----------------------------------------------------------

    def _llm_evaluate(
        self,
        candidates: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Delegate batch evaluation to :class:`MessageEvaluatorAgent`.

        Returns the legacy dict shape so ``_store_results`` keeps
        working unchanged.

        sensitivity_tier: 3
        """
        from src.agents.message_eval.agent import MessageEvaluatorAgent

        batch = [
            {
                "message_id": str(c.get("id", "")),
                "source": str(c.get("source", "")),
                "sender": str(
                    c.get("sender_name")
                    or c.get("sender", "Unknown"),
                ),
                "content": safe_str(c.get("content", ""), 200),
                "timestamp": str(c.get("timestamp", "")),
            }
            for c in candidates
        ]
        topics_map = {
            t.get("topic", ""): t
            for t in get_topic_contacts_for_prompt(
                context.get("topic_contacts", {}),
            )
        }
        try:
            logger.info(
                "MessageEvaluatorAgent: %d messages", len(candidates),
            )
            result = MessageEvaluatorAgent().evaluate(
                messages=batch,
                topics=topics_map,
                today_events=list(context.get("today_events", []) or []),
                existing_pending_ids=list(
                    context.get("existing_pending_ids", set()),
                ),
            )
        except Exception:
            logger.warning(
                "MessageEvaluatorAgent failed", exc_info=True,
            )
            return []
        if result is None:
            return []
        # Keep the legacy threshold and dict shape so _store_results
        # downstream consumers stay unchanged.
        filtered: list[dict[str, Any]] = []
        for draft in result.notifications:
            if draft.importance < 7:
                continue
            filtered.append({
                "message_id": draft.message_id,
                "type": draft.notification_type,
                "importance": draft.importance,
                "domain": draft.domain,
                "summary": draft.summary,
                "related_to": draft.related_to,
            })
        logger.info(
            "MessageEvaluatorAgent: %d drafts, %d with importance >= 7",
            len(result.notifications), len(filtered),
        )
        return filtered

    # ----------------------------------------------------------
    # Step 5: Store results
    # ----------------------------------------------------------

    def _store_results(
        self,
        llm_results: list[dict[str, Any]],
        all_candidates: list[dict[str, Any]],
        connector_id: str,
        target_table: str,
    ) -> list[MessageNotification]:
        """Store evaluation results and mark messages as evaluated.

        sensitivity_tier: 2
        """
        now = utc_now_iso()
        notifications: list[MessageNotification] = []

        # Build lookup for LLM-flagged messages
        flagged: dict[str, dict[str, Any]] = {}
        for item in llm_results:
            msg_id = str(item.get("message_id", ""))
            if msg_id:
                flagged[msg_id] = item

        # Mark ALL candidates as evaluated
        for msg in all_candidates:
            msg_id = str(msg.get("id", ""))
            if not msg_id:
                continue

            is_flagged = msg_id in flagged
            flag_data = flagged.get(msg_id, {})
            notif_type = str(flag_data.get("type", "")) if is_flagged else None
            importance = (
                int(flag_data.get("importance", 0)) if is_flagged else 0
            )

            try:
                self._db.execute(
                    """
                    INSERT INTO _evaluated_messages
                        (message_id, source_table, connector_id,
                         evaluated_at, notification_sent,
                         notification_type, importance)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        msg_id, target_table, connector_id,
                        now, is_flagged, notif_type, importance,
                    ],
                )
            except Exception:
                # Duplicate key — already evaluated (race condition)
                logger.debug(
                    "Message %s already in _evaluated_messages",
                    msg_id,
                )

        # Create notification records for flagged messages
        for item in llm_results:
            msg_id = str(item.get("message_id", ""))
            if not msg_id:
                continue

            notif_id = make_hash_id("msg_notif", msg_id, now)
            notif_type = str(item.get("type", "awareness"))
            importance = int(item.get("importance", 5))
            domain = str(item.get("domain", "general"))
            summary = str(item.get("summary", ""))
            related = str(item.get("related_to", ""))

            # Find sender name from candidates
            original = next(
                (c for c in all_candidates
                 if str(c.get("id", "")) == msg_id),
                None,
            )
            sender = str(
                (original or {}).get("sender_name")
                or (original or {}).get("sender", "Unknown"),
            )

            notif = MessageNotification(
                id=notif_id,
                message_ids=[msg_id],
                notification_type=notif_type,
                importance=importance,
                domain=domain,
                summary=summary,
                contacts=[sender],
                related_context=related,
                created_at=now,
            )
            notifications.append(notif)

            try:
                self._db.execute(
                    """
                    INSERT INTO _message_notifications
                        (id, message_ids, notification_type,
                         importance, domain, summary, contacts,
                         related_context, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        notif_id,
                        json.dumps([msg_id]),
                        notif_type,
                        importance,
                        domain,
                        summary,
                        json.dumps([sender]),
                        related,
                        now,
                    ],
                )
            except Exception:
                logger.debug(
                    "Failed to store notification %s",
                    notif_id,
                    exc_info=True,
                )

            # Enrich _topics when LLM links message to a topic
            if related and sender and sender != "Unknown":
                self._upsert_topic_from_eval(
                    sender, related, summary, importance, now,
                )

        return notifications

    def _upsert_topic_from_eval(
        self,
        contact_name: str,
        topic: str,
        description: str,
        importance: int,
        now: str,
    ) -> None:
        """Upsert a topic entry discovered during message evaluation.

        Zero additional LLM cost — piggybacks on the evaluation that
        already determined ``related_to`` and ``importance``.

        sensitivity_tier: 3
        """
        topic_id = make_hash_id(
            "topic", contact_name.lower(), topic.lower(),
        )
        try:
            # Try update first (existing topic)
            existing = self._db.query(
                "SELECT id, importance FROM _topics WHERE id = ?",
                [topic_id],
            )
            if existing:
                new_imp = max(existing[0]["importance"], importance)
                self._db.execute(
                    """UPDATE _topics
                       SET last_seen = ?, message_count = message_count + 1,
                           importance = ?, status = 'active',
                           description = CASE
                               WHEN ? > importance THEN ? ELSE description
                           END
                       WHERE id = ?""",
                    [now, new_imp, importance, description, topic_id],
                )
            else:
                self._db.execute(
                    """INSERT INTO _topics
                       (id, contact_name, topic, description,
                        importance, status, source,
                        first_seen, last_seen, sensitivity_tier)
                    VALUES (?, ?, ?, ?, ?, 'active', 'evaluator',
                            ?, ?, 3)""",
                    [
                        topic_id, contact_name, topic,
                        description, importance, now, now,
                    ],
                )
        except Exception:
            logger.debug(
                "Topic upsert failed for %s/%s",
                contact_name, topic, exc_info=True,
            )

    def _mark_evaluated(
        self,
        candidates: list[dict[str, Any]],
        connector_id: str,
        target_table: str,
    ) -> None:
        """Mark all candidates as evaluated (no notifications).

        sensitivity_tier: 1
        """
        now = utc_now_iso()
        for msg in candidates:
            msg_id = str(msg.get("id", ""))
            if not msg_id:
                continue
            try:
                self._db.execute(
                    """
                    INSERT INTO _evaluated_messages
                        (message_id, source_table, connector_id,
                         evaluated_at, notification_sent,
                         notification_type, importance)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        msg_id, target_table, connector_id,
                        now, False, None, 0,
                    ],
                )
            except Exception:
                pass  # Already evaluated

# ------------------------------------------------------------------
# Notification formatting
# ------------------------------------------------------------------


def format_realtime_notification(
    items: list[MessageNotification],
) -> str:
    """Format topic-driven evaluation results into a notification.

    sensitivity_tier: 2
    """
    actions = [
        i for i in items if i.notification_type == "topic_action"
    ]
    enrichment = [
        i for i in items if i.notification_type == "topic_enrichment"
    ]

    lines = ["Arandu - New Messages\n"]

    if actions:
        lines.append("Action Required:")
        for a in actions[:3]:
            contact = a.contacts[0] if a.contacts else ""
            prefix = f"{contact}: " if contact else ""
            lines.append(f"- {prefix}{a.summary}")
        lines.append("")

    if enrichment:
        lines.append("New Info:")
        for a in enrichment[:3]:
            contact = a.contacts[0] if a.contacts else ""
            prefix = f"{contact}: " if contact else ""
            lines.append(f"- {prefix}{a.summary}")

    return "\n".join(lines)
