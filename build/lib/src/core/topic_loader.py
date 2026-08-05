"""Shared topic intelligence and context loader.

Provides reusable functions to load contact topic importance, group
engagement data, upcoming events, and pending reply IDs from the
database — shared context used by multiple inference components.

Used by: message_evaluator, proactive_intelligence, query_engine,
insight_generator — all need the same context signals without
duplicating the SQL queries.

sensitivity_tier: 2 (reads contact names, topic importance scores,
event titles, message IDs)
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _topic_id(contact: str, topic: str) -> str:
    """Stable ID matching :func:`src.core.db_helpers.make_hash_id`
    output for ``("topic", contact.lower(), topic.lower())``.
    """
    raw = f"topic|{contact.lower()}|{topic.lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def sync_topics_table_from_cache(db_engine: Any) -> int:
    """Hydrate the ``_topics`` runtime table from ``_contact_topics_cache``.

    ``int_contact_topics`` is the source of truth for topic extraction;
    ``_topics`` exists only for the runtime evaluator to enrich and
    decay live.  This helper copies cached LLM verdicts into ``_topics``
    so MessageEvaluator never has to bootstrap from raw messages.

    Idempotent: missing topics are inserted, existing topics keep their
    runtime metadata (last_seen, message_count) untouched.

    Returns the number of rows inserted.

    sensitivity_tier: 2
    """
    try:
        cache_rows = db_engine.query(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='_contact_topics_cache'",
        )
        if not cache_rows:
            return 0
    except Exception:  # noqa: BLE001
        logger.debug("Topic cache existence check failed", exc_info=True)
        return 0

    try:
        rows = db_engine.query(
            "SELECT contact_name, topics_json, extracted_at "
            "FROM _contact_topics_cache",
        )
    except Exception:  # noqa: BLE001
        logger.debug("Topic cache read failed", exc_info=True)
        return 0

    inserted = 0
    for row in rows:
        contact = str(row.get("contact_name", "")).strip()
        if not contact:
            continue
        try:
            topics = json.loads(row.get("topics_json") or "[]")
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(topics, list):
            continue
        seen_at = str(row.get("extracted_at", "")) or None
        for t in topics:
            if not isinstance(t, dict):
                continue
            topic = str(t.get("topic", "")).strip()
            if not topic or len(topic) < 3:
                continue
            tid = _topic_id(contact, topic)
            try:
                importance = max(1, min(10, int(t.get("importance", 5))))
            except (TypeError, ValueError):
                importance = 5
            status = str(t.get("status", "active")).lower()
            if status not in ("active", "resolved", "stale"):
                status = "active"
            description = str(t.get("description", ""))[:500]
            raw_cat = t.get("category")
            if isinstance(raw_cat, str):
                cat = raw_cat.lower().strip()
                category = cat if cat in ("personal", "life", "work") else None
            else:
                category = None
            try:
                db_engine.execute(
                    """INSERT OR IGNORE INTO _topics
                       (id, contact_name, topic, description,
                        importance, status, source,
                        first_seen, last_seen, sensitivity_tier, category)
                    VALUES (?, ?, ?, ?, ?, ?, 'int_contact_topics',
                            ?, ?, 3, ?)""",
                    [
                        tid, contact, topic, description,
                        importance, status, seen_at, seen_at, category,
                    ],
                )
                inserted += 1
            except Exception:  # noqa: BLE001
                # category column might not exist yet on legacy DBs; retry
                # without it so existing topic hydration still works.
                try:
                    db_engine.execute(
                        """INSERT OR IGNORE INTO _topics
                           (id, contact_name, topic, description,
                            importance, status, source,
                            first_seen, last_seen, sensitivity_tier)
                        VALUES (?, ?, ?, ?, ?, ?, 'int_contact_topics',
                                ?, ?, 3)""",
                        [
                            tid, contact, topic, description,
                            importance, status, seen_at, seen_at,
                        ],
                    )
                    inserted += 1
                except Exception:  # noqa: BLE001
                    logger.debug(
                        "Topic hydration failed for %s/%s",
                        contact, topic, exc_info=True,
                    )
    if inserted:
        logger.info(
            "Hydrated %d topics from _contact_topics_cache into _topics",
            inserted,
        )
    return inserted


def load_topic_contacts(
    db_engine: Any,
    min_importance: int = 5,
    limit: int = 15,
) -> dict[str, dict[str, Any]]:
    """Load contacts with active important topics.

    Uses a two-source priority cascade:
    1. ``_topics`` table (primary, maintained by MessageEvaluator)
    2. ``mart_contact_summary`` (fallback, pipeline output)

    Args:
        db_engine: Database engine with ``query()`` and ``execute()``.
        min_importance: Minimum topic importance threshold.
        limit: Maximum number of contacts to return.

    Returns:
        Dict mapping ``contact_name.lower()`` to a dict with keys:
        ``name``, ``importance``, ``topics``, ``messages_7d``,
        ``notification_priority``, ``top_topic``.

    sensitivity_tier: 2
    """
    # Try _topics table first (most up-to-date)
    result = _load_from_topics_table(db_engine, min_importance, limit)
    if result:
        return result

    # Fallback to mart_contact_summary (pipeline output)
    return _load_from_mart_contact_summary(
        db_engine, min_importance, limit,
    )


def _load_from_topics_table(
    db_engine: Any,
    min_importance: int,
    limit: int,
) -> dict[str, dict[str, Any]]:
    """Load topic contacts from the ``_topics`` table.

    Groups topics by contact and aggregates into the standard format.

    sensitivity_tier: 2
    """
    result: dict[str, dict[str, Any]] = {}

    try:
        tables = db_engine.query(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='_topics'",
        )
        if not tables:
            return result

        rows = db_engine.query(f"""
            SELECT contact_name, topic, description,
                   importance, message_count
            FROM _topics
            WHERE status = 'active'
              AND importance >= {int(min_importance)}
            ORDER BY importance DESC, last_seen DESC
        """)

        if not rows:
            return result

        # Group by contact
        by_contact: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            name = r["contact_name"]
            by_contact.setdefault(name, []).append(r)

        # Build result dict (sorted by max importance, capped)
        contacts_sorted = sorted(
            by_contact.items(),
            key=lambda kv: max(t["importance"] for t in kv[1]),
            reverse=True,
        )[:limit]

        for name, topics_raw in contacts_sorted:
            max_imp = max(t["importance"] for t in topics_raw)
            topics = [
                {
                    "topic": t["topic"],
                    "description": t.get("description", ""),
                    "importance": t["importance"],
                }
                for t in topics_raw
            ]
            top_topic = max(
                topics_raw, key=lambda t: t["importance"],
            )["topic"]

            result[name.lower()] = {
                "name": name,
                "importance": max_imp,
                "topics": topics,
                "messages_7d": sum(
                    t.get("message_count", 0) for t in topics_raw
                ),
                "notification_priority": max_imp * 10,
                "top_topic": top_topic,
            }
    except Exception:
        logger.debug("_topics table fetch failed", exc_info=True)

    return result


def _load_from_mart_contact_summary(
    db_engine: Any,
    min_importance: int,
    limit: int,
) -> dict[str, dict[str, Any]]:
    """Fallback: load topic contacts from ``mart_contact_summary``.

    sensitivity_tier: 2
    """
    result: dict[str, dict[str, Any]] = {}

    try:
        tables = db_engine.query(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='mart_contact_summary'",
        )
        if not tables:
            return result

        rows = db_engine.query(f"""
            SELECT contact_name, top_topic,
                   max_topic_importance,
                   active_topics_json,
                   notification_priority,
                   messages_7d
            FROM mart_contact_summary
            WHERE max_topic_importance >= {int(min_importance)}
              AND top_topic IS NOT NULL
            ORDER BY notification_priority DESC
            LIMIT {int(limit)}
        """)

        for r in rows:
            name = r["contact_name"]
            topics_raw = r.get("active_topics_json")
            topics: list[dict[str, Any]] = []
            if topics_raw:
                try:
                    topics = json.loads(topics_raw)
                except (json.JSONDecodeError, TypeError):
                    pass
            if not topics:
                topics = [{"topic": r["top_topic"]}]

            result[name.lower()] = {
                "name": name,
                "importance": r["max_topic_importance"],
                "topics": topics,
                "messages_7d": r.get("messages_7d", 0),
                "notification_priority": r.get(
                    "notification_priority", 0,
                ),
                "top_topic": r["top_topic"],
            }
    except Exception:
        logger.debug("Topic contacts fetch failed", exc_info=True)

    return result


def load_group_engagement(
    db_engine: Any,
) -> dict[str, dict[str, Any]]:
    """Load group engagement stats from raw_messages.

    Returns a dict keyed by chat_name with message counts, user
    participation (sent), and member count.

    Args:
        db_engine: Database engine with ``query()``.

    Returns:
        Dict mapping ``chat_name`` to a dict with keys:
        ``sent``, ``members``, ``total``.

    sensitivity_tier: 2
    """
    result: dict[str, dict[str, Any]] = {}

    try:
        rows = db_engine.query("""
            SELECT chat_name,
                   COUNT(*) as total,
                   SUM(CASE WHEN is_from_me = 1
                       THEN 1 ELSE 0 END) as sent,
                   COUNT(DISTINCT sender) as members
            FROM raw_messages
            WHERE is_group = 1
            GROUP BY chat_name
        """)
        result = {
            r["chat_name"]: {
                "sent": r["sent"],
                "members": r["members"],
                "total": r["total"],
            }
            for r in rows
        }
    except Exception:
        logger.debug("Group engagement fetch failed", exc_info=True)

    return result


def get_topic_contacts_for_prompt(
    topic_contacts: dict[str, dict[str, Any]],
    max_contacts: int = 10,
) -> list[dict[str, Any]]:
    """Format topic contacts for LLM prompt injection.

    Converts the internal topic_contacts dict into a list of dicts
    suitable for JSON serialization in LLM prompts.

    Args:
        topic_contacts: Output of :func:`load_topic_contacts`.
        max_contacts: Maximum contacts to include.

    Returns:
        List of dicts with ``contact``, ``importance``, ``topics``.

    sensitivity_tier: 2
    """
    items = sorted(
        topic_contacts.values(),
        key=lambda tc: tc.get("notification_priority", 0),
        reverse=True,
    )[:max_contacts]

    return [
        {
            "contact": tc["name"],
            "importance": tc["importance"],
            "topics": [
                t.get("topic", "") for t in tc.get("topics", [])
            ],
        }
        for tc in items
    ]


def load_today_events(
    db_engine: Any,
    days_ahead: int = 1,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Load upcoming calendar events for LLM context injection.

    Queries ``raw_calendar_events`` for events within the next
    ``days_ahead`` days. Detects whether the table uses
    ``start_time`` or ``start_date`` column naming.

    Args:
        db_engine: Database engine with ``query()`` method.
        days_ahead: Number of days ahead to look.
        limit: Maximum events to return.

    Returns:
        List of dicts with ``title``, ``start``, ``attendees``,
        ``location`` keys.

    sensitivity_tier: 2
    """
    try:
        # Check table exists
        tables = db_engine.query(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='raw_calendar_events'"
        )
        if not tables:
            return []

        # Detect column name (start_time vs start_date)
        cols = {
            r["name"]
            for r in db_engine.query(
                "PRAGMA table_info(raw_calendar_events)",
            )
        }
        ts_col = "start_time" if "start_time" in cols else "start_date"

        has_origin = "event_origin" in cols
        origin_select = (
            ", COALESCE(event_origin, 'personal') AS event_origin"
            if has_origin
            else ", 'personal' AS event_origin"
        )
        rows = db_engine.query(f"""
            SELECT title, {ts_col} AS start_ts,
                   attendees, location{origin_select}
            FROM raw_calendar_events
            WHERE {ts_col} >= CURRENT_DATE
              AND {ts_col} < date('now', '+{int(days_ahead)} day')
            ORDER BY start_ts
            LIMIT {int(limit)}
        """)

        return [
            {
                "title": r.get("title", ""),
                "start": str(r.get("start_ts", "")),
                "attendees": str(r.get("attendees", ""))[:100],
                "location": str(r.get("location", ""))[:100],
                "event_origin": str(r.get("event_origin", "personal")),
            }
            for r in rows
        ]
    except Exception:
        logger.debug("Today events fetch failed", exc_info=True)
        return []


def load_pending_reply_ids(
    db_engine: Any,
) -> set[str]:
    """Load message IDs already flagged as pending replies.

    Used by MessageEvaluator to avoid duplicate flagging of messages
    that ProactiveIntelligence has already identified.

    Args:
        db_engine: Database engine with ``query()`` method.

    Returns:
        Set of message ID strings that are active pending replies.

    sensitivity_tier: 1
    """
    try:
        tables = db_engine.query(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='_pending_replies'"
        )
        if not tables:
            return set()

        rows = db_engine.query("""
            SELECT message_id
            FROM _pending_replies
            WHERE dismissed_at IS NULL
        """)
        return {str(r["message_id"]) for r in rows}
    except Exception:
        logger.debug("Pending reply IDs fetch failed", exc_info=True)
        return set()
