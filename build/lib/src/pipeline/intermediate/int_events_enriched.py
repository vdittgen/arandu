"""Python-based pipeline model that enriches calendar events.

Joins ``stg_calendar_events`` with ``stg_contacts`` to resolve
attendee identities, then asks :class:`EventCategorizerAgent` to
choose one ``event_category`` per event from a closed vocabulary
(``meeting``/``social``/``health``/``travel``/``other``).

The legacy ``int_events_enriched.sql`` used a brittle keyword
``CASE`` that mis-classified almost every real-world event as
``"other"`` — see ``CLAUDE.md`` pitfall on keyword filters.

A small per-event cache table (``_event_category_cache``) keyed on
``(event_id, content_fingerprint)`` is used to skip the LLM call
when an event hasn't materially changed since the last pipeline
run. Cache hits cost nothing; misses pay one LLM call per event.

sensitivity_tier: 2 (events carry attendee names and titles)
"""

from __future__ import annotations

import hashlib
import logging
import typing as t

logger = logging.getLogger(__name__)

# When the LLM is unavailable we still need a row per event so the
# downstream marts can keep running. Defaulting to "meeting" matches
# the design intent: a typical working person's calendar entry is a
# work meeting unless we have a clear signal otherwise.
_FALLBACK_CATEGORY = "meeting"
_FALLBACK_REASON = "fallback: llm unavailable"

# Categorising costs one LLM call per uncached event. Cap per run so
# a calendar import doesn't fan out into hundreds of calls in one go.
_MAX_LLM_CALLS_PER_RUN = 200

# Cache version — bump when the prompt changes so old verdicts are
# re-evaluated against the new system prompt.
_CACHE_VERSION = "v1"
_CACHE_TABLE = "_event_category_cache"

_ALLOWED_CATEGORIES = {
    "meeting", "social", "health", "travel", "other",
}

if t.TYPE_CHECKING:
    from src.core.sqlite.engine import DatabaseEngine


def _fingerprint(event: dict[str, t.Any]) -> str:
    """Hash the fields that influence the verdict.

    Includes ``_CACHE_VERSION`` so prompt changes invalidate cached
    entries automatically.

    sensitivity_tier: 1
    """
    parts = [
        _CACHE_VERSION,
        str(event.get("title", "") or ""),
        str(event.get("description", "") or ""),
        str(event.get("location", "") or ""),
        str(event.get("known_attendee_names", "") or ""),
    ]
    raw = "\x1f".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _ensure_cache_table(db: DatabaseEngine) -> None:
    """Create the cache table if it doesn't exist.

    sensitivity_tier: 1
    """
    try:
        db.execute(f"""
            CREATE TABLE IF NOT EXISTS {_CACHE_TABLE} (
                event_id    TEXT PRIMARY KEY,
                fingerprint TEXT NOT NULL,
                category    TEXT NOT NULL,
                reason      TEXT,
                decided_at  TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
    except Exception:  # noqa: BLE001
        logger.debug("Event category cache table creation skipped",
                     exc_info=True)


def _load_cache(
    db: DatabaseEngine,
) -> dict[str, tuple[str, str]]:
    """Return ``{event_id: (fingerprint, category)}`` for cached events.

    sensitivity_tier: 1
    """
    try:
        rows = db.query(
            f"SELECT event_id, fingerprint, category FROM {_CACHE_TABLE}",
        )
    except Exception:  # noqa: BLE001
        logger.debug("Event category cache lookup failed", exc_info=True)
        return {}
    return {
        str(r["event_id"]): (str(r["fingerprint"]), str(r["category"]))
        for r in rows
        if r.get("event_id")
    }


def _store_cache(
    db: DatabaseEngine,
    event_id: str,
    fingerprint: str,
    category: str,
    reason: str,
) -> None:
    """Insert-or-replace a single cache row.

    sensitivity_tier: 1
    """
    try:
        db.execute(
            f"INSERT OR REPLACE INTO {_CACHE_TABLE} "
            "(event_id, fingerprint, category, reason, decided_at) "
            "VALUES (?, ?, ?, ?, datetime('now'))",
            [event_id, fingerprint, category, reason],
        )
    except Exception:  # noqa: BLE001
        logger.debug("Could not cache category for event %s", event_id)


def _is_recurring(title: str) -> int:
    """Cheap recurrence heuristic — kept as a non-LLM check.

    Mirrors the recurrence detection from the legacy SQL. Recurrence
    is a structural signal, not a semantic one, so a small keyword
    list is appropriate here.

    sensitivity_tier: 1
    """
    lower = title.lower()
    markers = (
        "daily", "weekly", "monthly", "stand-up",
        "standup", "therapy", "1-on-1",
    )
    return 1 if any(m in lower for m in markers) else 0


def _select_events(db: DatabaseEngine) -> list[dict[str, t.Any]]:
    """Pull calendar events joined with known-attendee contact info.

    Uses the same JOIN shape as the legacy SQL so the per-row schema
    is preserved.

    sensitivity_tier: 2
    """
    return db.query("""
        WITH known_attendees AS (
            SELECT
                e.id AS event_id,
                GROUP_CONCAT(c.name, ', ')         AS known_attendee_names,
                GROUP_CONCAT(c.relationship, ', ') AS attendee_relationships
            FROM stg_calendar_events e
            CROSS JOIN stg_contacts c
            WHERE CAST(e.attendees AS TEXT) LIKE '%' || c.email || '%'
               OR CAST(e.attendees AS TEXT) LIKE '%"' || c.name || '"%'
            GROUP BY e.id
        )
        SELECT
            e.id,
            e.title,
            e.description,
            e.start_time,
            e.end_time,
            e.location,
            e.attendees,
            e.attendees_count,
            e.duration_minutes,
            e.sensitivity_tier,
            e.calendar_name,
            e.calendar_owner_email,
            e.is_shared_calendar,
            e.is_subscribed_calendar,
            e.self_response_status,
            e.event_origin,
            ka.known_attendee_names,
            ka.attendee_relationships
        FROM stg_calendar_events e
        LEFT JOIN known_attendees ka
            ON e.id = ka.event_id
    """)


def _categorize_via_llm(
    event: dict[str, t.Any],
) -> tuple[str, str] | None:
    """One LLM call → ``(category, reason)`` or None on failure.

    sensitivity_tier: 2
    """
    try:
        from src.agents.event_categorizer.agent import EventCategorizerAgent
    except Exception:  # noqa: BLE001
        logger.warning(
            "EventCategorizerAgent unavailable", exc_info=True,
        )
        return None
    try:
        decision = EventCategorizerAgent().categorize(
            title=str(event.get("title", "") or ""),
            description=str(event.get("description", "") or ""),
            location=str(event.get("location", "") or ""),
            start_time=str(event.get("start_time", "") or ""),
            attendees=str(event.get("attendees", "") or ""),
            attendee_names=str(
                event.get("known_attendee_names", "") or "",
            ),
        )
    except Exception as exc:  # noqa: BLE001
        # Preemption is the pipeline's shutdown signal — re-raise so
        # the runner can halt cleanly. Anything else is an LLM glitch
        # and falls through to the fallback category.
        from src.models.llm_provider import PreemptedError
        if isinstance(exc, PreemptedError):
            raise
        logger.warning(
            "Event categoriser failed for event %s",
            event.get("id"), exc_info=True,
        )
        return None
    if decision is None:
        return None
    category = (decision.category or "").lower().strip()
    if category not in _ALLOWED_CATEGORIES:
        logger.debug(
            "Event categoriser returned unknown category %r for %s "
            "— falling back to %r",
            category, event.get("id"), _FALLBACK_CATEGORY,
        )
        return None
    return category, (decision.reason or "")[:200]


def _build_row(
    event: dict[str, t.Any],
    category: str,
) -> dict[str, t.Any]:
    """Project one input row into the output schema.

    sensitivity_tier: 2
    """
    title = str(event.get("title", "") or "")
    return {
        "id": event.get("id"),
        "title": title,
        "description": event.get("description"),
        "start_time": event.get("start_time"),
        "end_time": event.get("end_time"),
        "location": event.get("location"),
        "attendees": event.get("attendees"),
        "attendees_count": event.get("attendees_count"),
        "duration_minutes": event.get("duration_minutes"),
        "sensitivity_tier": event.get("sensitivity_tier"),
        "calendar_name": event.get("calendar_name"),
        "calendar_owner_email": event.get("calendar_owner_email"),
        "is_shared_calendar": event.get("is_shared_calendar"),
        "is_subscribed_calendar": event.get("is_subscribed_calendar"),
        "self_response_status": event.get("self_response_status"),
        "event_origin": event.get("event_origin") or "personal",
        "known_attendee_names": event.get("known_attendee_names"),
        "attendee_relationships": event.get("attendee_relationships"),
        "event_category": category,
        "is_recurring": _is_recurring(title),
        "_loaded_at": _loaded_at(),
    }


def _loaded_at() -> str:
    """ISO timestamp matching ``datetime('now')`` in the legacy SQL.

    sensitivity_tier: 1
    """
    from datetime import datetime, timezone
    # Match the SQLite ``datetime('now')`` format: UTC, no tz suffix.
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def execute(db: DatabaseEngine) -> list[dict[str, t.Any]]:
    """Categorise every calendar event via :class:`EventCategorizerAgent`.

    Uses a fingerprint cache so unchanged events skip the LLM call. On
    LLM failure the row is still emitted with the ``meeting`` fallback
    so the marts always have data to work with.

    sensitivity_tier: 2
    """
    _ensure_cache_table(db)

    events = _select_events(db)
    if not events:
        logger.info("int_events_enriched: no events to process")
        return []

    cache = _load_cache(db)
    rows: list[dict[str, t.Any]] = []
    llm_calls = 0
    cache_hits = 0
    fallbacks = 0

    for event in events:
        event_id = str(event.get("id") or "")
        if not event_id:
            continue
        fingerprint = _fingerprint(event)
        cached = cache.get(event_id)
        if cached is not None and cached[0] == fingerprint:
            rows.append(_build_row(event, cached[1]))
            cache_hits += 1
            continue

        if llm_calls >= _MAX_LLM_CALLS_PER_RUN:
            logger.warning(
                "int_events_enriched: hit per-run LLM cap (%d) — "
                "deferring categorisation for %s to next run",
                _MAX_LLM_CALLS_PER_RUN, event_id,
            )
            rows.append(_build_row(event, _FALLBACK_CATEGORY))
            fallbacks += 1
            continue

        verdict = _categorize_via_llm(event)
        llm_calls += 1
        if verdict is None:
            rows.append(_build_row(event, _FALLBACK_CATEGORY))
            fallbacks += 1
            # Don't cache fallback verdicts — we want to retry next run.
            continue
        category, reason = verdict
        rows.append(_build_row(event, category))
        _store_cache(db, event_id, fingerprint, category, reason)

    logger.info(
        "int_events_enriched: %d events (%d cache hits, %d LLM calls, "
        "%d fallbacks)",
        len(rows), cache_hits, llm_calls, fallbacks,
    )
    return rows
