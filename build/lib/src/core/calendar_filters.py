"""Shared SQL filters for calendar event scoping.

The dashboard surfaces (daily brief, daily scheduler, suggestion chips,
domain mart awareness lists) all need the same notion of "events the
user actually owns" — primary calendar entries on the user's own
account where they haven't declined the invitation. Subscribed
calendars (sports schedules, public holidays) and team-awareness events
the user is not invited to belong to dedicated awareness surfaces,
never to plans / briefs / scheduler input.

Centralising the SQL guards against drift between callers (the brief
gatherer used the right filter; the scheduler did not — that is the
"meetings from other people's calendars in Today's plan" bug).

sensitivity_tier: 2 (returns calendar rows)
"""

from __future__ import annotations

from typing import Any

# Status values surfaced by the calendar bridges where the user is on
# the invitee list but has not committed. Treating them like declined
# entries keeps them out of the plan / brief but leaves them visible
# elsewhere (the awareness panels still query int_events_enriched
# directly).
_NON_COMMITTED_RESPONSES: tuple[str, ...] = ("declined", "noResponse")


def personal_events_for_date(
    db: Any,
    target_iso: str,
    *,
    columns: str = (
        "id, title, start_time, end_time, location"
    ),
    extra_where: str = "",
    order_by: str = "start_time",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Return the user's own committed events on ``target_iso``.

    Filters on ``event_origin = 'personal'`` (excludes team_awareness
    and subscribed calendars) and on ``self_response_status`` not being
    a non-committed value. Callers select the columns they need; the
    default covers the legacy ``_fetch_events_for_date`` shape.

    sensitivity_tier: 2
    """
    placeholders = ",".join("?" for _ in _NON_COMMITTED_RESPONSES)
    suffix_where = f" AND ({extra_where})" if extra_where else ""
    limit_sql = f" LIMIT {int(limit)}" if limit and limit > 0 else ""
    sql = (
        f"SELECT {columns} "  # noqa: S608 — columns is caller-controlled
        f"FROM raw_calendar_events "
        f"WHERE DATE(start_time) = DATE(?) "
        f"  AND COALESCE(event_origin, 'personal') = 'personal' "
        f"  AND COALESCE(self_response_status, 'accepted') "
        f"      NOT IN ({placeholders})"
        f"{suffix_where} "
        f"ORDER BY {order_by}{limit_sql}"
    )
    params: list[Any] = [target_iso, *_NON_COMMITTED_RESPONSES]
    try:
        rows = db.query(sql, params)
    except Exception:  # noqa: BLE001
        return []
    return [dict(r) for r in rows]


__all__ = ["personal_events_for_date"]
