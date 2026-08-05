"""Per-column sensitivity tier metadata for known tables.

When a table appears in COLUMN_TIERS, cmd_query_table attaches a ``tier``
field to each column entry in the JSON response. The UI uses per-column
tiers to mask only the truly sensitive fields (e.g. email/phone) while
keeping personal-but-not-sensitive fields (e.g. name) readable.

Tables NOT listed here return no column tiers; the UI falls back to the
row-level ``sensitivity_tier`` value for masking.

Tiers follow docs/PRIVACY.md:
    1 = low / public
    2 = personal
    3 = sensitive (masked until explicitly unlocked)
"""

from __future__ import annotations

COLUMN_TIERS: dict[str, dict[str, int]] = {
    "raw_messages": {
        "id": 1,
        "source": 1,
        "sender": 2,
        "recipient": 2,
        "content": 3,
        "timestamp": 1,
        "metadata": 2,
        "sender_name": 2,
        "is_from_me": 1,
        "sensitivity_tier": 1,
        "created_at": 1,
    },
    "raw_calendar_events": {
        "id": 1,
        "title": 2,
        "description": 3,
        "start_time": 1,
        "end_time": 1,
        "location": 3,
        "attendees": 2,
        "sensitivity_tier": 1,
        "created_at": 1,
    },
    "raw_notes": {
        "id": 1,
        "title": 2,
        "content": 3,
        "source": 1,
        "tags": 2,
        "sensitivity_tier": 1,
        "created_at": 1,
        "updated_at": 1,
    },
    "raw_health_metrics": {
        "id": 1,
        "metric_type": 3,
        "value": 3,
        "unit": 1,
        "recorded_at": 2,
        "source": 1,
        "sensitivity_tier": 1,
        "created_at": 1,
    },
    "raw_contacts": {
        "id": 1,
        "name": 2,
        "email": 3,
        "phone": 3,
        "relationship": 2,
        "notes": 2,
        "last_contact": 2,
        "sensitivity_tier": 1,
        "created_at": 1,
    },
    "raw_files": {
        "id": 1,
        "filepath": 2,
        "filename": 2,
        "filetype": 1,
        "size_bytes": 1,
        "content_preview": 3,
        "sensitivity_tier": 1,
        "created_at": 1,
        "modified_at": 1,
    },
}


def get_column_tier(table_name: str, column_name: str) -> int | None:
    """Return the documented tier for a column, or None if unknown.

    Callers fall back to the row-level ``sensitivity_tier`` when this
    returns None.
    """
    table = COLUMN_TIERS.get(table_name)
    if table is None:
        return None
    return table.get(column_name)
