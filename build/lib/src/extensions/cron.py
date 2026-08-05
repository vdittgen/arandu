"""Minimal cron expression matcher for agent scheduling.

Supports standard 5-field cron expressions: ``minute hour day month weekday``.

Field syntax:
- ``*`` — matches any value
- ``N`` — matches exact value
- ``*/N`` — matches every N-th value (step)
- ``N-M`` — matches range N through M inclusive
- ``N,M,...`` — matches any listed value

Weekday convention follows standard cron: 0=Sunday, 1=Monday, ..., 6=Saturday.
Internally converted to Python ``datetime.weekday()`` (0=Monday .. 6=Sunday).

No external dependencies.

sensitivity_tier: N/A (infrastructure utility)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Maximum minutes to scan in cron_is_due() to prevent runaway loops
# when last_run is very old (caps at 24 hours).
_MAX_SCAN_MINUTES = 1440


def _parse_field(field: str, min_val: int, max_val: int) -> set[int]:
    """Parse a single cron field into a set of matching integers.

    Args:
        field: Cron field string (e.g. ``"*"``, ``"5"``, ``"1-3"``, ``"*/15"``).
        min_val: Minimum valid value for this field.
        max_val: Maximum valid value for this field.

    Returns:
        Set of integers that this field matches.

    sensitivity_tier: N/A
    """
    result: set[int] = set()

    for part in field.split(","):
        part = part.strip()

        if part == "*":
            result.update(range(min_val, max_val + 1))
        elif part.startswith("*/"):
            step = int(part[2:])
            if step > 0:
                result.update(range(min_val, max_val + 1, step))
        elif "-" in part:
            lo, hi = part.split("-", 1)
            result.update(range(int(lo), int(hi) + 1))
        else:
            result.add(int(part))

    return result


def cron_matches(cron_expr: str, dt: datetime) -> bool:
    """Return True if *dt* matches the 5-field cron expression.

    Fields: ``minute hour day-of-month month day-of-week``.

    Args:
        cron_expr: Standard 5-field cron string.
        dt: Datetime to check.

    Returns:
        True if the datetime matches all five fields.

    sensitivity_tier: N/A
    """
    fields = cron_expr.strip().split()
    if len(fields) != 5:  # noqa: PLR2004
        logger.warning("Invalid cron expression (need 5 fields): %s", cron_expr)
        return False

    minutes = _parse_field(fields[0], 0, 59)
    hours = _parse_field(fields[1], 0, 23)
    days = _parse_field(fields[2], 1, 31)
    months = _parse_field(fields[3], 1, 12)

    # Convert standard cron weekdays (0=Sun) to Python weekdays (0=Mon).
    cron_weekdays = _parse_field(fields[4], 0, 6)
    py_weekdays = {(d - 1) % 7 for d in cron_weekdays}

    return (
        dt.minute in minutes
        and dt.hour in hours
        and dt.day in days
        and dt.month in months
        and dt.weekday() in py_weekdays
    )


def cron_is_due(
    cron_expr: str,
    last_run: datetime | None,
    now: datetime,
) -> bool:
    """Return True if the cron schedule fired between *last_run* and *now*.

    When *last_run* is None (never run before), checks only if *now*
    matches the expression.

    Scans at most 1440 minutes (24 hours) backwards from *now* to
    prevent runaway iteration when *last_run* is very old.

    Args:
        cron_expr: Standard 5-field cron string.
        last_run: Timestamp of the last successful run, or None.
        now: Current timestamp.

    Returns:
        True if the agent should run now.

    sensitivity_tier: N/A
    """
    if last_run is None:
        return cron_matches(cron_expr, now)

    # Truncate both to minute precision for clean iteration.
    start = last_run.replace(second=0, microsecond=0) + timedelta(minutes=1)
    end = now.replace(second=0, microsecond=0)

    if start > end:
        return False

    # Cap scan window at 24 hours.
    earliest = end - timedelta(minutes=_MAX_SCAN_MINUTES)
    if start < earliest:
        start = earliest

    cursor = start
    while cursor <= end:
        if cron_matches(cron_expr, cursor):
            return True
        cursor += timedelta(minutes=1)

    return False
