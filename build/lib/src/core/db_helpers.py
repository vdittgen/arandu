"""Shared database helper utilities.

Provides reusable functions for common database operations used across
multiple inference components (ProactiveIntelligence, MessageEvaluator,
FactLearner, InsightGenerator, etc.).

Eliminates duplication of ``_utc_now_iso``, ``_make_id``, ``_safe_str``,
``_table_exists``, ``_get_table_columns``, and the ``_ensure_tables``
pattern that was previously copy-pasted across 4+ modules.

sensitivity_tier: 1 (no user data access — pure infrastructure)
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)


def utc_now_iso() -> str:
    """Return current UTC timestamp as ISO 8601 text.

    sensitivity_tier: 1
    """
    return datetime.now(tz=timezone.utc).isoformat()


def utc_ago_iso(
    *, days: int = 0, hours: int = 0, minutes: int = 0,
) -> str:
    """Return a UTC timestamp ``days/hours/minutes`` ago as ISO 8601 text.

    Use this (as a bind parameter) for time-window predicates over
    ISO-8601 columns. Comparing them against SQLite's
    ``datetime('now', '-N minutes')`` is a string comparison against a
    space-separated timestamp, and since ``'T' > ' '`` every same-day
    row passes — the "window" silently becomes the whole UTC day.

    sensitivity_tier: 1
    """
    cutoff = datetime.now(tz=timezone.utc) - timedelta(
        days=days, hours=hours, minutes=minutes,
    )
    return cutoff.isoformat()


def make_hash_id(*parts: str) -> str:
    """Create a short deterministic ID from string parts.

    Joins parts with ``:``, hashes with SHA-256, returns first 16 hex chars.

    sensitivity_tier: 1
    """
    raw = ":".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def safe_str(value: Any, max_len: int = 200) -> str:
    """Convert value to string, truncate for prompt batching.

    sensitivity_tier: 1
    """
    s = str(value) if value is not None else ""
    return s[:max_len] if len(s) > max_len else s


def table_exists(db_engine: Any, table_name: str) -> bool:
    """Check if a table exists in SQLite.

    Args:
        db_engine: Database engine with ``query()`` method.
        table_name: Name of the table to check.

    Returns:
        True if the table exists, False otherwise.

    sensitivity_tier: 1
    """
    try:
        rows = db_engine.query(
            "SELECT 1 FROM sqlite_master "
            "WHERE type = 'table' AND name = ?",
            [table_name],
        )
        return len(rows) > 0
    except Exception:
        return False


def get_table_columns(db_engine: Any, table_name: str) -> set[str]:
    """Return column names for a SQLite table.

    Args:
        db_engine: Database engine with ``query()`` method.
        table_name: Name of the table.

    Returns:
        Set of column name strings, empty on failure.

    sensitivity_tier: 1
    """
    try:
        rows = db_engine.query(
            f"PRAGMA table_info({table_name})",
        )
        return {r["name"] for r in rows}
    except Exception:
        return set()


def ensure_tables(db_engine: Any, ddl_statements: list[str]) -> None:
    """Execute DDL statements, silently skip on read-only DB.

    Common pattern for components that create internal tables on init
    but must also work in read-only mode (e.g. Dashboard reads).

    Args:
        db_engine: Database engine with ``execute()`` method.
        ddl_statements: List of CREATE TABLE IF NOT EXISTS statements.

    sensitivity_tier: 1
    """
    try:
        for ddl in ddl_statements:
            db_engine.execute(ddl)
    except Exception:
        logger.debug(
            "Skipped table creation (read-only mode)",
            exc_info=True,
        )
