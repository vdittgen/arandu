"""Base DDL schemas for the Arandu raw-data layer (SQLite).

Every table stores data as ingested (no transformation).  The pipeline
runner in src/pipeline/ will stage, transform, and promote rows from
these tables into the analytical marts.

Data Sensitivity Tiers
----------------------
- Tier 1 (low):    general preferences, interests
- Tier 2 (medium): habits, routines, people names
- Tier 3 (high):   health, finances, emotions, traumas

Each table carries a *default* sensitivity_tier for new rows; individual
rows may override this value at insert time.

SQLite type mappings (from former DuckDB types):
    TIMESTAMPTZ → TEXT (ISO 8601 strings)
    JSON        → TEXT (JSON-encoded strings)
    DOUBLE      → REAL
    BIGINT      → INTEGER
    VARCHAR     → TEXT
    BOOLEAN     → INTEGER (0/1)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.sqlite.engine import DatabaseEngine

# ---------------------------------------------------------------------------
# DDL statements
# ---------------------------------------------------------------------------

# sensitivity_tier default 2 — messages contain people names / relationships
RAW_MESSAGES = """
CREATE TABLE IF NOT EXISTS raw_messages (
    id            TEXT     PRIMARY KEY,
    source        TEXT     NOT NULL,          -- e.g. 'gmail', 'slack', 'imessage'
    sender        TEXT     NOT NULL,
    recipient     TEXT     NOT NULL,
    content       TEXT     NOT NULL,
    timestamp     TEXT     NOT NULL,          -- ISO 8601
    metadata      TEXT,                       -- JSON
    sensitivity_tier  INTEGER  NOT NULL DEFAULT 2,
    created_at    TEXT     NOT NULL DEFAULT (datetime('now')),
    sender_name   TEXT,
    is_from_me    INTEGER  DEFAULT 0
);
"""

# sensitivity_tier default 2 — attendee names / locations
RAW_CALENDAR_EVENTS = """
CREATE TABLE IF NOT EXISTS raw_calendar_events (
    id            TEXT     PRIMARY KEY,
    title         TEXT     NOT NULL,
    description   TEXT,
    start_time    TEXT     NOT NULL,          -- ISO 8601
    end_time      TEXT     NOT NULL,          -- ISO 8601
    location      TEXT,
    attendees     TEXT,                       -- JSON array
    sensitivity_tier  INTEGER  NOT NULL DEFAULT 2,
    created_at    TEXT     NOT NULL DEFAULT (datetime('now'))
);
"""

# sensitivity_tier default 1 — notes can be general; override per-row
RAW_NOTES = """
CREATE TABLE IF NOT EXISTS raw_notes (
    id            TEXT     PRIMARY KEY,
    title         TEXT     NOT NULL,
    content       TEXT     NOT NULL,
    source        TEXT     NOT NULL,          -- e.g. 'obsidian', 'apple_notes'
    created_at    TEXT     NOT NULL DEFAULT (datetime('now')),
    updated_at    TEXT     NOT NULL DEFAULT (datetime('now')),
    tags          TEXT,                       -- JSON array
    sensitivity_tier  INTEGER  NOT NULL DEFAULT 1
);
"""

# sensitivity_tier default 3 — health data is always high sensitivity
RAW_HEALTH_METRICS = """
CREATE TABLE IF NOT EXISTS raw_health_metrics (
    id            TEXT     PRIMARY KEY,
    metric_type   TEXT     NOT NULL,          -- e.g. 'heart_rate', 'steps'
    value         REAL     NOT NULL,
    unit          TEXT     NOT NULL,
    recorded_at   TEXT     NOT NULL,          -- ISO 8601
    source        TEXT     NOT NULL,          -- e.g. 'apple_health', 'garmin'
    sensitivity_tier  INTEGER  NOT NULL DEFAULT 3,
    created_at    TEXT     NOT NULL DEFAULT (datetime('now'))
);
"""

# sensitivity_tier default 2 — contact details include personal info
RAW_CONTACTS = """
CREATE TABLE IF NOT EXISTS raw_contacts (
    id            TEXT     PRIMARY KEY,
    name          TEXT     NOT NULL,
    email         TEXT,
    phone         TEXT,
    relationship  TEXT,                       -- e.g. 'friend', 'colleague'
    notes         TEXT,
    last_contact  TEXT,                       -- ISO 8601
    sensitivity_tier  INTEGER  NOT NULL DEFAULT 2,
    created_at    TEXT     NOT NULL DEFAULT (datetime('now'))
);
"""

# sensitivity_tier default 1 — file metadata is generally low sensitivity
RAW_FILES = """
CREATE TABLE IF NOT EXISTS raw_files (
    id              TEXT     PRIMARY KEY,
    filepath        TEXT     NOT NULL,
    filename        TEXT     NOT NULL,
    filetype        TEXT     NOT NULL,        -- MIME type or extension
    size_bytes      INTEGER  NOT NULL,
    created_at      TEXT     NOT NULL DEFAULT (datetime('now')),
    modified_at     TEXT     NOT NULL DEFAULT (datetime('now')),
    content_preview TEXT,
    sensitivity_tier  INTEGER  NOT NULL DEFAULT 1
);
"""

_ALL_SCHEMAS: list[str] = [
    RAW_MESSAGES,
    RAW_CALENDAR_EVENTS,
    RAW_NOTES,
    RAW_HEALTH_METRICS,
    RAW_CONTACTS,
    RAW_FILES,
]

ALL_TABLE_NAMES: list[str] = [
    "raw_messages",
    "raw_calendar_events",
    "raw_notes",
    "raw_health_metrics",
    "raw_contacts",
    "raw_files",
]


def create_all_tables(engine: DatabaseEngine) -> None:
    """Create all raw-data tables in the given engine (idempotent).

    Args:
        engine: An open DatabaseEngine instance to run the DDL against.
    """
    for ddl in _ALL_SCHEMAS:
        engine.execute(ddl)
