"""SQLite schema migrations for connector-introduced tables and columns.

Creates new raw tables and adds columns to existing tables as required by
enabled connectors. Additive only — never drops tables or columns.

sensitivity_tier: N/A (infrastructure — schema metadata only)
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.core.sqlite.engine import DatabaseEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DDL for tables introduced by connectors (not in the original schemas.py)
# ---------------------------------------------------------------------------

# sensitivity_tier default 2 — email content is personal
RAW_EMAILS = """
CREATE TABLE IF NOT EXISTS raw_emails (
    id              TEXT     PRIMARY KEY,
    source          TEXT     NOT NULL DEFAULT 'unknown',
    message_id      TEXT,
    subject         TEXT,
    from_address    TEXT,
    to_addresses    TEXT,                    -- JSON
    date            TEXT,                    -- ISO 8601
    body_preview    TEXT,
    is_read         INTEGER  DEFAULT 0,
    folder          TEXT,
    labels          TEXT,                    -- JSON
    sensitivity_tier INTEGER NOT NULL DEFAULT 2,
    created_at      TEXT     NOT NULL DEFAULT (datetime('now'))
);
"""

# sensitivity_tier default 1 — reminder titles/completion are low sensitivity
RAW_REMINDERS = """
CREATE TABLE IF NOT EXISTS raw_reminders (
    id              TEXT     PRIMARY KEY,
    source          TEXT     NOT NULL DEFAULT 'apple-calendar',
    title           TEXT     NOT NULL,
    due_date        TEXT,                    -- ISO 8601
    notes           TEXT,
    completed       INTEGER  DEFAULT 0,
    list_name       TEXT,
    sensitivity_tier INTEGER NOT NULL DEFAULT 1,
    created_at      TEXT     NOT NULL DEFAULT (datetime('now'))
);
"""

# sensitivity_tier default 3 — workout metrics are health-sensitive
RAW_WORKOUTS = """
CREATE TABLE IF NOT EXISTS raw_workouts (
    id              TEXT     PRIMARY KEY,
    source          TEXT     NOT NULL DEFAULT 'apple-health',
    workout_type    TEXT     NOT NULL,
    duration_min    REAL,
    calories        REAL,
    heart_rate_avg  REAL,
    date            TEXT     NOT NULL,       -- ISO 8601
    sensitivity_tier INTEGER NOT NULL DEFAULT 3,
    created_at      TEXT     NOT NULL DEFAULT (datetime('now'))
);
"""

# sensitivity_tier default 2 — voice memos may contain personal content
RAW_VOICE_MEMOS = """
CREATE TABLE IF NOT EXISTS raw_voice_memos (
    id              TEXT     PRIMARY KEY,
    source          TEXT     NOT NULL DEFAULT 'apple-voice-memos',
    title           TEXT,
    duration_seconds INTEGER,
    recorded_at     TEXT,                    -- ISO 8601
    transcript      TEXT,
    sensitivity_tier INTEGER NOT NULL DEFAULT 2,
    created_at      TEXT     NOT NULL DEFAULT (datetime('now'))
);
"""

# sensitivity_tier default 1 — track/artist names are public
RAW_LISTENING_HISTORY = """
CREATE TABLE IF NOT EXISTS raw_listening_history (
    id              TEXT     PRIMARY KEY,
    source          TEXT     NOT NULL DEFAULT 'spotify',
    track_name      TEXT     NOT NULL,
    artist          TEXT,
    album           TEXT,
    played_at       TEXT     NOT NULL,       -- ISO 8601
    duration_ms     INTEGER,
    context_type    TEXT,
    sensitivity_tier INTEGER NOT NULL DEFAULT 1,
    created_at      TEXT     NOT NULL DEFAULT (datetime('now'))
);
"""

# ---------------------------------------------------------------------------
# Agent system tables (introduced in Phase 1 of the agentic refactor)
# ---------------------------------------------------------------------------

# Editable agent configuration overrides. Default config lives in code;
# rows here represent user customisation (prompt, model route, tools).
AGENT_CONFIGS = """
CREATE TABLE IF NOT EXISTS agent_configs (
    agent_id        TEXT PRIMARY KEY,
    system_prompt   TEXT,
    model_route     TEXT,
    model_override  TEXT,
    enabled_tools   TEXT,                       -- JSON array
    enabled_skills  TEXT,                       -- JSON array
    updated_at      TEXT NOT NULL,
    version         INTEGER NOT NULL DEFAULT 1
);
"""

# Per-provider consent records. Captures the user's explicit consent
# (or lack thereof) to send Tier 2/3 prompts to a remote LLM endpoint.
PROVIDER_CONSENTS = """
CREATE TABLE IF NOT EXISTS provider_consents (
    provider_url        TEXT PRIMARY KEY,
    provider_name       TEXT,
    max_tier_allowed    INTEGER NOT NULL DEFAULT 1,
    consent_granted_at  TEXT,
    revoked_at          TEXT,
    notes               TEXT
);
"""

# One row per deep-agent invocation. Captures the plan, status, and
# pointers into the audit chain.
DEEP_AGENT_RUNS = """
CREATE TABLE IF NOT EXISTS deep_agent_runs (
    run_id          TEXT PRIMARY KEY,
    agent_id        TEXT NOT NULL,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    status          TEXT NOT NULL DEFAULT 'running',
    plan_json       TEXT,
    workspace_path  TEXT,
    audit_head_hash TEXT,
    error           TEXT
);
"""

# Tool-call log inside a deep-agent run.
DEEP_AGENT_STEPS = """
CREATE TABLE IF NOT EXISTS deep_agent_steps (
    step_id         TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL,
    seq             INTEGER NOT NULL,
    tool            TEXT NOT NULL,
    args_json       TEXT,
    result_json     TEXT,
    ok              INTEGER NOT NULL DEFAULT 1,
    duration_ms     REAL NOT NULL DEFAULT 0,
    ts              TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES deep_agent_runs(run_id)
);
"""

# One row per pydantic-evals run triggered by an agent edit or the
# "Run eval" button. Powers the Agents page status banner.
AGENT_EVAL_RUNS = """
CREATE TABLE IF NOT EXISTS agent_eval_runs (
    run_id              TEXT PRIMARY KEY,
    agent_id            TEXT NOT NULL,
    suite               TEXT,
    trigger             TEXT NOT NULL DEFAULT 'manual',
    started_at          TEXT NOT NULL,
    finished_at         TEXT,
    status              TEXT NOT NULL DEFAULT 'pending',
    cases_total         INTEGER NOT NULL DEFAULT 0,
    cases_passed        INTEGER NOT NULL DEFAULT 0,
    cases_failed        INTEGER NOT NULL DEFAULT 0,
    failed_cases_json   TEXT,
    error               TEXT
)
"""

# Sibling index — applied in a second exec because sqlite3.execute()
# only takes one statement at a time.
AGENT_EVAL_RUNS_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_agent_eval_runs_agent_started "
    "ON agent_eval_runs(agent_id, started_at DESC)"
)

# Sampled LLM call metrics for the Agents page latency panel.
LLM_CALLS = """
CREATE TABLE IF NOT EXISTS llm_calls (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id        TEXT NOT NULL,
    tier            TEXT NOT NULL,
    route           TEXT NOT NULL,
    wait_ms         REAL NOT NULL DEFAULT 0,
    run_ms          REAL NOT NULL DEFAULT 0,
    ts              TEXT NOT NULL
);
"""


# Map of connector-raw table name → DDL statement. These tables share
# the raw-data convention (``id`` PK, ``sensitivity_tier``, ``created_at``)
# and are exercised by the connector catalog tests.
MIGRATION_SCHEMAS: dict[str, str] = {
    "raw_emails": RAW_EMAILS,
    "raw_reminders": RAW_REMINDERS,
    "raw_workouts": RAW_WORKOUTS,
    "raw_voice_memos": RAW_VOICE_MEMOS,
    "raw_listening_history": RAW_LISTENING_HISTORY,
}

# Agent-system tables introduced in Phase 1. These do NOT follow the
# raw-data convention — they store agent configuration, deep-agent run
# metadata, and call metrics. ``run_migrations`` applies them after the
# raw tables; the connector catalog tests intentionally skip them.
AGENT_SYSTEM_SCHEMAS: dict[str, str] = {
    "agent_configs": AGENT_CONFIGS,
    "provider_consents": PROVIDER_CONSENTS,
    "deep_agent_runs": DEEP_AGENT_RUNS,
    "deep_agent_steps": DEEP_AGENT_STEPS,
    "llm_calls": LLM_CALLS,
    "agent_eval_runs": AGENT_EVAL_RUNS,
}

# All connector-introduced table names (for reference)
MIGRATION_TABLE_NAMES: list[str] = sorted(MIGRATION_SCHEMAS.keys())

# ---------------------------------------------------------------------------
# Column additions to existing tables (for connector field mappings)
# ---------------------------------------------------------------------------
# Each entry: (table_name, column_name, column_def)
# These are columns needed by connectors but absent from the original schemas.

COLUMN_ADDITIONS: list[tuple[str, str, str]] = [
    # apple-calendar: is_all_day flag
    ("raw_calendar_events", "is_all_day", "INTEGER DEFAULT 0"),
    # apple-calendar: provenance — which calendar this event came from,
    # whether the user owns it, and whether they're an invited attendee.
    # Powers the dashboard's personal-vs-team_awareness-vs-subscribed split.
    ("raw_calendar_events", "calendar_name", "TEXT"),
    ("raw_calendar_events", "calendar_owner_email", "TEXT"),
    ("raw_calendar_events", "is_shared_calendar", "INTEGER DEFAULT 0"),
    ("raw_calendar_events", "is_subscribed_calendar", "INTEGER DEFAULT 0"),
    ("raw_calendar_events", "self_response_status", "TEXT"),
    ("raw_calendar_events", "event_origin", "TEXT"),
    # apple-contacts: birthday and address
    ("raw_contacts", "birthday", "TEXT"),
    ("raw_contacts", "address", "TEXT"),
    # apple-messages / whatsapp: chat metadata
    ("raw_messages", "is_from_me", "INTEGER"),
    ("raw_messages", "chat_name", "TEXT"),
    ("raw_messages", "is_group", "INTEGER"),
    ("raw_messages", "sender_name", "TEXT"),
    # obsidian notes: file path
    ("raw_notes", "filepath", "TEXT"),
]


def get_existing_tables(engine: DatabaseEngine) -> set[str]:
    """Return the set of table names already present in the database.

    sensitivity_tier: N/A
    """
    rows: list[dict[str, Any]] = engine.query(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    )
    return {row["name"] for row in rows}


def _get_table_columns(engine: DatabaseEngine, table_name: str) -> set[str]:
    """Return the set of column names for a given table.

    sensitivity_tier: N/A
    """
    rows = engine.query(f"PRAGMA table_info('{table_name}')")
    return {row["name"] for row in rows}


def run_column_additions(engine: DatabaseEngine) -> list[str]:
    """Add missing columns to existing tables (additive only).

    Returns a list of 'table.column' strings for newly added columns.

    Args:
        engine: An open DatabaseEngine instance.

    sensitivity_tier: N/A
    """
    added: list[str] = []
    existing_tables = get_existing_tables(engine)

    for table_name, col_name, col_def in COLUMN_ADDITIONS:
        if table_name not in existing_tables:
            continue
        existing_cols = _get_table_columns(engine, table_name)
        if col_name not in existing_cols:
            engine.execute(
                f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_def}"
            )
            added.append(f"{table_name}.{col_name}")
            logger.info("Added column: %s.%s", table_name, col_name)

    # One-time backfill: populate sender_name for existing WhatsApp messages
    if "raw_messages.sender_name" in added:
        _backfill_whatsapp_sender_names(engine)

    return added


def _backfill_whatsapp_sender_names(engine: DatabaseEngine) -> None:
    """Populate sender_name for existing WhatsApp messages.

    Runs once when the sender_name column is first created.  Uses
    chat_name (the contact/group name from Baileys) as the primary
    source, with phone formatting and contact cross-reference as
    fallbacks.

    SQLite has no ``REGEXP_REPLACE``, so the contact cross-reference
    step uses Python-side processing instead of a single SQL statement.

    sensitivity_tier: 2
    """
    # Own messages
    engine.execute(
        "UPDATE raw_messages SET sender_name = 'me' "
        "WHERE source = 'whatsapp' AND is_from_me = 1 "
        "AND sender_name IS NULL",
    )

    # Cross-reference with Apple Contacts by phone number (Python-based)
    try:
        contacts = engine.query(
            "SELECT phone, name FROM raw_contacts "
            "WHERE phone IS NOT NULL AND name IS NOT NULL AND name != ''"
        )
        for contact in contacts:
            phone_digits = re.sub(r"[^0-9]", "", contact["phone"])
            if len(phone_digits) < 4:
                continue
            phone_suffix = phone_digits[-10:]
            engine.execute(
                "UPDATE raw_messages SET sender_name = ? "
                "WHERE source = 'whatsapp' AND sender_name IS NULL "
                "AND sender LIKE '%@s.whatsapp.net' "
                "AND SUBSTR(REPLACE(sender, '@s.whatsapp.net', ''), -10) = ?",
                [contact["name"], phone_suffix],
            )
    except Exception:
        logger.debug("Contact cross-reference skipped", exc_info=True)

    # 1:1 chats: chat_name IS the contact name (when not a JID)
    engine.execute(
        "UPDATE raw_messages SET sender_name = chat_name "
        "WHERE source = 'whatsapp' "
        "AND is_from_me = 0 "
        "AND (is_group = 0 OR is_group IS NULL) "
        "AND sender_name IS NULL "
        "AND chat_name IS NOT NULL "
        "AND chat_name NOT LIKE '%@s.whatsapp.net' "
        "AND chat_name NOT LIKE '%@g.us' "
        "AND chat_name NOT LIKE '%@lid'",
    )

    # Phone JIDs: format as +phone
    engine.execute(
        "UPDATE raw_messages "
        "SET sender_name = '+' || REPLACE(sender, '@s.whatsapp.net', '') "
        "WHERE source = 'whatsapp' AND sender_name IS NULL "
        "AND sender LIKE '%@s.whatsapp.net'",
    )

    # @lid JIDs with no match
    engine.execute(
        "UPDATE raw_messages SET sender_name = 'Unknown' "
        "WHERE source = 'whatsapp' AND sender_name IS NULL "
        "AND sender LIKE '%@lid'",
    )

    logger.info("Backfilled sender_name for existing WhatsApp messages")


def run_migrations(engine: DatabaseEngine) -> list[str]:
    """Create any missing connector tables and add missing columns.

    Returns the list of table names that were newly created.
    Also runs column additions for existing tables.

    Args:
        engine: An open DatabaseEngine instance.

    sensitivity_tier: N/A
    """
    existing = get_existing_tables(engine)
    created: list[str] = []

    for table_name, ddl in MIGRATION_SCHEMAS.items():
        if table_name not in existing:
            engine.execute(ddl)
            created.append(table_name)
            logger.info("Created table: %s", table_name)
        else:
            logger.debug("Table already exists, skipping: %s", table_name)

    # Agent-system tables run alongside but are not surfaced in
    # ``created`` — callers use the return value to wire connectors,
    # and these tables are infrastructure unrelated to that flow.
    for table_name, ddl in AGENT_SYSTEM_SCHEMAS.items():
        if table_name not in existing:
            engine.execute(ddl)
            logger.info("Created agent-system table: %s", table_name)

    # Sibling indexes for agent-system tables.
    try:
        engine.execute(AGENT_EVAL_RUNS_INDEX)
    except Exception:  # noqa: BLE001
        logger.debug("agent_eval_runs index already exists", exc_info=True)

    run_column_additions(engine)

    return created


def ensure_table(engine: DatabaseEngine, table_name: str) -> bool:
    """Create a single table if it does not already exist.

    Returns True if the table was newly created, False if it already existed.

    Args:
        engine: An open DatabaseEngine instance.
        table_name: Name of the table to ensure exists.

    Raises:
        ValueError: If the table name is not in the migration registry.

    sensitivity_tier: N/A
    """
    all_schemas = {**MIGRATION_SCHEMAS, **AGENT_SYSTEM_SCHEMAS}
    if table_name not in all_schemas:
        msg = f"Unknown migration table: {table_name!r}"
        raise ValueError(msg)

    existing = get_existing_tables(engine)
    if table_name in existing:
        return False

    engine.execute(all_schemas[table_name])
    logger.info("Created table: %s", table_name)
    return True
