"""CLI entry point for the Arandu data layer.

Provides commands that operate on all three embedded databases:

    python -m src.core.cli init    — create database schemas
    python -m src.core.cli status  — print database stats and health
    python -m src.core.cli reset   — wipe and reinitialize all data
    python -m src.core.cli stats   — output database stats as JSON
    python -m src.core.cli query-messages --limit N  — recent messages (JSON)
    python -m src.core.cli query-events --days N     — upcoming events (JSON)
    python -m src.core.cli query-notes --limit N     — notes (JSON)
    python -m src.core.cli query-emails --limit N    — emails (JSON)
    python -m src.core.cli query-today               — today summary (JSON)
    python -m src.core.cli list-tables --prefix raw_ — list tables with counts (JSON)
    python -m src.core.cli query-table --table X     — sample rows from table (JSON)
    python -m src.core.cli ask "question"             — ask Brain Agent (JSON)
    python -m src.core.cli list-agents               — list discovered agents (JSON)
    python -m src.core.cli run-agent --agent-id ID   — run an agent (JSON)
    python -m src.core.cli get-agent-result --agent-id ID — last agent result (JSON)
    python -m src.core.cli list-skills               — list registered skills (JSON)
    python -m src.core.cli list-actions     — available action tools (JSON)
    python -m src.core.cli confirm-action  — execute confirmed action
    python -m src.core.cli cancel-action   — cancel proposed action
    python -m src.core.cli startup-sync    — sync stale connectors on launch
    python -m src.core.cli sync-all-stale  — periodic sync all enabled
    python -m src.core.cli run-scheduled-agents — run cron-due agents
    python -m src.core.cli health          — check all system components
    python -m src.core.cli get-interests   — interest profile (JSON)
    python -m src.core.cli get-domain-stats — per-domain query stats (JSON)
    python -m src.core.cli plan-refresh    — smart refresh plan (JSON)
    python -m src.core.cli get-insights    — active insights (JSON)
    python -m src.core.cli generate-insights — generate daily insights (JSON)
    python -m src.core.cli dismiss-insight  — dismiss an insight
    python -m src.core.cli follow-up-insight — follow up on an insight

The CLI uses the default data path (~/.arandu/data/) unless --data-dir is
passed explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.core.chromadb.engine import COLLECTION_NAMES
from src.core.data_layer import DataLayer
from src.core.db_helpers import (
    safe_str,
    table_exists,
    utc_ago_iso,
    utc_now_iso,
)
from src.core.kuzu.schema import ALL_NODE_TABLES, ALL_REL_TABLES
from src.core.sqlite.column_tiers import get_column_tier

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON serialization helper
# ---------------------------------------------------------------------------


class _DateTimeEncoder(json.JSONEncoder):
    """Encode datetime objects to ISO format strings.

    sensitivity_tier: N/A
    """

    def default(self, o: Any) -> Any:
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        if hasattr(o, "__str__") and not isinstance(
            o, (str, int, float, bool),
        ):
            return str(o)
        return super().default(o)


def _json_output(data: Any) -> str:
    """Serialize data to compact JSON with datetime support.

    sensitivity_tier: N/A
    """
    return json.dumps(data, cls=_DateTimeEncoder, ensure_ascii=False)


def _emit_json(data: Any) -> int:
    """Print a JSON object to stdout and return exit code 0.

    Convenience for the ``{ok: bool, ...}`` command contract the Rust
    bridge parses. Returns 0 even for ``{ok: false}`` payloads: the
    error is carried in the JSON, and a non-zero exit would make the
    bridge reject the invoke before the frontend can read it.

    sensitivity_tier: N/A
    """
    print(_json_output(data), flush=True)
    return 0


def _parse_reply_context_arg(raw: str | None) -> dict[str, Any] | None:
    """Parse the --reply-context CLI arg into a normalized dict.

    The Rust IPC serializes ``types::ReplyContext`` to JSON and passes
    it as a single positional value. We only accept entries with both
    ``source`` and ``message_id`` populated — anything else collapses
    to ``None`` so the Brain falls back to its inference path.

    sensitivity_tier: 2
    """
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    source = str(parsed.get("source") or "").strip()
    message_id = str(parsed.get("message_id") or "").strip()
    if not source or not message_id:
        return None
    contact_name = parsed.get("contact_name")
    return {
        "source": source,
        "message_id": message_id,
        "contact_name": (
            str(contact_name).strip() if contact_name else None
        ),
    }


def _parse_task_context_arg(raw: str | None) -> dict[str, Any] | None:
    """Parse the --task-context CLI arg into a normalized dict.

    sensitivity_tier: 2
    """
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    task_id = str(parsed.get("task_id") or "").strip()
    if not task_id:
        return None
    goal_id = parsed.get("goal_id")
    return {
        "task_id": task_id,
        "goal_id": str(goal_id).strip() if goal_id else None,
    }


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------


def cmd_init(layer: DataLayer) -> int:
    """Initialize schemas in all three databases.

    Args:
        layer: An open DataLayer instance.

    Returns:
        Exit code (0 = success, 1 = failure).
    """
    try:
        layer.initialize()
        print("✓  Arandu databases initialized successfully.")
        return 0
    except Exception as exc:
        print(f"✗  Initialization failed: {exc}", file=sys.stderr)
        logger.exception("init failed")
        return 1


def cmd_status(layer: DataLayer) -> int:
    """Print health status and document / row / node counts.

    Args:
        layer: An open DataLayer instance.

    Returns:
        Exit code (0 = all healthy, 1 = one or more engines down).
    """
    ok, report = layer.health_check()
    status_line = "HEALTHY" if ok else "DEGRADED"
    print(f"\nArandu data layer — {status_line}")
    print(f"  DuckDB  : {'OK' if report.duckdb_ok else 'FAIL'}")
    print(f"  Kuzu    : {'OK' if report.kuzu_ok else 'FAIL'}")
    print(f"  ChromaDB: {'OK' if report.chromadb_ok else 'FAIL'}")
    if report.errors:
        print("\nErrors:")
        for err in report.errors:
            print(f"  ! {err}")

    stats = layer.get_stats()
    print("\n── SQLite tables ──────────────────────────")
    for table, count in stats.sqlite.items():
        print(f"  {table:<30} {count:>6} rows")
    print(f"  {'TOTAL':<30} {stats.total_sqlite_rows:>6} rows")

    print("\n── Kuzu node types ────────────────────────")
    for node_type, count in stats.kuzu_nodes.items():
        print(f"  {node_type:<30} {count:>6} nodes")
    print(f"  {'TOTAL':<30} {stats.total_kuzu_nodes:>6} nodes")

    print("\n── ChromaDB collections ───────────────────")
    for collection, count in stats.chromadb.items():
        print(f"  {collection:<30} {count:>6} docs")
    print(f"  {'TOTAL':<30} {stats.total_chroma_docs:>6} docs")
    print()

    return 0 if ok else 1


def cmd_reset(layer: DataLayer) -> int:
    """Wipe all stored data and reinitialize from scratch.

    Args:
        layer: An open DataLayer instance.

    Returns:
        Exit code (0 = success, 1 = failure).
    """
    try:
        print("⚠  Resetting all Arandu data — this cannot be undone.")
        layer.reset()
        print("✓  Reset complete. All databases reinitialized.")
        return 0
    except Exception as exc:
        print(f"✗  Reset failed: {exc}", file=sys.stderr)
        logger.exception("reset failed")
        return 1


# ---------------------------------------------------------------------------
# Data ownership — export + delete (Settings › Privacy)
# ---------------------------------------------------------------------------


def _default_export_dir() -> Path:
    """Where an export archive lands when the caller doesn't specify.

    Prefers ``~/Downloads`` (where a user looks first); falls back to
    ``~/.arandu/exports`` when it doesn't exist (headless/CI).

    sensitivity_tier: 1
    """
    downloads = Path.home() / "Downloads"
    if downloads.is_dir():
        return downloads
    return Path.home() / ".arandu" / "exports"


def cmd_export_data(layer: DataLayer, output_dir: str | None = None) -> int:
    """Export all local user data as an inspectable JSON archive (zip).

    Dumps every SQLite table — raw_* ingested data, derived marts,
    learned facts, goals/tasks, the notification log, everything — to
    one JSON file per table, plus a ``manifest.json`` (table → row
    count), bundled into a single timestamped zip. The Kuzu graph and
    Chroma vectors are *derived* from these tables, so the SQLite dump
    is a complete, human-readable record of the user's own data.

    Emits JSON: ``{ok, path, tables, total_rows, bytes}``.

    sensitivity_tier: 3 (exports all personal data)
    """
    import zipfile

    try:
        db = layer.duckdb
        tables = [
            str(r["name"])
            for r in db.query(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            )
        ]

        dest_dir = Path(output_dir) if output_dir else _default_export_dir()
        dest_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        dest = dest_dir / f"arandu-export-{ts}.zip"

        manifest: dict[str, Any] = {
            "app": "arandu",
            "format_version": 1,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "tables": {},
        }
        total_rows = 0
        with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
            for table in tables:
                rows = db.query(f'SELECT * FROM "{table}"')  # noqa: S608
                total_rows += len(rows)
                manifest["tables"][table] = len(rows)
                zf.writestr(
                    f"tables/{table}.json",
                    _json_output(rows),
                )
            zf.writestr("manifest.json", _json_output(manifest))

        return _emit_json({
            "ok": True,
            "path": str(dest),
            "tables": len(tables),
            "total_rows": total_rows,
            "bytes": dest.stat().st_size,
        })
    except Exception as exc:  # noqa: BLE001
        logger.exception("export-data failed")
        return _emit_json({"ok": False, "error": str(exc)})


def cmd_delete_all_data(data_dir: str | None = None) -> int:
    """Permanently delete all local user data.

    Removes the entire data directory (``~/.arandu/data`` by default):
    the SQLite / Kuzu / Chroma stores, learned facts, goals & tasks,
    the audit chain, the WhatsApp listener session, and every derived
    cache. App preferences (``~/.arandu/settings.json``) and the Python
    runtime are preserved, so the app stays launchable and restarts
    into a fresh, empty state.

    The Pro cloud account and its keychain session are cleared
    separately by the Pro overlay (out of OSS scope).

    Runs *before* any DataLayer is opened (dispatched ahead of the
    engine context in ``main``) so no engine holds the directory open
    while it's wiped. Emits JSON: ``{ok, path, existed}``.

    sensitivity_tier: 3
    """
    import shutil

    try:
        base = (
            Path(data_dir) if data_dir else Path.home() / ".arandu" / "data"
        ).expanduser().resolve()

        # Safety rails — this is an irreversible recursive delete. Refuse
        # anything that isn't a specific data directory: never the
        # filesystem root, $HOME, or the process CWD (a bare or empty
        # ``--data-dir`` resolves to "." → the app install dir).
        forbidden = {
            Path(base.anchor),  # "/" (or a drive root on Windows)
            Path.home().resolve(),
            Path.cwd().resolve(),
        }
        if base in forbidden:
            return _emit_json({
                "ok": False,
                "error": f"refusing to delete unsafe path: {base}",
            })

        existed = base.exists()
        if existed:
            shutil.rmtree(base)
        # Leave an empty data dir so the next launch initializes cleanly.
        base.mkdir(parents=True, exist_ok=True)
        return _emit_json({
            "ok": True,
            "path": str(base),
            "existed": existed,
        })
    except Exception as exc:  # noqa: BLE001
        logger.exception("delete-all-data failed")
        return _emit_json({"ok": False, "error": str(exc)})


# ---------------------------------------------------------------------------
# JSON command implementations (for Tauri bridge)
# ---------------------------------------------------------------------------


def cmd_stats(layer: DataLayer) -> int:
    """Output database stats as JSON to stdout.

    Args:
        layer: An open DataLayer instance.

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 1
    """
    try:
        stats = layer.get_stats()
        ok, _report = layer.health_check()
        result = {
            "healthy": ok,
            "sqlite": stats.sqlite,
            "kuzu_nodes": stats.kuzu_nodes,
            "chromadb": stats.chromadb,
            "total_sqlite_rows": stats.total_sqlite_rows,
            "total_kuzu_nodes": stats.total_kuzu_nodes,
            "total_chroma_docs": stats.total_chroma_docs,
        }
        print(_json_output(result))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def _normalize_message_bools(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert SQLite 0/1 integer columns to Python bools for JSON output.

    Rust ``Message`` struct expects ``is_group`` as ``Option<bool>``.
    SQLite stores booleans as INTEGER 0/1, which serializes as JSON
    integers — serde rejects ``0`` for ``bool``. This normalizes them.

    sensitivity_tier: N/A
    """
    bool_cols = ("is_group", "is_from_me")
    for row in rows:
        for col in bool_cols:
            if col in row:
                row[col] = bool(row[col])
    return rows


def _message_select_columns(engine: Any) -> str:
    """Build a SELECT column list for raw_messages.

    Always includes core columns; adds migration-added columns only if
    they exist.  This avoids BinderErrors when the migration hasn't run
    yet.

    sensitivity_tier: 1
    """
    core = "id, source, sender, recipient, content, timestamp, sensitivity_tier"
    try:
        existing = {
            r["name"]
            for r in engine.query(
                "PRAGMA table_info(raw_messages)",
            )
        }
    except Exception:
        return core

    extras: list[str] = []
    for col in ("sender_name", "chat_name", "is_group", "is_from_me"):
        if col in existing:
            extras.append(col)
    if extras:
        return f"{core}, {', '.join(extras)}"
    return core


def cmd_query_messages(layer: DataLayer, limit: int, offset: int = 0) -> int:
    """Query recent messages and output as JSON to stdout.

    Args:
        layer: An open DataLayer instance.
        limit: Maximum number of messages to return.
        offset: Number of rows to skip (for pagination).

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 2
    """
    try:
        cols = _message_select_columns(layer.duckdb)
        rows = layer.duckdb.query(
            f"SELECT {cols} "  # noqa: S608
            "FROM raw_messages "
            "ORDER BY timestamp DESC "
            "LIMIT ? OFFSET ?",
            [limit, offset],
        )
        print(_json_output(_normalize_message_bools(rows)))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_query_events(
    layer: DataLayer,
    days: int,
    limit: int = 50,
    offset: int = 0,
) -> int:
    """Query calendar events and output as JSON to stdout.

    When *days* is ``0`` all events are returned (no date filter).
    Otherwise only events from today through today + *days* are returned.

    Args:
        layer: An open DataLayer instance.
        days: Number of days ahead to query.  ``0`` means all events.
        limit: Maximum number of events to return.
        offset: Number of rows to skip (for pagination).

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 2
    """
    try:
        if days > 0:
            today = date.today()
            end_date = today + timedelta(days=days)
            rows = layer.duckdb.query(
                "SELECT id, title, description, start_time, end_time, "
                "location, attendees, sensitivity_tier, "
                "COALESCE(event_origin, 'personal') AS event_origin "
                "FROM raw_calendar_events "
                "WHERE DATE(start_time) >= DATE(?) "
                "AND DATE(start_time) <= DATE(?) "
                "ORDER BY start_time "
                "LIMIT ? OFFSET ?",
                [today.isoformat(), end_date.isoformat(), limit, offset],
            )
        else:
            rows = layer.duckdb.query(
                "SELECT id, title, description, start_time, end_time, "
                "location, attendees, sensitivity_tier, "
                "COALESCE(event_origin, 'personal') AS event_origin "
                "FROM raw_calendar_events "
                "ORDER BY start_time DESC "
                "LIMIT ? OFFSET ?",
                [limit, offset],
            )
        print(_json_output(rows))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_query_contacts(
    layer: DataLayer,
    limit: int = 500,
    offset: int = 0,
) -> int:
    """Query contacts from raw_contacts and output as JSON.

    Args:
        layer: An open DataLayer instance.
        limit: Maximum number of contacts to return.
        offset: Number of rows to skip (for pagination).

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 2
    """
    try:
        rows = layer.duckdb.query(
            "SELECT id, name, email, phone, "
            "relationship, birthday, address, notes, "
            "sensitivity_tier "
            "FROM raw_contacts "
            "ORDER BY name "
            "LIMIT ? OFFSET ?",
            [limit, offset],
        )
        print(_json_output(rows))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_query_notes(
    layer: DataLayer,
    limit: int = 100,
    offset: int = 0,
) -> int:
    """Query notes from raw_notes and output as JSON.

    Args:
        layer: An open DataLayer instance.
        limit: Maximum number of notes to return.
        offset: Number of rows to skip (for pagination).

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 2
    """
    try:
        rows = layer.duckdb.query(
            "SELECT id, title, content, source, created_at, "
            "updated_at, tags, sensitivity_tier "
            "FROM raw_notes "
            "ORDER BY updated_at DESC "
            "LIMIT ? OFFSET ?",
            [limit, offset],
        )
        print(_json_output(rows))
        return 0
    except Exception as exc:
        if "does not exist" in str(exc).lower():
            print(_json_output([]))
            return 0
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_fix_notes_content(layer: DataLayer) -> int:
    """Re-extract note content from macOS Notes DB and update DuckDB rows.

    Reads the original Notes SQLite, re-parses protobuf blobs with the
    fixed parser, and updates any rows whose content has changed.

    Args:
        layer: An open DataLayer instance (read-write).

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 2
    """
    from src.extensions.bridges.apple.server import _read_notes_sqlite

    try:
        existing = layer.duckdb.query(
            "SELECT id, content FROM raw_notes",
        )
    except Exception:
        print(_json_output({"fixed": 0, "error": "raw_notes not found"}))
        return 0

    existing_map = {row["id"]: row["content"] for row in existing}
    if not existing_map:
        print(_json_output({"fixed": 0}))
        return 0

    fresh = _read_notes_sqlite(limit=len(existing_map) + 500)
    fresh_map = {n["id"]: n for n in fresh}

    fixed = 0
    for note_id, old_content in existing_map.items():
        new_note = fresh_map.get(note_id)
        if new_note is None:
            continue
        new_content = new_note.get("content", "")
        if new_content and new_content != old_content:
            layer.duckdb.execute(
                "UPDATE raw_notes SET content = ?, title = ? WHERE id = ?",
                [new_content, new_note.get("title", ""), note_id],
            )
            fixed += 1

    print(_json_output({"fixed": fixed, "total": len(existing_map)}))
    return 0


def cmd_query_emails(
    layer: DataLayer,
    limit: int = 200,
    offset: int = 0,
) -> int:
    """Query emails from raw_emails and output as JSON.

    Args:
        layer: An open DataLayer instance.
        limit: Maximum number of emails to return.
        offset: Number of rows to skip (for pagination).

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 2
    """
    try:
        rows = layer.duckdb.query(
            "SELECT id, subject, from_address, to_addresses, "
            "date, body_preview, is_read, folder, "
            "sensitivity_tier "
            "FROM raw_emails "
            "ORDER BY date DESC "
            "LIMIT ? OFFSET ?",
            [limit, offset],
        )
        for row in rows:
            if "is_read" in row:
                row["is_read"] = bool(row["is_read"])
        print(_json_output(rows))
        return 0
    except Exception as exc:
        if "does not exist" in str(exc).lower():
            print(_json_output([]))
            return 0
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# Generic table browsing commands
# ---------------------------------------------------------------------------

_ALLOWED_TABLE_PREFIXES = ("raw_", "stg_", "int_", "mart_", "ext_")


def _validate_table_prefix(table_name: str) -> bool:
    """Check table name starts with an allowed prefix.

    sensitivity_tier: N/A
    """
    return any(table_name.startswith(p) for p in _ALLOWED_TABLE_PREFIXES)


def _collect_table_info(
    layer: DataLayer,
    table_name: str,
) -> dict[str, Any]:
    """Build a table info dict for a given table.

    sensitivity_tier: 1
    """
    cols = layer.duckdb.query(
        f"PRAGMA table_info({table_name})",
    )
    columns = [
        {"name": c["name"], "type": c["type"] or "TEXT"}
        for c in cols
    ]

    try:
        count_rows = layer.duckdb.query(
            f"SELECT COUNT(*) AS cnt FROM {table_name}",
        )
        row_count = count_rows[0]["cnt"] if count_rows else 0
    except Exception:
        row_count = -1

    return {
        "table_name": table_name,
        "row_count": row_count,
        "column_count": len(columns),
        "columns": columns,
    }


def cmd_list_tables(layer: DataLayer, prefix: str = "") -> int:
    """List DuckDB tables matching a prefix with row counts and columns.

    Discovers tables from both the 'main' schema (raw tables) and
    SQLMesh-managed schemas (stg/int/mart pipeline tables).

    Args:
        layer: An open DataLayer instance.
        prefix: Table name prefix filter (e.g. 'raw_', 'mart_').

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 1
    """
    try:
        # Validate prefix if provided
        if prefix and not _validate_table_prefix(prefix):
            print(
                _json_output({"error": f"Invalid prefix: {prefix!r}"}),
                file=sys.stderr,
            )
            return 1

        result: list[dict[str, Any]] = []

        # Discover all tables from sqlite_master
        filter_clause = ""
        params: list[Any] = []
        if prefix:
            filter_clause = "AND name LIKE ?"
            params = [f"{prefix}%"]

        tables = layer.duckdb.query(
            "SELECT name FROM sqlite_master "
            f"WHERE type = 'table' {filter_clause} "
            "ORDER BY name",
            params,
        )

        for tbl in tables:
            name = tbl["name"]
            if not _validate_table_prefix(name):
                continue
            result.append(_collect_table_info(layer, name))

        # Sort by table name
        result.sort(key=lambda r: r["table_name"])

        print(_json_output(result))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_list_pipeline_models() -> int:
    """List all models registered in the pipeline manifest.

    Returns name, layer, model_type, and depends_on for each model so the
    Data Models page can surface registered-but-unmaterialized models with
    a "not built yet" indicator alongside the live SQLite tables.

    sensitivity_tier: 1
    """
    try:
        from src.pipeline.manifest import load_manifest

        manifest = load_manifest()
        models = [
            {
                "name": m.name,
                "layer": m.layer,
                "model_type": m.model_type,
                "depends_on": list(m.depends_on),
            }
            for m in manifest.models
        ]
        print(_json_output(models))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_query_table(
    layer: DataLayer,
    table_name: str,
    limit: int = 25,
    offset: int = 0,
) -> int:
    """Query sample rows from a whitelisted table.

    Args:
        layer: An open DataLayer instance.
        table_name: Exact table name (must pass whitelist check).
        limit: Maximum rows to return (capped at 100).
        offset: Number of rows to skip.

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 2
    """
    # Validate table name prefix
    if not _validate_table_prefix(table_name):
        print(
            _json_output(
                {"error": f"Table not allowed: {table_name!r}"},
            ),
            file=sys.stderr,
        )
        return 1

    # Cap limit
    limit = min(limit, 100)

    try:
        if not table_exists(layer.duckdb, table_name):
            print(_json_output({
                "table_name": table_name,
                "total_rows": 0,
                "columns": [],
                "rows": [],
            }))
            return 0

        # Get column metadata via PRAGMA
        col_rows = layer.duckdb.query(
            f"PRAGMA table_info({table_name})",
        )
        columns: list[dict[str, Any]] = []
        for c in col_rows:
            entry: dict[str, Any] = {
                "name": c["name"],
                "type": c["type"] or "TEXT",
            }
            tier = get_column_tier(table_name, c["name"])
            if tier is not None:
                entry["tier"] = tier
            columns.append(entry)

        # Find best ordering column (timestamp-like name)
        order_clause = ""
        for col in col_rows:
            col_name = col["name"]
            if col_name in (
                "timestamp", "created_at", "start_time",
                "recorded_at", "date", "occurred_at",
            ):
                order_clause = f"ORDER BY {col_name} DESC"
                break

        # Get total count
        count_rows = layer.duckdb.query(
            f"SELECT COUNT(*) AS cnt FROM {table_name}",
        )
        total = count_rows[0]["cnt"] if count_rows else 0

        # Get sample rows
        rows = layer.duckdb.query(
            f"SELECT * FROM {table_name} {order_clause} "
            f"LIMIT {limit} OFFSET {offset}",
        )

        print(_json_output({
            "table_name": table_name,
            "total_rows": total,
            "columns": columns,
            "rows": rows,
        }))
        return 0
    except Exception as exc:
        if "no such table" in str(exc).lower():
            print(_json_output({
                "table_name": table_name,
                "total_rows": 0,
                "columns": [],
                "rows": [],
            }))
            return 0
        print(
            _json_output({"error": str(exc)}),
            file=sys.stderr,
        )
        return 1


_GRAPH_BUSY_MESSAGE = (
    "Graph is being updated by the pipeline. Try again in a moment."
)


def _graph_error_message(exc: Exception) -> str:
    """Map a graph access failure to a user-facing message.

    Kuzu's read-write lock is held exclusively while the pipeline writes
    graph nodes, blocking every read with "Could not set lock on file".
    That is transient contention, not a real fault, so we surface a clear,
    retry-able message instead of the raw Kuzu IO exception.
    """
    if "set lock on file" in str(exc).lower():
        return _GRAPH_BUSY_MESSAGE
    return str(exc)


def cmd_graph_summary(layer: DataLayer) -> int:
    """Return node and relationship type counts from the Kuzu graph.

    Args:
        layer: An open DataLayer instance.

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 1
    """
    try:
        # Force the engine open up front so a failure to reach the graph
        # (e.g. the pipeline holding Kuzu's read-write lock) surfaces as an
        # error the UI can show, instead of being swallowed by the per-table
        # guards below and rendered as a misleading all-zeros "empty graph".
        graph = layer.kuzu

        nodes: list[dict[str, Any]] = []
        for node_type in ALL_NODE_TABLES:
            try:
                rows = graph.query(
                    f"MATCH (n:{node_type}) RETURN count(n) AS cnt",
                )
                count = rows[0]["cnt"] if rows else 0
            except Exception:
                count = 0
            nodes.append({"name": node_type, "count": count})

        relationships: list[dict[str, Any]] = []
        for rel_type in ALL_REL_TABLES:
            try:
                rows = graph.query(
                    f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS cnt",
                )
                count = rows[0]["cnt"] if rows else 0
            except Exception:
                count = 0
            relationships.append({"name": rel_type, "count": count})

        total_nodes = sum(n["count"] for n in nodes)
        total_rels = sum(r["count"] for r in relationships)

        print(_json_output({
            "nodes": nodes,
            "relationships": relationships,
            "total_nodes": total_nodes,
            "total_relationships": total_rels,
        }))
        return 0
    except Exception as exc:
        print(
            _json_output({"error": _graph_error_message(exc)}),
            file=sys.stderr,
        )
        return 1


def cmd_query_graph_nodes(
    layer: DataLayer,
    node_type: str,
    limit: int = 25,
) -> int:
    """Query sample nodes of a given type from the Kuzu graph.

    Args:
        layer: An open DataLayer instance.
        node_type: Node type name (must be in ALL_NODE_TABLES).
        limit: Maximum nodes to return (capped at 100).

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 2
    """
    if node_type not in ALL_NODE_TABLES:
        print(
            _json_output({"error": f"Unknown node type: {node_type!r}"}),
            file=sys.stderr,
        )
        return 1

    limit = min(limit, 100)

    try:
        count_rows = layer.kuzu.query(
            f"MATCH (n:{node_type}) RETURN count(n) AS cnt",
        )
        total = count_rows[0]["cnt"] if count_rows else 0

        rows = layer.kuzu.query(
            f"MATCH (n:{node_type}) RETURN n LIMIT {limit}",
        )
        # Kuzu returns nodes as dicts under the 'n' key
        nodes = []
        for row in rows:
            node = row.get("n", row)
            if isinstance(node, dict):
                # Filter out internal Kuzu keys
                cleaned = {
                    k: _serialize_value(v)
                    for k, v in node.items()
                    if not k.startswith("_")
                }
                nodes.append(cleaned)
            else:
                nodes.append({"value": str(node)})

        print(_json_output({
            "node_type": node_type,
            "total": total,
            "nodes": nodes,
        }))
        return 0
    except Exception as exc:
        print(
            _json_output({"error": _graph_error_message(exc)}),
            file=sys.stderr,
        )
        return 1


def cmd_query_graph_rels(
    layer: DataLayer,
    rel_type: str,
    limit: int = 25,
) -> int:
    """Query sample relationships of a given type from the Kuzu graph.

    Args:
        layer: An open DataLayer instance.
        rel_type: Relationship type name (must be in ALL_REL_TABLES).
        limit: Maximum relationships to return (capped at 100).

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 2
    """
    if rel_type not in ALL_REL_TABLES:
        print(
            _json_output(
                {"error": f"Unknown relationship type: {rel_type!r}"},
            ),
            file=sys.stderr,
        )
        return 1

    limit = min(limit, 100)

    try:
        count_rows = layer.kuzu.query(
            f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS cnt",
        )
        total = count_rows[0]["cnt"] if count_rows else 0

        rows = layer.kuzu.query(
            f"MATCH (a)-[r:{rel_type}]->(b) "
            f"RETURN a.id AS source_id, a.name AS source_name, "
            f"b.id AS target_id, b.name AS target_name, "
            f"r.weight AS weight, r.timestamp AS timestamp, "
            f"r.sensitivity_tier AS sensitivity_tier "
            f"LIMIT {limit}",
        )

        rels = []
        for row in rows:
            rels.append({
                "source_id": str(row.get("source_id", "")),
                "source_name": str(row.get("source_name", "")),
                "target_id": str(row.get("target_id", "")),
                "target_name": str(row.get("target_name", "")),
                "weight": row.get("weight"),
                "timestamp": _serialize_value(row.get("timestamp")),
                "sensitivity_tier": row.get("sensitivity_tier"),
            })

        print(_json_output({
            "rel_type": rel_type,
            "total": total,
            "relationships": rels,
        }))
        return 0
    except Exception as exc:
        print(
            _json_output({"error": _graph_error_message(exc)}),
            file=sys.stderr,
        )
        return 1


def _serialize_value(val: Any) -> Any:
    """Convert non-JSON-serializable values to strings.

    sensitivity_tier: N/A
    """
    if val is None:
        return None
    if isinstance(val, (str, int, float, bool)):
        return val
    if isinstance(val, datetime):
        return val.isoformat()
    return str(val)


def cmd_vector_summary(layer: DataLayer) -> int:
    """Return ChromaDB collection counts with sample documents.

    Args:
        layer: An open DataLayer instance.

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 2
    """
    try:
        collections: list[dict[str, Any]] = []
        for name in COLLECTION_NAMES:
            try:
                col = layer.chromadb.get_or_create_collection(name)
                count = col.count()
                samples: list[dict[str, Any]] = []
                if count > 0:
                    peek = col.peek(limit=min(5, count))
                    for i in range(len(peek["ids"])):
                        samples.append({
                            "id": peek["ids"][i],
                            "document": (
                                peek["documents"][i][:200]
                                if peek["documents"] and peek["documents"][i]
                                else ""
                            ),
                            "metadata": (
                                peek["metadatas"][i]
                                if peek["metadatas"]
                                else {}
                            ),
                        })
            except Exception:
                count = 0
                samples = []
            collections.append({
                "name": name,
                "count": count,
                "samples": samples,
            })

        print(_json_output({"collections": collections}))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_query_today(layer: DataLayer) -> int:
    """Query today's summary (events + recent messages) as JSON.

    Args:
        layer: An open DataLayer instance.

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 2
    """
    try:
        today = date.today()

        events = layer.duckdb.query(
            "SELECT id, title, description, start_time, end_time, "
            "location, attendees, sensitivity_tier, "
            "COALESCE(event_origin, 'personal') AS event_origin "
            "FROM raw_calendar_events "
            "WHERE DATE(start_time) = DATE(?) "
            "ORDER BY start_time",
            [today.isoformat()],
        )

        msg_cols = _message_select_columns(layer.duckdb)
        messages = layer.duckdb.query(
            f"SELECT {msg_cols} "  # noqa: S608
            "FROM raw_messages "
            "ORDER BY timestamp DESC LIMIT 10",
        )

        notes_count_rows = layer.duckdb.query(
            "SELECT COUNT(*) AS n FROM raw_notes",
        )
        notes_count = notes_count_rows[0]["n"] if notes_count_rows else 0

        result = {
            "date": today.isoformat(),
            "events": events,
            "recent_messages": _normalize_message_bools(messages),
            "notes_count": notes_count,
        }
        print(_json_output(result))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_profile(layer: DataLayer) -> int:
    """Run a standard benchmark suite and print a performance report.

    Uses a temporary data directory for clean, reproducible results.
    Exercises every major subsystem — init, queries, brain agent,
    and ChromaDB re-indexing — then prints a ranked table of the
    slowest operations.

    Args:
        layer: An open DataLayer instance (unused — profile creates
               its own temporary DataLayer for isolation).

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 3
    """
    import tempfile

    from src.core.profiler import PerformanceLog, timed_block

    perf = PerformanceLog.get()
    perf.clear()

    print("Arandu Performance Profiler")
    print("=" * 50)
    print()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / "profile_data"

        with DataLayer(base_path=tmp_path) as pl:
            # Phase 1 — Initialize
            print("  [1/5] Initializing databases...")
            with timed_block("cli.initialize"):
                pl.initialize()

            # Phase 2 — Health check + stats
            print("  [2/5] Health check + stats...")
            with timed_block("cli.health_check"):
                pl.health_check()
            with timed_block("cli.get_stats"):
                pl.get_stats()

            # Phase 3 — ChromaDB reindex
            print("  [3/5] ChromaDB full reindex...")
            with timed_block("cli.reindex"):
                pl.reindex()

            # Phase 4 — Query engine
            print("  [4/5] Query engine queries...")
            from src.core.query_engine import QueryEngine

            qe = QueryEngine(
                duckdb=pl.duckdb,
                kuzu=pl.kuzu,
                chromadb=pl.chromadb,
            )
            qe.query("What happened today?")
            qe.query("Tell me about Alice")
            qe.query("How is my health?")

            # Phase 5 — Brain Agent
            print("  [5/5] Brain Agent (ask)...")
            try:
                from src.agents.brain import BrainAgentV2
                from src.models.llm_provider import (
                    create_provider_from_settings,
                )

                provider = create_provider_from_settings(
                    background=True,
                )
                agent = BrainAgentV2(
                    query_engine=qe,
                    provider=provider,
                )
                agent.ask(
                    "What's on my schedule today?",
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"    (Ollama unavailable: {exc})",
                )

    print(perf.report())
    return 0


def _user_agent_chunks(
    agent_id: str,
    question: str,
    *,
    query_engine: Any,
) -> Any:
    """Yield streaming chunks for a user-authored agent's answer.

    Mirrors :meth:`BrainAgentV2.ask_stream` chunk shape so the
    frontend can reuse the existing stream handler.

    sensitivity_tier: 3
    """
    from src.agents.brain import bootstrap_agents
    from src.agents.core.registry import get_agent

    bootstrap_agents(query_engine=query_engine)
    definition = get_agent(agent_id)
    if definition is None or definition.factory is None:
        yield {"type": "error", "error": f"unknown agent: {agent_id}"}
        return
    agent = definition.factory()
    record = agent.run(question)
    if record.output is None:
        err = record.error or "agent returned no output"
        yield {"type": "error", "error": err}
        return
    response = record.output
    yield {
        "type": "context",
        "context_summary": getattr(response, "context_summary", ""),
        "sources": list(getattr(response, "sources", []) or []),
    }
    yield {
        "type": "token",
        "token": getattr(response, "answer", str(response)),
    }
    yield {
        "type": "done",
        "model": getattr(response, "model", agent_id),
        "latency_ms": record.duration_ms,
    }


def _emit_stream(
    chunks: Any,
    *,
    layer: DataLayer | None = None,
    session_id: str | None = None,
) -> None:
    """Forward stream chunks to stdout and optionally persist the result.

    The accumulator mirrors the Rust ``StreamCollector`` so what we
    store matches what the frontend renders. Persistence only happens
    when ``layer`` and ``session_id`` are both supplied.

    sensitivity_tier: 3
    """
    store = None
    if layer is not None and session_id:
        from src.core.chat_store import ChatStore

        store = ChatStore(layer.duckdb)

    parts_by_id: dict[str, dict[str, Any]] = {}
    parts_order: list[str] = []
    text_buf: list[str] = []
    thinking_buf: list[str] = []
    sources: list[Any] = []
    model: str | None = None
    latency_ms: float | None = None
    saw_error = False

    for chunk in chunks:
        print(_json_output(chunk), flush=True)
        if store is None:
            continue
        ty = chunk.get("type") if isinstance(chunk, dict) else None
        if ty == "context":
            srcs = chunk.get("sources")
            if isinstance(srcs, list):
                sources = list(srcs)
        elif ty == "token":
            tok = chunk.get("token")
            if isinstance(tok, str):
                text_buf.append(tok)
        elif ty == "thinking":
            tok = chunk.get("token") or chunk.get("text")
            if isinstance(tok, str):
                thinking_buf.append(tok)
        elif ty == "part_start":
            pid = chunk.get("part_id")
            if isinstance(pid, str) and pid not in parts_by_id:
                parts_order.append(pid)
                obj: dict[str, Any] = {"id": pid, "data": ""}
                for key in (
                    "mime", "title", "display",
                    "sensitivity_tier", "metadata",
                ):
                    if key in chunk:
                        obj[key] = chunk[key]
                parts_by_id[pid] = obj
        elif ty == "part_chunk":
            pid = chunk.get("part_id")
            data = chunk.get("data")
            if (
                isinstance(pid, str)
                and pid in parts_by_id
                and isinstance(data, str)
            ):
                cur = parts_by_id[pid].get("data") or ""
                parts_by_id[pid]["data"] = cur + data
        elif ty == "part_done":
            pid = chunk.get("part_id")
            if isinstance(pid, str) and pid in parts_by_id and "data" in chunk:
                parts_by_id[pid]["data"] = chunk["data"]
        elif ty == "done":
            m = chunk.get("model")
            if isinstance(m, str):
                model = m
            lat = chunk.get("latency_ms")
            if isinstance(lat, (int, float)):
                latency_ms = float(lat)
        elif ty == "error":
            saw_error = True

    if store is None or saw_error:
        return

    parts = [parts_by_id[pid] for pid in parts_order if pid in parts_by_id]
    text = "".join(text_buf)
    if not parts and not text:
        return

    store.append_message(
        session_id,  # type: ignore[arg-type]
        "assistant",
        text,
        parts=parts or None,
        sources=sources or None,
        latency_ms=latency_ms,
        model=model,
        thinking="".join(thinking_buf) or None,
    )


def cmd_ask_stream(
    layer: DataLayer,
    question: str,
    *,
    agent_id: str = "chat",
    session_id: str | None = None,
    reply_context: dict[str, Any] | None = None,
    task_context: dict[str, Any] | None = None,
    budget: str | None = None,
) -> int:
    """Ask an agent with streaming, outputting JSON lines to stdout.

    Each line is a complete JSON object:
      {"type": "context", "context_summary": "...", "sources": [...]}
      {"type": "token", "token": "Hello"}
      {"type": "done", "model": "...", "latency_ms": ...}
    or on error:
      {"type": "error", "error": "..."}

    Args:
        layer: An open DataLayer instance.
        question: The question to ask.
        agent_id: Target agent. Defaults to ``chat`` (the conversational
            orchestrator that calls Brain as a tool). Pass ``brain`` to
            bypass the chat layer and talk to Brain directly. Any
            other id routes to ``_stream_user_agent`` for that agent.

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 3
    """
    try:
        from src.agents.brain import BrainAgentV2, bootstrap_agents
        from src.agents.core.task_budget import TaskBudget, TaskClass
        from src.agents.tool_registry import ToolRegistry
        from src.core.query_engine import QueryEngine
        from src.extensions.connectors.catalog import ConnectorCatalog
        from src.extensions.connectors.registry import ExtensionRegistry
        from src.models.llm_provider import create_provider_from_settings

        budget_obj: TaskBudget | None = None
        if budget is not None:
            budget_obj = TaskBudget.for_class(TaskClass(budget))

        qe = QueryEngine(
            duckdb=layer.duckdb,
            kuzu=layer.kuzu,
            chromadb=layer.chromadb,
        )

        if session_id:
            from src.core.chat_store import ChatStore

            ChatStore(layer.duckdb).append_message(
                session_id, "user", question,
            )

        if agent_id == "chat":
            # Make sure brain + sub-agents are registered first so
            # ChatAgent's delegation tools can resolve them.
            bootstrap_agents(query_engine=qe)
            from src.agents.chat import ChatAgent
            provider = create_provider_from_settings()
            tool_registry = ToolRegistry(
                catalog=ConnectorCatalog(),
                registry=ExtensionRegistry(),
            )
            chat = ChatAgent(
                query_engine=qe,
                tool_registry=tool_registry,
                provider=provider,
            )
            _emit_stream(
                chat.ask_stream(
                    question,
                    reply_context=reply_context,
                    task_context=task_context,
                    budget=budget_obj,
                ),
                layer=layer,
                session_id=session_id,
            )
            return 0

        if agent_id and agent_id != "brain":
            _emit_stream(
                _user_agent_chunks(
                    agent_id, question, query_engine=qe,
                ),
                layer=layer,
                session_id=session_id,
            )
            return 0

        provider = create_provider_from_settings()
        tool_registry = ToolRegistry(
            catalog=ConnectorCatalog(),
            registry=ExtensionRegistry(),
        )
        v2 = BrainAgentV2(
            query_engine=qe,
            tool_registry=tool_registry,
            provider=provider,
        )
        _emit_stream(
            v2.ask_stream(
                question,
                reply_context=reply_context,
                budget=budget_obj,
            ),
            layer=layer,
            session_id=session_id,
        )
        return 0
    except Exception as exc:
        # The error chunk we just emitted already surfaces to the user
        # via the stream. Exiting non-zero would also cause the Rust
        # bridge to reject the invoke promise, which the frontend turns
        # into a *second* error bubble ("Python CLI stream exited with
        # code 1") on top of the chunk message. Return 0 so the user
        # sees one clean error.
        print(
            _json_output({"type": "error", "error": str(exc)}),
            flush=True,
        )
        return 0


def cmd_stop_research(run_id: str) -> int:
    """Signal a running orchestrator to stop researching.

    Looks up ``run_id`` in :mod:`src.agents.core.cancel_registry` and
    sets the cancel event. The in-flight orchestrator continues running;
    at its next reflection checkpoint it sees the event, injects a
    STOP_REQUEST user message into the pydantic-ai loop, and wraps up
    with whatever context it already has — no SIGTERM, no kill.

    Always returns 0 with ``{"ok": <found>}`` so the frontend can show
    "Already finished" vs "Stopping" gracefully.

    sensitivity_tier: 1
    """
    from src.agents.core.cancel_registry import request_cancel
    found = request_cancel(run_id)
    print(_json_output({"ok": True, "found": found}), flush=True)
    return 0


def cmd_ollama_status() -> int:
    """Check Ollama health and return status as JSON.

    Includes the configured LLM provider type in the response.

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 1
    """
    try:
        from src.models.llm_provider import load_llm_settings
        from src.models.ollama_manager import (
            DEFAULT_CHAT_MODEL,
            DEFAULT_HOST,
            OllamaManager,
        )

        settings = load_llm_settings()
        mgr = OllamaManager(
            host=settings.get("llm_host", DEFAULT_HOST),
            chat_model=settings.get("llm_model") or DEFAULT_CHAT_MODEL,
        )
        status = mgr.get_status_dict()
        status["provider"] = settings.get("llm_provider", "ollama")
        print(_json_output(status))
        return 0
    except Exception as exc:
        print(
            _json_output({"error": str(exc)}),
            file=sys.stderr,
        )
        return 1


def cmd_ollama_preload() -> int:
    """Start Ollama if needed, then preload the default chat model.

    Skips entirely when the configured provider is not Ollama,
    avoiding unnecessary memory usage from a loaded chat model.

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 1
    """
    try:
        from src.models.llm_provider import load_llm_settings
        from src.models.ollama_manager import (
            DEFAULT_CHAT_MODEL,
            DEFAULT_HOST,
            OllamaManager,
        )

        settings = load_llm_settings()
        if settings.get("llm_provider", "ollama") != "ollama":
            print(_json_output({"success": True, "skipped": True}))
            return 0

        # Don't pull/load before onboarding has chosen a model. Otherwise a
        # fresh launch would download the heavy default (llama3.1:70b, ~43 GB)
        # before the wizard's selection is saved. The wizard and the Settings
        # model picker call preload explicitly once a model is chosen.
        if not settings.get("onboarding_completed"):
            print(_json_output({"success": True, "skipped": "onboarding"}))
            return 0

        mgr = OllamaManager(
            host=settings.get("llm_host", DEFAULT_HOST),
            chat_model=settings.get("llm_model") or DEFAULT_CHAT_MODEL,
        )
        mgr.ensure_running()
        # Pull the configured model if it isn't present yet, then load it.
        pulled = mgr.ensure_model_pulled()
        ok = mgr.preload_model() if pulled else False
        print(_json_output({"success": ok, "pulled": pulled}))
        return 0 if ok else 1
    except Exception as exc:
        print(
            _json_output({"error": str(exc)}),
            file=sys.stderr,
        )
        return 1


def cmd_ollama_stop() -> int:
    """Stop the Ollama server if running.

    Called when the user switches to an external LLM provider
    so that Ollama doesn't consume memory in the background.

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 1
    """
    try:
        from src.models.ollama_manager import OllamaManager

        mgr = OllamaManager()
        ok = mgr.stop_server()
        print(_json_output({"success": ok}))
        return 0
    except Exception as exc:
        print(
            _json_output({"error": str(exc)}),
            file=sys.stderr,
        )
        return 1


def cmd_rebuild_vector_index(
    *,
    target_model: str,
    target_provider_kind: str,
    api_key: str | None,
    base_url: str | None,
    dimensions: int | None,
    dry_run: bool,
) -> int:
    """Rebuild ChromaDB + BM25 under a new embedding model.

    Wraps :mod:`src.core.chromadb.migrate` so the Settings UI can
    trigger a rebuild via one IPC call. Emits JSON suitable for the
    Tauri command to parse directly. Returns 0 on success, 1 on
    user-visible failure (printed under ``error``).

    sensitivity_tier: 1 (no record content in output)
    """
    import io
    import json as _json
    from contextlib import redirect_stderr, redirect_stdout

    from src.core.chromadb.engine import DEFAULT_DB_PATH
    from src.core.chromadb.migrate import migrate

    # migrate() writes progress to stdout — capture it so the IPC
    # caller sees the structured payload, not the progress narrative.
    out_buf = io.StringIO()
    err_buf = io.StringIO()
    try:
        with redirect_stdout(out_buf), redirect_stderr(err_buf):
            exit_code = migrate(
                target_model=target_model,
                target_provider_kind=target_provider_kind,
                db_path=DEFAULT_DB_PATH,
                api_key=api_key,
                base_url=base_url,
                dimensions=dimensions,
                apply=not dry_run,
            )
    except SystemExit as exc:
        print(_json.dumps({
            "ok": False,
            "error": str(exc),
            "progress": out_buf.getvalue(),
        }))
        return 1
    except Exception as exc:  # noqa: BLE001
        print(_json.dumps({
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "progress": out_buf.getvalue(),
        }))
        return 1

    print(_json.dumps({
        "ok": exit_code == 0,
        "exit_code": exit_code,
        "target_model": target_model,
        "provider": target_provider_kind,
        "dry_run": dry_run,
        "progress": out_buf.getvalue(),
    }))
    return 0 if exit_code == 0 else 1


def cmd_monitor() -> int:
    """Output memory usage and database file sizes as JSON.

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 1
    """
    try:
        from src.core.monitor import format_report, get_memory_usage

        report = get_memory_usage()
        print(_json_output(format_report(report)))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_pipeline_status(layer: DataLayer) -> int:
    """Output pipeline status as JSON (last run, staleness, pending changes).

    Args:
        layer: An open DataLayer instance.

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 1
    """
    try:
        result = layer.get_pipeline_status()
        print(_json_output(result))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_pipeline_run(layer: DataLayer, trigger: str = "manual") -> int:
    """Execute the SQLMesh pipeline and output the run result as JSON.

    Args:
        layer: An open DataLayer instance.
        trigger: Label indicating what initiated the run.

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 1
    """
    try:
        result = layer.run_pipeline(trigger=trigger)
        print(_json_output(result))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_pipeline_run_result(run_id: str) -> int:
    """Look up a specific PipelineRun by run_id and output as JSON.

    Returns ``{"status": "not_found"}`` when the run_id is unknown.

    Args:
        run_id: The UUID returned by a previous pipeline-run.

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 1
    """
    try:
        from dataclasses import asdict

        from src.pipeline.stats import ProcessingStats

        stats = ProcessingStats()
        for run in stats.get_run_history(limit=200):
            if run.run_id == run_id:
                result = asdict(run)
                result["started_at"] = run.started_at.isoformat()
                result["completed_at"] = run.completed_at.isoformat()
                print(_json_output(result))
                return 0
        print(_json_output({"status": "not_found", "run_id": run_id}))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_pipeline_run_stream(layer: DataLayer, trigger: str = "manual") -> int:
    """Execute the SQLMesh pipeline with streaming progress (JSON lines).

    Each line is a complete JSON object:
      {"type": "started", "step_index": 0, "total_steps": 13, ...}
      {"type": "sqlmesh_running", ...}
      {"type": "model_complete", "model_name": "...", "step_index": 1, ...}
      {"type": "done", "run_id": "...", "duration_seconds": ..., ...}
    or on error:
      {"type": "error", "error": "..."}

    Args:
        layer: An open DataLayer instance.
        trigger: Label indicating what initiated the run.

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 1
    """
    try:

        def _on_progress(event: dict) -> None:
            print(_json_output(event), flush=True)

        layer.run_pipeline_stream(
            trigger=trigger,
            on_progress=_on_progress,
        )
        return 0
    except Exception as exc:
        print(
            _json_output({"type": "error", "error": str(exc)}),
            flush=True,
        )
        return 1


def cmd_get_redaction_detail(payload_hash: str) -> int:
    """Return the stored original/redacted payload for an audit row.

    Args:
        payload_hash: SHA-256 from the audit row's ``payload_hash``.

    Returns:
        0 on success (JSON body printed even when the blob is absent —
        the frontend renders an empty state in that case), 1 on error.

    sensitivity_tier: 3
    """
    try:
        from src.models.redaction_store import default_redaction_store

        detail = default_redaction_store().get(payload_hash)
        print(_json_output({"detail": detail}))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_pipeline_run_history(limit: int = 5) -> int:
    """Return the last N pipeline runs as a JSON array.

    Args:
        limit: Maximum number of runs to return.

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 1
    """
    try:
        from dataclasses import asdict

        from src.pipeline.stats import ProcessingStats

        stats = ProcessingStats()
        runs = stats.get_run_history(limit=limit)
        result = []
        for run in runs:
            d = asdict(run)
            d["started_at"] = run.started_at.isoformat()
            d["completed_at"] = run.completed_at.isoformat()
            result.append(d)
        print(_json_output(result))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_ask(
    layer: DataLayer,
    question: str,
    *,
    session_id: str | None = None,
    reply_context: dict[str, Any] | None = None,
) -> int:
    """Ask the Brain Agent a question and output the response as JSON.

    Args:
        layer: An open DataLayer instance.
        question: The question to ask.
        session_id: If provided, persist both the user question and the
            assistant reply under this chat session.

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 3
    """
    try:
        from src.agents.brain import BrainAgentV2
        from src.agents.tool_registry import ToolRegistry
        from src.core.chat_store import ChatStore
        from src.core.query_engine import QueryEngine
        from src.extensions.connectors.catalog import ConnectorCatalog
        from src.extensions.connectors.registry import ExtensionRegistry
        from src.models.llm_provider import create_provider_from_settings

        qe = QueryEngine(
            duckdb=layer.duckdb,
            kuzu=layer.kuzu,
            chromadb=layer.chromadb,
        )

        store = ChatStore(layer.duckdb) if session_id else None
        if store is not None and session_id:
            store.append_message(session_id, "user", question)

        provider = create_provider_from_settings()
        tool_registry = ToolRegistry(
            catalog=ConnectorCatalog(),
            registry=ExtensionRegistry(),
        )
        v2 = BrainAgentV2(
            query_engine=qe,
            tool_registry=tool_registry,
            provider=provider,
        )
        resp = v2.ask(question, reply_context=reply_context)
        result = {
            "answer": resp.answer,
            "sources": list(resp.sources),
            "context_summary": resp.context_summary,
            "model": resp.model,
            "latency_ms": resp.latency_ms,
        }
        if store is not None and session_id:
            store.append_message(
                session_id,
                "assistant",
                resp.answer,
                sources=list(resp.sources),
                latency_ms=resp.latency_ms,
                model=resp.model,
            )
        print(_json_output(result))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# Chat session commands (persisted via ChatStore)
# ---------------------------------------------------------------------------


def cmd_chat_session_create(
    layer: DataLayer, title: str | None,
) -> int:
    """Create a new chat session row.

    sensitivity_tier: 2
    """
    try:
        from src.core.chat_store import ChatStore

        store = ChatStore(layer.duckdb)
        session_id = store.create_session(title)
        print(_json_output({"session_id": session_id}))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_chat_session_list(layer: DataLayer, limit: int) -> int:
    """List recent chat session summaries.

    sensitivity_tier: 3
    """
    try:
        from src.core.chat_store import ChatStore

        store = ChatStore(layer.duckdb)
        print(_json_output({"sessions": store.list_sessions(limit=limit)}))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_chat_session_load(layer: DataLayer, session_id: str) -> int:
    """Load all messages for a chat session.

    sensitivity_tier: 3
    """
    try:
        from src.core.chat_store import ChatStore

        store = ChatStore(layer.duckdb)
        messages = store.load_session(session_id)
        print(
            _json_output(
                {"session_id": session_id, "messages": messages},
            )
        )
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_chat_session_delete(layer: DataLayer, session_id: str) -> int:
    """Delete a chat session and its messages.

    sensitivity_tier: 1
    """
    try:
        from src.core.chat_store import ChatStore

        store = ChatStore(layer.duckdb)
        store.delete_session(session_id)
        print(_json_output({"ok": True}))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# Action tool commands (JSON output for Tauri bridge)
# ---------------------------------------------------------------------------


def cmd_list_actions() -> int:
    """List available MCP action tools from enabled connectors.

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 1
    """
    try:
        from dataclasses import asdict

        from src.agents.tool_registry import ToolRegistry
        from src.extensions.connectors.catalog import ConnectorCatalog
        from src.extensions.connectors.registry import ExtensionRegistry

        tool_registry = ToolRegistry(
            catalog=ConnectorCatalog(),
            registry=ExtensionRegistry(),
        )
        actions = tool_registry.get_available_actions()
        print(_json_output([asdict(a) for a in actions]))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_confirm_action(
    layer: DataLayer,
    proposal_json: str,
) -> int:
    """Execute a confirmed MCP action, then re-sync the connector.

    After a successful action (e.g. ``create_event``), syncs the
    connector so newly created data appears in queries immediately.

    Args:
        layer: An open DataLayer instance.
        proposal_json: JSON string of the ActionProposal.

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 3
    """
    try:
        from dataclasses import asdict

        from src.agents.action_executor import ActionExecutor

        proposal = json.loads(proposal_json)
        executor = ActionExecutor()
        result = executor.execute(
            connector_id=proposal["connector_id"],
            command=proposal["command"],
            args=tuple(proposal["args"]),
            tool_name=proposal["tool_name"],
            arguments=proposal.get("arguments", {}),
            proposal_id=proposal["proposal_id"],
        )
        result_dict = asdict(result)

        # Re-sync connector after successful action so new data
        # (e.g. a created event) is immediately queryable.
        if result.status == "success":
            try:
                from src.extensions.connectors.connection_manager import (
                    ConnectionManager,
                )

                manager = ConnectionManager(
                    db_engine=layer.duckdb,
                )
                sync_stats = manager.sync_now(
                    proposal["connector_id"],
                )
                result_dict["post_sync"] = {
                    "status": sync_stats.status,
                    "rows_synced": sync_stats.rows_synced,
                }
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Post-action re-sync failed: %s", exc,
                )
                result_dict["post_sync"] = {
                    "status": "error",
                    "error": str(exc),
                }

            # Evaluate whether to send a WhatsApp notification.
            # Non-fatal: never fails the action result.
            try:
                _maybe_notify_action(
                    layer.duckdb, result_dict, proposal,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Action notification evaluation failed: %s", exc,
                )

            # For apple-mail send_email, record the outbound row
            # ourselves so the sweep below can see it immediately.
            # Mail.app's hourly re-ingest will eventually overwrite it.
            if (
                proposal.get("connector_id") == "apple-mail"
                and proposal.get("tool_name") == "send_email"
            ):
                try:
                    _record_outbound_email(
                        layer.duckdb,
                        arguments=proposal.get("arguments", {}),
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Failed to record outbound email: %s", exc,
                    )

            # Eager sweep — dismiss pending replies for which an
            # outbound message now exists. Catches the in-app send
            # path so the Today dashboard reflects reality without
            # waiting for the next proactive cycle.
            try:
                from src.agents.proactive import ProactiveIntelligence

                dismissed = ProactiveIntelligence(
                    db_engine=layer.duckdb,
                ).sweep_resolved_pending_replies()
                if dismissed:
                    logger.info(
                        "Post-send sweep dismissed %d pending reply(ies)",
                        dismissed,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Post-send sweep failed: %s", exc)

        print(_json_output(result_dict))
        return 0 if result.status == "success" else 1
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def _record_outbound_email(
    db: Any,
    *,
    arguments: dict[str, Any],
) -> None:
    """Insert a synthetic ``raw_emails`` row for an in-app sent email.

    The apple-mail bridge sends via AppleScript and returns success but
    does not write back to ``raw_emails`` — Mail.app's next ingestion
    (hourly) is the normal path. This function bridges that gap so the
    pending-reply sweep can detect the send immediately.

    Uses ``INSERT OR IGNORE`` on a synthetic UUID-keyed id, so the
    eventual real ingest with the actual Message-ID does not collide.

    sensitivity_tier: 2
    """
    import uuid as _uuid

    to = str(arguments.get("to") or "").strip()
    if not to:
        return
    subject = str(arguments.get("subject") or "").strip()
    body = str(
        arguments.get("body") or arguments.get("content") or "",
    ).strip()
    recipients = [
        addr.strip() for addr in to.split(",") if addr.strip()
    ]
    if not recipients:
        return

    user_email = _resolve_user_email(db)
    now_iso = datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z",
    )
    synthetic_id = f"apple-sent-{_uuid.uuid4()}"

    db.execute(
        """
        INSERT OR IGNORE INTO raw_emails
            (id, source, message_id, subject, from_address,
             to_addresses, date, body_preview, is_read, folder,
             labels, sensitivity_tier, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            synthetic_id,
            "apple_mail",
            synthetic_id,
            subject,
            user_email or "",
            json.dumps(recipients),
            now_iso,
            body[:200],
            1,
            "Sent",
            json.dumps([]),
            2,
            now_iso,
        ],
    )


_USER_EMAIL_CACHE: str | None = None
_USER_EMAIL_RESOLVED: bool = False


def _resolve_user_email(db: Any) -> str | None:
    """Discover the user's own email address from past Sent rows.

    Picks the most frequent ``from_address`` among ingested rows whose
    folder looks like Sent. Cached per process — fresh installs may
    have no Sent rows yet, in which case we return ``None`` and the
    synthetic outbound is still written (without an authoritative
    ``from_address``).

    sensitivity_tier: 1
    """
    global _USER_EMAIL_CACHE, _USER_EMAIL_RESOLVED
    if _USER_EMAIL_RESOLVED:
        return _USER_EMAIL_CACHE
    try:
        rows = db.query(
            """
            SELECT from_address, COUNT(*) AS c
            FROM raw_emails
            WHERE from_address IS NOT NULL
              AND LOWER(COALESCE(folder, '')) LIKE '%sent%'
            GROUP BY from_address
            ORDER BY c DESC
            LIMIT 1
            """,
        )
    except Exception:  # noqa: BLE001
        rows = []
    if rows:
        _USER_EMAIL_CACHE = str(rows[0].get("from_address") or "") or None
    _USER_EMAIL_RESOLVED = True
    return _USER_EMAIL_CACHE


def cmd_resume_action_with_recipient(
    layer: DataLayer,
    disambiguation_json: str,
    candidate_json: str,
) -> int:
    """Build a confirmed ActionProposal from a disambiguation choice.

    Takes the serialised RecipientDisambiguationProposal and the
    candidate dict the user picked; merges the candidate's handle
    into the draft arguments and emits a normal ActionProposal that
    the frontend renders as the standard confirmation card.

    sensitivity_tier: 3
    """
    try:
        from dataclasses import asdict

        from src.agents.brain.actions import (
            resume_action_from_disambiguation,
        )

        disambiguation = json.loads(disambiguation_json)
        candidate = json.loads(candidate_json)
        proposal = resume_action_from_disambiguation(
            disambiguation=disambiguation,
            candidate=candidate,
            duckdb=layer.duckdb,
        )
        print(_json_output({
            "type": "action_proposal",
            "proposal": asdict(proposal),
        }))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_search_recipient_candidates(
    layer: DataLayer,
    query: str,
    channel: str,
    include_apple: bool,
    limit: int,
) -> int:
    """Search the user's contacts for a Send Message recipient.

    Powers the inline search box on the recipient disambiguation
    card. ``include_apple=False`` searches only the local mart +
    raw_contacts (fast — meant for every keystroke). ``include_apple=
    True`` additionally spawns the apple-contacts MCP subprocess
    (slower — gated behind an explicit button in the UI).

    sensitivity_tier: 3
    """
    try:
        from dataclasses import asdict

        from src.agents.brain.recipient_resolver import resolve_recipient

        tool_registry: Any = None
        mcp_client_factory: Any = None
        if include_apple:
            from src.agents.tool_registry import ToolRegistry
            from src.extensions.connectors.catalog import ConnectorCatalog
            from src.extensions.connectors.registry import ExtensionRegistry
            from src.extensions.mcp.client import McpClient

            tool_registry = ToolRegistry(
                catalog=ConnectorCatalog(),
                registry=ExtensionRegistry(),
            )
            mcp_client_factory = McpClient

        resolution = resolve_recipient(
            query,
            channel,
            layer.duckdb,
            tool_registry=tool_registry,
            mcp_client_factory=mcp_client_factory,
            limit=limit,
        )
        print(_json_output({
            "candidates": [asdict(c) for c in resolution.candidates],
        }))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_cancel_action(proposal_id: str) -> int:
    """Record cancellation of a proposed action.

    Args:
        proposal_id: The UUID of the cancelled proposal.

    Returns:
        Exit code (0 = success).

    sensitivity_tier: 1
    """
    print(_json_output({
        "status": "cancelled",
        "proposal_id": proposal_id,
    }))
    return 0


def _maybe_notify_action(
    db: Any,
    result_dict: dict[str, Any],
    proposal: dict[str, Any],
) -> None:
    """Evaluate and notify about an action result.

    Delivers over every configured channel — native macOS
    notifications by default, WhatsApp additionally when a phone
    number is configured. Non-fatal: callers wrap this in try/except
    so notification failures never degrade the action response.

    sensitivity_tier: 2
    """
    from src.notifications.preference_service import PreferenceService

    prefs = PreferenceService(db_engine=db)

    if prefs.is_muted_globally():
        return

    # WhatsApp phone from settings.json — optional; native delivery
    # doesn't need it.
    phone = _read_whatsapp_phone()

    from src.models.llm_provider import create_provider_from_settings
    from src.notifications.models import (
        DeliveryResult,
        NotificationRecord,
    )
    from src.notifications.notifier import deliver_notification, get_opt_out_text
    from src.notifications.orchestrator import (
        BrainNotificationOrchestrator,
    )

    try:
        notif_llm = create_provider_from_settings(background=True)
    except Exception:  # noqa: BLE001
        notif_llm = None

    orchestrator = BrainNotificationOrchestrator(
        preference_service=prefs,
        db_engine=db,
        llm_provider=notif_llm,
    )

    decision = orchestrator.evaluate_action_result(
        action_result=result_dict,
        proposal=proposal,
    )

    delivery: DeliveryResult
    if decision.should_notify:
        delivery = deliver_notification(
            decision.message, decision.category, whatsapp_phone=phone,
        )
    else:
        delivery = DeliveryResult(
            status="skipped",
            timestamp=utc_now_iso(),
        )

    prefs.log_notification(
        NotificationRecord(
            id=prefs.new_record_id(),
            dedupe_key=decision.dedupe_key,
            category=decision.category,
            importance_score=decision.importance_score,
            decision="send" if decision.should_notify else "skip",
            delivery_status=delivery.status,
            message=decision.message,
            opt_out_text=get_opt_out_text(decision.category),
            source_type="action",
            source_id=proposal.get("proposal_id", "unknown"),
            error=delivery.error,
            created_at=utc_now_iso(),
            message_id=delivery.message_id,
        ),
    )


def _maybe_notify_insights(
    db: Any,
    insights: list[dict[str, Any]],
) -> None:
    """Evaluate and notify about newly generated insights.

    Delivers over every configured channel — native macOS
    notifications by default, WhatsApp additionally when a phone
    number is configured. Non-fatal: callers wrap this in try/except
    so notification failures never degrade the insight generation
    response.

    sensitivity_tier: 2
    """
    if not insights:
        return

    from src.notifications.preference_service import PreferenceService

    prefs = PreferenceService(db_engine=db)

    if prefs.is_muted_globally():
        return

    # WhatsApp phone from settings.json — optional; native delivery
    # doesn't need it.
    phone = _read_whatsapp_phone()

    from src.models.llm_provider import create_provider_from_settings
    from src.notifications.models import (
        DeliveryResult,
        NotificationRecord,
    )
    from src.notifications.notifier import deliver_notification, get_opt_out_text
    from src.notifications.orchestrator import (
        BrainNotificationOrchestrator,
    )

    try:
        notif_llm = create_provider_from_settings(background=True)
    except Exception:  # noqa: BLE001
        notif_llm = None

    orchestrator = BrainNotificationOrchestrator(
        preference_service=prefs,
        db_engine=db,
        llm_provider=notif_llm,
    )

    decision = orchestrator.evaluate_insight_result(insights)

    delivery: DeliveryResult
    if decision.should_notify:
        delivery = deliver_notification(
            decision.message, decision.category, whatsapp_phone=phone,
        )
    else:
        delivery = DeliveryResult(
            status="skipped",
            timestamp=utc_now_iso(),
        )

    source_id = insights[0].get("id", "unknown")
    prefs.log_notification(
        NotificationRecord(
            id=prefs.new_record_id(),
            dedupe_key=decision.dedupe_key,
            category=decision.category,
            importance_score=decision.importance_score,
            decision="send" if decision.should_notify else "skip",
            delivery_status=delivery.status,
            message=decision.message,
            opt_out_text=get_opt_out_text(decision.category),
            source_type="insight",
            source_id=source_id,
            error=delivery.error,
            created_at=utc_now_iso(),
            message_id=delivery.message_id,
        ),
    )


def _read_whatsapp_phone() -> str | None:
    """Read WhatsApp phone from settings.json.

    sensitivity_tier: 1
    """
    try:
        settings_file = Path.home() / ".arandu" / "settings.json"
        if settings_file.exists():
            data = json.loads(
                settings_file.read_text(encoding="utf-8"),
            )
            if data.get("notifications_enabled"):
                return (
                    data.get("whatsapp_notification_phone")
                    or None
                )
    except Exception:  # noqa: BLE001
        pass
    return None


# ---------------------------------------------------------------------------
# Connector commands (JSON output for Tauri bridge)
# ---------------------------------------------------------------------------


def cmd_connector_catalog() -> int:
    """Return all connectors with their current status.

    sensitivity_tier: 1
    """
    try:
        from src.extensions.connectors.connection_manager import ConnectionManager

        manager = ConnectionManager()
        result = manager.get_connector_catalog()
        print(_json_output(result))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def _build_system_health(layer: DataLayer) -> dict[str, Any]:
    """Aggregate connector, pipeline, graph, and vector state into one
    health payload: an ``overall`` verdict, the data-flow ``stages``
    (connectors -> ingest -> transform -> graph -> vectors), and a flat,
    severity-sorted list of actionable ``issues``.

    This is the single source of truth the unified status surface reads,
    so the frontend stops stitching four separate calls together. Each
    section is best-effort; a failure in one never blanks the rest.

    sensitivity_tier: 1
    """
    stages: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    # --- Connectors -------------------------------------------------
    try:
        from src.extensions.connectors.connection_manager import (
            ConnectionManager,
        )

        catalog = ConnectionManager().get_connector_catalog()
    except Exception:  # noqa: BLE001
        catalog = []
    enabled = [c for c in catalog if c.get("enabled")]
    errored = [c for c in enabled if c.get("status") == "error"]
    connected = [c for c in enabled if c.get("status") == "connected"]
    empty = [
        c
        for c in connected
        if (c.get("stats") or {}).get("records_synced", 0) == 0
        and not (c.get("stats") or {}).get("last_success")
    ]
    last_syncs = [
        (c.get("stats") or {}).get("last_sync")
        for c in enabled
        if (c.get("stats") or {}).get("last_sync")
    ]
    conn_status = (
        "error" if errored
        else "warning" if empty
        else "ok" if connected
        else "idle"
    )
    stages.append({
        "id": "connectors",
        "label": "Connectors",
        "status": conn_status,
        "summary": (
            f"{len(connected)}/{len(enabled)} connected"
            if enabled else "none enabled"
        ),
        "last_run_at": max(last_syncs) if last_syncs else None,
        "route": "/connectors",
    })
    for c in errored:
        issues.append({
            "id": f"connector:{c['connector_id']}",
            "stage": "connectors",
            "severity": "error",
            "title": f"{c.get('name')} sync failed",
            "detail": (c.get("stats") or {}).get("error") or "Connector error",
            "action": {
                "label": "Retry",
                "kind": "retry_connector",
                "target": c["connector_id"],
            },
        })
    for c in empty:
        issues.append({
            "id": f"connector-empty:{c['connector_id']}",
            "stage": "connectors",
            "severity": "warning",
            "title": f"{c.get('name')} has no data yet",
            "detail": "Connected, but nothing has been ingested.",
            "action": {
                "label": "Sync now",
                "kind": "retry_connector",
                "target": c["connector_id"],
            },
        })

    # --- Pipeline status (drives ingest / transform / graph / vectors)
    try:
        ps = layer.get_pipeline_status()
    except Exception:  # noqa: BLE001
        ps = {}
    last = ps.get("last_run") or {}
    is_stale = bool(ps.get("is_stale"))

    # Ingest — total raw rows currently landed.
    raw_total = 0
    raw_sources = 0
    try:
        rows = layer.duckdb.query(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name LIKE 'raw_%'",
        )
        for r in rows:
            try:
                cnt = layer.duckdb.query(
                    f"SELECT COUNT(*) AS n FROM {r['name']}",
                )
                n = cnt[0]["n"] if cnt else 0
            except Exception:  # noqa: BLE001
                n = 0
            raw_total += n
            if n > 0:
                raw_sources += 1
    except Exception:  # noqa: BLE001
        pass
    stages.append({
        "id": "ingest",
        "label": "Ingest",
        "status": "ok" if raw_total else "idle",
        "summary": (
            f"{raw_total:,} rows · {raw_sources} sources"
            if raw_total else "no data"
        ),
        "last_run_at": None,
        "route": "/data?tab=sources",
    })

    # Transform — last pipeline run over the staging/mart models.
    run_status = last.get("status")
    transform_status = (
        "error" if run_status == "failed"
        else "warning" if is_stale
        else "ok" if last
        else "idle"
    )
    stages.append({
        "id": "transform",
        "label": "Transform",
        "status": transform_status,
        "summary": (
            "Failed" if run_status == "failed"
            else "Stale" if is_stale
            else "Up to date" if last
            else "never run"
        ),
        "last_run_at": last.get("completed_at"),
        "route": "/data?tab=models",
    })
    if run_status == "failed":
        issues.append({
            "id": "transform:failed",
            "stage": "transform",
            "severity": "error",
            "title": "Last pipeline run failed",
            "detail": last.get("error") or "Pipeline run failed",
            "action": {
                "label": "Run now",
                "kind": "run_pipeline",
                "target": None,
            },
        })

    # Graph — Kuzu node count + reindex outcome.
    graph_nodes = 0
    try:
        for nt in ALL_NODE_TABLES:
            try:
                gr = layer.kuzu.query(
                    f"MATCH (n:{nt}) RETURN count(n) AS c",
                )
                graph_nodes += gr[0]["c"] if gr else 0
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass
    graph_idx = last.get("graph_index_status")
    stages.append({
        "id": "graph",
        "label": "Graph",
        "status": (
            "error" if graph_idx == "error"
            else "ok" if graph_nodes else "idle"
        ),
        "summary": f"{graph_nodes:,} nodes" if graph_nodes else "empty",
        "last_run_at": last.get("completed_at"),
        "route": "/data?tab=graph",
    })
    if graph_idx == "error":
        issues.append({
            "id": "graph:index",
            "stage": "graph",
            "severity": "error",
            "title": "Knowledge graph index failed",
            "detail": last.get("index_error") or "Graph re-index failed",
            "action": {
                "label": "View",
                "kind": "open_route",
                "target": "/data?tab=graph",
            },
        })

    # Vectors — ChromaDB doc count + reindex outcome.
    doc_total = 0
    try:
        for name in COLLECTION_NAMES:
            try:
                doc_total += layer.chromadb.get_or_create_collection(
                    name,
                ).count()
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass
    vec_idx = last.get("vector_index_status")
    stages.append({
        "id": "vectors",
        "label": "Vectors",
        "status": (
            "error" if vec_idx == "error"
            else "ok" if doc_total else "idle"
        ),
        "summary": f"{doc_total:,} documents" if doc_total else "empty",
        "last_run_at": last.get("completed_at"),
        "route": "/data?tab=vectors",
    })
    if vec_idx == "error":
        issues.append({
            "id": "vectors:index",
            "stage": "vectors",
            "severity": "error",
            "title": "Vector index failed",
            "detail": last.get("index_error") or "Vector re-index failed",
            "action": {
                "label": "View",
                "kind": "open_route",
                "target": "/data?tab=vectors",
            },
        })

    has_error = any(i["severity"] == "error" for i in issues)
    has_warning = (
        any(i["severity"] == "warning" for i in issues) or is_stale
    )
    overall = (
        "failing" if has_error
        else "degraded" if has_warning
        else "healthy"
    )
    issues.sort(key=lambda i: 0 if i["severity"] == "error" else 1)

    return {"overall": overall, "stages": stages, "issues": issues}


def cmd_system_health(layer: DataLayer) -> int:
    """Output the aggregated system-health payload as JSON.

    sensitivity_tier: 1
    """
    try:
        print(_json_output(_build_system_health(layer)))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_toggle_connector(
    layer: DataLayer,
    connector_id: str,
    enabled: bool,
    user_inputs_json: str | None = None,
) -> int:
    """Toggle a connector on or off.

    sensitivity_tier: 1
    """
    try:
        from src.extensions.connectors.connection_manager import ConnectionManager

        user_inputs: dict[str, Any] = {}
        if user_inputs_json:
            parsed = json.loads(user_inputs_json)
            # Backward compatibility: older UI payloads could be double-encoded.
            if isinstance(parsed, str):
                parsed = json.loads(parsed)
            if not isinstance(parsed, dict):
                raise ValueError("user_inputs must be a JSON object")
            user_inputs = parsed

        manager = ConnectionManager(db_engine=layer.duckdb)
        result = manager.toggle_connector(
            connector_id, enabled, user_inputs,
        )

        from dataclasses import asdict

        payload = asdict(result)
        should_refresh_after_enable = (
            enabled
            and payload.get("status") == "connected"
            and int(payload.get("records_synced", 0)) > 0
        )
        if should_refresh_after_enable:
            run = _run_smart_pipeline_and_reindex(
                layer, trigger="connector_enable",
            )
            payload["pipeline"] = {
                "triggered": True,
                "status": run.get("status"),
                "plan_summary": run.get("plan_summary"),
            }
        else:
            payload["pipeline"] = {"triggered": False}

        print(_json_output(payload))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_sync_connector(
    layer: DataLayer, connector_id: str,
) -> int:
    """Trigger an immediate sync for a connector.

    sensitivity_tier: 1
    """
    try:
        from src.extensions.connectors.connection_manager import ConnectionManager

        manager = ConnectionManager(db_engine=layer.duckdb)
        stats = manager.sync_now(connector_id)

        print(
            _json_output(
                {
                    "connector_id": stats.connector_id,
                    "status": stats.status,
                    "rows_synced": stats.rows_synced,
                    "duration_seconds": stats.duration_seconds,
                    "error": stats.error,
                }
            )
        )
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_ensure_whatsapp_listener() -> int:
    """Ensure the WhatsApp listener is running if the connector is enabled.

    Intended for app startup: stops any stale listener from a previous
    session, then starts a fresh one.  If WhatsApp is not enabled, this
    is a no-op.

    sensitivity_tier: 1
    """
    try:
        from src.extensions.bridges.whatsapp.listener import (
            WhatsAppListenerService,
        )
        from src.extensions.connectors.catalog import ConnectorCatalog
        from src.extensions.connectors.registry import ExtensionRegistry

        registry = ExtensionRegistry()
        enabled_ids = {e.connector_id for e in registry.get_enabled()}
        if "whatsapp" not in enabled_ids:
            print(_json_output({"status": "skipped", "reason": "not_enabled"}))
            return 0

        catalog = ConnectorCatalog()
        template = catalog.get("whatsapp")
        if template is None:
            print(_json_output({"status": "skipped", "reason": "no_template"}))
            return 0

        service = WhatsAppListenerService()

        # Stop stale listener from previous app session (pid may be dead).
        current = service.status()
        if current.get("pid"):
            logger.info(
                "Stopping previous WhatsApp listener (pid=%s)",
                current["pid"],
            )
            service.stop()

        # Start a fresh listener.
        status = service.start(template.command, template.args)
        print(_json_output(status))
        return 0
    except Exception as exc:
        logger.warning("ensure-whatsapp-listener failed: %s", exc)
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_whatsapp_listener_spec() -> int:
    """Emit the WhatsApp listener spec for the Rust supervisor.

    Returns enabled state plus the runtime paths and MCP command the
    supervisor needs to spawn the listener subprocess. Kept cheap so
    the Rust loop can call it without significant overhead.

    sensitivity_tier: 1
    """
    try:
        from src.extensions.bridges.whatsapp.listener import (
            _LOG_PATH,
            _PID_PATH,
            _RUNTIME_DIR,
        )
        from src.extensions.connectors.catalog import ConnectorCatalog
        from src.extensions.connectors.registry import ExtensionRegistry

        registry = ExtensionRegistry()
        enabled_ids = {e.connector_id for e in registry.get_enabled()}
        enabled = "whatsapp" in enabled_ids

        command: str | None = None
        args: list[str] = []
        if enabled:
            template = ConnectorCatalog().get("whatsapp")
            if template is None:
                enabled = False
            else:
                command = template.command
                args = list(template.args)

        payload = {
            "enabled": enabled,
            "command": command,
            "args": args,
            "runtime_dir": str(_RUNTIME_DIR),
            "pid_path": str(_PID_PATH),
            "log_path": str(_LOG_PATH),
        }
        print(_json_output(payload))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_whatsapp_listener_start() -> int:
    """Start the persistent WhatsApp listener process.

    sensitivity_tier: 1
    """
    try:
        from src.extensions.bridges.whatsapp.listener import (
            WhatsAppListenerService,
        )
        from src.extensions.connectors.catalog import ConnectorCatalog

        catalog = ConnectorCatalog()
        template = catalog.get("whatsapp")
        if template is None:
            print(
                _json_output(
                    {"error": "WhatsApp connector template not found"},
                ),
                file=sys.stderr,
            )
            return 1

        status = WhatsAppListenerService().ensure_running(
            template.command,
            template.args,
        )
        print(_json_output(status))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_whatsapp_listener_stop() -> int:
    """Stop the persistent WhatsApp listener process.

    sensitivity_tier: 1
    """
    try:
        from src.extensions.bridges.whatsapp.listener import (
            WhatsAppListenerService,
        )

        status = WhatsAppListenerService().stop()
        print(_json_output(status))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_whatsapp_listener_status() -> int:
    """Return persistent WhatsApp listener status.

    sensitivity_tier: 1
    """
    try:
        from src.extensions.bridges.whatsapp.listener import (
            WhatsAppListenerService,
        )

        status = WhatsAppListenerService().status()
        print(_json_output(status))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_whatsapp_listener_run(
    command: str,
    mcp_args: list[str],
    mcp_timeout_seconds: float = 45.0,
    scan_interval_seconds: float = 2.0,
    reconnect_backoff_seconds: float = 5.0,
) -> int:
    """Run the foreground WhatsApp listener loop.

    sensitivity_tier: 2
    """
    from src.extensions.bridges.whatsapp.listener import run_whatsapp_listener

    return run_whatsapp_listener(
        command=command,
        args=tuple(mcp_args),
        mcp_timeout_seconds=mcp_timeout_seconds,
        scan_interval_seconds=scan_interval_seconds,
        reconnect_backoff_seconds=reconnect_backoff_seconds,
    )


def cmd_connector_details(connector_id: str) -> int:
    """Return full details for a single connector.

    sensitivity_tier: 1
    """
    try:
        from src.extensions.connectors.connection_manager import ConnectionManager

        manager = ConnectionManager()
        details = manager.get_connector_details(connector_id)

        if details is None:
            print(
                _json_output(
                    {"error": f"Unknown connector: {connector_id}"}
                ),
                file=sys.stderr,
            )
            return 1

        print(_json_output(details))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# Extension installer commands (JSON output for Tauri bridge)
# ---------------------------------------------------------------------------


def _parse_env_json(raw: str | None) -> dict[str, str] | None:
    """Parse a JSON env-vars argument into a flat str→str dict.

    Returns None for empty/missing input. Raises ValueError if the JSON
    is malformed or values aren't strings.

    sensitivity_tier: 1
    """
    if not raw:
        return None
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        msg = "--env must be a JSON object"
        raise ValueError(msg)
    result: dict[str, str] = {}
    for k, v in parsed.items():
        if v is None or v == "":
            continue
        result[str(k)] = str(v)
    return result or None


def cmd_discover_extension(
    command: str,
    ext_args: list[str],
    name: str | None = None,
    env: dict[str, str] | None = None,
) -> int:
    """Discover tools and schema from an MCP server command.

    Args:
        command: The MCP server command to run.
        ext_args: Arguments for the command.
        name: Optional human-friendly name override.
        env: Extra environment variables to pass to the server
            (e.g. API tokens). Tier 3 secrets — not logged.

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 1
    """
    try:
        from dataclasses import asdict

        from src.extensions.mcp.installer import ExtensionInstaller

        installer = ExtensionInstaller(mcp_timeout=60.0)
        preview = installer.discover(
            command, tuple(ext_args), name=name, env=env,
        )

        result = asdict(preview)
        # Convert tuples to lists for JSON
        result["args"] = list(result["args"])
        result["new_tables"] = list(result["new_tables"])
        result["existing_tables"] = list(result["existing_tables"])
        result["warnings"] = list(result["warnings"])
        for t in result["tools"]:
            t["warnings"] = list(t["warnings"])

        print(_json_output(result))
        return 0
    except Exception as exc:
        # Write error to stdout so the Rust bridge can parse it as JSON.
        # stderr is used for logging and gets mixed with INFO lines.
        print(_json_output({"error": str(exc)}))
        return 1


def cmd_confirm_extension(
    layer: DataLayer,
    preview_json: str,
    name: str | None = None,
    env: dict[str, str] | None = None,
) -> int:
    """Confirm and finalize an extension install.

    Args:
        layer: An open DataLayer instance.
        preview_json: JSON string of the InstallPreview.
        name: Optional name override.
        env: Extra environment variables persisted to the connector
            registry so future syncs can relaunch the MCP server.
            Tier 3 secrets.

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 1
    """
    try:
        from dataclasses import asdict

        from src.extensions.mcp.installer import (
            ExtensionInstaller,
            InstallPreview,
            ToolPreview,
        )

        data = json.loads(preview_json)

        # Reconstruct InstallPreview from JSON
        tools = tuple(
            ToolPreview(
                tool_name=t["tool_name"],
                tool_type=t["tool_type"],
                target_table=t.get("target_table"),
                is_new_table=t.get("is_new_table", False),
                field_count=t.get("field_count", 0),
                sensitivity_tiers=t.get("sensitivity_tiers", {}),
                confidence=t.get("confidence", 0.0),
                warnings=tuple(t.get("warnings", ())),
            )
            for t in data.get("tools", [])
        )

        preview = InstallPreview(
            server_name=data["server_name"],
            command=data["command"],
            args=tuple(data.get("args", ())),
            tools=tools,
            data_tools=data.get("data_tools", 0),
            action_tools=data.get("action_tools", 0),
            new_tables=tuple(data.get("new_tables", ())),
            existing_tables=tuple(data.get("existing_tables", ())),
            overall_confidence=data.get("overall_confidence", 0.0),
            warnings=tuple(data.get("warnings", ())),
        )

        installer = ExtensionInstaller(db_engine=layer.duckdb)
        result = installer.confirm(preview, name=name, env=env)

        output = asdict(result)
        output["tables_created"] = list(output["tables_created"])
        print(_json_output(output))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}))
        return 1


# ---------------------------------------------------------------------------
# Model generator commands (JSON output for Tauri bridge)
# ---------------------------------------------------------------------------


def cmd_generate_models(
    connector_id: str,
    mapping_json: str,
) -> int:
    """Generate pipeline models for a new data source.

    Args:
        connector_id: The extension connector ID.
        mapping_json: JSON string of the DiscoveredMapping.

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 1
    """
    try:
        from src.extensions.ingestion.model_generator import ModelGenerator
        from src.extensions.ingestion.review_flow import (
            ReviewFlow,
            _serialize_preview,
        )
        from src.extensions.ingestion.schema_discovery import (
            DiscoveredMapping,
            FieldMapping,
        )

        data = json.loads(mapping_json)

        # Reconstruct DiscoveredMapping from JSON
        fields = tuple(
            FieldMapping(
                source_name=f["source_name"],
                target_column=f["target_column"],
                source_type=f.get("source_type", "string"),
                target_type=f.get("target_type", "VARCHAR"),
                sensitivity_tier=f.get("sensitivity_tier", 2),
                confidence=f.get("confidence", 0.5),
                tier_source=f.get("tier_source", "default"),
                transform=f.get("transform"),
                is_new_column=f.get("is_new_column", False),
            )
            for f in data.get("fields", [])
        )

        mapping = DiscoveredMapping(
            tool_name=data["tool_name"],
            target_table=data["target_table"],
            is_new_table=data.get("is_new_table", True),
            domain=data.get("domain", "general"),
            confidence=data.get("confidence", 0.5),
            analysis_method=data.get("analysis_method", "rules_only"),
            fields=fields,
            dedup_key=tuple(data.get("dedup_key", ())),
            suggested_schedule=data.get("suggested_schedule", "daily"),
            warnings=tuple(data.get("warnings", ())),
        )

        generator = ModelGenerator()
        preview = generator.generate(mapping, connector_id)

        review = ReviewFlow()
        review.stage(preview)

        print(_json_output(_serialize_preview(preview)))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_approve_models(
    layer: DataLayer,
    connector_id: str,
) -> int:
    """Approve staged models and install into the pipeline.

    Args:
        layer: An open DataLayer instance.
        connector_id: The connector ID to approve.

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 1
    """
    try:
        from dataclasses import asdict

        from src.extensions.ingestion.review_flow import ReviewFlow

        review = ReviewFlow(db_engine=layer.duckdb)
        result = review.approve(connector_id)

        output = asdict(result)
        output["files_created"] = list(output["files_created"])
        output["pipeline_models_added"] = list(output["pipeline_models_added"])
        print(_json_output(output))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_reject_models(connector_id: str) -> int:
    """Reject staged models without installing.

    Args:
        connector_id: The connector ID to reject.

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 1
    """
    try:
        from src.extensions.ingestion.review_flow import ReviewFlow

        review = ReviewFlow()
        review.reject(connector_id)
        print(_json_output({"status": "rejected", "connector_id": connector_id}))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# Agent runner commands
# ---------------------------------------------------------------------------


def cmd_list_agents() -> int:
    """List all discovered agents with status.

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 1
    """
    try:
        from src.agent_runtime.runner import AgentRunner

        runner = AgentRunner()
        statuses = runner.list_agents()
        print(_json_output([
            {
                "agent_id": s.agent_id,
                "name": s.name,
                "description": s.description,
                "category": s.category,
                "status": s.status,
                "builtin": s.builtin,
                "triggers": list(s.triggers),
                "max_sensitivity_tier": s.max_sensitivity_tier,
                "last_run_at": s.last_run_at,
                "last_result": s.last_result,
                "error": s.error,
            }
            for s in statuses
        ]))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_run_agent(
    layer: DataLayer,
    agent_id: str,
    params: str | None = None,
) -> int:
    """Run an agent by ID and return the result.

    Args:
        layer: An open DataLayer instance.
        agent_id: Agent identifier.
        params: Optional JSON-encoded parameters.

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 2
    """
    try:
        from src.agent_runtime.runner import AgentRunner

        runner = AgentRunner(db_engine=layer.duckdb)
        parsed_params = json.loads(params) if params else None
        result = runner.run_agent(agent_id, params=parsed_params)
        print(_json_output({
            "agent_id": result.agent_id,
            "status": result.status,
            "output": result.output,
            "tables_written": list(result.tables_written),
            "rows_written": result.rows_written,
            "llm_calls": result.llm_calls,
            "duration_ms": result.duration_ms,
            "error": result.error,
        }))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_get_agent_result(agent_id: str) -> int:
    """Get the last result from an agent run.

    Args:
        agent_id: Agent identifier.

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 1
    """
    try:
        from src.agent_runtime.runner import AgentRunner

        runner = AgentRunner()
        result = runner.get_agent_result(agent_id)
        if result is None:
            print(_json_output({
                "agent_id": agent_id,
                "status": "no_result",
                "output": "",
                "tables_written": [],
                "rows_written": 0,
                "llm_calls": 0,
                "duration_ms": 0.0,
                "error": None,
            }))
        else:
            print(_json_output({
                "agent_id": result.agent_id,
                "status": result.status,
                "output": result.output,
                "tables_written": list(result.tables_written),
                "rows_written": result.rows_written,
                "llm_calls": result.llm_calls,
                "duration_ms": result.duration_ms,
                "error": result.error,
            }))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_run_scheduled_agents(layer: DataLayer) -> int:
    """Run agents whose cron schedule is due.

    Reads schedule state from disk, checks each scheduled agent's
    cron expression, runs due agents, and persists updated state.

    Args:
        layer: An open DataLayer instance (read-write).

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 1
    """
    from datetime import datetime, timezone
    from pathlib import Path

    from src.agent_runtime.models import TriggerMode
    from src.agent_runtime.runner import AgentRunner
    from src.extensions.cron import cron_is_due

    state_path = Path.home() / ".arandu" / "data" / "agent_schedule_state.json"
    now = datetime.now(timezone.utc)

    # Load persisted state.
    state: dict[str, str] = {}
    if state_path.exists():
        try:
            state = json.loads(
                state_path.read_text(encoding="utf-8"),
            )
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupt schedule state, resetting")
            state = {}

    runner = AgentRunner(db_engine=layer.duckdb)
    manifests = runner.discover_agents()

    agents_run: list[dict[str, str]] = []
    errors: list[str] = []
    checked = 0

    for manifest in manifests:
        if TriggerMode.SCHEDULED not in manifest.triggers:
            continue
        if not manifest.schedule:
            continue

        checked += 1
        last_run_str = state.get(manifest.id)
        last_run = (
            datetime.fromisoformat(last_run_str)
            if last_run_str
            else None
        )

        if not cron_is_due(manifest.schedule, last_run, now):
            continue

        logger.info(
            "Agent '%s' is due (schedule=%s)",
            manifest.id,
            manifest.schedule,
        )
        try:
            result = runner.run_agent(
                manifest.id,
                trigger=TriggerMode.SCHEDULED,
            )
            agents_run.append({
                "agent_id": manifest.id,
                "status": result.status,
            })
            if result.status in ("success", "timeout"):
                state[manifest.id] = now.isoformat()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Scheduled agent '%s' failed: %s",
                manifest.id,
                exc,
            )
            errors.append(f"{manifest.id}: {exc}")

    # ----- User-authored agents -----
    # The same tick also walks ``user_agents`` rows whose
    # ``schedule_enabled=1`` and fires the ones whose cron is due.
    # Without this, ``schedule_cron`` is collected by the UI + stored
    # in SQLite but never honored — silent no-op for every "every
    # hour at xx:00" the user sets. Kept inside the same handler so
    # the persisted ``state`` file remains the single source of truth
    # for last-fire timestamps across both populations.
    try:
        ua_checked, ua_run, ua_errors = _tick_scheduled_user_agents(
            layer=layer, state=state, now=now,
        )
        checked += ua_checked
        agents_run.extend(ua_run)
        errors.extend(ua_errors)
    except Exception as exc:  # noqa: BLE001
        logger.warning("User-agent scheduler tick failed: %s", exc)
        errors.append(f"user-agents-tick: {exc}")

    # ----- Daily plan tick -----
    # Fire the daily scheduler at 06:00 local time once per day.
    # Same persisted-state file so the next tick after 06:00 sees it
    # has already run today and skips.
    try:
        if _tick_daily_plan(layer=layer, state=state, now=now):
            agents_run.append({
                "agent_id": "daily_scheduler",
                "status": "success",
            })
        checked += 1
    except Exception as exc:  # noqa: BLE001
        logger.warning("Daily plan tick failed: %s", exc)
        errors.append(f"daily-plan-tick: {exc}")

    # Persist updated state.
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps(state, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("Failed to persist schedule state: %s", exc)

    print(_json_output({
        "agents_checked": checked,
        "agents_run": agents_run,
        "errors": errors,
    }))
    return 0


def _tick_daily_plan(
    *, layer: DataLayer, state: dict[str, str], now: Any,
) -> bool:
    """Fire the daily scheduler once per day after 06:00 local.

    Reuses the persisted ``state`` file under the synthetic id
    ``daily_scheduler`` so the same once-per-day guard applies. The
    actual scheduler call goes through ``TaskCurator.regenerate_daily_
    schedule``, which the curator caches in ``_schedule_suggestions``.

    sensitivity_tier: 2
    """
    key = "daily_scheduler"
    last_run_str = state.get(key)
    if last_run_str:
        try:
            last_run = datetime.fromisoformat(last_run_str)
        except ValueError:
            last_run = None
    else:
        last_run = None

    if now.hour < 6:
        return False
    if last_run is not None and last_run.date() == now.date():
        return False

    from src.agents.tasks import TaskCurator

    curator = TaskCurator(db_engine=layer.duckdb)
    record = curator.regenerate_daily_schedule()
    if record is not None:
        state[key] = now.isoformat()
        # Also refresh habits on the daily tick so atomic habits track
        # the latest goal set.
        try:
            curator.regenerate_habits()
        except Exception:  # noqa: BLE001
            logger.warning(
                "Daily habit refresh failed", exc_info=True,
            )
        return True
    return False


def _tick_scheduled_user_agents(
    *,
    layer: DataLayer,
    state: dict[str, str],
    now: Any,
) -> tuple[int, list[dict[str, str]], list[str]]:
    """Fire user-authored agents whose cron is due.

    Delegates the actual invocation to
    :mod:`src.agents.user_agents.runner` — rows with at least one
    catalog ``data`` tool in ``enabled_mcp_tools`` take the batch
    path (one LLM call per unprocessed item, followed by the
    post-batch delivery hook when ``delivery_tools`` is non-empty);
    other rows fall back to the generic Portuguese trigger that has
    been the behavior since the original schedule wiring landed.

    Returns ``(checked, run_list, error_list)``.

    sensitivity_tier: 1
    """
    from src.agents.user_agents.runner import (
        data_tool_ids_for_row,
        run_user_agent_batch,
        run_user_agent_generic,
    )
    from src.agents.user_agents.store import UserAgentStore
    from src.extensions.cron import cron_is_due

    checked = 0
    run_list: list[dict[str, str]] = []
    error_list: list[str] = []

    store = UserAgentStore()
    try:
        rows = store.list_all()
    finally:
        store.close()

    for row in rows:
        if not row.schedule_enabled or not row.schedule_cron:
            continue
        checked += 1
        last_run_str = state.get(row.agent_id)
        last_run = (
            datetime.fromisoformat(last_run_str)
            if last_run_str
            else None
        )
        if not cron_is_due(row.schedule_cron, last_run, now):
            continue

        logger.info(
            "User agent '%s' is due (schedule=%s)",
            row.agent_id, row.schedule_cron,
        )

        try:
            if data_tool_ids_for_row(row):
                summary = run_user_agent_batch(layer, row.agent_id)
            else:
                summary = run_user_agent_generic(layer, row.agent_id)
            status = "success" if summary.errors == 0 else "error"
            run_list.append({
                "agent_id": row.agent_id,
                "status": status,
            })
            # Only advance schedule state on full success. Generic
            # failures must retry (no per-item cursor), and batch
            # failures retry too — though for batch the next tick is a
            # near no-op against already-processed items via the
            # ``_user_agent_processed_items`` cursor.
            if summary.errors == 0:
                state[row.agent_id] = now.isoformat()
            for msg in summary.error_messages:
                error_list.append(f"{row.agent_id}: {msg}")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Scheduled user agent '%s' failed: %s",
                row.agent_id, exc,
            )
            error_list.append(f"{row.agent_id}: {exc}")

    return checked, run_list, error_list


def cmd_list_skills() -> int:
    """List all registered skills (built-in + user-authored).

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 1
    """
    try:
        from src.agent_runtime.skills import (
            SkillRegistry,
            register_user_skills_from_db,
        )

        registry = SkillRegistry()
        registry.register_builtin_skills()
        builtin_ids = {s.id for s in registry.list_skills()}
        try:
            from src.agents.user_agents.skill_store import UserSkillStore

            store = UserSkillStore()
            try:
                register_user_skills_from_db(registry, store)
            finally:
                store.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load user skills: %s", exc)

        skills = registry.list_skills()
        print(_json_output([
            {
                "id": s.id,
                "name": s.name,
                "description": s.description,
                "category": s.category,
                "uses_llm": s.uses_llm,
                "builtin": s.id in builtin_ids,
                "parameters": dict(s.parameters),
            }
            for s in skills
        ]))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# Extension management commands
# ---------------------------------------------------------------------------


def cmd_uninstall_extension(
    connector_id: str,
    preserve_data: bool = True,
) -> int:
    """Uninstall an extension: disable connector, remove from registry, clean up.

    Args:
        connector_id: The connector to uninstall.
        preserve_data: If True, keep raw data tables. If False, drop them.

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 1
    """
    try:
        from src.extensions.connectors.connection_manager import ConnectionManager
        from src.extensions.connectors.registry import ExtensionRegistry

        tables_removed: list[str] = []

        # Disable the connector (stops MCP client, cancels scheduled syncs)
        manager = ConnectionManager()
        try:
            manager.disable(connector_id)
        except Exception:
            pass  # Already disabled or unknown — continue cleanup

        # Remove from persistent registry
        registry = ExtensionRegistry()
        registry.remove(connector_id)

        # Clean up staged/generated model files
        ext_dir = Path.home() / ".arandu" / "extensions" / connector_id
        if ext_dir.exists():
            import shutil
            shutil.rmtree(ext_dir)
            tables_removed.append(f"extensions/{connector_id}/")

        print(_json_output({
            "status": "uninstalled",
            "connector_id": connector_id,
            "tables_removed": tables_removed,
            "data_preserved": preserve_data,
            "error": None,
        }))
        return 0
    except Exception as exc:
        print(_json_output({
            "status": "error",
            "connector_id": connector_id,
            "tables_removed": [],
            "data_preserved": True,
            "error": str(exc),
        }))
        return 1


def cmd_connector_history(
    connector_id: str,
    limit: int = 20,
) -> int:
    """Get sync history for a connector.

    Args:
        connector_id: The connector to query.
        limit: Maximum number of history entries.

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 1
    """
    try:
        history_file = (
            Path.home()
            / ".arandu"
            / "data"
            / "sync_history"
            / f"{connector_id}.json"
        )
        if not history_file.exists():
            print(_json_output([]))
            return 0

        data = json.loads(history_file.read_text())
        entries = data if isinstance(data, list) else []
        # Return most recent first, limited
        entries = entries[-limit:]
        entries.reverse()
        print(_json_output(entries))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_health(layer: DataLayer) -> int:
    """Check all system components and report status as JSON.

    Returns a JSON object with ``ok`` (bool) and ``checks`` (list).
    Each check has ``component``, ``ok``, ``detail`` or ``error``.

    Args:
        layer: An open DataLayer instance.

    Returns:
        Exit code (0 = all healthy, 1 = one or more degraded).

    sensitivity_tier: 1
    """
    checks: list[dict[str, Any]] = []

    # 1. SQLite
    try:
        stats = layer.get_stats()
        total = stats.total_sqlite_rows
        checks.append({
            "component": "sqlite",
            "ok": True,
            "detail": (
                f"{len(stats.sqlite)} tables, {total} rows"
            ),
        })
    except Exception as exc:  # noqa: BLE001
        checks.append({
            "component": "sqlite",
            "ok": False,
            "error": str(exc),
        })

    # 2. Kuzu
    try:
        nodes = stats.total_kuzu_nodes
        checks.append({
            "component": "kuzu",
            "ok": True,
            "detail": (
                f"{len(stats.kuzu_nodes)} types, "
                f"{nodes} nodes"
            ),
        })
    except Exception as exc:  # noqa: BLE001
        checks.append({
            "component": "kuzu",
            "ok": False,
            "error": str(exc),
        })

    # 3. ChromaDB
    try:
        docs = stats.total_chroma_docs
        checks.append({
            "component": "chromadb",
            "ok": True,
            "detail": (
                f"{len(stats.chromadb)} collections, "
                f"{docs} docs"
            ),
        })
    except Exception as exc:  # noqa: BLE001
        checks.append({
            "component": "chromadb",
            "ok": False,
            "error": str(exc),
        })

    # 4. Pipeline
    try:
        ps = layer.get_pipeline_status()
        stale = ps["is_stale"]
        last = ps["last_run"]
        detail = f"stale={stale}"
        if last:
            detail += f", last: {last['status']}"
        checks.append({
            "component": "pipeline",
            "ok": True,
            "detail": detail,
        })
    except Exception as exc:  # noqa: BLE001
        checks.append({
            "component": "pipeline",
            "ok": False,
            "error": str(exc),
        })

    # 5. Ollama
    try:
        from src.models.ollama_manager import OllamaManager

        mgr = OllamaManager()
        status = mgr.check_health()
        checks.append({
            "component": "ollama",
            "ok": status.server_reachable,
            "detail": (
                f"{status.chat_model} "
                f"({status.chat_model_status.value})"
            ),
        })
    except Exception as exc:  # noqa: BLE001
        checks.append({
            "component": "ollama",
            "ok": False,
            "error": str(exc),
        })

    # 6. Enabled connectors
    try:
        from src.extensions.connectors.registry import ExtensionRegistry

        registry = ExtensionRegistry()
        enabled = registry.get_enabled()
        checks.append({
            "component": "connectors",
            "ok": True,
            "detail": f"{len(enabled)} enabled",
        })
    except Exception as exc:  # noqa: BLE001
        checks.append({
            "component": "connectors",
            "ok": False,
            "error": str(exc),
        })

    # 7. Tool Registry
    try:
        from src.agents.tool_registry import ToolRegistry
        from src.extensions.connectors.catalog import ConnectorCatalog

        tr = ToolRegistry(
            catalog=ConnectorCatalog(),
            registry=ExtensionRegistry(),
        )
        actions = tr.get_available_actions()
        checks.append({
            "component": "tool_registry",
            "ok": True,
            "detail": f"{len(actions)} action tools",
        })
    except Exception as exc:  # noqa: BLE001
        checks.append({
            "component": "tool_registry",
            "ok": False,
            "error": str(exc),
        })

    all_ok = all(c["ok"] for c in checks)
    print(_json_output({"ok": all_ok, "checks": checks}))
    return 0 if all_ok else 1


# -------------------------------------------------------------------
# Interest profile commands (JSON output for Tauri bridge)
# -------------------------------------------------------------------


def _load_interest_overrides() -> dict[str, int] | None:
    """Load interest_overrides from settings.json if present.

    sensitivity_tier: 1
    """
    try:
        settings_file = (
            Path.home() / ".arandu" / "settings.json"
        )
        if settings_file.exists():
            data = json.loads(settings_file.read_text())
            overrides = data.get("interest_overrides")
            if isinstance(overrides, dict) and overrides:
                return overrides
    except Exception:  # noqa: BLE001
        pass
    return None


def cmd_get_interests(layer: DataLayer) -> int:
    """Return the interest profile as JSON.

    Args:
        layer: An open DataLayer instance.

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 1
    """
    try:
        from dataclasses import asdict

        from src.core.query_tracker import QueryTracker

        tracker = QueryTracker(db_engine=layer.duckdb)
        overrides = _load_interest_overrides()
        profile = tracker.get_interest_profile(
            overrides=overrides,
        )
        print(_json_output([asdict(area) for area in profile]))
        return 0
    except Exception as exc:
        print(
            _json_output({"error": str(exc)}),
            file=sys.stderr,
        )
        return 1


def cmd_get_domain_stats(layer: DataLayer) -> int:
    """Return per-domain query statistics as JSON.

    Args:
        layer: An open DataLayer instance.

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 1
    """
    try:
        from dataclasses import asdict

        from src.core.query_tracker import QueryTracker

        tracker = QueryTracker(db_engine=layer.duckdb)
        stats = tracker.get_domain_stats()
        print(_json_output([asdict(s) for s in stats]))
        return 0
    except Exception as exc:
        print(
            _json_output({"error": str(exc)}),
            file=sys.stderr,
        )
        return 1


def cmd_plan_refresh(layer: DataLayer) -> int:
    """Return a smart pipeline refresh plan as JSON.

    Creates a :class:`PipelineBrain` and generates a prioritized
    :class:`RefreshPlan` based on user interest profile and data
    freshness.

    Args:
        layer: An open DataLayer instance.

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 1
    """
    try:
        from src.core.query_tracker import QueryTracker
        from src.pipeline.pipeline_brain import PipelineBrain
        from src.pipeline.runner import PipelineRunner

        tracker = QueryTracker(db_engine=layer.duckdb)
        runner = PipelineRunner(duckdb=layer.duckdb)
        brain = PipelineBrain(
            query_tracker=tracker,
            pipeline_runner=runner,
        )
        plan = brain.plan_refresh()
        print(_json_output(plan.to_dict()))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(
            _json_output({"error": str(exc)}),
            file=sys.stderr,
        )
        return 1


def cmd_get_insights(layer: DataLayer, limit: int = 3) -> int:
    """Return active (non-dismissed) insights as JSON.

    Args:
        layer: An open DataLayer instance.
        limit: Maximum insights to return.

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 1
    """
    try:
        from dataclasses import asdict

        from src.agents.brain import BrainAgentV2
        from src.agents.insight import InsightGenerator
        from src.core.query_engine import QueryEngine
        from src.core.query_tracker import QueryTracker

        tracker = QueryTracker(db_engine=layer.duckdb)
        qe = QueryEngine(
            duckdb=layer.duckdb,
            kuzu=layer.kuzu,
            chromadb=layer.chromadb,
        )
        brain = BrainAgentV2(query_engine=qe)
        gen = InsightGenerator(
            db_engine=layer.duckdb,
            query_tracker=tracker,
            brain_agent=brain,
        )
        insights = gen.get_active_insights(limit=limit)
        print(_json_output([asdict(i) for i in insights]))
        return 0
    except Exception as exc:
        print(
            _json_output({"error": str(exc)}),
            file=sys.stderr,
        )
        return 1


def cmd_generate_insights(layer: DataLayer) -> int:
    """Generate daily insights from recent question patterns.

    Uses the configured LLM provider.  Returns empty list on failure.

    Args:
        layer: An open DataLayer instance.

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 3
    """
    try:
        from dataclasses import asdict

        from src.agents.brain import BrainAgentV2
        from src.agents.insight import InsightGenerator
        from src.core.query_engine import QueryEngine
        from src.core.query_tracker import QueryTracker
        from src.models.llm_provider import create_provider_from_settings

        provider = create_provider_from_settings(background=True)
        tracker = QueryTracker(db_engine=layer.duckdb)
        qe = QueryEngine(
            duckdb=layer.duckdb,
            kuzu=layer.kuzu,
            chromadb=layer.chromadb,
        )
        brain = BrainAgentV2(
            query_engine=qe, provider=provider,
        )
        gen = InsightGenerator(
            db_engine=layer.duckdb,
            query_tracker=tracker,
            brain_agent=brain,
        )
        insights = gen.generate_daily_insights()
        insight_dicts = [asdict(i) for i in insights]

        try:
            _maybe_notify_insights(layer.duckdb, insight_dicts)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Insight notification evaluation failed: %s", exc,
            )

        print(_json_output(insight_dicts))
        return 0
    except Exception as exc:
        print(
            _json_output({"error": str(exc)}),
            file=sys.stderr,
        )
        return 1


def cmd_dismiss_insight(
    layer: DataLayer,
    insight_id: str,
) -> int:
    """Dismiss an insight by its ID.

    Args:
        layer: An open DataLayer instance.
        insight_id: UUID of the insight to dismiss.

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 1
    """
    try:
        from src.agents.brain import BrainAgentV2
        from src.agents.insight import InsightGenerator
        from src.core.query_engine import QueryEngine
        from src.core.query_tracker import QueryTracker

        tracker = QueryTracker(db_engine=layer.duckdb)
        qe = QueryEngine(
            duckdb=layer.duckdb,
            kuzu=layer.kuzu,
            chromadb=layer.chromadb,
        )
        brain = BrainAgentV2(query_engine=qe)
        gen = InsightGenerator(
            db_engine=layer.duckdb,
            query_tracker=tracker,
            brain_agent=brain,
        )
        gen.dismiss_insight(insight_id)
        print(_json_output({"status": "dismissed"}))
        return 0
    except Exception as exc:
        print(
            _json_output({"error": str(exc)}),
            file=sys.stderr,
        )
        return 1


def cmd_follow_up_insight(
    layer: DataLayer,
    insight_id: str,
) -> int:
    """Mark an insight as followed-up and boost its domain.

    Args:
        layer: An open DataLayer instance.
        insight_id: UUID of the insight to follow up.

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 1
    """
    try:
        from src.agents.brain import BrainAgentV2
        from src.agents.insight import InsightGenerator
        from src.core.query_engine import QueryEngine
        from src.core.query_tracker import QueryTracker

        tracker = QueryTracker(db_engine=layer.duckdb)
        qe = QueryEngine(
            duckdb=layer.duckdb,
            kuzu=layer.kuzu,
            chromadb=layer.chromadb,
        )
        brain = BrainAgentV2(query_engine=qe)
        gen = InsightGenerator(
            db_engine=layer.duckdb,
            query_tracker=tracker,
            brain_agent=brain,
        )
        gen.follow_up_insight(insight_id)
        print(_json_output({"status": "followed_up"}))
        return 0
    except Exception as exc:
        print(
            _json_output({"error": str(exc)}),
            file=sys.stderr,
        )
        return 1


# ---------------------------------------------------------------------------
# Proactive intelligence commands (JSON output for Tauri bridge)
# ---------------------------------------------------------------------------


def cmd_evaluate_proactive(layer: DataLayer) -> int:
    """Run proactive intelligence evaluation (all 3 pillars).

    Evaluates pending replies, contact contexts, and actionable events.
    Sends WhatsApp notifications for high-priority items.

    Args:
        layer: An open DataLayer instance.

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 3
    """
    try:
        from dataclasses import asdict

        from src.agents.proactive import ProactiveIntelligence

        # Wire per-sender streaming notifications.
        # Each sender's LLM result fires a WhatsApp message immediately
        # instead of waiting for the full pipeline to complete.
        notifier = _ProactiveSenderNotifier(layer.duckdb)

        pi = ProactiveIntelligence(
            db_engine=layer.duckdb,
        )
        result = pi.evaluate_all(
            on_sender_result=notifier.on_sender_result,
        )
        result_dict = asdict(result)

        # Send a summary for events/contexts (non-sender items)
        try:
            _maybe_notify_events_and_contexts(
                layer.duckdb, result_dict,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Proactive event/context notification failed: %s", exc,
            )

        print(_json_output(result_dict))
        return 0
    except Exception as exc:
        print(
            _json_output({"error": str(exc)}),
            file=sys.stderr,
        )
        return 1


def cmd_get_pending_replies(layer: DataLayer) -> int:
    """Return active pending replies as JSON.

    Args:
        layer: An open DataLayer instance.

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 2
    """
    try:
        from dataclasses import asdict

        from src.agents.proactive import ProactiveIntelligence

        pi = ProactiveIntelligence(
            db_engine=layer.duckdb,
        )
        replies = pi.get_pending_replies()
        print(_json_output([asdict(r) for r in replies]))
        return 0
    except Exception as exc:
        print(
            _json_output({"error": str(exc)}),
            file=sys.stderr,
        )
        return 1


def cmd_get_contact_contexts(layer: DataLayer) -> int:
    """Return contact contexts as JSON.

    Args:
        layer: An open DataLayer instance.

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 3
    """
    try:
        from dataclasses import asdict

        from src.agents.proactive import ProactiveIntelligence

        pi = ProactiveIntelligence(
            db_engine=layer.duckdb,
        )
        contexts = pi.get_contact_contexts()
        print(_json_output([asdict(c) for c in contexts]))
        return 0
    except Exception as exc:
        print(
            _json_output({"error": str(exc)}),
            file=sys.stderr,
        )
        return 1


def cmd_get_actionable_events(layer: DataLayer) -> int:
    """Return actionable events as JSON.

    Args:
        layer: An open DataLayer instance.

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 2
    """
    try:
        from dataclasses import asdict

        from src.agents.proactive import ProactiveIntelligence

        pi = ProactiveIntelligence(
            db_engine=layer.duckdb,
        )
        events = pi.get_actionable_events()
        print(_json_output([asdict(e) for e in events]))
        return 0
    except Exception as exc:
        print(
            _json_output({"error": str(exc)}),
            file=sys.stderr,
        )
        return 1


def cmd_dismiss_pending_reply(
    layer: DataLayer, reply_id: str,
) -> int:
    """Dismiss a pending reply.

    Args:
        layer: An open DataLayer instance.
        reply_id: ID of the pending reply to dismiss.

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 1
    """
    try:
        from src.agents.proactive import ProactiveIntelligence

        pi = ProactiveIntelligence(
            db_engine=layer.duckdb,
        )
        pi.dismiss_pending_reply(reply_id)
        print(_json_output({"status": "dismissed"}))
        return 0
    except Exception as exc:
        print(
            _json_output({"error": str(exc)}),
            file=sys.stderr,
        )
        return 1


def cmd_dismiss_actionable_event(
    layer: DataLayer, event_id: str,
) -> int:
    """Dismiss an actionable event.

    Args:
        layer: An open DataLayer instance.
        event_id: ID of the actionable event to dismiss.

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 1
    """
    try:
        from src.agents.proactive import ProactiveIntelligence

        pi = ProactiveIntelligence(
            db_engine=layer.duckdb,
        )
        pi.dismiss_actionable_event(event_id)
        print(_json_output({"status": "dismissed"}))
        return 0
    except Exception as exc:
        print(
            _json_output({"error": str(exc)}),
            file=sys.stderr,
        )
        return 1


# ---------------------------------------------------------------------------
# Tasks / Goals / Habits / Schedule commands
# ---------------------------------------------------------------------------


def _task_curator(layer: DataLayer):  # type: ignore[no-untyped-def]
    """Build a TaskCurator over the analytical SQLite engine.

    sensitivity_tier: 1
    """
    from src.agents.tasks import TaskCurator

    return TaskCurator(db_engine=layer.duckdb)


def _dataclass_list_json(rows: list) -> str:  # type: ignore[type-arg]
    """sensitivity_tier: 1"""
    from dataclasses import asdict

    return _json_output([asdict(r) for r in rows])


def cmd_goals_list(
    layer: DataLayer, status: str | None, category: str | None,
) -> int:
    """List goals enriched with a derived ``urgency_score``.

    ``urgency_score`` blends tasks_today, overdue work, target-date
    proximity and horizon so the dashboard can sort by what's pressing
    rather than by importance alone (which biases toward big, slow-
    moving aspirations). Importance is still the tie-breaker.

    sensitivity_tier: 2
    """
    try:
        from dataclasses import asdict

        curator = _task_curator(layer)
        goals = curator.list_goals(
            status=status or None, category=category or None,
        )

        enriched: list[dict[str, Any]] = []
        for g in goals:
            try:
                progress = _compute_goal_progress(layer, g.id)
                tasks_today = len(progress.get("tasks_today", []))
                overdue = int(progress.get("overdue_tasks", 0))
                streak = int(progress.get("habit_streak_days", 0))
            except Exception:  # noqa: BLE001
                tasks_today, overdue, streak = 0, 0, 0
            urgency = _compute_goal_urgency(
                horizon=g.horizon,
                target_date=g.target_date,
                tasks_today=tasks_today,
                overdue_tasks=overdue,
                habit_streak_days=streak,
            )
            enriched.append({**asdict(g), "urgency_score": urgency})

        enriched.sort(
            key=lambda d: (
                int(d.get("urgency_score") or 0),
                int(d.get("importance") or 0),
            ),
            reverse=True,
        )
        print(_json_output(enriched))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_goals_create(
    layer: DataLayer,
    *,
    title: str,
    category: str,
    description: str,
    horizon: str,
    target_date: str | None,
    importance: int,
    why: str,
) -> int:
    """sensitivity_tier: 2"""
    from dataclasses import asdict

    try:
        curator = _task_curator(layer)
        g = curator.create_goal(
            title=title,
            category=category,
            description=description,
            horizon=horizon,
            target_date=target_date or None,
            importance=importance,
            why=why,
        )
        print(_json_output(asdict(g)))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_goals_update(
    layer: DataLayer, goal_id: str, patch_json: str,
) -> int:
    """sensitivity_tier: 2"""
    from dataclasses import asdict

    try:
        patch = json.loads(patch_json or "{}")
        if not isinstance(patch, dict):
            raise ValueError("patch must be a JSON object")
        curator = _task_curator(layer)
        g = curator.update_goal(goal_id, **patch)
        print(_json_output(asdict(g) if g else None))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_goals_mine(layer: DataLayer) -> int:
    """sensitivity_tier: 3"""
    try:
        curator = _task_curator(layer)
        created = curator.mine_goals()
        if created is None:
            # Failed run, not an empty result — the UI must not render
            # this as "no goals found".
            raise RuntimeError(
                "goal mining failed — the model call did not complete",
            )
        print(_dataclass_list_json(created))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_projects_list(
    layer: DataLayer, status: str | None, category: str | None,
) -> int:
    """sensitivity_tier: 2"""
    try:
        curator = _task_curator(layer)
        projects = curator.list_projects(
            status=status or None, category=category or None,
        )
        print(_dataclass_list_json(projects))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_projects_create(
    layer: DataLayer,
    *,
    name: str,
    category: str,
    goal_id: str | None,
) -> int:
    """sensitivity_tier: 2"""
    from dataclasses import asdict

    try:
        curator = _task_curator(layer)
        p = curator.create_project(
            name=name, category=category, goal_id=goal_id or None,
        )
        print(_json_output(asdict(p)))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_projects_archive(layer: DataLayer, project_id: str) -> int:
    """sensitivity_tier: 2"""
    try:
        curator = _task_curator(layer)
        curator.archive_project(project_id)
        print(_json_output({"status": "archived"}))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_tasks_list(
    layer: DataLayer,
    *,
    status: str | None,
    project_id: str | None,
    goal_id: str | None,
    parent_task_id: str | None,
) -> int:
    """sensitivity_tier: 2"""
    try:
        curator = _task_curator(layer)
        tasks = curator.list_tasks(
            status=status or None,
            project_id=project_id or None,
            goal_id=goal_id or None,
            parent_task_id=parent_task_id or None,
        )
        print(_dataclass_list_json(tasks))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_tasks_create(
    layer: DataLayer,
    *,
    title: str,
    project_id: str | None,
    parent_task_id: str | None,
    goal_id: str | None,
    notes: str,
    importance: int,
    due_at: str | None,
) -> int:
    """sensitivity_tier: 2"""
    from dataclasses import asdict

    try:
        curator = _task_curator(layer)
        t = curator.create_task(
            title=title,
            project_id=project_id or None,
            parent_task_id=parent_task_id or None,
            goal_id=goal_id or None,
            notes=notes,
            importance=importance,
            due_at=due_at or None,
        )
        print(_json_output(asdict(t)))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_tasks_update(
    layer: DataLayer, task_id: str, patch_json: str,
) -> int:
    """sensitivity_tier: 2"""
    from dataclasses import asdict

    try:
        patch = json.loads(patch_json or "{}")
        if not isinstance(patch, dict):
            raise ValueError("patch must be a JSON object")
        curator = _task_curator(layer)
        t = curator.update_task(task_id, **patch)
        print(_json_output(asdict(t) if t else None))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_tasks_toggle(
    layer: DataLayer, task_id: str, completion_note: str | None,
) -> int:
    """sensitivity_tier: 2"""
    from dataclasses import asdict

    try:
        curator = _task_curator(layer)
        t = curator.toggle_task_done(
            task_id, completion_note=completion_note or None,
        )
        print(_json_output(asdict(t) if t else None))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_tasks_delete(layer: DataLayer, task_id: str) -> int:
    """sensitivity_tier: 2"""
    try:
        curator = _task_curator(layer)
        curator.delete_task(task_id)
        print(_json_output({"status": "deleted"}))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_habits_list(
    layer: DataLayer, status: str | None, goal_id: str | None,
) -> int:
    """sensitivity_tier: 1"""
    try:
        curator = _task_curator(layer)
        habits = curator.list_habits(
            status=status or None, goal_id=goal_id or None,
        )
        print(_dataclass_list_json(habits))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_habits_create(
    layer: DataLayer,
    *,
    title: str,
    goal_id: str,
    cadence: str,
    days_of_week_json: str,
    preferred_window: str,
    why: str,
) -> int:
    """sensitivity_tier: 1"""
    from dataclasses import asdict

    try:
        days = json.loads(days_of_week_json or "[]")
        if not isinstance(days, list):
            days = []
        curator = _task_curator(layer)
        h = curator.create_habit(
            title=title,
            goal_id=goal_id,
            cadence=cadence,
            days_of_week=tuple(str(d) for d in days),
            preferred_window=preferred_window,
            why=why,
        )
        print(_json_output(asdict(h)))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_habits_toggle(layer: DataLayer, habit_id: str) -> int:
    """sensitivity_tier: 1"""
    try:
        curator = _task_curator(layer)
        curator.toggle_habit(habit_id)
        print(_json_output({"status": "toggled"}))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_habits_delete(layer: DataLayer, habit_id: str) -> int:
    """sensitivity_tier: 1"""
    try:
        curator = _task_curator(layer)
        curator.delete_habit(habit_id)
        print(_json_output({"status": "deleted"}))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_habits_regenerate(layer: DataLayer) -> int:
    """sensitivity_tier: 2"""
    try:
        curator = _task_curator(layer)
        habits = curator.regenerate_habits()
        print(_dataclass_list_json(habits))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_schedule_get(
    layer: DataLayer, schedule_date: str | None,
) -> int:
    """sensitivity_tier: 2"""
    from dataclasses import asdict

    try:
        curator = _task_curator(layer)
        record = curator.get_daily_schedule(
            schedule_date or None,
        )
        if record is None:
            print(_json_output(None))
            return 0
        # Slots are dataclasses; asdict recurses.
        print(_json_output(asdict(record)))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_schedule_regenerate(
    layer: DataLayer, schedule_date: str | None,
) -> int:
    """sensitivity_tier: 2"""
    from dataclasses import asdict

    try:
        curator = _task_curator(layer)
        record = curator.regenerate_daily_schedule(
            schedule_date=schedule_date or None,
        )
        if record is None:
            print(_json_output(None))
            return 0
        print(_json_output(asdict(record)))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# Mission Control dashboard commands
# ---------------------------------------------------------------------------


_DASHBOARD_BRIEF_CACHE_PATH = (
    Path.home() / ".arandu" / "data" / "dashboard_brief.json"
)


def _dashboard_brief_cache_key(layer: DataLayer) -> str:
    """Cache key for the daily brief — (date, last pipeline completed_at).

    The brief invalidates when the day changes or when the pipeline
    runs (which may bring in new events / replies / topics).

    sensitivity_tier: 1
    """
    try:
        status = layer.get_pipeline_status()
        last_run = status.get("last_run") or {}
        last_completed = str(last_run.get("completed_at") or "")
    except Exception:  # noqa: BLE001
        last_completed = ""
    return f"{date.today().isoformat()}|{last_completed}"


def _load_brief_cache() -> dict[str, Any]:
    """Read the brief cache file, returning an empty dict on any failure.

    sensitivity_tier: 3 (cached brief is tier-3 narrative)
    """
    try:
        if _DASHBOARD_BRIEF_CACHE_PATH.exists():
            return json.loads(_DASHBOARD_BRIEF_CACHE_PATH.read_text())
    except Exception:  # noqa: BLE001
        pass
    return {}


def _write_brief_cache(payload: dict[str, Any]) -> None:
    """Best-effort write of the brief cache; ignores filesystem errors.

    sensitivity_tier: 3
    """
    try:
        _DASHBOARD_BRIEF_CACHE_PATH.parent.mkdir(
            parents=True, exist_ok=True,
        )
        _DASHBOARD_BRIEF_CACHE_PATH.write_text(json.dumps(payload))
    except Exception:  # noqa: BLE001
        pass


def _gather_brief_context(layer: DataLayer) -> tuple[str, dict[str, int]]:
    """Build the structured prompt context for the daily brief.

    Returns a tuple of ``(prompt_context, source_counts)`` where
    ``prompt_context`` is a plain-text summary the brain reads to
    produce the narrative, and ``source_counts`` powers the UI
    "synthesized from N events / M replies" label.

    sensitivity_tier: 3
    """
    from dataclasses import asdict

    from src.agents.proactive import ProactiveIntelligence

    # Brief covers the user's own day — events on shared/subscribed
    # calendars where they're not invited belong elsewhere (LifeDomains
    # "Team awareness" / "Subscribed" sections), not in the narrative.
    from src.core.calendar_filters import personal_events_for_date

    today_events = personal_events_for_date(
        layer.duckdb,
        date.today().isoformat(),
        columns=(
            "title, start_time, end_time, location, "
            "COALESCE(event_origin, 'personal') AS event_origin"
        ),
        limit=10,
    )

    pi = ProactiveIntelligence(db_engine=layer.duckdb)
    try:
        pending = [asdict(r) for r in pi.get_pending_replies()][:3]
    except Exception:  # noqa: BLE001
        pending = []
    try:
        actionable = [asdict(e) for e in pi.get_actionable_events()][:3]
    except Exception:  # noqa: BLE001
        actionable = []

    top_threads: list[dict[str, Any]] = []
    try:
        top_threads = layer.duckdb.query(
            "SELECT contact_name, top_topic, max_topic_importance, "
            "notification_priority, last_message_at "
            "FROM mart_contact_summary "
            "WHERE topic_count > 0 AND notification_priority >= 30 "
            "ORDER BY notification_priority DESC LIMIT 5",
        )
    except Exception:  # noqa: BLE001
        top_threads = []

    lines: list[str] = ["Today is " + date.today().isoformat() + "."]
    if today_events:
        lines.append("Today's events:")
        for evt in today_events:
            lines.append(
                f"- {evt.get('title', 'Untitled')} at "
                f"{evt.get('start_time', '?')}"
                + (f" ({evt['location']})" if evt.get("location") else "")
            )
    else:
        lines.append("No events on the calendar today.")

    if pending:
        lines.append("Pending replies:")
        for reply in pending:
            lines.append(
                f"- {reply.get('contact_name', '?')} via "
                f"{reply.get('source', '?')}: "
                f"{reply.get('reason') or reply.get('preview') or ''}"
            )

    if actionable:
        lines.append("Actionable events:")
        for ev in actionable:
            lines.append(
                f"- {ev.get('title', '?')} on "
                f"{ev.get('event_date', '?')}: "
                f"{ev.get('action_needed', '')}"
            )

    if top_threads:
        lines.append("Active threads:")
        for thread in top_threads:
            lines.append(
                f"- {thread.get('contact_name', '?')}: "
                f"{thread.get('top_topic', '')}"
            )

    counts = {
        "events": len(today_events),
        "threads": len(top_threads),
        "pending_replies": len(pending),
        "actionable_events": len(actionable),
    }
    return "\n".join(lines), counts


def cmd_get_daily_brief(layer: DataLayer, force: bool = False) -> int:
    """Return today's synthesized brief as JSON (cached).

    On a cache hit (same date + same pipeline ``completed_at``), the
    brain LLM is NOT invoked — the cached narrative is returned. The
    Dashboard regenerate button passes ``--force`` to bypass the cache.

    Output schema::

        {
          "brief": str,
          "generated_at": iso-8601,
          "source_counts": {events, threads, pending_replies, actionable_events}
        }

    sensitivity_tier: 3
    """
    try:
        cache_key = _dashboard_brief_cache_key(layer)
        if not force:
            cache = _load_brief_cache()
            if cache.get("key") == cache_key and cache.get("brief"):
                print(_json_output({
                    "brief": cache["brief"],
                    "generated_at": cache.get("generated_at", ""),
                    "source_counts": cache.get("source_counts", {}),
                }))
                return 0

        context, counts = _gather_brief_context(layer)
        is_empty_day = all(v == 0 for v in counts.values())
        if is_empty_day:
            brief = (
                "Your day is open — no scheduled events, no pending "
                "replies. A good window to set your own agenda."
            )
        else:
            from src.agents.brain import BrainAgentV2
            from src.agents.core.task_budget import TaskBudget
            from src.agents.tool_registry import ToolRegistry
            from src.core.query_engine import QueryEngine
            from src.extensions.connectors.catalog import ConnectorCatalog
            from src.extensions.connectors.registry import ExtensionRegistry
            from src.models.llm_provider import create_provider_from_settings

            qe = QueryEngine(
                duckdb=layer.duckdb,
                kuzu=layer.kuzu,
                chromadb=layer.chromadb,
            )
            provider = create_provider_from_settings()
            tool_registry = ToolRegistry(
                catalog=ConnectorCatalog(),
                registry=ExtensionRegistry(),
            )
            brain = BrainAgentV2(
                query_engine=qe,
                tool_registry=tool_registry,
                provider=provider,
            )
            prompt = (
                "You are writing a 2-3 sentence morning brief for the "
                "user. Be concrete and synthesize across sources. "
                "Don't list items — narrate.\n\n"
                "Context:\n" + context
            )
            try:
                # Daily brief is the canonical background-deep caller:
                # the user expects a slower, more thorough synthesis
                # than a chat question. ``background_deep`` reflects at
                # 30s/60s rather than at 10s/10s.
                resp = brain.ask(prompt, budget=TaskBudget.background_deep())
                brief = resp.answer.strip()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "daily brief LLM call failed, returning fallback: %s",
                    exc,
                )
                brief = (
                    f"{counts['events']} events, "
                    f"{counts['pending_replies']} pending replies, "
                    f"{counts['threads']} active threads. "
                    "Brief synthesis unavailable — open Chat for details."
                )

        generated_at = datetime.now(tz=timezone.utc).isoformat()
        _write_brief_cache({
            "key": cache_key,
            "brief": brief,
            "generated_at": generated_at,
            "source_counts": counts,
        })
        print(_json_output({
            "brief": brief,
            "generated_at": generated_at,
            "source_counts": counts,
        }))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def _thread_status(
    *, last_activity: str | None, kind: str,
    waiting_on_user: bool,
) -> str:
    """Compute a coarse status label for a Thread.

    Status codes — surfaced as colored dots in the UI:
      - ``waiting``: stale + needs the user's attention.
      - ``soon``: events within 48h or fresh high-importance.
      - ``healthy``: recent two-way activity, no friction.
      - ``quiet``: no activity for > 7 days.

    sensitivity_tier: 1
    """
    if not last_activity:
        return "quiet"
    try:
        last_dt = datetime.fromisoformat(
            last_activity.replace("Z", "+00:00"),
        )
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)
    except Exception:  # noqa: BLE001
        return "quiet"

    now = datetime.now(tz=timezone.utc)
    hours = (now - last_dt).total_seconds() / 3600.0

    if kind == "event":
        return "soon" if abs(hours) <= 48 else "healthy"
    if waiting_on_user and hours > 48:
        return "waiting"
    if hours > 24 * 7:
        return "quiet"
    return "healthy"


def cmd_get_active_threads(layer: DataLayer, limit: int = 10) -> int:
    """Return cross-source "threads of activity" as JSON.

    A *thread* aggregates signals across data sources into a single
    unit of attention (a project, a relationship arc, an in-flight
    decision). Phase 1 sources:

      - Per-contact topics from ``mart_contact_summary`` — each active
        topic with notification_priority >= 20 becomes one Thread.
      - Actionable events from ``_actionable_events`` (birthdays,
        upcoming events needing prep) — each becomes one Thread.

    Output schema::

        [{
          "id": str,
          "kind": "conversation"|"event",
          "title": str,
          "subtitle": str | None,
          "status": "waiting"|"soon"|"healthy"|"quiet",
          "sources": [str],
          "last_activity": iso-8601 | None,
          "suggested_actions": [{"label": str, "intent": str,
                                  "payload": dict}],
        }]

    sensitivity_tier: 3
    """
    try:
        from dataclasses import asdict

        from src.agents.proactive import ProactiveIntelligence

        threads: list[dict[str, Any]] = []

        # ---- Conversation threads from mart_contact_summary ----
        try:
            contact_rows = layer.duckdb.query(
                "SELECT contact_name, top_topic, active_topics_json, "
                "max_topic_importance, notification_priority, "
                "last_message_at, primary_channel, messages_7d "
                "FROM mart_contact_summary "
                "WHERE topic_count > 0 AND notification_priority >= 20 "
                "ORDER BY notification_priority DESC LIMIT ?",
                [int(limit)],
            )
        except Exception:  # noqa: BLE001
            contact_rows = []

        for row in contact_rows:
            contact = row.get("contact_name") or "Unknown"
            topic = row.get("top_topic") or "Ongoing conversation"
            channel = row.get("primary_channel") or "messages"
            messages_7d = int(row.get("messages_7d") or 0)
            # No reply from the user when they haven't messaged back
            # recently — proxied by very low messages_7d on a high
            # priority contact. The pending-replies pillar handles
            # the precise case; this is the coarse rollup.
            waiting = messages_7d <= 1 and int(
                row.get("notification_priority") or 0,
            ) >= 50

            threads.append({
                "id": f"thread:contact:{contact}",
                "kind": "conversation",
                "title": f"{topic} — {contact}",
                "subtitle": (
                    f"{messages_7d} messages this week"
                    if messages_7d > 0
                    else "No activity this week"
                ),
                "status": _thread_status(
                    last_activity=row.get("last_message_at"),
                    kind="conversation",
                    waiting_on_user=waiting,
                ),
                "sources": [channel],
                "last_activity": row.get("last_message_at"),
                "suggested_actions": [
                    {
                        "label": "Draft reply",
                        "intent": "draft_reply",
                        "payload": {
                            "contact": contact, "topic": topic,
                        },
                    },
                    {
                        "label": "Open thread",
                        "intent": "open_thread",
                        "payload": {"contact": contact},
                    },
                ],
            })

        # ---- Event threads from actionable events ----
        pi = ProactiveIntelligence(db_engine=layer.duckdb)
        try:
            events = pi.get_actionable_events()
        except Exception:  # noqa: BLE001
            events = []
        for ev_obj in events[:limit]:
            ev = asdict(ev_obj)
            threads.append({
                "id": f"thread:event:{ev['id']}",
                "kind": "event",
                "title": ev.get("title", "Upcoming event"),
                "subtitle": ev.get("action_needed") or None,
                "status": _thread_status(
                    last_activity=ev.get("event_date"),
                    kind="event",
                    waiting_on_user=False,
                ),
                "sources": ["calendar"]
                + (["contacts"] if ev.get("contact_name") else []),
                "last_activity": ev.get("event_date"),
                "suggested_actions": [
                    {
                        "label": "Open prep brief",
                        "intent": "open_prep",
                        "payload": {
                            "event_id": ev.get("event_id"),
                            "title": ev.get("title"),
                        },
                    },
                    {
                        "label": "Dismiss",
                        "intent": "dismiss_event",
                        "payload": {"id": ev.get("id")},
                    },
                ],
            })

        # Sort by status priority (waiting > soon > healthy > quiet)
        # then by last_activity recency.
        priority = {
            "waiting": 0, "soon": 1, "healthy": 2, "quiet": 3,
        }
        threads.sort(key=lambda t: (
            priority.get(t["status"], 9),
            t.get("last_activity") or "",
        ))
        print(_json_output(threads[:limit]))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_get_agent_stream(layer: DataLayer) -> int:  # noqa: ARG001
    """Return live agent activity as JSON for the Mission Control panel.

    Combines three sources:
      - ``awaiting_review``: pending replies (proactive intel) + active
        high-importance insights (importance >= 7).
      - ``recently_completed``: agent run-log entries from the last
        24 hours across all registered agents, status=success.
      - ``running``: empty in Phase 1; the Rust handler merges its
        in-memory ``active_tasks`` map into this slot.

    sensitivity_tier: 2
    """
    try:
        from dataclasses import asdict

        from src.agents.brain import bootstrap_agents
        from src.agents.core.registry import all_agents
        from src.agents.core.run_log import default_run_log
        from src.agents.proactive import ProactiveIntelligence

        # Awaiting review — pending replies
        awaiting: list[dict[str, Any]] = []
        try:
            pi = ProactiveIntelligence(db_engine=layer.duckdb)
            replies = [asdict(r) for r in pi.get_pending_replies()][:5]
            for reply in replies:
                awaiting.append({
                    "id": f"review:reply:{reply['id']}",
                    "agent_name": "Pending-reply scanner",
                    "summary": (
                        f"Reply needed for {reply.get('contact_name')}"
                        + (
                            f" — {reply['reason']}"
                            if reply.get("reason") else ""
                        )
                    ),
                    "kind": "reply",
                    "payload_ref": reply.get("id", ""),
                })
        except Exception:  # noqa: BLE001
            pass

        # Awaiting review — high-importance insights
        try:
            insights_rows = layer.duckdb.query(
                "SELECT id, title, content, domain, generated_at "
                "FROM ext_mart_insights "
                "WHERE dismissed = 0 "
                "ORDER BY generated_at DESC LIMIT 5",
            )
            for ins in insights_rows:
                awaiting.append({
                    "id": f"review:insight:{ins['id']}",
                    "agent_name": "Insight generator",
                    "summary": ins.get("title", ""),
                    "kind": "insight",
                    "payload_ref": ins.get("id", ""),
                })
        except Exception:  # noqa: BLE001
            pass

        # Recently completed — across registered agents from run-log
        recently: list[dict[str, Any]] = []
        try:
            bootstrap_agents()
            cutoff = (
                datetime.now(tz=timezone.utc)
                - timedelta(hours=24)
            ).isoformat()
            log = default_run_log()
            for definition in all_agents():
                try:
                    entries = log.recent(definition.agent_id, limit=5)
                except Exception:  # noqa: BLE001
                    continue
                for entry in entries:
                    if entry.status != "success":
                        continue
                    if (entry.ts or "") < cutoff:
                        continue
                    recently.append({
                        "id": f"completed:{definition.agent_id}:{entry.id}",
                        "agent_name": definition.name,
                        "summary": (
                            entry.input[:80] + "…"
                            if entry.input and len(entry.input) > 80
                            else (entry.input or "Completed")
                        ),
                        "finished_at": entry.ts,
                    })
        except Exception:  # noqa: BLE001
            pass
        # Keep the 10 most recent.
        recently.sort(key=lambda r: r.get("finished_at") or "", reverse=True)
        recently = recently[:10]

        print(_json_output({
            "running": [],  # Rust handler merges its active_tasks here
            "awaiting_review": awaiting,
            "recently_completed": recently,
        }))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


_DOMAIN_MARTS = {
    "work": "mart_work",
    "personal": "mart_personal",
    "health": "mart_health",
}

# pending_reply.domain enum is personal|work|family|social|health.
# "personal" tab is a broad bucket: anything not strictly work or health.
_DOMAIN_REPLY_FILTER = {
    "work": ("work",),
    "personal": ("personal", "family", "social"),
    "health": ("health",),
}


# Which event_category rows belong on each domain tab. Health uses
# metrics, not events, so it has no entry here.
_DOMAIN_EVENT_CATEGORIES = {
    "work": ("meeting",),
    "personal": ("social", "health"),
}


def _domain_event_items(
    layer: DataLayer, mart: str,
) -> list[dict[str, Any]]:
    """Today's event-shaped items from a work/personal mart.

    The mart already filters to ``event_origin = 'personal'``, so every
    row here is something the user owns or is invited to.

    sensitivity_tier: 3 (mart rows can carry tier-3 details)
    """
    try:
        rows = layer.duckdb.query(
            f"SELECT id, title, detail, occurred_at, contact_name "  # noqa: S608
            f"FROM {mart} "
            f"WHERE item_type = 'event' "
            f"AND DATE(occurred_at) = DATE('now') "
            f"ORDER BY occurred_at LIMIT 20",
        )
    except Exception:  # noqa: BLE001
        return []
    items: list[dict[str, Any]] = []
    for row in rows:
        items.append({
            "id": str(row.get("id", "")),
            "kind": "event",
            "title": row.get("title") or "Untitled event",
            "subtitle": (row.get("detail") or "")[:120] or None,
            "when": row.get("occurred_at"),
            "badge": None,
            "contact": row.get("contact_name"),
            "event_origin": "personal",
        })
    return items


def _domain_awareness_event_items(
    layer: DataLayer, domain: str,
) -> list[dict[str, Any]]:
    """Today's events the user is *not* invited to, scoped to a domain.

    Reads ``int_events_enriched`` directly because the per-domain marts
    are filtered to personal events only.

    sensitivity_tier: 3
    """
    categories = _DOMAIN_EVENT_CATEGORIES.get(domain, ())
    if not categories:
        return []
    placeholders = ",".join("?" for _ in categories)
    try:
        rows = layer.duckdb.query(
            f"SELECT id, title, description AS detail, start_time AS occurred_at, "  # noqa: S608
            f"known_attendee_names AS contact_name, "
            f"COALESCE(event_origin, 'personal') AS event_origin "
            f"FROM int_events_enriched "
            f"WHERE event_category IN ({placeholders}) "
            f"  AND COALESCE(event_origin, 'personal') <> 'personal' "
            f"  AND DATE(start_time) = DATE('now') "
            f"ORDER BY start_time LIMIT 20",
            list(categories),
        )
    except Exception:  # noqa: BLE001
        return []
    items: list[dict[str, Any]] = []
    for row in rows:
        items.append({
            "id": str(row.get("id", "")),
            "kind": "event",
            "title": row.get("title") or "Untitled event",
            "subtitle": (row.get("detail") or "")[:120] or None,
            "when": row.get("occurred_at"),
            "badge": None,
            "contact": row.get("contact_name"),
            "event_origin": str(row.get("event_origin") or "team_awareness"),
        })
    return items


def _domain_metric_items(layer: DataLayer) -> list[dict[str, Any]]:
    """Health metrics: latest value per type + 7-day anomaly flag.

    sensitivity_tier: 3
    """
    try:
        rows = layer.duckdb.query(
            "SELECT id, metric_type, value, unit, recorded_at, "
            "is_anomaly FROM mart_health "
            "WHERE is_latest = 1 "
            "ORDER BY recorded_at DESC LIMIT 10",
        )
    except Exception:  # noqa: BLE001
        return []
    items: list[dict[str, Any]] = []
    for row in rows:
        unit = row.get("unit") or ""
        value = row.get("value")
        badge = "anomaly" if row.get("is_anomaly") else None
        items.append({
            "id": str(row.get("id", "")),
            "kind": "metric",
            "title": str(row.get("metric_type") or "Metric"),
            "subtitle": (
                f"{value} {unit}".strip() if value is not None else None
            ),
            "when": row.get("recorded_at"),
            "badge": badge,
            "contact": None,
        })
    return items


def _domain_open_loops(
    layer: DataLayer, domain: str,
) -> list[dict[str, Any]]:
    """Open loops in a domain: pending replies whose domain matches.

    sensitivity_tier: 2
    """
    from dataclasses import asdict

    from src.agents.proactive import ProactiveIntelligence

    domains = _DOMAIN_REPLY_FILTER.get(domain, ())
    if not domains:
        return []

    try:
        pi = ProactiveIntelligence(db_engine=layer.duckdb)
        replies = [asdict(r) for r in pi.get_pending_replies()]
    except Exception:  # noqa: BLE001
        return []

    loops: list[dict[str, Any]] = []
    now = datetime.now(tz=timezone.utc)
    for reply in replies:
        if reply.get("domain") not in domains:
            continue
        # Compute age in days from message_at.
        age_days = 0
        msg_at = reply.get("message_at")
        if msg_at:
            try:
                dt = datetime.fromisoformat(
                    str(msg_at).replace("Z", "+00:00"),
                )
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                age_days = max(0, (now - dt).days)
            except Exception:  # noqa: BLE001
                age_days = 0
        loops.append({
            "id": f"loop:reply:{reply.get('id', '')}",
            "kind": "reply",
            "label": f"Reply to {reply.get('contact_name', '?')}",
            "context": reply.get("reason") or reply.get("preview") or "",
            "age_days": age_days,
            "suggested_action": "Draft reply",
            "source": reply.get("source") or None,
            "message_id": reply.get("message_id") or None,
            "contact_name": reply.get("contact_name") or None,
        })
    # Surface the most overdue first; cap at 8.
    loops.sort(key=lambda loop: loop["age_days"], reverse=True)
    return loops[:8]


def cmd_get_domain_summary(layer: DataLayer, domain: str) -> int:
    """Return today's items + open loops in a life domain (JSON).

    Phase 2 of the Mission Control redesign. ``domain`` is one of
    ``work`` / ``personal`` / ``health`` — each backed by the
    corresponding mart (``mart_work`` / ``mart_personal`` /
    ``mart_health``).

    Output schema::

        {
          "domain": str,
          "items": [{"id","kind","title","subtitle","when","badge","contact"}],
          "open_loops": [{"id","kind","label","context","age_days",
                           "suggested_action"}]
        }

    Items are events (work/personal) or latest metrics (health).
    Open loops are pending replies in that domain bucket; health
    currently surfaces no loops (anomaly handling is a follow-up).

    sensitivity_tier: 3
    """
    try:
        normalized = (domain or "").strip().lower()
        if normalized not in _DOMAIN_MARTS:
            print(
                _json_output({
                    "error": (
                        f"unknown domain {normalized!r}; expected one of "
                        f"{sorted(_DOMAIN_MARTS)}"
                    ),
                }),
                file=sys.stderr,
            )
            return 1

        if normalized == "health":
            items = _domain_metric_items(layer)
        else:
            items = _domain_event_items(
                layer, _DOMAIN_MARTS[normalized],
            )
            items.extend(
                _domain_awareness_event_items(layer, normalized),
            )

        loops = _domain_open_loops(layer, normalized)

        print(_json_output({
            "domain": normalized,
            "items": items,
            "open_loops": loops,
        }))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


# Domain → goal category mapping for the unified LifeBoard. The Health
# domain doesn't have a 1:1 goal category — we route "life" goals
# (health, fitness, lifestyle) there because that's the user-facing
# bucket users intuitively associate with the Health column.
_DOMAIN_TO_GOAL_CATEGORY: dict[str, str] = {
    "work": "work",
    "personal": "personal",
    "health": "life",
}


def cmd_get_life_board(layer: DataLayer) -> int:
    """Return the unified Life board for the dashboard (JSON).

    Replaces the two separate dashboard surfaces (Goals widget + Life
    snapshot) with a single canvas keyed on the three life domains
    (work / personal / health). For each domain we return:

    - ``goals`` — active goals in the mapped category, enriched with
      ``urgency_score`` and sorted by it so the most pressing goals
      float to the top of each column.
    - ``today_actions`` — the union of every goal's ``tasks_today`` +
      ``habits_today``, each carrying the parent goal's title/id so
      the UI can render "from <goal>" suffixes.
    - ``items`` / ``open_loops`` — reuses the existing domain summary
      computations (calendar events / metrics / pending replies).

    sensitivity_tier: 3
    """
    try:
        from dataclasses import asdict

        curator = _task_curator(layer)
        all_goals = curator.list_goals(status="active")

        domains: list[dict[str, Any]] = []
        for domain_name in ("work", "personal", "health"):
            category = _DOMAIN_TO_GOAL_CATEGORY[domain_name]

            # Goals in this domain's category, enriched + sorted.
            enriched_goals: list[dict[str, Any]] = []
            today_actions: list[dict[str, Any]] = []
            today_progress: dict[str, dict[str, int]] = {}
            for g in all_goals:
                if g.category != category:
                    continue
                try:
                    progress = _compute_goal_progress(layer, g.id)
                except Exception:  # noqa: BLE001
                    progress = {
                        "tasks_today": [],
                        "tasks_today_total": 0,
                        "tasks_today_done": 0,
                        "habits_today": [],
                        "overdue_tasks": 0,
                        "habit_streak_days": 0,
                    }
                tasks_today = progress.get("tasks_today", [])
                habits_today = progress.get("habits_today", [])
                urgency = _compute_goal_urgency(
                    horizon=g.horizon,
                    target_date=g.target_date,
                    tasks_today=len(tasks_today),
                    overdue_tasks=int(progress.get("overdue_tasks", 0)),
                    habit_streak_days=int(
                        progress.get("habit_streak_days", 0),
                    ),
                )
                enriched_goals.append({
                    **asdict(g),
                    "urgency_score": urgency,
                })
                today_progress[g.id] = {
                    "total": (
                        int(progress.get("tasks_today_total", 0))
                        + len(habits_today)
                    ),
                    "done": int(progress.get("tasks_today_done", 0)),
                }
                for t in tasks_today:
                    today_actions.append({
                        "id": f"task:{t.get('id')}",
                        "kind": "task",
                        "title": t.get("title", ""),
                        "goal_id": g.id,
                        "goal_title": g.title,
                        "when": t.get("due_at"),
                        "preferred_window": None,
                    })
                for h in habits_today:
                    today_actions.append({
                        "id": f"habit:{h.get('id')}",
                        "kind": "habit",
                        "title": h.get("title", ""),
                        "goal_id": g.id,
                        "goal_title": g.title,
                        "when": None,
                        "preferred_window": h.get("preferred_window"),
                    })

            enriched_goals.sort(
                key=lambda d: (
                    int(d.get("urgency_score") or 0),
                    int(d.get("importance") or 0),
                ),
                reverse=True,
            )

            # Today's shape — reuse existing domain summary helpers.
            if domain_name == "health":
                items = _domain_metric_items(layer)
            else:
                items = _domain_event_items(
                    layer, _DOMAIN_MARTS[domain_name],
                )
                items.extend(
                    _domain_awareness_event_items(layer, domain_name),
                )
            loops = _domain_open_loops(layer, domain_name)

            domains.append({
                "domain": domain_name,
                "goals": enriched_goals,
                "today_actions": today_actions,
                "today_progress": today_progress,
                "items": items,
                "open_loops": loops,
            })

        print(_json_output({"domains": domains}))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_today_board(layer: DataLayer) -> int:
    """Return the dashboard's prioritized Today board (JSON).

    Slices the persisted daily schedule into Now / Up Next / Loops and
    blends in the highest-importance pending replies. No new mart, no
    new LLM call — this is the dashboard's "single canonical surface
    for what to do right now" view.

    Output schema::

        {
          "now": [ScheduleSlot],
          "up_next": [ScheduleSlot],
          "todays_loops": [{
              "id","kind","label","context","importance","age_days"
          }],
          "rationale": str,
          "schedule_date": str | None
        }

    sensitivity_tier: 3
    """
    try:
        from dataclasses import asdict
        from datetime import datetime, timezone

        from src.agents.proactive import ProactiveIntelligence
        from src.agents.tasks import TaskCurator

        curator = TaskCurator(db_engine=layer.duckdb)
        record = curator.get_daily_schedule()
        slots = (
            [asdict(s) for s in record.slots] if record is not None else []
        )

        # Enrich slots with goal_title for frontend goal chips.
        goal_ids = {
            s.get("goal_id") for s in slots if s.get("goal_id")
        }
        goal_titles: dict[str, str] = {}
        if goal_ids:
            try:
                from src.agents.tasks.persistence import list_goals
                for g in list_goals(layer.duckdb):
                    if g.id in goal_ids:
                        goal_titles[g.id] = g.title
            except Exception:  # noqa: BLE001
                pass
        for slot in slots:
            gid = slot.get("goal_id")
            slot["goal_title"] = goal_titles.get(gid) if gid else None

        now = datetime.now(tz=timezone.utc)

        def _to_dt(s: dict[str, Any]) -> datetime | None:
            try:
                iso = str(s.get("start", "")).replace("Z", "+00:00")
                dt = datetime.fromisoformat(iso)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except Exception:  # noqa: BLE001
                return None

        def _end_dt(s: dict[str, Any]) -> datetime | None:
            try:
                iso = str(s.get("end", "")).replace("Z", "+00:00")
                dt = datetime.fromisoformat(iso)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except Exception:  # noqa: BLE001
                return None

        now_slots: list[dict[str, Any]] = []
        up_next: list[dict[str, Any]] = []
        for slot in slots:
            start = _to_dt(slot)
            end = _end_dt(slot)
            if start is None:
                continue
            if end is not None and start <= now <= end:
                now_slots.append(slot)
            elif start > now:
                up_next.append(slot)
        up_next = up_next[:4]

        loops: list[dict[str, Any]] = []
        try:
            pi = ProactiveIntelligence(db_engine=layer.duckdb)
            replies = [asdict(r) for r in pi.get_pending_replies()]
        except Exception:  # noqa: BLE001
            replies = []
        for reply in sorted(
            replies,
            key=lambda r: int(r.get("importance") or 0),
            reverse=True,
        )[:6]:
            age_days = 0
            msg_at = reply.get("message_at")
            if msg_at:
                try:
                    dt = datetime.fromisoformat(
                        str(msg_at).replace("Z", "+00:00"),
                    )
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    age_days = max(0, (now - dt).days)
                except Exception:  # noqa: BLE001
                    age_days = 0
            loops.append({
                "id": f"loop:reply:{reply.get('id', '')}",
                "kind": "reply",
                "label": f"Reply to {reply.get('contact_name', '?')}",
                "context": (
                    reply.get("reason") or reply.get("preview") or ""
                )[:200],
                "importance": int(reply.get("importance") or 0),
                "age_days": int(age_days),
                "source": reply.get("source") or None,
                "message_id": reply.get("message_id") or None,
                "contact_name": reply.get("contact_name") or None,
            })

        payload = {
            "now": now_slots,
            "up_next": up_next,
            "todays_loops": loops,
            "rationale": record.rationale if record is not None else "",
            "schedule_date": (
                record.schedule_date if record is not None else None
            ),
        }
        print(_json_output(payload))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def _compute_goal_progress(
    layer: DataLayer, goal_id: str,
) -> dict[str, Any]:
    """In-process variant of ``cmd_goal_progress``.

    Returns the same JSON payload shape so that ``cmd_goal_progress``,
    ``cmd_goals_list`` and ``cmd_get_life_board`` can share the data
    without re-shelling into the CLI. Adds ``overdue_tasks`` (count of
    open tasks whose ``due_at`` is before today) so the urgency formula
    can weigh missed deadlines without a second pass.

    sensitivity_tier: 3
    """
    from dataclasses import asdict
    from datetime import date as _date
    from datetime import datetime, timedelta, timezone

    from src.agents.tasks import TaskCurator

    curator = TaskCurator(db_engine=layer.duckdb)
    tasks = curator.list_tasks(goal_id=goal_id)
    habits = curator.list_habits(goal_id=goal_id)
    today_iso = _date.today().isoformat()

    def _scheduled_today(t: Any) -> bool:
        if t.due_at is None:
            return False
        try:
            return t.due_at[:10] == today_iso
        except Exception:  # noqa: BLE001
            return False

    def _completed_today(t: Any) -> bool:
        if t.status != "done" or t.completed_at is None:
            return False
        try:
            return str(t.completed_at)[:10] == today_iso
        except Exception:  # noqa: BLE001
            return False

    def _due_today(t: Any) -> bool:
        return _scheduled_today(t) and t.status != "done"

    # In scope = scheduled today (still open) OR completed today, so a
    # task finished today without a today `due_at` still counts.
    def _in_scope_today(t: Any) -> bool:
        if t.status == "done":
            return _completed_today(t)
        return _scheduled_today(t)

    def _overdue(t: Any) -> bool:
        if t.due_at is None or t.status == "done":
            return False
        try:
            return t.due_at[:10] < today_iso
        except Exception:  # noqa: BLE001
            return False

    tasks_today = [asdict(t) for t in tasks if _due_today(t)]
    tasks_today_total = sum(1 for t in tasks if _in_scope_today(t))
    tasks_today_done = sum(1 for t in tasks if _completed_today(t))
    tasks_open = sum(
        1 for t in tasks if t.status in ("todo", "in_progress")
    )
    overdue_tasks = sum(1 for t in tasks if _overdue(t))

    # 7-day window for tasks_done_7d and last_evidence_at
    cutoff = (
        datetime.now(tz=timezone.utc) - timedelta(days=7)
    ).isoformat()
    completed_recent = [
        t for t in tasks
        if t.status == "done"
        and t.completed_at
        and t.completed_at >= cutoff
    ]
    tasks_done_7d = len(completed_recent)
    last_evidence_at: str | None = None
    if completed_recent:
        last_evidence_at = max(
            str(t.completed_at) for t in completed_recent
        )

    # Habit streak proxy: count consecutive days from today back
    # where at least one goal-anchored task was completed.
    completed_days: set[str] = set()
    for t in tasks:
        if t.status == "done" and t.completed_at:
            completed_days.add(str(t.completed_at)[:10])
    streak = 0
    d = _date.today()
    while d.isoformat() in completed_days:
        streak += 1
        d = d - timedelta(days=1)

    # Rolled-up topics: join via _projects.topic_id.
    topics: list[dict[str, Any]] = []
    try:
        rows = layer.duckdb.query(
            "SELECT DISTINCT p.topic_id AS topic_id, "
            "       ict.topic AS title, "
            "       COALESCE(ict.importance, 0) AS importance, "
            "       mcs.last_message_at AS last_activity, "
            "       ict.contact_name AS contact_name "
            "FROM _projects p "
            "LEFT JOIN int_contact_topics ict "
            "  ON ict.topic = p.topic_id "
            "LEFT JOIN mart_contact_summary mcs "
            "  ON mcs.contact_name = ict.contact_name "
            "WHERE p.goal_id = ? AND p.topic_id IS NOT NULL "
            "ORDER BY importance DESC LIMIT 10",
            [goal_id],
        )
    except Exception:  # noqa: BLE001
        rows = []
    for row in rows:
        tid = row.get("topic_id")
        if not tid:
            continue
        topics.append({
            "topic_id": str(tid),
            "title": str(row.get("title") or tid),
            "importance": int(row.get("importance") or 0),
            "last_activity": row.get("last_activity"),
            "contact_name": row.get("contact_name"),
        })

    return {
        "goal_id": goal_id,
        "rolled_up_topics": topics,
        "tasks_today": tasks_today,
        "tasks_today_total": int(tasks_today_total),
        "tasks_today_done": int(tasks_today_done),
        "tasks_open": int(tasks_open),
        "overdue_tasks": int(overdue_tasks),
        "tasks_done_7d": int(tasks_done_7d),
        "habits_today": [asdict(h) for h in habits],
        "habit_streak_days": int(streak),
        "last_evidence_at": last_evidence_at,
    }


def _compute_goal_urgency(
    *,
    horizon: str,
    target_date: str | None,
    tasks_today: int,
    overdue_tasks: int,
    habit_streak_days: int,
) -> int:
    """Deterministic urgency score for goal ordering.

    Weights tasks due today highest, then overdue work, then near-term
    target dates, then horizon, with a tiny bonus for an active habit
    streak so an actively-worked-on goal beats a dormant one with the
    same task count. Pure function — testable in isolation.

    sensitivity_tier: 1
    """
    from datetime import date as _date

    score = tasks_today * 10 + overdue_tasks * 8
    if horizon == "short":
        score += 5
    elif horizon == "medium":
        score += 2
    if target_date:
        try:
            tgt = _date.fromisoformat(str(target_date)[:10])
            days = (tgt - _date.today()).days
            if days <= 7:
                score += 6
            elif days <= 30:
                score += 3
            elif days <= 90:
                score += 1
        except Exception:  # noqa: BLE001
            pass
    if habit_streak_days > 0:
        score += 1
    return score


def cmd_goal_progress(layer: DataLayer, goal_id: str) -> int:
    """Return progress + today's moves for a single goal (JSON).

    Joins ``_projects`` (for ``topic_id`` linkage), ``_tasks`` (filtered
    by goal_id directly or via project_id), ``_habits`` (by goal_id),
    and ``int_contact_topics`` (joined through ``_projects.topic_id``)
    to give the Goals page everything it needs for drill-down: rolled-up
    topics, tasks due today, open task count, last completion evidence,
    and a 7-day habit streak proxy (consecutive days with at least one
    goal-anchored completed task).

    sensitivity_tier: 3
    """
    try:
        payload = _compute_goal_progress(layer, goal_id)
        print(_json_output(payload))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_list_inbox(
    layer: DataLayer,
    *,
    domain: str | None = None,
    topic: str | None = None,
) -> int:
    """Return the unified inbox: pending replies + tasks + habits (JSON).

    Output schema::

        {
          "replies": [PendingReply],
          "tasks":   [InboxTask],     # tasks due/overdue/scheduled today
          "habits":  [InboxHabit],    # active habits due today
          "topics":  [GoalTopic]      # only when domain is None
        }

    ``domain``: optional filter (work / personal / health).
    ``topic``: optional filter that scopes replies by conversation thread.

    sensitivity_tier: 3
    """
    try:
        from dataclasses import asdict
        from datetime import date as _date

        from src.agents.proactive import ProactiveIntelligence
        from src.agents.tasks.persistence import (
            get_actionable_tasks_today,
            get_habits_today,
        )

        today_iso = _date.today().isoformat()
        domain_filter = (
            _DOMAIN_REPLY_FILTER.get(domain) if domain else None
        )

        pi = ProactiveIntelligence(db_engine=layer.duckdb)
        replies = [asdict(r) for r in pi.get_pending_replies()]
        if domain_filter:
            replies = [
                r for r in replies if r.get("domain") in domain_filter
            ]
        if topic:
            try:
                matching = layer.duckdb.query(
                    "SELECT contact_name FROM mart_contact_summary "
                    "WHERE top_topic = ?",
                    [topic],
                )
                contact_names = {
                    str(r.get("contact_name") or "") for r in matching
                }
            except Exception:  # noqa: BLE001
                contact_names = set()
            replies = [
                r for r in replies
                if str(r.get("contact_name", "")) in contact_names
            ]

        # Task categories (personal/life/work) differ from reply
        # domains (personal/family/social/work/health).
        _task_domain_filter = {
            "work": ("work",),
            "personal": ("personal", "life"),
            "health": ("health",),
        }
        task_domain = (
            _task_domain_filter.get(domain) if domain else None
        )

        # Tasks due/overdue/scheduled today, with goal metadata.
        try:
            tasks = get_actionable_tasks_today(layer.duckdb, today_iso)
        except Exception:  # noqa: BLE001
            tasks = []
        if task_domain:
            tasks = [
                t for t in tasks
                if t.get("category") in task_domain
            ]

        # Active habits due today, with goal metadata.
        try:
            habits = get_habits_today(layer.duckdb, today_iso)
        except Exception:  # noqa: BLE001
            habits = []
        if task_domain:
            habits = [
                h for h in habits
                if h.get("category") in task_domain
            ]

        topics: list[dict[str, Any]] = []
        if domain is None and topic is None:
            try:
                rows = layer.duckdb.query(
                    "SELECT contact_name, top_topic, "
                    "max_topic_importance, last_message_at "
                    "FROM mart_contact_summary "
                    "WHERE topic_count > 0 AND notification_priority >= 20 "
                    "ORDER BY notification_priority DESC LIMIT 20",
                )
            except Exception:  # noqa: BLE001
                rows = []
            for row in rows:
                topic_name = row.get("top_topic") or ""
                if not topic_name:
                    continue
                topics.append({
                    "topic_id": str(topic_name),
                    "title": str(topic_name),
                    "importance": int(
                        row.get("max_topic_importance") or 0,
                    ),
                    "last_activity": row.get("last_message_at"),
                    "contact_name": row.get("contact_name"),
                })

        print(_json_output({
            "replies": replies,
            "tasks": tasks,
            "habits": habits,
            "topics": topics,
        }))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_get_suggested_actions(layer: DataLayer, limit: int = 3) -> int:
    """Return Command-Bar suggestion chips derived from current state.

    Deterministic template generation — no LLM call. Sources from top
    threads + nearest actionable event + most-urgent pending reply.

    Output schema::

        {"chips": [{"label": str, "prefilled_prompt": str}]}

    sensitivity_tier: 2
    """
    try:
        from dataclasses import asdict

        from src.agents.proactive import ProactiveIntelligence

        chips: list[dict[str, str]] = []

        # Top thread (highest priority contact with a topic)
        try:
            top = layer.duckdb.query(
                "SELECT contact_name, top_topic FROM mart_contact_summary "
                "WHERE topic_count > 0 AND notification_priority >= 20 "
                "ORDER BY notification_priority DESC LIMIT 1",
            )
            if top:
                row = top[0]
                topic = row.get("top_topic") or row.get("contact_name")
                chips.append({
                    "label": f"Catch me up on {topic}",
                    "prefilled_prompt": (
                        f"Catch me up on {topic} with "
                        f"{row.get('contact_name')}."
                    ),
                })
        except Exception:  # noqa: BLE001
            pass

        # Most-urgent pending reply
        try:
            pi = ProactiveIntelligence(db_engine=layer.duckdb)
            replies = pi.get_pending_replies()
            if replies:
                top_reply = asdict(replies[0])
                chips.append({
                    "label": f"Draft a reply to {top_reply['contact_name']}",
                    "prefilled_prompt": (
                        f"Draft a reply to {top_reply['contact_name']} "
                        f"about: {top_reply.get('preview', '')[:120]}"
                    ),
                })
        except Exception:  # noqa: BLE001
            pass

        # Today's first event
        try:
            from src.core.calendar_filters import personal_events_for_date

            events = personal_events_for_date(
                layer.duckdb,
                date.today().isoformat(),
                columns="title",
                limit=1,
            )
            if events:
                title = events[0].get("title", "")
                chips.append({
                    "label": f"Brief me on {title}",
                    "prefilled_prompt": (
                        f"Brief me on the {title} meeting today — "
                        "who, why, what's the context."
                    ),
                })
        except Exception:  # noqa: BLE001
            pass

        # Fallback if everything was empty
        if not chips:
            chips = [
                {
                    "label": "What's on today?",
                    "prefilled_prompt": "What's on my calendar today?",
                },
                {
                    "label": "Catch me up",
                    "prefilled_prompt": (
                        "Catch me up on what happened in the last day."
                    ),
                },
            ]

        print(_json_output({"chips": chips[:limit]}))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_process_whatsapp_replies(layer: DataLayer) -> int:
    """Manually trigger processing of WhatsApp self-chat replies.

    Finds new user messages in the self-chat and routes them to
    Brain v2 or handles STOP opt-out commands.

    Args:
        layer: An open DataLayer instance.

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 3
    """
    try:
        from src.agents.brain import BrainAgentV2
        from src.agents.tool_registry import ToolRegistry
        from src.core.query_engine import QueryEngine
        from src.extensions.bridges.whatsapp.paths import resolve_self_jid
        from src.extensions.connectors.catalog import ConnectorCatalog
        from src.extensions.connectors.registry import ExtensionRegistry
        from src.models.llm_provider import create_provider_from_settings
        from src.notifications.reply_handler import ReplyHandler

        phone = _read_whatsapp_phone()
        if not phone:
            print(_json_output({"status": "not_configured", "processed": 0}))
            return 0

        self_jid = resolve_self_jid() or phone.lstrip("+")
        qe = QueryEngine(
            duckdb=layer.duckdb,
            kuzu=layer.kuzu,
            chromadb=layer.chromadb,
        )
        provider = create_provider_from_settings(background=True)
        tool_registry = ToolRegistry(
            catalog=ConnectorCatalog(),
            registry=ExtensionRegistry(),
        )
        brain = BrainAgentV2(
            query_engine=qe,
            provider=provider,
            tool_registry=tool_registry,
        )
        handler = ReplyHandler(
            db_engine=layer.duckdb, brain_agent=brain, phone=phone,
            self_jid=self_jid,
        )
        count = handler.process_new_replies()
        print(_json_output({"status": "ok", "processed": count}))
        return 0
    except Exception as exc:
        print(
            _json_output({"error": str(exc)}),
            file=sys.stderr,
        )
        return 1


class _ProactiveSenderNotifier:
    """Stream per-sender notifications as LLM results arrive.

    Instead of waiting for all senders to complete and building one
    unified digest, each sender's evaluation fires an immediate
    notification (native macOS by default, WhatsApp additionally when
    configured) with their actionable items.

    sensitivity_tier: 2
    """

    _IMPORTANCE_THRESHOLD = 6

    def __init__(self, db: Any) -> None:
        # WhatsApp phone from settings.json — optional; native
        # delivery doesn't need it.
        self._phone = _read_whatsapp_phone()
        self._db = db
        self._sent_count = 0

        from src.notifications.preference_service import PreferenceService

        self._prefs: Any = PreferenceService(db_engine=db)

    def on_sender_result(
        self,
        sender_name: str,
        llm_results: list[dict[str, Any]],
        raw_candidates: list[dict[str, Any]],
    ) -> None:
        """Callback fired per sender after LLM evaluation.

        Sends a notification immediately if any result has importance
        >= threshold.

        sensitivity_tier: 2
        """
        if self._prefs.is_muted_globally():
            return

        # Filter to actionable items
        actionable = [
            r for r in llm_results
            if r.get("importance", 0) >= self._IMPORTANCE_THRESHOLD
        ]
        if not actionable:
            logger.info(
                "Proactive sender [%s]: below threshold, skipping",
                sender_name,
            )
            return

        message = _format_sender_notification(
            sender_name, actionable, raw_candidates,
        )
        if not message:
            return

        _send_proactive_notification(
            self._prefs, self._phone,
            category="important_people",
            source_id=f"sender:{sender_name}",
            message=message,
        )
        self._sent_count += 1


def _format_sender_notification(
    sender_name: str,
    actionable: list[dict[str, Any]],
    raw_candidates: list[dict[str, Any]],
) -> str:
    """Format a per-sender notification message.

    Short, actionable, no LLM call needed — the LLM already
    provided the reason and domain during evaluation.

    sensitivity_tier: 2
    """
    lines = [f"🧠 {sender_name}"]

    for item in actionable[:3]:
        reason = item.get("reason", "")
        domain = item.get("domain", "")
        importance = item.get("importance", 0)

        if reason:
            lines.append(f"  → {reason}")
        if domain:
            lines.append(f"  📎 {domain} · importance {importance}")

        # Add message preview for context
        msg_id = str(item.get("message_id", ""))
        original = next(
            (c for c in raw_candidates
             if str(c.get("id", "")) == msg_id),
            None,
        )
        if original:
            preview = safe_str(
                original.get("content")
                or original.get("body_preview")
                or original.get("subject", ""),
                120,
            )
            if preview:
                lines.append(f'  "{preview}"')

    return "\n".join(lines)


def _send_proactive_notification(
    prefs: Any,
    phone: str | None,
    *,
    category: str,
    source_id: str,
    message: str,
) -> None:
    """Send a proactive notification over every configured channel.

    Native macOS notifications are the default channel; WhatsApp
    (via the running listener IPC — no extra MCP server spawn) is
    additional when ``phone`` is configured. Dedup is per source_id
    per 2h window (allows multiple senders).

    sensitivity_tier: 2
    """
    from src.notifications.models import NotificationRecord
    from src.notifications.notifier import deliver_notification, get_opt_out_text

    if not prefs.is_category_enabled(category):
        return

    # Dedup: one notification per source_id per 2h window
    now = datetime.now(tz=timezone.utc)
    window = f"{now.strftime('%Y-%m-%d')}T{now.hour // 2 * 2:02d}"
    dedupe_key = hashlib.sha256(
        f"proactive:{source_id}:{window}".encode(),
    ).hexdigest()[:16]

    if prefs.has_recent_dedup(dedupe_key):
        logger.info(
            "Proactive notify [%s]: deduped", source_id,
        )
        return

    opt_out = get_opt_out_text(category)

    logger.info(
        "Proactive notify [%s]: delivering (%d chars)",
        source_id, len(message),
    )
    # Raw message: deliver_notification -> WhatsAppNotifier.send()
    # appends the opt-out text itself. Pre-appending it here (as the
    # old direct-listener call required) would double it up on
    # WhatsApp and put WhatsApp-specific opt-out text in the native
    # macOS banner.
    delivery = deliver_notification(message, category, whatsapp_phone=phone)

    logger.info(
        "Proactive notify [%s]: status=%s error=%s",
        source_id, delivery.status, delivery.error,
    )

    if delivery.status not in ("sent", "failed"):
        # not_configured — no channel available to have attempted.
        return

    prefs.log_notification(
        NotificationRecord(
            id=prefs.new_record_id(),
            dedupe_key=dedupe_key,
            category=category,
            importance_score=8,
            decision="send",
            delivery_status=delivery.status,
            message=message,
            opt_out_text=opt_out,
            source_type="proactive",
            source_id=source_id,
            error=delivery.error,
            created_at=utc_now_iso(),
            message_id=delivery.message_id,
        ),
    )


def _maybe_notify_events_and_contexts(
    db: Any,
    result: dict[str, Any],
) -> None:
    """Send a short notification for events/birthdays/contexts.

    Called after all sender notifications have streamed.
    No LLM call — just structured formatting.

    Non-fatal: callers wrap this in try/except.

    sensitivity_tier: 2
    """
    # WhatsApp phone from settings.json — optional; native delivery
    # doesn't need it.
    phone = _read_whatsapp_phone()

    from src.notifications.preference_service import PreferenceService

    prefs = PreferenceService(db_engine=db)
    if prefs.is_muted_globally():
        return

    events = result.get("actionable_events", [])
    contacts = [
        c for c in result.get("contact_contexts", [])
        if c.get("active_context")
    ]

    if not events and not contacts:
        return

    lines = ["🧠 Events & Context"]

    for e in events[:5]:
        title = e.get("title", "Event")
        action = e.get("action_needed", "")
        name = e.get("contact_name", "")
        event_type = e.get("event_type", "")
        if event_type == "birthday" and name:
            lines.append(
                f"  🎂 {name}: {action or 'Send wishes'}",
            )
        else:
            label = f"  📅 {title}"
            if action:
                label += f" — {action}"
            lines.append(label)

    for c in contacts[:5]:
        name = c.get("contact_name", "Unknown")
        ctx = c.get("active_context", "")
        if ctx:
            lines.append(f"  👤 {name}: {ctx}")

    if len(lines) <= 1:
        return

    _send_proactive_notification(
        prefs, phone,
        category="important_people",
        source_id="events_and_contexts",
        message="\n".join(lines),
    )


# ---------------------------------------------------------------------------
# Notification commands (JSON output for Tauri bridge)
# ---------------------------------------------------------------------------


def cmd_notification_prefs_get(layer: DataLayer) -> int:
    """Return notification preferences as JSON.

    Args:
        layer: An open DataLayer instance.

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 1
    """
    try:
        from dataclasses import asdict

        from src.notifications.preference_service import (
            PreferenceService,
        )

        prefs = PreferenceService(db_engine=layer.duckdb)
        result = prefs.get_preferences()
        print(_json_output([asdict(p) for p in result]))
        return 0
    except Exception as exc:
        print(
            _json_output({"error": str(exc)}),
            file=sys.stderr,
        )
        return 1


def cmd_notification_prefs_set(
    layer: DataLayer,
    category: str,
    enabled: bool,
) -> int:
    """Update a notification preference.

    Args:
        layer: An open DataLayer instance.
        category: Notification category name.
        enabled: Whether to enable or disable.

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 1
    """
    try:
        from src.notifications.preference_service import (
            PreferenceService,
        )

        prefs = PreferenceService(db_engine=layer.duckdb)
        prefs.update_preference(category, enabled=enabled)
        print(_json_output({
            "status": "updated",
            "category": category,
            "enabled": enabled,
        }))
        return 0
    except Exception as exc:
        print(
            _json_output({"error": str(exc)}),
            file=sys.stderr,
        )
        return 1


def cmd_notification_prefs_mute_all(
    layer: DataLayer,
    until: str | None,
) -> int:
    """Mute all notifications.

    Args:
        layer: An open DataLayer instance.
        until: Optional ISO 8601 timestamp for mute expiration.

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 1
    """
    try:
        from src.notifications.preference_service import (
            PreferenceService,
        )

        prefs = PreferenceService(db_engine=layer.duckdb)
        prefs.mute_all(until=until)
        print(_json_output({"status": "muted"}))
        return 0
    except Exception as exc:
        print(
            _json_output({"error": str(exc)}),
            file=sys.stderr,
        )
        return 1


def cmd_notification_prefs_unmute_all(
    layer: DataLayer,
) -> int:
    """Unmute all notifications.

    Args:
        layer: An open DataLayer instance.

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 1
    """
    try:
        from src.notifications.preference_service import (
            PreferenceService,
        )

        prefs = PreferenceService(db_engine=layer.duckdb)
        prefs.unmute_all()
        print(_json_output({"status": "unmuted"}))
        return 0
    except Exception as exc:
        print(
            _json_output({"error": str(exc)}),
            file=sys.stderr,
        )
        return 1


def cmd_notification_log(
    layer: DataLayer,
    limit: int,
    offset: int,
) -> int:
    """Return paginated notification log as JSON.

    Args:
        layer: An open DataLayer instance.
        limit: Max records to return.
        offset: Pagination offset.

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 2
    """
    try:
        from dataclasses import asdict

        from src.notifications.preference_service import (
            PreferenceService,
        )

        prefs = PreferenceService(db_engine=layer.duckdb)
        records = prefs.get_notification_log(
            limit=limit, offset=offset,
        )
        print(_json_output([asdict(r) for r in records]))
        return 0
    except Exception as exc:
        print(
            _json_output({"error": str(exc)}),
            file=sys.stderr,
        )
        return 1


def cmd_infer_profile(layer: DataLayer) -> int:
    """Infer user profile from available data and merge into settings.

    Analyzes WhatsApp phone number (country code), contacts, and emails
    to auto-detect the user's name, location, timezone, and language.
    Only fills fields that are currently unset — never overwrites.

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 2
    """
    try:
        from src.core.user_context import infer_user_profile

        inferred = infer_user_profile(layer)
        if not inferred:
            print(_json_output({
                "inferred": {},
                "applied": {},
                "message": "No data available for inference",
            }))
            return 0

        # Load current settings, merge only None/empty fields
        settings_path = Path.home() / ".arandu" / "settings.json"
        current: dict[str, Any] = {}
        if settings_path.exists():
            try:
                current = json.loads(
                    settings_path.read_text(encoding="utf-8"),
                )
            except (json.JSONDecodeError, OSError):
                pass

        applied: dict[str, str] = {}
        for key, value in inferred.items():
            if not current.get(key):
                current[key] = value
                applied[key] = value

        if applied:
            settings_path.parent.mkdir(parents=True, exist_ok=True)
            settings_path.write_text(
                json.dumps(current, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        print(_json_output({
            "inferred": inferred,
            "applied": applied,
        }))
        return 0
    except Exception as exc:
        logger.exception("infer-profile failed")
        print(
            _json_output({"error": str(exc)}),
            file=sys.stderr,
        )
        return 1


# ------------------------------------------------------------------
# Learned facts commands
# ------------------------------------------------------------------


def cmd_get_learned_facts(
    layer: DataLayer,
    limit: int = 20,
    category: str | None = None,
) -> int:
    """Return active learned facts as JSON.

    sensitivity_tier: 2
    """
    try:
        from src.agents.fact_extractor import FactLearner

        learner = FactLearner(db_engine=layer.duckdb)
        facts = learner.get_active_facts(
            limit=limit, category=category,
        )
        print(_json_output([
            {
                "id": f.id,
                "category": f.category,
                "subject": f.subject,
                "predicate": f.predicate,
                "content": f.content,
                "confidence": f.confidence,
                "source_type": f.source_type,
                "extracted_at": f.extracted_at,
                "confirmed_at": f.confirmed_at,
                "sensitivity_tier": f.sensitivity_tier,
                "times_used": f.times_used,
            }
            for f in facts
        ]))
        return 0
    except Exception as exc:
        logger.exception("get-learned-facts failed")
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_get_facts_for_review(
    layer: DataLayer,
    limit: int = 50,
) -> int:
    """Return facts pending user review as JSON.

    sensitivity_tier: 2
    """
    try:
        from src.agents.fact_extractor import FactLearner

        learner = FactLearner(db_engine=layer.duckdb)
        facts = learner.get_facts_for_review(limit=limit)
        print(_json_output([
            {
                "id": f.id,
                "category": f.category,
                "subject": f.subject,
                "predicate": f.predicate,
                "content": f.content,
                "confidence": f.confidence,
                "source_type": f.source_type,
                "extracted_at": f.extracted_at,
                "sensitivity_tier": f.sensitivity_tier,
                "times_used": f.times_used,
            }
            for f in facts
        ]))
        return 0
    except Exception as exc:
        logger.exception("get-facts-for-review failed")
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_get_fact_stats(layer: DataLayer) -> int:
    """Return fact statistics as JSON.

    sensitivity_tier: 1
    """
    try:
        from src.agents.fact_extractor import FactLearner

        learner = FactLearner(db_engine=layer.duckdb)
        stats = learner.get_fact_count()
        print(_json_output(stats))
        return 0
    except Exception as exc:
        logger.exception("get-fact-stats failed")
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_confirm_fact(layer: DataLayer, fact_id: str) -> int:
    """Confirm a learned fact.

    sensitivity_tier: 2
    """
    try:
        from src.agents.fact_extractor import FactLearner

        learner = FactLearner(db_engine=layer.duckdb)
        learner.confirm_fact(fact_id)
        print(_json_output({"status": "confirmed", "id": fact_id}))
        return 0
    except Exception as exc:
        logger.exception("confirm-fact failed")
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_dismiss_fact(layer: DataLayer, fact_id: str) -> int:
    """Dismiss a learned fact.

    sensitivity_tier: 1
    """
    try:
        from src.agents.fact_extractor import FactLearner

        learner = FactLearner(db_engine=layer.duckdb)
        learner.dismiss_fact(fact_id)
        print(_json_output({"status": "dismissed", "id": fact_id}))
        return 0
    except Exception as exc:
        logger.exception("dismiss-fact failed")
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_edit_fact(
    layer: DataLayer, fact_id: str, content: str,
) -> int:
    """Edit a learned fact's content.

    sensitivity_tier: 2
    """
    try:
        from src.agents.fact_extractor import FactLearner

        learner = FactLearner(db_engine=layer.duckdb)
        learner.edit_fact(fact_id, content)
        print(_json_output({"status": "edited", "id": fact_id}))
        return 0
    except Exception as exc:
        logger.exception("edit-fact failed")
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_transcribe_audio(
    audio_input: str,
    model_size: str | None = None,
    language: str | None = None,
) -> int:
    """Transcribe audio to text using local ASR model.

    Uses Qwen3-ASR (MLX) as primary backend, with faster-whisper as
    fallback. ``audio_input`` is either a file path or base64-encoded
    audio data. When base64, the data is decoded to a temp file first.

    Does NOT require a DataLayer — operates independently.

    sensitivity_tier: 3
    """
    import base64

    from src.models.voice_transcriber import VoiceTranscriber, is_available

    if not is_available():
        print(
            _json_output({
                "error": "No ASR backend installed",
                "hint": (
                    "pip install 'arandu[voice]' (Qwen3-ASR, recommended) "
                    "or pip install 'arandu[voice-fallback]' (faster-whisper)"
                ),
            }),
            file=sys.stderr,
        )
        return 1

    # Determine model size from settings if not specified
    if model_size is None:
        settings_path = Path.home() / ".arandu" / "settings.json"
        if settings_path.exists():
            try:
                with open(settings_path) as f:
                    settings = json.load(f)
                model_size = settings.get("whisper_model_size", "base")
            except (json.JSONDecodeError, OSError):
                model_size = "base"
        else:
            model_size = "base"

    transcriber = VoiceTranscriber(model_size=model_size)

    try:
        # Check if input is a file path
        input_path = Path(audio_input)
        if input_path.exists():
            result = transcriber.transcribe(str(input_path), language=language)
        else:
            # Treat as base64-encoded audio
            try:
                audio_bytes = base64.b64decode(audio_input)
            except Exception:
                print(
                    _json_output({
                        "error": f"Invalid input: {audio_input[:40]}...",
                    }),
                    file=sys.stderr,
                )
                return 1
            result = transcriber.transcribe_bytes(audio_bytes, language=language)

        print(_json_output({
            "text": result.text,
            "language": result.language,
            "duration": result.duration,
            "segments": [
                {
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                }
                for s in result.segments
            ],
        }))
        return 0
    except Exception as exc:
        logger.exception("transcribe-audio failed")
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def _reindex_kuzu_from_layer(layer: DataLayer) -> dict[str, int]:
    """Re-index Kuzu graph from DuckDB raw tables.

    Full reindex when graph is empty, otherwise incremental.

    Non-fatal: caller should wrap in try/except.

    sensitivity_tier: 2
    """
    from src.core.kuzu.indexer import GraphIndexer
    from src.core.kuzu.schema import create_schema

    create_schema(layer.kuzu)
    indexer = GraphIndexer(duckdb=layer.duckdb, kuzu=layer.kuzu)

    rows = layer.kuzu.query("MATCH (n) RETURN count(n) AS cnt")
    node_count = rows[0]["cnt"] if rows else 0

    if node_count == 0:
        return indexer.full_reindex()
    return indexer.incremental_index()


def _run_smart_pipeline_and_reindex(
    layer: DataLayer,
    trigger: str,
) -> dict[str, Any]:
    """Run a smart pipeline refresh, then re-index ChromaDB and Kuzu.

    Falls back to a full pipeline run if smart planning raises.

    sensitivity_tier: 2
    """
    from dataclasses import asdict
    from datetime import datetime

    from src.core.query_tracker import QueryTracker
    from src.pipeline.pipeline_brain import PipelineBrain
    from src.pipeline.runner import PipelineRunner
    from src.pipeline.stats import ProcessingStats

    try:
        stats = ProcessingStats()
        runner = PipelineRunner(duckdb=layer.duckdb, stats=stats)
        tracker = QueryTracker(db_engine=layer.duckdb)
        brain = PipelineBrain(
            query_tracker=tracker,
            pipeline_runner=runner,
        )

        # Opportunistically stage new marts for high-demand domains.
        brain.check_demand_for_new_marts()

        plan = brain.plan_refresh()
        selected = plan.get_ordered()
        if not selected:
            return {
                "status": "nothing_to_do",
                "trigger": trigger,
                "plan_summary": plan.summary(),
                "models_processed": [],
            }

        run = runner.run(
            trigger=trigger,
            select_models=selected,
        )
        run.plan_summary = plan.summary()
        result = asdict(run)
        result["started_at"] = run.started_at.isoformat()
        result["completed_at"] = run.completed_at.isoformat()

        if result["status"] == "success":
            try:
                from src.core.chromadb.engine import COLLECTION_NAMES

                total_docs = sum(
                    layer.chromadb.get_collection_count(name)
                    for name in COLLECTION_NAMES
                )
                if total_docs == 0:
                    counts = layer.indexer.full_reindex()
                else:
                    since = datetime.fromisoformat(result["started_at"])
                    counts = layer.indexer.incremental_index(since=since)
                result["reindex_counts"] = counts
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Post-pipeline re-index failed: %s",
                    exc,
                )
                result["reindex_error"] = str(exc)
            try:
                _reindex_kuzu_from_layer(layer)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Post-pipeline Kuzu re-index failed: %s", exc)
        return result
    except Exception:  # noqa: BLE001
        logger.warning(
            "Smart pipeline run failed, falling back to full run",
            exc_info=True,
        )
        return layer.run_pipeline_and_reindex(trigger=trigger)


def _sync_single_connector(
    layer: DataLayer,
    connector_id: str,
) -> int:
    """Sync one connector using the DataLayer's existing connection.

    Reuses the parent DataLayer's DatabaseEngine to avoid opening a
    second SQLite write connection in the same process, which causes
    self-deadlock under WAL mode.

    After syncing message-bearing connectors with new rows, triggers
    real-time message evaluation for proactive notifications.

    Returns the number of rows synced.

    sensitivity_tier: 2
    """
    from src.extensions.connectors.connection_manager import ConnectionManager

    manager = ConnectionManager(db_engine=layer.duckdb)
    stats = manager.sync_now(connector_id)
    rows = stats.rows_synced if stats.status == "success" else 0

    # Real-time message evaluation for message-bearing connectors
    if rows > 0 and _is_message_connector(connector_id):
        try:
            _maybe_evaluate_new_messages(layer, connector_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Real-time message evaluation failed for %s: %s",
                connector_id, exc,
            )

    return rows


def _is_message_connector(connector_id: str) -> bool:
    """Check if a connector produces messages or emails.

    sensitivity_tier: 1
    """
    from src.agents.message_eval import MESSAGE_CONNECTORS

    return connector_id in MESSAGE_CONNECTORS


def _maybe_evaluate_new_messages(
    layer: DataLayer,
    connector_id: str,
) -> None:
    """Evaluate newly ingested messages for proactive notifications.

    Non-fatal: callers wrap this in try/except.

    sensitivity_tier: 3
    """
    from src.agents.message_eval import (
        _CONNECTOR_TABLES,
        MessageEvaluator,
        format_realtime_notification,
    )

    target_tables = _CONNECTOR_TABLES.get(connector_id, [])

    # Task curator post-sync hook runs independently of the
    # notification pipeline — it writes to _tasks/_goals and doesn't
    # need a configured phone or unmuted prefs.
    try:
        _maybe_curate_tasks(layer, target_tables)
    except Exception:  # noqa: BLE001
        logger.warning(
            "Task curator post-sync pass failed",
            exc_info=True,
        )

    # WhatsApp phone from settings.json — optional; native delivery
    # doesn't need it.
    phone = _read_whatsapp_phone()

    from src.notifications.preference_service import PreferenceService

    prefs = PreferenceService(db_engine=layer.duckdb)
    if prefs.is_muted_globally():
        logger.info("Message eval: globally muted")
        return

    evaluator = MessageEvaluator(db_engine=layer.duckdb)

    all_notifications = []

    for table in target_tables:
        results = evaluator.evaluate_new_messages(
            connector_id, table,
        )
        all_notifications.extend(results)

    if not all_notifications:
        return

    # Group by notification type and send
    actions = [
        n for n in all_notifications
        if n.notification_type == "action"
    ]
    awareness = [
        n for n in all_notifications
        if n.notification_type == "awareness"
    ]

    if actions:
        msg = format_realtime_notification(actions)
        _send_proactive_notification(
            prefs, phone,
            category="realtime_action",
            source_id=actions[0].id,
            message=msg,
        )

    if awareness:
        msg = format_realtime_notification(awareness)
        _send_proactive_notification(
            prefs, phone,
            category="realtime_awareness",
            source_id=awareness[0].id,
            message=msg,
        )


def _maybe_curate_tasks(
    layer: DataLayer,
    target_tables: list[str],
    lookback_minutes: int = 60,
) -> None:
    """Run the task curator on freshly ingested message rows.

    Pulls the most recent rows from each ``target_table`` (last
    ``lookback_minutes`` minutes), feeds them to the proposer (which
    may insert new ``_tasks``) and the completion detector (which may
    close open ``_tasks``).

    sensitivity_tier: 3
    """
    from src.agents.tasks import TaskCurator

    curator = TaskCurator(db_engine=layer.duckdb)
    recent: list[dict[str, Any]] = []
    # ISO-T columns: bind a Python cutoff. SQLite's datetime('now', …)
    # compares as a space-separated string and 'T' > ' ' admits the
    # whole UTC day instead of the lookback window.
    cutoff = utc_ago_iso(minutes=int(lookback_minutes))
    for table in target_tables:
        try:
            rows = layer.duckdb.query(
                "SELECT id, source, sender, content, timestamp "
                f"FROM {table} "  # noqa: S608
                "WHERE timestamp >= ? "
                "ORDER BY timestamp DESC LIMIT 50",
                [cutoff],
            )
        except Exception:  # noqa: BLE001
            continue
        recent.extend(dict(r) for r in rows)
    if not recent:
        return

    proposed = curator.propose_from_messages(recent)
    if proposed:
        logger.info(
            "Task curator: %d new tasks proposed from sync",
            len(proposed),
        )
    closed = curator.detect_completions(recent)
    if closed:
        logger.info(
            "Task curator: %d open tasks auto-closed from sync",
            len(closed),
        )


def cmd_startup_sync(layer: DataLayer) -> int:
    """Sync stale connectors and refresh pipeline on app launch.

    Each connector sync opens and closes its own DuckDB connection to
    avoid holding the write lock for the entire duration, which would
    block user-initiated commands (toggle, chat, etc.).

    Args:
        layer: An open DataLayer instance.

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 2
    """
    try:
        from src.extensions.connectors.registry import ExtensionRegistry

        registry = ExtensionRegistry()
        enabled = registry.get_enabled()

        synced_connectors = 0
        total_rows = 0
        errors: list[str] = []

        for ext in enabled:
            try:
                rows = _sync_single_connector(layer, ext.connector_id)
                if rows > 0:
                    synced_connectors += 1
                    total_rows += rows
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Startup sync failed for %s: %s",
                    ext.connector_id,
                    exc,
                )
                errors.append(f"{ext.connector_id}: {exc}")

        # Run pipeline + reindex if data is stale after syncs.
        # Reuse the existing DataLayer — opening a second one causes
        # SQLite self-deadlock (two write connections in the same process).
        pipeline_ran = False
        from src.pipeline.runner import PipelineRunner
        from src.pipeline.worker import PIPELINE_LOCK_PATH

        runner = PipelineRunner(duckdb=layer.duckdb)
        if runner.is_stale() and not PIPELINE_LOCK_PATH.exists():
            result = _run_smart_pipeline_and_reindex(
                layer, trigger="startup",
            )
            pipeline_ran = result["status"] == "success"

        # Belt-and-suspenders: if the user set an ingest cutoff (typically
        # right after fresh_restart.sh), mark every row dated before the
        # cutoff as already-evaluated so the MessageEvaluator doesn't burn
        # LLM tokens on a connector's first-sync backfill.
        seeded = 0
        try:
            from src.models.llm_provider import load_llm_settings

            cutoff = load_llm_settings().get("ingest_cutoff_iso")
            if cutoff:
                seeded = layer.seed_evaluated_messages_pre_cutoff(cutoff)
        except Exception:  # noqa: BLE001
            logger.exception("startup-sync: cutoff seeding failed")

        print(_json_output({
            "synced_connectors": synced_connectors,
            "total_rows": total_rows,
            "pipeline_ran": pipeline_ran,
            "pre_cutoff_rows_seeded": seeded,
            "errors": errors,
        }))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_sync_all_stale(layer: DataLayer) -> int:
    """Sync all enabled connectors and re-index if new data arrived.

    Intended for periodic background use (called by Rust-side timer).
    Each connector sync opens and closes its own DuckDB connection to
    avoid holding the write lock for the entire duration.

    Args:
        layer: An open DataLayer instance (read-only).

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 2
    """
    try:
        from src.extensions.connectors.registry import ExtensionRegistry

        registry = ExtensionRegistry()

        total_rows = 0
        results: list[dict[str, Any]] = []

        for ext in registry.get_enabled():
            try:
                rows = _sync_single_connector(layer, ext.connector_id)
                results.append({
                    "connector_id": ext.connector_id,
                    "status": "success",
                    "rows_synced": rows,
                })
                total_rows += rows
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Periodic sync failed for %s: %s",
                    ext.connector_id,
                    exc,
                )
                results.append({
                    "connector_id": ext.connector_id,
                    "status": "error",
                    "rows_synced": 0,
                    "error": str(exc),
                })

        # Run pipeline + reindex whenever the pipeline is stale.
        # Reuse the existing DataLayer — opening a second one causes
        # SQLite self-deadlock (two write connections in the same process).
        pipeline_ran = False
        from src.pipeline.runner import PipelineRunner
        from src.pipeline.worker import PIPELINE_LOCK_PATH as _PL

        runner = PipelineRunner(duckdb=layer.duckdb)
        if runner.is_stale() and not _PL.exists():
            result = _run_smart_pipeline_and_reindex(
                layer, trigger="periodic",
            )
            pipeline_ran = result["status"] == "success"

        print(_json_output({
            "connectors": results,
            "total_rows": total_rows,
            "pipeline_ran": pipeline_ran,
        }))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_evaluate_messages(
    layer: DataLayer,
    connector_id: str,
) -> int:
    """Evaluate newly ingested messages for a connector.

    Triggers real-time message evaluation and sends notifications
    for high-importance items.

    Args:
        layer: An open DataLayer instance.
        connector_id: The connector to evaluate messages for.

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 3
    """
    try:
        from src.agents.message_eval import (
            _CONNECTOR_TABLES,
            MessageEvaluator,
            format_realtime_notification,
        )

        evaluator = MessageEvaluator(db_engine=layer.duckdb)

        target_tables = _CONNECTOR_TABLES.get(connector_id, [])
        all_notifications = []

        for table in target_tables:
            results = evaluator.evaluate_new_messages(
                connector_id, table,
            )
            all_notifications.extend(results)

        # Notify over every configured channel (native macOS by
        # default; WhatsApp additionally when a phone is configured).
        if all_notifications:
            # Optional; native delivery doesn't need it.
            phone = _read_whatsapp_phone()
            from src.notifications.preference_service import (
                PreferenceService,
            )

            prefs = PreferenceService(db_engine=layer.duckdb)
            actions = [
                n for n in all_notifications
                if n.notification_type == "action"
            ]
            awareness = [
                n for n in all_notifications
                if n.notification_type == "awareness"
            ]

            if actions:
                _send_proactive_notification(
                    prefs, phone,
                    category="realtime_action",
                    source_id=actions[0].id,
                    message=format_realtime_notification(
                        actions,
                    ),
                )
            if awareness:
                _send_proactive_notification(
                    prefs, phone,
                    category="realtime_awareness",
                    source_id=awareness[0].id,
                    message=format_realtime_notification(
                        awareness,
                    ),
                )

        print(_json_output({
            "connector_id": connector_id,
            "notifications": len(all_notifications),
            "tables_evaluated": target_tables,
        }))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


def cmd_extension_logs(
    extension_id: str,
    lines: int = 50,
) -> int:
    """Get recent log lines for an extension.

    Args:
        extension_id: The extension to query.
        lines: Number of lines to return.

    Returns:
        Exit code (0 = success, 1 = failure).

    sensitivity_tier: 1
    """
    try:
        log_file = (
            Path.home()
            / ".arandu"
            / "data"
            / "logs"
            / f"{extension_id}.log"
        )
        if not log_file.exists():
            print(_json_output({
                "extension_id": extension_id,
                "lines": [],
            }))
            return 0

        all_lines = log_file.read_text().splitlines()
        tail = all_lines[-lines:] if len(all_lines) > lines else all_lines
        print(_json_output({
            "extension_id": extension_id,
            "lines": tail,
        }))
        return 0
    except Exception as exc:
        print(_json_output({"error": str(exc)}), file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src.core.cli",
        description="Arandu data layer management CLI",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help=("Override the data directory (default: ~/.arandu/data/)"),
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    subparsers.add_parser("init", help="Initialize database schemas")
    subparsers.add_parser("status", help="Show health and row/node/doc counts")
    subparsers.add_parser("reset", help="Wipe all data and reinitialize")

    # JSON output commands (for Tauri bridge)
    subparsers.add_parser("stats", help="Output database stats as JSON")

    export_parser = subparsers.add_parser(
        "export-data",
        help="Export all local user data to a JSON zip archive",
    )
    export_parser.add_argument(
        "--output-dir",
        default=None,
        metavar="PATH",
        help="Directory to write the archive to (default: ~/Downloads)",
    )
    subparsers.add_parser(
        "delete-all-data",
        help="Permanently delete all local user data (keeps settings)",
    )

    msg_parser = subparsers.add_parser(
        "query-messages",
        help="Query recent messages (JSON)",
    )
    msg_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum messages to return (default: 20)",
    )
    msg_parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Number of rows to skip (default: 0)",
    )

    events_parser = subparsers.add_parser(
        "query-events",
        help="Query upcoming events (JSON)",
    )
    events_parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Days ahead to query (default: 7)",
    )
    events_parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum events to return (default: 50)",
    )
    events_parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Number of rows to skip (default: 0)",
    )

    contacts_parser = subparsers.add_parser(
        "query-contacts",
        help="Query contacts (JSON)",
    )
    contacts_parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Maximum contacts to return (default: 500)",
    )
    contacts_parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Number of rows to skip (default: 0)",
    )

    notes_parser = subparsers.add_parser(
        "query-notes",
        help="Query notes (JSON)",
    )
    notes_parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum notes to return (default: 100)",
    )
    notes_parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Number of rows to skip (default: 0)",
    )

    emails_parser = subparsers.add_parser(
        "query-emails",
        help="Query emails (JSON)",
    )
    emails_parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Maximum emails to return (default: 200)",
    )
    emails_parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Number of rows to skip (default: 0)",
    )

    subparsers.add_parser(
        "fix-notes-content",
        help="Re-extract notes content from macOS Notes DB (fixes charset)",
    )

    # Generic table browsing commands
    list_tables_parser = subparsers.add_parser(
        "list-tables",
        help="List DuckDB tables with row counts and column info (JSON)",
    )
    list_tables_parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Filter tables by prefix (e.g. 'raw_', 'mart_')",
    )

    subparsers.add_parser(
        "list-pipeline-models",
        help="List all models registered in the pipeline manifest (JSON)",
    )

    query_table_parser = subparsers.add_parser(
        "query-table",
        help="Query sample rows from any whitelisted DuckDB table (JSON)",
    )
    query_table_parser.add_argument(
        "--table",
        type=str,
        required=True,
        help="Table name to query",
    )
    query_table_parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Maximum rows to return (default: 25, max: 100)",
    )
    query_table_parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Number of rows to skip (default: 0)",
    )

    # Graph exploration commands
    subparsers.add_parser(
        "graph-summary",
        help="Node and relationship type counts from Kuzu graph (JSON)",
    )

    graph_nodes_parser = subparsers.add_parser(
        "query-graph-nodes",
        help="Sample nodes of a given type from Kuzu graph (JSON)",
    )
    graph_nodes_parser.add_argument(
        "--type",
        type=str,
        required=True,
        dest="node_type",
        help="Node type to query (e.g. Person, Event)",
    )
    graph_nodes_parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Maximum nodes to return (default: 25, max: 100)",
    )

    graph_rels_parser = subparsers.add_parser(
        "query-graph-rels",
        help="Sample relationships of a given type from Kuzu graph (JSON)",
    )
    graph_rels_parser.add_argument(
        "--type",
        type=str,
        required=True,
        dest="rel_type",
        help="Relationship type to query (e.g. KNOWS, PARTICIPATED_IN)",
    )
    graph_rels_parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Maximum relationships to return (default: 25, max: 100)",
    )

    # Vector exploration commands
    subparsers.add_parser(
        "vector-summary",
        help="ChromaDB collection counts with sample docs (JSON)",
    )

    subparsers.add_parser("query-today", help="Today's summary (JSON)")

    subparsers.add_parser(
        "profile",
        help="Run performance benchmark and print report",
    )

    ask_parser = subparsers.add_parser(
        "ask",
        help="Ask Brain Agent a question (JSON)",
    )
    ask_parser.add_argument(
        "question",
        type=str,
        help="The question to ask",
    )

    ask_stream_parser = subparsers.add_parser(
        "ask-stream",
        help="Ask Brain Agent with streaming JSON-lines output",
    )
    ask_stream_parser.add_argument(
        "question",
        type=str,
        help="The question to ask",
    )
    ask_stream_parser.add_argument(
        "--agent-id",
        default="brain",
        help=(
            "Target agent id. Defaults to 'brain'. Pass a 'user.<slug>' "
            "id to route the question to a user-authored agent."
        ),
    )
    ask_stream_parser.add_argument(
        "--session-id",
        default=None,
        help=(
            "Chat session id to persist this exchange under. If omitted, "
            "no persistence happens (callers that want a stateless ask "
            "should leave this unset)."
        ),
    )
    ask_stream_parser.add_argument(
        "--reply-context",
        default=None,
        help=(
            "JSON {source, message_id, contact_name} identifying the "
            "inbound message this ask originated from (a 'Draft reply' "
            "click). Hard-locks the proposed action channel and seeds "
            "the original message into context."
        ),
    )
    ask_stream_parser.add_argument(
        "--task-context",
        default=None,
        help=(
            "JSON {task_id, goal_id} identifying the task the user "
            "clicked 'Work on this' for. Seeds the task + goal details "
            "into context so the assistant can help complete it."
        ),
    )
    ask_stream_parser.add_argument(
        "--budget",
        default=None,
        choices=("interactive_fast", "interactive_deep", "background_deep"),
        help=(
            "Wall-clock budget class for the reflective runner. "
            "Defaults to 'interactive_fast'. Background callers like "
            "the daily brief pass 'background_deep'."
        ),
    )

    stop_research_parser = subparsers.add_parser(
        "stop-research",
        help=(
            "Signal an in-flight ask-stream run to stop researching "
            "and finalize its answer with the current context."
        ),
    )
    stop_research_parser.add_argument(
        "run_id",
        type=str,
        help="Short hex run id from the 'run_started' stream chunk.",
    )
    ask_parser.add_argument(
        "--session-id",
        default=None,
        help="Chat session id to persist this exchange under.",
    )
    ask_parser.add_argument(
        "--reply-context",
        default=None,
        help=(
            "JSON {source, message_id, contact_name} identifying the "
            "inbound message this ask originated from."
        ),
    )

    chat_session_create_parser = subparsers.add_parser(
        "chat-session-create",
        help="Create a new persistent chat session (JSON)",
    )
    chat_session_create_parser.add_argument(
        "--title",
        default=None,
        help="Optional initial title (defaults to 'New chat')",
    )

    chat_session_list_parser = subparsers.add_parser(
        "chat-session-list",
        help="List recent chat session summaries (JSON)",
    )
    chat_session_list_parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of sessions to return",
    )

    chat_session_load_parser = subparsers.add_parser(
        "chat-session-load",
        help="Load all messages for a chat session (JSON)",
    )
    chat_session_load_parser.add_argument(
        "session_id",
        type=str,
        help="The session id to load",
    )

    chat_session_delete_parser = subparsers.add_parser(
        "chat-session-delete",
        help="Delete a chat session and its messages",
    )
    chat_session_delete_parser.add_argument(
        "session_id",
        type=str,
        help="The session id to delete",
    )

    subparsers.add_parser(
        "ollama-status",
        help="Ollama server and model status (JSON)",
    )
    subparsers.add_parser(
        "ollama-preload",
        help="Preload default chat model into memory",
    )
    subparsers.add_parser(
        "ollama-stop",
        help="Stop the Ollama server",
    )
    subparsers.add_parser(
        "monitor",
        help="Memory usage and database file sizes (JSON)",
    )

    rebuild_idx_parser = subparsers.add_parser(
        "rebuild-vector-index",
        help=(
            "Rebuild ChromaDB + BM25 under a new embedding model (JSON). "
            "Wraps src.core.chromadb.migrate for the Settings UI."
        ),
    )
    rebuild_idx_parser.add_argument(
        "--to-model", required=True,
        help="Target embedding model (e.g. bge-m3, text-embedding-3-large).",
    )
    rebuild_idx_parser.add_argument(
        "--provider", choices=("ollama", "openai"), default="ollama",
    )
    rebuild_idx_parser.add_argument("--api-key", default=None)
    rebuild_idx_parser.add_argument("--base-url", default=None)
    rebuild_idx_parser.add_argument(
        "--dimensions", type=int, default=None,
    )
    rebuild_idx_parser.add_argument(
        "--dry-run", action="store_true",
        help="Estimate cost without dropping or rebuilding.",
    )

    subparsers.add_parser(
        "pipeline-status",
        help="Pipeline status: last run, staleness, pending changes (JSON)",
    )

    pipeline_run_parser = subparsers.add_parser(
        "pipeline-run",
        help="Execute the SQLMesh pipeline and output run stats (JSON)",
    )
    pipeline_run_parser.add_argument(
        "--trigger",
        type=str,
        default="manual",
        help="Trigger label (default: manual)",
    )

    pipeline_run_stream_parser = subparsers.add_parser(
        "pipeline-run-stream",
        help="Execute pipeline with streaming progress (JSON lines)",
    )
    pipeline_run_stream_parser.add_argument(
        "--trigger",
        type=str,
        default="manual",
        help="Trigger label (default: manual)",
    )

    pipeline_result_parser = subparsers.add_parser(
        "pipeline-run-result",
        help="Look up a pipeline run by ID (JSON)",
    )
    pipeline_result_parser.add_argument(
        "run_id",
        type=str,
        help="The run_id returned by pipeline-run",
    )

    pipeline_history_parser = subparsers.add_parser(
        "pipeline-run-history",
        help="Return recent pipeline runs as JSON array",
    )
    pipeline_history_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of runs to return (default: 5)",
    )

    # Action tool commands
    subparsers.add_parser(
        "list-actions",
        help="List available MCP action tools from enabled connectors (JSON)",
    )

    confirm_action_parser = subparsers.add_parser(
        "confirm-action",
        help="Execute a confirmed MCP action (JSON)",
    )
    confirm_action_parser.add_argument(
        "--proposal-json",
        type=str,
        required=True,
        help="JSON string of the ActionProposal",
    )

    cancel_action_parser = subparsers.add_parser(
        "cancel-action",
        help="Record cancellation of a proposed action (JSON)",
    )
    cancel_action_parser.add_argument(
        "--proposal-id",
        type=str,
        required=True,
        help="UUID of the cancelled proposal",
    )

    resume_parser = subparsers.add_parser(
        "resume-action-with-recipient",
        help=(
            "Resume a disambiguation proposal with the user's chosen "
            "candidate; emits a normal ActionProposal (JSON)"
        ),
    )
    resume_parser.add_argument(
        "--disambiguation-json",
        type=str,
        required=True,
        help="JSON string of the RecipientDisambiguationProposal",
    )
    resume_parser.add_argument(
        "--candidate-json",
        type=str,
        required=True,
        help="JSON string of the chosen ContactCandidate",
    )

    search_recipient_parser = subparsers.add_parser(
        "search-recipient-candidates",
        help=(
            "Search the user's contacts for a Send Message recipient "
            "(powers the disambiguation card's search input)"
        ),
    )
    search_recipient_parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Free-text name to search for",
    )
    search_recipient_parser.add_argument(
        "--channel",
        type=str,
        required=True,
        choices=["whatsapp", "email", "imessage"],
        help="Messaging channel — picks phone vs email as the handle",
    )
    search_recipient_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of candidates to return (default 5)",
    )
    search_recipient_parser.add_argument(
        "--include-apple",
        action="store_true",
        help=(
            "Also search the macOS AddressBook via the apple-contacts "
            "MCP server (slower — opt-in via UI button)"
        ),
    )

    # Connector commands
    subparsers.add_parser(
        "connector-catalog",
        help="List all connectors with status (JSON)",
    )

    subparsers.add_parser(
        "system-health",
        help="Aggregated connector/pipeline/graph/vector health (JSON)",
    )

    toggle_parser = subparsers.add_parser(
        "toggle-connector",
        help="Enable or disable a connector (JSON)",
    )
    toggle_parser.add_argument(
        "connector_id",
        type=str,
        help="Connector ID (e.g. 'apple-calendar')",
    )
    toggle_parser.add_argument(
        "--enabled",
        type=str,
        choices=["true", "false"],
        required=True,
        help="Toggle state",
    )
    toggle_parser.add_argument(
        "--user-inputs",
        type=str,
        default=None,
        help="JSON string of user-provided values",
    )

    sync_parser = subparsers.add_parser(
        "sync-connector",
        help="Trigger immediate sync for a connector (JSON)",
    )
    sync_parser.add_argument(
        "connector_id",
        type=str,
        help="Connector ID to sync",
    )

    subparsers.add_parser(
        "ensure-whatsapp-listener",
        help="Ensure WhatsApp listener is running if enabled (JSON)",
    )
    subparsers.add_parser(
        "whatsapp-listener-spec",
        help="WhatsApp listener spec for supervisor (JSON)",
    )
    subparsers.add_parser(
        "whatsapp-listener-start",
        help="Start persistent WhatsApp listener (JSON)",
    )
    subparsers.add_parser(
        "whatsapp-listener-stop",
        help="Stop persistent WhatsApp listener (JSON)",
    )
    subparsers.add_parser(
        "whatsapp-listener-status",
        help="WhatsApp listener runtime status (JSON)",
    )

    wa_listener_run = subparsers.add_parser(
        "whatsapp-listener-run",
        help=argparse.SUPPRESS,
    )
    wa_listener_run.add_argument(
        "--mcp-command",
        dest="mcp_command",
        type=str,
        default="npx",
    )
    wa_listener_run.add_argument(
        "--mcp-arg",
        action="append",
        default=[],
    )
    wa_listener_run.add_argument(
        "--mcp-timeout-seconds",
        type=float,
        default=45.0,
    )
    wa_listener_run.add_argument(
        "--scan-interval-seconds",
        type=float,
        default=2.0,
    )
    wa_listener_run.add_argument(
        "--reconnect-backoff-seconds",
        type=float,
        default=5.0,
    )

    details_parser = subparsers.add_parser(
        "connector-details",
        help="Full details for a single connector (JSON)",
    )
    details_parser.add_argument(
        "connector_id",
        type=str,
        help="Connector ID",
    )

    # Extension installer commands
    discover_parser = subparsers.add_parser(
        "discover-extension",
        help="Discover tools/schema from an MCP server (JSON)",
    )
    discover_parser.add_argument(
        "ext_command",
        type=str,
        help="MCP server command (e.g. 'npx')",
    )
    discover_parser.add_argument(
        "ext_args",
        nargs="*",
        default=[],
        help="MCP server arguments (e.g. '-y @scope/mcp-server-name')",
    )
    discover_parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Human-friendly name override",
    )
    discover_parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="JSON object of env vars to pass to the MCP server",
    )

    confirm_parser = subparsers.add_parser(
        "confirm-extension",
        help="Confirm and finalize an extension install (JSON)",
    )
    confirm_parser.add_argument(
        "--preview-json",
        type=str,
        required=True,
        help="JSON string of the InstallPreview from discover",
    )
    confirm_parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name override for the connector",
    )
    confirm_parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="JSON object of env vars to persist with the connector",
    )

    # Model generator commands
    gen_models_parser = subparsers.add_parser(
        "generate-models",
        help="Generate pipeline models for a new data source (JSON)",
    )
    gen_models_parser.add_argument(
        "--connector-id",
        type=str,
        required=True,
        help="Connector ID (e.g. 'custom-spotify')",
    )
    gen_models_parser.add_argument(
        "--mapping-json",
        type=str,
        required=True,
        help="JSON string of the DiscoveredMapping",
    )

    approve_parser = subparsers.add_parser(
        "approve-models",
        help="Approve staged models and install into pipeline (JSON)",
    )
    approve_parser.add_argument(
        "--connector-id",
        type=str,
        required=True,
        help="Connector ID to approve",
    )

    reject_parser = subparsers.add_parser(
        "reject-models",
        help="Reject staged models without installing (JSON)",
    )
    reject_parser.add_argument(
        "--connector-id",
        type=str,
        required=True,
        help="Connector ID to reject",
    )

    # Agent runner commands
    subparsers.add_parser(
        "list-agents",
        help="List all discovered agents with status (JSON)",
    )

    run_agent_parser = subparsers.add_parser(
        "run-agent",
        help="Run an agent by ID (JSON)",
    )
    run_agent_parser.add_argument(
        "--agent-id",
        type=str,
        required=True,
        help="Agent ID to run (e.g. 'weekly-digest')",
    )
    run_agent_parser.add_argument(
        "--params",
        type=str,
        default=None,
        help="JSON-encoded parameters for the agent",
    )

    agent_result_parser = subparsers.add_parser(
        "get-agent-result",
        help="Get last result from an agent run (JSON)",
    )
    agent_result_parser.add_argument(
        "--agent-id",
        type=str,
        required=True,
        help="Agent ID to query",
    )

    subparsers.add_parser(
        "list-skills",
        help="List all registered skills (JSON)",
    )

    # Extension management commands
    uninstall_parser = subparsers.add_parser(
        "uninstall-extension",
        help="Uninstall an extension (JSON)",
    )
    uninstall_parser.add_argument(
        "connector_id",
        type=str,
        help="Connector ID to uninstall",
    )
    uninstall_parser.add_argument(
        "--preserve-data",
        type=str,
        choices=["true", "false"],
        default="true",
        help="Preserve raw data tables (default: true)",
    )

    conn_history_parser = subparsers.add_parser(
        "connector-history",
        help="Sync history for a connector (JSON)",
    )
    conn_history_parser.add_argument(
        "connector_id",
        type=str,
        help="Connector ID to query",
    )
    conn_history_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum entries to return (default: 20)",
    )

    ext_logs_parser = subparsers.add_parser(
        "extension-logs",
        help="Recent log lines for an extension (JSON)",
    )
    ext_logs_parser.add_argument(
        "extension_id",
        type=str,
        help="Extension ID to query",
    )
    ext_logs_parser.add_argument(
        "--lines",
        type=int,
        default=50,
        help="Number of lines to return (default: 50)",
    )

    # Sync lifecycle commands
    subparsers.add_parser(
        "startup-sync",
        help="Sync stale connectors and refresh pipeline on launch",
    )
    subparsers.add_parser(
        "sync-all-stale",
        help="Sync all enabled connectors (periodic background)",
    )
    subparsers.add_parser(
        "run-scheduled-agents",
        help="Run agents whose cron schedule is due",
    )
    subparsers.add_parser(
        "health",
        help="Check all system components and report status (JSON)",
    )
    subparsers.add_parser(
        "get-interests",
        help="Interest profile (JSON)",
    )
    subparsers.add_parser(
        "get-domain-stats",
        help="Per-domain query statistics (JSON)",
    )
    subparsers.add_parser(
        "plan-refresh",
        help="Smart pipeline refresh plan (JSON)",
    )

    # Insight commands
    p_get_insights = subparsers.add_parser(
        "get-insights",
        help="Active insights (JSON)",
    )
    p_get_insights.add_argument(
        "--limit", type=int, default=3,
    )
    subparsers.add_parser(
        "generate-insights",
        help="Generate daily insights (JSON)",
    )
    p_dismiss = subparsers.add_parser(
        "dismiss-insight",
        help="Dismiss an insight",
    )
    p_dismiss.add_argument(
        "--insight-id", required=True,
    )
    p_followup = subparsers.add_parser(
        "follow-up-insight",
        help="Follow up on an insight",
    )
    p_followup.add_argument(
        "--insight-id", required=True,
    )

    # Proactive intelligence commands
    subparsers.add_parser(
        "evaluate-proactive",
        help="Run proactive intelligence evaluation (JSON)",
    )
    subparsers.add_parser(
        "get-pending-replies",
        help="Get pending replies (JSON)",
    )
    subparsers.add_parser(
        "get-contact-contexts",
        help="Get contact contexts (JSON)",
    )
    subparsers.add_parser(
        "get-actionable-events",
        help="Get actionable events (JSON)",
    )
    p_dismiss_reply = subparsers.add_parser(
        "dismiss-pending-reply",
        help="Dismiss a pending reply",
    )
    p_dismiss_reply.add_argument(
        "--id", required=True, dest="reply_id",
    )
    p_dismiss_event = subparsers.add_parser(
        "dismiss-actionable-event",
        help="Dismiss an actionable event",
    )
    p_dismiss_event.add_argument(
        "--id", required=True, dest="event_id",
    )

    # Tasks / Goals / Habits / Schedule commands
    p_goals_list = subparsers.add_parser(
        "goals-list", help="List goals (JSON)",
    )
    p_goals_list.add_argument("--status", default="active")
    p_goals_list.add_argument("--category", default=None)
    p_goals_create = subparsers.add_parser(
        "goals-create", help="Create a goal",
    )
    p_goals_create.add_argument("--title", required=True)
    p_goals_create.add_argument(
        "--category", required=True,
        choices=["personal", "life", "work"],
    )
    p_goals_create.add_argument("--description", default="")
    p_goals_create.add_argument(
        "--horizon", default="medium",
        choices=["short", "medium", "long"],
    )
    p_goals_create.add_argument("--target-date", default=None, dest="target_date")
    p_goals_create.add_argument("--importance", type=int, default=5)
    p_goals_create.add_argument("--why", default="")
    p_goals_update = subparsers.add_parser(
        "goals-update", help="Update mutable goal fields",
    )
    p_goals_update.add_argument("--id", required=True, dest="goal_id")
    p_goals_update.add_argument(
        "--patch", required=True, dest="patch_json",
        help="JSON object of fields to update",
    )
    subparsers.add_parser(
        "goals-mine",
        help="Run the goal extractor over recent evidence (JSON)",
    )

    p_proj_list = subparsers.add_parser(
        "projects-list", help="List projects (JSON)",
    )
    p_proj_list.add_argument("--status", default="active")
    p_proj_list.add_argument("--category", default=None)
    p_proj_create = subparsers.add_parser(
        "projects-create", help="Create a project",
    )
    p_proj_create.add_argument("--name", required=True)
    p_proj_create.add_argument(
        "--category", default="personal",
        choices=["personal", "life", "work"],
    )
    p_proj_create.add_argument("--goal-id", default=None, dest="goal_id")
    p_proj_archive = subparsers.add_parser(
        "projects-archive", help="Archive a project",
    )
    p_proj_archive.add_argument("--id", required=True, dest="project_id")

    p_tasks_list = subparsers.add_parser(
        "tasks-list", help="List tasks (JSON)",
    )
    p_tasks_list.add_argument("--status", default=None)
    p_tasks_list.add_argument("--project-id", default=None, dest="project_id")
    p_tasks_list.add_argument("--goal-id", default=None, dest="goal_id")
    p_tasks_list.add_argument(
        "--parent-task-id", default=None, dest="parent_task_id",
    )
    p_tasks_create = subparsers.add_parser(
        "tasks-create", help="Create a task",
    )
    p_tasks_create.add_argument("--title", required=True)
    p_tasks_create.add_argument("--project-id", default=None, dest="project_id")
    p_tasks_create.add_argument(
        "--parent-task-id", default=None, dest="parent_task_id",
    )
    p_tasks_create.add_argument("--goal-id", default=None, dest="goal_id")
    p_tasks_create.add_argument("--notes", default="")
    p_tasks_create.add_argument("--importance", type=int, default=5)
    p_tasks_create.add_argument("--due-at", default=None, dest="due_at")
    p_tasks_update = subparsers.add_parser(
        "tasks-update", help="Update mutable task fields",
    )
    p_tasks_update.add_argument("--id", required=True, dest="task_id")
    p_tasks_update.add_argument(
        "--patch", required=True, dest="patch_json",
        help="JSON object of fields to update",
    )
    p_tasks_toggle = subparsers.add_parser(
        "tasks-toggle", help="Toggle a task's done state",
    )
    p_tasks_toggle.add_argument("--id", required=True, dest="task_id")
    p_tasks_toggle.add_argument(
        "--note", default=None, dest="completion_note",
    )
    p_tasks_delete = subparsers.add_parser(
        "tasks-delete", help="Delete a task",
    )
    p_tasks_delete.add_argument("--id", required=True, dest="task_id")

    p_habits_list = subparsers.add_parser(
        "habits-list", help="List habits (JSON)",
    )
    p_habits_list.add_argument("--status", default="active")
    p_habits_list.add_argument("--goal-id", default=None, dest="goal_id")
    p_habits_create = subparsers.add_parser(
        "habits-create", help="Create a habit",
    )
    p_habits_create.add_argument("--title", required=True)
    p_habits_create.add_argument("--goal-id", required=True, dest="goal_id")
    p_habits_create.add_argument(
        "--cadence", default="daily",
        choices=["daily", "weekly", "specific_days"],
    )
    p_habits_create.add_argument(
        "--days-of-week", default="[]", dest="days_of_week_json",
        help="JSON array of day strings (mon..sun)",
    )
    p_habits_create.add_argument(
        "--preferred-window", default="any", dest="preferred_window",
        choices=["morning", "midday", "evening", "any"],
    )
    p_habits_create.add_argument("--why", default="")
    p_habits_toggle = subparsers.add_parser(
        "habits-toggle", help="Toggle habit active/paused",
    )
    p_habits_toggle.add_argument("--id", required=True, dest="habit_id")
    p_habits_delete = subparsers.add_parser(
        "habits-delete", help="Delete a habit",
    )
    p_habits_delete.add_argument("--id", required=True, dest="habit_id")
    subparsers.add_parser(
        "habits-regenerate",
        help="Regenerate brain-sourced habits from current goals",
    )

    p_sched_get = subparsers.add_parser(
        "schedule-get", help="Get the saved daily schedule (JSON)",
    )
    p_sched_get.add_argument(
        "--date", default=None, dest="schedule_date",
        help="ISO date (defaults to today)",
    )
    p_sched_regen = subparsers.add_parser(
        "schedule-regenerate",
        help="Regenerate the daily schedule (JSON)",
    )
    p_sched_regen.add_argument(
        "--date", default=None, dest="schedule_date",
    )

    # Mission Control dashboard commands
    p_brief = subparsers.add_parser(
        "get-daily-brief",
        help="Today's synthesized brief (JSON, cached)",
    )
    p_brief.add_argument(
        "--force",
        action="store_true",
        help="Bypass the brief cache and regenerate",
    )
    p_threads = subparsers.add_parser(
        "get-active-threads",
        help="Cross-source threads of attention (JSON)",
    )
    p_threads.add_argument(
        "--limit", type=int, default=10,
    )
    subparsers.add_parser(
        "get-agent-stream",
        help="Live agent activity for Mission Control (JSON)",
    )
    p_sugg = subparsers.add_parser(
        "get-suggested-actions",
        help="Command Bar suggestion chips (JSON)",
    )
    p_sugg.add_argument(
        "--limit", type=int, default=3,
    )
    p_domain = subparsers.add_parser(
        "get-domain-summary",
        help="Life-domain summary (work/personal/health) as JSON",
    )
    p_domain.add_argument(
        "--domain",
        type=str,
        required=True,
        choices=["work", "personal", "health"],
    )

    # New dashboard subcommands — Today board / inbox / goal progress.
    subparsers.add_parser(
        "today-board",
        help="Prioritized Now / Up Next / Loops board for the dashboard (JSON)",
    )
    subparsers.add_parser(
        "get-life-board",
        help=(
            "Unified Work/Personal/Health board (goals + today's actions "
            "+ domain items) for the dashboard (JSON)"
        ),
    )
    p_goal_progress = subparsers.add_parser(
        "goal-progress",
        help="Progress + today's moves for one goal (JSON)",
    )
    p_goal_progress.add_argument("--id", type=str, required=True)
    p_inbox = subparsers.add_parser(
        "list-inbox",
        help="Canonical inbox of pending replies (JSON)",
    )
    p_inbox.add_argument("--domain", type=str, default=None)
    p_inbox.add_argument("--topic", type=str, default=None)

    subparsers.add_parser(
        "process-whatsapp-replies",
        help="Process WhatsApp self-chat replies (JSON)",
    )

    # Real-time message evaluation
    p_eval_msgs = subparsers.add_parser(
        "evaluate-messages",
        help="Evaluate new messages for a connector (JSON)",
    )
    p_eval_msgs.add_argument(
        "--connector",
        type=str,
        required=True,
        dest="eval_connector_id",
        help="Connector ID to evaluate messages for",
    )

    # Notification commands
    subparsers.add_parser(
        "notification-prefs-get",
        help="Get notification preferences (JSON)",
    )
    p_notif_set = subparsers.add_parser(
        "notification-prefs-set",
        help="Set a notification preference",
    )
    p_notif_set.add_argument(
        "--category", required=True,
    )
    p_notif_set.add_argument(
        "--enabled",
        type=str,
        choices=["true", "false"],
        required=True,
    )
    p_notif_mute = subparsers.add_parser(
        "notification-prefs-mute-all",
        help="Mute all notifications",
    )
    p_notif_mute.add_argument(
        "--until", type=str, default=None,
    )
    subparsers.add_parser(
        "notification-prefs-unmute-all",
        help="Unmute all notifications",
    )
    p_notif_log = subparsers.add_parser(
        "notification-log",
        help="Notification log (JSON)",
    )
    p_notif_log.add_argument(
        "--limit", type=int, default=20,
    )
    p_notif_log.add_argument(
        "--offset", type=int, default=0,
    )

    # --- User profile inference ---
    subparsers.add_parser(
        "infer-profile",
        help="Infer user profile from available data (JSON)",
    )

    # --- Learned facts ---
    subparsers.add_parser(
        "get-learned-facts",
        help="Get active learned facts (JSON)",
    )
    subparsers.add_parser(
        "get-facts-for-review",
        help="Get facts pending user review (JSON)",
    )
    subparsers.add_parser(
        "get-fact-stats",
        help="Get fact statistics (JSON)",
    )
    p_confirm_fact = subparsers.add_parser(
        "confirm-fact",
        help="Confirm a learned fact",
    )
    p_confirm_fact.add_argument("--fact-id", required=True)
    p_dismiss_fact = subparsers.add_parser(
        "dismiss-fact",
        help="Dismiss a learned fact",
    )
    p_dismiss_fact.add_argument("--fact-id", required=True)
    p_edit_fact = subparsers.add_parser(
        "edit-fact",
        help="Edit a learned fact",
    )
    p_edit_fact.add_argument("--fact-id", required=True)
    p_edit_fact.add_argument("--content", required=True)

    # --- Voice transcription ---
    p_transcribe = subparsers.add_parser(
        "transcribe-audio",
        help="Transcribe audio file or base64 data to text (JSON)",
    )
    p_transcribe.add_argument(
        "audio_input",
        help="Path to audio file or base64-encoded audio data",
    )
    p_transcribe.add_argument(
        "--model-size",
        default=None,
        help="Whisper model size: tiny, base, small (reads settings if omitted)",
    )
    p_transcribe.add_argument(
        "--language",
        default=None,
        help="Optional ISO-639-1 language hint (e.g. 'en', 'es'); omit to auto-detect",
    )

    # --- Agents page (Phase 4) ---
    subparsers.add_parser(
        "agents-list",
        help="List every registered Pydantic AI agent (JSON)",
    )
    p_agents_get = subparsers.add_parser(
        "agents-get",
        help="Resolved config + registry info for one agent (JSON)",
    )
    p_agents_get.add_argument("--agent-id", required=True)
    p_agents_update = subparsers.add_parser(
        "agents-update",
        help="Apply a patch to one editable agent's config (JSON)",
    )
    p_agents_update.add_argument("--agent-id", required=True)
    p_agents_update.add_argument(
        "--patch", required=True,
        help="JSON object with any of system_prompt, model_route, "
             "model_override, enabled_tools, enabled_skills",
    )
    p_agents_reset = subparsers.add_parser(
        "agents-reset",
        help="Drop override row for one editable agent (JSON)",
    )
    p_agents_reset.add_argument("--agent-id", required=True)
    p_agents_run_eval = subparsers.add_parser(
        "agents-run-eval",
        help="Run an agent's eval suite synchronously (JSON)",
    )
    p_agents_run_eval.add_argument("--agent-id", required=True)
    p_agents_run_eval.add_argument(
        "--trigger",
        default="manual",
        choices=("manual", "auto"),
    )
    p_agents_run_eval_proposal = subparsers.add_parser(
        "agents-run-eval-proposal",
        help=(
            "Run an agent's eval suite against a proposed model "
            "override without persisting the config change (JSON)"
        ),
    )
    p_agents_run_eval_proposal.add_argument("--agent-id", required=True)
    p_agents_run_eval_proposal.add_argument(
        "--override", required=True,
        help="Candidate model id to evaluate against",
    )
    p_set_local_infer = subparsers.add_parser(
        "set-local-inference-for-sensitive",
        help=(
            "Flip the privacy mode. When --enabled=true the handler "
            "runs every registered agent's eval suite against the "
            "current local model; the flag commits only if every "
            "agent's run status is 'passed'."
        ),
    )
    p_set_local_infer.add_argument(
        "--enabled", required=True, choices=("true", "false"),
    )
    p_agents_eval_status = subparsers.add_parser(
        "agents-eval-status",
        help="Latest (and optional history) eval row(s) for an agent",
    )
    p_agents_eval_status.add_argument("--agent-id", required=True)
    p_agents_eval_status.add_argument(
        "--limit", type=int, default=1,
    )
    p_agents_activity = subparsers.add_parser(
        "agents-activity",
        help="Most-recent input/output entries for an agent (JSON)",
    )
    p_agents_activity.add_argument("--agent-id", required=True)
    p_agents_activity.add_argument(
        "--limit", type=int, default=100,
    )
    p_agents_eval_dataset = subparsers.add_parser(
        "agents-eval-dataset",
        help="Return the eval dataset YAML for an agent (JSON)",
    )
    p_agents_eval_dataset.add_argument("--agent-id", required=True)
    p_agents_validate_dataset = subparsers.add_parser(
        "agents-validate-dataset",
        help="Validate (and persist on success) a user dataset upload",
    )
    p_agents_validate_dataset.add_argument("--agent-id", required=True)
    p_agents_validate_dataset.add_argument(
        "--content", required=True,
        help="Raw YAML content to validate",
    )
    p_agents_suggest_dataset = subparsers.add_parser(
        "agents-suggest-dataset",
        help=(
            "Propose a starter eval dataset for a user agent (JSON). "
            "Pass either --agent-id (saved agent) or --unsaved-spec "
            "(JSON payload from the create-agent modal)."
        ),
    )
    p_agents_suggest_dataset.add_argument("--agent-id", default=None)
    p_agents_suggest_dataset.add_argument(
        "--unsaved-spec", default=None,
        help=(
            "JSON object with name, description, system_prompt, "
            "max_sensitivity_tier, optional output_schema and "
            "available_tools. Used for the create-modal preview."
        ),
    )
    p_agents_suggest_model = subparsers.add_parser(
        "agents-suggest-model",
        help=(
            "Recommend best-overall + cost-effective models for an "
            "agent spec (JSON). The live /models lists for the remote "
            "and local routes are fetched server-side so the LLM picks "
            "from real ids only."
        ),
    )
    p_agents_suggest_model.add_argument(
        "--unsaved-spec", required=True,
        help=(
            "JSON object with name, description, system_prompt, "
            "max_sensitivity_tier, optional output_schema, "
            "enabled_skills, enabled_mcp_tools, and agent_id."
        ),
    )
    p_agents_suggest_prompt = subparsers.add_parser(
        "agents-suggest-prompt-improvements",
        help=(
            "Rewrite a user agent's system prompt + description for "
            "clarity, expected output, language pinning, format "
            "strictness, scope, and safety. Returns a PromptSuggestion "
            "(JSON) with both a full rewrite and a surgical-additions "
            "list."
        ),
    )
    p_agents_suggest_prompt.add_argument(
        "--unsaved-spec", required=True,
        help=(
            "JSON object with name, description, system_prompt, "
            "max_sensitivity_tier, plus optional output_schema, "
            "available_tools, available_skills, enabled_mcp_tools, "
            "agent_id, has_dataset, and prior_eval_failures."
        ),
    )
    p_agents_apply_prompt_edit = subparsers.add_parser(
        "agents-user-apply-prompt-edit",
        help=(
            "Apply a prompt-engineer rewrite to a user agent. "
            "Snapshots the current system_prompt + description into "
            "pre_ai_* columns, writes the new values, and mirrors the "
            "new system_prompt into the config overlay."
        ),
    )
    p_agents_apply_prompt_edit.add_argument("--agent-id", required=True)
    p_agents_apply_prompt_edit.add_argument(
        "--payload", required=True,
        help="JSON object with system_prompt and description.",
    )
    p_agents_revert_ai = subparsers.add_parser(
        "agents-user-revert-ai-edit",
        help=(
            "Restore a user agent's system_prompt + description from "
            "the pre-AI-edit snapshot recorded at the most recent "
            "prompt-engineer apply."
        ),
    )
    p_agents_revert_ai.add_argument("--agent-id", required=True)
    p_agents_create = subparsers.add_parser(
        "agents-create",
        help="Create a new user-authored agent (JSON payload)",
    )
    p_agents_create.add_argument("--payload", required=True)
    p_agents_user_update = subparsers.add_parser(
        "agents-user-update",
        help="Update a user-authored agent's row (JSON payload)",
    )
    p_agents_user_update.add_argument("--agent-id", required=True)
    p_agents_user_update.add_argument("--payload", required=True)
    p_agents_delete = subparsers.add_parser(
        "agents-delete",
        help="Delete a user-authored agent",
    )
    p_agents_delete.add_argument("--agent-id", required=True)
    p_agents_set_schedule = subparsers.add_parser(
        "agents-set-schedule",
        help="Set the schedule cron + enabled flag for a user agent",
    )
    p_agents_set_schedule.add_argument("--agent-id", required=True)
    p_agents_set_schedule.add_argument(
        "--cron", default=None,
        help="Cron expression, or empty to clear",
    )
    p_agents_set_schedule.add_argument(
        "--enabled", default="false",
        choices=("true", "false"),
    )
    p_agents_run_now = subparsers.add_parser(
        "agents-run-now",
        help="Invoke a user agent immediately (batch if sources, "
             "else generic trigger)",
    )
    p_agents_run_now.add_argument("--agent-id", required=True)
    p_agents_user_status = subparsers.add_parser(
        "agents-user-status",
        help="Return scheduling + last/next/pending status for one "
             "user agent (JSON)",
    )
    p_agents_user_status.add_argument("--agent-id", required=True)
    subparsers.add_parser(
        "agents-list-mcp-tools",
        help="List MCP action tools from enabled connectors (JSON)",
    )
    p_agents_list_models = subparsers.add_parser(
        "agents-list-models",
        help=(
            "List model ids exposed by a route's endpoint "
            "(JSON; chat-family ids sorted first)"
        ),
    )
    p_agents_list_models.add_argument(
        "--route",
        choices=("remote", "local", "inherit"),
        default="remote",
    )
    p_skills_create = subparsers.add_parser(
        "skills-create",
        help="Create a new user-authored skill (JSON payload)",
    )
    p_skills_create.add_argument("--payload", required=True)
    p_skills_update = subparsers.add_parser(
        "skills-update",
        help="Update an existing user-authored skill (JSON payload)",
    )
    p_skills_update.add_argument("--skill-id", required=True)
    p_skills_update.add_argument("--payload", required=True)
    p_skills_delete = subparsers.add_parser(
        "skills-delete",
        help="Delete a user-authored skill",
    )
    p_skills_delete.add_argument("--skill-id", required=True)
    p_skills_get = subparsers.add_parser(
        "skills-get",
        help="Return one skill's metadata + prompt template (JSON)",
    )
    p_skills_get.add_argument("--skill-id", required=True)

    # ----- Skills v2 (SKILL.md-based) -----
    subparsers.add_parser(
        "skills-list-v2",
        help="List all SKILL.md-based skills (L1 metadata)",
    )
    p_skills_get_v2 = subparsers.add_parser(
        "skills-get-v2",
        help="Return one SKILL.md skill's full content (L2)",
    )
    p_skills_get_v2.add_argument("--skill-id", required=True)
    p_skills_create_v2 = subparsers.add_parser(
        "skills-create-v2",
        help="Create a new SKILL.md-based skill",
    )
    p_skills_create_v2.add_argument("--name", required=True)
    p_skills_create_v2.add_argument("--content", required=True)
    p_skills_update_v2 = subparsers.add_parser(
        "skills-update-v2",
        help="Update a SKILL.md-based skill's content",
    )
    p_skills_update_v2.add_argument("--skill-id", required=True)
    p_skills_update_v2.add_argument("--content", required=True)
    p_skills_delete_v2 = subparsers.add_parser(
        "skills-delete-v2",
        help="Delete a SKILL.md-based skill",
    )
    p_skills_delete_v2.add_argument("--skill-id", required=True)

    p_redaction_detail = subparsers.add_parser(
        "get-redaction-detail",
        help="Return original/redacted payload for one audit row (JSON)",
    )
    p_redaction_detail.add_argument(
        "--payload-hash", required=True,
        help="SHA-256 from the audit row's payload_hash field",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, open a DataLayer, and dispatch to the right command.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    kwargs: dict[str, object] = {}
    if args.data_dir is not None:
        kwargs["base_path"] = args.data_dir

    # Commands that don't need a DataLayer.
    if args.command == "delete-all-data":
        # Runs before any engine opens so nothing holds the data
        # directory open while it's wiped.
        data_dir = str(args.data_dir) if args.data_dir is not None else None
        return cmd_delete_all_data(data_dir)
    if args.command == "transcribe-audio":
        return cmd_transcribe_audio(
            args.audio_input, args.model_size, args.language,
        )
    if args.command == "ensure-whatsapp-listener":
        return cmd_ensure_whatsapp_listener()
    if args.command == "whatsapp-listener-spec":
        return cmd_whatsapp_listener_spec()
    if args.command == "whatsapp-listener-start":
        return cmd_whatsapp_listener_start()
    if args.command == "whatsapp-listener-stop":
        return cmd_whatsapp_listener_stop()
    if args.command == "whatsapp-listener-status":
        return cmd_whatsapp_listener_status()
    if args.command == "whatsapp-listener-run":
        return cmd_whatsapp_listener_run(
            command=args.mcp_command,
            mcp_args=args.mcp_arg,
            mcp_timeout_seconds=args.mcp_timeout_seconds,
            scan_interval_seconds=args.scan_interval_seconds,
            reconnect_backoff_seconds=args.reconnect_backoff_seconds,
        )

    # Commands that write to the database need read-write access.
    # All other commands open DuckDB in read-only mode to avoid
    # file-locking conflicts with concurrent pipeline runs.
    write_commands = {
        "init", "reset", "pipeline-run", "pipeline-run-stream",
        "toggle-connector", "discover-extension", "confirm-extension",
        "sync-connector",
        "generate-models", "approve-models", "run-agent",
        "run-scheduled-agents",
        "confirm-action",
        "ask", "ask-stream",  # query tracker writes
        "chat-session-create", "chat-session-delete",
        "generate-insights", "dismiss-insight", "follow-up-insight",
        "evaluate-proactive",
        "dismiss-pending-reply", "dismiss-actionable-event",
        "process-whatsapp-replies",
        "evaluate-messages",
        "notification-prefs-set",
        "notification-prefs-mute-all",
        "notification-prefs-unmute-all",
        "fix-notes-content",
        "infer-profile",
        "confirm-fact", "dismiss-fact", "edit-fact",
        "agents-update", "agents-reset", "agents-run-eval",
        "agents-create", "agents-user-update", "agents-delete",
        "agents-user-apply-prompt-edit", "agents-user-revert-ai-edit",
        "agents-set-schedule", "agents-run-now", "agents-validate-dataset",
        "set-local-inference-for-sensitive",
        "skills-create", "skills-update", "skills-delete",
    }
    is_read_only = args.command not in write_commands
    if is_read_only:
        kwargs["read_only"] = True

    # Kuzu has its own writer set. Most "write" commands write SQLite
    # (QueryTracker, prefs, agents, etc.) and only *read* the graph, so
    # they can safely open Kuzu in shared read-only mode. Forcing Kuzu
    # read-only by default lets long-running readers (WhatsApp
    # listener serving BrainAgent reply context) coexist with chat
    # without lock contention.
    kuzu_write_commands = {
        "init", "reset",
        "pipeline-run", "pipeline-run-stream",
        "discover-extension", "confirm-extension",
        "generate-models", "approve-models",
    }
    kuzu_read_only = args.command not in kuzu_write_commands
    if kuzu_read_only:
        kwargs["kuzu_read_only"] = True

    logger.info(
        "CLI command=%s sqlite=%s kuzu=%s",
        args.command,
        "read_only" if is_read_only else "read_write",
        "read_only" if kuzu_read_only else "read_write",
    )

    # Fast-path: commands that don't need the DataLayer at all.
    # Avoids SQLite/Kuzu/ChromaDB init overhead for simple health checks.
    if args.command == "ollama-status":
        return cmd_ollama_status()
    if args.command == "ollama-preload":
        return cmd_ollama_preload()
    if args.command == "ollama-stop":
        return cmd_ollama_stop()
    if args.command == "monitor":
        return cmd_monitor()
    if args.command == "rebuild-vector-index":
        return cmd_rebuild_vector_index(
            target_model=args.to_model,
            target_provider_kind=args.provider,
            api_key=args.api_key,
            base_url=args.base_url,
            dimensions=args.dimensions,
            dry_run=args.dry_run,
        )
    # Agents page (Phase 4): registry + SQLite-backed config store —
    # no DataLayer init needed, only the agent_configs table.
    if args.command == "agents-list":
        from src.agents.cli_handlers import cmd_agents_list
        return cmd_agents_list()
    if args.command == "agents-get":
        from src.agents.cli_handlers import cmd_agents_get
        return cmd_agents_get(args.agent_id)
    if args.command == "agents-update":
        from src.agents.cli_handlers import cmd_agents_update
        return cmd_agents_update(args.agent_id, args.patch)
    if args.command == "agents-reset":
        from src.agents.cli_handlers import cmd_agents_reset
        return cmd_agents_reset(args.agent_id)
    if args.command == "agents-list-models":
        from src.agents.cli_handlers import cmd_agents_list_models
        return cmd_agents_list_models(args.route)
    if args.command == "agents-run-eval":
        from src.agents.cli_handlers import cmd_agents_run_eval
        return cmd_agents_run_eval(args.agent_id, trigger=args.trigger)
    if args.command == "agents-run-eval-proposal":
        from src.agents.cli_handlers import cmd_agents_run_eval_proposal
        return cmd_agents_run_eval_proposal(
            args.agent_id, proposed_override=args.override,
        )
    if args.command == "set-local-inference-for-sensitive":
        from src.agents.cli_handlers import (
            cmd_set_local_inference_for_sensitive,
        )
        return cmd_set_local_inference_for_sensitive(args.enabled)
    if args.command == "agents-eval-status":
        from src.agents.cli_handlers import cmd_agents_eval_status
        return cmd_agents_eval_status(args.agent_id, limit=args.limit)
    if args.command == "agents-activity":
        from src.agents.cli_handlers import cmd_agents_activity
        return cmd_agents_activity(args.agent_id, limit=args.limit)
    if args.command == "agents-eval-dataset":
        from src.agents.cli_handlers import cmd_agents_eval_dataset
        return cmd_agents_eval_dataset(args.agent_id)
    if args.command == "agents-validate-dataset":
        from src.agents.cli_handlers import cmd_agents_validate_dataset
        return cmd_agents_validate_dataset(args.agent_id, args.content)
    if args.command == "agents-suggest-dataset":
        from src.agents.cli_handlers import cmd_agents_suggest_dataset
        return cmd_agents_suggest_dataset(
            args.agent_id, args.unsaved_spec,
        )
    if args.command == "agents-suggest-model":
        from src.agents.cli_handlers import cmd_agents_suggest_model
        return cmd_agents_suggest_model(args.unsaved_spec)
    if args.command == "agents-suggest-prompt-improvements":
        from src.agents.cli_handlers import (
            cmd_agents_suggest_prompt_improvements,
        )
        return cmd_agents_suggest_prompt_improvements(args.unsaved_spec)
    if args.command == "agents-user-apply-prompt-edit":
        from src.agents.cli_handlers import (
            cmd_agents_user_apply_prompt_edit,
        )
        return cmd_agents_user_apply_prompt_edit(
            args.agent_id, args.payload,
        )
    if args.command == "agents-user-revert-ai-edit":
        from src.agents.cli_handlers import cmd_agents_user_revert_ai_edit
        return cmd_agents_user_revert_ai_edit(args.agent_id)
    if args.command == "agents-create":
        from src.agents.cli_handlers import cmd_agents_create
        return cmd_agents_create(args.payload)
    if args.command == "agents-user-update":
        from src.agents.cli_handlers import cmd_agents_user_update
        return cmd_agents_user_update(args.agent_id, args.payload)
    if args.command == "agents-delete":
        from src.agents.cli_handlers import cmd_agents_delete
        return cmd_agents_delete(args.agent_id)
    if args.command == "agents-set-schedule":
        from src.agents.cli_handlers import cmd_agents_set_schedule
        return cmd_agents_set_schedule(
            args.agent_id,
            cron=(args.cron if args.cron else None),
            enabled=(args.enabled == "true"),
        )
    if args.command == "agents-run-now":
        from src.agents.cli_handlers import cmd_agents_run_now
        return cmd_agents_run_now(args.agent_id)
    if args.command == "agents-user-status":
        from src.agents.cli_handlers import cmd_agents_user_status
        return cmd_agents_user_status(args.agent_id)
    if args.command == "agents-list-mcp-tools":
        from src.agents.cli_handlers import cmd_agents_list_mcp_tools
        return cmd_agents_list_mcp_tools()
    if args.command == "skills-create":
        from src.agents.cli_handlers import cmd_skills_create
        return cmd_skills_create(args.payload)
    if args.command == "skills-update":
        from src.agents.cli_handlers import cmd_skills_update
        return cmd_skills_update(args.skill_id, args.payload)
    if args.command == "skills-delete":
        from src.agents.cli_handlers import cmd_skills_delete
        return cmd_skills_delete(args.skill_id)
    if args.command == "skills-get":
        from src.agents.cli_handlers import cmd_skills_get
        return cmd_skills_get(args.skill_id)
    if args.command == "skills-list-v2":
        from src.agents.cli_handlers import cmd_skills_list_v2
        return cmd_skills_list_v2()
    if args.command == "skills-get-v2":
        from src.agents.cli_handlers import cmd_skills_get_v2
        return cmd_skills_get_v2(args.skill_id)
    if args.command == "skills-create-v2":
        from src.agents.cli_handlers import cmd_skills_create_v2
        return cmd_skills_create_v2(args.name, args.content)
    if args.command == "skills-update-v2":
        from src.agents.cli_handlers import cmd_skills_update_v2
        return cmd_skills_update_v2(args.skill_id, args.content)
    if args.command == "skills-delete-v2":
        from src.agents.cli_handlers import cmd_skills_delete_v2
        return cmd_skills_delete_v2(args.skill_id)
    if args.command == "get-redaction-detail":
        return cmd_get_redaction_detail(payload_hash=args.payload_hash)
    if args.command == "list-pipeline-models":
        return cmd_list_pipeline_models()

    with DataLayer(**kwargs) as layer:
        # Commands that need all three engines — warmup eagerly.
        if args.command in ("init", "status", "reset"):
            layer.warmup()

        if args.command == "init":
            return cmd_init(layer)
        if args.command == "status":
            return cmd_status(layer)
        if args.command == "reset":
            return cmd_reset(layer)
        # JSON commands — engines lazy-init as needed.
        if args.command == "stats":
            return cmd_stats(layer)
        if args.command == "export-data":
            return cmd_export_data(layer, args.output_dir)
        if args.command == "query-messages":
            return cmd_query_messages(layer, args.limit, args.offset)
        if args.command == "query-events":
            return cmd_query_events(layer, args.days, args.limit, args.offset)
        if args.command == "query-contacts":
            return cmd_query_contacts(layer, args.limit, args.offset)
        if args.command == "query-notes":
            return cmd_query_notes(layer, args.limit, args.offset)
        if args.command == "query-emails":
            return cmd_query_emails(layer, args.limit, args.offset)
        if args.command == "fix-notes-content":
            return cmd_fix_notes_content(layer)
        if args.command == "list-tables":
            return cmd_list_tables(layer, args.prefix)
        if args.command == "query-table":
            return cmd_query_table(
                layer, args.table, args.limit, args.offset,
            )
        # Graph exploration commands
        if args.command == "graph-summary":
            return cmd_graph_summary(layer)
        if args.command == "query-graph-nodes":
            return cmd_query_graph_nodes(
                layer, args.node_type, args.limit,
            )
        if args.command == "query-graph-rels":
            return cmd_query_graph_rels(
                layer, args.rel_type, args.limit,
            )
        # Vector exploration commands
        if args.command == "vector-summary":
            return cmd_vector_summary(layer)
        if args.command == "query-today":
            return cmd_query_today(layer)
        if args.command == "profile":
            return cmd_profile(layer)
        if args.command == "ask":
            return cmd_ask(
                layer,
                args.question,
                session_id=args.session_id,
                reply_context=_parse_reply_context_arg(args.reply_context),
            )
        if args.command == "ask-stream":
            return cmd_ask_stream(
                layer,
                args.question,
                agent_id=args.agent_id,
                session_id=args.session_id,
                reply_context=_parse_reply_context_arg(args.reply_context),
                task_context=_parse_task_context_arg(
                    getattr(args, "task_context", None),
                ),
                budget=getattr(args, "budget", None),
            )
        if args.command == "stop-research":
            return cmd_stop_research(args.run_id)
        if args.command == "chat-session-create":
            return cmd_chat_session_create(layer, args.title)
        if args.command == "chat-session-list":
            return cmd_chat_session_list(layer, args.limit)
        if args.command == "chat-session-load":
            return cmd_chat_session_load(layer, args.session_id)
        if args.command == "chat-session-delete":
            return cmd_chat_session_delete(layer, args.session_id)
        if args.command == "ollama-status":
            return cmd_ollama_status()
        if args.command == "ollama-preload":
            return cmd_ollama_preload()
        if args.command == "ollama-stop":
            return cmd_ollama_stop()
        if args.command == "monitor":
            return cmd_monitor()
        if args.command == "pipeline-status":
            return cmd_pipeline_status(layer)
        if args.command == "pipeline-run":
            return cmd_pipeline_run(layer, trigger=args.trigger)
        if args.command == "pipeline-run-stream":
            return cmd_pipeline_run_stream(layer, trigger=args.trigger)
        if args.command == "pipeline-run-result":
            return cmd_pipeline_run_result(args.run_id)
        if args.command == "pipeline-run-history":
            return cmd_pipeline_run_history(limit=args.limit)
        # Action tool commands
        if args.command == "list-actions":
            return cmd_list_actions()
        if args.command == "confirm-action":
            return cmd_confirm_action(layer, args.proposal_json)
        if args.command == "cancel-action":
            return cmd_cancel_action(args.proposal_id)
        if args.command == "resume-action-with-recipient":
            return cmd_resume_action_with_recipient(
                layer,
                args.disambiguation_json,
                args.candidate_json,
            )
        if args.command == "search-recipient-candidates":
            return cmd_search_recipient_candidates(
                layer,
                args.query,
                args.channel,
                bool(args.include_apple),
                int(args.limit),
            )
        # Connector commands
        if args.command == "connector-catalog":
            return cmd_connector_catalog()
        if args.command == "system-health":
            return cmd_system_health(layer)
        if args.command == "toggle-connector":
            return cmd_toggle_connector(
                layer,
                args.connector_id,
                args.enabled == "true",
                args.user_inputs,
            )
        if args.command == "sync-connector":
            return cmd_sync_connector(layer, args.connector_id)
        if args.command == "connector-details":
            return cmd_connector_details(args.connector_id)
        # Extension installer commands
        if args.command == "discover-extension":
            env_arg = _parse_env_json(args.env)
            return cmd_discover_extension(
                args.ext_command, args.ext_args, args.name, env=env_arg,
            )
        if args.command == "confirm-extension":
            env_arg = _parse_env_json(args.env)
            return cmd_confirm_extension(
                layer, args.preview_json, args.name, env=env_arg,
            )
        # Model generator commands
        if args.command == "generate-models":
            return cmd_generate_models(
                args.connector_id, args.mapping_json,
            )
        if args.command == "approve-models":
            return cmd_approve_models(layer, args.connector_id)
        if args.command == "reject-models":
            return cmd_reject_models(args.connector_id)
        # Agent runner commands
        if args.command == "list-agents":
            return cmd_list_agents()
        if args.command == "run-agent":
            return cmd_run_agent(layer, args.agent_id, args.params)
        if args.command == "get-agent-result":
            return cmd_get_agent_result(args.agent_id)
        if args.command == "list-skills":
            return cmd_list_skills()
        # Extension management commands
        if args.command == "uninstall-extension":
            return cmd_uninstall_extension(
                args.connector_id,
                args.preserve_data == "true",
            )
        if args.command == "connector-history":
            return cmd_connector_history(args.connector_id, args.limit)
        if args.command == "extension-logs":
            return cmd_extension_logs(args.extension_id, args.lines)
        # Sync lifecycle commands
        if args.command == "startup-sync":
            return cmd_startup_sync(layer)
        if args.command == "sync-all-stale":
            return cmd_sync_all_stale(layer)
        if args.command == "run-scheduled-agents":
            return cmd_run_scheduled_agents(layer)
        if args.command == "health":
            return cmd_health(layer)
        # Interest profile commands
        if args.command == "get-interests":
            return cmd_get_interests(layer)
        if args.command == "get-domain-stats":
            return cmd_get_domain_stats(layer)
        if args.command == "plan-refresh":
            return cmd_plan_refresh(layer)
        # Insight commands
        if args.command == "get-insights":
            return cmd_get_insights(layer, args.limit)
        if args.command == "generate-insights":
            return cmd_generate_insights(layer)
        if args.command == "dismiss-insight":
            return cmd_dismiss_insight(layer, args.insight_id)
        if args.command == "follow-up-insight":
            return cmd_follow_up_insight(
                layer, args.insight_id,
            )
        # Proactive intelligence commands
        if args.command == "evaluate-proactive":
            return cmd_evaluate_proactive(layer)
        if args.command == "get-pending-replies":
            return cmd_get_pending_replies(layer)
        if args.command == "get-contact-contexts":
            return cmd_get_contact_contexts(layer)
        if args.command == "get-actionable-events":
            return cmd_get_actionable_events(layer)
        if args.command == "dismiss-pending-reply":
            return cmd_dismiss_pending_reply(
                layer, args.reply_id,
            )
        if args.command == "dismiss-actionable-event":
            return cmd_dismiss_actionable_event(
                layer, args.event_id,
            )
        # Tasks / Goals / Habits / Schedule
        if args.command == "goals-list":
            return cmd_goals_list(
                layer, args.status, args.category,
            )
        if args.command == "goals-create":
            return cmd_goals_create(
                layer,
                title=args.title,
                category=args.category,
                description=args.description,
                horizon=args.horizon,
                target_date=args.target_date,
                importance=args.importance,
                why=args.why,
            )
        if args.command == "goals-update":
            return cmd_goals_update(
                layer, args.goal_id, args.patch_json,
            )
        if args.command == "goals-mine":
            return cmd_goals_mine(layer)
        if args.command == "projects-list":
            return cmd_projects_list(
                layer, args.status, args.category,
            )
        if args.command == "projects-create":
            return cmd_projects_create(
                layer,
                name=args.name,
                category=args.category,
                goal_id=args.goal_id,
            )
        if args.command == "projects-archive":
            return cmd_projects_archive(layer, args.project_id)
        if args.command == "tasks-list":
            return cmd_tasks_list(
                layer,
                status=args.status,
                project_id=args.project_id,
                goal_id=args.goal_id,
                parent_task_id=args.parent_task_id,
            )
        if args.command == "tasks-create":
            return cmd_tasks_create(
                layer,
                title=args.title,
                project_id=args.project_id,
                parent_task_id=args.parent_task_id,
                goal_id=args.goal_id,
                notes=args.notes,
                importance=args.importance,
                due_at=args.due_at,
            )
        if args.command == "tasks-update":
            return cmd_tasks_update(
                layer, args.task_id, args.patch_json,
            )
        if args.command == "tasks-toggle":
            return cmd_tasks_toggle(
                layer, args.task_id, args.completion_note,
            )
        if args.command == "tasks-delete":
            return cmd_tasks_delete(layer, args.task_id)
        if args.command == "habits-list":
            return cmd_habits_list(
                layer, args.status, args.goal_id,
            )
        if args.command == "habits-create":
            return cmd_habits_create(
                layer,
                title=args.title,
                goal_id=args.goal_id,
                cadence=args.cadence,
                days_of_week_json=args.days_of_week_json,
                preferred_window=args.preferred_window,
                why=args.why,
            )
        if args.command == "habits-toggle":
            return cmd_habits_toggle(layer, args.habit_id)
        if args.command == "habits-delete":
            return cmd_habits_delete(layer, args.habit_id)
        if args.command == "habits-regenerate":
            return cmd_habits_regenerate(layer)
        if args.command == "schedule-get":
            return cmd_schedule_get(layer, args.schedule_date)
        if args.command == "schedule-regenerate":
            return cmd_schedule_regenerate(
                layer, args.schedule_date,
            )
        # Mission Control dashboard commands
        if args.command == "get-daily-brief":
            return cmd_get_daily_brief(layer, force=args.force)
        if args.command == "get-active-threads":
            return cmd_get_active_threads(layer, args.limit)
        if args.command == "get-agent-stream":
            return cmd_get_agent_stream(layer)
        if args.command == "get-suggested-actions":
            return cmd_get_suggested_actions(layer, args.limit)
        if args.command == "get-domain-summary":
            return cmd_get_domain_summary(layer, args.domain)
        if args.command == "today-board":
            return cmd_today_board(layer)
        if args.command == "get-life-board":
            return cmd_get_life_board(layer)
        if args.command == "goal-progress":
            return cmd_goal_progress(layer, args.id)
        if args.command == "list-inbox":
            return cmd_list_inbox(
                layer, domain=args.domain, topic=args.topic,
            )
        if args.command == "infer-profile":
            return cmd_infer_profile(layer)
        if args.command == "process-whatsapp-replies":
            return cmd_process_whatsapp_replies(layer)
        if args.command == "evaluate-messages":
            return cmd_evaluate_messages(
                layer, args.eval_connector_id,
            )
        # Notification commands
        if args.command == "notification-prefs-get":
            return cmd_notification_prefs_get(layer)
        if args.command == "notification-prefs-set":
            return cmd_notification_prefs_set(
                layer,
                args.category,
                args.enabled == "true",
            )
        if args.command == "notification-prefs-mute-all":
            return cmd_notification_prefs_mute_all(
                layer, args.until,
            )
        if args.command == "notification-prefs-unmute-all":
            return cmd_notification_prefs_unmute_all(layer)
        if args.command == "notification-log":
            return cmd_notification_log(
                layer, args.limit, args.offset,
            )

    # --- Learned facts commands ---
    if args.command == "get-learned-facts":
        return cmd_get_learned_facts(layer)
    if args.command == "get-facts-for-review":
        return cmd_get_facts_for_review(layer)
    if args.command == "get-fact-stats":
        return cmd_get_fact_stats(layer)
    if args.command == "confirm-fact":
        return cmd_confirm_fact(layer, args.fact_id)
    if args.command == "dismiss-fact":
        return cmd_dismiss_fact(layer, args.fact_id)
    if args.command == "edit-fact":
        return cmd_edit_fact(
            layer, args.fact_id, args.content,
        )

    # argparse guarantees one of the above is hit
    return 0  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
