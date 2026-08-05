"""Zero-friction extension installer — discover and install MCP servers.

Orchestrates the full flow: MCP handshake → tool discovery → sample data
probing → schema analysis → preview → confirm → table creation → registry.

Two-phase API:
  1. discover(command, args) → InstallPreview  (read-only, reversible)
  2. confirm(preview)       → InstallResult    (creates tables, registers)

sensitivity_tier: 1 (manages connector setup, no user data accessed)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.sqlite.engine import DatabaseEngine
from src.extensions.connectors.registry import ExtensionRegistry
from src.extensions.ingestion.schema_discovery import (
    DiscoveredMapping,
    FieldMapping,
    SchemaDiscoveryAgent,
    to_tool_template,
)
from src.extensions.mcp.client import (
    McpClient,
    McpTimeoutError,
    McpToolError,
    McpToolInfo,
)
from src.extensions.mcp.tool_classifier import classify_tool
from src.extensions.models import ToolTemplate

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_SAMPLE_RECORDS = 5

# Parameter names that suggest a "limit" or "count" value
_LIMIT_PARAM_NAMES: frozenset[str] = frozenset({
    "limit", "count", "max", "max_results", "page_size",
    "maxResults", "pageSize", "num",
})

# Parameter names safe to pass as empty string
_QUERY_PARAM_NAMES: frozenset[str] = frozenset({
    "query", "search", "filter", "q",
})

DISCOVERY_CACHE_VERSION = 1
DEFAULT_DISCOVERY_CACHE_DIR = (
    Path.home() / ".arandu" / "extensions" / "discovery_cache"
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolPreview:
    """Preview of a single discovered tool.

    sensitivity_tier: 1
    """

    tool_name: str
    tool_type: str  # "data" or "action"
    target_table: str | None = None  # None for action tools
    is_new_table: bool = False
    field_count: int = 0
    sensitivity_tiers: dict[int, int] = field(default_factory=dict)
    confidence: float = 0.0
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class InstallPreview:
    """Complete preview of what installing this MCP server will do.

    sensitivity_tier: 1
    """

    server_name: str
    command: str
    args: tuple[str, ...]
    tools: tuple[ToolPreview, ...]
    data_tools: int = 0
    action_tools: int = 0
    new_tables: tuple[str, ...] = ()
    existing_tables: tuple[str, ...] = ()
    overall_confidence: float = 0.0
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class InstallResult:
    """Result of confirming an extension install.

    sensitivity_tier: 1
    """

    status: str  # "installed" | "error"
    connector_id: str
    tables_created: tuple[str, ...] = ()
    tools_registered: int = 0
    models_staged: int = 0
    error: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _derive_server_name(command: str, args: tuple[str, ...]) -> str:
    """Derive a human-friendly server name from command and args.

    Examples:
        npx -y @anthropic/mcp-server-weather → "weather"
        uvx mcp-server-fetch → "fetch"
        /usr/local/bin/my-server → "my-server"

    sensitivity_tier: 1
    """
    # Look for npm-style package names in args
    for arg in reversed(args):
        if arg.startswith("-"):
            continue
        # @scope/mcp-server-name or mcp-server-name
        match = re.search(r"(?:mcp-server-|mcp-)(.+)$", arg)
        if match:
            return match.group(1)
        # Bare package name as last non-flag arg
        if "/" in arg:
            # @scope/package → package
            return arg.rsplit("/", 1)[-1]
        if not arg.startswith("."):
            return arg

    # Fallback: derive from command itself
    match = re.search(r"(?:mcp-server-|mcp-)(.+)$", command)
    if match:
        return match.group(1)

    # Last resort: command basename without extension
    name = command.rsplit("/", 1)[-1]
    return name.rsplit(".", 1)[0]


def _make_connector_id(server_name: str) -> str:
    """Generate a connector ID from the server name.

    sensitivity_tier: 1
    """
    # Lowercase, replace non-alphanumeric with hyphens
    slug = re.sub(r"[^a-z0-9]+", "-", server_name.lower()).strip("-")
    return f"custom-{slug}"


def _build_probe_args(
    tool: McpToolInfo,
) -> dict[str, Any]:
    """Build minimal arguments for probing a tool's sample output.

    sensitivity_tier: 1
    """
    schema = tool.input_schema
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    probe_args: dict[str, Any] = {}
    for param_name in required:
        param_lower = param_name.lower()
        if param_lower in _LIMIT_PARAM_NAMES:
            probe_args[param_name] = MAX_SAMPLE_RECORDS
        elif param_lower in _QUERY_PARAM_NAMES:
            probe_args[param_name] = ""
        else:
            # Try to infer type from schema
            param_schema = properties.get(param_name, {})
            param_type = param_schema.get("type", "string")
            if param_type == "number" or param_type == "integer":
                probe_args[param_name] = 0
            elif param_type == "boolean":
                probe_args[param_name] = False
            elif param_type == "array":
                probe_args[param_name] = []
            elif param_type == "object":
                probe_args[param_name] = {}
            else:
                probe_args[param_name] = ""

    # Add optional limit if not already required
    for param_name in properties:
        if (
            param_name.lower() in _LIMIT_PARAM_NAMES
            and param_name not in probe_args
        ):
            probe_args[param_name] = MAX_SAMPLE_RECORDS
            break

    return probe_args


def _build_create_table_ddl(
    table_name: str,
    fields: tuple[Any, ...],
    dedup_key: tuple[str, ...],
    default_tier: int = 2,
) -> str:
    """Generate CREATE TABLE IF NOT EXISTS DDL from discovered fields.

    sensitivity_tier: 1
    """
    columns: list[str] = []
    has_id = False

    for f in fields:
        col_def = f"    {f.target_column:<20} {f.target_type}"
        if f.target_column in dedup_key and f.target_column == "id":
            col_def += " PRIMARY KEY"
            has_id = True
        columns.append(col_def)

    # Add standard columns if not already present
    existing_cols = {f.target_column for f in fields}

    if "id" not in existing_cols and not has_id:
        columns.insert(0, "    id                   VARCHAR PRIMARY KEY")

    if "sensitivity_tier" not in existing_cols:
        columns.append(
            f"    sensitivity_tier     INTEGER NOT NULL DEFAULT {default_tier}"
        )

    if "created_at" not in existing_cols:
        columns.append(
            "    created_at           TEXT NOT NULL"
            " DEFAULT current_timestamp"
        )

    col_str = ",\n".join(columns)
    return f"CREATE TABLE IF NOT EXISTS {table_name} (\n{col_str}\n);"


def _get_existing_table_schemas(
    db_engine: DatabaseEngine | None,
) -> dict[str, list[str]]:
    """Get existing table names and their columns from DuckDB.

    sensitivity_tier: 1
    """
    if db_engine is None:
        return {}

    try:
        from src.core.sqlite.migrations import get_existing_tables

        tables = get_existing_tables(db_engine)
        result: dict[str, list[str]] = {}
        for table_name in tables:
            rows = db_engine.query(
                f"PRAGMA table_info({table_name})"
            )
            result[table_name] = [r["name"] for r in rows]
        return result
    except Exception:
        logger.warning("Failed to read existing table schemas", exc_info=True)
        return {}


def _discovery_cache_key(command: str, args: tuple[str, ...]) -> str:
    """Build a stable cache key for a command + args tuple.

    sensitivity_tier: 1
    """
    payload = json.dumps(
        {"command": command, "args": list(args)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def _serialize_preview(preview: InstallPreview) -> dict[str, Any]:
    """Serialize InstallPreview to a JSON-friendly dict.

    sensitivity_tier: 1
    """
    data = asdict(preview)
    data["args"] = list(preview.args)
    data["tools"] = [
        {
            "tool_name": t.tool_name,
            "tool_type": t.tool_type,
            "target_table": t.target_table,
            "is_new_table": t.is_new_table,
            "field_count": t.field_count,
            "sensitivity_tiers": dict(t.sensitivity_tiers),
            "confidence": t.confidence,
            "warnings": list(t.warnings),
        }
        for t in preview.tools
    ]
    data["new_tables"] = list(preview.new_tables)
    data["existing_tables"] = list(preview.existing_tables)
    data["warnings"] = list(preview.warnings)
    return data


def _deserialize_preview(data: dict[str, Any]) -> InstallPreview:
    """Deserialize InstallPreview from cached JSON data.

    sensitivity_tier: 1
    """
    tools: list[ToolPreview] = []
    for raw in data.get("tools", []):
        raw_tiers = raw.get("sensitivity_tiers", {})
        tiers: dict[int, int] = {}
        if isinstance(raw_tiers, dict):
            for key, value in raw_tiers.items():
                try:
                    tiers[int(key)] = int(value)
                except (TypeError, ValueError):
                    continue
        tools.append(ToolPreview(
            tool_name=str(raw.get("tool_name", "")),
            tool_type=str(raw.get("tool_type", "data")),
            target_table=raw.get("target_table"),
            is_new_table=bool(raw.get("is_new_table", False)),
            field_count=int(raw.get("field_count", 0)),
            sensitivity_tiers=tiers,
            confidence=float(raw.get("confidence", 0.0)),
            warnings=tuple(str(w) for w in raw.get("warnings", [])),
        ))

    return InstallPreview(
        server_name=str(data.get("server_name", "")),
        command=str(data.get("command", "")),
        args=tuple(str(a) for a in data.get("args", [])),
        tools=tuple(tools),
        data_tools=int(data.get("data_tools", 0)),
        action_tools=int(data.get("action_tools", 0)),
        new_tables=tuple(str(t) for t in data.get("new_tables", [])),
        existing_tables=tuple(str(t) for t in data.get("existing_tables", [])),
        overall_confidence=float(data.get("overall_confidence", 0.0)),
        warnings=tuple(str(w) for w in data.get("warnings", [])),
    )


def _serialize_mapping(mapping: DiscoveredMapping) -> dict[str, Any]:
    """Serialize DiscoveredMapping to cache JSON.

    sensitivity_tier: 1
    """
    return {
        "tool_name": mapping.tool_name,
        "target_table": mapping.target_table,
        "is_new_table": mapping.is_new_table,
        "domain": mapping.domain,
        "confidence": mapping.confidence,
        "analysis_method": mapping.analysis_method,
        "fields": [asdict(f) for f in mapping.fields],
        "dedup_key": list(mapping.dedup_key),
        "suggested_schedule": mapping.suggested_schedule,
        "unmapped_fields": list(mapping.unmapped_fields),
        "warnings": list(mapping.warnings),
    }


def _deserialize_mapping(data: dict[str, Any]) -> DiscoveredMapping:
    """Deserialize DiscoveredMapping from cache JSON.

    sensitivity_tier: 1
    """
    fields = tuple(
        FieldMapping(
            source_name=str(f.get("source_name", "")),
            target_column=str(f.get("target_column", "")),
            source_type=str(f.get("source_type", "string")),
            target_type=str(f.get("target_type", "VARCHAR")),
            sensitivity_tier=int(f.get("sensitivity_tier", 2)),
            confidence=float(f.get("confidence", 0.0)),
            tier_source=str(f.get("tier_source", "default")),
            transform=f.get("transform"),
            is_new_column=bool(f.get("is_new_column", False)),
        )
        for f in data.get("fields", [])
    )

    return DiscoveredMapping(
        tool_name=str(data.get("tool_name", "")),
        target_table=str(data.get("target_table", "")),
        is_new_table=bool(data.get("is_new_table", True)),
        domain=str(data.get("domain", "general")),
        confidence=float(data.get("confidence", 0.0)),
        analysis_method=str(data.get("analysis_method", "rules_only")),
        fields=fields,
        dedup_key=tuple(str(x) for x in data.get("dedup_key", [])),
        suggested_schedule=str(data.get("suggested_schedule", "daily")),
        unmapped_fields=tuple(
            str(x) for x in data.get("unmapped_fields", [])
        ),
        warnings=tuple(str(x) for x in data.get("warnings", [])),
    )


# ---------------------------------------------------------------------------
# ExtensionInstaller
# ---------------------------------------------------------------------------


class ExtensionInstaller:
    """Zero-friction MCP server installer.

    Two-phase flow:
    1. discover() — connect to server, probe tools, analyze schemas
    2. confirm() — create tables, register connector

    sensitivity_tier: 1
    """

    def __init__(
        self,
        db_engine: DatabaseEngine | None = None,
        registry: ExtensionRegistry | None = None,
        mcp_timeout: float = 10.0,
        llm_model: str = "llama3.1:8b",
        llm_host: str = "http://localhost:11434",
        cache_dir: Path | None = None,
    ) -> None:
        self._db_engine = db_engine
        self._registry = registry or ExtensionRegistry()
        self._mcp_timeout = mcp_timeout
        self._llm_model = llm_model
        self._llm_host = llm_host
        self._cache_dir = cache_dir or DEFAULT_DISCOVERY_CACHE_DIR
        # Cache discovered mappings between discover() and confirm()
        self._last_discovered: dict[str, DiscoveredMapping] = {}
        self._last_preview: InstallPreview | None = None

    def _cache_path(self, command: str, args: tuple[str, ...]) -> Path:
        """Return the cache file path for a command signature.

        sensitivity_tier: 1
        """
        cache_key = _discovery_cache_key(command, args)
        return self._cache_dir / f"{cache_key}.json"

    def _persist_discovery_cache(
        self,
        command: str,
        args: tuple[str, ...],
        preview: InstallPreview,
        mappings: dict[str, DiscoveredMapping],
    ) -> None:
        """Persist discovery preview + mappings for reuse in confirm().

        sensitivity_tier: 1
        """
        path = self._cache_path(command, args)
        payload = {
            "version": DISCOVERY_CACHE_VERSION,
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "command": command,
            "args": list(args),
            "preview": _serialize_preview(preview),
            "mappings": {
                tool_name: _serialize_mapping(mapping)
                for tool_name, mapping in mappings.items()
            },
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except OSError as exc:
            logger.warning("Failed to persist discovery cache %s: %s", path, exc)

    def _load_discovery_cache(
        self,
        command: str,
        args: tuple[str, ...],
    ) -> tuple[InstallPreview, dict[str, DiscoveredMapping]] | None:
        """Load cached discovery payload for a command signature.

        sensitivity_tier: 1
        """
        path = self._cache_path(command, args)
        if not path.exists():
            return None

        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load discovery cache %s: %s", path, exc)
            return None

        if payload.get("version") != DISCOVERY_CACHE_VERSION:
            return None

        try:
            preview = _deserialize_preview(payload.get("preview", {}))
            mappings_raw = payload.get("mappings", {})
            mappings = {
                str(tool_name): _deserialize_mapping(mapping_data)
                for tool_name, mapping_data in mappings_raw.items()
                if isinstance(mapping_data, dict)
            }
            return (preview, mappings)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to parse discovery cache payload: %s", exc)
            return None

    def _hydrate_mappings_for_confirm(
        self,
        preview: InstallPreview,
        env: dict[str, str] | None = None,
    ) -> None:
        """Ensure `_last_discovered` is available before confirm().

        sensitivity_tier: 1
        """
        if self._last_discovered:
            return

        # Only consult the on-disk cache when no secrets are in play;
        # the cache key doesn't include env, so a cached preview for the
        # same command+args could mask auth-dependent differences.
        if not env:
            cached = self._load_discovery_cache(
                preview.command, preview.args,
            )
            if cached is not None:
                cached_preview, mappings = cached
                self._last_preview = cached_preview
                self._last_discovered = mappings
                if self._last_discovered:
                    return

        # Last resort: re-run discover to rebuild mappings.
        try:
            self.discover(
                preview.command,
                preview.args,
                name=preview.server_name,
                env=env,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to re-hydrate discovery during confirm: %s", exc,
            )

    def discover(
        self,
        command: str,
        args: tuple[str, ...] = (),
        name: str | None = None,
        env: dict[str, str] | None = None,
    ) -> InstallPreview:
        """Connect to an MCP server and discover its capabilities.

        Args:
            command: The command to run the MCP server.
            args: Arguments for the command.
            name: Optional human-friendly name override.
            env: Extra environment variables to pass to the server
                (e.g. API tokens). Treated as secrets — never cached.

        Returns:
            InstallPreview with all discovered tools and schema analysis.

        Raises:
            McpConnectionError: If the server cannot be reached.

        sensitivity_tier: 1
        """
        # Discovery cache is only safe to reuse when no secrets are needed,
        # since the cache file lives on disk without encryption.
        if not env:
            cached = self._load_discovery_cache(command, args)
            if cached is not None:
                cached_preview, mappings = cached
                if name and cached_preview.server_name != name:
                    cached_preview = replace(cached_preview, server_name=name)
                self._last_discovered = mappings
                self._last_preview = cached_preview
                logger.info("Loaded discovery cache for command: %s", command)
                return cached_preview

        server_name = name or _derive_server_name(command, args)
        existing_tables = _get_existing_table_schemas(self._db_engine)

        # Initialize schema discovery agent
        agent = SchemaDiscoveryAgent(
            existing_tables=existing_tables,
            model=self._llm_model,
            host=self._llm_host,
        )

        tool_previews: list[ToolPreview] = []
        all_warnings: list[str] = []
        new_tables: list[str] = []
        matched_tables: list[str] = []
        self._last_discovered = {}

        with McpClient(
            command, args, timeout=self._mcp_timeout, env=env,
        ) as client:
            tools = client.list_tools()

            if not tools:
                all_warnings.append("Server exposes no tools")
                preview = InstallPreview(
                    server_name=server_name,
                    command=command,
                    args=args,
                    tools=(),
                    warnings=tuple(all_warnings),
                )
                self._last_preview = preview
                return preview

            for tool in tools:
                tool_type = classify_tool(tool)

                if tool_type == "action":
                    tool_previews.append(ToolPreview(
                        tool_name=tool.name,
                        tool_type="action",
                    ))
                    continue

                # DATA tool — probe for sample records
                sample_records = self._probe_tool(client, tool)

                if not sample_records:
                    tool_previews.append(ToolPreview(
                        tool_name=tool.name,
                        tool_type="data",
                        confidence=0.3,
                        warnings=(
                            "Could not get sample data; "
                            "schema may be incomplete",
                        ),
                    ))
                    all_warnings.append(
                        f"Tool '{tool.name}': no sample data obtained"
                    )
                    continue

                # Run schema discovery
                mapping = agent.discover(
                    tool_name=tool.name,
                    sample_records=sample_records,
                    tool_description=tool.description,
                )
                self._last_discovered[tool.name] = mapping

                # Build tier counts
                tier_counts: dict[int, int] = {}
                for f in mapping.fields:
                    tier_counts[f.sensitivity_tier] = (
                        tier_counts.get(f.sensitivity_tier, 0) + 1
                    )

                if mapping.is_new_table:
                    new_tables.append(mapping.target_table)
                else:
                    matched_tables.append(mapping.target_table)

                tool_previews.append(ToolPreview(
                    tool_name=tool.name,
                    tool_type="data",
                    target_table=mapping.target_table,
                    is_new_table=mapping.is_new_table,
                    field_count=len(mapping.fields),
                    sensitivity_tiers=tier_counts,
                    confidence=mapping.confidence,
                    warnings=mapping.warnings,
                ))
                all_warnings.extend(mapping.warnings)

        data_count = sum(
            1 for t in tool_previews if t.tool_type == "data"
        )
        action_count = sum(
            1 for t in tool_previews if t.tool_type == "action"
        )

        # Compute overall confidence
        data_confidences = [
            t.confidence for t in tool_previews if t.tool_type == "data"
        ]
        overall_confidence = (
            sum(data_confidences) / len(data_confidences)
            if data_confidences
            else 0.0
        )

        preview = InstallPreview(
            server_name=server_name,
            command=command,
            args=args,
            tools=tuple(tool_previews),
            data_tools=data_count,
            action_tools=action_count,
            new_tables=tuple(sorted(set(new_tables))),
            existing_tables=tuple(sorted(set(matched_tables))),
            overall_confidence=round(overall_confidence, 2),
            warnings=tuple(all_warnings),
        )
        self._last_preview = preview
        # Don't persist cache when env (secrets) were used — the cache key
        # only covers command+args, so a cached preview could be served on
        # a subsequent run with different (or no) secrets, masking real
        # auth failures.
        if not env:
            self._persist_discovery_cache(
                command, args, preview, self._last_discovered,
            )
        return preview

    def confirm(
        self,
        preview: InstallPreview,
        name: str | None = None,
        env: dict[str, str] | None = None,
    ) -> InstallResult:
        """Finalize the extension install: create tables and register.

        Args:
            preview: The InstallPreview from a prior discover() call.
            name: Optional name override for the connector.

        Returns:
            InstallResult with status, tables created, and tool count.

        sensitivity_tier: 1
        """
        server_name = name or preview.server_name
        connector_id = _make_connector_id(server_name)

        tables_created: list[str] = []
        tool_templates: list[ToolTemplate] = []

        try:
            self._hydrate_mappings_for_confirm(preview, env=env)

            # Create tables for data tools
            for tp in preview.tools:
                if tp.tool_type != "data" or not tp.target_table:
                    continue

                mapping = self._last_discovered.get(tp.tool_name)
                if mapping is None:
                    continue

                # Create new tables dynamically
                if tp.is_new_table and self._db_engine is not None:
                    max_tier = max(
                        (f.sensitivity_tier for f in mapping.fields),
                        default=2,
                    )
                    ddl = _build_create_table_ddl(
                        tp.target_table,
                        mapping.fields,
                        mapping.dedup_key,
                        default_tier=max_tier,
                    )
                    self._db_engine.execute(ddl)
                    tables_created.append(tp.target_table)
                    logger.info(
                        "Created table: %s", tp.target_table,
                    )

                # Convert to ToolTemplate
                tool_templates.append(to_tool_template(mapping))

            if preview.data_tools > 0 and not any(
                t.tool_type == "data" for t in tool_templates
            ):
                msg = (
                    "No discovered mappings available for data tools; "
                    "run discover before confirm"
                )
                raise RuntimeError(msg)

            # Add action tools as ToolTemplates
            for tp in preview.tools:
                if tp.tool_type == "action":
                    tool_templates.append(ToolTemplate(
                        tool_name=tp.tool_name,
                        tool_type="action",
                        target_table=None,
                    ))

            # Register in extension registry. env_values persists Tier 3
            # secrets needed to relaunch the MCP server on every sync.
            cmd_line = f"{preview.command} {' '.join(preview.args)}"
            self._registry.register(
                connector_id,
                tools_count=len(preview.tools),
                command_line=cmd_line.strip(),
                env_values=env or None,
            )

            # Persist install metadata for detail view
            self._save_install_metadata(
                connector_id, preview,
            )

            # Auto-generate pipeline models for new tables
            models_staged = self._stage_models_for_new_tables(
                preview, connector_id,
            )

            return InstallResult(
                status="installed",
                connector_id=connector_id,
                tables_created=tuple(tables_created),
                tools_registered=len(tool_templates),
                models_staged=models_staged,
            )

        except Exception as exc:
            logger.error(
                "Extension install failed: %s", exc, exc_info=True,
            )
            return InstallResult(
                status="error",
                connector_id=connector_id,
                error=str(exc),
            )

    def _save_install_metadata(
        self,
        connector_id: str,
        preview: InstallPreview,
    ) -> None:
        """Persist preview + discovered mappings for detail view.

        Saved to ``~/.arandu/extensions/{connector_id}/metadata.json``.

        sensitivity_tier: 1
        """
        base = (
            Path.home() / ".arandu" / "extensions" / connector_id
        )
        try:
            base.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning(
                "Skipping metadata persistence for %s: %s",
                connector_id,
                exc,
            )
            return

        # Build tools list with field-level detail
        tools: list[dict[str, Any]] = []
        for tp in preview.tools:
            tool_dict: dict[str, Any] = {
                "tool_name": tp.tool_name,
                "tool_type": tp.tool_type,
                "target_table": tp.target_table,
                "is_new_table": tp.is_new_table,
                "field_count": tp.field_count,
                "sensitivity_tiers": dict(tp.sensitivity_tiers),
                "confidence": tp.confidence,
                "warnings": list(tp.warnings),
            }
            mapping = self._last_discovered.get(tp.tool_name)
            if mapping is not None:
                tool_dict["fields"] = [
                    asdict(f) for f in mapping.fields
                ]
                tool_dict["dedup_key"] = list(mapping.dedup_key)
                tool_dict["domain"] = mapping.domain
            tools.append(tool_dict)

        metadata = {
            "server_name": preview.server_name,
            "command": preview.command,
            "args": list(preview.args),
            "tools": tools,
            "data_tools": preview.data_tools,
            "action_tools": preview.action_tools,
            "new_tables": list(preview.new_tables),
            "existing_tables": list(preview.existing_tables),
            "overall_confidence": preview.overall_confidence,
        }

        path = base / "metadata.json"
        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        except OSError as exc:
            logger.warning(
                "Failed to save install metadata for %s: %s",
                connector_id,
                exc,
            )
            return

        logger.info("Saved install metadata: %s", path)

    def _stage_models_for_new_tables(
        self,
        preview: InstallPreview,
        connector_id: str,
    ) -> int:
        """Generate and stage pipeline models for new tables.

        sensitivity_tier: 1
        """
        from src.extensions.ingestion.model_generator import ModelGenerator
        from src.extensions.ingestion.review_flow import ReviewFlow

        new_table_mappings = [
            (tp, self._last_discovered[tp.tool_name])
            for tp in preview.tools
            if tp.is_new_table
            and tp.tool_type == "data"
            and tp.target_table
            and tp.tool_name in self._last_discovered
        ]

        if not new_table_mappings:
            return 0

        model_gen = ModelGenerator(
            model=self._llm_model,
            host=self._llm_host,
        )
        review = ReviewFlow(db_engine=self._db_engine)
        staged = 0

        for tp, mapping in new_table_mappings:
            try:
                model_preview = model_gen.generate(mapping, connector_id)
                if model_preview.total_models > 0:
                    review.stage(model_preview)
                    staged += 1
                    logger.info(
                        "Staged %d models for tool '%s'",
                        model_preview.total_models,
                        tp.tool_name,
                    )
            except Exception as exc:
                logger.warning(
                    "Model generation failed for '%s': %s",
                    tp.tool_name,
                    exc,
                )

        return staged

    def _probe_tool(
        self,
        client: McpClient,
        tool: McpToolInfo,
    ) -> list[dict[str, Any]]:
        """Call a tool to get sample records for schema discovery.

        Tries with minimal arguments first. Captures up to
        MAX_SAMPLE_RECORDS records.

        sensitivity_tier: 1
        """
        try:
            probe_args = _build_probe_args(tool)
            records = client.call_tool(tool.name, probe_args)
            return records[:MAX_SAMPLE_RECORDS]
        except McpToolError as exc:
            logger.warning(
                "Tool probe failed for '%s': %s", tool.name, exc,
            )
            # Try with empty args as fallback
            if _build_probe_args(tool):
                try:
                    records = client.call_tool(tool.name, {})
                    return records[:MAX_SAMPLE_RECORDS]
                except (McpToolError, McpTimeoutError):
                    pass
            return []
        except McpTimeoutError as exc:
            logger.warning(
                "Tool probe timed out for '%s': %s", tool.name, exc,
            )
            return []
