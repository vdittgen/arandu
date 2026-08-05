"""Data models for the connector catalog.

Defines the schema for pre-verified MCP server connectors, their tools,
and field mappings used by the extension system.

sensitivity_tier: 1 (catalog metadata contains no user data)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FieldTemplate:
    """A single field mapping from an MCP tool output to a DuckDB column.

    sensitivity_tier: 1
    """

    source_name: str  # field name returned by the MCP tool
    target_column: str  # DuckDB column name
    source_type: str  # "string", "number", "boolean", "array", "object"
    target_type: str  # SQLite SQL type: "TEXT", "INTEGER", "REAL", etc.
    sensitivity_tier: int  # 1, 2, or 3
    transform: str | None = None  # e.g. "iso_to_timestamp", "json_array"


@dataclass(frozen=True)
class ToolTemplate:
    """A tool exposed by an MCP server with pre-verified field mappings.

    sensitivity_tier: 1
    """

    tool_name: str  # MCP tool name, e.g. "list_calendar_events"
    tool_type: str  # "data" (produces rows) or "action" (side-effect only)
    target_table: str | None  # DuckDB table name, None for actions
    fields: tuple[FieldTemplate, ...] = ()
    dedup_key: tuple[str, ...] = ()
    input_schema: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConnectorTemplate:
    """A pre-verified MCP server connector in the standard catalog.

    sensitivity_tier: 1
    """

    id: str  # unique slug, e.g. "apple-calendar"
    name: str  # human-readable name
    category: str  # "apple" | "files" | "email" | "notes" | "lifestyle"
    icon: str  # emoji or icon name
    description: str

    # How to run the MCP server
    command: str  # e.g. "npx"
    args: tuple[str, ...]  # e.g. ("-y", "@supermemoryai/apple-mcp")
    transport: str  # "stdio"

    # Pre-verified field mappings
    tools: tuple[ToolTemplate, ...]

    # Requirements
    requires_auth: str | None = None  # "google_oauth", "spotify_oauth", etc.
    requires_env: dict[str, str] = field(default_factory=dict)
    requires_app: str | None = None  # "WhatsApp Desktop", etc.
    requires_permission: str | None = None  # "macOS Calendar", etc.

    # Defaults
    default_enabled: bool = False
    default_schedule: str = "hourly"
    estimated_first_sync_seconds: int = 30

    # Availability
    platforms: tuple[str, ...] = ("macos",)
    min_version: str | None = None

    # Optional note displayed during setup
    note: str | None = None
