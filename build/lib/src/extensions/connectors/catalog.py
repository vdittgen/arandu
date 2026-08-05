"""Standard connector catalog — bundled registry of pre-verified MCP servers.

Loads connector templates from the bundled catalog_data.json and provides
lookup, filtering, and search capabilities.

sensitivity_tier: 1 (catalog metadata contains no user data)
"""

from __future__ import annotations

import json
import logging
import platform
from pathlib import Path
from typing import Any

from src.extensions.models import ConnectorTemplate, FieldTemplate, ToolTemplate

logger = logging.getLogger(__name__)

_CATALOG_PATH = Path(__file__).parent / "catalog_data.json"


def _parse_field(raw: dict[str, Any]) -> FieldTemplate:
    """Parse a single field mapping from JSON.

    sensitivity_tier: 1
    """
    return FieldTemplate(
        source_name=raw["source_name"],
        target_column=raw["target_column"],
        source_type=raw["source_type"],
        target_type=raw["target_type"],
        sensitivity_tier=raw["sensitivity_tier"],
        transform=raw.get("transform"),
    )


def _parse_tool(raw: dict[str, Any]) -> ToolTemplate:
    """Parse a single tool template from JSON.

    sensitivity_tier: 1
    """
    return ToolTemplate(
        tool_name=raw["tool_name"],
        tool_type=raw["tool_type"],
        target_table=raw.get("target_table"),
        fields=tuple(_parse_field(f) for f in raw.get("fields", [])),
        dedup_key=tuple(raw.get("dedup_key", [])),
        input_schema=raw.get("input_schema", {}),
    )


def _parse_connector(raw: dict[str, Any]) -> ConnectorTemplate:
    """Parse a single connector template from JSON.

    sensitivity_tier: 1
    """
    return ConnectorTemplate(
        id=raw["id"],
        name=raw["name"],
        category=raw["category"],
        icon=raw["icon"],
        description=raw["description"],
        command=raw["command"],
        args=tuple(raw["args"]),
        transport=raw["transport"],
        tools=tuple(_parse_tool(t) for t in raw.get("tools", [])),
        requires_auth=raw.get("requires_auth"),
        requires_env=raw.get("requires_env", {}),
        requires_app=raw.get("requires_app"),
        requires_permission=raw.get("requires_permission"),
        default_enabled=raw.get("default_enabled", False),
        default_schedule=raw.get("default_schedule", "hourly"),
        estimated_first_sync_seconds=raw.get("estimated_first_sync_seconds", 30),
        platforms=tuple(raw.get("platforms", ["macos"])),
        min_version=raw.get("min_version"),
        note=raw.get("note"),
    )


def _current_platform() -> str:
    """Return the normalised platform name.

    sensitivity_tier: 1
    """
    system = platform.system().lower()
    if system == "darwin":
        return "macos"
    return system  # "linux", "windows"


class ConnectorCatalog:
    """Registry of pre-verified MCP server connectors.

    Loads from the bundled catalog_data.json file. Provides filtering
    by platform, category, enabled state, and free-text search.

    sensitivity_tier: 1
    """

    def __init__(self, catalog_path: Path = _CATALOG_PATH) -> None:
        self._connectors: list[ConnectorTemplate] = []
        self._by_id: dict[str, ConnectorTemplate] = {}
        self._load(catalog_path)

    def _load(self, path: Path) -> None:
        """Load and parse the catalog JSON file.

        sensitivity_tier: 1
        """
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        for raw in data.get("connectors", []):
            connector = _parse_connector(raw)
            self._connectors.append(connector)
            self._by_id[connector.id] = connector

        logger.info("Loaded %d connectors from catalog", len(self._connectors))

    @property
    def all(self) -> list[ConnectorTemplate]:
        """Return all connectors in the catalog.

        sensitivity_tier: 1
        """
        return list(self._connectors)

    def get(self, connector_id: str) -> ConnectorTemplate | None:
        """Look up a connector by ID.

        sensitivity_tier: 1
        """
        return self._by_id.get(connector_id)

    def get_available(
        self, target_platform: str | None = None,
    ) -> list[ConnectorTemplate]:
        """Return connectors available on the given (or current) platform.

        Args:
            target_platform: Override platform detection. If None, auto-detect.

        sensitivity_tier: 1
        """
        plat = target_platform or _current_platform()
        return [c for c in self._connectors if plat in c.platforms]

    def get_by_category(
        self, target_platform: str | None = None,
    ) -> dict[str, list[ConnectorTemplate]]:
        """Return available connectors grouped by category.

        sensitivity_tier: 1
        """
        result: dict[str, list[ConnectorTemplate]] = {}
        for c in self.get_available(target_platform):
            result.setdefault(c.category, []).append(c)
        return result

    def get_enabled(self) -> list[ConnectorTemplate]:
        """Return connectors that are enabled by default.

        sensitivity_tier: 1
        """
        return [c for c in self._connectors if c.default_enabled]

    def search(self, query: str) -> list[ConnectorTemplate]:
        """Filter connectors whose name or description matches the query.

        Case-insensitive substring match on name, description, and category.

        sensitivity_tier: 1
        """
        q = query.lower()
        return [
            c
            for c in self._connectors
            if q in c.name.lower()
            or q in c.description.lower()
            or q in c.category.lower()
        ]

    def get_all_target_tables(self) -> set[str]:
        """Return the set of all DuckDB tables referenced by data tools.

        sensitivity_tier: 1
        """
        tables: set[str] = set()
        for connector in self._connectors:
            for tool in connector.tools:
                if tool.tool_type == "data" and tool.target_table:
                    tables.add(tool.target_table)
        return tables
