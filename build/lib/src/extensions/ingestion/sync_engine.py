"""Sync engine — orchestrates ingestion adapters for a connector.

Creates one IngestionAdapter per data tool, runs them in sequence,
and aggregates results into a single ``SyncStats`` that the
``SyncScheduler`` expects.

sensitivity_tier: 2 (coordinates user data ingestion)
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.sqlite.engine import DatabaseEngine
from src.extensions.connectors.catalog import ConnectorCatalog
from src.extensions.connectors.registry import ExtensionRegistry
from src.extensions.connectors.sync_scheduler import SyncStats
from src.extensions.ingestion.adapter import (
    IngestionAdapter,
    SyncError,
    SyncResult,
)
from src.extensions.models import (
    FieldTemplate,
    ToolTemplate,
)

logger = logging.getLogger(__name__)

_EXTENSIONS_DIR = Path.home() / ".arandu" / "extensions"

# Connectors that sync via direct function calls (no MCP subprocess).
NATIVE_SYNC_CONNECTORS: frozenset[str] = frozenset({
    "apple-calendar",
    "apple-contacts",
    "apple-notes",
    "apple-mail",
    "apple-messages",
    "filesystem",
    "whatsapp",
})


class _NoopMcpClient:
    """Placeholder MCP client for store-driven WhatsApp sync."""

    def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        msg = f"Unexpected MCP tool call in noop client: {tool_name}"
        raise RuntimeError(msg)


class SyncEngine:
    """Orchestrates data ingestion for a single connector.

    Provides ``sync_connector(connector_id) -> SyncStats`` which is
    the callable that plugs into ``SyncScheduler.sync_fn``.

    sensitivity_tier: 2
    """

    def __init__(
        self,
        mcp_client_factory: Callable[
            [str, tuple[str, ...], float], Any
        ],
        db_engine: DatabaseEngine,
        catalog: ConnectorCatalog,
        registry: ExtensionRegistry,
        mcp_timeout: float = 30.0,
    ) -> None:
        """Initialise the sync engine.

        Args:
            mcp_client_factory: Creates an MCP client given
                ``(command, args, timeout)``. Must support the
                context-manager protocol.
            db_engine: DuckDB engine for reads and writes.
            catalog: Bundled connector catalog.
            registry: Enabled-extension registry.
            mcp_timeout: Timeout in seconds for MCP operations.

        sensitivity_tier: 1
        """
        self._factory = mcp_client_factory
        self._db = db_engine
        self._catalog = catalog
        self._registry = registry
        self._timeout = mcp_timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sync_connector(self, connector_id: str) -> SyncStats:
        """Run a full sync cycle for *connector_id*.

        1. Resolve connector config (bundled catalog or custom metadata).
        2. Open ONE MCP client for this sync call.
        3. Run each data tool through an ``IngestionAdapter``.
        4. Aggregate results into ``SyncStats``.

        This method is the ``sync_fn`` that ``SyncScheduler`` calls.

        sensitivity_tier: 2
        """
        started_at = datetime.now(tz=timezone.utc)
        start_mono = time.monotonic()

        try:
            command, args, data_tools = self._get_connector_config(
                connector_id,
            )
        except LookupError as exc:
            return SyncStats(
                connector_id=connector_id,
                started_at=started_at,
                completed_at=datetime.now(tz=timezone.utc),
                status="error",
                error=str(exc),
            )

        if not data_tools:
            return SyncStats(
                connector_id=connector_id,
                started_at=started_at,
                completed_at=datetime.now(tz=timezone.utc),
                status="success",
                rows_synced=0,
            )

        results: list[SyncResult] = []
        errors: list[str] = []

        if self._should_sync_without_live_mcp(connector_id, data_tools):
            self._sync_tools(
                connector_id, data_tools, _NoopMcpClient(), results, errors,
            )
        else:
            try:
                with self._factory(command, args, self._timeout) as client:
                    self._sync_tools(
                        connector_id, data_tools, client, results, errors,
                    )
            except Exception as exc:
                logger.exception(
                    "MCP connection failed for %s: %s",
                    connector_id,
                    exc,
                )
                return SyncStats(
                    connector_id=connector_id,
                    started_at=started_at,
                    completed_at=datetime.now(tz=timezone.utc),
                    status="error",
                    error=f"MCP connection failed: {exc}",
                    duration_seconds=round(
                        time.monotonic() - start_mono, 3,
                    ),
                )

        return self._aggregate_results(
            connector_id, results, errors, started_at, start_mono,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_connector_config(
        self,
        connector_id: str,
    ) -> tuple[str, tuple[str, ...], list[ToolTemplate]]:
        """Resolve ``(command, args, data_tools)`` for a connector.

        Tries the bundled catalog first, then falls back to custom
        install metadata at ``~/.arandu/extensions/{id}/metadata.json``.

        Raises ``LookupError`` if neither source has config.

        sensitivity_tier: 1
        """
        # 1) Try bundled catalog
        template = self._catalog.get(connector_id)
        if template is not None:
            data_tools = [
                t
                for t in template.tools
                if t.tool_type == "data" and t.target_table
            ]
            return (template.command, template.args, data_tools)

        # 2) Try custom metadata
        meta_path = _EXTENSIONS_DIR / connector_id / "metadata.json"
        if meta_path.exists():
            return self._parse_custom_metadata(
                connector_id, meta_path,
            )

        msg = f"No config found for connector '{connector_id}'"
        raise LookupError(msg)

    def _parse_custom_metadata(
        self,
        connector_id: str,
        path: Path,
    ) -> tuple[str, tuple[str, ...], list[ToolTemplate]]:
        """Parse custom connector metadata into tool templates.

        sensitivity_tier: 1
        """
        with path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        command = meta.get("command", "")
        args = tuple(meta.get("args", []))

        data_tools: list[ToolTemplate] = []
        for raw_tool in meta.get("tools", []):
            if raw_tool.get("tool_type") != "data":
                continue
            if not raw_tool.get("target_table"):
                continue

            fields: list[FieldTemplate] = []
            for raw_field in raw_tool.get("fields", []):
                fields.append(
                    FieldTemplate(
                        source_name=raw_field["source_name"],
                        target_column=raw_field["target_column"],
                        source_type=raw_field.get(
                            "source_type", "string",
                        ),
                        target_type=raw_field.get(
                            "target_type", "VARCHAR",
                        ),
                        sensitivity_tier=raw_field.get(
                            "sensitivity_tier", 2,
                        ),
                        transform=raw_field.get("transform"),
                    ),
                )

            data_tools.append(
                ToolTemplate(
                    tool_name=raw_tool["tool_name"],
                    tool_type="data",
                    target_table=raw_tool["target_table"],
                    fields=tuple(fields),
                    dedup_key=tuple(
                        raw_tool.get("dedup_key", []),
                    ),
                ),
            )

        return (command, args, data_tools)

    @staticmethod
    def _should_sync_without_live_mcp(
        connector_id: str,
        data_tools: list[ToolTemplate],
    ) -> bool:
        """Return True when data tools can run without live MCP calls."""
        return connector_id in NATIVE_SYNC_CONNECTORS

    def _sync_tools(
        self,
        connector_id: str,
        data_tools: list[ToolTemplate],
        client: Any,
        results: list[SyncResult],
        errors: list[str],
    ) -> None:
        """Run all adapters and collect results/errors."""
        for tool in data_tools:
            try:
                adapter = IngestionAdapter(
                    connector_id, tool, client, self._db,
                )
                result = adapter.sync()
                results.append(result)
            except SyncError as exc:
                logger.warning(
                    "Tool %s failed for %s: %s",
                    tool.tool_name,
                    connector_id,
                    exc,
                )
                errors.append(f"{tool.tool_name}: {exc}")

    def _aggregate_results(
        self,
        connector_id: str,
        results: list[SyncResult],
        errors: list[str],
        started_at: datetime,
        start_mono: float,
    ) -> SyncStats:
        """Combine per-tool results into a single ``SyncStats``.

        sensitivity_tier: 1
        """
        rows_synced = sum(
            r.rows_new + r.rows_updated for r in results
        )
        status = "error" if errors else "success"
        error = errors[0] if errors else None

        return SyncStats(
            connector_id=connector_id,
            started_at=started_at,
            completed_at=datetime.now(tz=timezone.utc),
            status=status,
            rows_synced=rows_synced,
            error=error,
            duration_seconds=round(
                time.monotonic() - start_mono, 3,
            ),
        )
