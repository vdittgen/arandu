"""Connection manager — orchestrates connector enable/disable flow.

Handles the full lifecycle: requirement checking, table creation,
registry management, and sync scheduling.

sensitivity_tier: 1 (manages connector state, no user data accessed)
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.core.sqlite.engine import DatabaseEngine
from src.core.sqlite.migrations import ensure_table
from src.extensions.connectors.catalog import ConnectorCatalog
from src.extensions.connectors.registry import ExtensionRegistry
from src.extensions.connectors.requirements import (
    MANUAL_GRANT_PERMISSIONS,
    RequirementChecker,
)
from src.extensions.connectors.sync_scheduler import SyncScheduler, SyncStats
from src.extensions.ingestion.sync_engine import NATIVE_SYNC_CONNECTORS, SyncEngine
from src.extensions.mcp.client import McpClient
from src.extensions.models import ConnectorTemplate

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnableResult:
    """Result of enabling a connector.

    sensitivity_tier: 1
    """

    status: str  # "connected" | "needs_setup" | "error"
    connector_id: str = ""
    records_synced: int = 0
    tools_available: int = 0
    next_sync_at: str | None = None
    missing: list[dict[str, str]] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class DisableResult:
    """Result of disabling a connector.

    sensitivity_tier: 1
    """

    status: str  # "disabled" | "error"
    connector_id: str = ""
    data_preserved: bool = True
    error: str | None = None


@dataclass(frozen=True)
class ConnectorStatus:
    """Full status of a connector for the UI.

    sensitivity_tier: 1
    """

    connector_id: str
    name: str
    icon: str
    description: str
    category: str
    enabled: bool
    # "connected" | "needs_setup" | "syncing" | "error" | "disabled"
    status: str
    records_synced: int = 0
    last_sync: str | None = None
    next_sync: str | None = None
    tools_available: int = 0
    missing_requirements: list[dict[str, str]] = field(
        default_factory=list,
    )
    error: str | None = None


class ConnectionManager:
    """Orchestrates the connector toggle-on/toggle-off flow.

    sensitivity_tier: 1
    """

    def __init__(
        self,
        catalog: ConnectorCatalog | None = None,
        registry: ExtensionRegistry | None = None,
        scheduler: SyncScheduler | None = None,
        checker: RequirementChecker | None = None,
        db_engine: DatabaseEngine | None = None,
        mcp_timeout: float = 30.0,
        mcp_client_factory: Callable[
            [str, tuple[str, ...], float], Any
        ] | None = None,
        sync_engine: SyncEngine | None = None,
    ) -> None:
        self.catalog = catalog or ConnectorCatalog()
        self.registry = registry or ExtensionRegistry()
        self.checker = checker or RequirementChecker()
        self.db_engine = db_engine
        self._mcp_timeout = mcp_timeout
        self._mcp_client_factory = (
            mcp_client_factory
            if mcp_client_factory is not None
            else self._default_mcp_client_factory
        )

        # Build SyncEngine when db_engine is available
        if sync_engine is not None:
            self._sync_engine: SyncEngine | None = sync_engine
        elif db_engine is not None:
            self._sync_engine = SyncEngine(
                mcp_client_factory=self._mcp_client_factory,
                db_engine=db_engine,
                catalog=self.catalog,
                registry=self.registry,
                mcp_timeout=mcp_timeout,
            )
        else:
            self._sync_engine = None

        # Wire sync_fn into scheduler
        sync_fn = (
            self._sync_engine.sync_connector
            if self._sync_engine
            else None
        )
        self.scheduler = scheduler or SyncScheduler(
            sync_fn=sync_fn,
        )

    def enable_connector(
        self,
        connector_id: str,
        user_inputs: dict[str, Any] | None = None,
    ) -> EnableResult:
        """Enable a connector (called when user toggles ON).

        Flow:
        1. Load ConnectorTemplate from catalog
        2. Check requirements (auth, env vars, permissions, apps)
        3. If requirements met: create tables, register, schedule
        4. If requirements not met: return what's needed

        sensitivity_tier: 1
        """
        user_inputs = user_inputs or {}

        template = self.catalog.get(connector_id)
        if template is None:
            return EnableResult(
                status="error",
                connector_id=connector_id,
                error=f"Unknown connector: {connector_id}",
            )

        # Allow the caller to provide OAuth material in user_inputs.
        self._hydrate_oauth_requirements(template, user_inputs)

        # Step 1: Check requirements
        req_status = self.checker.check_all(
            requires_permission=template.requires_permission,
            requires_auth=template.requires_auth,
            requires_env=template.requires_env or None,
            requires_app=template.requires_app,
            user_inputs=user_inputs,
        )

        if not req_status.all_met:
            # Some missing requirements can be resolved at runtime by the
            # MCP server itself: a connector that scripts Calendar.app via
            # AppleScript will trigger the Automation prompt on its first
            # call, so we can let it through and let macOS handle the
            # dialog. Others — OAuth, env vars, and any permission in
            # MANUAL_GRANT_PERMISSIONS (today: Full Disk Access) — have
            # no runtime prompt and must be set up before we enable, or
            # the connector silently registers as "connected" while its
            # read path fails permission-denied on every sync.
            only_permission_missing = all(
                m.requirement_type == "permission"
                for m in req_status.missing
            )
            has_manual_grant_permission = any(
                m.requirement_type == "permission"
                and m.key in MANUAL_GRANT_PERMISSIONS
                for m in req_status.missing
            )
            if not only_permission_missing or has_manual_grant_permission:
                self.registry.register_needs_setup(
                    connector_id,
                    missing=[
                        {
                            "type": m.requirement_type,
                            "key": m.key,
                            "label": m.label,
                            "action": m.action,
                        }
                        for m in req_status.missing
                    ],
                )
                return EnableResult(
                    status="needs_setup",
                    connector_id=connector_id,
                    missing=[
                        {
                            "type": m.requirement_type,
                            "key": m.key,
                            "label": m.label,
                            "action": m.action,
                        }
                        for m in req_status.missing
                    ],
                )
            # Only runtime-promptable permissions are missing — proceed
            # to MCP handshake and let the first API call trigger the
            # macOS Automation dialog.
            logger.info(
                "Permission '%s' not yet granted for %s — "
                "attempting MCP handshake to trigger macOS dialog",
                template.requires_permission,
                connector_id,
            )

        # Step 2: Ensure tables exist
        self._ensure_tables(template)

        # Step 3: Register in extension registry
        env_vals = {
            k: str(user_inputs[k])
            for k in (template.requires_env or {})
            if k in user_inputs
        }
        self.registry.register(connector_id, env_values=env_vals)

        # Step 4: Verify MCP lifecycle (handshake + tools/list)
        # Native-sync connectors bypass MCP entirely — no subprocess needed.
        if connector_id in NATIVE_SYNC_CONNECTORS:
            tools_count = len(template.tools)
            self.registry.update_tools_count(connector_id, tools_count)
        else:
            tools_count, handshake_error = self._discover_runtime_tools(
                template,
            )
            if handshake_error:
                self.registry.update_tools_count(connector_id, 0)
                self.registry.update_status(
                    connector_id,
                    status="error",
                    error=handshake_error,
                )
                return EnableResult(
                    status="error",
                    connector_id=connector_id,
                    error=handshake_error,
                )
            self.registry.update_tools_count(connector_id, tools_count)

        # Step 4.5: Connector-specific setup handshakes (QR pairing etc.)
        extra_setup_missing = self._resolve_extra_setup_missing(template)
        if extra_setup_missing:
            self.registry.register_needs_setup(
                connector_id,
                missing=extra_setup_missing,
            )
            return EnableResult(
                status="needs_setup",
                connector_id=connector_id,
                missing=extra_setup_missing,
            )

        # Step 4.75: Start connector runtime services (if any).
        try:
            self._ensure_connector_runtime_services(template)
        except Exception as exc:  # noqa: BLE001
            self.registry.update_status(
                connector_id,
                status="error",
                error=str(exc),
            )
            return EnableResult(
                status="error",
                connector_id=connector_id,
                tools_available=tools_count,
                error=f"Runtime service failed: {exc}",
            )

        # Step 5: Schedule syncs
        self.scheduler.schedule(
            connector_id, template.default_schedule,
        )

        # Step 6: Run first sync immediately.  "skipped" (a sync for
        # this connector is already in flight) is not a failure — the
        # running sync delivers the data this one would have.
        first_sync = self.sync_now(connector_id)
        if first_sync.status not in ("success", "skipped"):
            return EnableResult(
                status="error",
                connector_id=connector_id,
                tools_available=tools_count,
                error=first_sync.error or "First sync failed",
            )

        # Build next_sync time
        next_times = self.scheduler.get_next_sync_times()
        next_sync = next_times.get(connector_id)

        return EnableResult(
            status="connected",
            connector_id=connector_id,
            records_synced=first_sync.rows_synced,
            tools_available=tools_count,
            next_sync_at=(
                next_sync.isoformat() if next_sync else None
            ),
        )

    def disable_connector(
        self, connector_id: str,
    ) -> DisableResult:
        """Disable a connector (called when user toggles OFF).

        Stops scheduled syncs and marks as disabled.
        Does NOT delete data — user may re-enable.

        sensitivity_tier: 1
        """
        template = self.catalog.get(connector_id)
        if template is None:
            return DisableResult(
                status="error",
                connector_id=connector_id,
                error=f"Unknown connector: {connector_id}",
            )

        # Stop scheduled syncs
        self.scheduler.unschedule(connector_id)

        # Stop connector runtime services (if any).
        try:
            self._stop_connector_runtime_services(template)
        except Exception:  # noqa: BLE001
            logger.warning(
                "Failed stopping runtime services for %s",
                connector_id,
                exc_info=True,
            )

        # Mark as disabled in registry
        self.registry.unregister(connector_id)

        return DisableResult(
            status="disabled",
            connector_id=connector_id,
            data_preserved=True,
        )

    def reconnect(
        self,
        connector_id: str,
        user_inputs: dict[str, Any] | None = None,
    ) -> EnableResult:
        """Re-enable a previously connected connector.

        Data still exists, just re-check requirements and restart.

        sensitivity_tier: 1
        """
        return self.enable_connector(connector_id, user_inputs)

    def sync_now(self, connector_id: str) -> SyncStats:
        """Trigger an immediate sync for a connector.

        Updates the registry timestamp so the UI shows the sync.

        sensitivity_tier: 1
        """
        template = self.catalog.get(connector_id)
        if template is not None:
            self._ensure_connector_runtime_services(template)
            self._ensure_tables(template)

        stats = self.scheduler.run_now(connector_id)
        # Persist sync timestamp in registry for catalog refetches
        self.registry.update_sync_stats(
            connector_id,
            records_synced=stats.rows_synced,
            error=stats.error,
        )
        return stats

    def get_connector_catalog(self) -> list[dict[str, Any]]:
        """Return all connectors with their current status.

        Each entry includes template info + enabled state + sync stats.
        This is what the UI calls to render the connector list.

        sensitivity_tier: 1
        """
        result: list[dict[str, Any]] = []
        next_times = self.scheduler.get_next_sync_times()

        for template in self.catalog.get_available():
            registered = self.registry.get(template.id)
            enabled = registered is not None and registered.enabled

            # Determine status
            if not enabled:
                status = "disabled"
            elif registered and registered.status:
                status = registered.status
            else:
                status = "disabled"

            # Check requirements for disabled or needs_setup connectors
            missing: list[dict[str, str]] = []
            if status == "needs_setup" and registered:
                if registered.missing_requirements:
                    missing = list(registered.missing_requirements)
                else:
                    req = self.checker.check_all(
                        requires_permission=template.requires_permission,
                        requires_auth=template.requires_auth,
                        requires_env=template.requires_env or None,
                        requires_app=template.requires_app,
                    )
                    missing = [
                        {
                            "type": m.requirement_type,
                            "key": m.key,
                            "label": m.label,
                            "action": m.action,
                        }
                        for m in req.missing
                    ]
            elif not enabled:
                req = self.checker.check_all(
                    requires_permission=template.requires_permission,
                    requires_auth=template.requires_auth,
                    requires_env=template.requires_env or None,
                    requires_app=template.requires_app,
                )
                missing = [
                    {
                        "type": m.requirement_type,
                        "key": m.key,
                        "label": m.label,
                        "action": m.action,
                    }
                    for m in req.missing
                ]

            # A connector can get "stuck" in needs_setup after the
            # user installs/primes prerequisites outside the app.
            # Treat that as retryable (toggle-on again) in the UI.
            if status == "needs_setup" and not missing:
                enabled = False
                status = "disabled"

            next_sync = next_times.get(template.id)

            result.append({
                "connector_id": template.id,
                "name": template.name,
                "icon": template.icon,
                "description": template.description,
                "category": template.category,
                "enabled": enabled,
                "status": status,
                "stats": {
                    "records_synced": (
                        registered.records_synced if registered else 0
                    ),
                    "last_sync": (
                        registered.last_sync_at if registered else None
                    ),
                    "last_success": (
                        registered.last_success_at if registered else None
                    ),
                    "error": registered.error if registered else None,
                    "next_sync": (
                        next_sync.isoformat() if next_sync else None
                    ),
                },
                "missing_requirements": missing,
                "tools_available": len(template.tools),
                "default_schedule": template.default_schedule,
                "note": template.note,
            })

        # Include user-installed extensions not in the bundled catalog
        bundled_ids = {t.id for t in self.catalog.get_available()}
        for ext in self.registry.get_all():
            if ext.connector_id in bundled_ids:
                continue
            next_sync = next_times.get(ext.connector_id)
            display_name = ext.connector_id.replace("custom-", "", 1)
            display_name = display_name.replace("-", " ").title()
            result.append({
                "connector_id": ext.connector_id,
                "name": display_name,
                "icon": "puzzle",
                "description": "User-installed MCP extension",
                "category": "custom",
                "enabled": ext.enabled,
                "status": ext.status or "disabled",
                "stats": {
                    "records_synced": ext.records_synced,
                    "last_sync": ext.last_sync_at,
                    "last_success": ext.last_success_at,
                    "error": ext.error,
                    "next_sync": (
                        next_sync.isoformat() if next_sync else None
                    ),
                },
                "missing_requirements": [],
                "tools_available": ext.tools_count,
                "default_schedule": "manual",
                "note": ext.command_line or None,
            })

        return result

    def get_connector_details(
        self, connector_id: str,
    ) -> dict[str, Any] | None:
        """Return full details for a single connector.

        Includes field mappings, sync history, schedule info.

        sensitivity_tier: 1
        """
        template = self.catalog.get(connector_id)
        registered = self.registry.get(connector_id)
        schedule_info = self.scheduler.get_schedule_info(connector_id)

        # User-installed extension (not in bundled catalog)
        if template is None:
            if registered is None:
                return None
            return self._build_custom_details(
                connector_id, registered, schedule_info,
            )

        tools_info = []
        for tool in template.tools:
            tool_dict: dict[str, Any] = {
                "tool_name": tool.tool_name,
                "tool_type": tool.tool_type,
                "target_table": tool.target_table,
                "field_count": len(tool.fields),
                "dedup_key": list(tool.dedup_key),
            }
            if tool.fields:
                tool_dict["fields"] = [
                    {
                        "source": f.source_name,
                        "target": f.target_column,
                        "type": f.target_type,
                        "tier": f.sensitivity_tier,
                        "transform": f.transform,
                    }
                    for f in tool.fields
                ]
            tools_info.append(tool_dict)

        return {
            "connector_id": template.id,
            "name": template.name,
            "icon": template.icon,
            "description": template.description,
            "category": template.category,
            "command": template.command,
            "args": list(template.args),
            "transport": template.transport,
            "enabled": (
                registered is not None and registered.enabled
            ),
            "status": (
                registered.status if registered else "disabled"
            ),
            "requires_auth": template.requires_auth,
            "requires_env": template.requires_env,
            "requires_app": template.requires_app,
            "requires_permission": template.requires_permission,
            "default_schedule": template.default_schedule,
            "platforms": list(template.platforms),
            "note": template.note,
            "tools": tools_info,
            "schedule": schedule_info,
            "stats": {
                "records_synced": (
                    registered.records_synced if registered else 0
                ),
                "last_sync": (
                    registered.last_sync_at if registered else None
                ),
                "last_success": (
                    registered.last_success_at if registered else None
                ),
                "error": (
                    registered.error if registered else None
                ),
            },
        }

    @staticmethod
    def _load_install_metadata(
        connector_id: str,
    ) -> dict[str, Any] | None:
        """Load persisted install metadata for a user-installed extension.

        sensitivity_tier: 1
        """
        path = (
            Path.home()
            / ".arandu"
            / "extensions"
            / connector_id
            / "metadata.json"
        )
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Failed to load metadata for %s: %s",
                connector_id, exc,
            )
            return None

    def _build_custom_details(
        self,
        connector_id: str,
        registered: Any,
        schedule_info: Any,
    ) -> dict[str, Any]:
        """Build details dict for a user-installed extension.

        Loads saved install metadata for tool and field information.

        sensitivity_tier: 1
        """
        display = connector_id.replace("custom-", "", 1)
        display = display.replace("-", " ").title()

        meta = self._load_install_metadata(connector_id)

        # Build tools list from metadata
        tools_info: list[dict[str, Any]] = []
        if meta and meta.get("tools"):
            for t in meta["tools"]:
                tool_dict: dict[str, Any] = {
                    "tool_name": t.get("tool_name", ""),
                    "tool_type": t.get("tool_type", "data"),
                    "target_table": t.get("target_table"),
                    "field_count": t.get("field_count", 0),
                    "dedup_key": t.get("dedup_key", []),
                }
                if t.get("fields"):
                    tool_dict["fields"] = [
                        {
                            "source": f.get("source_name", ""),
                            "target": f.get("target_column", ""),
                            "type": f.get("target_type", "VARCHAR"),
                            "tier": f.get("sensitivity_tier", 2),
                            "transform": f.get("transform"),
                        }
                        for f in t["fields"]
                    ]
                tools_info.append(tool_dict)

        cmd_parts = (
            registered.command_line.split()
            if registered.command_line
            else []
        )

        return {
            "connector_id": connector_id,
            "name": meta.get("server_name", display) if meta else display,
            "icon": "puzzle",
            "description": "User-installed MCP extension",
            "category": "custom",
            "command": cmd_parts[0] if cmd_parts else "",
            "args": cmd_parts[1:],
            "transport": "stdio",
            "enabled": registered.enabled,
            "status": registered.status or "disabled",
            "requires_auth": None,
            "requires_env": None,
            "requires_app": None,
            "requires_permission": None,
            "default_schedule": "manual",
            "platforms": ["macos"],
            "note": registered.command_line or None,
            "tools": tools_info,
            "schedule": schedule_info,
            "stats": {
                "records_synced": registered.records_synced,
                "last_sync": registered.last_sync_at,
                "last_success": registered.last_success_at,
                "error": registered.error,
            },
        }

    def toggle_connector(
        self,
        connector_id: str,
        enabled: bool,
        user_inputs: dict[str, Any] | None = None,
    ) -> EnableResult | DisableResult:
        """Single toggle call — the UI entry point.

        sensitivity_tier: 1
        """
        if enabled:
            return self.enable_connector(connector_id, user_inputs)
        return self.disable_connector(connector_id)

    @staticmethod
    def _ensure_connector_runtime_services(
        template: ConnectorTemplate,
    ) -> None:
        """Start long-lived runtime services required by a connector."""
        if template.id != "whatsapp":
            return

        from src.extensions.bridges.whatsapp.listener import WhatsAppListenerService

        status = WhatsAppListenerService().ensure_running(
            template.command,
            template.args,
        )
        if not bool(status.get("running")):
            msg = "WhatsApp listener failed to start"
            raise RuntimeError(msg)

    @staticmethod
    def _stop_connector_runtime_services(
        template: ConnectorTemplate,
    ) -> None:
        """Stop long-lived runtime services for a connector."""
        if template.id != "whatsapp":
            return

        from src.extensions.bridges.whatsapp.listener import WhatsAppListenerService

        WhatsAppListenerService().stop()

    def _ensure_tables(self, template: ConnectorTemplate) -> None:
        """Create any missing DuckDB tables for the connector's tools.

        sensitivity_tier: 1
        """
        if self.db_engine is None:
            logger.debug(
                "No db_engine — skipping table creation for %s",
                template.id,
            )
            return

        from src.core.sqlite.migrations import (
            MIGRATION_SCHEMAS,
            run_column_additions,
        )
        from src.core.sqlite.schemas import create_all_tables

        # Ensure baseline raw tables exist even on first run where only a
        # connector toggle path is executed (without a prior full init).
        create_all_tables(self.db_engine)

        for tool in template.tools:
            if tool.tool_type == "data" and tool.target_table:
                if tool.target_table in MIGRATION_SCHEMAS:
                    ensure_table(self.db_engine, tool.target_table)

        # Also run column additions for existing tables
        run_column_additions(self.db_engine)

    @staticmethod
    def _default_mcp_client_factory(
        command: str,
        args: tuple[str, ...],
        timeout: float,
    ) -> McpClient:
        """Default MCP client factory used for runtime handshake checks."""
        return McpClient(command=command, args=args, timeout=timeout)

    def _discover_runtime_tools(
        self,
        template: ConnectorTemplate,
    ) -> tuple[int, str | None]:
        """Run a lightweight runtime health check by calling tools/list."""
        try:
            with self._mcp_client_factory(
                template.command,
                template.args,
                self._mcp_timeout,
            ) as client:
                tools = client.list_tools()
        except Exception as exc:  # noqa: BLE001
            return (0, f"MCP handshake failed: {exc}")
        if not tools:
            return (0, "Server started but not responding")
        return (len(tools), None)

    def _resolve_extra_setup_missing(
        self,
        template: ConnectorTemplate,
    ) -> list[dict[str, str]]:
        """Return connector-specific setup blockers after base handshake."""
        if template.id != "whatsapp":
            return []

        try:
            with self._mcp_client_factory(
                template.command,
                template.args,
                self._mcp_timeout,
            ) as client:
                tools = {t.name for t in client.list_tools()}
                if "get_connection_status" not in tools:
                    return []
                status_rows = client.call_tool("get_connection_status", {})
        except Exception as exc:  # noqa: BLE001
            logger.info(
                "Could not determine WhatsApp connection status: %s",
                exc,
            )
            return []

        if self._is_whatsapp_connected(status_rows):
            return []

        has_stored_creds = self._has_whatsapp_stored_credentials(status_rows)

        # Align with Baileys/OpenClaw flow: attempt one explicit connect.
        # This can recover from transient credential-state issues (for
        # example, creds restored from backup) without requiring the user to
        # manually re-trigger setup.
        if "connect" in tools:
            try:
                with self._mcp_client_factory(
                    template.command,
                    template.args,
                    self._mcp_timeout,
                ) as client:
                    _ = client.call_tool("connect", {})
                    status_rows = client.call_tool(
                        "get_connection_status", {},
                    )
            except Exception as exc:  # noqa: BLE001
                logger.info(
                    "WhatsApp connect probe failed: %s",
                    exc,
                )
            else:
                if self._is_whatsapp_connected(status_rows):
                    return []
                has_stored_creds = (
                    has_stored_creds
                    or self._has_whatsapp_stored_credentials(status_rows)
                )

        # If credentials exist, do not hard-block as setup-required.
        # Let first sync attempt surface any actionable runtime error.
        if has_stored_creds:
            return []

        return [
            {
                "type": "setup",
                "key": "whatsapp_qr",
                "label": (
                    "WhatsApp is not paired yet. Complete QR pairing in the "
                    "WhatsApp MCP setup flow, then click Retry Connection."
                ),
                "action": "scan_qr",
            }
        ]

    @staticmethod
    def _is_whatsapp_connected(rows: list[dict[str, Any]]) -> bool:
        """Best-effort check for connected status from MCP status payloads."""
        if not rows:
            return False

        for row in rows:
            for key in ("connected", "is_connected", "authenticated", "ready"):
                value = row.get(key)
                if isinstance(value, bool) and value:
                    return True
            status = str(row.get("status", "")).strip().lower()
            if status in {"connected", "ready", "authenticated", "online"}:
                return True
            raw = str(row.get("_raw_text", "")).strip().lower()
            if raw:
                if "not connected" in raw or "not authenticated" in raw:
                    continue
                if "connected" in raw or "authenticated" in raw:
                    return True

        return False

    @staticmethod
    def _has_whatsapp_stored_credentials(rows: list[dict[str, Any]]) -> bool:
        """Best-effort check for persisted WhatsApp credential state."""
        for row in rows:
            has_creds = row.get("hasStoredCredentials")
            if isinstance(has_creds, bool) and has_creds:
                return True
            saved_user = row.get("savedUser")
            if isinstance(saved_user, dict) and saved_user.get("id"):
                return True
        return False

    def _hydrate_oauth_requirements(
        self,
        template: ConnectorTemplate,
        user_inputs: dict[str, Any],
    ) -> None:
        """Store OAuth tokens or trigger flow when requested by user inputs."""
        provider = template.requires_auth
        if not provider:
            return
        if self.checker.check_oauth(provider):
            return

        token_keys = (
            f"{provider}_token",
            f"{provider}_oauth_token",
            "oauth_token",
            "access_token",
        )
        for key in token_keys:
            raw = user_inputs.get(key)
            if raw is None:
                continue
            token = str(raw).strip()
            if token and self.checker.store_oauth_token(provider, token):
                return

        trigger = user_inputs.get("start_oauth") or user_inputs.get("oauth_provider")
        if trigger in (True, "true", "1", provider):
            oauth_result = self.checker.start_oauth_flow(provider)
            if not oauth_result.success:
                logger.warning(
                    "OAuth flow failed for %s: %s",
                    provider,
                    oauth_result.error or "unknown error",
                )
