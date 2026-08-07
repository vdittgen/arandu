"""Unit tests for the ConnectionManager.

Tests the enable/disable flow, requirement checking integration,
table creation, registry management, and connector catalog output.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from src.core.sqlite.engine import DatabaseEngine
from src.core.sqlite.schemas import create_all_tables
from src.extensions.connectors.catalog import ConnectorCatalog
from src.extensions.connectors.connection_manager import (
    ConnectionManager,
    DisableResult,
    EnableResult,
)
from src.extensions.connectors.registry import ExtensionRegistry
from src.extensions.connectors.requirements import RequirementChecker
from src.extensions.connectors.sync_scheduler import SyncScheduler, SyncStats
from src.extensions.mcp.client import McpToolInfo

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_db(tmp_path: Path) -> DatabaseEngine:
    """Fresh DuckDB with original schemas applied."""
    db_path = tmp_path / "test_conn_mgr.duckdb"
    engine = DatabaseEngine(db_path=db_path)
    create_all_tables(engine)
    yield engine
    engine.close()


@pytest.fixture()
def registry(tmp_path: Path) -> ExtensionRegistry:
    """Extension registry backed by a temp file."""
    return ExtensionRegistry(
        registry_path=tmp_path / "extensions.json",
    )


@pytest.fixture()
def mcp_client_state() -> dict[str, object]:
    """Controllable fake MCP client behavior for lifecycle tests."""
    state: dict[str, object] = {
        "tools": [
            McpToolInfo(
                name="list_files",
                description="List files",
                input_schema={},
            ),
        ],
        "error": None,
        "calls": 0,
    }

    class FakeMcpClient:
        def __init__(
            self,
            tools: list[McpToolInfo],
            error: Exception | None = None,
        ) -> None:
            self._tools = tools
            self._error = error

        def __enter__(self) -> FakeMcpClient:
            if self._error is not None:
                raise self._error
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
            return None

        def list_tools(self) -> list[McpToolInfo]:
            return self._tools

    def factory(
        command: str,
        args: tuple[str, ...],
        timeout: float,
    ) -> FakeMcpClient:
        state["calls"] = int(state["calls"]) + 1
        # Keep assertions available for tests that care about wiring.
        state["last_command"] = command
        state["last_args"] = args
        state["last_timeout"] = timeout
        return FakeMcpClient(
            tools=list(state["tools"]),  # type: ignore[arg-type]
            error=state["error"],  # type: ignore[arg-type]
        )

    state["factory"] = factory
    return state


@pytest.fixture()
def manager(
    tmp_db: DatabaseEngine,
    registry: ExtensionRegistry,
    mcp_client_state: dict[str, object],
) -> ConnectionManager:
    """ConnectionManager with all real components except checker."""
    return ConnectionManager(
        catalog=ConnectorCatalog(),
        registry=registry,
        scheduler=SyncScheduler(),
        checker=RequirementChecker(),
        db_engine=tmp_db,
        mcp_client_factory=mcp_client_state["factory"],  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Enable flow — no requirements
# ---------------------------------------------------------------------------


class TestEnableNoRequirements:
    def test_enable_filesystem_connector(
        self, manager: ConnectionManager,
    ) -> None:
        """filesystem connector has no auth/app/permission reqs."""
        result = manager.enable_connector("filesystem")
        assert isinstance(result, EnableResult)
        assert result.status == "connected"
        assert result.connector_id == "filesystem"
        assert result.tools_available >= 1

    def test_enable_registers_in_registry(
        self,
        manager: ConnectionManager,
        registry: ExtensionRegistry,
    ) -> None:
        """Enabling should register the connector."""
        manager.enable_connector("filesystem")
        assert registry.is_enabled("filesystem") is True

    def test_enable_schedules_sync(
        self, manager: ConnectionManager,
    ) -> None:
        """Enabling should create a sync schedule."""
        manager.enable_connector("filesystem")
        info = manager.scheduler.get_schedule_info("filesystem")
        assert info is not None
        assert info["enabled"] is True

    def test_enable_creates_tables(
        self, manager: ConnectionManager, tmp_db: DatabaseEngine,
    ) -> None:
        """Enabling apple-mail should create raw_emails table."""
        # Mock all requirements as met
        with patch.object(
            RequirementChecker, "check_all",
        ) as mock_check:
            from src.extensions.connectors.requirements import (
                RequirementsStatus,
            )

            mock_check.return_value = RequirementsStatus(
                all_met=True,
            )
            manager.enable_connector("apple-mail")

        from src.core.sqlite.migrations import (
            get_existing_tables,
        )

        tables = get_existing_tables(tmp_db)
        assert "raw_emails" in tables

    def test_enable_whatsapp_bootstraps_base_tables_on_fresh_db(
        self,
        tmp_path: Path,
        registry: ExtensionRegistry,
        mcp_client_state: dict[str, object],
    ) -> None:
        """Enable path should create base schemas even without prior init."""
        db_path = tmp_path / "fresh_enable_path.duckdb"
        engine = DatabaseEngine(db_path=db_path)
        manager = ConnectionManager(
            catalog=ConnectorCatalog(),
            registry=registry,
            scheduler=SyncScheduler(),
            checker=RequirementChecker(),
            db_engine=engine,
            mcp_client_factory=mcp_client_state["factory"],  # type: ignore[arg-type]
        )

        now = datetime.now(timezone.utc)
        first_sync = SyncStats(
            connector_id="whatsapp",
            started_at=now,
            completed_at=now,
            status="success",
            rows_synced=0,
        )

        with patch.object(
            ConnectionManager,
            "_ensure_connector_runtime_services",
            return_value=None,
        ):
            with patch.object(manager, "sync_now", return_value=first_sync):
                result = manager.enable_connector("whatsapp")

        from src.core.sqlite.migrations import get_existing_tables

        tables = get_existing_tables(engine)
        assert result.status == "connected"
        assert "raw_messages" in tables
        engine.close()


# ---------------------------------------------------------------------------
# Enable flow — lifecycle checks (MCP handshake + first sync)
# ---------------------------------------------------------------------------


class TestEnableLifecycle:
    def test_enable_errors_when_mcp_tools_empty(
        self,
        manager: ConnectionManager,
        mcp_client_state: dict[str, object],
        registry: ExtensionRegistry,
    ) -> None:
        """A running server with zero tools should fail fast."""
        mcp_client_state["tools"] = []
        with patch.object(
            RequirementChecker, "check_oauth", return_value=True,
        ):
            result = manager.enable_connector("spotify")
        assert result.status == "error"
        assert "not responding" in (result.error or "").lower()
        reg = registry.get("spotify")
        assert reg is not None
        assert reg.status == "error"

    def test_enable_errors_when_handshake_fails(
        self,
        manager: ConnectionManager,
        mcp_client_state: dict[str, object],
    ) -> None:
        """Handshake failures should surface an error result."""
        mcp_client_state["error"] = RuntimeError("boom")
        with patch.object(
            RequirementChecker, "check_oauth", return_value=True,
        ):
            result = manager.enable_connector("spotify")
        assert result.status == "error"
        assert "handshake failed" in (result.error or "").lower()

    def test_enable_runs_first_sync_and_returns_rows(
        self,
        manager: ConnectionManager,
    ) -> None:
        """First sync rows should be included in the enable result."""
        now = datetime.now(timezone.utc)
        first_sync = SyncStats(
            connector_id="filesystem",
            started_at=now,
            completed_at=now,
            status="success",
            rows_synced=42,
        )
        with patch.object(manager.scheduler, "run_now", return_value=first_sync):
            result = manager.enable_connector("filesystem")
        assert result.status == "connected"
        assert result.records_synced == 42

    def test_enable_handshake_only_once(
        self,
        manager: ConnectionManager,
        mcp_client_state: dict[str, object],
    ) -> None:
        """Enable path should perform exactly one handshake/list_tools pass."""
        with patch.object(
            RequirementChecker, "check_oauth", return_value=True,
        ):
            result = manager.enable_connector("spotify")
        assert result.status == "connected"
        assert mcp_client_state["calls"] == 1

    def test_enable_persists_discovered_tools_count(
        self,
        manager: ConnectionManager,
        registry: ExtensionRegistry,
        mcp_client_state: dict[str, object],
    ) -> None:
        """Registry should store runtime-discovered tools_count."""
        mcp_client_state["tools"] = [
            McpToolInfo(name="a", description="A", input_schema={}),
            McpToolInfo(name="b", description="B", input_schema={}),
            McpToolInfo(name="c", description="C", input_schema={}),
        ]

        result = manager.enable_connector("filesystem")

        assert result.status == "connected"
        reg = registry.get("filesystem")
        assert reg is not None
        assert reg.tools_count == 3

    def test_enable_whatsapp_starts_runtime_service(
        self,
        manager: ConnectionManager,
    ) -> None:
        """WhatsApp enable should start its persistent runtime service."""
        now = datetime.now(timezone.utc)
        first_sync = SyncStats(
            connector_id="whatsapp",
            started_at=now,
            completed_at=now,
            status="success",
            rows_synced=0,
        )
        with patch.object(
            ConnectionManager,
            "_ensure_connector_runtime_services",
            return_value=None,
        ) as mock_runtime:
            with patch.object(manager, "sync_now", return_value=first_sync):
                result = manager.enable_connector("whatsapp")

        assert result.status == "connected"
        assert mock_runtime.call_count == 1

    def test_enable_whatsapp_runtime_failure_returns_error(
        self,
        manager: ConnectionManager,
    ) -> None:
        """Runtime bootstrap failures should bubble as enable errors."""
        with patch.object(
            ConnectionManager,
            "_ensure_connector_runtime_services",
            side_effect=RuntimeError("listener boom"),
        ):
            result = manager.enable_connector("whatsapp")

        assert result.status == "error"
        assert "Runtime service failed" in (result.error or "")

    def test_sync_now_whatsapp_ensures_runtime_service(
        self,
        manager: ConnectionManager,
    ) -> None:
        """sync_now should ensure WhatsApp runtime is up before syncing."""
        now = datetime.now(timezone.utc)
        stats = SyncStats(
            connector_id="whatsapp",
            started_at=now,
            completed_at=now,
            status="success",
            rows_synced=0,
        )
        with patch.object(
            ConnectionManager,
            "_ensure_connector_runtime_services",
            return_value=None,
        ) as mock_runtime:
            with patch.object(manager.scheduler, "run_now", return_value=stats):
                got = manager.sync_now("whatsapp")

        assert got.status == "success"
        assert mock_runtime.call_count == 1


# ---------------------------------------------------------------------------
# Enable flow — needs requirements
# ---------------------------------------------------------------------------


class TestEnableNeedsRequirements:
    def test_enable_blocks_when_full_disk_access_missing(
        self, manager: ConnectionManager,
    ) -> None:
        """Full Disk Access has no runtime dialog — block as needs_setup.

        apple-* connectors read protected SQLite databases directly under
        ~/Library, so without FDA every sync fails with permission
        denied. Letting enable succeed would register the connector as
        "connected" while it silently produces zero rows.
        """
        with patch.object(
            RequirementChecker,
            "check_macos_permission",
            return_value=False,
        ):
            result = manager.enable_connector("apple-calendar")
            assert result.status == "needs_setup"
            assert any(
                m["key"] == "Full Disk Access" for m in result.missing
            )

    def test_enable_proceeds_when_only_runtime_promptable_perm_missing(
        self, manager: ConnectionManager,
    ) -> None:
        """Runtime-promptable permissions (e.g. macOS Calendar) still
        proceed to MCP handshake so the first API call can trigger the
        Automation dialog. No production connector exercises this today,
        but the bypass is preserved for future TCC-based connectors."""
        from src.extensions.connectors.requirements import (
            MissingRequirement,
            RequirementsStatus,
        )

        tcc_status = RequirementsStatus(
            all_met=False,
            missing=[
                MissingRequirement(
                    requirement_type="permission",
                    key="macOS Calendar",
                    label="Grant macOS Calendar access",
                    action="grant_permission",
                ),
            ],
        )
        with patch.object(
            RequirementChecker, "check_all", return_value=tcc_status,
        ):
            result = manager.enable_connector("apple-calendar")
            assert result.status != "needs_setup"

    def test_enable_whatsapp_needs_qr_setup_when_unpaired(
        self, manager: ConnectionManager,
    ) -> None:
        """Unpaired WhatsApp should surface explicit QR setup requirement."""
        with patch.object(
            ConnectionManager,
            "_resolve_extra_setup_missing",
            return_value=[
                {
                    "type": "setup",
                    "key": "whatsapp_qr",
                    "label": "Pair first",
                    "action": "scan_qr",
                }
            ],
        ):
            result = manager.enable_connector("whatsapp")
            assert result.status == "needs_setup"
            types = {m["type"] for m in result.missing}
            assert "setup" in types

    def test_enable_needs_oauth_spotify(
        self, manager: ConnectionManager,
    ) -> None:
        """spotify needs spotify_oauth — should return needs_setup."""
        with patch.object(
            RequirementChecker,
            "check_oauth",
            return_value=False,
        ):
            result = manager.enable_connector("spotify")
            assert result.status == "needs_setup"
            assert len(result.missing) >= 1
            assert result.missing[0]["type"] == "oauth"


# ---------------------------------------------------------------------------
# Enable flow — error cases
# ---------------------------------------------------------------------------


class TestEnableErrors:
    def test_enable_unknown_connector(
        self, manager: ConnectionManager,
    ) -> None:
        """Unknown connector ID should return error."""
        result = manager.enable_connector("nonexistent")
        assert result.status == "error"
        assert "Unknown connector" in (result.error or "")


# ---------------------------------------------------------------------------
# Disable flow
# ---------------------------------------------------------------------------


class TestDisable:
    def test_disable_preserves_data(
        self, manager: ConnectionManager,
    ) -> None:
        """Disabling should not delete data."""
        manager.enable_connector("filesystem")
        result = manager.disable_connector("filesystem")
        assert isinstance(result, DisableResult)
        assert result.status == "disabled"
        assert result.data_preserved is True

    def test_disable_unschedules(
        self, manager: ConnectionManager,
    ) -> None:
        """Disabling should remove the sync schedule."""
        manager.enable_connector("filesystem")
        manager.disable_connector("filesystem")
        info = manager.scheduler.get_schedule_info("filesystem")
        assert info is None

    def test_disable_marks_registry(
        self,
        manager: ConnectionManager,
        registry: ExtensionRegistry,
    ) -> None:
        """Disabling should mark connector as disabled in registry."""
        manager.enable_connector("filesystem")
        manager.disable_connector("filesystem")
        assert registry.is_enabled("filesystem") is False

    def test_disable_unknown_connector(
        self, manager: ConnectionManager,
    ) -> None:
        """Disabling unknown connector should return error."""
        result = manager.disable_connector("nonexistent")
        assert result.status == "error"

    def test_disable_whatsapp_stops_runtime_services(
        self,
        manager: ConnectionManager,
    ) -> None:
        """WhatsApp disable should stop the listener runtime service."""
        now = datetime.now(timezone.utc)
        first_sync = SyncStats(
            connector_id="whatsapp",
            started_at=now,
            completed_at=now,
            status="success",
            rows_synced=0,
        )
        with patch.object(
            ConnectionManager,
            "_ensure_connector_runtime_services",
            return_value=None,
        ):
            with patch.object(manager, "sync_now", return_value=first_sync):
                manager.enable_connector("whatsapp")

        with patch.object(
            ConnectionManager,
            "_stop_connector_runtime_services",
            return_value=None,
        ) as mock_stop:
            result = manager.disable_connector("whatsapp")

        assert result.status == "disabled"
        assert mock_stop.call_count == 1


# ---------------------------------------------------------------------------
# Reconnect
# ---------------------------------------------------------------------------


class TestReconnect:
    def test_reconnect_reenables(
        self,
        manager: ConnectionManager,
        registry: ExtensionRegistry,
    ) -> None:
        """Reconnect should re-enable a disabled connector."""
        manager.enable_connector("filesystem")
        manager.disable_connector("filesystem")
        assert registry.is_enabled("filesystem") is False

        result = manager.reconnect("filesystem")
        assert result.status == "connected"
        assert registry.is_enabled("filesystem") is True


# ---------------------------------------------------------------------------
# Toggle shorthand
# ---------------------------------------------------------------------------


class TestToggle:
    def test_toggle_on(
        self, manager: ConnectionManager,
    ) -> None:
        """toggle_connector(enabled=True) should enable."""
        result = manager.toggle_connector(
            "filesystem", enabled=True,
        )
        assert isinstance(result, EnableResult)
        assert result.status == "connected"

    def test_toggle_off(
        self, manager: ConnectionManager,
    ) -> None:
        """toggle_connector(enabled=False) should disable."""
        manager.enable_connector("filesystem")
        result = manager.toggle_connector(
            "filesystem", enabled=False,
        )
        assert isinstance(result, DisableResult)
        assert result.status == "disabled"


# ---------------------------------------------------------------------------
# Connector catalog output
# ---------------------------------------------------------------------------


class TestConnectorCatalog:
    def test_catalog_returns_all_available(
        self, manager: ConnectionManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """get_connector_catalog should list all available connectors.

        Forces the platform to macOS so the assertion is portable across
        CI environments — the catalog filters by host platform and Linux
        only sees the 3 cross-platform connectors.
        """
        monkeypatch.setattr(
            "src.extensions.connectors.catalog._current_platform",
            lambda: "macos",
        )
        entries = manager.get_connector_catalog()
        assert len(entries) >= 8  # at least cross-platform ones
        # All should have required fields
        for entry in entries:
            assert "connector_id" in entry
            assert "name" in entry
            assert "enabled" in entry
            assert "status" in entry

    def test_catalog_reflects_enabled_state(
        self, manager: ConnectionManager,
    ) -> None:
        """Enabled connectors should show 'connected' status."""
        manager.enable_connector("filesystem")
        entries = manager.get_connector_catalog()
        fs_entry = next(
            e for e in entries
            if e["connector_id"] == "filesystem"
        )
        assert fs_entry["enabled"] is True
        assert fs_entry["status"] == "connected"

    def test_catalog_clears_stale_needs_setup_without_missing_requirements(
        self,
        manager: ConnectionManager,
        registry: ExtensionRegistry,
    ) -> None:
        """Needs-setup entries with no missing reqs should be retryable."""
        from src.extensions.connectors.requirements import RequirementsStatus

        registry.register_needs_setup("whatsapp")
        with patch.object(
            RequirementChecker,
            "check_all",
            return_value=RequirementsStatus(all_met=True),
        ):
            entries = manager.get_connector_catalog()

        wa_entry = next(
            e for e in entries
            if e["connector_id"] == "whatsapp"
        )
        assert wa_entry["enabled"] is False
        assert wa_entry["status"] == "disabled"

    def test_catalog_preserves_registered_needs_setup_missing_requirements(
        self,
        manager: ConnectionManager,
        registry: ExtensionRegistry,
    ) -> None:
        """Explicit needs_setup blockers should remain visible in catalog."""
        registry.register_needs_setup(
            "whatsapp",
            missing=[
                {
                    "type": "setup",
                    "key": "whatsapp_qr",
                    "label": "Pair first",
                    "action": "scan_qr",
                }
            ],
        )

        entries = manager.get_connector_catalog()
        wa_entry = next(
            e for e in entries
            if e["connector_id"] == "whatsapp"
        )
        assert wa_entry["enabled"] is True
        assert wa_entry["status"] == "needs_setup"
        assert wa_entry["missing_requirements"]
        assert wa_entry["missing_requirements"][0]["key"] == "whatsapp_qr"


# ---------------------------------------------------------------------------
# Connector details
# ---------------------------------------------------------------------------


class TestConnectorDetails:
    def test_details_returns_full_info(
        self, manager: ConnectionManager,
    ) -> None:
        """get_connector_details should return comprehensive info."""
        details = manager.get_connector_details("filesystem")
        assert details is not None
        assert details["connector_id"] == "filesystem"
        assert "tools" in details
        assert len(details["tools"]) >= 1
        assert "command" in details
        assert "stats" in details

    def test_details_includes_field_mappings(
        self, manager: ConnectionManager,
    ) -> None:
        """Details should include field mappings for data tools."""
        details = manager.get_connector_details(
            "apple-calendar",
        )
        assert details is not None
        data_tools = [
            t for t in details["tools"]
            if t["tool_type"] == "data"
        ]
        assert len(data_tools) >= 1
        assert "fields" in data_tools[0]

    def test_details_unknown_connector(
        self, manager: ConnectionManager,
    ) -> None:
        """Unknown connector should return None."""
        details = manager.get_connector_details("nonexistent")
        assert details is None


# ---------------------------------------------------------------------------
# SyncEngine wiring
# ---------------------------------------------------------------------------


class TestSyncEngineWiring:
    def test_default_creates_sync_engine_with_db(
        self,
        tmp_db: DatabaseEngine,
        registry: ExtensionRegistry,
    ) -> None:
        """When db_engine is provided, SyncEngine should be created."""
        mgr = ConnectionManager(
            catalog=ConnectorCatalog(),
            registry=registry,
            checker=RequirementChecker(),
            db_engine=tmp_db,
        )
        assert mgr._sync_engine is not None
        assert mgr.scheduler._sync_fn is not None

    def test_explicit_scheduler_preserved(
        self,
        tmp_db: DatabaseEngine,
        registry: ExtensionRegistry,
    ) -> None:
        """Passing an explicit scheduler should use it as-is."""
        explicit = SyncScheduler()
        mgr = ConnectionManager(
            catalog=ConnectorCatalog(),
            registry=registry,
            scheduler=explicit,
            db_engine=tmp_db,
        )
        assert mgr.scheduler is explicit

    def test_no_db_engine_no_sync_engine(
        self,
        registry: ExtensionRegistry,
    ) -> None:
        """Without db_engine, _sync_engine should be None."""
        mgr = ConnectionManager(
            catalog=ConnectorCatalog(),
            registry=registry,
            db_engine=None,
        )
        assert mgr._sync_engine is None


class TestOAuthTriggerContract:
    """The keys the UI sends must be the keys the manager acts on.

    Regression: the Extensions page rendered an ``oauth`` requirement's
    label as plain text with no control, and its "Retry Connection"
    button forwarded only the ``env`` inputs. So nothing ever set
    ``start_oauth``, the PKCE flow was never launched, and the Google
    connectors sat on "Setup Required" / "Error" indefinitely — the
    retry just re-ran the check and found the same missing token.

    These pin the contract from the UI's side: given exactly what the
    button now sends, the flow must start.
    """

    @staticmethod
    def _hydrate(manager: ConnectionManager, user_inputs: dict) -> bool:
        """Run the oauth hydration and report whether the flow started."""
        template = next(
            (t for t in manager.catalog.all
             if getattr(t, "requires_auth", None)),
            None,
        )
        if template is None:  # pragma: no cover - catalog has no oauth
            pytest.skip("no oauth connector in the catalog")

        with (
            patch.object(
                RequirementChecker, "check_oauth", return_value=False,
            ),
            patch.object(
                RequirementChecker, "start_oauth_flow",
            ) as mock_flow,
        ):
            mock_flow.return_value = type(
                "R", (), {"success": True, "error": None},
            )()
            manager._hydrate_oauth_requirements(template, user_inputs)
            return mock_flow.called

    def test_button_payload_starts_the_flow(
        self, manager: ConnectionManager,
    ) -> None:
        """Exactly what the Sign-in button sends must launch PKCE."""
        assert self._hydrate(
            manager, {"start_oauth": "true", "oauth_provider": "google_oauth"},
        ) is True

    def test_env_only_payload_does_not_start_the_flow(
        self, manager: ConnectionManager,
    ) -> None:
        """The old Retry payload — this is what left it stuck."""
        assert self._hydrate(manager, {"some_env_key": "value"}) is False
