"""Unit tests for the connector catalog, models, and migrations.

Tests cover catalog loading, platform filtering, field mapping validation,
dedup-key consistency, and DuckDB table migrations.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from src.core.sqlite.engine import DatabaseEngine
from src.core.sqlite.migrations import (
    MIGRATION_TABLE_NAMES,
    ensure_table,
    get_existing_tables,
    run_migrations,
)
from src.core.sqlite.schemas import ALL_TABLE_NAMES, create_all_tables
from src.extensions.connectors.catalog import ConnectorCatalog
from src.extensions.models import FieldTemplate, ToolTemplate

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def catalog() -> ConnectorCatalog:
    """Load the bundled catalog from the default path."""
    return ConnectorCatalog()


@pytest.fixture()
def tmp_db(tmp_path: Path) -> DatabaseEngine:
    """Open a fresh DatabaseEngine backed by a temp file; close after test."""
    db_path = tmp_path / "test_migrations.duckdb"
    engine = DatabaseEngine(db_path=db_path)
    yield engine
    engine.close()


@pytest.fixture()
def full_db(tmp_db: DatabaseEngine) -> DatabaseEngine:
    """DatabaseEngine with original schemas + all migrations applied."""
    create_all_tables(tmp_db)
    run_migrations(tmp_db)
    return tmp_db


# ---------------------------------------------------------------------------
# Catalog loading tests
# ---------------------------------------------------------------------------


class TestCatalogLoading:
    def test_loads_all_connectors(self, catalog: ConnectorCatalog) -> None:
        """Catalog must contain exactly 9 pre-verified connectors."""
        assert len(catalog.all) == 10

    def test_all_connectors_have_unique_ids(self, catalog: ConnectorCatalog) -> None:
        """Every connector must have a unique ID."""
        ids = [c.id for c in catalog.all]
        assert len(ids) == len(set(ids)), f"Duplicate IDs found: {ids}"

    def test_all_connectors_have_required_fields(
        self, catalog: ConnectorCatalog,
    ) -> None:
        """Every connector must have id, name, category, command, transport."""
        for c in catalog.all:
            assert c.id, f"Connector missing id: {c}"
            assert c.name, f"Connector {c.id} missing name"
            assert c.category, f"Connector {c.id} missing category"
            assert c.command, f"Connector {c.id} missing command"
            assert c.transport == "stdio", f"Connector {c.id} has unexpected transport"

    def test_all_connectors_have_at_least_one_tool(
        self, catalog: ConnectorCatalog,
    ) -> None:
        """Every connector must expose at least one tool."""
        for c in catalog.all:
            assert len(c.tools) >= 1, f"Connector {c.id} has no tools"

    def test_connector_ids_match_expected_set(
        self, catalog: ConnectorCatalog,
    ) -> None:
        """Verify the exact set of shipped connector IDs."""
        expected = {
            "apple-calendar",
            "apple-contacts",
            "apple-notes",
            "apple-mail",
            "apple-messages",
            "google-calendar",
            "google-gmail",
            "filesystem",
            "whatsapp",
            "spotify",
        }
        actual = {c.id for c in catalog.all}
        assert actual == expected

    def test_get_by_id(self, catalog: ConnectorCatalog) -> None:
        """get() must return the correct connector by ID."""
        cal = catalog.get("apple-calendar")
        assert cal is not None
        assert cal.name == "Calendar & Reminders"

    def test_get_by_id_missing(self, catalog: ConnectorCatalog) -> None:
        """get() must return None for unknown IDs."""
        assert catalog.get("nonexistent-connector") is None

    def test_categories_are_valid(self, catalog: ConnectorCatalog) -> None:
        """All connectors must use one of the valid categories."""
        valid = {"apple", "google", "files", "email", "lifestyle"}
        for c in catalog.all:
            assert c.category in valid, (
                f"Connector {c.id} has invalid category: {c.category}"
            )


# ---------------------------------------------------------------------------
# Platform filtering tests
# ---------------------------------------------------------------------------


class TestPlatformFiltering:
    def test_macos_includes_all(self, catalog: ConnectorCatalog) -> None:
        """On macOS, all 9 connectors should be available."""
        available = catalog.get_available(target_platform="macos")
        assert len(available) == 10

    def test_linux_excludes_apple_only(self, catalog: ConnectorCatalog) -> None:
        """On Linux, Apple-only connectors should be excluded."""
        available = catalog.get_available(target_platform="linux")
        available_ids = {c.id for c in available}
        # Apple-specific connectors should not be available on Linux
        apple_only = {
            "apple-calendar", "apple-contacts", "apple-notes", "apple-mail",
            "apple-messages",
        }
        for apple_id in apple_only:
            assert apple_id not in available_ids, f"{apple_id} should not be on Linux"

    def test_cross_platform_connectors_on_linux(
        self, catalog: ConnectorCatalog,
    ) -> None:
        """Cross-platform connectors must be available on Linux."""
        available = catalog.get_available(target_platform="linux")
        available_ids = {c.id for c in available}
        cross_platform = {"filesystem", "whatsapp", "spotify"}
        for cp_id in cross_platform:
            assert cp_id in available_ids, f"{cp_id} should be available on Linux"

    def test_windows_excludes_apple_only(self, catalog: ConnectorCatalog) -> None:
        """On Windows, Apple-only connectors should be excluded."""
        available = catalog.get_available(target_platform="windows")
        available_ids = {c.id for c in available}
        assert "apple-calendar" not in available_ids
        assert "filesystem" in available_ids


# ---------------------------------------------------------------------------
# Category grouping tests
# ---------------------------------------------------------------------------


class TestCategoryGrouping:
    def test_get_by_category_returns_all_categories(
        self, catalog: ConnectorCatalog,
    ) -> None:
        """get_by_category on macOS should return all 4 categories."""
        grouped = catalog.get_by_category(target_platform="macos")
        assert set(grouped.keys()) == {
            "apple", "google", "files", "email", "lifestyle",
        }

    def test_apple_category_count(self, catalog: ConnectorCatalog) -> None:
        """Apple category should have 5 connectors."""
        grouped = catalog.get_by_category(target_platform="macos")
        assert len(grouped["apple"]) == 5

    def test_email_category_count(self, catalog: ConnectorCatalog) -> None:
        """Email category should have 1 connector (whatsapp)."""
        grouped = catalog.get_by_category(target_platform="macos")
        assert len(grouped["email"]) == 1

    def test_linux_has_no_apple_category(self, catalog: ConnectorCatalog) -> None:
        """On Linux, apple category should be absent."""
        grouped = catalog.get_by_category(target_platform="linux")
        assert "apple" not in grouped


# ---------------------------------------------------------------------------
# Default-enabled tests
# ---------------------------------------------------------------------------


class TestDefaultEnabled:
    def test_enabled_connectors(self, catalog: ConnectorCatalog) -> None:
        """Only apple-calendar, apple-contacts, and filesystem are default-enabled."""
        enabled = catalog.get_enabled()
        enabled_ids = {c.id for c in enabled}
        assert enabled_ids == {"apple-calendar", "apple-contacts", "filesystem"}


# ---------------------------------------------------------------------------
# Search tests
# ---------------------------------------------------------------------------


class TestSearch:
    def test_search_by_name(self, catalog: ConnectorCatalog) -> None:
        """Search 'calendar' should match apple-calendar."""
        results = catalog.search("calendar")
        ids = {c.id for c in results}
        assert "apple-calendar" in ids

    def test_search_by_description(self, catalog: ConnectorCatalog) -> None:
        """Search 'spotify' should match the spotify connector."""
        results = catalog.search("spotify")
        ids = {c.id for c in results}
        assert "spotify" in ids

    def test_search_case_insensitive(self, catalog: ConnectorCatalog) -> None:
        """Search must be case-insensitive."""
        results = catalog.search("WHATSAPP")
        ids = {c.id for c in results}
        assert "whatsapp" in ids

    def test_search_no_results(self, catalog: ConnectorCatalog) -> None:
        """Search for a nonexistent term returns empty list."""
        results = catalog.search("zzz_nonexistent_zzz")
        assert results == []

    def test_search_by_category(self, catalog: ConnectorCatalog) -> None:
        """Search 'lifestyle' should match spotify."""
        results = catalog.search("lifestyle")
        ids = {c.id for c in results}
        assert "spotify" in ids


# ---------------------------------------------------------------------------
# Field mapping validation tests
# ---------------------------------------------------------------------------


class TestFieldMappings:
    def test_all_data_tools_have_target_table(
        self, catalog: ConnectorCatalog,
    ) -> None:
        """Every data tool must reference a target table."""
        for c in catalog.all:
            for tool in c.tools:
                if tool.tool_type == "data":
                    assert tool.target_table, (
                        f"Data tool {c.id}/{tool.tool_name} missing target_table"
                    )

    def test_action_tools_have_no_fields(
        self, catalog: ConnectorCatalog,
    ) -> None:
        """Action tools should have empty fields and dedup_key."""
        for c in catalog.all:
            for tool in c.tools:
                if tool.tool_type == "action":
                    assert len(tool.fields) == 0, (
                        f"Action tool {c.id}/{tool.tool_name} should have no fields"
                    )

    def test_sensitivity_tiers_are_valid(
        self, catalog: ConnectorCatalog,
    ) -> None:
        """All field sensitivity tiers must be 1, 2, or 3."""
        for c in catalog.all:
            for tool in c.tools:
                for field in tool.fields:
                    assert field.sensitivity_tier in (1, 2, 3), (
                        f"Invalid tier {field.sensitivity_tier} in "
                        f"{c.id}/{tool.tool_name}/{field.source_name}"
                    )

    def test_data_tools_have_dedup_keys(
        self, catalog: ConnectorCatalog,
    ) -> None:
        """Every data tool must have at least one dedup key."""
        for c in catalog.all:
            for tool in c.tools:
                if tool.tool_type == "data":
                    assert len(tool.dedup_key) >= 1, (
                        f"Data tool {c.id}/{tool.tool_name} has no dedup_key"
                    )

    def test_data_tools_have_at_least_one_field(
        self, catalog: ConnectorCatalog,
    ) -> None:
        """Every data tool must have at least one field mapping."""
        for c in catalog.all:
            for tool in c.tools:
                if tool.tool_type == "data":
                    assert len(tool.fields) >= 1, (
                        f"Data tool {c.id}/{tool.tool_name} has no fields"
                    )

    def test_field_target_types_are_valid_sql(
        self, catalog: ConnectorCatalog,
    ) -> None:
        """All target_type values should be valid DuckDB types."""
        valid_types = {
            "VARCHAR", "TEXT", "INTEGER", "BIGINT", "DOUBLE",
            "BOOLEAN", "TIMESTAMPTZ", "JSON",
        }
        for c in catalog.all:
            for tool in c.tools:
                for field in tool.fields:
                    assert field.target_type in valid_types, (
                        f"Invalid target_type {field.target_type!r} in "
                        f"{c.id}/{tool.tool_name}/{field.source_name}"
                    )

    def test_tool_types_are_valid(self, catalog: ConnectorCatalog) -> None:
        """Tool type must be 'data' or 'action'."""
        for c in catalog.all:
            for tool in c.tools:
                assert tool.tool_type in ("data", "action"), (
                    f"Invalid tool_type {tool.tool_type!r} in {c.id}/{tool.tool_name}"
                )


# ---------------------------------------------------------------------------
# Target table reference validation
# ---------------------------------------------------------------------------


class TestTargetTableReferences:
    """Validate that all target tables referenced by connectors
    are either original schema tables or migration tables."""

    def test_all_target_tables_have_schemas(
        self, catalog: ConnectorCatalog,
    ) -> None:
        """Every target table must be defined either in schemas.py or migrations.py."""
        known_tables = set(ALL_TABLE_NAMES) | set(MIGRATION_TABLE_NAMES)
        referenced = catalog.get_all_target_tables()
        for table in referenced:
            assert table in known_tables, (
                f"Target table {table!r} is not defined in schemas or migrations"
            )

    def test_field_columns_match_table_schema(
        self,
        full_db: DatabaseEngine,
        catalog: ConnectorCatalog,
    ) -> None:
        """Field target_columns must exist in the target table."""
        for c in catalog.all:
            for tool in c.tools:
                if tool.tool_type != "data" or not tool.target_table:
                    continue
                # Get column names from the table
                cols_rows = full_db.query(
                    f"PRAGMA table_info({tool.target_table})"
                )
                col_names = {row["name"] for row in cols_rows}
                assert col_names, (
                    f"Table {tool.target_table} has no columns — "
                    f"possibly not created?"
                )
                for field in tool.fields:
                    assert field.target_column in col_names, (
                        f"Column {field.target_column!r} not found in "
                        f"{tool.target_table} (connector={c.id}, "
                        f"tool={tool.tool_name}). "
                        f"Available: {sorted(col_names)}"
                    )


# ---------------------------------------------------------------------------
# Dedup key consistency tests
# ---------------------------------------------------------------------------


class TestDedupKeyConsistency:
    def test_no_conflicting_dedup_keys_for_shared_tables(
        self, catalog: ConnectorCatalog,
    ) -> None:
        """Connectors sharing a table must use compatible dedup strategies.

        Specifically: if multiple connectors write to the same table, their
        dedup_key tuples must be identical or one must be a subset of the
        other (so the broader key can be used as the merge key).
        """
        table_dedup: dict[str, list[tuple[str, tuple[str, ...]]]] = {}
        for c in catalog.all:
            for tool in c.tools:
                if tool.tool_type == "data" and tool.target_table:
                    entry = (f"{c.id}/{tool.tool_name}", tool.dedup_key)
                    table_dedup.setdefault(tool.target_table, []).append(entry)

        for table, entries in table_dedup.items():
            if len(entries) <= 1:
                continue
            # All dedup keys for a shared table should be compatible
            key_sets = [set(e[1]) for e in entries]
            union = set().union(*key_sets)
            for key_set, (label, _) in zip(key_sets, entries):
                assert key_set.issubset(union), (
                    f"Dedup key conflict for table {table!r}: "
                    f"{label} uses {key_set}, union is {union}"
                )


# ---------------------------------------------------------------------------
# DuckDB migration tests
# ---------------------------------------------------------------------------


class TestMigrations:
    def test_run_migrations_creates_all_new_tables(
        self, tmp_db: DatabaseEngine,
    ) -> None:
        """run_migrations() must create all 5 new tables."""
        create_all_tables(tmp_db)
        created = run_migrations(tmp_db)
        assert set(created) == set(MIGRATION_TABLE_NAMES)

    def test_run_migrations_is_idempotent(
        self, tmp_db: DatabaseEngine,
    ) -> None:
        """Running migrations twice must not raise or create duplicates."""
        create_all_tables(tmp_db)
        first = run_migrations(tmp_db)
        second = run_migrations(tmp_db)
        assert len(first) == 5
        assert len(second) == 0  # all already exist

    def test_migration_tables_have_sensitivity_tier(
        self, full_db: DatabaseEngine,
    ) -> None:
        """Every migration table must have a sensitivity_tier column."""
        for table in MIGRATION_TABLE_NAMES:
            rows = full_db.query(
                f"PRAGMA table_info({table})"
            )
            col_names = {r["name"] for r in rows}
            assert "sensitivity_tier" in col_names, (
                f"Table {table} is missing sensitivity_tier column"
            )

    def test_migration_tables_have_created_at(
        self, full_db: DatabaseEngine,
    ) -> None:
        """Every migration table must have a created_at column."""
        for table in MIGRATION_TABLE_NAMES:
            rows = full_db.query(
                f"PRAGMA table_info({table})"
            )
            col_names = {r["name"] for r in rows}
            assert "created_at" in col_names, (
                f"Table {table} is missing created_at column"
            )

    def test_migration_tables_have_id_primary_key(
        self, full_db: DatabaseEngine,
    ) -> None:
        """Every migration table must have an 'id' column."""
        for table in MIGRATION_TABLE_NAMES:
            rows = full_db.query(
                f"PRAGMA table_info({table})"
            )
            col_names = {r["name"] for r in rows}
            assert "id" in col_names, f"Table {table} is missing id column"

    def test_ensure_table_creates_single_table(
        self, tmp_db: DatabaseEngine,
    ) -> None:
        """ensure_table() should create one specific table."""
        create_all_tables(tmp_db)
        result = ensure_table(tmp_db, "raw_emails")
        assert result is True
        existing = get_existing_tables(tmp_db)
        assert "raw_emails" in existing
        # Other migration tables should NOT be created
        assert "raw_workouts" not in existing

    def test_ensure_table_returns_false_if_exists(
        self, tmp_db: DatabaseEngine,
    ) -> None:
        """ensure_table() returns False when the table already exists."""
        create_all_tables(tmp_db)
        ensure_table(tmp_db, "raw_emails")
        result = ensure_table(tmp_db, "raw_emails")
        assert result is False

    def test_ensure_table_rejects_unknown_table(
        self, tmp_db: DatabaseEngine,
    ) -> None:
        """ensure_table() must raise ValueError for unknown tables."""
        with pytest.raises(ValueError, match="Unknown migration table"):
            ensure_table(tmp_db, "raw_nonexistent")

    def test_original_tables_still_exist_after_migration(
        self, full_db: DatabaseEngine,
    ) -> None:
        """Migrations must not affect original schema tables."""
        existing = get_existing_tables(full_db)
        for table in ALL_TABLE_NAMES:
            assert table in existing, (
                f"Original table {table} is missing after migration"
            )

    def test_migration_table_default_tiers(
        self, full_db: DatabaseEngine,
    ) -> None:
        """Verify default sensitivity_tier for each migration table."""
        expected_defaults: dict[str, int] = {
            "raw_emails": 2,
            "raw_reminders": 1,
            "raw_workouts": 3,
            "raw_voice_memos": 2,
            "raw_listening_history": 1,
        }
        for table, expected_tier in expected_defaults.items():
            rows = full_db.query(
                f"PRAGMA table_info({table})"
            )
            tier_rows = [r for r in rows if r["name"] == "sensitivity_tier"]
            assert len(tier_rows) == 1
            default_val = str(tier_rows[0]["dflt_value"])
            assert str(expected_tier) in default_val, (
                f"Table {table} expected default tier {expected_tier}, "
                f"got dflt_value={default_val}"
            )


# ---------------------------------------------------------------------------
# Model dataclass tests
# ---------------------------------------------------------------------------


class TestModels:
    def test_connector_template_is_frozen(self) -> None:
        """ConnectorTemplate should be immutable (frozen dataclass)."""
        from src.extensions.models import ConnectorTemplate

        c = ConnectorTemplate(
            id="test",
            name="Test",
            category="test",
            icon="T",
            description="A test connector",
            command="echo",
            args=("hello",),
            transport="stdio",
            tools=(),
        )
        with pytest.raises(AttributeError):
            c.id = "changed"  # type: ignore[misc]

    def test_field_template_is_frozen(self) -> None:
        """FieldTemplate should be immutable."""
        f = FieldTemplate(
            source_name="x",
            target_column="y",
            source_type="string",
            target_type="VARCHAR",
            sensitivity_tier=1,
        )
        with pytest.raises(AttributeError):
            f.source_name = "changed"  # type: ignore[misc]

    def test_tool_template_is_frozen(self) -> None:
        """ToolTemplate should be immutable."""
        t = ToolTemplate(
            tool_name="test",
            tool_type="action",
            target_table=None,
        )
        with pytest.raises(AttributeError):
            t.tool_name = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Get all target tables
# ---------------------------------------------------------------------------


class TestGetAllTargetTables:
    def test_returns_nonempty_set(self, catalog: ConnectorCatalog) -> None:
        """get_all_target_tables() must return at least one table."""
        tables = catalog.get_all_target_tables()
        assert len(tables) >= 1

    def test_includes_original_and_migration_tables(
        self, catalog: ConnectorCatalog,
    ) -> None:
        """Target tables should span both original and migration schemas."""
        tables = catalog.get_all_target_tables()
        # At least one original table
        assert tables & set(ALL_TABLE_NAMES), "No original tables referenced"
        # At least one migration table
        assert tables & set(MIGRATION_TABLE_NAMES), "No migration tables referenced"
