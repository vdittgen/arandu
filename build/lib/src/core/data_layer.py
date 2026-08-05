"""Unified data access layer for Arandu.

Coordinates the three embedded databases — SQLite (analytical), Kuzu (graph),
and ChromaDB (vector) — through a single facade.  Callers interact with this
class instead of instantiating engines directly, making it straightforward to
swap storage backends or add new engines in future.

Engines are initialized **lazily** on first access.  This means that CLI
commands which only need SQLite (e.g. ``query-messages``) never pay the cost
of opening ChromaDB or Kuzu, which is a significant win when Ollama is
offline (ChromaDB embedding retries can take 30+ seconds).

Typical usage::

    with DataLayer() as layer:
        layer.initialize()          # create schemas
        ok, report = layer.health_check()
        stats = layer.get_stats()

sensitivity_tier: N/A — infrastructure / orchestration layer.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.core.chromadb.engine import COLLECTION_NAMES, VectorEngine
from src.core.chromadb.indexer import Indexer
from src.core.kuzu.engine import GraphEngine
from src.core.kuzu.schema import ALL_NODE_TABLES
from src.core.kuzu.schema import create_schema as create_kuzu_schema
from src.core.sqlite.engine import DatabaseEngine
from src.core.sqlite.schemas import ALL_TABLE_NAMES, create_all_tables

logger = logging.getLogger(__name__)

DEFAULT_BASE_PATH = Path.home() / ".arandu" / "data"


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@dataclass
class HealthReport:
    """Result of a health_check() call across all three engines."""

    sqlite_ok: bool = False
    kuzu_ok: bool = False
    chromadb_ok: bool = False
    errors: list[str] = field(default_factory=list)

    # Backwards-compat alias used by existing callers
    @property
    def duckdb_ok(self) -> bool:  # noqa: N802
        return self.sqlite_ok

    @duckdb_ok.setter
    def duckdb_ok(self, value: bool) -> None:
        self.sqlite_ok = value

    @property
    def all_ok(self) -> bool:
        """True only when every engine is healthy."""
        return self.sqlite_ok and self.kuzu_ok and self.chromadb_ok


@dataclass
class LayerStats:
    """Aggregate counts from all three embedded databases."""

    # SQLite: row count per raw table
    sqlite: dict[str, int] = field(default_factory=dict)
    # Kuzu: node count per node-table name
    kuzu_nodes: dict[str, int] = field(default_factory=dict)
    # ChromaDB: document count per collection
    chromadb: dict[str, int] = field(default_factory=dict)

    # Backwards-compat aliases used by existing callers
    @property
    def duckdb(self) -> dict[str, int]:
        return self.sqlite

    @property
    def total_duckdb_rows(self) -> int:
        return sum(self.sqlite.values())

    @property
    def total_sqlite_rows(self) -> int:
        return sum(self.sqlite.values())

    @property
    def total_kuzu_nodes(self) -> int:
        return sum(self.kuzu_nodes.values())

    @property
    def total_chroma_docs(self) -> int:
        return sum(self.chromadb.values())


# ---------------------------------------------------------------------------
# DataLayer
# ---------------------------------------------------------------------------


class DataLayer:
    """Facade that owns and coordinates all three embedded database engines.

    Engines are initialized **lazily** on first access via the ``.duckdb``,
    ``.kuzu``, and ``.chromadb`` properties.  Call ``warmup()`` to eagerly
    initialize all engines when you know they will all be needed.

    Args:
        base_path: Root directory under which each engine stores its data.
                   Defaults to ~/.arandu/data/.

    sensitivity_tier: N/A
    """

    def __init__(
        self,
        base_path: Path = DEFAULT_BASE_PATH,
        read_only: bool = False,
        *,
        kuzu_read_only: bool | None = None,
    ) -> None:
        """Initialize the data layer.

        Args:
            base_path: Root directory under which each engine stores
                its data. Defaults to ``~/.arandu/data``.
            read_only: Open SQLite in read-only mode. Set True for
                query-only callers that must not contend with the
                pipeline writer.
            kuzu_read_only: Open the Kuzu graph engine read-only when
                True. ``None`` (default) inherits from ``read_only``.
                Long-running query consumers (WhatsApp listener,
                ``ask`` / ``ask-stream`` chat) should pass True so
                they can coexist; only the pipeline + ingestion
                review flow need Kuzu writes.
        """
        self._base = base_path
        self._base.mkdir(parents=True, exist_ok=True)
        self._read_only = read_only
        self._kuzu_read_only = (
            read_only if kuzu_read_only is None else kuzu_read_only
        )

        self._duck: DatabaseEngine | None = None
        self._kuzu: GraphEngine | None = None
        self._chroma: VectorEngine | None = None
        self._indexer: Indexer | None = None
        logger.info(
            "DataLayer created (base=%s), engines deferred", self._base
        )

    # ------------------------------------------------------------------
    # Lazy engine access
    # ------------------------------------------------------------------

    @property
    def duckdb(self) -> DatabaseEngine:
        """The underlying SQLite engine. Initialized on first access.

        Named ``duckdb`` for backwards compatibility with callers.

        sensitivity_tier: N/A
        """
        if self._duck is None:
            logger.info("Lazy-initializing SQLite engine…")
            self._duck = DatabaseEngine(
                db_path=self._base / "arandu.sqlite3",
                read_only=self._read_only,
            )
        return self._duck

    @property
    def kuzu(self) -> GraphEngine:
        """The underlying Kuzu graph engine. Initialized on first access.

        sensitivity_tier: N/A
        """
        if self._kuzu is None:
            logger.info("Lazy-initializing Kuzu engine…")
            self._kuzu = GraphEngine(
                db_path=self._base / "kuzu_db",
                read_only=self._kuzu_read_only,
            )
            # Kuzu's schema (Person/Event/Place/... node tables) only
            # exists if someone has explicitly run `cli init` or `cli
            # reset` against this data directory. SQLite and ChromaDB
            # self-bootstrap on first write, so after a fresh wipe Kuzu
            # is the only engine that stays empty — every query returns
            # -1 and the AmbientBar flips to "DB issue". The DDL uses
            # CREATE NODE TABLE IF NOT EXISTS, so doing this on every
            # read-write open is idempotent and cheap.
            if not self._kuzu_read_only:
                create_kuzu_schema(self._kuzu)
        return self._kuzu

    @property
    def chromadb(self) -> VectorEngine:
        """The underlying ChromaDB vector engine. Initialized on first access.

        sensitivity_tier: N/A
        """
        if self._chroma is None:
            logger.info("Lazy-initializing ChromaDB engine…")
            self._chroma = VectorEngine(db_path=self._base / "chromadb")
        return self._chroma

    @property
    def indexer(self) -> Indexer:
        """The underlying DuckDB-to-ChromaDB indexer.

        sensitivity_tier: N/A
        """
        if self._indexer is None:
            self._indexer = Indexer(duckdb=self.duckdb, chromadb=self.chromadb)
        return self._indexer

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def warmup(self) -> None:
        """Eagerly initialize all three database engines.

        Call this when you know all engines will be needed (e.g. for
        ``init``, ``status``, or ``reset`` commands).

        sensitivity_tier: N/A
        """
        logger.info("Warming up all engines…")
        _ = self.duckdb
        _ = self.kuzu
        _ = self.chromadb
        _ = self.indexer
        logger.info("All engines warmed up.")

    # ------------------------------------------------------------------
    # Core lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Create all schemas in every database.

        Safe to call on a fresh or an already-initialized database —
        all operations are idempotent.
        """
        logger.info("Initializing SQLite schemas…")
        create_all_tables(self.duckdb)

        logger.info("Initializing Kuzu schema…")
        create_kuzu_schema(self.kuzu)

        logger.info("Initialization complete.")

    def health_check(self) -> tuple[bool, HealthReport]:
        """Verify that every engine can execute a trivial read operation.

        Returns:
            Tuple of (all_ok: bool, report: HealthReport).
        """
        report = HealthReport()

        # SQLite — count tables in sqlite_master
        try:
            result = self.duckdb.query(
                "SELECT COUNT(*) AS n FROM sqlite_master"
                " WHERE type = 'table'"
            )
            assert result[0]["n"] >= 0
            report.sqlite_ok = True
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"SQLite: {exc}")

        # Kuzu — list tables (safe even on an empty schema)
        try:
            self.kuzu.query("CALL show_tables() RETURN *")
            report.kuzu_ok = True
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"Kuzu: {exc}")

        # ChromaDB — list collections
        try:
            self.chromadb._client.list_collections()
            report.chromadb_ok = True
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"ChromaDB: {exc}")

        return report.all_ok, report

    def get_stats(self) -> LayerStats:
        """Return document / row / node counts across all three databases.

        Returns:
            LayerStats dataclass with counts grouped by engine.
        """
        stats = LayerStats()

        # SQLite — row count per raw table
        for table in self._list_raw_tables():
            try:
                rows = self.duckdb.query(f"SELECT COUNT(*) AS n FROM {table}")
                stats.sqlite[table] = rows[0]["n"]
            except Exception:  # noqa: BLE001
                stats.sqlite[table] = -1

        # Kuzu — node count per node type
        for node_type in ALL_NODE_TABLES:
            try:
                rows = self.kuzu.query(
                    f"MATCH (n:{node_type}) RETURN count(n) AS n"
                )
                stats.kuzu_nodes[node_type] = rows[0]["n"]
            except Exception:  # noqa: BLE001
                stats.kuzu_nodes[node_type] = -1

        # ChromaDB — document count per collection
        for name in COLLECTION_NAMES:
            try:
                col = self.chromadb.get_or_create_collection(name)
                stats.chromadb[name] = col.count()
            except Exception:  # noqa: BLE001
                stats.chromadb[name] = -1

        return stats

    def reset(self) -> None:
        """Drop all data and reinitialize from scratch.

        Closes all engines, deletes their on-disk storage, then reopens and
        re-initializes.  Use with caution — all stored data is permanently
        deleted.

        Note: ``reset`` runs *before* any ingestion, so the post-reset
        ``raw_messages`` table is empty.  To skip historical messages
        that a connector backfills on the first sync, call
        :meth:`seed_evaluated_messages_pre_cutoff` after that sync
        completes (``cmd_startup_sync`` does this automatically when
        ``ingest_cutoff_iso`` is set in settings).
        """
        logger.warning(
            "DataLayer.reset() — deleting all data under %s",
            self._base,
        )
        self.close()
        self._delete_engine_files()
        self._reopen_engines()
        self.initialize()
        logger.info("DataLayer reset complete.")

    def seed_evaluated_messages_pre_cutoff(self, cutoff_iso: str) -> int:
        """Mark every pre-cutoff message as already-evaluated.

        Belt-and-suspenders to ``_apply_ingest_cutoff`` in the ingestion
        adapter: anything that slips past the cutoff filter (or that was
        ingested before the cutoff was set) still won't be sent to the
        LLM evaluator on the next sync.

        Returns the number of rows inserted.

        sensitivity_tier: 1
        """
        from src.agents.message_eval.persistence import _CONNECTOR_TABLES

        sqlite = self.duckdb
        # Ensure the dedup table exists even when MessageEvaluator hasn't
        # been instantiated yet in this process.
        sqlite.execute(
            """
            CREATE TABLE IF NOT EXISTS _evaluated_messages (
                message_id      VARCHAR PRIMARY KEY,
                source_table    VARCHAR NOT NULL,
                connector_id    VARCHAR NOT NULL,
                evaluated_at    TEXT DEFAULT CURRENT_TIMESTAMP,
                notification_sent INTEGER DEFAULT 0,
                notification_type VARCHAR,
                importance      INTEGER DEFAULT 0
            )
            """,
        )
        # Walk each (connector, table) pair, but record the distinct set
        # of (message_id, source_table) we've added so the returned count
        # reflects rows actually inserted (one connector wins per id).
        seen: set[tuple[str, str]] = set()
        for connector_id, tables in _CONNECTOR_TABLES.items():
            for table in tables:
                ts_col = "date" if table == "raw_emails" else "timestamp"
                try:
                    rows = sqlite.query(
                        f"SELECT CAST(id AS VARCHAR) AS id "
                        f"FROM {table} WHERE {ts_col} < ?",
                        [cutoff_iso],
                    )
                except Exception:  # noqa: BLE001
                    logger.debug(
                        "seed cutoff: skipping missing table %s",
                        table,
                        exc_info=True,
                    )
                    continue
                for row in rows:
                    key = (row["id"], table)
                    if key in seen:
                        continue
                    sqlite.execute(
                        "INSERT OR IGNORE INTO _evaluated_messages "
                        "(message_id, source_table, connector_id, "
                        "importance) VALUES (?, ?, ?, 0)",
                        [row["id"], table, connector_id],
                    )
                    seen.add(key)
        total = len(seen)
        logger.info(
            "Seeded %d pre-cutoff rows into _evaluated_messages "
            "(cutoff=%s)",
            total,
            cutoff_iso,
        )
        return total

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def reindex(self) -> dict[str, int]:
        """Reindex all DuckDB data into ChromaDB collections.

        Clears existing embeddings and rebuilds from raw tables.

        Returns:
            Dict mapping collection name to document count.

        sensitivity_tier: 3
        """
        return self.indexer.full_reindex()

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def run_pipeline_and_reindex(
        self,
        trigger: str = "manual",
    ) -> dict[str, Any]:
        """Execute the SQLMesh pipeline, then re-index ChromaDB.

        Calls ``run_pipeline`` and, on success, incrementally indexes
        any records created since the run started.

        Args:
            trigger: Label indicating the caller.

        Returns:
            JSON-serializable dict with pipeline run stats and
            ``reindex_counts`` on success.

        sensitivity_tier: 1
        """
        from datetime import datetime

        result = self.run_pipeline(trigger=trigger)
        if result["status"] == "success":
            try:
                total_docs = sum(
                    self.chromadb.get_collection_count(name)
                    for name in COLLECTION_NAMES
                )
                if total_docs == 0:
                    counts = self.indexer.full_reindex()
                else:
                    since = datetime.fromisoformat(result["started_at"])
                    counts = self.indexer.incremental_index(since=since)
                result["reindex_counts"] = counts
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Post-pipeline re-index failed: %s", exc,
                )
                result["reindex_error"] = str(exc)
        return result

    def run_pipeline(self, trigger: str = "manual") -> dict[str, Any]:
        """Execute the SQLMesh pipeline and return run stats.

        Args:
            trigger: Label indicating the caller
                     (``"manual"``, ``"scheduled"``, ``"startup"``).

        Returns:
            JSON-serializable dict representation of the PipelineRun.

        sensitivity_tier: 1
        """
        from dataclasses import asdict

        from src.pipeline.runner import PipelineRunner
        from src.pipeline.stats import ProcessingStats

        runner = PipelineRunner(duckdb=self.duckdb, stats=ProcessingStats())
        run = runner.run(trigger=trigger)
        result = asdict(run)
        result["started_at"] = run.started_at.isoformat()
        result["completed_at"] = run.completed_at.isoformat()
        return result

    def run_pipeline_stream(
        self,
        trigger: str = "manual",
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Execute the SQLMesh pipeline with streaming progress callbacks.

        Args:
            trigger: Label indicating the caller.
            on_progress: Called with each progress event dict.

        Returns:
            JSON-serializable dict representation of the PipelineRun.

        sensitivity_tier: 1
        """
        from dataclasses import asdict

        from src.pipeline.runner import PipelineRunner
        from src.pipeline.stats import ProcessingStats

        runner = PipelineRunner(duckdb=self.duckdb, stats=ProcessingStats())
        run = runner.run(trigger=trigger, on_progress=on_progress)
        result = asdict(run)
        result["started_at"] = run.started_at.isoformat()
        result["completed_at"] = run.completed_at.isoformat()
        return result

    def get_pipeline_status(self) -> dict[str, Any]:
        """Return pipeline health: last run, staleness, pending changes.

        Returns:
            Dict with keys ``last_run``, ``is_stale``,
            ``pending_changes``, ``estimated_refresh_time``.

        sensitivity_tier: 1
        """
        from dataclasses import asdict

        from src.pipeline.runner import PipelineRunner
        from src.pipeline.stats import ProcessingStats

        stats = ProcessingStats()
        runner = PipelineRunner(duckdb=self.duckdb, stats=stats)

        last_run = stats.get_last_run()
        estimate = runner.dry_run()

        last_run_dict: dict[str, Any] | None = None
        if last_run is not None:
            last_run_dict = asdict(last_run)
            last_run_dict["started_at"] = last_run.started_at.isoformat()
            last_run_dict["completed_at"] = (
                last_run.completed_at.isoformat()
            )

        return {
            "last_run": last_run_dict,
            "is_stale": runner.is_stale(),
            "pending_changes": estimate.pending_changes,
            "estimated_refresh_time": (
                estimate.estimated_duration_seconds
            ),
        }

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Cleanly shut down all initialized database engines.

        Only closes engines that have been created (handles partial init).

        sensitivity_tier: N/A
        """
        if self._duck is not None:
            self._duck.close()
        if self._kuzu is not None:
            self._kuzu.close()
        if self._chroma is not None:
            self._chroma.close()
        logger.info("DataLayer closed.")

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> DataLayer:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _delete_engine_files(self) -> None:
        """Remove all engine-specific storage paths under the base directory."""
        targets = [
            self._base / "arandu.sqlite3",
            self._base / "arandu.sqlite3-wal",
            self._base / "arandu.sqlite3-shm",
            self._base / "kuzu_db",
            self._base / "chromadb",
        ]
        for target in targets:
            if target.is_file():
                target.unlink()
                logger.debug("Deleted file: %s", target)
            elif target.is_dir():
                shutil.rmtree(target)
                logger.debug("Deleted directory: %s", target)

    def _reopen_engines(self) -> None:
        """Recreate engine instances after storage has been wiped."""
        self._duck = DatabaseEngine(
            db_path=self._base / "arandu.sqlite3",
        )
        self._kuzu = GraphEngine(
            db_path=self._base / "kuzu_db",
            read_only=self._kuzu_read_only,
        )
        self._chroma = VectorEngine(
            db_path=self._base / "chromadb",
        )
        self._indexer = Indexer(
            duckdb=self._duck,
            chromadb=self._chroma,
        )

    def _list_raw_tables(self) -> list[str]:
        """Return all existing ``raw_*`` tables in SQLite.

        Falls back to the static schema list when the sqlite_master
        query fails.

        sensitivity_tier: 1
        """
        try:
            rows = self.duckdb.query(
                "SELECT name "
                "FROM sqlite_master "
                "WHERE type = 'table' "
                "AND name LIKE 'raw_%' "
                "ORDER BY name",
            )
            tables = [str(r["name"]) for r in rows]
            if tables:
                return tables
        except Exception:  # noqa: BLE001
            logger.warning(
                "Failed to list raw tables dynamically",
                exc_info=True,
            )
        return list(ALL_TABLE_NAMES)
