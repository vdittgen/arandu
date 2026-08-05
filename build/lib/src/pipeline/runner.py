"""Pipeline runner — executes SQL/Python transforms with stats collection.

Provides :class:`PipelineRunner` which executes the pipeline via the
manifest-driven executor, captures per-model timing and row counts,
records results via :class:`ProcessingStats`, and offers staleness
detection and dry-run estimation.

sensitivity_tier: 1 (infrastructure metrics only)
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.sqlite.engine import DatabaseEngine
from src.core.sqlite.schemas import ALL_TABLE_NAMES
from src.pipeline.executor import execute_pipeline
from src.pipeline.manifest import (
    load_manifest,
    resolve_execution_order,
)
from src.pipeline.stats import PipelineEstimate, PipelineRun, ProcessingStats

ProgressCallback = Callable[[dict[str, Any]], None]

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class PipelineRunner:
    """Executes the pipeline with timing, row counting, and stats.

    Args:
        duckdb: The SQLite engine for querying row counts.
        stats: Stats store (defaults to the standard file path).
        project_root: Root directory containing ``pipeline_manifest.json``.

    sensitivity_tier: 1
    """

    def __init__(
        self,
        duckdb: DatabaseEngine,
        stats: ProcessingStats | None = None,
        project_root: Path = PROJECT_ROOT,
    ) -> None:
        self._db = duckdb
        self._stats = stats or ProcessingStats()
        self._project_root = project_root
        self._manifest = load_manifest(
            project_root / "pipeline_manifest.json",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        trigger: str = "manual",
        on_progress: ProgressCallback | None = None,
        cancel_check: Callable[[], bool] | None = None,
        select_models: list[str] | None = None,
    ) -> PipelineRun:
        """Execute the pipeline and record stats.

        Args:
            trigger: Label indicating what initiated the run.
            on_progress: Optional callback invoked with progress events.
            cancel_check: Optional callable returning True when the run
                should be cancelled.
            select_models: Optional list of model names to run.

        Returns:
            PipelineRun with full execution metadata.

        sensitivity_tier: 1
        """
        models = resolve_execution_order(
            self._manifest, select_models,
        )
        run_id = str(uuid.uuid4())
        started_at = datetime.now(tz=timezone.utc)
        start_time = time.monotonic()
        total_steps = len(models)

        status = "success"
        error: str | None = None
        rows_processed: dict[str, int] = {}
        rows_changed: dict[str, int] = {}
        models_processed: list[str] = []

        self._emit(on_progress, {
            "type": "started",
            "step_index": 0,
            "total_steps": total_steps,
            "status": "starting",
            "elapsed_seconds": 0.0,
        })

        try:
            if cancel_check is not None and cancel_check():
                status = "cancelled"
            else:
                self._emit(on_progress, {
                    "type": "sqlmesh_running",
                    "step_index": 0,
                    "total_steps": total_steps,
                    "status": "executing",
                    "elapsed_seconds": round(
                        time.monotonic() - start_time, 2,
                    ),
                })

                counts = execute_pipeline(
                    db=self._db,
                    models=models,
                    on_progress=on_progress,
                    cancel_check=cancel_check,
                )

                rows_processed = counts
                rows_changed = dict(counts)
                models_processed = [
                    name for name, c in counts.items() if c >= 0
                ]

                if cancel_check is not None and cancel_check():
                    status = "cancelled"
        except Exception as exc:  # noqa: BLE001
            logger.exception("Pipeline run failed: %s", exc)
            status = "failed"
            error = str(exc)
            self._emit(on_progress, {
                "type": "error",
                "error": str(exc),
                "elapsed_seconds": round(
                    time.monotonic() - start_time, 2,
                ),
            })

        completed_at = datetime.now(tz=timezone.utc)
        duration = time.monotonic() - start_time

        pipeline_run = PipelineRun(
            run_id=run_id,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=round(duration, 3),
            status=status,
            models_processed=models_processed,
            rows_processed=rows_processed,
            rows_changed=rows_changed,
            trigger=trigger,
            error=error,
        )

        self._stats.record_run(pipeline_run)

        if status == "success":
            self._emit(on_progress, {
                "type": "done",
                "run_id": run_id,
                "duration_seconds": round(duration, 3),
                "step_index": total_steps,
                "total_steps": total_steps,
                "status": "success",
                "elapsed_seconds": round(duration, 2),
            })
        elif status == "cancelled":
            self._emit(on_progress, {
                "type": "cancelled",
                "models_completed": len(models_processed),
                "total_models": total_steps,
                "elapsed_seconds": round(duration, 2),
            })

        return pipeline_run

    def dry_run(self) -> PipelineEstimate:
        """Return an estimate without executing the pipeline.

        sensitivity_tier: 1
        """
        raw_counts = self._get_raw_table_counts()
        total_rows = sum(raw_counts.values())
        pending = self.get_pending_changes()

        estimated_duration = self._stats.estimate_next_duration(
            total_rows,
        )
        last_run = self._stats.get_last_run()
        last_run_at = last_run.completed_at if last_run else None

        return PipelineEstimate(
            estimated_duration_seconds=estimated_duration,
            models_to_process=self._manifest.model_names,
            estimated_rows=raw_counts,
            last_run_at=last_run_at,
            pending_changes=pending,
        )

    def is_stale(self) -> bool:
        """Return True if new raw records exist since last successful run.

        sensitivity_tier: 1
        """
        last = self._stats.get_last_successful_run()
        if last is None:
            return True

        last_run_ts = last.completed_at

        latest_ingestion: datetime | None = None
        for table in self._get_existing_raw_tables():
            try:
                rows = self._db.query(
                    f"SELECT MAX(created_at) AS max_ts FROM {table}",
                )
                if rows and rows[0]["max_ts"] is not None:
                    ts = rows[0]["max_ts"]
                    if isinstance(ts, str):
                        ts = datetime.fromisoformat(ts)
                    if latest_ingestion is None or ts > latest_ingestion:
                        latest_ingestion = ts
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Could not query MAX(created_at) for %s", table,
                )

        if latest_ingestion is None:
            return False

        # Normalize timezone awareness for comparison.
        if latest_ingestion.tzinfo is None:
            latest_ingestion = latest_ingestion.replace(
                tzinfo=timezone.utc,
            )
        if last_run_ts.tzinfo is None:
            last_run_ts = last_run_ts.replace(tzinfo=timezone.utc)

        return latest_ingestion > last_run_ts

    def get_pending_changes(self) -> dict[str, int]:
        """Return per-table count of new rows since last successful run.

        sensitivity_tier: 1
        """
        last = self._stats.get_last_successful_run()

        if last is None:
            return self._get_raw_table_counts()

        last_run_ts = last.completed_at
        if last_run_ts.tzinfo is None:
            last_run_ts = last_run_ts.replace(tzinfo=timezone.utc)

        pending: dict[str, int] = {}
        for table in self._get_existing_raw_tables():
            try:
                rows = self._db.query(
                    f"SELECT COUNT(*) AS n FROM {table} "
                    "WHERE created_at > ?",
                    [last_run_ts.isoformat()],
                )
                pending[table] = rows[0]["n"] if rows else 0
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Could not count pending rows for %s", table,
                )
                pending[table] = -1
        return pending

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_raw_table_counts(self) -> dict[str, int]:
        """Query current row count for every raw table.

        sensitivity_tier: 1
        """
        counts: dict[str, int] = {}
        for table in self._get_existing_raw_tables():
            try:
                rows = self._db.query(
                    f"SELECT COUNT(*) AS n FROM {table}",
                )
                counts[table] = rows[0]["n"] if rows else 0
            except Exception:  # noqa: BLE001
                counts[table] = 0
        return counts

    def _get_existing_raw_tables(self) -> list[str]:
        """Return all currently existing ``raw_*`` tables.

        sensitivity_tier: 1
        """
        try:
            rows = self._db.query(
                "SELECT name AS table_name "
                "FROM sqlite_master "
                "WHERE type = 'table' "
                "AND name LIKE 'raw_%' "
                "ORDER BY name",
            )
            tables = [str(r["table_name"]) for r in rows]
            if tables:
                return tables
        except Exception:  # noqa: BLE001
            logger.warning(
                "Falling back to static raw table list",
                exc_info=True,
            )

        return list(ALL_TABLE_NAMES)

    @staticmethod
    def _emit(
        callback: ProgressCallback | None,
        event: dict[str, Any],
    ) -> None:
        """Safely invoke the progress callback if provided.

        sensitivity_tier: 1
        """
        if callback is not None:
            callback(event)
