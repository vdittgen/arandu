"""Pipeline processing statistics — append-only JSONL persistence.

Records every SQLMesh pipeline run with timing, row counts, and status.
Provides duration estimation via linear regression on historical data.

The stats file lives at ``~/.arandu/data/pipeline_stats.jsonl``.  Each
line is one JSON-encoded :class:`PipelineRun`.

sensitivity_tier: 1 (infrastructure metrics only, no user data)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean

logger = logging.getLogger(__name__)

DEFAULT_STATS_PATH = (
    Path.home() / ".arandu" / "data" / "pipeline_stats.jsonl"
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PipelineRun:
    """Record of a single SQLMesh pipeline execution.

    sensitivity_tier: 1
    """

    run_id: str
    started_at: datetime
    completed_at: datetime
    duration_seconds: float
    status: str  # "success" | "failed" | "partial"
    models_processed: list[str]
    rows_processed: dict[str, int]
    rows_changed: dict[str, int]
    trigger: str  # "manual" | "scheduled" | "startup"
    error: str | None = None
    plan_summary: str | None = None
    # Re-index outcomes are recorded after the SQLMesh marts complete.
    # The run can be a "success" at producing marts while the vector or
    # graph index separately fails — these surface that distinction.
    vector_index_status: str | None = None  # "success" | "error" | None
    graph_index_status: str | None = None  # "success" | "error" | None
    index_error: str | None = None


@dataclass
class PipelineEstimate:
    """Non-executing estimate produced by dry_run().

    sensitivity_tier: 1
    """

    estimated_duration_seconds: float
    models_to_process: list[str]
    estimated_rows: dict[str, int]
    last_run_at: datetime | None
    pending_changes: dict[str, int]


# ---------------------------------------------------------------------------
# ProcessingStats
# ---------------------------------------------------------------------------


class ProcessingStats:
    """Append-only pipeline run statistics store.

    Persists :class:`PipelineRun` records to a JSONL file.  Reads are
    O(n) file scans — for the small history volumes expected (hundreds
    of runs at most), this is acceptable.

    sensitivity_tier: 1
    """

    def __init__(self, stats_path: Path = DEFAULT_STATS_PATH) -> None:
        self._path = stats_path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record_run(self, run: PipelineRun) -> None:
        """Append a PipelineRun to the JSONL file.

        sensitivity_tier: 1
        """
        record = asdict(run)
        record["started_at"] = run.started_at.isoformat()
        record["completed_at"] = run.completed_at.isoformat()
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def update_index_status(
        self,
        run_id: str,
        vector_index_status: str | None,
        graph_index_status: str | None,
        index_error: str | None,
    ) -> None:
        """Patch a previously-recorded run with its re-index outcome.

        The run record is written by :meth:`record_run` before the
        vector/graph re-index runs, so this rewrites the matching line
        in place.  The history file is small (hundreds of runs), so a
        full read/patch/rewrite is acceptable and keeps all mutation of
        the JSONL in one place.

        sensitivity_tier: 1
        """
        if not self._path.exists():
            return
        runs = self._load_all()
        patched = False
        for run in runs:
            if run.run_id == run_id:
                run.vector_index_status = vector_index_status
                run.graph_index_status = graph_index_status
                run.index_error = index_error
                patched = True
                break
        if not patched:
            return

        lines: list[str] = []
        for run in runs:
            record = asdict(run)
            record["started_at"] = run.started_at.isoformat()
            record["completed_at"] = run.completed_at.isoformat()
            lines.append(json.dumps(record))
        with self._path.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def _load_all(self) -> list[PipelineRun]:
        """Read all records from disk.  Returns [] if file is missing.

        sensitivity_tier: 1
        """
        if not self._path.exists():
            return []
        runs: list[PipelineRun] = []
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    d["started_at"] = datetime.fromisoformat(
                        d["started_at"],
                    )
                    d["completed_at"] = datetime.fromisoformat(
                        d["completed_at"],
                    )
                    runs.append(PipelineRun(**d))
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "Skipping malformed stats line: %r",
                        line[:80],
                    )
        return runs

    def get_last_run(self) -> PipelineRun | None:
        """Return the most recent run regardless of status.

        sensitivity_tier: 1
        """
        runs = self._load_all()
        return runs[-1] if runs else None

    def get_last_successful_run(self) -> PipelineRun | None:
        """Return the most recent run with status == 'success'.

        sensitivity_tier: 1
        """
        for run in reversed(self._load_all()):
            if run.status == "success":
                return run
        return None

    def get_run_history(self, limit: int = 20) -> list[PipelineRun]:
        """Return the last *limit* runs, newest first.

        sensitivity_tier: 1
        """
        runs = self._load_all()
        return list(reversed(runs[-limit:]))

    def get_average_duration(self, last_n: int = 5) -> float | None:
        """Return average duration of the last *last_n* successful runs.

        Returns ``None`` if there are no successful runs.

        sensitivity_tier: 1
        """
        successful = [
            r for r in self._load_all() if r.status == "success"
        ]
        if not successful:
            return None
        sample = successful[-last_n:]
        return mean(r.duration_seconds for r in sample)

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------

    def estimate_next_duration(self, data_size: int) -> float:
        """Estimate next run duration using linear regression on history.

        Uses ``(total_rows_processed, duration_seconds)`` pairs from
        successful runs.  Falls back to ``avg * 1.2`` when fewer than
        3 data points are available, or ``60.0`` with no history at all.

        Args:
            data_size: Total raw row count (independent variable).

        Returns:
            Estimated duration in seconds (always >= 1.0).

        sensitivity_tier: 1
        """
        successful = [
            r for r in self._load_all() if r.status == "success"
        ]

        if len(successful) < 3:
            avg = self.get_average_duration()
            if avg is None:
                return 60.0
            return avg * 1.2

        points = [
            (sum(r.rows_processed.values()), r.duration_seconds)
            for r in successful
        ]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x_mean = mean(xs)
        y_mean = mean(ys)

        denominator = sum((x - x_mean) ** 2 for x in xs)
        if denominator == 0:
            return y_mean

        numerator = sum(
            (x - x_mean) * (y - y_mean) for x, y in points
        )
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean

        return max(intercept + slope * data_size, 1.0)
