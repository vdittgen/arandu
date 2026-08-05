"""Memory and resource monitoring for Arandu.

Provides RSS memory usage and database file sizes for system health
monitoring. Warns when total resource usage exceeds safe thresholds.

sensitivity_tier: 1 (no user data — infrastructure metrics only)
"""

from __future__ import annotations

import logging
import resource
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path.home() / ".arandu" / "data"
WARNING_THRESHOLD_MB = 2048.0

# Database subdirectories relative to the data dir
_DB_PATHS: dict[str, str | list[str]] = {
    "duckdb": "arandu.duckdb",
    "kuzu": "kuzu_db",
    "chromadb": "chromadb",
}


@dataclass
class MemoryReport:
    """System resource snapshot.

    sensitivity_tier: 1
    """

    rss_mb: float
    db_sizes_mb: dict[str, float] = field(default_factory=dict)
    total_db_mb: float = 0.0
    warning: str | None = None


def _dir_size_bytes(path: Path) -> int:
    """Recursively sum file sizes in a directory.

    sensitivity_tier: 1
    """
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            total += entry.stat().st_size
    return total


def get_memory_usage(data_dir: Path | None = None) -> MemoryReport:
    """Collect current memory usage and database file sizes.

    Args:
        data_dir: Override the data directory path.
                  Defaults to ~/.arandu/data/.

    Returns:
        MemoryReport with RSS, DB sizes, and optional warning.

    sensitivity_tier: 1
    """
    base = data_dir or DEFAULT_DATA_DIR

    # RSS: resource.getrusage reports in bytes on macOS, KB on Linux
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == "darwin":
        rss_mb = usage.ru_maxrss / (1024 * 1024)
    else:
        rss_mb = usage.ru_maxrss / 1024

    db_sizes: dict[str, float] = {}
    for db_name, sub_path in _DB_PATHS.items():
        full_path = base / sub_path
        size_bytes = _dir_size_bytes(full_path)
        db_sizes[db_name] = round(size_bytes / (1024 * 1024), 2)

    total_db = sum(db_sizes.values())

    warning = None
    total_usage = rss_mb + total_db
    if total_usage > WARNING_THRESHOLD_MB:
        warning = (
            f"Total resource usage ({total_usage:.0f} MB) "
            f"exceeds {WARNING_THRESHOLD_MB:.0f} MB threshold"
        )

    return MemoryReport(
        rss_mb=round(rss_mb, 2),
        db_sizes_mb=db_sizes,
        total_db_mb=round(total_db, 2),
        warning=warning,
    )


def format_report(report: MemoryReport) -> dict[str, Any]:
    """Convert a MemoryReport to a JSON-serializable dict.

    sensitivity_tier: 1
    """
    return {
        "rss_mb": report.rss_mb,
        "db_sizes_mb": report.db_sizes_mb,
        "total_db_mb": report.total_db_mb,
        "warning": report.warning,
    }
