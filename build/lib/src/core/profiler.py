"""Performance instrumentation for Arandu.

Provides a ``@timed`` decorator and a ``PerformanceLog`` class that records
operation timings to ``~/.arandu/data/perf_log.jsonl``.

Each log entry contains::

    { "operation": str, "duration_ms": float, "timestamp": str,
      "data_size_hint": int | null }

sensitivity_tier: 1 (only operation names and durations, no user data)
"""

from __future__ import annotations

import contextlib
import functools
import json
import logging
import time
from collections import defaultdict
from collections.abc import Callable, Generator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

DEFAULT_LOG_PATH = Path.home() / ".arandu" / "data" / "perf_log.jsonl"

F = TypeVar("F", bound=Callable[..., Any])


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------


@dataclass(frozen=True)
class PerfEntry:
    """A single performance measurement.

    sensitivity_tier: 1
    """

    operation: str
    duration_ms: float
    timestamp: str
    data_size_hint: int | None = None


# ------------------------------------------------------------------
# PerformanceLog
# ------------------------------------------------------------------


class PerformanceLog:
    """Append-only performance log backed by a JSONL file.

    Uses a module-level singleton so that ``@timed`` decorators across
    the codebase write to the same log without explicit wiring.

    sensitivity_tier: 1
    """

    _instance: PerformanceLog | None = None

    def __init__(self, log_path: Path = DEFAULT_LOG_PATH) -> None:
        """Initialize the log.

        Args:
            log_path: File path for the JSONL log.
        """
        self._log_path = log_path
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list[PerfEntry] = []

    @classmethod
    def get(cls, log_path: Path = DEFAULT_LOG_PATH) -> PerformanceLog:
        """Return the singleton instance, creating it if needed."""
        if cls._instance is None:
            cls._instance = cls(log_path=log_path)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Discard the singleton (useful for testing)."""
        cls._instance = None

    def record(self, entry: PerfEntry) -> None:
        """Append an entry to both the in-memory buffer and the JSONL file."""
        self._entries.append(entry)
        try:
            with open(self._log_path, "a") as fh:
                fh.write(
                    json.dumps(
                        {
                            "operation": entry.operation,
                            "duration_ms": round(entry.duration_ms, 2),
                            "timestamp": entry.timestamp,
                            "data_size_hint": entry.data_size_hint,
                        }
                    )
                    + "\n"
                )
        except OSError:
            logger.warning("Failed to write perf entry to %s", self._log_path)

    @property
    def entries(self) -> list[PerfEntry]:
        """Return a copy of in-memory entries."""
        return list(self._entries)

    def load_from_disk(self) -> None:
        """Replace in-memory entries with those from the JSONL file."""
        if not self._log_path.exists():
            return
        self._entries.clear()
        with open(self._log_path) as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    data = json.loads(stripped)
                    self._entries.append(
                        PerfEntry(
                            operation=data["operation"],
                            duration_ms=data["duration_ms"],
                            timestamp=data["timestamp"],
                            data_size_hint=data.get("data_size_hint"),
                        )
                    )
                except (json.JSONDecodeError, KeyError):
                    continue

    def report(self) -> str:
        """Return a human-readable summary of the slowest operations."""
        entries = self._entries
        if not entries:
            self.load_from_disk()
            entries = self._entries
        if not entries:
            return "No performance data recorded."

        stats: dict[str, list[float]] = defaultdict(list)
        for entry in entries:
            stats[entry.operation].append(entry.duration_ms)

        sorted_ops = sorted(
            stats.items(),
            key=lambda x: max(x[1]),
            reverse=True,
        )

        lines: list[str] = [
            "",
            "=" * 72,
            "  Arandu Performance Report",
            "=" * 72,
            f"  Total entries: {len(entries)}",
            f"  Unique operations: {len(stats)}",
            "-" * 72,
            f"  {'Operation':<40} {'Calls':>6} {'Avg ms':>10} {'Max ms':>10}",
            "-" * 72,
        ]

        for op, durations in sorted_ops:
            avg = sum(durations) / len(durations)
            mx = max(durations)
            lines.append(
                f"  {op:<40} {len(durations):>6} {avg:>10.1f} {mx:>10.1f}"
            )

        lines.append("-" * 72)
        lines.append("")
        lines.append("  TOP 5 BOTTLENECKS:")
        for i, (op, durations) in enumerate(sorted_ops[:5], 1):
            mx = max(durations)
            avg = sum(durations) / len(durations)
            lines.append(f"    {i}. {op} -- {mx:.1f}ms max, {avg:.1f}ms avg")

        lines.append("")
        lines.append("=" * 72)
        return "\n".join(lines)

    def clear(self) -> None:
        """Remove all entries from memory and delete the log file."""
        self._entries.clear()
        if self._log_path.exists():
            self._log_path.unlink()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _result_len(result: Any) -> int | None:
    """Return len(result) for lists/dicts, None otherwise.

    sensitivity_tier: N/A
    """
    if isinstance(result, (list, dict)):
        return len(result)
    return None


# ------------------------------------------------------------------
# @timed decorator
# ------------------------------------------------------------------


def timed(
    operation: str | None = None,
    *,
    size_fn: Callable[[Any], int | None] | None = _result_len,
) -> Callable[[F], F]:
    """Decorator that records execution time of a function.

    Args:
        operation: Human-readable operation name.  Defaults to
                   ``ClassName.method_name`` for methods, or
                   ``module.func_name`` for top-level functions.
        size_fn: Callable that extracts a data-size hint from the return
                 value.  Defaults to ``len(result)`` for lists/dicts.

    sensitivity_tier: 1
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            op_name = operation
            if op_name is None:
                if (
                    args
                    and hasattr(args[0], "__class__")
                    and not isinstance(args[0], type)
                ):
                    op_name = f"{args[0].__class__.__name__}.{fn.__name__}"
                else:
                    op_name = fn.__qualname__

            start = time.perf_counter()
            raised = False
            result = None
            try:
                result = fn(*args, **kwargs)
                return result
            except Exception:
                raised = True
                raise
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                data_size: int | None = None
                if not raised and size_fn is not None:
                    try:
                        data_size = size_fn(result)
                    except Exception:  # noqa: BLE001
                        pass
                entry = PerfEntry(
                    operation=op_name,
                    duration_ms=elapsed_ms,
                    timestamp=datetime.now().isoformat(),
                    data_size_hint=data_size,
                )
                PerformanceLog.get().record(entry)
                logger.debug(
                    "PERF %s: %.1fms (size=%s)",
                    op_name,
                    elapsed_ms,
                    data_size,
                )

        return wrapper  # type: ignore[return-value]

    return decorator


# ------------------------------------------------------------------
# timed_block context manager
# ------------------------------------------------------------------


@contextlib.contextmanager
def timed_block(
    operation: str,
    data_size_hint: int | None = None,
) -> Generator[None, None, None]:
    """Context manager for timing inline code blocks.

    Args:
        operation: Name for this timed block.
        data_size_hint: Optional pre-known size hint.

    sensitivity_tier: 1
    """
    start = time.perf_counter()
    yield
    elapsed_ms = (time.perf_counter() - start) * 1000
    entry = PerfEntry(
        operation=operation,
        duration_ms=elapsed_ms,
        timestamp=datetime.now().isoformat(),
        data_size_hint=data_size_hint,
    )
    PerformanceLog.get().record(entry)
    logger.debug("PERF %s: %.1fms", operation, elapsed_ms)
