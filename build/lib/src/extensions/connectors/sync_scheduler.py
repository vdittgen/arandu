"""Sync scheduler for enabled MCP connectors.

Manages periodic sync timers for each enabled connector. On app launch,
starts schedulers for all enabled connectors. Each connector gets its own
schedule based on the template default.

If a sync fails, retries with exponential backoff (1min, 5min, 15min),
then disables the connector's schedule.

sensitivity_tier: 1 (manages scheduling metadata, no user data)
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Schedule intervals in seconds
SCHEDULE_INTERVALS: dict[str, int] = {
    "every_15min": 900,
    "hourly": 3600,
    "daily": 86400,
    "manual": 0,  # no automatic scheduling
}

# Retry backoff intervals in seconds (1min, 5min, 15min)
RETRY_BACKOFF: tuple[int, ...] = (60, 300, 900)

MAX_RETRIES: int = len(RETRY_BACKOFF)


@dataclass
class SyncStats:
    """Statistics for a single sync run.

    sensitivity_tier: 1
    """

    connector_id: str
    started_at: datetime
    completed_at: datetime | None = None
    status: str = "running"  # "running" | "success" | "error" | "skipped"
    rows_synced: int = 0
    error: str | None = None
    duration_seconds: float = 0.0


@dataclass
class ScheduleEntry:
    """Internal state for a scheduled connector.

    sensitivity_tier: 1
    """

    connector_id: str
    interval_seconds: int
    timer: threading.Timer | None = None
    last_sync: datetime | None = None
    next_sync: datetime | None = None
    consecutive_failures: int = 0
    enabled: bool = True
    last_stats: SyncStats | None = None


class SyncScheduler:
    """Timer-based sync scheduler for MCP connectors.

    Uses threading.Timer for non-blocking periodic scheduling.
    Each connector has its own timer and retry state.

    sensitivity_tier: 1
    """

    def __init__(
        self,
        sync_fn: Callable[[str], SyncStats] | None = None,
    ) -> None:
        """Initialize the scheduler.

        Args:
            sync_fn: Callable that performs the actual sync for a
                connector_id. Returns SyncStats. If None, syncs
                are no-ops (useful for testing schedule logic).

        sensitivity_tier: 1
        """
        self._entries: dict[str, ScheduleEntry] = {}
        self._lock = threading.Lock()
        self._sync_fn = sync_fn
        self._stopped = False
        # Connectors with a sync currently running.  Guards against
        # overlapping runs of the SAME connector (timer + retry timer,
        # timer + run_now, re-schedule while a fire is mid-sync), which
        # would race in the ingestion adapter and abort on UNIQUE
        # constraint violations.
        self._in_flight: set[str] = set()

    def schedule(
        self,
        connector_id: str,
        interval: str,
    ) -> None:
        """Start or update the schedule for a connector.

        Args:
            connector_id: Unique connector ID.
            interval: Schedule key from SCHEDULE_INTERVALS.

        sensitivity_tier: 1
        """
        interval_seconds = SCHEDULE_INTERVALS.get(interval, 0)

        with self._lock:
            # Cancel existing timer
            if connector_id in self._entries:
                self._cancel_timer(connector_id)

            entry = ScheduleEntry(
                connector_id=connector_id,
                interval_seconds=interval_seconds,
                enabled=True,
            )
            self._entries[connector_id] = entry

            if interval_seconds > 0 and not self._stopped:
                self._schedule_next(entry)
                logger.info(
                    "Scheduled %s every %ds",
                    connector_id,
                    interval_seconds,
                )
            else:
                logger.info(
                    "Registered %s with manual schedule",
                    connector_id,
                )

    def unschedule(self, connector_id: str) -> None:
        """Stop and remove the schedule for a connector.

        sensitivity_tier: 1
        """
        with self._lock:
            self._cancel_timer(connector_id)
            self._entries.pop(connector_id, None)
            logger.info("Unscheduled %s", connector_id)

    def run_now(self, connector_id: str) -> SyncStats:
        """Trigger an immediate sync for a connector.

        sensitivity_tier: 1
        """
        return self._execute_sync(connector_id)

    def get_next_sync_times(self) -> dict[str, datetime | None]:
        """Return the next scheduled sync time for each connector.

        sensitivity_tier: 1
        """
        with self._lock:
            return {
                cid: entry.next_sync
                for cid, entry in self._entries.items()
            }

    def get_schedule_info(
        self, connector_id: str,
    ) -> dict[str, Any] | None:
        """Return scheduling info for a specific connector.

        sensitivity_tier: 1
        """
        with self._lock:
            return self._schedule_info_unlocked(connector_id)

    def get_all_schedules(self) -> list[dict[str, Any]]:
        """Return scheduling info for all connectors.

        sensitivity_tier: 1
        """
        with self._lock:
            result = []
            for cid in self._entries:
                info = self._schedule_info_unlocked(cid)
                if info is not None:
                    result.append(info)
            return result

    def _schedule_info_unlocked(
        self, connector_id: str,
    ) -> dict[str, Any] | None:
        """Build schedule info dict (caller must hold _lock).

        sensitivity_tier: 1
        """
        entry = self._entries.get(connector_id)
        if entry is None:
            return None
        return {
            "connector_id": entry.connector_id,
            "interval_seconds": entry.interval_seconds,
            "last_sync": (
                entry.last_sync.isoformat() if entry.last_sync else None
            ),
            "next_sync": (
                entry.next_sync.isoformat() if entry.next_sync else None
            ),
            "consecutive_failures": entry.consecutive_failures,
            "enabled": entry.enabled,
            "last_status": (
                entry.last_stats.status if entry.last_stats else None
            ),
        }

    def stop_all(self) -> None:
        """Cancel all timers and stop the scheduler.

        sensitivity_tier: 1
        """
        with self._lock:
            self._stopped = True
            for connector_id in list(self._entries):
                self._cancel_timer(connector_id)
            logger.info("All schedules stopped")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cancel_timer(self, connector_id: str) -> None:
        """Cancel the timer for a connector (must hold _lock).

        sensitivity_tier: 1
        """
        entry = self._entries.get(connector_id)
        if entry and entry.timer is not None:
            entry.timer.cancel()
            entry.timer = None

    def _schedule_next(self, entry: ScheduleEntry) -> None:
        """Schedule the next sync timer (must hold _lock).

        sensitivity_tier: 1
        """
        if self._stopped or not entry.enabled:
            return

        interval = entry.interval_seconds
        if interval <= 0:
            return

        entry.next_sync = datetime.now(timezone.utc).replace(
            microsecond=0,
        )
        # Add interval offset
        from datetime import timedelta

        entry.next_sync += timedelta(seconds=interval)

        entry.timer = threading.Timer(
            interval,
            self._on_timer_fire,
            args=[entry.connector_id],
        )
        entry.timer.daemon = True
        entry.timer.start()

    def _on_timer_fire(self, connector_id: str) -> None:
        """Called when a timer fires — runs sync and reschedules.

        sensitivity_tier: 1
        """
        stats = self._execute_sync(connector_id)

        with self._lock:
            entry = self._entries.get(connector_id)
            if entry is None or self._stopped:
                return

            if stats.status == "skipped":
                # Another run of this connector is still in flight —
                # neither a success nor a failure; just keep the
                # regular cadence.
                self._schedule_next(entry)
            elif stats.status == "success":
                entry.consecutive_failures = 0
                self._schedule_next(entry)
            else:
                entry.consecutive_failures += 1
                if entry.consecutive_failures > MAX_RETRIES:
                    entry.enabled = False
                    logger.warning(
                        "Disabled schedule for %s after %d consecutive failures",
                        connector_id,
                        entry.consecutive_failures,
                    )
                else:
                    # Retry with backoff
                    backoff_idx = min(
                        entry.consecutive_failures - 1,
                        len(RETRY_BACKOFF) - 1,
                    )
                    backoff = RETRY_BACKOFF[backoff_idx]
                    logger.info(
                        "Retrying %s in %ds (attempt %d/%d)",
                        connector_id,
                        backoff,
                        entry.consecutive_failures,
                        MAX_RETRIES,
                    )
                    from datetime import timedelta

                    entry.next_sync = datetime.now(
                        timezone.utc,
                    ) + timedelta(seconds=backoff)
                    entry.timer = threading.Timer(
                        backoff,
                        self._on_timer_fire,
                        args=[connector_id],
                    )
                    entry.timer.daemon = True
                    entry.timer.start()

    def _execute_sync(self, connector_id: str) -> SyncStats:
        """Run the sync function for a connector.

        sensitivity_tier: 1
        """
        now = datetime.now(timezone.utc)
        stats = SyncStats(
            connector_id=connector_id,
            started_at=now,
        )

        with self._lock:
            if connector_id in self._in_flight:
                stats.status = "skipped"
                stats.completed_at = now
                logger.info(
                    "Sync already in flight for %s — skipping",
                    connector_id,
                )
                return stats
            self._in_flight.add(connector_id)

        try:
            try:
                if self._sync_fn is not None:
                    stats = self._sync_fn(connector_id)
                else:
                    stats.status = "success"
                    stats.completed_at = datetime.now(timezone.utc)
            except Exception as exc:
                stats.status = "error"
                stats.error = str(exc)
                stats.completed_at = datetime.now(timezone.utc)
                logger.exception(
                    "Sync failed for %s: %s", connector_id, exc,
                )

            if stats.completed_at and stats.started_at:
                stats.duration_seconds = (
                    stats.completed_at - stats.started_at
                ).total_seconds()

            with self._lock:
                entry = self._entries.get(connector_id)
                if entry:
                    entry.last_sync = stats.completed_at or now
                    entry.last_stats = stats
        finally:
            with self._lock:
                self._in_flight.discard(connector_id)

        return stats
