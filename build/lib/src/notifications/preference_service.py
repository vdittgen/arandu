"""Notification preference management with DuckDB persistence.

Manages per-category notification preferences and the notification log.
Follows the ``QueryTracker`` / ``InsightGenerator`` pattern for table
creation and CRUD.

sensitivity_tier: 1
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from src.notifications.models import (
    NotificationPreference,
    NotificationRecord,
)

if TYPE_CHECKING:
    from src.core.sqlite.engine import DatabaseEngine

logger = logging.getLogger(__name__)


class PreferenceService:
    """Manage notification preferences and log in DuckDB.

    sensitivity_tier: 1
    """

    def __init__(self, db_engine: DatabaseEngine) -> None:
        self._db = db_engine
        self._ensure_tables()

    # ----------------------------------------------------------
    # Table setup
    # ----------------------------------------------------------

    def _ensure_tables(self) -> None:
        """Create notification tables if they don't exist.

        sensitivity_tier: 1
        """
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS _notification_preferences (
                category    VARCHAR PRIMARY KEY,
                enabled     INTEGER NOT NULL DEFAULT 1,
                muted_until TEXT,
                created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS _notification_log (
                id               VARCHAR PRIMARY KEY,
                dedupe_key       VARCHAR NOT NULL,
                category         VARCHAR NOT NULL,
                importance_score DOUBLE DEFAULT 0.0,
                decision         VARCHAR NOT NULL,
                delivery_status  VARCHAR NOT NULL DEFAULT 'pending',
                message          TEXT,
                opt_out_text     VARCHAR,
                error            TEXT,
                source_type      VARCHAR NOT NULL,
                source_id        VARCHAR NOT NULL,
                message_id       VARCHAR,
                delivered_at     TEXT,
                created_at       TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        # Idempotent column adds for DBs created before these columns existed.
        for ddl in (
            "ALTER TABLE _notification_log ADD COLUMN message_id VARCHAR",
            "ALTER TABLE _notification_log ADD COLUMN delivered_at TEXT",
        ):
            try:
                self._db.execute(ddl)
            except Exception:  # noqa: BLE001
                pass  # column already exists

    # ----------------------------------------------------------
    # Preference CRUD
    # ----------------------------------------------------------

    def get_preferences(self) -> list[NotificationPreference]:
        """Return all notification preferences.

        sensitivity_tier: 1
        """
        rows = self._db.query(
            "SELECT category, enabled, muted_until, created_at, updated_at "
            "FROM _notification_preferences "
            "ORDER BY category"
        )
        return [self._row_to_pref(r) for r in rows]

    def update_preference(self, category: str, *, enabled: bool) -> None:
        """Insert or update a notification preference.

        sensitivity_tier: 1
        """
        now_ts = datetime.now(timezone.utc).isoformat()
        # DuckDB doesn't have native UPSERT, so delete + insert.
        self._db.execute(
            "DELETE FROM _notification_preferences WHERE category = ?",
            [category],
        )
        self._db.execute(
            "INSERT INTO _notification_preferences "
            "(category, enabled, created_at, updated_at) "
            "VALUES (?, ?, ?, ?)",
            [category, enabled, now_ts, now_ts],
        )

    def is_category_enabled(self, category: str) -> bool:
        """Check whether a category is enabled (default: True).

        sensitivity_tier: 1
        """
        rows = self._db.query(
            "SELECT enabled FROM _notification_preferences "
            "WHERE category = ?",
            [category],
        )
        if not rows:
            return True  # unknown categories default to enabled
        return bool(rows[0]["enabled"])

    def is_muted_globally(self) -> bool:
        """Check whether all notifications are globally muted.

        sensitivity_tier: 1
        """
        rows = self._db.query(
            "SELECT muted_until FROM _notification_preferences "
            "WHERE category = '_global'"
        )
        if not rows or rows[0]["muted_until"] is None:
            return False
        muted_until = rows[0]["muted_until"]
        if isinstance(muted_until, str):
            muted_until = datetime.fromisoformat(muted_until)
        if muted_until.tzinfo is None:
            muted_until = muted_until.replace(tzinfo=timezone.utc)
        return muted_until > datetime.now(timezone.utc)

    def mute_all(self, until: str | None = None) -> None:
        """Mute all notifications until a given timestamp.

        If *until* is ``None``, mutes for 24 hours.

        sensitivity_tier: 1
        """
        if until is None:
            until = (
                datetime.now(timezone.utc) + timedelta(hours=24)
            ).isoformat()
        now_ts = datetime.now(timezone.utc).isoformat()
        self._db.execute(
            "DELETE FROM _notification_preferences WHERE category = '_global'",
        )
        self._db.execute(
            "INSERT INTO _notification_preferences "
            "(category, enabled, muted_until, created_at, updated_at) "
            "VALUES ('_global', false, ?, ?, ?)",
            [until, now_ts, now_ts],
        )

    def unmute_all(self) -> None:
        """Remove the global mute.

        sensitivity_tier: 1
        """
        self._db.execute(
            "DELETE FROM _notification_preferences WHERE category = '_global'",
        )

    # ----------------------------------------------------------
    # Notification log
    # ----------------------------------------------------------

    def log_notification(self, record: NotificationRecord) -> None:
        """Persist a notification decision+delivery record.

        sensitivity_tier: 2
        """
        self._db.execute(
            "INSERT INTO _notification_log "
            "(id, dedupe_key, category, importance_score, decision, "
            "delivery_status, message, opt_out_text, error, "
            "source_type, source_id, message_id, delivered_at, "
            "created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                record.id,
                record.dedupe_key,
                record.category,
                record.importance_score,
                record.decision,
                record.delivery_status,
                record.message,
                record.opt_out_text,
                record.error,
                record.source_type,
                record.source_id,
                record.message_id,
                record.delivered_at,
                record.created_at or datetime.now(timezone.utc).isoformat(),
            ],
        )

    def get_notification_log(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> list[NotificationRecord]:
        """Return paginated notification log entries, newest first.

        sensitivity_tier: 2
        """
        rows = self._db.query(
            "SELECT id, dedupe_key, category, importance_score, "
            "decision, delivery_status, message, opt_out_text, "
            "error, source_type, source_id, message_id, delivered_at, "
            "created_at "
            "FROM _notification_log "
            "ORDER BY created_at DESC "
            f"LIMIT {int(limit)} OFFSET {int(offset)}"
        )
        return [self._row_to_record(r) for r in rows]

    def has_recent_dedup(
        self,
        dedupe_key: str,
        hours: int = 24,
    ) -> bool:
        """Check whether a notification with this key was sent recently.

        sensitivity_tier: 1
        """
        rows = self._db.query(
            "SELECT COUNT(*) AS cnt FROM _notification_log "
            "WHERE dedupe_key = ? AND delivery_status = 'sent' "
            f"AND created_at >= datetime('now', '-{int(hours)} hours')",
            [dedupe_key],
        )
        return bool(rows and rows[0]["cnt"] > 0)

    def get_recent_log_summary(
        self,
        hours: int = 24,
        limit: int = 10,
    ) -> list[dict[str, str]]:
        """Return a brief summary of recent notifications for LLM context.

        sensitivity_tier: 1
        """
        rows = self._db.query(
            "SELECT category, message, created_at "
            "FROM _notification_log "
            "WHERE delivery_status = 'sent' "
            f"AND created_at >= datetime('now', '-{int(hours)} hours') "
            "ORDER BY created_at DESC "
            f"LIMIT {int(limit)}"
        )
        return [
            {
                "category": r["category"],
                "message": r["message"] or "",
                "sent_at": str(r["created_at"]),
            }
            for r in rows
        ]

    # ----------------------------------------------------------
    # Private helpers
    # ----------------------------------------------------------

    @staticmethod
    def _row_to_pref(row: dict) -> NotificationPreference:
        """Convert a DB row to a NotificationPreference.

        sensitivity_tier: 1
        """
        return NotificationPreference(
            category=row["category"],
            enabled=bool(row["enabled"]),
            muted_until=str(row["muted_until"]) if row.get("muted_until") else None,
            created_at=str(row.get("created_at", "")),
            updated_at=str(row.get("updated_at", "")),
        )

    @staticmethod
    def _row_to_record(row: dict) -> NotificationRecord:
        """Convert a DB row to a NotificationRecord.

        sensitivity_tier: 2
        """
        return NotificationRecord(
            id=row["id"],
            dedupe_key=row["dedupe_key"],
            category=row["category"],
            importance_score=float(row.get("importance_score", 0.0)),
            decision=row["decision"],
            delivery_status=row["delivery_status"],
            message=row.get("message") or "",
            opt_out_text=row.get("opt_out_text") or "",
            source_type=row["source_type"],
            source_id=row["source_id"],
            error=row.get("error"),
            created_at=str(row.get("created_at", "")),
            message_id=row.get("message_id"),
            delivered_at=row.get("delivered_at"),
        )

    @staticmethod
    def new_record_id() -> str:
        """Generate a new UUID for a notification record.

        sensitivity_tier: 1
        """
        return str(uuid.uuid4())
