"""Tests for the pipeline worker's notification call site.

Issue #43: native macOS notifications are the default channel now —
_maybe_notify_pipeline must still evaluate and deliver when no
WhatsApp phone is configured, instead of bailing out entirely.

sensitivity_tier: N/A — test infrastructure
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from src.core.sqlite.engine import DatabaseEngine
from src.notifications.models import DeliveryResult
from src.notifications.preference_service import PreferenceService
from src.pipeline.worker import _maybe_notify_pipeline


@pytest.fixture()
def tmp_db(tmp_path: Path) -> DatabaseEngine:
    """Fresh DuckDB engine backed by a temp file."""
    engine = DatabaseEngine(db_path=tmp_path / "test_worker_notify.duckdb")
    yield engine
    engine.close()


@pytest.fixture()
def prefs(tmp_db: DatabaseEngine) -> PreferenceService:
    return PreferenceService(db_engine=tmp_db)


class _FailedRun:
    """Stand-in for a pipeline RunResult with a failure status."""

    def __init__(self) -> None:
        self.status = "failed"
        self.error = "boom"
        self.run_id = "run-1"


class TestMaybeNotifyPipeline:
    """Pipeline-failure notifications no longer require WhatsApp."""

    def test_notifies_without_whatsapp_configured(self, tmp_db) -> None:
        """The acceptance criterion: still delivers with no phone set."""
        with (
            patch(
                "src.pipeline.worker._read_whatsapp_phone", return_value=None,
            ),
            patch(
                "src.notifications.notifier.deliver_notification",
                return_value=DeliveryResult(status="sent"),
            ) as mock_deliver,
        ):
            _maybe_notify_pipeline(tmp_db, _FailedRun())

        assert mock_deliver.call_count == 1
        assert mock_deliver.call_args.kwargs["whatsapp_phone"] is None

    def test_forwards_configured_whatsapp_phone(self, tmp_db) -> None:
        with (
            patch(
                "src.pipeline.worker._read_whatsapp_phone",
                return_value="+15551234",
            ),
            patch(
                "src.notifications.notifier.deliver_notification",
                return_value=DeliveryResult(status="sent"),
            ) as mock_deliver,
        ):
            _maybe_notify_pipeline(tmp_db, _FailedRun())

        assert mock_deliver.call_args.kwargs["whatsapp_phone"] == "+15551234"

    def test_logs_delivery_result(self, tmp_db, prefs) -> None:
        with (
            patch(
                "src.pipeline.worker._read_whatsapp_phone", return_value=None,
            ),
            patch(
                "src.notifications.notifier.deliver_notification",
                return_value=DeliveryResult(status="sent"),
            ),
        ):
            _maybe_notify_pipeline(tmp_db, _FailedRun())

        log = prefs.get_notification_log()
        assert len(log) == 1
        assert log[0].delivery_status == "sent"
        assert log[0].source_type == "pipeline"

    def test_still_respects_global_mute(self, tmp_db, prefs) -> None:
        prefs.mute_all()
        with patch(
            "src.notifications.notifier.deliver_notification",
        ) as mock_deliver:
            _maybe_notify_pipeline(tmp_db, _FailedRun())

        mock_deliver.assert_not_called()

    def test_routine_completion_does_not_deliver(self, tmp_db) -> None:
        """A successful (non-failure) run must not attempt delivery."""

        class _OkRun:
            status = "success"
            run_id = "run-2"

        with (
            patch(
                "src.pipeline.worker._read_whatsapp_phone", return_value=None,
            ),
            patch(
                "src.notifications.notifier.deliver_notification",
            ) as mock_deliver,
        ):
            _maybe_notify_pipeline(tmp_db, _OkRun())

        mock_deliver.assert_not_called()
