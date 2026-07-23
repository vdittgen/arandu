"""Tests for cli.py's proactive-notification call sites.

Issue #43: native macOS notifications are the default channel now —
these functions must still evaluate and deliver when no WhatsApp
phone is configured, instead of bailing out entirely. Delivery itself
is exercised in tests/unit/notifications/test_notifier.py; these
tests cover the call sites' control flow (do they still evaluate +
attempt delivery + log, and do they forward the phone correctly).

sensitivity_tier: N/A — test infrastructure
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from src.core import cli as cli_mod
from src.core.sqlite.engine import DatabaseEngine
from src.notifications.models import DeliveryResult
from src.notifications.preference_service import PreferenceService


@pytest.fixture()
def tmp_db(tmp_path: Path) -> DatabaseEngine:
    """Fresh DuckDB engine backed by a temp file."""
    engine = DatabaseEngine(db_path=tmp_path / "test_cli_notify.duckdb")
    yield engine
    engine.close()


@pytest.fixture()
def prefs(tmp_db: DatabaseEngine) -> PreferenceService:
    return PreferenceService(db_engine=tmp_db)


# ================================================================
# _maybe_notify_action
# ================================================================


class TestMaybeNotifyAction:
    """Action-result notifications no longer require WhatsApp."""

    def test_notifies_without_whatsapp_configured(self, tmp_db) -> None:
        """The acceptance criterion: still delivers with no phone set."""
        with (
            patch("src.core.cli._read_whatsapp_phone", return_value=None),
            patch(
                "src.notifications.notifier.deliver_notification",
                return_value=DeliveryResult(status="sent"),
            ) as mock_deliver,
        ):
            cli_mod._maybe_notify_action(
                tmp_db,
                {"status": "success"},
                {"tool_name": "send_email", "proposal_id": "p1"},
            )

        assert mock_deliver.call_count == 1
        assert mock_deliver.call_args.kwargs["whatsapp_phone"] is None

    def test_forwards_configured_whatsapp_phone(self, tmp_db) -> None:
        """When a phone IS configured, it's still passed through (opt-in add-on)."""
        with (
            patch("src.core.cli._read_whatsapp_phone", return_value="+15551234"),
            patch(
                "src.notifications.notifier.deliver_notification",
                return_value=DeliveryResult(status="sent"),
            ) as mock_deliver,
        ):
            cli_mod._maybe_notify_action(
                tmp_db,
                {"status": "success"},
                {"tool_name": "send_email", "proposal_id": "p2"},
            )

        assert mock_deliver.call_args.kwargs["whatsapp_phone"] == "+15551234"

    def test_logs_delivery_result(self, tmp_db, prefs) -> None:
        """A successful delivery is recorded in the notification log."""
        with (
            patch("src.core.cli._read_whatsapp_phone", return_value=None),
            patch(
                "src.notifications.notifier.deliver_notification",
                return_value=DeliveryResult(status="sent"),
            ),
        ):
            cli_mod._maybe_notify_action(
                tmp_db,
                {"status": "success"},
                {"tool_name": "send_email", "proposal_id": "p3"},
            )

        log = prefs.get_notification_log()
        assert len(log) == 1
        assert log[0].delivery_status == "sent"
        assert log[0].source_type == "action"

    def test_still_respects_global_mute(self, tmp_db, prefs) -> None:
        """Global mute must still short-circuit before any delivery attempt."""
        prefs.mute_all()
        with patch(
            "src.notifications.notifier.deliver_notification",
        ) as mock_deliver:
            cli_mod._maybe_notify_action(
                tmp_db,
                {"status": "success"},
                {"tool_name": "send_email", "proposal_id": "p4"},
            )

        mock_deliver.assert_not_called()


# ================================================================
# _maybe_notify_insights
# ================================================================


class TestMaybeNotifyInsights:
    """Insight notifications no longer require WhatsApp."""

    _HIGH_IMPORTANCE_INSIGHT = [
        {"id": "i1", "summary": "Urgent deadline approaching", "importance": 6},
    ]

    def test_notifies_without_whatsapp_configured(self, tmp_db) -> None:
        with (
            patch("src.core.cli._read_whatsapp_phone", return_value=None),
            patch(
                "src.models.llm_provider.create_provider_from_settings",
                side_effect=RuntimeError("no provider in tests"),
            ),
            patch(
                "src.notifications.notifier.deliver_notification",
                return_value=DeliveryResult(status="sent"),
            ) as mock_deliver,
        ):
            cli_mod._maybe_notify_insights(
                tmp_db, self._HIGH_IMPORTANCE_INSIGHT,
            )

        assert mock_deliver.call_count == 1
        assert mock_deliver.call_args.kwargs["whatsapp_phone"] is None

    def test_low_importance_insight_does_not_deliver(self, tmp_db) -> None:
        """Below-threshold insights still correctly produce no delivery attempt."""
        with (
            patch("src.core.cli._read_whatsapp_phone", return_value=None),
            patch(
                "src.models.llm_provider.create_provider_from_settings",
                side_effect=RuntimeError("no provider in tests"),
            ),
            patch(
                "src.notifications.notifier.deliver_notification",
            ) as mock_deliver,
        ):
            cli_mod._maybe_notify_insights(
                tmp_db, [{"id": "i2", "summary": "Routine update", "importance": 1}],
            )

        mock_deliver.assert_not_called()


# ================================================================
# _send_proactive_notification (shared by the realtime message-eval
# and per-sender proactive paths)
# ================================================================


class TestSendProactiveNotification:
    """The shared proactive-delivery helper behind #43's real fix."""

    def test_delivers_with_phone_none(self, tmp_db, prefs) -> None:
        """Core acceptance criterion: works with WhatsApp entirely unconfigured."""
        with patch(
            "src.notifications.notifier.deliver_notification",
            return_value=DeliveryResult(status="sent"),
        ) as mock_deliver:
            cli_mod._send_proactive_notification(
                prefs, None,
                category="realtime_action",
                source_id="msg-1",
                message="You have a new urgent reply",
            )

        assert mock_deliver.call_args.kwargs["whatsapp_phone"] is None
        log = prefs.get_notification_log()
        assert len(log) == 1
        assert log[0].delivery_status == "sent"

    def test_forwards_phone_when_configured(self, tmp_db, prefs) -> None:
        with patch(
            "src.notifications.notifier.deliver_notification",
            return_value=DeliveryResult(status="sent"),
        ) as mock_deliver:
            cli_mod._send_proactive_notification(
                prefs, "+15551234",
                category="realtime_action",
                source_id="msg-2",
                message="You have a new urgent reply",
            )

        assert mock_deliver.call_args.kwargs["whatsapp_phone"] == "+15551234"

    def test_not_configured_result_is_not_logged(self, tmp_db, prefs) -> None:
        """Neither channel available -> nothing was attempted, nothing logged
        (matches the prior behavior for 'listener not running')."""
        with patch(
            "src.notifications.notifier.deliver_notification",
            return_value=DeliveryResult(status="not_configured"),
        ):
            cli_mod._send_proactive_notification(
                prefs, None,
                category="realtime_action",
                source_id="msg-3",
                message="hello",
            )

        assert prefs.get_notification_log() == []

    def test_dedup_prevents_second_send_within_window(self, tmp_db, prefs) -> None:
        with patch(
            "src.notifications.notifier.deliver_notification",
            return_value=DeliveryResult(status="sent"),
        ) as mock_deliver:
            for _ in range(2):
                cli_mod._send_proactive_notification(
                    prefs, None,
                    category="realtime_action",
                    source_id="msg-dup",
                    message="hello",
                )

        assert mock_deliver.call_count == 1

    def test_disabled_category_skips_delivery(self, tmp_db, prefs) -> None:
        prefs.update_preference("realtime_action", enabled=False)
        with patch(
            "src.notifications.notifier.deliver_notification",
        ) as mock_deliver:
            cli_mod._send_proactive_notification(
                prefs, None,
                category="realtime_action",
                source_id="msg-4",
                message="hello",
            )

        mock_deliver.assert_not_called()
