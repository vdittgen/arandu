"""Tests for MacNotifier, WhatsAppNotifier, and deliver_notification.

WhatsApp sends route through the persistent listener IPC
(``send_text_via_running_listener``).  No MCP client involvement.
Mac sends shell out to ``osascript``.

sensitivity_tier: N/A — test infrastructure
"""

from __future__ import annotations

import subprocess
from unittest.mock import patch

from src.notifications.models import DeliveryResult
from src.notifications.notifier import (
    DEFAULT_OPT_OUT,
    OPT_OUT_TEMPLATES,
    MacNotifier,
    WhatsAppNotifier,
    deliver_notification,
    get_opt_out_text,
)

# ================================================================
# Opt-out text
# ================================================================


class TestOptOutText:
    """Opt-out template tests."""

    def test_known_category(self) -> None:
        """Known categories return specific text."""
        text = get_opt_out_text("calendar_conflicts")
        assert "CALENDAR" in text

    def test_unknown_category(self) -> None:
        """Unknown category returns default text."""
        text = get_opt_out_text("some_unknown_category")
        assert text == DEFAULT_OPT_OUT

    def test_all_templates_have_stop(self) -> None:
        """All opt-out templates include STOP keyword."""
        for text in OPT_OUT_TEMPLATES.values():
            assert "STOP" in text


# ================================================================
# Configuration
# ================================================================


class TestConfiguration:
    """Notifier configuration tests."""

    def test_is_configured_true(self) -> None:
        """Configured when phone is set."""
        notifier = WhatsAppNotifier(
            whatsapp_phone="+1234567890",
        )
        assert notifier.is_configured() is True

    def test_is_configured_false_no_phone(self) -> None:
        """Not configured without phone."""
        notifier = WhatsAppNotifier(
            whatsapp_phone=None,
        )
        assert notifier.is_configured() is False

    def test_is_configured_with_phone_only(self) -> None:
        """Configured with phone only (no mcp_command needed)."""
        notifier = WhatsAppNotifier(
            whatsapp_phone="+1234567890",
            mcp_command=None,
        )
        assert notifier.is_configured() is True

    def test_not_configured_returns_status(self) -> None:
        """Send on unconfigured notifier returns not_configured."""
        notifier = WhatsAppNotifier(whatsapp_phone=None)
        result = notifier.send("test", "calendar_conflicts")
        assert result.status == "not_configured"
        assert result.error is None

    def test_backward_compat_params_accepted(self) -> None:
        """mcp_command/mcp_args/prefer_listener_ipc are accepted."""
        notifier = WhatsAppNotifier(
            whatsapp_phone="+1234567890",
            mcp_command="npx",
            mcp_args=("-y", "whatsapp-mcp-lifeosai"),
            prefer_listener_ipc=True,
        )
        assert notifier.is_configured() is True


# ================================================================
# Delivery via listener IPC
# ================================================================


class TestDelivery:
    """Delivery tests with mocked listener IPC."""

    def test_send_success(self) -> None:
        """Successful send via listener IPC returns 'sent' status."""
        notifier = WhatsAppNotifier(whatsapp_phone="+1234567890")

        with patch(
            "src.extensions.bridges.whatsapp.listener.send_text_via_running_listener",
            return_value={
                "status": "sent",
                "message_id": "3AA_test_123",
            },
        ):
            result = notifier.send("Hello!", "calendar_conflicts")

        assert result.status == "sent"
        assert result.message_id == "3AA_test_123"

    def test_send_appends_opt_out(self) -> None:
        """Sent message includes opt-out text."""
        captured_args: dict = {}

        def _capture_send(to: str, message: str, timeout_seconds: float):  # noqa: ANN202, ARG001
            captured_args["to"] = to
            captured_args["message"] = message
            return {"status": "sent", "message_id": "msg_1"}

        notifier = WhatsAppNotifier(whatsapp_phone="+1")

        with patch(
            "src.extensions.bridges.whatsapp.listener.send_text_via_running_listener",
            side_effect=_capture_send,
        ):
            notifier.send("Test message", "health_alerts")

        assert "Test message" in captured_args["message"]
        assert "STOP HEALTH" in captured_args["message"]

    def test_send_uses_lid_jid_for_self_chat(self) -> None:
        """Listener IPC uses @lid JID so reply lands in phone's self-chat."""
        captured_args: dict = {}

        def _capture_send(to: str, message: str, timeout_seconds: float):  # noqa: ANN202, ARG001
            captured_args["to"] = to
            return {"status": "sent", "message_id": "msg_1"}

        notifier = WhatsAppNotifier(whatsapp_phone="+5548992011083")

        with patch(
            "src.extensions.bridges.whatsapp.listener.send_text_via_running_listener",
            side_effect=_capture_send,
        ), patch(
            "src.extensions.bridges.whatsapp.paths.resolve_self_lid",
            return_value="161048623628515",
        ):
            notifier.send("Hello", "action_results")

        # Must use @lid JID so replies land in the phone's self-chat thread
        assert captured_args["to"] == "161048623628515@lid"

    def test_send_falls_back_to_jid_when_no_lid(self) -> None:
        """Falls back to @s.whatsapp.net JID if LID is not available."""
        captured_args: dict = {}

        def _capture_send(to: str, message: str, timeout_seconds: float):  # noqa: ANN202, ARG001
            captured_args["to"] = to
            return {"status": "sent", "message_id": "msg_1"}

        notifier = WhatsAppNotifier(whatsapp_phone="+5548992011083")

        with patch(
            "src.extensions.bridges.whatsapp.listener.send_text_via_running_listener",
            side_effect=_capture_send,
        ), patch(
            "src.extensions.bridges.whatsapp.paths.resolve_self_lid",
            return_value=None,
        ), patch(
            "src.extensions.bridges.whatsapp.paths.resolve_self_jid",
            return_value="554892011083",
        ):
            notifier.send("Hello", "action_results")

        # No LID → falls back to Baileys-normalized JID
        assert captured_args["to"] == "554892011083@s.whatsapp.net"

    def test_send_falls_back_to_phone_when_no_creds(self) -> None:
        """Falls back to settings phone if creds.json is not available."""
        captured_args: dict = {}

        def _capture_send(to: str, message: str, timeout_seconds: float):  # noqa: ANN202, ARG001
            captured_args["to"] = to
            return {"status": "sent", "message_id": "msg_1"}

        notifier = WhatsAppNotifier(whatsapp_phone="+5548992011083")

        with patch(
            "src.extensions.bridges.whatsapp.listener.send_text_via_running_listener",
            side_effect=_capture_send,
        ), patch(
            "src.extensions.bridges.whatsapp.paths.resolve_self_lid",
            return_value=None,
        ), patch(
            "src.extensions.bridges.whatsapp.paths.resolve_self_jid",
            return_value=None,
        ):
            notifier.send("Hello", "action_results")

        # No creds → falls back to raw settings phone
        assert captured_args["to"] == "+5548992011083"

    def test_listener_not_running_returns_failed(self) -> None:
        """None response from listener IPC means not running."""
        notifier = WhatsAppNotifier(whatsapp_phone="+1")

        with patch(
            "src.extensions.bridges.whatsapp.listener.send_text_via_running_listener",
            return_value=None,
        ):
            result = notifier.send("test", "action_results")

        assert result.status == "failed"
        assert result.error is not None
        assert "not running" in result.error

    def test_listener_send_failure(self) -> None:
        """Error response from listener IPC returns failed."""
        notifier = WhatsAppNotifier(whatsapp_phone="+1")

        with patch(
            "src.extensions.bridges.whatsapp.listener.send_text_via_running_listener",
            return_value={
                "status": "error",
                "error": "Connection lost",
            },
        ):
            result = notifier.send("test", "action_results")

        assert result.status == "failed"
        assert "Connection lost" in (result.error or "")

    def test_listener_import_failure(self) -> None:
        """Import failure of listener module returns failed."""
        notifier = WhatsAppNotifier(whatsapp_phone="+1")

        with patch(
            "src.notifications.notifier.WhatsAppNotifier._send_via_listener",
        ) as mock_send:
            mock_send.return_value = __import__(
                "src.notifications.models", fromlist=["DeliveryResult"],
            ).DeliveryResult(
                status="failed",
                error="WhatsApp listener module not available",
            )
            result = notifier.send("test", "action_results")

        assert result.status == "failed"

    def test_send_has_timestamp(self) -> None:
        """All delivery results include a timestamp."""
        notifier = WhatsAppNotifier(whatsapp_phone="+1")

        with patch(
            "src.extensions.bridges.whatsapp.listener.send_text_via_running_listener",
            return_value={"status": "sent", "message_id": "m1"},
        ):
            result = notifier.send("test", "action_results")

        assert result.timestamp is not None


# ================================================================
# MacNotifier
# ================================================================


class TestMacNotifier:
    """Native macOS notification tests (osascript, mocked)."""

    def test_is_configured_on_macos(self) -> None:
        """Configured when platform.system() reports Darwin."""
        notifier = MacNotifier()
        with patch("platform.system", return_value="Darwin"):
            assert notifier.is_configured() is True

    def test_is_not_configured_off_macos(self) -> None:
        """Not configured on any non-Darwin platform."""
        notifier = MacNotifier()
        for other in ("Linux", "Windows"):
            with patch("platform.system", return_value=other):
                assert notifier.is_configured() is False

    def test_send_not_configured_returns_status(self) -> None:
        """Send on a non-macOS platform returns not_configured, no subprocess call."""
        notifier = MacNotifier()
        with (
            patch("platform.system", return_value="Linux"),
            patch("subprocess.run") as mock_run,
        ):
            result = notifier.send("test", "action_results")

        assert result.status == "not_configured"
        mock_run.assert_not_called()

    def test_send_success(self) -> None:
        """A clean osascript exit returns 'sent'."""
        notifier = MacNotifier()
        with (
            patch("platform.system", return_value="Darwin"),
            patch("subprocess.run") as mock_run,
        ):
            result = notifier.send("Hello there", "action_results")

        assert result.status == "sent"
        assert mock_run.call_count == 1
        args = mock_run.call_args.args[0]
        assert args[0] == "osascript"
        assert "Hello there" in args[2]

    def test_send_quotes_message_safely(self) -> None:
        """Quotes/backslashes in the message don't break the AppleScript literal."""
        notifier = MacNotifier()
        with (
            patch("platform.system", return_value="Darwin"),
            patch("subprocess.run") as mock_run,
        ):
            notifier.send('She said "hi" \\ bye', "action_results")

        script = mock_run.call_args.args[0][2]
        assert script.count('"') % 2 == 0  # every quote is paired
        # The escaped payload round-trips through AppleScript's own
        # quoting rules (backslash-escaped backslash, then quote).
        assert '\\"hi\\"' in script
        assert "\\\\ bye" in script

    def test_send_failure_returns_failed(self) -> None:
        """A non-zero osascript exit returns 'failed' with the error."""
        notifier = MacNotifier()
        with (
            patch("platform.system", return_value="Darwin"),
            patch(
                "subprocess.run",
                side_effect=subprocess.CalledProcessError(1, "osascript"),
            ),
        ):
            result = notifier.send("test", "action_results")

        assert result.status == "failed"
        assert result.error is not None

    def test_send_timeout_returns_failed(self) -> None:
        """A hung osascript call (timeout) returns 'failed', not an exception."""
        notifier = MacNotifier()
        with (
            patch("platform.system", return_value="Darwin"),
            patch(
                "subprocess.run",
                side_effect=subprocess.TimeoutExpired("osascript", 5.0),
            ),
        ):
            result = notifier.send("test", "action_results")

        assert result.status == "failed"

    def test_send_has_timestamp(self) -> None:
        """All delivery results include a timestamp."""
        notifier = MacNotifier()
        with (
            patch("platform.system", return_value="Darwin"),
            patch("subprocess.run"),
        ):
            result = notifier.send("test", "action_results")

        assert result.timestamp is not None


# ================================================================
# deliver_notification (composite channel routing)
# ================================================================


class _FakeNotifier:
    """Minimal notifier double for composite-routing tests."""

    def __init__(self, *, configured: bool, result: DeliveryResult) -> None:
        self._configured = configured
        self._result = result
        self.send_calls: list[tuple[str, str]] = []

    def is_configured(self) -> bool:
        return self._configured

    def send(self, message: str, category: str) -> DeliveryResult:
        self.send_calls.append((message, category))
        return self._result


class TestDeliverNotification:
    """Composite routing across the native + WhatsApp channels."""

    def test_not_configured_when_neither_channel_available(self) -> None:
        """Neither macOS nor WhatsApp configured -> not_configured, nothing sent."""
        mac = _FakeNotifier(configured=False, result=DeliveryResult(status="sent"))
        wa = _FakeNotifier(configured=False, result=DeliveryResult(status="sent"))

        result = deliver_notification(
            "hi", "action_results",
            whatsapp_phone=None,
            mac_notifier=mac,
            whatsapp_notifier=wa,
        )

        assert result.status == "not_configured"
        assert mac.send_calls == []
        assert wa.send_calls == []

    def test_native_only_delivers_with_whatsapp_unconfigured(self) -> None:
        """The acceptance criterion: native alone is enough to notify."""
        mac = _FakeNotifier(
            configured=True, result=DeliveryResult(status="sent"),
        )
        wa = _FakeNotifier(configured=False, result=DeliveryResult(status="sent"))

        result = deliver_notification(
            "hi", "action_results",
            whatsapp_phone=None,
            mac_notifier=mac,
            whatsapp_notifier=wa,
        )

        assert result.status == "sent"
        assert mac.send_calls == [("hi", "action_results")]
        assert wa.send_calls == []  # not configured -> never attempted

    def test_both_channels_attempted_when_both_configured(self) -> None:
        """WhatsApp is additive on top of native, not a replacement."""
        mac = _FakeNotifier(configured=True, result=DeliveryResult(status="sent"))
        wa = _FakeNotifier(configured=True, result=DeliveryResult(status="sent"))

        deliver_notification(
            "hi", "action_results",
            whatsapp_phone="+1",
            mac_notifier=mac,
            whatsapp_notifier=wa,
        )

        assert mac.send_calls == [("hi", "action_results")]
        assert wa.send_calls == [("hi", "action_results")]

    def test_returns_sent_if_any_channel_succeeds(self) -> None:
        """One channel failing must not mask another channel's success."""
        mac = _FakeNotifier(
            configured=True,
            result=DeliveryResult(status="failed", error="osascript missing"),
        )
        wa = _FakeNotifier(configured=True, result=DeliveryResult(status="sent"))

        result = deliver_notification(
            "hi", "action_results",
            whatsapp_phone="+1",
            mac_notifier=mac,
            whatsapp_notifier=wa,
        )

        assert result.status == "sent"

    def test_returns_failed_when_all_configured_channels_fail(self) -> None:
        """Both channels configured but both fail -> surfaces a failure."""
        mac = _FakeNotifier(
            configured=True,
            result=DeliveryResult(status="failed", error="osascript missing"),
        )
        wa = _FakeNotifier(
            configured=True,
            result=DeliveryResult(status="failed", error="listener down"),
        )

        result = deliver_notification(
            "hi", "action_results",
            whatsapp_phone="+1",
            mac_notifier=mac,
            whatsapp_notifier=wa,
        )

        assert result.status == "failed"

    def test_default_notifiers_constructed_when_not_injected(self) -> None:
        """Without injected fakes, real MacNotifier/WhatsAppNotifier are built."""
        with (
            patch("platform.system", return_value="Linux"),
        ):
            result = deliver_notification(
                "hi", "action_results", whatsapp_phone=None,
            )

        # Off-macOS, no phone: both real channels are unconfigured.
        assert result.status == "not_configured"
