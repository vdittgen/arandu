"""Notification delivery — native macOS + WhatsApp via listener IPC.

Native macOS notifications (:class:`MacNotifier`) are the default
channel: no account setup, works the moment the app is installed.
WhatsApp (:class:`WhatsAppNotifier`) routes all sends through the
running listener subprocess, which owns the sole Baileys connection —
see :func:`send_text_via_running_listener`. :func:`deliver_notification`
composites both, so callers get native delivery even when WhatsApp
isn't configured.

sensitivity_tier: 2 (sends messages containing personal data summaries)
"""

from __future__ import annotations

import logging
import platform
import subprocess
from datetime import datetime, timezone

from src.notifications.models import DeliveryResult

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Opt-out text templates per category
# ------------------------------------------------------------------

OPT_OUT_TEMPLATES: dict[str, str] = {
    "calendar_conflicts": "Reply STOP CALENDAR to opt out of calendar notifications.",
    "health_alerts": "Reply STOP HEALTH to opt out of health notifications.",
    "action_results": "Reply STOP ACTIONS to opt out of action notifications.",
    "pipeline_summary": "Reply STOP PIPELINE to opt out of pipeline notifications.",
    "pending_replies": "Reply STOP REPLIES to opt out of reply notifications.",
    "important_people": "Reply STOP PEOPLE to opt out of people notifications.",
    "birthday_reminders": "Reply STOP BIRTHDAYS to opt out of birthday notifications.",
    "event_actions": "Reply STOP EVENTS to opt out of event notifications.",
    "topic_action": "Reply STOP ALERTS to opt out of action notifications.",
    "topic_enrichment": "Reply STOP ENRICHMENT to opt out of enrichment notifications.",
    "conversation_digest": "Reply STOP DIGEST to opt out of conversation digests.",
}

DEFAULT_OPT_OUT = "Reply STOP ALL to opt out of all notifications."


def get_opt_out_text(category: str) -> str:
    """Return the opt-out hint for a notification category.

    sensitivity_tier: 1
    """
    return OPT_OUT_TEMPLATES.get(category, DEFAULT_OPT_OUT)


def _osascript_quote(text: str) -> str:
    """Quote a string for embedding as an AppleScript literal.

    sensitivity_tier: 1
    """
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


class MacNotifier:
    """Send native macOS notifications via ``osascript``.

    The default notification channel: unlike WhatsApp, it needs no
    account setup and works the moment the app is installed. A no-op
    (``not_configured``) on non-macOS platforms — other OSes are
    tracked separately (issue #43 is "macOS native first").

    sensitivity_tier: 2
    """

    def __init__(self, timeout: float = 5.0) -> None:
        self._timeout = timeout

    def is_configured(self) -> bool:
        """True on macOS only.

        sensitivity_tier: 1
        """
        return platform.system() == "Darwin"

    def send(self, message: str, category: str) -> DeliveryResult:  # noqa: ARG002
        """Display a native notification via ``osascript``.

        ``category`` is accepted for interface parity with
        :class:`WhatsAppNotifier` (native notifications carry no
        opt-out text — that's a WhatsApp-specific reply convention).

        sensitivity_tier: 2
        """
        now_ts = datetime.now(timezone.utc).isoformat()
        if not self.is_configured():
            return DeliveryResult(status="not_configured", timestamp=now_ts)

        script = (
            f"display notification {_osascript_quote(message)} "
            f'with title "Arandu"'
        )
        try:
            subprocess.run(
                ["osascript", "-e", script],
                check=True,
                capture_output=True,
                timeout=self._timeout,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            return DeliveryResult(
                status="failed", error=str(exc), timestamp=now_ts,
            )
        return DeliveryResult(status="sent", timestamp=now_ts)


class WhatsAppNotifier:
    """Send notifications via the running WhatsApp listener process.

    All sends route through
    :func:`~src.extensions.bridges.whatsapp.listener.send_text_via_running_listener`
    (file-based outbox IPC).  Only one Baileys connection per phone can
    exist, so the listener subprocess is the single owner of the session.

    If the phone number is not configured, all sends return
    ``DeliveryResult(status="not_configured")`` without error.

    The *mcp_command* and *mcp_args* parameters are accepted for
    backward compatibility but are **unused**.

    sensitivity_tier: 2
    """

    def __init__(
        self,
        whatsapp_phone: str | None,
        mcp_command: str | None = None,
        mcp_args: tuple[str, ...] = (),
        mcp_timeout: float = 10.0,
        prefer_listener_ipc: bool = True,
    ) -> None:
        self._phone = whatsapp_phone
        self._timeout = mcp_timeout

    def is_configured(self) -> bool:
        """Check whether WhatsApp delivery is configured.

        sensitivity_tier: 1
        """
        return bool(self._phone)

    def send(self, message: str, category: str) -> DeliveryResult:
        """Send a WhatsApp notification with opt-out text appended.

        Returns a ``DeliveryResult`` with status ``"not_configured"``
        if the notifier isn't set up, or ``"failed"`` if the listener
        process isn't running.

        Args:
            message: The notification body text.
            category: The notification category (for opt-out text).

        Returns:
            A delivery result describing what happened.

        sensitivity_tier: 2
        """
        now_ts = datetime.now(timezone.utc).isoformat()

        if not self.is_configured():
            return DeliveryResult(
                status="not_configured",
                timestamp=now_ts,
            )

        opt_out = get_opt_out_text(category)
        full_message = f"{message}\n\n---\n{opt_out}"

        return self._send_via_listener(full_message, now_ts)

    def _send_via_listener(
        self,
        full_message: str,
        now_ts: str,
    ) -> DeliveryResult:
        """Send through the running persistent listener process.

        sensitivity_tier: 2
        """
        try:
            from src.extensions.bridges.whatsapp.listener import (
                send_text_via_running_listener,
            )
            from src.extensions.bridges.whatsapp.paths import (
                resolve_self_jid,
                resolve_self_lid,
            )
        except Exception:  # noqa: BLE001
            return DeliveryResult(
                status="failed",
                error="WhatsApp listener module not available",
                timestamp=now_ts,
            )

        # In multi-device Baileys, the phone's self-chat thread uses @lid
        # JIDs (Linked Device IDs), NOT @s.whatsapp.net.  Sending to a
        # phone-number @s.whatsapp.net JID creates a SEPARATE chat thread
        # on the phone.  Use @lid when available, fall back to @s.whatsapp.net.
        self_lid = resolve_self_lid()
        if self_lid:
            to_jid = f"{self_lid}@lid"
        else:
            self_jid = resolve_self_jid()
            to_jid = f"{self_jid}@s.whatsapp.net" if self_jid else str(self._phone)

        response = send_text_via_running_listener(
            to=to_jid,
            message=full_message,
            timeout_seconds=max(8.0, self._timeout * 2.0),
        )
        if response is None:
            return DeliveryResult(
                status="failed",
                error="WhatsApp listener is not running",
                timestamp=now_ts,
            )

        status = str(response.get("status") or "").strip().lower()
        if status == "sent":
            return DeliveryResult(
                status="sent",
                timestamp=now_ts,
                message_id=(
                    str(response.get("message_id"))
                    if response.get("message_id")
                    else None
                ),
            )

        return DeliveryResult(
            status="failed",
            error=str(
                response.get("error") or "Listener send failed",
            ),
            timestamp=now_ts,
        )


def deliver_notification(
    message: str,
    category: str,
    *,
    whatsapp_phone: str | None,
    mac_notifier: MacNotifier | None = None,
    whatsapp_notifier: WhatsAppNotifier | None = None,
) -> DeliveryResult:
    """Deliver a notification over every configured channel.

    Native macOS notifications are the default channel and need no
    setup — they're attempted whenever running on macOS. WhatsApp is
    an additional, opt-in channel sent when ``whatsapp_phone`` is
    configured. Both are attempted independently; either one landing
    is enough to notify the user.

    Returns the first "sent" result if any channel delivered,
    "not_configured" if no channel is available at all, or a "failed"
    result (from whichever channel actually ran) otherwise. Callers
    that log a single ``DeliveryResult`` per notification (the
    existing schema — see ``NotificationRecord``) get one coherent
    outcome without a database migration for per-channel tracking.

    sensitivity_tier: 2
    """
    mac = mac_notifier if mac_notifier is not None else MacNotifier()
    wa = (
        whatsapp_notifier
        if whatsapp_notifier is not None
        else WhatsAppNotifier(whatsapp_phone=whatsapp_phone)
    )

    results = [
        n.send(message, category)
        for n in (mac, wa)
        if n.is_configured()
    ]

    if not results:
        return DeliveryResult(
            status="not_configured",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    sent = next((r for r in results if r.status == "sent"), None)
    return sent if sent is not None else results[0]
