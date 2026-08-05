"""Notification system data models.

Frozen dataclasses for notification preferences, decisions, delivery
results, and log records.

sensitivity_tier: 2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class NotificationPreference:
    """Per-category notification preference.

    sensitivity_tier: 1
    """

    category: str
    enabled: bool
    muted_until: str | None = None
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True)
class NotificationDecision:
    """Result of the Brain Agent's notification evaluation.

    sensitivity_tier: 3
    """

    should_notify: bool
    category: str
    importance_score: float  # 0.0–10.0
    message: str
    reason: str
    dedupe_key: str
    event_context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DeliveryResult:
    """Result of attempting to deliver a WhatsApp notification.

    sensitivity_tier: 1
    """

    # "sent" | "failed" | "not_configured" | "skipped_dedup" | "skipped_muted"
    status: str
    error: str | None = None
    timestamp: str = ""
    message_id: str | None = None


@dataclass(frozen=True)
class NotificationRecord:
    """Full notification log entry (decision + delivery).

    sensitivity_tier: 2
    """

    id: str
    dedupe_key: str
    category: str
    importance_score: float
    decision: str  # "send" | "skip" | "muted" | "disabled"
    delivery_status: str  # from DeliveryResult.status
    message: str
    opt_out_text: str
    source_type: str  # "pipeline" | "action"
    source_id: str  # run_id or proposal_id
    error: str | None = None
    created_at: str = ""
    # WhatsApp-side id; set when delivery_status == "sent".
    message_id: str | None = None
    # Set by the ack handler when WhatsApp confirms receipt (status >= 3).
    delivered_at: str | None = None
