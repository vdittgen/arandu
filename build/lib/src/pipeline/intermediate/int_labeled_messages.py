"""Python-based pipeline model that labels messages with emotional metadata.

Reads from stg_messages, runs recent messages through the
EmotionalLabeler (LLM provider), and outputs structured emotional labels.
Older messages get an 'unlabeled' fallback; messages dropped by
:class:`MessageTriager` (promos, automated, ack-only, system events,
media placeholders) get a neutral 'triaged-out' row so we never pay the
labeling cost for noise.

sensitivity_tier: 3 (processes raw message content)
"""

from __future__ import annotations

import json
import logging
import typing as t

logger = logging.getLogger(__name__)

# Default lower bound for LLM labeling when no ingest window is set —
# a cost-control floor. Older messages get the 'unlabeled' fallback.
_LABEL_SINCE_FLOOR = "2026-01-01"


def _label_since() -> str:
    """Effective lower bound for LLM labeling.

    Aligns with the user's ingest window (``ingest_cutoff_iso``, set at
    onboarding) so that choosing e.g. "last 30 days" also bounds
    labeling to ~30 days instead of always reaching back to the static
    floor. Falls back to the floor when no cutoff is configured.

    sensitivity_tier: 1
    """
    try:
        from src.models.llm_provider import load_llm_settings

        cutoff = load_llm_settings().get("ingest_cutoff_iso")
    except Exception:  # noqa: BLE001
        cutoff = None
    if isinstance(cutoff, str) and cutoff:
        return cutoff
    return _LABEL_SINCE_FLOOR

_UNLABELED_ROW = {
    "primary_emotion": "unlabeled",
    "intensity": 0.0,
    "feelings_json": "[]",
    "desires_json": "[]",
    "actors_json": "[]",
    "environment": "",
    "domain": "personal",
    "sensitivity_tier": 3,
}

# Messages the triager drops get a neutral row with a "triaged_out"
# breadcrumb in feelings_json so it's auditable downstream.
_TRIAGED_OUT_ROW = {
    "primary_emotion": "trust",
    "intensity": 0.2,
    "feelings": ["triaged_out"],
    "desires": [],
    "actors": [],
    "environment": "",
    "domain": "personal",
}

if t.TYPE_CHECKING:
    from src.core.sqlite.engine import DatabaseEngine


def _make_row(msg_id: str, label: dict[str, t.Any] | None) -> dict[str, t.Any]:
    """Build an output row from a message id and optional label."""
    if label is not None:
        return {
            "message_id": msg_id,
            "primary_emotion": label["primary_emotion"],
            "intensity": label["intensity"],
            "feelings_json": json.dumps(label["feelings"]),
            "desires_json": json.dumps(label["desires"]),
            "actors_json": json.dumps(label["actors"]),
            "environment": label["environment"],
            "domain": label["domain"],
            "sensitivity_tier": 3,
        }
    return {"message_id": msg_id, **_UNLABELED_ROW}


def execute(db: DatabaseEngine) -> list[dict[str, t.Any]]:
    """Execute the labeling pipeline.

    Labels only messages since the effective cutoff (the configured
    ingest window, else the static floor) via LLM. The triager drops
    promos/automated/ack-only/system noise before labelling so we don't
    pay per-message cost on trash.

    sensitivity_tier: 3
    """
    since = _label_since()
    recent = db.query(
        "SELECT id, content FROM stg_messages WHERE timestamp >= ? "
        "ORDER BY timestamp",
        [since],
    )
    older = db.query(
        "SELECT id FROM stg_messages WHERE timestamp < ?",
        [since],
    )

    logger.info(
        "Labeling %d recent messages (since %s), %d older as unlabeled",
        len(recent),
        since,
        len(older),
    )

    # Older messages → unlabeled
    rows: list[dict[str, t.Any]] = [
        _make_row(m["id"], None) for m in older
    ]

    if not recent:
        return rows

    # AI triage replaces the old regex-based _is_trivial check.
    try:
        from src.agents.triage import MessageTriager

        triager: MessageTriager | None = MessageTriager(db)
    except Exception:  # noqa: BLE001
        logger.warning(
            "Triage unavailable — falling open and labeling all",
            exc_info=True,
        )
        triager = None

    decisions = (
        triager.triage(list(recent))
        if triager is not None
        else None
    )

    to_label: list[dict[str, t.Any]] = []
    triaged_out = 0
    for i, msg in enumerate(recent):
        keep = (
            True
            if decisions is None
            else decisions[i].keep
        )
        if keep:
            to_label.append(msg)
        else:
            rows.append(_make_row(msg["id"], _TRIAGED_OUT_ROW))
            triaged_out += 1

    logger.info(
        "Triage: %d/%d recent messages dropped (kept %d for labeling)",
        triaged_out, len(recent), len(to_label),
    )

    if not to_label:
        return rows

    # Attempt LLM labeling for kept messages only
    try:
        from src.models.labeler import EmotionalLabeler

        labeler = EmotionalLabeler()
        labels = labeler.batch_label([m["content"] for m in to_label])
    except Exception:  # noqa: BLE001
        logger.warning(
            "LLM unavailable — falling back to 'unlabeled' for %d messages",
            len(to_label),
        )
        labels = [None] * len(to_label)

    for i, msg in enumerate(to_label):
        label = labels[i] if i < len(labels) else None
        rows.append(_make_row(msg["id"], label))

    return rows
