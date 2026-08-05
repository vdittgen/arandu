"""Recipient profile lookup for action-proposal tone calibration.

When the agent proposes a reply, the body has to match the relationship
between the user and the recipient: spouses get an intimate first-name
register; colleagues get something more reserved. The agent also has to
get *grammatical number* right — a single recipient takes singular
pronouns ("você", "you") even when the inbound message used plural
forms or when the user's own command was ambiguous.

This module is the structured-data lookup that gives the param
extractor + judge the signals they need:

- ``relationship`` from ``raw_contacts.relationship`` ("wife",
  "husband", "friend", "colleague", "manager", …) — the user-curated
  source of truth.
- ``addressing`` — the user's own recent outbound style toward this
  person (samples from ``raw_messages`` with ``is_from_me = 1``).
  This is the best tone proxy: the agent should mimic how the user
  *actually* writes to them.
- ``count_recent_messages`` — intimacy proxy (more recent traffic
  → casual / familiar).

Failure modes are deliberate no-ops: if the DB call raises, we return
an empty :class:`RecipientProfile` and the upstream pipeline keeps
working with a generic-tone fallback. The profile is a hint, not a
hard gate.

sensitivity_tier: 3 (relationship labels + message samples are Tier 3)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecipientProfile:
    """Tone-calibration hint for a single named recipient.

    Every field is optional; an empty profile means "no signal", and
    the upstream prompt will leave the body in a neutral register.

    sensitivity_tier: 3
    """

    name: str = ""
    relationship: str = ""  # "wife" / "friend" / "colleague" / ...
    recent_outbound_samples: tuple[str, ...] = field(default_factory=tuple)
    count_recent_messages: int = 0

    @property
    def is_present(self) -> bool:
        """sensitivity_tier: 1"""
        return bool(
            self.relationship
            or self.recent_outbound_samples
            or self.count_recent_messages,
        )


def lookup_recipient_profile(
    name: str | None,
    db: Any,
    *,
    sample_limit: int = 3,
    recent_message_window: int = 50,
) -> RecipientProfile:
    """Pull relationship label + recent outbound tone samples for ``name``.

    Strategy:

    1. ``raw_contacts.relationship`` (LIKE-match on ``name`` or
       ``email`` or ``phone``) — the user's explicit relationship
       declaration is the truth.
    2. ``raw_messages`` rows where ``is_from_me=1`` and the recipient
       matches ``name`` — the user's actual writing style.
    3. Count of recent messages (intimacy proxy).

    All swallowed on error — this is best-effort tone hinting, never
    a hard dependency. Callers fall back to a neutral register when
    the profile is empty.

    sensitivity_tier: 3
    """
    if not name or not name.strip():
        return RecipientProfile()
    if db is None:
        return RecipientProfile()
    clean = name.strip()

    relationship = _query_relationship(db, clean)
    samples = _query_recent_outbound(db, clean, limit=sample_limit)
    count = _count_recent_messages(db, clean, window=recent_message_window)

    return RecipientProfile(
        name=clean,
        relationship=relationship,
        recent_outbound_samples=samples,
        count_recent_messages=count,
    )


def _query_relationship(db: Any, name: str) -> str:
    """Read ``raw_contacts.relationship`` for the first row matching
    ``name``. LIKE-matches on ``name`` so we catch nicknames embedded
    in the canonical contact entry.

    sensitivity_tier: 2
    """
    try:
        rows = db.query(
            "SELECT relationship FROM raw_contacts "
            "WHERE relationship IS NOT NULL "
            "  AND relationship != '' "
            "  AND (name = ? OR name LIKE ? OR name LIKE ?) "
            "ORDER BY LENGTH(name) ASC "
            "LIMIT 1",
            [name, f"%{name}%", f"{name}%"],
        )
    except Exception:  # noqa: BLE001
        logger.debug(
            "recipient_profile: raw_contacts lookup failed",
            exc_info=True,
        )
        return ""
    for row in rows or []:
        value = row.get("relationship") if isinstance(row, dict) else row[0]
        if value:
            return str(value).strip().lower()
    return ""


def _query_recent_outbound(
    db: Any, name: str, *, limit: int,
) -> tuple[str, ...]:
    """Return up to ``limit`` of the user's most recent outbound
    messages to ``name``. These are the best tone-calibration samples
    because they show how the user actually writes to this person.

    sensitivity_tier: 3 (message content)
    """
    try:
        rows = db.query(
            "SELECT content FROM raw_messages "
            "WHERE is_from_me = 1 "
            "  AND (recipient = ? OR chat_name = ? OR sender_name = ?) "
            "  AND content IS NOT NULL "
            "  AND content != '' "
            "ORDER BY timestamp DESC "
            "LIMIT ?",
            [name, name, name, int(limit)],
        )
    except Exception:  # noqa: BLE001
        logger.debug(
            "recipient_profile: raw_messages outbound lookup failed",
            exc_info=True,
        )
        return ()
    out: list[str] = []
    for row in rows or []:
        value = row.get("content") if isinstance(row, dict) else row[0]
        if isinstance(value, str) and value.strip():
            # Cap each sample so the eventual prompt block stays small.
            out.append(value.strip()[:160])
    return tuple(out)


def _count_recent_messages(db: Any, name: str, *, window: int) -> int:
    """Count messages in either direction with ``name`` within the
    last ``window`` rows. Intimacy proxy — close contacts have
    sustained traffic.

    sensitivity_tier: 2
    """
    try:
        rows = db.query(
            "SELECT COUNT(*) AS n FROM raw_messages "
            "WHERE recipient = ? OR chat_name = ? OR sender_name = ?",
            [name, name, name],
        )
    except Exception:  # noqa: BLE001
        logger.debug(
            "recipient_profile: count lookup failed", exc_info=True,
        )
        return 0
    if not rows:
        return 0
    row = rows[0]
    value = row.get("n") if isinstance(row, dict) else row[0]
    try:
        return min(int(value or 0), window)
    except Exception:  # noqa: BLE001
        return 0


def format_profile_for_prompt(profile: RecipientProfile) -> str:
    """Render a :class:`RecipientProfile` as a prompt block.

    Returns an empty string when the profile carries no signal — the
    caller appends only when there's something to say so we don't
    waste tokens on noise.

    sensitivity_tier: 3
    """
    if not profile.is_present:
        return ""
    lines: list[str] = [
        "Recipient profile (match this register; default to singular "
        "addressing for one named recipient):",
        f"- name: {profile.name}",
    ]
    if profile.relationship:
        lines.append(f"- relationship: {profile.relationship}")
    if profile.count_recent_messages:
        lines.append(
            f"- recent message count: {profile.count_recent_messages}",
        )
    if profile.recent_outbound_samples:
        lines.append("- recent outbound samples (the user's own style):")
        for sample in profile.recent_outbound_samples:
            lines.append(f"    • {sample}")
    return "\n".join(lines)


__all__ = [
    "RecipientProfile",
    "format_profile_for_prompt",
    "lookup_recipient_profile",
]
