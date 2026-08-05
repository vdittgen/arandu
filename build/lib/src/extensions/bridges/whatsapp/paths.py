"""Path helpers for WhatsApp auth/session artifacts.

sensitivity_tier: 1 (filesystem metadata only)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def resolve_whatsapp_auth_dir() -> Path:
    """Return the best available WhatsApp auth directory.

    Preference order:
    1. ``WHATSAPP_AUTH_DIR`` (if set and exists)
    2. ``~/.claude/connectors/whatsapp/auth``
    3. ``~/.whatsapp-mcp/auth``
    4. ``WHATSAPP_AUTH_DIR`` fallback value (if set but missing)
    5. ``~/.claude/connectors/whatsapp/auth`` default

    sensitivity_tier: 1
    """
    env_val = os.getenv("WHATSAPP_AUTH_DIR")
    env_path = Path(env_val).expanduser() if env_val else None

    candidates: list[Path] = []
    if env_path is not None:
        candidates.append(env_path)
    candidates.extend(
        [
            Path.home() / ".claude" / "connectors" / "whatsapp" / "auth",
            Path.home() / ".whatsapp-mcp" / "auth",
        ]
    )

    for path in candidates:
        if (path / "creds.json").exists() or (path / "store.json").exists():
            return path

    if env_path is not None:
        return env_path
    return candidates[0]


def resolve_whatsapp_store_path() -> Path:
    """Return the expected WhatsApp store JSON path.

    sensitivity_tier: 1
    """
    return resolve_whatsapp_auth_dir() / "store.json"


def resolve_self_jid() -> str | None:
    """Return the user's bare WhatsApp JID from ``creds.json``.

    Reads the ``me.id`` field (e.g. ``"554892011083:34@s.whatsapp.net"``)
    and strips the device suffix to return just the number
    (e.g. ``"554892011083"``).

    This is the canonical JID that Baileys uses in ``store.messages``
    keys and in ``raw_messages.chat_name`` / ``raw_messages.recipient``.
    It may differ from the user's international phone number (e.g.
    Brazil numbers drop a leading "9" in the subscriber part).

    Returns ``None`` if ``creds.json`` is missing or unreadable.

    sensitivity_tier: 1
    """
    creds_path = resolve_whatsapp_auth_dir() / "creds.json"
    if not creds_path.exists():
        return None

    try:
        creds = json.loads(creds_path.read_text(encoding="utf-8"))
        me_id: str = creds.get("me", {}).get("id", "")
        if not me_id:
            return None
        # Strip device suffix (":34") and domain ("@s.whatsapp.net")
        bare = me_id.split(":")[0].split("@")[0]
        return bare if bare else None
    except Exception:  # noqa: BLE001
        logger.debug("Failed to read creds.json for self JID", exc_info=True)
        return None


def resolve_self_lid() -> str | None:
    """Return the user's bare Linked Device ID from ``creds.json``.

    Reads the ``me.lid`` field (e.g. ``"161048623628515:34@lid"``)
    and strips the device suffix to return just the number
    (e.g. ``"161048623628515"``).

    In multi-device Baileys, self-chat messages sent from the phone
    arrive under ``@lid`` JIDs rather than the phone number JID.
    This LID is needed to match those messages as self-chat.

    Returns ``None`` if ``creds.json`` is missing or has no ``me.lid``.

    sensitivity_tier: 1
    """
    creds_path = resolve_whatsapp_auth_dir() / "creds.json"
    if not creds_path.exists():
        return None

    try:
        creds = json.loads(creds_path.read_text(encoding="utf-8"))
        me_lid: str = creds.get("me", {}).get("lid", "")
        if not me_lid:
            return None
        bare = me_lid.split(":")[0].split("@")[0]
        return bare if bare else None
    except Exception:  # noqa: BLE001
        logger.debug("Failed to read creds.json for self LID", exc_info=True)
        return None
