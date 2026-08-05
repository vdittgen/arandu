"""Recipient resolution for messaging action proposals.

When the user asks to send a message ("send a whatsapp to Elmara"),
the brain must resolve the bare name to a real contact before
drafting the action — otherwise the proposal ships with a name
string the connector cannot route. This module does that lookup
across three sources, ranked so the contacts the user is most
likely to mean surface first:

1. ``mart_contact_summary`` — the user's saved contacts joined
   with their active conversation topics. Ranked by name-match
   quality (exact > prefix > substring) then by
   ``notification_priority`` (which already factors topic
   importance, recency, and volume).
2. ``raw_contacts`` — the staging table. Catches contacts that
   exist in Apple Contacts but haven't generated enough message
   traffic yet to land in the mart.
3. Apple Contacts MCP — the macOS AddressBook, queried via
   ``search_contacts``. Always called on a DB miss so the brain
   can find people the user knows but never messaged from this
   device.

The chat surface (or WhatsApp listener) always asks the user
which candidate to send to, even when there is exactly one
exact match. This is intentional: a wrong recipient on a private
message is a worse failure than one extra confirmation tap.

sensitivity_tier: 3 (contact details are Tier 2/3)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

logger = logging.getLogger(__name__)


Channel = Literal["whatsapp", "email", "imessage"]
Source = Literal["mart", "stg_contacts", "apple_mcp"]


@dataclass(frozen=True)
class ContactCandidate:
    """One ranked candidate for the disambiguation card.

    ``handle`` is the channel-appropriate destination — phone for
    WhatsApp / iMessage, email for mail. ``active_topic`` and
    ``topic_importance`` come from the contact's most-important
    active topic so the UI can render the topic chip that helps
    the user pick the right Elmara.

    sensitivity_tier: 3
    """

    name: str
    handle: str | None
    relationship: str
    active_topic: str
    topic_importance: int
    notification_priority: int
    source: Source


@dataclass(frozen=True)
class ResolvedRecipient:
    """Result of a recipient lookup.

    ``candidates`` is ordered: the most likely match first. May be
    empty when neither the DB nor the MCP fallback found anything,
    in which case the chat surface should still show a disambiguation
    card with a "type a phone / email manually" affordance.

    sensitivity_tier: 3
    """

    original_name: str
    channel: Channel
    candidates: tuple[ContactCandidate, ...]


def handle_field_for_channel(channel: str) -> str:
    """Return the contact attribute used as the destination for ``channel``.

    sensitivity_tier: 1
    """
    if channel == "email":
        return "email"
    return "phone"


def resolve_recipient(
    name: str,
    channel: str,
    db: Any,
    *,
    tool_registry: Any = None,
    mcp_client_factory: Any = None,
    limit: int = 5,
) -> ResolvedRecipient:
    """Look up ``name`` across the marts, staging, and Apple MCP.

    Returns up to ``limit`` ranked :class:`ContactCandidate`. The
    function is best-effort — DB or MCP failures degrade silently
    to an empty / partial result rather than raising, so the
    proposal pipeline can still surface a disambiguation card.

    sensitivity_tier: 3
    """
    clean = (name or "").strip()
    if not clean:
        return ResolvedRecipient(
            original_name=name or "", channel=channel, candidates=(),
        )

    candidates: list[ContactCandidate] = []
    seen_keys: set[str] = set()

    def _add(c: ContactCandidate) -> None:
        key = (c.handle or "").strip().lower() or c.name.lower()
        if key in seen_keys:
            return
        seen_keys.add(key)
        candidates.append(c)

    for c in _query_mart(db, clean, channel, limit):
        _add(c)

    if len(candidates) < limit:
        remaining = limit - len(candidates)
        for c in _query_raw_contacts(db, clean, channel, remaining):
            _add(c)

    if mcp_client_factory is not None and tool_registry is not None:
        mcp_candidates = _query_apple_mcp(
            clean, channel,
            tool_registry=tool_registry,
            mcp_client_factory=mcp_client_factory,
            limit=limit,
        )
        for c in mcp_candidates:
            _add(c)

    return ResolvedRecipient(
        original_name=clean,
        channel=channel,  # type: ignore[arg-type]
        candidates=tuple(candidates[:limit]),
    )


def _query_mart(
    db: Any, name: str, channel: str, limit: int,
) -> list[ContactCandidate]:
    """Rank ``mart_contact_summary`` rows by name-match quality + priority.

    sensitivity_tier: 3
    """
    if db is None:
        return []
    handle_col = handle_field_for_channel(channel)
    lowered = name.lower()
    sql = f"""
        SELECT
            contact_name,
            {handle_col} AS handle,
            COALESCE(relationship, '')          AS relationship,
            COALESCE(top_topic, '')             AS top_topic,
            COALESCE(max_topic_importance, 0)   AS topic_importance,
            COALESCE(notification_priority, 0)  AS priority,
            CASE
                WHEN LOWER(contact_name) = ? THEN 1
                WHEN LOWER(contact_name) LIKE ? THEN 2
                WHEN LOWER(contact_name) LIKE ? THEN 3
                ELSE 9
            END AS match_rank
        FROM mart_contact_summary
        WHERE {handle_col} IS NOT NULL
          AND {handle_col} != ''
          AND (
              LOWER(contact_name) = ?
              OR LOWER(contact_name) LIKE ?
              OR LOWER(contact_name) LIKE ?
              OR LOWER(COALESCE(whatsapp_name, '')) LIKE ?
          )
        ORDER BY match_rank ASC,
                 priority DESC,
                 topic_importance DESC,
                 last_message_at DESC
        LIMIT ?
    """
    params: list[Any] = [
        lowered, f"{lowered}%", f"%{lowered}%",
        lowered, f"{lowered}%", f"%{lowered}%",
        f"%{lowered}%",
        int(limit),
    ]
    try:
        rows = db.query(sql, params)
    except Exception:  # noqa: BLE001
        logger.debug(
            "recipient_resolver: mart_contact_summary lookup failed",
            exc_info=True,
        )
        return []
    return [
        ContactCandidate(
            name=str(row.get("contact_name") or "").strip(),
            handle=_clean(row.get("handle")),
            relationship=str(row.get("relationship") or "").strip().lower(),
            active_topic=str(row.get("top_topic") or "").strip(),
            topic_importance=int(row.get("topic_importance") or 0),
            notification_priority=int(row.get("priority") or 0),
            source="mart",
        )
        for row in (rows or [])
        if row.get("contact_name")
    ]


def _query_raw_contacts(
    db: Any, name: str, channel: str, limit: int,
) -> list[ContactCandidate]:
    """Fuzzy-match ``raw_contacts`` for entries not yet in the mart.

    sensitivity_tier: 3
    """
    if db is None or limit <= 0:
        return []
    handle_col = handle_field_for_channel(channel)
    lowered = name.lower()
    sql = f"""
        SELECT
            name,
            {handle_col} AS handle,
            COALESCE(relationship, '') AS relationship,
            CASE
                WHEN LOWER(name) = ? THEN 1
                WHEN LOWER(name) LIKE ? THEN 2
                WHEN LOWER(name) LIKE ? THEN 3
                ELSE 9
            END AS match_rank
        FROM raw_contacts
        WHERE {handle_col} IS NOT NULL
          AND {handle_col} != ''
          AND (
              LOWER(name) = ?
              OR LOWER(name) LIKE ?
              OR LOWER(name) LIKE ?
          )
        ORDER BY match_rank ASC, LENGTH(name) ASC
        LIMIT ?
    """
    params: list[Any] = [
        lowered, f"{lowered}%", f"%{lowered}%",
        lowered, f"{lowered}%", f"%{lowered}%",
        int(limit),
    ]
    try:
        rows = db.query(sql, params)
    except Exception:  # noqa: BLE001
        logger.debug(
            "recipient_resolver: raw_contacts lookup failed",
            exc_info=True,
        )
        return []
    return [
        ContactCandidate(
            name=str(row.get("name") or "").strip(),
            handle=_clean(row.get("handle")),
            relationship=str(row.get("relationship") or "").strip().lower(),
            active_topic="",
            topic_importance=0,
            notification_priority=0,
            source="stg_contacts",
        )
        for row in (rows or [])
        if row.get("name")
    ]


def _query_apple_mcp(
    name: str,
    channel: str,
    *,
    tool_registry: Any,
    mcp_client_factory: Any,
    limit: int,
) -> list[ContactCandidate]:
    """Search the macOS AddressBook via the apple-contacts MCP server.

    Best-effort: any error degrades to an empty list so the
    disambiguation card still renders DB candidates (or shows the
    "no matches" affordance).

    sensitivity_tier: 2
    """
    command, args = _apple_bridge_command(tool_registry)
    if not command:
        return []
    handle_field = handle_field_for_channel(channel)
    try:
        with mcp_client_factory(command, args, 10.0) as client:
            results = client.call_tool(
                "search_contacts",
                {"query": name, "limit": limit},
            )
    except Exception:  # noqa: BLE001
        logger.debug(
            "recipient_resolver: apple_mcp search_contacts failed",
            exc_info=True,
        )
        return []
    candidates: list[ContactCandidate] = []
    for item in (results or [])[:limit]:
        if not isinstance(item, dict):
            continue
        handle = _clean(item.get(handle_field))
        if not handle:
            continue
        candidates.append(
            ContactCandidate(
                name=str(item.get("name") or "").strip(),
                handle=handle,
                relationship=str(
                    item.get("relationship") or "",
                ).strip().lower(),
                active_topic="",
                topic_importance=0,
                notification_priority=0,
                source="apple_mcp",
            ),
        )
    return candidates


def _apple_bridge_command(
    tool_registry: Any,
) -> tuple[str, tuple[str, ...]]:
    """Return the (command, args) tuple for the apple-contacts MCP.

    Falls through to ``("", ())`` when the catalog doesn't have it
    registered (e.g. running on non-macOS, or apple-contacts was
    never installed).

    sensitivity_tier: 1
    """
    if tool_registry is None:
        return ("", ())
    try:
        catalog = tool_registry._catalog  # noqa: SLF001
        template = catalog.get("apple-contacts")
    except Exception:  # noqa: BLE001
        return ("", ())
    if template is None:
        return ("", ())
    return (template.command, tuple(template.args))


def _clean(value: Any) -> str | None:
    """sensitivity_tier: 1"""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = [
    "ContactCandidate",
    "ResolvedRecipient",
    "handle_field_for_channel",
    "resolve_recipient",
]
