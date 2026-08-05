"""Ingestion adapter — bridges a single MCP data tool to a DuckDB raw table.

Each adapter handles one sync cycle: fetch records from an MCP tool,
transform fields per ToolTemplate mappings, deduplicate against existing
rows, and upsert new/changed records.

sensitivity_tier: 2 (processes user data during sync)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from src.core.sqlite.engine import DatabaseEngine
from src.extensions.bridges.whatsapp.paths import resolve_whatsapp_store_path
from src.extensions.ingestion.transforms import apply_transform
from src.extensions.models import ToolTemplate

logger = logging.getLogger(__name__)

_CALENDAR_FAKE_TITLE_MARKERS = (
    "calendar operations too slow",
    "no events available - calendar operations too slow",
)
_CALENDAR_FAKE_NOTE_MARKERS = (
    "notoriously slow and unreliable",
    "calendar.app applescript queries are notoriously slow",
)
_REMINDER_FAKE_TEXT_MARKERS = (
    "found 0 lists and 0 reminders",
    "reminder_search_not_implemented_for_performance",
    "found_lists_but_reminders_query_too_slow",
    "reminders_by_id_not_implemented_for_performance",
)
_WHATSAPP_FAKE_TEXT_MARKERS = (
    "invalid time value",
)

_APPLE_TOOL_DEFAULT_ARGS: dict[str, dict[str, Any]] = {
    "list_calendar_events": {"limit": 500},
    "list_reminders": {"limit": 200},
    "list_contacts": {"limit": 2000},
    "list_notes": {"limit": 200},
    "list_emails": {"limit": 500},
    "list_messages": {"limit": 1000},
}

# Record fields scanned for an ingestion timestamp.  The first present
# field wins; this matches the spread of source schemas (iMessage emits
# ``timestamp``, email emits ``date``, etc).
_INGEST_TIMESTAMP_FIELDS = (
    "timestamp", "date", "created_at", "modified_at",
    "startDate", "start_time",
)


def _load_ingest_cutoff() -> datetime | None:
    """Return the configured ingest cutoff or ``None`` when unset.

    Reads ``ingest_cutoff_iso`` from ``~/.arandu/settings.json``.

    sensitivity_tier: 1
    """
    try:
        from src.models.llm_provider import load_llm_settings
    except Exception:  # noqa: BLE001
        return None
    raw = load_llm_settings().get("ingest_cutoff_iso")
    if not raw:
        return None
    return IngestionAdapter._coerce_utc_timestamp(raw)


def _apply_ingest_cutoff(
    records: list[dict[str, Any]],
    cutoff: datetime | None,
) -> list[dict[str, Any]]:
    """Drop records older than ``cutoff``.

    Records without any recognised timestamp field are kept — the
    cutoff is a safety net, not a primary filter.

    sensitivity_tier: 1
    """
    if cutoff is None or not records:
        return records
    kept: list[dict[str, Any]] = []
    for record in records:
        ts: datetime | None = None
        for field in _INGEST_TIMESTAMP_FIELDS:
            if field in record:
                ts = IngestionAdapter._coerce_utc_timestamp(record[field])
                if ts is not None:
                    break
        if ts is None or ts >= cutoff:
            kept.append(record)
    dropped = len(records) - len(kept)
    if dropped:
        logger.info(
            "ingest cutoff dropped %d/%d pre-%s records",
            dropped, len(records), cutoff.isoformat(),
        )
    return kept



def _collapse_by_key(
    records: list[dict[str, Any]],
    dedup_key: list[str],
) -> list[dict[str, Any]]:
    """Drop intra-batch duplicates so two records never share a dedup key.

    Bridges occasionally surface the same logical entity from multiple
    source rows (e.g. one Gmail message in INBOX + label folders, all
    carrying the same ``message_id``). Without this collapse the second
    ``INSERT`` would hit a UNIQUE constraint and abort the sync. First
    occurrence wins so the bridge's ``ORDER BY`` decides which copy is
    retained.

    sensitivity_tier: N/A
    """
    if not dedup_key or not records:
        return records
    seen: dict[tuple[str, ...], dict[str, Any]] = {}
    for record in records:
        key = tuple(str(record.get(col, "")) for col in dedup_key)
        seen.setdefault(key, record)
    return list(seen.values())


def _normalize_value(val: Any) -> str:
    """Normalise a value to a stable string for comparison.

    SQLite returns ``datetime`` objects for TEXT timestamp columns,
    while incoming records carry ISO-8601 strings.  This helper
    converts both representations to a naive ISO string so that
    ``"2025-06-02T10:00:00"`` matches ``datetime(2025,6,2,10,0,
    tzinfo=utc)``.

    sensitivity_tier: 1
    """
    if val is None:
        return "None"
    if isinstance(val, datetime):
        return val.replace(tzinfo=None).isoformat()
    s = str(val)
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.replace(tzinfo=None).isoformat()
    except (ValueError, TypeError):
        return s


def _normalize_phone(raw: str) -> str:
    """Strip a phone string to its last 10 digits for fuzzy matching.

    Works across formatting variants: ``+55 (48) 9201-1083``,
    ``554892011083``, ``554892011083@s.whatsapp.net``.

    sensitivity_tier: 1
    """
    digits = re.sub(r"\D", "", raw)
    return digits[-10:] if len(digits) >= 10 else digits


def _resolve_sender_name(
    sender_jid: str,
    *,
    is_group: bool,
    chat_name: str,
    contact_lookup: dict[str, str],
    push_name: str = "",
    remote_jid_alt: str = "",
) -> str:
    """Resolve a human-readable display name for a WhatsApp sender JID.

    Priority:
    1. ``"me"`` for own messages.
    2. Direct JID match in *contact_lookup*.
    3. Normalized phone match in *contact_lookup* (for ``@s.whatsapp.net``).
    4. ``remoteJidAlt`` phone lookup in *contact_lookup* (for ``@lid``).
    5. *push_name* from the Baileys message (WhatsApp profile name).
    6. *chat_name* for 1:1 chats (the chat name IS the contact name)
       — only if it doesn't look like a raw JID.
    7. Formatted phone number for ``@s.whatsapp.net`` JIDs.
    8. ``"Unknown"`` for ``@lid`` JIDs that can't be resolved.

    sensitivity_tier: 2
    """
    if sender_jid == "me":
        return "me"

    # Direct JID lookup (exact match in chats dict)
    if sender_jid in contact_lookup:
        return contact_lookup[sender_jid]

    # Phone-based lookup for @s.whatsapp.net JIDs
    if sender_jid.endswith("@s.whatsapp.net"):
        phone_raw = sender_jid.removesuffix("@s.whatsapp.net")
        phone_norm = _normalize_phone(phone_raw)
        if phone_norm in contact_lookup:
            return contact_lookup[phone_norm]
        if push_name:
            return push_name
        if not is_group and not _looks_like_jid(chat_name):
            return chat_name
        return f"+{phone_raw}" if phone_raw else sender_jid

    # @lid JIDs: try remoteJidAlt phone lookup, then pushName
    if sender_jid.endswith("@lid"):
        # remoteJidAlt gives the phone-based JID for @lid contacts
        if remote_jid_alt and remote_jid_alt.endswith(
            "@s.whatsapp.net",
        ):
            alt_phone = remote_jid_alt.removesuffix(
                "@s.whatsapp.net",
            )
            alt_norm = _normalize_phone(alt_phone)
            if alt_norm in contact_lookup:
                return contact_lookup[alt_norm]
        # pushName is the WhatsApp profile display name
        if push_name:
            return push_name
        if not is_group and not _looks_like_jid(chat_name):
            return chat_name
        return "Unknown"

    # Fallback for any other format
    if push_name:
        return push_name
    if not is_group and not _looks_like_jid(chat_name):
        return chat_name
    return sender_jid


def _looks_like_jid(text: str) -> bool:
    """Return True if *text* looks like a raw WhatsApp JID.

    sensitivity_tier: 1
    """
    return (
        "@s.whatsapp.net" in text
        or "@g.us" in text
        or "@lid" in text
        or "@broadcast" in text
    )


class SyncError(Exception):
    """Error during a sync cycle.

    sensitivity_tier: 1
    """


@dataclass(frozen=True)
class SyncResult:
    """Result of a single tool sync cycle.

    sensitivity_tier: 1
    """

    connector_id: str
    tool_name: str
    target_table: str
    timestamp: datetime
    rows_fetched: int = 0
    rows_new: int = 0
    rows_updated: int = 0
    rows_unchanged: int = 0
    duration_seconds: float = 0.0
    status: str = "success"  # "success" | "error"
    error: str | None = None


class IngestionAdapter:
    """Bridges a single MCP data tool to a DuckDB raw table.

    Steps per sync cycle:
    1. Call MCP tool to fetch raw records
    2. Transform each record's fields per FieldTemplate mappings
    3. Generate stable IDs for records missing an explicit ``id`` field
    4. Dedup against existing rows using the tool's ``dedup_key``
    5. INSERT new records, UPDATE changed records, skip unchanged
    6. Wrap all DML in a transaction (BEGIN/COMMIT/ROLLBACK)

    sensitivity_tier: 2
    """

    def __init__(
        self,
        connector_id: str,
        tool: ToolTemplate,
        mcp_client: Any,
        db_engine: DatabaseEngine | None,
    ) -> None:
        """Initialize the adapter.

        Args:
            connector_id: The connector this adapter belongs to.
            tool: The ToolTemplate with field mappings and dedup config.
            mcp_client: An already-connected McpClient (or compatible).
            db_engine: The DuckDB engine for reads and writes.

        sensitivity_tier: 2
        """
        self._connector_id = connector_id
        self._tool = tool
        self._client = mcp_client
        self._db = db_engine

        # Pre-compute column list from field mappings
        self._target_columns = [f.target_column for f in tool.fields]
        self._table_columns = self._load_table_columns()

        # Compute max sensitivity tier from fields
        self._max_tier = max(
            (f.sensitivity_tier for f in tool.fields),
            default=2,
        )

    def _load_table_columns(self) -> set[str]:
        """Return target table columns, if discoverable.

        Empty set means unknown/unavailable (e.g. pure transform unit tests).

        sensitivity_tier: 1
        """
        table = self._tool.target_table
        if not table or self._db is None:
            return set()

        try:
            rows = self._db.query(
                f"PRAGMA table_info({table})",
            )
            return {str(r.get("name", "")) for r in rows if r.get("name")}
        except Exception:  # noqa: BLE001
            logger.debug(
                "Could not introspect columns for table %s",
                table,
                exc_info=True,
            )
            return set()

    def _has_column(self, name: str) -> bool:
        """Check whether a column exists in the target table.

        If schema introspection is unavailable, default to True so
        transform-only tests keep existing behavior.

        sensitivity_tier: 1
        """
        if not self._table_columns:
            return True
        return name in self._table_columns

    def sync(self, full: bool = False) -> SyncResult:
        """Execute one sync cycle for this tool.

        Args:
            full: Reserved for future full-refresh mode. Currently
                all syncs use incremental dedup.

        Returns:
            SyncResult with counts and status.

        sensitivity_tier: 2
        """
        start = time.monotonic()
        target = self._tool.target_table or ""

        try:
            raw_records = self._fetch_records()
            if not raw_records:
                return SyncResult(
                    connector_id=self._connector_id,
                    tool_name=self._tool.tool_name,
                    target_table=target,
                    timestamp=datetime.now(tz=timezone.utc),
                    rows_fetched=0,
                    duration_seconds=round(
                        time.monotonic() - start, 3,
                    ),
                )

            filtered_records = self._filter_placeholder_records(raw_records)
            dropped_count = len(raw_records) - len(filtered_records)
            if dropped_count > 0:
                logger.info(
                    "Dropped %d placeholder rows for %s/%s",
                    dropped_count,
                    self._connector_id,
                    self._tool.tool_name,
                )

            if not filtered_records:
                return SyncResult(
                    connector_id=self._connector_id,
                    tool_name=self._tool.tool_name,
                    target_table=target,
                    timestamp=datetime.now(tz=timezone.utc),
                    rows_fetched=0,
                    duration_seconds=round(
                        time.monotonic() - start, 3,
                    ),
                )

            transformed = [
                self._transform_record(r) for r in filtered_records
            ]

            rows_new, rows_updated, rows_unchanged = (
                self._dedup_and_upsert(transformed)
            )

            return SyncResult(
                connector_id=self._connector_id,
                tool_name=self._tool.tool_name,
                target_table=target,
                timestamp=datetime.now(tz=timezone.utc),
                rows_fetched=len(filtered_records),
                rows_new=rows_new,
                rows_updated=rows_updated,
                rows_unchanged=rows_unchanged,
                duration_seconds=round(
                    time.monotonic() - start, 3,
                ),
            )
        except SyncError:
            raise
        except Exception as exc:
            logger.exception(
                "Sync failed for %s/%s: %s",
                self._connector_id,
                self._tool.tool_name,
                exc,
            )
            raise SyncError(
                f"Sync failed for {self._tool.tool_name}: {exc}",
            ) from exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_records(self) -> list[dict[str, Any]]:
        """Fetch records from the appropriate source.

        Routes to native functions for Apple/filesystem/WhatsApp connectors
        and falls back to MCP tool calls for everything else.

        sensitivity_tier: 2
        """
        # WhatsApp native path (store-driven)
        if (
            self._connector_id == "whatsapp"
            and self._tool.tool_name == "list_chats"
        ):
            return self._fetch_whatsapp_store_messages_incremental()

        # Apple native path (direct SQLite reads)
        if self._connector_id.startswith("apple-"):
            return self._fetch_apple_native()

        # Filesystem native path (pathlib scan)
        if self._connector_id == "filesystem":
            return self._fetch_filesystem_native()

        # Default: MCP tool call
        try:
            call_args = self._build_default_call_args()
            return self._client.call_tool(self._tool.tool_name, call_args)
        except Exception as exc:
            raise SyncError(
                f"MCP tool call failed for "
                f"{self._tool.tool_name}: {exc}",
            ) from exc

    def _build_default_call_args(self) -> dict[str, Any] | None:
        """Build default tool arguments for connectors that need hints.

        sensitivity_tier: 1
        """
        call_args: dict[str, Any] | None = None
        if self._connector_id.startswith("apple-"):
            defaults = _APPLE_TOOL_DEFAULT_ARGS.get(self._tool.tool_name)
            if defaults:
                call_args = dict(defaults)

        if (
            self._connector_id == "apple-calendar"
            and self._tool.tool_name == "list_calendar_events"
        ):
            now = datetime.now(tz=timezone.utc)
            call_args = call_args or {}
            from_date = now - timedelta(days=90)
            cutoff = _load_ingest_cutoff()
            if cutoff is not None and cutoff > from_date:
                from_date = cutoff
            call_args["fromDate"] = from_date.isoformat()
            call_args["toDate"] = (
                (now + timedelta(days=90)).isoformat()
            )

        return call_args

    def _fetch_apple_native(self) -> list[dict[str, Any]]:
        """Call apple_bridge_mcp functions directly (no MCP subprocess).

        sensitivity_tier: 2
        """
        from src.extensions.bridges.apple.server import (
            list_calendar_events,
            list_contacts,
            list_emails,
            list_messages,
            list_notes,
            list_reminders,
        )

        tool_map: dict[str, Any] = {
            "list_calendar_events": list_calendar_events,
            "list_reminders": list_reminders,
            "list_contacts": list_contacts,
            "list_notes": list_notes,
            "list_emails": list_emails,
            "list_messages": list_messages,
        }

        fn = tool_map.get(self._tool.tool_name)
        if fn is None:
            logger.warning(
                "No native function for %s/%s — returning empty",
                self._connector_id,
                self._tool.tool_name,
            )
            return []

        args = self._build_default_call_args() or {}
        try:
            records = fn(args)
        except Exception as exc:
            # A genuinely empty result must stay a clean success; only an
            # actual failure (TCC permission denied, AppleScript error,
            # SQLite read failure) is an error.  Swallowing this hid the
            # failure behind a "connected / 0 rows" status.
            logger.warning(
                "Native Apple fetch failed for %s/%s: %s",
                self._connector_id,
                self._tool.tool_name,
                exc,
            )
            raise SyncError(
                f"Apple native fetch failed for "
                f"{self._connector_id}/{self._tool.tool_name}: {exc}",
            ) from exc
        return _apply_ingest_cutoff(records, _load_ingest_cutoff())

    def _fetch_filesystem_native(self) -> list[dict[str, Any]]:
        """Scan ~/Documents and ~/Desktop directly (no MCP subprocess).

        sensitivity_tier: 2
        """
        from pathlib import Path as _Path

        scan_dirs = [
            _Path.home() / "Documents",
            _Path.home() / "Desktop",
        ]
        records: list[dict[str, Any]] = []
        for base_dir in scan_dirs:
            if not base_dir.exists():
                continue
            try:
                for entry in base_dir.rglob("*"):
                    if not entry.is_file():
                        continue
                    try:
                        stat = entry.stat()
                        records.append({
                            "id": hashlib.sha256(
                                str(entry).encode()
                            ).hexdigest()[:16],
                            "source": "filesystem",
                            "filename": entry.name,
                            "filepath": str(entry),
                            "filetype": entry.suffix.lstrip(".") or "unknown",
                            "size_bytes": stat.st_size,
                            "created_at": datetime.fromtimestamp(
                                stat.st_birthtime, tz=timezone.utc,
                            ).isoformat(),
                            "modified_at": datetime.fromtimestamp(
                                stat.st_mtime, tz=timezone.utc,
                            ).isoformat(),
                        })
                    except (OSError, ValueError):
                        continue
            except PermissionError:
                logger.warning(
                    "Permission denied scanning %s", base_dir,
                )
        return records

    def _fetch_whatsapp_store_messages_incremental(self) -> list[dict[str, Any]]:
        """Fetch incremental WhatsApp rows from Baileys store.json.

        The persistent WhatsApp listener keeps ``store.json`` updated by
        ``messages.upsert`` events. This ingestion path reads those events
        directly and never polls ``get_chat_messages``.

        sensitivity_tier: 2
        """
        payload = self._load_whatsapp_store_payload()
        if not payload:
            return []

        chats = payload.get("chats", {})
        messages = payload.get("messages", {})
        if not isinstance(messages, dict):
            return []

        contact_lookup = self._build_whatsapp_contact_lookup(
            chats, messages,
        )

        last_seen_by_chat = self._load_whatsapp_last_seen_by_chat()
        cutoff = _load_ingest_cutoff()
        records: list[dict[str, Any]] = []
        dropped_pre_cutoff = 0

        for chat_id, raw_items in messages.items():
            chat_id_text = str(chat_id).strip()
            if not chat_id_text:
                continue
            if not isinstance(raw_items, list):
                continue

            chat_data = (
                chats.get(chat_id_text, {})
                if isinstance(chats, dict)
                else {}
            )
            if not isinstance(chat_data, dict):
                chat_data = {}

            chat_name = str(
                chat_data.get("name")
                or chat_data.get("subject")
                or chat_id_text,
            ).strip() or chat_id_text
            is_group = chat_id_text.endswith("@g.us")
            if "isGroup" in chat_data:
                is_group = bool(chat_data.get("isGroup"))

            last_seen = last_seen_by_chat.get(chat_id_text)
            for entry in raw_items:
                normalized = self._normalize_whatsapp_store_message_row(
                    chat_id_text,
                    chat_name,
                    is_group,
                    entry,
                    contact_lookup,
                )
                if normalized is None:
                    continue

                parsed_ts = normalized.pop("_parsed_ts", None)
                if (
                    isinstance(parsed_ts, datetime)
                    and last_seen is not None
                    and parsed_ts <= last_seen
                ):
                    continue
                if (
                    cutoff is not None
                    and isinstance(parsed_ts, datetime)
                    and parsed_ts < cutoff
                ):
                    dropped_pre_cutoff += 1
                    continue
                records.append(normalized)

        if dropped_pre_cutoff:
            logger.info(
                "ingest cutoff dropped %d pre-%s WhatsApp messages",
                dropped_pre_cutoff,
                cutoff.isoformat() if cutoff else "?",
            )
        return records

    def _build_whatsapp_contact_lookup(
        self,
        chats: dict[str, Any],
        messages: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        """Build a JID/phone → display name lookup for WhatsApp contacts.

        Sources (in priority order):
        1. ``raw_contacts`` table — Apple Contacts with phone numbers.
        2. Baileys ``chats`` dict — entries with real names.
        3. Message ``pushName`` + ``remoteJidAlt`` — resolves @lid JIDs.

        sensitivity_tier: 2
        """
        lookup: dict[str, str] = {}

        # Source 1: Apple Contacts (phone → name)
        if self._db is not None:
            try:
                rows = self._db.query(
                    "SELECT name, phone FROM raw_contacts "
                    "WHERE phone IS NOT NULL AND name IS NOT NULL "
                    "AND phone != '' AND name != ''",
                )
                for row in rows:
                    phone = _normalize_phone(str(row["phone"]))
                    if phone:
                        lookup[phone] = str(row["name"])
            except Exception:
                logger.debug(
                    "Could not query raw_contacts for WhatsApp lookup",
                    exc_info=True,
                )

        # Source 2: Baileys chats dict (JID → name)
        if isinstance(chats, dict):
            for jid, chat_data in chats.items():
                if not isinstance(chat_data, dict):
                    continue
                name = str(
                    chat_data.get("name")
                    or chat_data.get("subject")
                    or "",
                ).strip()
                if not name:
                    continue
                jid_str = str(jid).strip()
                lookup[jid_str] = name
                if jid_str.endswith("@s.whatsapp.net"):
                    phone = _normalize_phone(
                        jid_str.removesuffix("@s.whatsapp.net"),
                    )
                    if phone and phone not in lookup:
                        lookup[phone] = name

        # Source 3: Message pushName + remoteJidAlt for @lid resolution
        if isinstance(messages, dict):
            for _chat_id, msg_list in messages.items():
                if not isinstance(msg_list, list):
                    continue
                for entry in msg_list:
                    if not isinstance(entry, dict):
                        continue
                    push = str(
                        entry.get("pushName") or "",
                    ).strip()
                    if not push:
                        continue
                    key = entry.get("key")
                    if not isinstance(key, dict):
                        continue
                    if key.get("fromMe"):
                        continue
                    # Map @lid JID → pushName
                    remote = str(
                        key.get("remoteJid") or "",
                    ).strip()
                    if (
                        remote.endswith("@lid")
                        and remote not in lookup
                    ):
                        lookup[remote] = push
                    # Map remoteJidAlt phone → pushName
                    alt = str(
                        key.get("remoteJidAlt") or "",
                    ).strip()
                    if alt.endswith("@s.whatsapp.net"):
                        alt_phone = _normalize_phone(
                            alt.removesuffix("@s.whatsapp.net"),
                        )
                        if alt_phone and alt_phone not in lookup:
                            lookup[alt_phone] = push

        return lookup

    def _load_whatsapp_store_payload(self) -> dict[str, Any]:
        """Load WhatsApp store JSON from the auth directory.

        sensitivity_tier: 1
        """
        store_path = resolve_whatsapp_store_path()
        if not store_path.exists():
            return {}

        try:
            payload = json.loads(store_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.debug(
                "Could not parse WhatsApp store file: %s",
                store_path,
                exc_info=True,
            )
            return {}

        if not isinstance(payload, dict):
            return {}
        return payload

    def _normalize_whatsapp_store_message_row(
        self,
        chat_id: str,
        chat_name: str,
        is_group: bool,
        entry: Any,
        contact_lookup: dict[str, str] | None = None,
    ) -> dict[str, Any] | None:
        """Normalize one store.json message entry to raw_messages shape.

        sensitivity_tier: 2
        """
        if not isinstance(entry, dict):
            return None

        key = entry.get("key")
        message_payload = entry.get("message")
        timestamp_value = entry.get("messageTimestamp")

        # Some snapshots keep a wrapper under entry["message"].
        if (
            not isinstance(key, dict)
            and isinstance(message_payload, dict)
            and isinstance(message_payload.get("key"), dict)
        ):
            key = message_payload.get("key")
            timestamp_value = message_payload.get("messageTimestamp", timestamp_value)
            message_payload = message_payload.get("message")

        if not isinstance(key, dict):
            return None

        remote_jid = str(key.get("remoteJid") or chat_id).strip()
        if not remote_jid:
            return None
        if remote_jid.endswith("@status") or remote_jid.endswith("@broadcast"):
            return None

        parsed_ts = self._coerce_utc_timestamp(timestamp_value)
        if parsed_ts is None:
            return None

        from_me = bool(key.get("fromMe", False))
        participant = str(
            key.get("participant")
            or key.get("participantAlt")
            or "",
        ).strip()

        sender = "me" if from_me else (participant or remote_jid)

        # Extract pushName and remoteJidAlt for @lid resolution
        push_name = str(entry.get("pushName") or "").strip()
        remote_jid_alt = str(
            key.get("remoteJidAlt") or "",
        ).strip()

        # Resolve a human-readable sender name
        sender_name = _resolve_sender_name(
            sender_jid=sender,
            is_group=is_group or remote_jid.endswith("@g.us"),
            chat_name=chat_name,
            contact_lookup=contact_lookup or {},
            push_name=push_name,
            remote_jid_alt=remote_jid_alt,
        )

        content = self._extract_whatsapp_message_text(message_payload)
        if content is None:
            return None

        raw_id = str(key.get("id", "")).strip()
        if raw_id:
            record_id = f"{remote_jid}:{raw_id}"
        else:
            payload = (
                f"{remote_jid}|{parsed_ts.isoformat()}|{sender}|{content}"
            )
            digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
            record_id = f"{remote_jid}:{digest}"

        resolved_chat_name = chat_name or remote_jid

        return {
            "id": record_id,
            "sender": sender,
            "sender_name": sender_name,
            "recipient": remote_jid,
            "content": content,
            "source": "whatsapp",
            "timestamp": parsed_ts.isoformat(),
            "chat_name": resolved_chat_name,
            "is_group": is_group or remote_jid.endswith("@g.us"),
            "is_from_me": from_me,
            "_parsed_ts": parsed_ts,
        }

    @staticmethod
    def _extract_whatsapp_message_text(message_payload: Any) -> str | None:
        """Extract readable text/placeholder from a WhatsApp message payload.

        Returns ``None`` for protocol/control envelopes that should not be
        persisted as user chat messages.

        sensitivity_tier: 2
        """
        if not isinstance(message_payload, dict):
            return "[message]"

        if "protocolMessage" in message_payload:
            return None

        if "conversation" in message_payload:
            return str(message_payload.get("conversation") or "").strip() or "[text]"

        ext = message_payload.get("extendedTextMessage")
        if isinstance(ext, dict):
            txt = str(ext.get("text") or "").strip()
            if txt:
                return txt

        img = message_payload.get("imageMessage")
        if isinstance(img, dict):
            caption = str(img.get("caption") or "").strip()
            return caption or "[image]"

        vid = message_payload.get("videoMessage")
        if isinstance(vid, dict):
            caption = str(vid.get("caption") or "").strip()
            return caption or "[video]"

        doc = message_payload.get("documentMessage")
        if isinstance(doc, dict):
            caption = str(doc.get("caption") or "").strip()
            name = str(doc.get("fileName") or "").strip()
            return caption or name or "[document]"

        if isinstance(message_payload.get("audioMessage"), dict):
            # Check if transcription is available via audio_cache
            return "[audio]"  # Transcription applied post-sync by listener

        if isinstance(message_payload.get("stickerMessage"), dict):
            return "[sticker]"

        if isinstance(message_payload.get("reactionMessage"), dict):
            reaction = message_payload["reactionMessage"]
            emoji = str(reaction.get("text") or "").strip()
            return emoji or "[reaction]"

        first_type = next(iter(message_payload.keys()), "message")
        return f"[{first_type}]"

    def _load_whatsapp_last_seen_by_chat(self) -> dict[str, datetime]:
        """Return max synced timestamp per WhatsApp chat ID.

        sensitivity_tier: 1
        """
        if self._db is None:
            return {}

        try:
            rows = self._db.query(
                "SELECT recipient AS chat_id, MAX(timestamp) AS last_ts "
                "FROM raw_messages WHERE source = ? GROUP BY recipient",
                ["whatsapp"],
            )
        except Exception:  # noqa: BLE001
            logger.debug(
                "Could not load WhatsApp last-seen timestamps",
                exc_info=True,
            )
            return {}

        result: dict[str, datetime] = {}
        for row in rows:
            chat_id = str(row.get("chat_id", "")).strip()
            parsed = self._coerce_utc_timestamp(row.get("last_ts"))
            if chat_id and parsed is not None:
                result[chat_id] = parsed
        return result

    @staticmethod
    def _coerce_utc_timestamp(value: Any) -> datetime | None:
        """Parse multiple timestamp representations into UTC datetimes.

        sensitivity_tier: 1
        """
        if value is None:
            return None

        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)

        if isinstance(value, (int, float)):
            num = float(value)
            if num > 1e12:
                num /= 1000.0
            try:
                return datetime.fromtimestamp(num, tz=timezone.utc)
            except (OSError, OverflowError, ValueError):
                return None

        # Protobuf Long format: {"low": <epoch>, "high": 0, "unsigned": true}
        if isinstance(value, dict) and "low" in value:
            return IngestionAdapter._coerce_utc_timestamp(value["low"])

        text = str(value).strip()
        if not text:
            return None

        try:
            num = float(text)
        except ValueError:
            num = 0.0
        else:
            if num != 0.0:
                if num > 1e12:
                    num /= 1000.0
                try:
                    return datetime.fromtimestamp(num, tz=timezone.utc)
                except (OSError, OverflowError, ValueError):
                    return None

        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def _transform_record(
        self, raw: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply field transforms to a single raw record.

        Maps ``source_name`` → ``target_column`` and applies the
        named transform for each field.  Adds implicit columns
        (``id``, ``source``, ``sensitivity_tier``, ``created_at``)
        if not already present.

        sensitivity_tier: 2
        """
        result: dict[str, Any] = {}

        for field in self._tool.fields:
            raw_value = raw.get(field.source_name)
            result[field.target_column] = apply_transform(
                field.transform, raw_value,
            )

        # Add implicit columns
        if "source" not in result and self._has_column("source"):
            result["source"] = self._connector_id

        if "id" not in result and self._has_column("id"):
            raw_id = raw.get("id")
            raw_id_text = ""
            if raw_id is not None:
                raw_id_text = str(raw_id).strip()
            if raw_id_text and raw_id_text.lower() not in {"none", "null"}:
                result["id"] = raw_id_text
            else:
                result["id"] = self._generate_id(result)

        if (
            "sensitivity_tier" not in result
            and self._has_column("sensitivity_tier")
        ):
            result["sensitivity_tier"] = self._max_tier

        if "created_at" not in result and self._has_column("created_at"):
            result["created_at"] = (
                datetime.now(tz=timezone.utc).isoformat()
            )

        return result

    def _generate_id(
        self, record: dict[str, Any],
    ) -> str:
        """Generate a deterministic ID from dedup key values.

        Uses SHA-256 of ``connector_id:tool_name:sorted_key_values``,
        truncated to 16 hex characters.

        sensitivity_tier: 1
        """
        parts = [self._connector_id, self._tool.tool_name]
        dedup_values: list[str] = []
        for key in sorted(self._tool.dedup_key):
            val = str(record.get(key, "")).strip()
            if val:
                dedup_values.append(f"{key}={val}")

        if dedup_values:
            parts.extend(dedup_values)
        else:
            # Dedup keys can be empty at transform time (for example, when id
            # itself is the dedup key and was missing in the source payload).
            # Fall back to full row content for deterministic ID generation.
            for key, value in sorted(record.items()):
                parts.append(f"{key}={value}")

        digest = hashlib.sha256(
            ":".join(parts).encode(),
        ).hexdigest()
        return digest[:16]

    def _filter_placeholder_records(
        self,
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Drop known synthetic/placeholder records from flaky connectors.

        sensitivity_tier: 1
        """
        if self._connector_id != "apple-calendar":
            if (
                self._connector_id == "whatsapp"
                and self._tool.tool_name == "list_chats"
            ):
                return [
                    r
                    for r in records
                    if not self._is_fake_whatsapp_record(r)
                ]
            return records

        if self._tool.tool_name == "list_calendar_events":
            return [
                r for r in records
                if not self._is_fake_calendar_record(r)
            ]

        if self._tool.tool_name == "list_reminders":
            return [
                r for r in records
                if not self._is_fake_reminder_record(r)
            ]

        return records

    @staticmethod
    def _is_fake_calendar_record(record: dict[str, Any]) -> bool:
        """Identify synthetic calendar placeholders that must never persist.

        sensitivity_tier: 1
        """
        raw_text = str(record.get("_raw_text", "")).strip().lower()
        if "no events found" in raw_text or "too slow" in raw_text:
            return True

        event_id = str(record.get("id", "")).strip().lower()
        if event_id in {"dummy-event-1", "dummy-event", "none", "null"}:
            return True

        title = str(record.get("title", "")).strip().lower()
        notes = str(
            record.get("description", "") or record.get("notes", ""),
        ).strip().lower()
        if any(marker in title for marker in _CALENDAR_FAKE_TITLE_MARKERS):
            return True
        if any(marker in notes for marker in _CALENDAR_FAKE_NOTE_MARKERS):
            return True

        has_start = bool(
            str(record.get("start_time", "") or record.get("startDate", "")).strip(),
        )
        has_end = bool(
            str(record.get("end_time", "") or record.get("endDate", "")).strip(),
        )
        return not (has_start and has_end)

    @staticmethod
    def _is_fake_reminder_record(record: dict[str, Any]) -> bool:
        """Identify synthetic reminder placeholders that must never persist.

        sensitivity_tier: 1
        """
        raw_text = str(record.get("_raw_text", "")).strip().lower()
        if any(marker in raw_text for marker in _REMINDER_FAKE_TEXT_MARKERS):
            return True

        reminder_id = str(record.get("id", "")).strip().lower()
        if reminder_id in {"none", "null"}:
            reminder_id = ""

        title = str(record.get("title", "")).strip()
        notes = str(record.get("notes", "")).strip()
        list_name = str(record.get("list_name", "")).strip()
        due = str(
            record.get("due_date", "") or record.get("dueDate", ""),
        ).strip()

        if title.lower() == "untitled reminder":
            has_meaningful_fields = any(
                [reminder_id, notes, list_name, due],
            )
            if not has_meaningful_fields:
                return True

        if not title:
            return True

        return False

    @staticmethod
    def _is_fake_whatsapp_record(record: dict[str, Any]) -> bool:
        """Identify WhatsApp MCP plain-text error rows.

        sensitivity_tier: 1
        """
        raw_text = str(record.get("_raw_text", "")).strip().lower()
        if not raw_text:
            return False
        if raw_text.startswith("error:"):
            return True
        return any(marker in raw_text for marker in _WHATSAPP_FAKE_TEXT_MARKERS)

    def _dedup_and_upsert(
        self,
        records: list[dict[str, Any]],
    ) -> tuple[int, int, int]:
        """Dedup records against existing rows and upsert.

        Returns: ``(rows_new, rows_updated, rows_unchanged)``.

        sensitivity_tier: 2
        """
        dedup_key = list(self._tool.dedup_key)
        if self._table_columns:
            dedup_key = [k for k in dedup_key if k in self._table_columns]

        # No dedup key → insert all
        if not dedup_key:
            self._db.execute("BEGIN TRANSACTION")
            try:
                self._insert_batch(records)
                self._db.execute("COMMIT")
            except Exception:
                self._db.execute("ROLLBACK")
                raise
            return (len(records), 0, 0)

        # Collapse intra-batch duplicates before any DB work. Bridges
        # that surface the same logical entity from multiple source rows
        # — e.g. macOS Mail's Envelope Index repeating each Gmail
        # message once per label folder (Inbox + Important + All Mail +
        # any user labels), all sharing the same ``message_id`` — would
        # otherwise feed two rows with the same key into
        # :meth:`_insert_batch`, hit the UNIQUE constraint on the second
        # INSERT, and roll back the entire sync. First occurrence wins;
        # the bridge's ORDER BY decides which copy that is.
        records = _collapse_by_key(records, dedup_key)

        # Take the write lock up front (IMMEDIATE queues on busy_timeout)
        # so the existing-rows lookup below sees a snapshot no concurrent
        # writer can invalidate before our INSERTs run.  With the lookup
        # outside the transaction, a row inserted in the gap by another
        # connector or the listener hit the UNIQUE constraint and rolled
        # back the whole batch.
        self._db.execute("BEGIN IMMEDIATE TRANSACTION")
        try:
            # Build lookup of existing records
            existing = self._build_existing_lookup(records, dedup_key)

            new_records: list[dict[str, Any]] = []
            update_records: list[dict[str, Any]] = []
            unchanged = 0

            for record in records:
                key_tuple = tuple(
                    str(record.get(k, "")) for k in dedup_key
                )
                existing_row = existing.get(key_tuple)

                if existing_row is None:
                    new_records.append(record)
                elif self._record_has_changed(record, existing_row):
                    update_records.append(record)
                else:
                    unchanged += 1

            if new_records:
                self._insert_batch(new_records)
            for record in update_records:
                key_values = {
                    k: record.get(k) for k in dedup_key
                }
                self._update_record(record, key_values)
            self._db.execute("COMMIT")
        except Exception:
            self._db.execute("ROLLBACK")
            raise

        return (len(new_records), len(update_records), unchanged)

    def _build_existing_lookup(
        self,
        records: list[dict[str, Any]],
        dedup_key: list[str],
    ) -> dict[tuple[str, ...], dict[str, Any]]:
        """Query existing rows matching the dedup keys.

        Returns a dict mapping dedup key tuples to row dicts.

        sensitivity_tier: 2
        """
        table = self._tool.target_table
        if not table or not records:
            return {}

        # Build set of unique dedup-key tuples for the incoming batch.
        unique_keys: set[tuple[str, ...]] = set()
        for record in records:
            key_tuple = tuple(
                str(record.get(k, "")) for k in dedup_key
            )
            unique_keys.add(key_tuple)

        if not unique_keys:
            return {}

        result: dict[tuple[str, ...], dict[str, Any]] = {}

        # Chunk size keeps each query well under SQLite's
        # SQLITE_LIMIT_EXPR_DEPTH (default 1000) so large batches
        # don't blow up the parser with a giant OR-chain.
        chunk = 400
        unique_list = list(unique_keys)

        # Fast path: single-column dedup key → use IN (...) which is a
        # single expression node regardless of value count.
        if len(dedup_key) == 1:
            col = dedup_key[0]
            for start in range(0, len(unique_list), chunk):
                batch = unique_list[start:start + chunk]
                placeholders = ", ".join("?" for _ in batch)
                sql = (
                    f"SELECT * FROM {table} "  # noqa: S608
                    f"WHERE {col} IN ({placeholders})"
                )
                params = [k[0] for k in batch]
                try:
                    rows = self._db.query(sql, params)
                except Exception:
                    logger.warning(
                        "Dedup lookup failed for %s",
                        table,
                        exc_info=True,
                    )
                    return {}
                for row in rows:
                    key_tuple = (str(row.get(col, "")),)
                    result[key_tuple] = row
            return result

        # General path: multi-column dedup key → batched OR-chain.
        for start in range(0, len(unique_list), chunk):
            batch = unique_list[start:start + chunk]
            conditions = []
            params: list[Any] = []
            for key_vals in batch:
                clause = " AND ".join(
                    f"{col} = ?" for col in dedup_key
                )
                conditions.append(f"({clause})")
                params.extend(key_vals)
            where = " OR ".join(conditions)
            sql = f"SELECT * FROM {table} WHERE {where}"  # noqa: S608
            try:
                rows = self._db.query(sql, params)
            except Exception:
                logger.warning(
                    "Dedup lookup failed for %s",
                    table,
                    exc_info=True,
                )
                return {}
            for row in rows:
                key_tuple = tuple(
                    str(row.get(k, "")) for k in dedup_key
                )
                result[key_tuple] = row
        return result

    def _insert_batch(
        self, records: list[dict[str, Any]],
    ) -> None:
        """Batch INSERT new records into the target table.

        sensitivity_tier: 2
        """
        if not records:
            return

        table = self._tool.target_table
        if not table:
            return

        # Use column set from first record
        columns = list(records[0].keys())
        if self._table_columns:
            columns = [c for c in columns if c in self._table_columns]
        if not columns:
            logger.warning(
                "No writable columns for %s; skipping insert",
                table,
            )
            return
        placeholders = ", ".join("?" for _ in columns)
        col_list = ", ".join(columns)
        sql = (
            f"INSERT INTO {table} ({col_list}) "
            f"VALUES ({placeholders})"
        )

        for record in records:
            params = [record.get(col) for col in columns]
            self._db.execute(sql, params)

    def _update_record(
        self,
        record: dict[str, Any],
        dedup_key_values: dict[str, Any],
    ) -> None:
        """UPDATE a single existing record by dedup key.

        sensitivity_tier: 2
        """
        table = self._tool.target_table
        if not table:
            return

        if self._table_columns:
            dedup_key_values = {
                k: v
                for k, v in dedup_key_values.items()
                if k in self._table_columns
            }
            if not dedup_key_values:
                return

        # Columns to update (exclude dedup key columns)
        update_cols = [
            c for c in record
            if c not in dedup_key_values
        ]
        if self._table_columns:
            update_cols = [c for c in update_cols if c in self._table_columns]
        if not update_cols:
            return

        set_clause = ", ".join(f"{c} = ?" for c in update_cols)
        where_clause = " AND ".join(
            f"{c} = ?" for c in dedup_key_values
        )

        sql = (
            f"UPDATE {table} SET {set_clause} "
            f"WHERE {where_clause}"
        )
        params = [record.get(c) for c in update_cols]
        params.extend(dedup_key_values.values())
        self._db.execute(sql, params)

    def _record_has_changed(
        self,
        new: dict[str, Any],
        existing: dict[str, Any],
    ) -> bool:
        """Compare two records to determine if an UPDATE is needed.

        Only compares columns defined in the ToolTemplate fields.
        Ignores metadata columns (``created_at``, ``_synced_at``).

        Handles SQLite returning ``datetime`` objects for TEXT timestamp
        columns by normalising both sides to naive ISO strings before
        comparison.

        sensitivity_tier: 1
        """
        for field in self._tool.fields:
            col = field.target_column
            new_val = _normalize_value(new.get(col))
            existing_val = _normalize_value(existing.get(col))
            if new_val != existing_val:
                # Never downgrade transcribed voice notes back to
                # [audio] placeholder — transcription is authoritative.
                # Patch the incoming record so that even if OTHER
                # fields trigger an update, the content column keeps
                # the transcribed text.
                if (
                    col == "content"
                    and new_val == "[audio]"
                    and isinstance(existing_val, str)
                    and existing_val.startswith("[voice note]")
                ):
                    new["content"] = existing_val
                    continue
                return True
        return False
