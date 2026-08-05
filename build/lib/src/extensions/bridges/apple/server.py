"""Local MCP bridge for macOS Calendar, Reminders, and Contacts.

Uses direct SQLite reads for Calendar, Reminders, and Contacts data
(instant, all calendars/accounts) and AppleScript for write operations.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

SERVER_NAME = "arandu-apple-bridge"
SERVER_VERSION = "1.0.0"
PROTOCOL_VERSION = "2024-11-05"
OSASCRIPT_TIMEOUT_SECONDS = 15

FIELD_SEP = chr(31)
RECORD_SEP = chr(30)

# macOS Calendar SQLite database path (Core Data epoch: 2001-01-01)
_CALENDAR_DB_PATH = (
    Path.home()
    / "Library"
    / "Group Containers"
    / "group.com.apple.calendar"
    / "Calendar.sqlitedb"
)
_REMINDERS_DB_DIR = (
    Path.home()
    / "Library"
    / "Group Containers"
    / "group.com.apple.reminders"
    / "Container_v1"
    / "Stores"
)
_CONTACTS_DB_DIR = (
    Path.home() / "Library" / "Application Support" / "AddressBook" / "Sources"
)
_NOTES_DB_PATH = (
    Path.home()
    / "Library"
    / "Group Containers"
    / "group.com.apple.notes"
    / "NoteStore.sqlite"
)
_MAIL_DB_PATH = (
    Path.home()
    / "Library"
    / "Mail"
    / "V10"
    / "MailData"
    / "Envelope Index"
)
_MESSAGES_DB_PATH = Path.home() / "Library" / "Messages" / "chat.db"
# iMessage date epoch: 2001-01-01 in nanoseconds
_IMESSAGE_EPOCH_OFFSET = 978307200
_CORE_DATA_EPOCH_OFFSET = 978307200  # seconds from Unix to Core Data
# AddressBook Core Data entity type for contacts (ABCDContact)
_ABCD_CONTACT_ENT = 22


_T = Any  # generic return type alias


def _query_macos_db(
    db_path: Path,
    query_fn: Any,
) -> Any:
    """Run query_fn(conn) on a macOS SQLite database.

    macOS protects certain databases (Notes, Mail, Messages) with a SQLite
    authorizer that denies queries on user tables even when the file is
    readable.  The authorizer is tied to the original file path; copying
    the database to a temp directory bypasses it.

    First tries a direct read-only connection (fast path).  On
    "authorization denied", copies the DB + WAL/SHM to a temp dir and
    retries.

    sensitivity_tier: 1
    """
    # --- fast path: direct connection ---
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return query_fn(conn)
    except sqlite3.DatabaseError as exc:
        if "authorization denied" not in str(exc).lower():
            raise
    finally:
        if conn is not None:
            conn.close()

    # --- fallback: copy to temp to bypass macOS authorizer ---
    tmp_dir = Path(tempfile.mkdtemp())
    tmp = tmp_dir / db_path.name
    conn = None
    try:
        shutil.copy2(db_path, tmp)
        for suffix in ("-wal", "-shm"):
            wal = db_path.parent / (db_path.name + suffix)
            if wal.exists():
                shutil.copy2(wal, tmp_dir / (db_path.name + suffix))
        conn = sqlite3.connect(f"file:{tmp}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return query_fn(conn)
    finally:
        if conn is not None:
            conn.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)


TOOLS: list[dict[str, Any]] = [
    {
        "name": "list_calendar_events",
        "description": "List macOS Calendar events in a date range.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 500},
                "fromDate": {"type": "string"},
                "toDate": {"type": "string"},
            },
            "additionalProperties": True,
        },
    },
    {
        "name": "list_reminders",
        "description": "List macOS Reminders across all lists.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 500},
            },
            "additionalProperties": True,
        },
    },
    {
        "name": "list_contacts",
        "description": "List macOS Contacts from the AddressBook.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 2000},
                "query": {"type": "string"},
            },
            "additionalProperties": True,
        },
    },
    {
        "name": "search_contacts",
        "description": "Search macOS Contacts by name, email, or phone.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
            },
            "required": ["query"],
            "additionalProperties": True,
        },
    },
    {
        "name": "list_notes",
        "description": "List macOS Notes from the Notes app.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 1000,
                },
            },
            "additionalProperties": True,
        },
    },
    {
        "name": "create_note",
        "description": "Create a new note in macOS Notes.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "body": {"type": "string"},
                "folder": {"type": "string"},
            },
            "required": ["title"],
            "additionalProperties": True,
        },
    },
    {
        "name": "list_emails",
        "description": "List emails from macOS Mail.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 1000,
                },
            },
            "additionalProperties": True,
        },
    },
    {
        "name": "list_messages",
        "description": "List iMessage/SMS conversations.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 2000,
                },
            },
            "additionalProperties": True,
        },
    },
    {
        "name": "send_message",
        "description": "Send an iMessage.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "text": {"type": "string"},
            },
            "required": ["to", "text"],
            "additionalProperties": True,
        },
    },
    {
        "name": "create_event",
        "description": "Create a Calendar event in the default calendar.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "start_time": {"type": "string"},
                "end_time": {"type": "string"},
                "location": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["title"],
            "additionalProperties": True,
        },
    },
    {
        "name": "create_reminder",
        "description": "Create a Reminder in a target list.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "list_name": {"type": "string"},
                "notes": {"type": "string"},
                "due_date": {"type": "string"},
            },
            "required": ["title"],
            "additionalProperties": True,
        },
    },
    {
        "name": "delete_event",
        "description": "Delete a Calendar event by title.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
            },
            "required": ["title"],
            "additionalProperties": True,
        },
    },
    {
        "name": "delete_reminder",
        "description": "Delete a Reminder by title.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
            },
            "required": ["title"],
            "additionalProperties": True,
        },
    },
    {
        "name": "delete_note",
        "description": "Delete a note from macOS Notes by title.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
            },
            "required": ["title"],
            "additionalProperties": True,
        },
    },
    {
        "name": "search_notes",
        "description": "Search macOS Notes by title or content.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
            },
            "required": ["query"],
            "additionalProperties": True,
        },
    },
    {
        "name": "update_note",
        "description": "Update an existing note in macOS Notes.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["title", "body"],
            "additionalProperties": True,
        },
    },
    {
        "name": "search_emails",
        "description": "Search macOS Mail by subject, sender, or content.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
            },
            "required": ["query"],
            "additionalProperties": True,
        },
    },
    {
        "name": "send_email",
        "description": "Compose and send an email via macOS Mail.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
                "cc": {"type": "string"},
            },
            "required": ["to", "subject"],
            "additionalProperties": True,
        },
    },
    {
        "name": "reply_email",
        "description": "Reply to an email found by subject in macOS Mail.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["subject", "body"],
            "additionalProperties": True,
        },
    },
    {
        "name": "delete_email",
        "description": "Move an email to Trash in macOS Mail.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
            },
            "required": ["subject"],
            "additionalProperties": True,
        },
    },
    {
        "name": "move_email",
        "description": "Move an email to a different mailbox in macOS Mail.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "folder": {"type": "string"},
            },
            "required": ["subject", "folder"],
            "additionalProperties": True,
        },
    },
    {
        "name": "flag_email",
        "description": "Flag or unflag an email in macOS Mail.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "flagged": {"type": "boolean"},
            },
            "required": ["subject"],
            "additionalProperties": True,
        },
    },
]


def _send(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _error_response(
    req_id: Any,
    code: int,
    message: str,
) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": code, "message": message},
    }


def _tool_error_result(message: str) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": message}],
        "isError": True,
    }


def _run_osascript(script: str) -> str:
    try:
        proc = subprocess.run(
            ["osascript"],
            input=script,
            text=True,
            capture_output=True,
            timeout=OSASCRIPT_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "AppleScript timed out. Grant Calendar/Reminders automation "
            "permission and try again.",
        ) from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        if not detail:
            detail = "Unknown AppleScript error"
        raise RuntimeError(detail)
    return proc.stdout.strip()


def _ensure_app_running(
    app_name: str,
    *,
    timeout_s: float = 2.0,
    poll_interval_s: float = 0.1,
) -> None:
    # macOS app launch is async: `open -a` / `launch application` return
    # before the target app has registered its Apple Event handlers, so an
    # immediate `tell application "X"` can fail with -600 "Application
    # isn't running". This helper closes the race by pre-launching in the
    # background and polling readiness before the caller runs its script.
    try:
        subprocess.run(
            ["open", "-g", "-a", app_name],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Launching {app_name} timed out via `open -g -a`.",
        ) from exc

    probe = f'tell application "{app_name}" to return (running as text)'
    deadline = time.monotonic() + timeout_s
    last_detail = ""
    while True:
        try:
            if _run_osascript(probe).strip().lower() == "true":
                return
        except RuntimeError as exc:
            last_detail = str(exc)
        if time.monotonic() >= deadline:
            break
        time.sleep(poll_interval_s)
    suffix = f" Last probe error: {last_detail}" if last_detail else ""
    raise RuntimeError(
        f"{app_name} did not become ready within {timeout_s:.1f}s.{suffix}",
    )


def _escape_apple_text(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", " ")
        .replace("\r", " ")
    )


def _arg_str(arguments: dict[str, Any], key: str) -> str | None:
    """Return ``arguments[key]`` as a clean string or ``None``.

    The param-extractor LLM may emit JSON ``null`` (decoded to Python
    ``None``) for optional fields. The naive ``str(d.get(k, "")).strip()
    or None`` pattern stringifies that ``None`` into the literal string
    ``"None"`` and downstream parsers (``datetime.fromisoformat``) then
    raise ``Invalid isoformat string: 'None'``. This helper normalises
    missing / ``None`` / empty / whitespace / literal-``"None"`` /
    literal-``"null"`` values to a real ``None``.

    sensitivity_tier: 1
    """
    raw = arguments.get(key)
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return text


def _parse_iso_to_epoch(value: str | None) -> int | None:
    if value is None:
        return None
    text = value.strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    if len(text) == 10:
        text = f"{text}T00:00:00"
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    local_tz = datetime.now().astimezone().tzinfo or timezone.utc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=local_tz)
    return int(dt.timestamp())


def _stable_id(prefix: str, parts: list[str]) -> str:
    digest = hashlib.sha256(":".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "yes", "1"}


def _epoch_text_to_iso(value: str) -> str:
    text = value.strip()
    if not text:
        return ""
    try:
        return datetime.fromtimestamp(
            float(text),
            tz=timezone.utc,
        ).isoformat().replace("+00:00", "Z")
    except (TypeError, ValueError, OSError):
        return ""


def _parse_delimited_rows(
    text: str,
    columns: list[str],
) -> list[dict[str, str]]:
    if not text:
        return []
    rows: list[dict[str, str]] = []
    for chunk in text.split(RECORD_SEP):
        if not chunk:
            continue
        parts = chunk.split(FIELD_SEP)
        if len(parts) < len(columns):
            parts.extend([""] * (len(columns) - len(parts)))
        row = {columns[idx]: parts[idx] for idx in range(len(columns))}
        rows.append(row)
    return rows


def _calendar_placeholder(
    title: str,
    description: str,
    event_id: str,
) -> bool:
    t = title.strip().lower()
    d = description.strip().lower()
    eid = event_id.strip().lower()
    return (
        "calendar operations too slow" in t
        or "calendar.app applescript queries are notoriously slow" in d
        or eid in {"dummy-event-1", "dummy-event"}
    )


def _reminder_placeholder(
    title: str,
    reminder_id: str,
    notes: str,
    list_name: str,
    due_date: str,
) -> bool:
    rid = reminder_id.strip().lower()
    if rid in {"none", "null"}:
        rid = ""
    if title.strip().lower() == "untitled reminder":
        return not any([rid, notes.strip(), list_name.strip(), due_date.strip()])
    return False


def _list_calendar_names_script() -> str:
    """AppleScript that returns all calendar names, newline-separated."""
    return """
tell application "Calendar"
    set calNames to {}
    repeat with cal in calendars
        set end of calNames to name of cal
    end repeat
    set AppleScript's text item delimiters to linefeed
    set outText to calNames as text
    set AppleScript's text item delimiters to ""
    return outText
end tell
""".strip()


def _calendar_list_script(
    from_epoch: int,
    to_epoch: int,
    limit: int,
    calendar_name: str | None = None,
) -> str:
    cal_target = (
        f'set calList to {{calendar "{_escape_apple_text(calendar_name)}"}}'
        if calendar_name
        else "set calList to (get calendars)"
    )
    return f"""
on replace_text(theText, oldItem, newItem)
    set AppleScript's text item delimiters to oldItem
    set textItems to every text item of theText
    set AppleScript's text item delimiters to newItem
    set outText to textItems as text
    set AppleScript's text item delimiters to ""
    return outText
end replace_text

on normalize_text(v)
    if v is missing value then return ""
    set t to v as text
    set t to my replace_text(t, (character id 30), " ")
    set t to my replace_text(t, (character id 31), " ")
    set t to my replace_text(t, linefeed, " ")
    set t to my replace_text(t, return, " ")
    return t
end normalize_text

set fieldSep to character id 31
set recordSep to character id 30
set outputRows to {{}}

set maxItems to {limit}
set epochZero to date "1/1/1970 00:00:00"

tell application "Calendar"
    {cal_target}
    repeat with cal in calList
        try
            set calName to my normalize_text(name of cal)
            set evs to events of cal
            repeat with ev in evs
                if (count of outputRows) >= maxItems then exit repeat
                try
                    set evId to ""
                    try
                        set evId to my normalize_text(uid of ev)
                    on error
                        try
                            set evId to my normalize_text(id of ev)
                        end try
                    end try
                    set evTitle to ""
                    try
                        set evTitle to my normalize_text(summary of ev)
                    end try
                    set evLocation to ""
                    try
                        set evLocation to my normalize_text(location of ev)
                    end try
                    set evDesc to ""
                    try
                        set evDesc to my normalize_text(description of ev)
                    end try
                    set evAllDay to false
                    set startEpoch to (|start date| of ev) - epochZero
                    set startEpoch to startEpoch as integer
                    set endEpoch to (|end date| of ev) - epochZero
                    set endEpoch to endEpoch as integer
                    set rowText to evId & fieldSep ¬
                        & evTitle & fieldSep ¬
                        & (startEpoch as text) & fieldSep ¬
                        & (endEpoch as text) & fieldSep ¬
                        & evLocation & fieldSep ¬
                        & evDesc & fieldSep ¬
                        & calName & fieldSep ¬
                        & (evAllDay as text)
                    set end of outputRows to rowText
                end try
            end repeat
        end try
        if (count of outputRows) >= maxItems then exit repeat
    end repeat
end tell

set AppleScript's text item delimiters to recordSep
set outputText to outputRows as text
set AppleScript's text item delimiters to ""
return outputText
""".strip()


def _reminders_list_script(limit: int) -> str:
    return f"""
on replace_text(theText, oldItem, newItem)
    set AppleScript's text item delimiters to oldItem
    set textItems to every text item of theText
    set AppleScript's text item delimiters to newItem
    set outText to textItems as text
    set AppleScript's text item delimiters to ""
    return outText
end replace_text

on normalize_text(v)
    if v is missing value then return ""
    set t to v as text
    set t to my replace_text(t, (character id 30), " ")
    set t to my replace_text(t, (character id 31), " ")
    set t to my replace_text(t, linefeed, " ")
    set t to my replace_text(t, return, " ")
    return t
end normalize_text

set fieldSep to character id 31
set recordSep to character id 30
set outputRows to {{}}
set maxItems to {limit}

set epochZero to date "1/1/1970 00:00:00"

tell application "Reminders"
    repeat with lst in lists
        set listName to ""
        try
            set listName to my normalize_text(name of lst)
        end try
        repeat with rem in reminders of lst
            if (count of outputRows) >= maxItems then exit repeat
            set remId to ""
            try
                set remId to my normalize_text(id of rem)
            end try
            set remTitle to ""
            try
                set remTitle to my normalize_text(name of rem)
            end try
            set remNotes to ""
            try
                set remNotes to my normalize_text(body of rem)
            end try
            set remDueIso to ""
            set remCompleted to false
            try
                set remCompleted to (completed of rem)
            end try
            set rowText to remId & fieldSep & remTitle & fieldSep & remDueIso & fieldSep & remNotes & fieldSep & listName & fieldSep & (remCompleted as text)
            set end of outputRows to rowText
        end repeat
        if (count of outputRows) >= maxItems then exit repeat
    end repeat
end tell

set AppleScript's text item delimiters to recordSep
set outputText to outputRows as text
set AppleScript's text item delimiters to ""
return outputText
""".strip()


def _calendar_create_script(
    title: str,
    start_epoch: int,
    end_epoch: int,
    location: str,
    notes: str,
) -> str:
    # macOS 26 (Tahoe) tightened Calendar's autosave: creating the event
    # with ``summary`` only and then assigning ``start date`` / ``end
    # date`` separately triggers ``-10025 "No end date has been set"``
    # because the autosave fires between the two property assignments.
    # We now create the event in one shot via a property record, which
    # works on Tahoe; the old ``-1700 errAECoercionFail`` issue on
    # earlier macOS is no longer reproducible.
    #
    # We also skip subscribed / read-only calendars (Apple's "Calendars
    # I follow") because ``item 1 of (get calendars)`` may otherwise
    # land on a Birthdays / Holidays / shared-team calendar where the
    # write fails with permissions errors.
    return f"""
set eventTitle to "{_escape_apple_text(title)}"
set eventLocation to "{_escape_apple_text(location)}"
set eventNotes to "{_escape_apple_text(notes)}"
set startEpoch to {start_epoch}
set endEpoch to {end_epoch}

set epochZero to date "1/1/1970 00:00:00"
set startDate to epochZero + startEpoch
set endDate to epochZero + endEpoch

if application "Calendar" is not running then
    tell application "Calendar" to launch
    repeat 30 times
        if application "Calendar" is running then exit repeat
        delay 0.1
    end repeat
end if

tell application "Calendar"
    set calList to (get calendars)
    if (count of calList) is 0 then error "No calendars available."
    set targetCalendar to missing value
    repeat with cal in calList
        try
            if (writable of cal) is true then
                set targetCalendar to cal
                exit repeat
            end if
        on error
            -- Older macOS versions don't expose ``writable``; fall
            -- back to the first non-subscribed calendar.
            try
                if (subscribed of cal) is false then
                    set targetCalendar to cal
                    exit repeat
                end if
            on error
                set targetCalendar to cal
                exit repeat
            end try
        end try
    end repeat
    if targetCalendar is missing value then set targetCalendar to item 1 of calList
    set newEvent to make new event at end of events of targetCalendar with properties {{summary:eventTitle, start date:startDate, end date:endDate}}
    if eventLocation is not "" then
        try
            set location of newEvent to eventLocation
        end try
    end if
    if eventNotes is not "" then
        try
            set description of newEvent to eventNotes
        end try
    end if
    try
        return (uid of newEvent) as text
    on error
        return (id of newEvent) as text
    end try
end tell
""".strip()


def _reminder_create_script(
    title: str,
    list_name: str,
    notes: str,
    due_epoch: int | None,
) -> str:
    due_clause = ""
    if due_epoch is not None and due_epoch > 0:
        due_clause = f"""
set epochZero to date "1/1/1970 00:00:00"
set dueDate to epochZero + {due_epoch}
try
    set due date of newReminder to dueDate
end try
"""
    return f"""
set reminderTitle to "{_escape_apple_text(title)}"
set listName to "{_escape_apple_text(list_name)}"
set reminderNotes to "{_escape_apple_text(notes)}"

tell application "Reminders"
    if not (exists list listName) then
        make new list with properties {{name:listName}}
    end if
    set targetList to list listName
    set newReminder to make new reminder at end of reminders of targetList with properties {{name:reminderTitle}}
    if reminderNotes is not "" then
        try
            set body of newReminder to reminderNotes
        end try
    end if
{due_clause}
    return (id of newReminder) as text
end tell
""".strip()


def _normalize_limit(arguments: dict[str, Any]) -> int:
    value = arguments.get("limit", 200)
    try:
        limit = int(value)
    except (TypeError, ValueError):
        limit = 200
    return max(1, min(limit, 500))


def _resolve_range(arguments: dict[str, Any]) -> tuple[int, int]:
    now = datetime.now().astimezone()
    default_from = int(now.timestamp())
    default_to = int((now + timedelta(days=30)).timestamp())
    from_epoch = _parse_iso_to_epoch(
        str(arguments.get("fromDate", "")).strip() or None,
    )
    to_epoch = _parse_iso_to_epoch(
        str(arguments.get("toDate", "")).strip() or None,
    )
    if from_epoch is None:
        from_epoch = default_from
    if to_epoch is None:
        to_epoch = default_to
    if to_epoch <= from_epoch:
        to_epoch = from_epoch + 3600
    return (from_epoch, to_epoch)


# macOS Calendar Participant.status code → human-readable RSVP string.
# Source: EventKit ``EKParticipantStatus`` enum.
_PARTICIPANT_STATUS_MAP: dict[int, str] = {
    0: "needs_action",
    1: "accepted",
    2: "declined",
    3: "tentative",
    4: "delegated",
}


def _classify_event_origin(
    *,
    is_shared_calendar: bool,
    is_subscribed_calendar: bool,
    is_self_invited: bool,
) -> str:
    """Pick one of personal/team_awareness/subscribed.

    Invitation supersedes calendar type: if the user is an invited
    participant the event is theirs no matter where it lives.

    sensitivity_tier: 1
    """
    if is_self_invited:
        return "personal"
    if not is_shared_calendar and not is_subscribed_calendar:
        return "personal"
    if is_subscribed_calendar:
        return "subscribed"
    return "team_awareness"


def _is_shared_to_me(
    *,
    sharing_status: int | None,
    shared_owner_address: str | None,
    self_identity_email: str | None,
) -> bool:
    """True when the calendar belongs to someone else but is shared with us.

    sensitivity_tier: 1
    """
    if sharing_status and int(sharing_status) > 0:
        return True
    if not shared_owner_address:
        return False
    if not self_identity_email:
        return True
    return self_identity_email.lower() not in shared_owner_address.lower()


def _read_calendar_events_sqlite(
    from_epoch: int,
    to_epoch: int,
    limit: int,
) -> list[dict[str, Any]]:
    """Read calendar events directly from the macOS Calendar SQLite DB.

    This is orders of magnitude faster than AppleScript for large
    calendars and reads from all calendars in a single query.

    sensitivity_tier: 2
    """
    if not _CALENDAR_DB_PATH.exists():
        return []

    from_cd = from_epoch - _CORE_DATA_EPOCH_OFFSET
    to_cd = to_epoch - _CORE_DATA_EPOCH_OFFSET

    conn = sqlite3.connect(
        f"file:{_CALENDAR_DB_PATH}?mode=ro",
        uri=True,
    )
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT
                ci.ROWID                    AS ci_rowid,
                ci.UUID                     AS event_id,
                ci.summary                  AS title,
                ci.start_date               AS start_cd,
                ci.end_date                 AS end_cd,
                ci.all_day                  AS is_all_day,
                ci.description              AS description,
                c.title                     AS calendar_name,
                c.shared_owner_address      AS shared_owner_address,
                c.owner_identity_email      AS owner_identity_email,
                c.self_identity_email       AS self_identity_email,
                c.sharing_status            AS sharing_status,
                c.subcal_url                AS subcal_url,
                c.subscription_id           AS subscription_id,
                l.title                     AS location
            FROM CalendarItem ci
            JOIN Calendar c ON ci.calendar_id = c.ROWID
            LEFT JOIN Location l
                ON ci.location_id = l.ROWID
            WHERE ci.start_date IS NOT NULL
              AND ci.start_date >= ?
              AND ci.start_date <= ?
            ORDER BY ci.start_date
            LIMIT ?
            """,
            (from_cd, to_cd, limit),
        ).fetchall()

        # Batch participant lookup keyed by CalendarItem.ROWID.
        # Avoids N+1: one query for every event on screen.
        participants_by_event: dict[int, list[dict[str, Any]]] = {}
        ci_rowids = [int(r["ci_rowid"]) for r in rows if r["ci_rowid"] is not None]
        if ci_rowids:
            placeholders = ",".join("?" for _ in ci_rowids)
            p_rows = conn.execute(
                f"""
                SELECT owner_id, email, is_self, status, role
                FROM Participant
                WHERE owner_id IN ({placeholders})
                """,  # noqa: S608 — placeholders, not user input
                ci_rowids,
            ).fetchall()
            for p in p_rows:
                owner_id = int(p["owner_id"])
                participants_by_event.setdefault(owner_id, []).append({
                    "email": (p["email"] or "").strip() or None,
                    "is_self": int(p["is_self"] or 0),
                    "status": int(p["status"] or 0),
                    "role": int(p["role"] or 0),
                })
    finally:
        conn.close()

    events: list[dict[str, Any]] = []
    for row in rows:
        start_unix = row["start_cd"] + _CORE_DATA_EPOCH_OFFSET
        end_cd = row["end_cd"]
        end_unix = (
            (end_cd + _CORE_DATA_EPOCH_OFFSET)
            if end_cd is not None
            else start_unix + 3600
        )

        start_iso = datetime.fromtimestamp(
            start_unix, tz=timezone.utc,
        ).isoformat().replace("+00:00", "Z")
        end_iso = datetime.fromtimestamp(
            end_unix, tz=timezone.utc,
        ).isoformat().replace("+00:00", "Z")

        event_id = (row["event_id"] or "").strip()
        title = (row["title"] or "").strip()
        description = (row["description"] or "").strip()
        location = (row["location"] or "").strip()

        if _calendar_placeholder(title, description, event_id):
            continue
        if not event_id:
            event_id = _stable_id(
                "evt",
                [title, start_iso, end_iso, location, description],
            )

        calendar_name = (row["calendar_name"] or "").strip() or None
        owner_email = (row["owner_identity_email"] or "").strip() or None
        self_email = (row["self_identity_email"] or "").strip() or None
        shared_owner = (row["shared_owner_address"] or "").strip() or None
        is_shared_calendar = _is_shared_to_me(
            sharing_status=row["sharing_status"],
            shared_owner_address=shared_owner,
            self_identity_email=self_email,
        )
        is_subscribed_calendar = bool(
            (row["subcal_url"] or row["subscription_id"]) or False,
        )

        participants = participants_by_event.get(int(row["ci_rowid"]), [])
        self_participant = next(
            (p for p in participants if p["is_self"] == 1),
            None,
        )
        is_self_invited = self_participant is not None
        self_response_status = (
            _PARTICIPANT_STATUS_MAP.get(self_participant["status"])
            if self_participant is not None
            else None
        )

        event_origin = _classify_event_origin(
            is_shared_calendar=is_shared_calendar,
            is_subscribed_calendar=is_subscribed_calendar,
            is_self_invited=is_self_invited,
        )

        attendees_payload = [
            {
                "email": p["email"],
                "is_self": p["is_self"],
                "status": _PARTICIPANT_STATUS_MAP.get(p["status"]),
            }
            for p in participants
        ]

        events.append({
            "id": event_id,
            "title": title or "Untitled Event",
            "start_time": start_iso,
            "end_time": end_iso,
            "location": location or None,
            "attendees": attendees_payload,
            "description": description or None,
            "is_all_day": bool(row["is_all_day"]),
            "calendar_name": calendar_name,
            "calendar_owner_email": owner_email,
            "is_shared_calendar": int(is_shared_calendar),
            "is_subscribed_calendar": int(is_subscribed_calendar),
            "self_response_status": self_response_status,
            "event_origin": event_origin,
        })
    return events


def list_calendar_events(
    arguments: dict[str, Any],
) -> list[dict[str, Any]]:
    """Fetch events from all macOS calendars.

    Uses direct SQLite reads for speed; falls back to AppleScript
    if the database file is not accessible.

    sensitivity_tier: 2
    """
    limit = _normalize_limit(arguments)
    from_epoch, to_epoch = _resolve_range(arguments)

    # Fast path: direct SQLite read.
    if _CALENDAR_DB_PATH.exists():
        try:
            return _read_calendar_events_sqlite(
                from_epoch, to_epoch, limit,
            )
        except (sqlite3.Error, OSError):
            pass  # Fall through to AppleScript fallback.

    # Fallback: AppleScript (single calendar, slow).
    try:
        output = _run_osascript(
            _calendar_list_script(from_epoch, to_epoch, limit),
        )
    except RuntimeError:
        return []
    parsed = _parse_delimited_rows(
        output,
        [
            "id", "title", "start_iso", "end_iso",
            "location", "description",
            "calendar_name", "is_all_day",
        ],
    )
    rows: list[dict[str, Any]] = []
    for row in parsed:
        event_id = row["id"].strip()
        title = row["title"].strip()
        try:
            start_val = float(row["start_iso"].strip())
        except ValueError:
            continue
        if start_val < from_epoch or start_val > to_epoch:
            continue
        start_iso = _epoch_text_to_iso(row["start_iso"])
        end_iso = _epoch_text_to_iso(row["end_iso"])
        location = row["location"].strip()
        description = row["description"].strip()
        if _calendar_placeholder(title, description, event_id):
            continue
        if not start_iso or not end_iso:
            continue
        if not event_id:
            event_id = _stable_id(
                "evt",
                [title, start_iso, end_iso, location, description],
            )
        rows.append({
            "id": event_id,
            "title": title or "Untitled Event",
            "start_time": start_iso,
            "end_time": end_iso,
            "location": location or None,
            "attendees": [],
            "description": description or None,
            "is_all_day": _parse_bool(row["is_all_day"]),
            # AppleScript fallback can't read participants or calendar
            # ownership flags; downstream pipeline treats NULL origin as
            # 'personal' so this stays safe.
            "calendar_name": row["calendar_name"].strip() or None,
            "calendar_owner_email": None,
            "is_shared_calendar": 0,
            "is_subscribed_calendar": 0,
            "self_response_status": None,
            "event_origin": "personal",
        })
        if len(rows) >= limit:
            break
    return rows


def _find_reminders_db() -> Path | None:
    """Find the Reminders SQLite file that contains data.

    macOS stores reminders across multiple SQLite files; we
    pick the one with the most rows.

    sensitivity_tier: 1
    """
    if not _REMINDERS_DB_DIR.is_dir():
        return None
    best: tuple[int, Path | None] = (0, None)
    for db_file in _REMINDERS_DB_DIR.glob("*.sqlite"):
        try:
            conn = sqlite3.connect(
                f"file:{db_file}?mode=ro", uri=True,
            )
            count = conn.execute(
                "SELECT COUNT(*) FROM ZREMCDREMINDER"
                " WHERE ZMARKEDFORDELETION = 0",
            ).fetchone()[0]
            conn.close()
            if count > best[0]:
                best = (count, db_file)
        except (sqlite3.Error, OSError):
            continue
    return best[1]


def _read_reminders_sqlite(
    limit: int,
) -> list[dict[str, Any]]:
    """Read reminders directly from the macOS Reminders SQLite DB.

    sensitivity_tier: 2
    """
    db_path = _find_reminders_db()
    if db_path is None:
        return []

    conn = sqlite3.connect(
        f"file:{db_path}?mode=ro", uri=True,
    )
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT
                r.ZEXTERNALIDENTIFIER AS ext_id,
                r.ZTITLE              AS title,
                r.ZDUEDATE            AS due_cd,
                r.ZNOTES              AS notes,
                r.ZCOMPLETED          AS completed,
                l.ZNAME               AS list_name
            FROM ZREMCDREMINDER r
            LEFT JOIN ZREMCDBASELIST l
                ON r.ZLIST = l.Z_PK
            WHERE r.ZMARKEDFORDELETION = 0
            ORDER BY r.ZCOMPLETED ASC,
                     r.ZDUEDATE ASC NULLS LAST
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    finally:
        conn.close()

    reminders: list[dict[str, Any]] = []
    for row in rows:
        title = (row["title"] or "").strip()
        if not title:
            continue
        reminder_id = (row["ext_id"] or "").strip()
        notes = (row["notes"] or "").strip()
        list_name = (row["list_name"] or "").strip()
        due_iso = ""
        if row["due_cd"] is not None:
            due_unix = row["due_cd"] + _CORE_DATA_EPOCH_OFFSET
            due_iso = datetime.fromtimestamp(
                due_unix, tz=timezone.utc,
            ).isoformat().replace("+00:00", "Z")

        if not reminder_id:
            reminder_id = _stable_id(
                "rem",
                [title, due_iso, notes, list_name],
            )
        reminders.append({
            "id": reminder_id,
            "title": title,
            "due_date": due_iso or None,
            "notes": notes or None,
            "completed": bool(row["completed"]),
            "list_name": list_name or None,
        })
    return reminders


def list_reminders(arguments: dict[str, Any]) -> list[dict[str, Any]]:
    """Fetch reminders from macOS Reminders.

    Uses direct SQLite reads for speed; falls back to AppleScript
    if the database is not accessible.

    sensitivity_tier: 2
    """
    limit = _normalize_limit(arguments)

    # Fast path: direct SQLite read.
    if _REMINDERS_DB_DIR.is_dir():
        try:
            result = _read_reminders_sqlite(limit)
            if result:
                return result
        except (sqlite3.Error, OSError):
            pass  # Fall through to AppleScript.

    # Fallback: AppleScript.
    try:
        output = _run_osascript(_reminders_list_script(limit))
    except RuntimeError:
        return []
    parsed = _parse_delimited_rows(
        output,
        ["id", "title", "due_iso", "notes", "list_name", "completed"],
    )
    rows: list[dict[str, Any]] = []
    for row in parsed:
        reminder_id = row["id"].strip()
        title = row["title"].strip()
        due_iso = _epoch_text_to_iso(row["due_iso"])
        notes = row["notes"].strip()
        list_name = row["list_name"].strip()
        if _reminder_placeholder(
            title, reminder_id, notes, list_name, due_iso,
        ):
            continue
        if not title:
            continue
        if not reminder_id:
            reminder_id = _stable_id(
                "rem",
                [title, due_iso, notes, list_name],
            )
        rows.append({
            "id": reminder_id,
            "title": title,
            "due_date": due_iso or None,
            "notes": notes or None,
            "completed": _parse_bool(row["completed"]),
            "list_name": list_name or None,
        })
        if len(rows) >= limit:
            break
    return rows


def _read_contacts_sqlite(
    limit: int,
    query: str | None = None,
) -> list[dict[str, Any]]:
    """Read contacts from macOS AddressBook SQLite databases.

    Iterates over all Sources databases to capture contacts from
    every account (iCloud, Gmail, Exchange, etc.).

    sensitivity_tier: 2
    """
    if not _CONTACTS_DB_DIR.is_dir():
        return []

    seen_ids: set[str] = set()
    contacts: list[dict[str, Any]] = []

    for db_file in sorted(_CONTACTS_DB_DIR.glob("*/AddressBook-v22.abcddb")):
        if len(contacts) >= limit:
            break
        try:
            conn = sqlite3.connect(
                f"file:{db_file}?mode=ro", uri=True,
            )
            conn.row_factory = sqlite3.Row
            _read_contacts_from_db(
                conn, contacts, seen_ids, limit, query,
            )
            conn.close()
        except (sqlite3.Error, OSError):
            continue

    return contacts[:limit]


def _read_contacts_from_db(
    conn: sqlite3.Connection,
    contacts: list[dict[str, Any]],
    seen_ids: set[str],
    limit: int,
    query: str | None,
) -> None:
    """Read contacts from a single AddressBook SQLite database.

    sensitivity_tier: 2
    """
    where_clauses = ["r.Z_ENT = ?"]
    params: list[Any] = [_ABCD_CONTACT_ENT]

    if query:
        q = f"%{query}%"
        where_clauses.append(
            "(r.ZFIRSTNAME LIKE ? OR r.ZLASTNAME LIKE ? "
            "OR r.ZORGANIZATION LIKE ? OR r.ZNICKNAME LIKE ?)"
        )
        params.extend([q, q, q, q])

    where_sql = " AND ".join(where_clauses)
    rows = conn.execute(
        f"""
        SELECT
            r.Z_PK          AS pk,
            r.ZUNIQUEID      AS unique_id,
            r.ZFIRSTNAME     AS first_name,
            r.ZLASTNAME      AS last_name,
            r.ZORGANIZATION  AS organization,
            r.ZBIRTHDAY      AS birthday_cd,
            r.ZNICKNAME      AS nickname,
            r.ZJOBTITLE      AS job_title,
            r.ZDEPARTMENT    AS department
        FROM ZABCDRECORD r
        WHERE {where_sql}
        ORDER BY r.ZSORTINGFIRSTNAME, r.ZSORTINGLASTNAME
        LIMIT ?
        """,
        (*params, limit),
    ).fetchall()

    for row in rows:
        unique_id = (row["unique_id"] or "").strip()
        if not unique_id:
            unique_id = _stable_id(
                "contact",
                [
                    row["first_name"] or "",
                    row["last_name"] or "",
                    row["organization"] or "",
                ],
            )
        if unique_id in seen_ids:
            continue
        seen_ids.add(unique_id)

        first = (row["first_name"] or "").strip()
        last = (row["last_name"] or "").strip()
        name = f"{first} {last}".strip() if (first or last) else None
        organization = (row["organization"] or "").strip() or None
        if not name and not organization:
            continue

        pk = row["pk"]

        # Email (first/primary)
        email = _first_value(
            conn,
            "SELECT ZADDRESS FROM ZABCDEMAILADDRESS"
            " WHERE ZOWNER = ? ORDER BY ZISPRIMARY DESC, ZORDERINGINDEX"
            " LIMIT 1",
            pk,
        )

        # Phone (first/primary)
        phone = _first_value(
            conn,
            "SELECT ZFULLNUMBER FROM ZABCDPHONENUMBER"
            " WHERE ZOWNER = ? ORDER BY ZISPRIMARY DESC, ZORDERINGINDEX"
            " LIMIT 1",
            pk,
        )

        # Postal address (formatted)
        address = _first_address(conn, pk)

        # Notes
        notes = _first_value(
            conn,
            "SELECT ZTEXT FROM ZABCDNOTE WHERE ZCONTACT = ? LIMIT 1",
            pk,
        )

        # Relationship (related name)
        relationship = _first_value(
            conn,
            "SELECT ZNAME FROM ZABCDRELATEDNAME"
            " WHERE ZOWNER = ? ORDER BY ZORDERINGINDEX LIMIT 1",
            pk,
        )

        # Birthday
        birthday = None
        if row["birthday_cd"] is not None:
            try:
                bday_unix = row["birthday_cd"] + _CORE_DATA_EPOCH_OFFSET
                birthday = datetime.fromtimestamp(
                    bday_unix, tz=timezone.utc,
                ).strftime("%Y-%m-%d")
            except (TypeError, ValueError, OSError):
                pass

        display_name = name or organization or "Unknown"

        contacts.append({
            "id": unique_id,
            "name": display_name,
            "email": email,
            "phone": phone,
            "relationship": relationship,
            "birthday": birthday,
            "address": address,
            "notes": notes,
        })

        if len(contacts) >= limit:
            break


def _first_value(
    conn: sqlite3.Connection,
    sql: str,
    owner_pk: int,
) -> str | None:
    """Fetch first non-empty string from a single-column query.

    sensitivity_tier: 3
    """
    try:
        row = conn.execute(sql, (owner_pk,)).fetchone()
        if row and row[0]:
            return str(row[0]).strip() or None
    except sqlite3.Error:
        pass
    return None


def _first_address(
    conn: sqlite3.Connection,
    owner_pk: int,
) -> str | None:
    """Build a formatted address string from the first postal address.

    sensitivity_tier: 3
    """
    try:
        row = conn.execute(
            "SELECT ZSTREET, ZCITY, ZSTATE, ZZIPCODE, ZCOUNTRYNAME"
            " FROM ZABCDPOSTALADDRESS"
            " WHERE ZOWNER = ? ORDER BY ZISPRIMARY DESC, ZORDERINGINDEX"
            " LIMIT 1",
            (owner_pk,),
        ).fetchone()
        if not row:
            return None
        parts = [
            (row[0] or "").strip(),  # street
            (row[1] or "").strip(),  # city
            (row[2] or "").strip(),  # state
            (row[3] or "").strip(),  # zip
            (row[4] or "").strip(),  # country
        ]
        formatted = ", ".join(p for p in parts if p)
        return formatted or None
    except sqlite3.Error:
        return None


def list_contacts(arguments: dict[str, Any]) -> list[dict[str, Any]]:
    """Fetch contacts from macOS AddressBook via SQLite.

    sensitivity_tier: 2
    """
    raw_limit = arguments.get("limit", 500)
    try:
        limit = int(raw_limit)
    except (TypeError, ValueError):
        limit = 500
    limit = max(1, min(limit, 2000))

    query = (str(arguments.get("query", "")).strip() or None)

    return _read_contacts_sqlite(limit, query)


def search_contacts(arguments: dict[str, Any]) -> list[dict[str, Any]]:
    """Search contacts by name, email, or phone.

    sensitivity_tier: 2
    """
    query = str(arguments.get("query", "")).strip()
    if not query:
        return []

    raw_limit = arguments.get("limit", 20)
    try:
        limit = int(raw_limit)
    except (TypeError, ValueError):
        limit = 20
    limit = max(1, min(limit, 100))

    # SQLite-level search covers name/org/nickname.
    # For email/phone matches we do a broader search and
    # post-filter.
    results = _read_contacts_sqlite(limit * 3, query)

    # Also match on email/phone fields
    q_lower = query.lower()
    matched = [
        c for c in results
        if q_lower in (c.get("name") or "").lower()
        or q_lower in (c.get("email") or "").lower()
        or q_lower in (c.get("phone") or "").lower()
        or q_lower in (c.get("relationship") or "").lower()
    ]
    return matched[:limit]


# ---------------------------------------------------------------------------
# Notes (NoteStore.sqlite)
# ---------------------------------------------------------------------------


def _cd_to_iso(cd_timestamp: float | None) -> str:
    """Convert a Core Data timestamp to ISO-8601 string.

    sensitivity_tier: 1
    """
    if cd_timestamp is None:
        return ""
    try:
        unix_ts = cd_timestamp + _CORE_DATA_EPOCH_OFFSET
        return (
            datetime.fromtimestamp(unix_ts, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
    except (TypeError, ValueError, OSError):
        return ""


def _extract_note_text(zdata: bytes | None) -> str:
    """Extract plaintext from a Notes ZDATA blob (gzip + protobuf).

    The blob is gzip-compressed protobuf.  We decompress and parse the
    protobuf wire format to extract only UTF-8 string fields, avoiding
    garbled output from naively decoding the whole blob.

    sensitivity_tier: 2
    """
    if not zdata:
        return ""
    import gzip as _gzip

    try:
        raw = _gzip.decompress(zdata)
    except Exception:  # noqa: BLE001
        return ""

    strings = _pb_extract_strings(raw, depth=0)
    if not strings:
        return ""
    # The note body is the longest extracted string.
    return max(strings, key=len)


def _pb_read_varint(data: bytes, pos: int) -> tuple[int, int]:
    """Read a protobuf varint, return ``(value, new_position)``.

    sensitivity_tier: 1
    """
    result = 0
    shift = 0
    while pos < len(data):
        b = data[pos]
        result |= (b & 0x7F) << shift
        pos += 1
        if not (b & 0x80):
            return result, pos
        shift += 7
        if shift > 63:
            break
    return result, pos


def _pb_extract_strings(data: bytes, depth: int) -> list[str]:
    """Extract UTF-8 string fields from protobuf wire format.

    Recurses into embedded messages up to *depth* 5 to find text
    stored inside nested protobuf structures (as Apple Notes does).

    sensitivity_tier: 1
    """
    if depth > 5 or len(data) < 2:
        return []
    strings: list[str] = []
    pos = 0
    while pos < len(data):
        start = pos
        tag, pos = _pb_read_varint(data, pos)
        if pos <= start:
            break
        wire_type = tag & 0x07

        if wire_type == 0:  # varint
            _, pos = _pb_read_varint(data, pos)
        elif wire_type == 1:  # 64-bit fixed
            pos += 8
        elif wire_type == 2:  # length-delimited (string / bytes / msg)
            length, pos = _pb_read_varint(data, pos)
            if length < 0 or length > len(data) - pos:
                break
            chunk = data[pos : pos + length]
            pos += length
            # Try as UTF-8 string first
            try:
                text = chunk.decode("utf-8")
                total = len(text)
                if total >= 2:
                    ok = sum(
                        1
                        for c in text
                        if c.isprintable() or c in "\n\t\r"
                    )
                    if ok / total > 0.8:
                        stripped = text.strip()
                        if stripped:
                            strings.append(stripped)
                        continue
            except UnicodeDecodeError:
                pass
            # Not a clean string — try as embedded message
            strings.extend(_pb_extract_strings(chunk, depth + 1))
        elif wire_type == 5:  # 32-bit fixed
            pos += 4
        else:
            break
    return strings


def _sqlite_table_exists(
    conn: sqlite3.Connection,
    table_name: str,
) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table' AND name = ?
        LIMIT 1
        """,
        (table_name,),
    ).fetchone()
    return row is not None


def _sqlite_table_columns(
    conn: sqlite3.Connection,
    table_name: str,
) -> set[str]:
    rows = conn.execute(
        f'PRAGMA table_info("{table_name}")',
    ).fetchall()
    return {str(row["name"]) for row in rows if row["name"]}


def _pick_first_column(
    available: set[str],
    candidates: list[str],
) -> str | None:
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def _read_notes_sqlite(limit: int) -> list[dict[str, Any]]:
    """Read notes from macOS Notes NoteStore.sqlite directly.

    Uses a schema-tolerant query because Notes columns vary across
    macOS releases.

    sensitivity_tier: 2
    """
    if not _NOTES_DB_PATH.exists():
        return []

    def _query(conn: sqlite3.Connection) -> list[sqlite3.Row]:
        if not _sqlite_table_exists(conn, "ZICCLOUDSYNCINGOBJECT"):
            raise RuntimeError("Unsupported Notes schema (missing sync table)")
        if not _sqlite_table_exists(conn, "ZICNOTEDATA"):
            raise RuntimeError("Unsupported Notes schema (missing note data)")

        note_cols = _sqlite_table_columns(conn, "ZICCLOUDSYNCINGOBJECT")
        data_cols = _sqlite_table_columns(conn, "ZICNOTEDATA")
        if "Z_PK" not in note_cols:
            raise RuntimeError("Unsupported Notes schema (missing Z_PK)")
        if "ZNOTE" not in data_cols or "ZDATA" not in data_cols:
            raise RuntimeError(
                "Unsupported Notes schema (missing ZNOTE/ZDATA)",
            )

        id_col = _pick_first_column(
            note_cols,
            ["ZIDENTIFIER", "ZUNIQUEIDENTIFIER"],
        )
        title_col = _pick_first_column(
            note_cols,
            ["ZTITLE1", "ZTITLE2", "ZTITLE"],
        )
        snippet_col = _pick_first_column(
            note_cols,
            ["ZSNIPPET", "ZSUMMARY", "ZSUBTITLE"],
        )
        created_col = _pick_first_column(
            note_cols,
            [
                "ZCREATIONDATE3",
                "ZCREATIONDATE2",
                "ZCREATIONDATE1",
                "ZCREATIONDATE",
            ],
        )
        updated_col = _pick_first_column(
            note_cols,
            [
                "ZMODIFICATIONDATE3",
                "ZMODIFICATIONDATE2",
                "ZMODIFICATIONDATE1",
                "ZMODIFICATIONDATE",
            ],
        )
        folder_fk_col = _pick_first_column(
            note_cols,
            ["ZFOLDER", "ZPARENT"],
        )
        folder_title_col = _pick_first_column(
            note_cols,
            ["ZTITLE1", "ZTITLE2", "ZTITLE"],
        )

        select_note_id = f"n.{id_col}" if id_col else "''"
        select_title = f"n.{title_col}" if title_col else "''"
        select_snippet = f"n.{snippet_col}" if snippet_col else "''"
        select_created = f"n.{created_col}" if created_col else "NULL"
        if updated_col:
            select_updated = f"n.{updated_col}"
        elif created_col:
            select_updated = f"n.{created_col}"
        else:
            select_updated = "NULL"

        join_folder_sql = ""
        select_folder = "''"
        if folder_fk_col and folder_title_col:
            join_folder_sql = (
                "LEFT JOIN ZICCLOUDSYNCINGOBJECT f "
                f"ON n.{folder_fk_col} = f.Z_PK"
            )
            select_folder = f"f.{folder_title_col}"

        where_clauses: list[str] = []
        if "ZMARKEDFORDELETION" in note_cols:
            where_clauses.append("COALESCE(n.ZMARKEDFORDELETION, 0) = 0")
        if "ZTRASHED" in note_cols:
            where_clauses.append("COALESCE(n.ZTRASHED, 0) = 0")
        if "ZISPASSWORDPROTECTED" in note_cols:
            where_clauses.append("COALESCE(n.ZISPASSWORDPROTECTED, 0) = 0")
        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        if updated_col:
            order_sql = f"ORDER BY n.{updated_col} DESC, n.Z_PK DESC"
        elif created_col:
            order_sql = f"ORDER BY n.{created_col} DESC, n.Z_PK DESC"
        else:
            order_sql = "ORDER BY n.Z_PK DESC"

        query = f"""
            SELECT
                n.Z_PK AS pk,
                {select_note_id} AS note_id,
                {select_title} AS title,
                {select_snippet} AS snippet,
                {select_created} AS created_cd,
                {select_updated} AS updated_cd,
                {select_folder} AS folder_name,
                d.ZDATA AS zdata
            FROM ZICNOTEDATA d
            INNER JOIN ZICCLOUDSYNCINGOBJECT n
                ON d.ZNOTE = n.Z_PK
            {join_folder_sql}
            {where_sql}
            {order_sql}
            LIMIT ?
        """
        return conn.execute(query, (limit,)).fetchall()

    rows = _query_macos_db(_NOTES_DB_PATH, _query)

    notes: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in rows:
        snippet = str(row["snippet"] or "").strip()
        body = _extract_note_text(row["zdata"]).strip() or snippet
        title = str(row["title"] or "").strip()
        if not title:
            title = snippet[:120] or body[:120] or "Untitled Note"

        created_at = _cd_to_iso(row["created_cd"])
        updated_at = _cd_to_iso(row["updated_cd"])
        folder = str(row["folder_name"] or "").strip()
        note_id = str(row["note_id"] or "").strip()
        if not note_id:
            note_id = _stable_id(
                "note",
                [title, snippet, body, created_at, updated_at, folder],
            )
        if note_id in seen_ids:
            continue
        seen_ids.add(note_id)

        now_iso = datetime.now(tz=timezone.utc).isoformat()
        notes.append({
            "id": note_id,
            "title": title,
            "content": body or "",
            "source": "apple_notes",
            "folder": folder or None,
            "created_at": created_at or now_iso,
            "updated_at": updated_at or now_iso,
            "tags": [],
        })
        if len(notes) >= limit:
            break
    return notes


def _read_notes_jxa(limit: int) -> list[dict[str, Any]]:
    """Read notes via JXA (JavaScript for Automation).

    Uses osascript -l JavaScript to call the Notes app scripting bridge.
    This triggers the proper macOS permission dialog on first run.

    sensitivity_tier: 2
    """
    jxa_script = f"""
'use strict';
const Notes = Application("Notes");
const limit = {limit};
const all = Notes.notes();
const count = Math.min(all.length, limit);
const result = [];
for (let i = 0; i < count; i++) {{
    const n = all[i];
    try {{
        result.push({{
            id: n.id() || "",
            title: n.name() || "",
            content: n.plaintext() || "",
            folder: n.container().name() || "",
            created_at: n.creationDate().toISOString(),
            updated_at: n.modificationDate().toISOString(),
        }});
    }} catch(e) {{}}
}}
JSON.stringify(result);
"""
    try:
        proc = subprocess.run(
            ["osascript", "-l", "JavaScript", "-e", jxa_script],
            capture_output=True,
            text=True,
            timeout=OSASCRIPT_TIMEOUT_SECONDS * 2,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "JXA script failed")
        raw = proc.stdout.strip()
        if not raw:
            return []
        notes_raw: list[dict] = json.loads(raw)
    except (json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Notes JXA failed: {exc}") from exc

    notes: list[dict[str, Any]] = []
    for entry in notes_raw:
        note_id = str(entry.get("id", "")).strip()
        if not note_id:
            continue
        title = str(entry.get("title", "")).strip() or "Untitled Note"
        now_iso = datetime.now(tz=timezone.utc).isoformat()
        notes.append({
            "id": note_id,
            "title": title,
            "content": entry.get("content") or "",
            "source": "apple_notes",
            "folder": entry.get("folder") or None,
            "created_at": entry.get("created_at") or now_iso,
            "updated_at": entry.get("updated_at") or now_iso,
            "tags": [],
        })
    return notes


def list_notes(arguments: dict[str, Any]) -> list[dict[str, Any]]:
    """Fetch notes from macOS Notes.

    Uses direct SQLite reads (Contacts-style fast path for large
    datasets). This avoids JXA timeouts on large Notes libraries.

    sensitivity_tier: 2
    """
    raw_limit = arguments.get("limit", 500)
    try:
        limit = int(raw_limit)
    except (TypeError, ValueError):
        limit = 500
    limit = max(1, min(limit, 1000))

    try:
        return _read_notes_sqlite(limit)
    except PermissionError as exc:
        raise RuntimeError(
            "Access to Notes database was denied. Grant Full Disk Access "
            "to Arandu and python3, then retry.",
        ) from exc
    except (sqlite3.Error, OSError, RuntimeError) as exc:
        detail = str(exc).lower()
        if (
            "permission denied" in detail
            or "operation not permitted" in detail
            or "unable to open database file" in detail
        ):
            raise RuntimeError(
                "Access to Notes database was denied. Grant Full Disk "
                "Access to Arandu and python3, then retry.",
            ) from exc
        raise RuntimeError(
            f"Failed to read Notes database directly: {exc}",
        ) from exc


def _list_notes_jxa(arguments: dict[str, Any]) -> list[dict[str, Any]]:
    """Legacy JXA reader kept for manual debugging.

    sensitivity_tier: 2
    """
    raw_limit = arguments.get("limit", 500)
    try:
        limit = int(raw_limit)
    except (TypeError, ValueError):
        limit = 500
    limit = max(1, min(limit, 1000))
    return _read_notes_jxa(limit)


def create_note(arguments: dict[str, Any]) -> dict[str, Any]:
    """Create a note in macOS Notes via AppleScript.

    sensitivity_tier: 2
    """
    title = str(arguments.get("title", "")).strip()
    if not title:
        raise ValueError("create_note requires a non-empty title")
    body = str(arguments.get("body") or arguments.get("content") or "").strip()
    folder = str(arguments.get("folder", "Notes")).strip() or "Notes"

    # Apple Notes derives `name` from the first line of `body`.
    # Setting name then body would override the title.  Instead,
    # prepend the title as the first line of the body content.
    if body:
        full_body = f"{title}\n{body}"
    else:
        full_body = title

    script = f"""
tell application "Notes"
    set targetFolder to missing value
    repeat with f in folders
        if name of f is "{_escape_apple_text(folder)}" then
            set targetFolder to f
            exit repeat
        end if
    end repeat
    if targetFolder is missing value then
        set targetFolder to default account's folder "Notes"
    end if
    set newNote to make new note at targetFolder
    set body of newNote to "{_escape_apple_text(full_body)}"
    return id of newNote as text
end tell
""".strip()

    created_id = _run_osascript(script).strip()
    return {"success": True, "id": created_id or None, "title": title}


def search_notes(arguments: dict[str, Any]) -> list[dict[str, Any]]:
    """Search macOS Notes by title or content.

    Reads notes from NoteStore.sqlite and filters in Python.

    sensitivity_tier: 2
    """
    query = str(arguments.get("query", "")).strip()
    if not query:
        return []

    raw_limit = arguments.get("limit", 20)
    try:
        limit = int(raw_limit)
    except (TypeError, ValueError):
        limit = 20
    limit = max(1, min(limit, 100))

    all_notes = _read_notes_sqlite(500)
    q_lower = query.lower()

    matched = [
        note
        for note in all_notes
        if q_lower in (note.get("title") or "").lower()
        or q_lower in (note.get("content") or "").lower()
    ]
    return matched[:limit]


def update_note(arguments: dict[str, Any]) -> dict[str, Any]:
    """Update an existing note in macOS Notes via AppleScript.

    Finds the note by title and replaces its body content.

    sensitivity_tier: 2
    """
    title = str(arguments.get("title", "")).strip()
    if not title:
        raise ValueError("update_note requires a 'title' to find the note")

    body = str(
        arguments.get("body") or arguments.get("content") or "",
    ).strip()
    if not body:
        raise ValueError("update_note requires a 'body' with the new content")

    script = f"""
tell application "Notes"
    set matchedNotes to (every note whose name contains \
"{_escape_apple_text(title)}")
    if (count of matchedNotes) is 0 then
        error "No note found with title containing: \
{_escape_apple_text(title)}"
    end if
    set targetNote to item 1 of matchedNotes
    set body of targetNote to "{_escape_apple_text(body)}"
    return name of targetNote as text
end tell
""".strip()

    updated_title = _run_osascript(script).strip()
    return {"success": True, "title": updated_title or title}


def delete_note(arguments: dict[str, Any]) -> dict[str, Any]:
    """Delete a note from macOS Notes via AppleScript.

    Finds the note by title and moves it to the trash.

    sensitivity_tier: 2
    """
    title = str(arguments.get("title", "")).strip()
    if not title:
        raise ValueError("delete_note requires a 'title' to find the note")

    script = f"""
tell application "Notes"
    set matchedNotes to (every note whose name contains \
"{_escape_apple_text(title)}")
    if (count of matchedNotes) is 0 then
        error "No note found with title containing: \
{_escape_apple_text(title)}"
    end if
    delete item 1 of matchedNotes
end tell
return "deleted"
""".strip()

    _run_osascript(script)
    return {"success": True, "title": title}


# ---------------------------------------------------------------------------
# Mail (Envelope Index)
# ---------------------------------------------------------------------------


def _read_emails_sqlite(limit: int) -> list[dict[str, Any]]:
    """Read emails from macOS Mail Envelope Index SQLite database.

    Reads directly from the Mail.app database, joining the messages,
    subjects, addresses, summaries, and mailboxes tables.  This is
    far more reliable than JXA (which returns 0 for un-viewed mboxes).

    sensitivity_tier: 3
    """
    if not _MAIL_DB_PATH.exists():
        return []

    def _extract_folder(url: str | None) -> str:
        """Extract a human-readable folder name from a mailbox URL."""
        if not url:
            return "Unknown"
        from urllib.parse import unquote
        parts = unquote(url).split("/")
        # Last non-empty segment is the folder name
        for part in reversed(parts):
            if part:
                return part
        return "Unknown"

    def _query(conn: sqlite3.Connection) -> list:
        return conn.execute(
            """
            SELECT
                m.ROWID          AS rowid,
                m.message_id     AS message_id,
                s.subject        AS subject,
                a.address        AS sender_addr,
                a.comment        AS sender_name,
                m.date_received  AS date_received,
                m.read           AS is_read,
                mb.url           AS mailbox_url,
                sm.summary       AS body_preview
            FROM messages m
            LEFT JOIN subjects s ON m.subject = s.ROWID
            LEFT JOIN addresses a ON m.sender = a.ROWID
            LEFT JOIN mailboxes mb ON m.mailbox = mb.ROWID
            LEFT JOIN summaries sm ON m.summary = sm.ROWID
            WHERE m.deleted = 0
            ORDER BY m.date_received DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    rows = _query_macos_db(_MAIL_DB_PATH, _query)

    emails: list[dict[str, Any]] = []
    for row in rows:
        subject = str(row["subject"] or "").strip()
        if not subject:
            continue

        msg_id = str(row["message_id"] or row["rowid"])
        sender_addr = str(row["sender_addr"] or "").strip() or None
        sender_name = str(row["sender_name"] or "").strip()
        from_address = (
            f"{sender_name} <{sender_addr}>"
            if sender_name and sender_addr
            else sender_addr
        )

        date_received = row["date_received"]
        date_iso = None
        if date_received is not None:
            try:
                date_iso = (
                    datetime.fromtimestamp(date_received, tz=timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z")
                )
            except (TypeError, ValueError, OSError):
                pass

        folder = _extract_folder(row["mailbox_url"])
        body_preview = str(row["body_preview"] or "").strip() or None

        emails.append({
            "id": msg_id,
            "subject": subject,
            "source": "apple_mail",
            "from_address": from_address,
            "to_addresses": [],
            "date": date_iso,
            "body_preview": body_preview,
            "is_read": bool(row["is_read"]),
            "folder": folder,
        })

    return emails


def list_emails(arguments: dict[str, Any]) -> list[dict[str, Any]]:
    """Fetch emails from macOS Mail via Envelope Index SQLite.

    sensitivity_tier: 3
    """
    raw_limit = arguments.get("limit", 500)
    try:
        limit = int(raw_limit)
    except (TypeError, ValueError):
        limit = 500
    limit = max(1, min(limit, 2000))
    return _read_emails_sqlite(limit)


def search_emails(arguments: dict[str, Any]) -> list[dict[str, Any]]:
    """Search macOS Mail by subject, sender, or body preview.

    Reads from the Envelope Index SQLite and filters in Python.

    sensitivity_tier: 3
    """
    query = str(arguments.get("query", "")).strip()
    if not query:
        return []

    raw_limit = arguments.get("limit", 20)
    try:
        limit = int(raw_limit)
    except (TypeError, ValueError):
        limit = 20
    limit = max(1, min(limit, 100))

    all_emails = _read_emails_sqlite(1000)
    q_lower = query.lower()

    matched = [
        email
        for email in all_emails
        if q_lower in (email.get("subject") or "").lower()
        or q_lower in (email.get("from_address") or "").lower()
        or q_lower in (email.get("body_preview") or "").lower()
    ]
    return matched[:limit]


def send_email(arguments: dict[str, Any]) -> dict[str, Any]:
    """Compose and send an email via macOS Mail.app.

    sensitivity_tier: 3
    """
    to = str(arguments.get("to", "")).strip()
    subject = str(arguments.get("subject", "")).strip()
    body = str(
        arguments.get("body") or arguments.get("content") or "",
    ).strip()

    if not to:
        raise ValueError("send_email requires a 'to' email address")
    if not subject:
        raise ValueError("send_email requires a 'subject'")

    # Build recipient lines for AppleScript
    to_lines = ""
    for addr in to.split(","):
        addr = addr.strip()
        if addr:
            to_lines += (
                "\n        make new to recipient at end of to recipients"
                f' with properties {{address:"{_escape_apple_text(addr)}"}}'
            )

    cc = str(arguments.get("cc", "")).strip()
    cc_lines = ""
    if cc:
        for addr in cc.split(","):
            addr = addr.strip()
            if addr:
                cc_lines += (
                    "\n        make new cc recipient at end of cc recipients"
                    f' with properties {{address:'
                    f'"{_escape_apple_text(addr)}"}}'
                )

    script = f"""
tell application "Mail"
    set newMessage to make new outgoing message with properties \
{{subject:"{_escape_apple_text(subject)}", \
content:"{_escape_apple_text(body)}", visible:false}}
    tell newMessage{to_lines}{cc_lines}
    end tell
    send newMessage
end tell
return "sent"
""".strip()

    _run_osascript(script)
    return {"success": True, "to": to, "subject": subject}


def reply_email(arguments: dict[str, Any]) -> dict[str, Any]:
    """Reply to an email found by subject in macOS Mail.app.

    Finds the most recent email matching the subject and sends a reply
    to the original sender.

    sensitivity_tier: 3
    """
    subject = str(arguments.get("subject", "")).strip()
    body = str(
        arguments.get("body") or arguments.get("content") or "",
    ).strip()

    if not subject:
        raise ValueError("reply_email requires a 'subject' to find the email")
    if not body:
        raise ValueError("reply_email requires a 'body' for the reply text")

    script = f"""
tell application "Mail"
    set matchedMsgs to (every message of inbox whose subject contains \
"{_escape_apple_text(subject)}")
    if (count of matchedMsgs) is 0 then
        error "No email found with subject containing: \
{_escape_apple_text(subject)}"
    end if
    set targetMsg to item 1 of matchedMsgs
    set senderAddr to extract address from sender of targetMsg
    set origSubject to subject of targetMsg
    set replySubject to "Re: " & origSubject
    set newMessage to make new outgoing message with properties \
{{subject:replySubject, \
content:"{_escape_apple_text(body)}", visible:false}}
    tell newMessage
        make new to recipient at end of to recipients \
with properties {{address:senderAddr}}
    end tell
    send newMessage
end tell
return "replied"
""".strip()

    _run_osascript(script)
    return {"success": True, "subject": "Re: " + subject}


def delete_email(arguments: dict[str, Any]) -> dict[str, Any]:
    """Move an email to Trash in macOS Mail.app.

    Finds the most recent inbox email matching the subject and deletes it.

    sensitivity_tier: 3
    """
    subject = str(arguments.get("subject", "")).strip()
    if not subject:
        raise ValueError("delete_email requires a 'subject' to find the email")

    script = f"""
tell application "Mail"
    set matchedMsgs to (every message of inbox whose subject contains \
"{_escape_apple_text(subject)}")
    if (count of matchedMsgs) is 0 then
        error "No email found with subject containing: \
{_escape_apple_text(subject)}"
    end if
    delete item 1 of matchedMsgs
end tell
return "deleted"
""".strip()

    _run_osascript(script)
    return {"success": True, "subject": subject}


def move_email(arguments: dict[str, Any]) -> dict[str, Any]:
    """Move an email to a different mailbox in macOS Mail.app.

    sensitivity_tier: 3
    """
    subject = str(arguments.get("subject", "")).strip()
    folder = str(arguments.get("folder", "")).strip()

    if not subject:
        raise ValueError("move_email requires a 'subject' to find the email")
    if not folder:
        raise ValueError("move_email requires a 'folder' destination")

    script = f"""
tell application "Mail"
    set matchedMsgs to (every message of inbox whose subject contains \
"{_escape_apple_text(subject)}")
    if (count of matchedMsgs) is 0 then
        error "No email found with subject containing: \
{_escape_apple_text(subject)}"
    end if
    set targetMsg to item 1 of matchedMsgs
    set targetBox to missing value
    repeat with acct in accounts
        try
            set targetBox to mailbox "{_escape_apple_text(folder)}" of acct
            exit repeat
        end try
    end repeat
    if targetBox is missing value then
        error "Mailbox not found: {_escape_apple_text(folder)}"
    end if
    move targetMsg to targetBox
end tell
return "moved"
""".strip()

    _run_osascript(script)
    return {"success": True, "subject": subject, "folder": folder}


def flag_email(arguments: dict[str, Any]) -> dict[str, Any]:
    """Flag or unflag an email in macOS Mail.app.

    sensitivity_tier: 3
    """
    subject = str(arguments.get("subject", "")).strip()
    flagged = _parse_bool(str(arguments.get("flagged", "true")))

    if not subject:
        raise ValueError("flag_email requires a 'subject' to find the email")

    flag_val = "true" if flagged else "false"

    script = f"""
tell application "Mail"
    set matchedMsgs to (every message of inbox whose subject contains \
"{_escape_apple_text(subject)}")
    if (count of matchedMsgs) is 0 then
        error "No email found with subject containing: \
{_escape_apple_text(subject)}"
    end if
    set flagged status of item 1 of matchedMsgs to {flag_val}
end tell
return "flagged"
""".strip()

    _run_osascript(script)
    return {"success": True, "subject": subject, "flagged": flagged}


# ---------------------------------------------------------------------------
# Messages (chat.db)
# ---------------------------------------------------------------------------


def _read_messages_sqlite(
    limit: int,
) -> list[dict[str, Any]]:
    """Read iMessage/SMS from the macOS Messages database.

    sensitivity_tier: 3
    """
    if not _MESSAGES_DB_PATH.exists():
        return []

    def _query(conn: sqlite3.Connection) -> list:
        return conn.execute(
            """
            SELECT
                m.ROWID          AS msg_id,
                m.guid           AS guid,
                m.text           AS text,
                m.date           AS date_ns,
                m.is_from_me     AS is_from_me,
                h.id             AS handle_id,
                c.display_name   AS chat_name
            FROM message m
            LEFT JOIN handle h ON m.handle_id = h.ROWID
            LEFT JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
            LEFT JOIN chat c ON cmj.chat_id = c.ROWID
            WHERE m.text IS NOT NULL AND m.text != ''
            ORDER BY m.date DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    rows = _query_macos_db(_MESSAGES_DB_PATH, _query)

    messages: list[dict[str, Any]] = []
    for row in rows:
        guid = (row["guid"] or "").strip()
        text = (row["text"] or "").strip()
        if not text:
            continue

        msg_id = guid or str(row["msg_id"])
        handle = (row["handle_id"] or "").strip()
        is_from_me = bool(row["is_from_me"])
        chat_name = (row["chat_name"] or "").strip()

        # iMessage dates are in nanoseconds since 2001-01-01
        date_iso = ""
        if row["date_ns"] is not None:
            try:
                unix_ts = (row["date_ns"] / 1_000_000_000) + (
                    _IMESSAGE_EPOCH_OFFSET
                )
                date_iso = (
                    datetime.fromtimestamp(unix_ts, tz=timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z")
                )
            except (TypeError, ValueError, OSError):
                pass

        # Determine sender/recipient based on direction
        sender = "me" if is_from_me else (handle or "unknown")
        recipient = (handle or "unknown") if is_from_me else "me"

        if not date_iso:
            continue

        messages.append({
            "id": msg_id,
            "sender": sender,
            "recipient": recipient,
            "content": text,
            "source": "imessage",
            "timestamp": date_iso,
            "is_from_me": is_from_me,
            "chat_name": chat_name or None,
        })

    return messages


def list_messages(arguments: dict[str, Any]) -> list[dict[str, Any]]:
    """Fetch iMessages/SMS from macOS Messages via SQLite.

    sensitivity_tier: 3
    """
    raw_limit = arguments.get("limit", 1000)
    try:
        limit = int(raw_limit)
    except (TypeError, ValueError):
        limit = 1000
    limit = max(1, min(limit, 2000))
    return _read_messages_sqlite(limit)


def send_message(arguments: dict[str, Any]) -> dict[str, Any]:
    """Send an iMessage via AppleScript.

    sensitivity_tier: 3
    """
    to = str(arguments.get("to", "")).strip()
    text = str(arguments.get("text", "")).strip()
    if not to or not text:
        raise ValueError("send_message requires 'to' and 'text'")

    script = f"""
tell application "Messages"
    set targetService to 1st account whose service type = iMessage
    set targetBuddy to participant "{_escape_apple_text(to)}" ¬
        of targetService
    send "{_escape_apple_text(text)}" to targetBuddy
end tell
return "sent"
""".strip()

    _run_osascript(script)
    return {"success": True, "to": to}


def create_event(arguments: dict[str, Any]) -> dict[str, Any]:
    title = (_arg_str(arguments, "title") or "").strip()
    if not title:
        raise ValueError("create_event requires a non-empty title")

    start_epoch = _parse_iso_to_epoch(_arg_str(arguments, "start_time"))
    end_epoch = _parse_iso_to_epoch(_arg_str(arguments, "end_time"))
    if start_epoch is None:
        start_epoch = int(time.time()) + 300
    if end_epoch is None or end_epoch <= start_epoch:
        end_epoch = start_epoch + 3600

    location = _arg_str(arguments, "location") or ""
    description = _arg_str(arguments, "description") or ""
    _ensure_app_running("Calendar")
    created_id = _run_osascript(
        _calendar_create_script(
            title,
            start_epoch,
            end_epoch,
            location,
            description,
        ),
    ).strip()
    return {
        "success": True,
        "id": created_id or None,
        "title": title,
    }


def create_reminder(arguments: dict[str, Any]) -> dict[str, Any]:
    title = (_arg_str(arguments, "title") or "").strip()
    if not title:
        raise ValueError("create_reminder requires a non-empty title")
    list_name = (_arg_str(arguments, "list_name") or "Reminders").strip() or "Reminders"
    notes = _arg_str(arguments, "notes") or ""
    due_epoch = _parse_iso_to_epoch(_arg_str(arguments, "due_date"))
    created_id = _run_osascript(
        _reminder_create_script(title, list_name, notes, due_epoch),
    ).strip()
    return {
        "success": True,
        "id": created_id or None,
        "title": title,
        "list_name": list_name,
    }


def delete_event(arguments: dict[str, Any]) -> dict[str, Any]:
    """Delete a calendar event from macOS Calendar via AppleScript.

    Finds the event by title and deletes it.

    sensitivity_tier: 2
    """
    title = str(arguments.get("title", "")).strip()
    if not title:
        raise ValueError("delete_event requires a 'title' to find the event")

    script = f"""
tell application "Calendar"
    set matchedEvents to {{}}
    repeat with cal in calendars
        set matchedEvents to matchedEvents & \
(every event of cal whose summary contains \
"{_escape_apple_text(title)}")
    end repeat
    if (count of matchedEvents) is 0 then
        error "No event found with title containing: \
{_escape_apple_text(title)}"
    end if
    delete item 1 of matchedEvents
end tell
return "deleted"
""".strip()

    _ensure_app_running("Calendar")
    _run_osascript(script)
    return {"success": True, "title": title}


def delete_reminder(arguments: dict[str, Any]) -> dict[str, Any]:
    """Delete a reminder from macOS Reminders via AppleScript.

    Finds the reminder by title and deletes it.

    sensitivity_tier: 2
    """
    title = str(arguments.get("title", "")).strip()
    if not title:
        raise ValueError(
            "delete_reminder requires a 'title' to find the reminder",
        )

    script = f"""
tell application "Reminders"
    set matchedReminders to (every reminder whose name contains \
"{_escape_apple_text(title)}")
    if (count of matchedReminders) is 0 then
        error "No reminder found with title containing: \
{_escape_apple_text(title)}"
    end if
    delete item 1 of matchedReminders
end tell
return "deleted"
""".strip()

    _run_osascript(script)
    return {"success": True, "title": title}


def _handle_tool_call(
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    if tool_name == "list_calendar_events":
        rows = list_calendar_events(arguments)
        return {"content": [{"type": "json", "json": rows}], "isError": False}
    if tool_name == "list_reminders":
        rows = list_reminders(arguments)
        return {"content": [{"type": "json", "json": rows}], "isError": False}
    if tool_name == "list_contacts":
        rows = list_contacts(arguments)
        return {"content": [{"type": "json", "json": rows}], "isError": False}
    if tool_name == "search_contacts":
        rows = search_contacts(arguments)
        return {"content": [{"type": "json", "json": rows}], "isError": False}
    if tool_name == "list_notes":
        rows = list_notes(arguments)
        return {"content": [{"type": "json", "json": rows}], "isError": False}
    if tool_name == "create_note":
        result = create_note(arguments)
        return {"content": [{"type": "json", "json": result}], "isError": False}
    if tool_name == "list_emails":
        rows = list_emails(arguments)
        return {"content": [{"type": "json", "json": rows}], "isError": False}
    if tool_name == "list_messages":
        rows = list_messages(arguments)
        return {"content": [{"type": "json", "json": rows}], "isError": False}
    if tool_name == "send_message":
        result = send_message(arguments)
        return {"content": [{"type": "json", "json": result}], "isError": False}
    if tool_name == "create_event":
        result = create_event(arguments)
        return {"content": [{"type": "json", "json": result}], "isError": False}
    if tool_name == "create_reminder":
        result = create_reminder(arguments)
        return {"content": [{"type": "json", "json": result}], "isError": False}
    if tool_name == "delete_event":
        result = delete_event(arguments)
        return {"content": [{"type": "json", "json": result}], "isError": False}
    if tool_name == "delete_reminder":
        result = delete_reminder(arguments)
        return {"content": [{"type": "json", "json": result}], "isError": False}
    if tool_name == "delete_note":
        result = delete_note(arguments)
        return {"content": [{"type": "json", "json": result}], "isError": False}
    if tool_name == "search_notes":
        rows = search_notes(arguments)
        return {"content": [{"type": "json", "json": rows}], "isError": False}
    if tool_name == "update_note":
        result = update_note(arguments)
        return {"content": [{"type": "json", "json": result}], "isError": False}
    if tool_name == "search_emails":
        rows = search_emails(arguments)
        return {"content": [{"type": "json", "json": rows}], "isError": False}
    if tool_name == "send_email":
        result = send_email(arguments)
        return {"content": [{"type": "json", "json": result}], "isError": False}
    if tool_name == "reply_email":
        result = reply_email(arguments)
        return {"content": [{"type": "json", "json": result}], "isError": False}
    if tool_name == "delete_email":
        result = delete_email(arguments)
        return {"content": [{"type": "json", "json": result}], "isError": False}
    if tool_name == "move_email":
        result = move_email(arguments)
        return {"content": [{"type": "json", "json": result}], "isError": False}
    if tool_name == "flag_email":
        result = flag_email(arguments)
        return {"content": [{"type": "json", "json": result}], "isError": False}
    return _tool_error_result(f"Unknown tool: {tool_name}")


def _handle_request(request: dict[str, Any]) -> dict[str, Any] | None:
    req_id = request.get("id")
    method = request.get("method")
    params = request.get("params") or {}

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": SERVER_NAME,
                    "version": SERVER_VERSION,
                },
            },
        }

    if method == "notifications/initialized":
        return None

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": TOOLS},
        }

    if method == "tools/call":
        name = str(params.get("name", ""))
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            arguments = {}
        try:
            result = _handle_tool_call(name, arguments)
        except Exception as exc:  # noqa: BLE001
            result = _tool_error_result(str(exc))
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": result,
        }

    if method == "shutdown":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    if method == "exit":
        raise SystemExit(0)

    if req_id is None:
        return None
    return _error_response(req_id, -32601, f"Method not found: {method}")


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue
        try:
            response = _handle_request(request)
        except SystemExit:
            return 0
        except Exception as exc:  # noqa: BLE001
            req_id = request.get("id")
            response = _error_response(
                req_id,
                -32000,
                f"Bridge error: {exc}",
            )
        if response is not None:
            _send(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
