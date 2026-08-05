"""Minimal MCP client — JSON-RPC 2.0 over stdio.

Communicates with MCP servers via subprocess. Supports both
newline-delimited JSON (JSONL, the current MCP stdio default) and
Content-Length framing (LSP-style, used by older servers).  The
protocol is auto-detected from the first server response.

sensitivity_tier: 1 (protocol metadata, no user data)
"""

from __future__ import annotations

import json
import logging
import os
import select
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class McpConnectionError(Exception):
    """MCP server subprocess could not start or handshake failed.

    sensitivity_tier: 1
    """


class McpTimeoutError(Exception):
    """MCP server did not respond within the timeout.

    sensitivity_tier: 1
    """


class McpToolError(Exception):
    """MCP tool call returned an error.

    sensitivity_tier: 1
    """


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class McpToolInfo:
    """A tool discovered from an MCP server.

    sensitivity_tier: 1
    """

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Wire protocol helpers
# ---------------------------------------------------------------------------

_PROTOCOL_VERSION = "2024-11-05"
_STDERR_TAIL_MAX_LINES = 500

# Legacy catalog tool names mapped to newer MCP tool names.
_LEGACY_TOOL_FALLBACKS: dict[str, tuple[str, dict[str, Any]]] = {
    # Apple MCP aliases
    "list_calendar_events": ("calendar", {"operation": "list", "limit": 200}),
    "list_reminders": ("reminders", {"operation": "list", "limit": 200}),
    # WhatsApp MCP aliases (Baileys server naming)
    "list_chats": ("get_recent_chats", {"limit": 50}),
    "send_message": ("send_text_message", {}),
}


def _encode_jsonl(payload: dict[str, Any]) -> bytes:
    """Encode a JSON-RPC message as newline-delimited JSON.

    sensitivity_tier: 1
    """
    return json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n"


def _encode_framed(payload: dict[str, Any]) -> bytes:
    """Encode a JSON-RPC message with Content-Length framing.

    sensitivity_tier: 1
    """
    body = json.dumps(payload).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    return header + body


class _TimeoutReader:
    """Binary stream reader with per-read timeout via select.

    sensitivity_tier: 1
    """

    def __init__(self, stream: Any, timeout: float) -> None:
        self._stream = stream
        self._timeout = timeout

    def read_byte(self) -> bytes:
        """Read exactly one byte, raising on timeout or EOF.

        sensitivity_tier: 1
        """
        if sys.platform != "win32":
            ready, _, _ = select.select([self._stream], [], [], self._timeout)
            if not ready:
                msg = f"Timed out waiting for response ({self._timeout}s)"
                raise McpTimeoutError(msg)
        b = self._stream.read(1)
        if not b:
            raise McpConnectionError("MCP server closed connection")
        return b

    def read_exact(self, n: int) -> bytes:
        """Read exactly *n* bytes, raising on timeout or EOF.

        sensitivity_tier: 1
        """
        buf = b""
        while len(buf) < n:
            if sys.platform != "win32":
                ready, _, _ = select.select(
                    [self._stream], [], [], self._timeout,
                )
                if not ready:
                    msg = f"Timed out reading body ({self._timeout}s)"
                    raise McpTimeoutError(msg)
            chunk = self._stream.read(n - len(buf))
            if not chunk:
                raise McpConnectionError(
                    "MCP server closed connection during body read",
                )
            buf += chunk
        return buf


def _read_jsonl(reader: _TimeoutReader) -> dict[str, Any]:
    """Read one JSONL message (raw JSON + newline).

    sensitivity_tier: 1
    """
    buf = b""
    while True:
        b = reader.read_byte()
        if b == b"\n":
            break
        buf += b
    if not buf:
        raise McpConnectionError("Empty JSONL response")
    return json.loads(buf.decode("utf-8"))


def _read_framed(reader: _TimeoutReader) -> dict[str, Any]:
    """Read one Content-Length-framed message.

    sensitivity_tier: 1
    """
    # Finish reading headers (first byte already consumed)
    header = b"C"
    while not header.endswith(b"\r\n\r\n"):
        header += reader.read_byte()

    content_length = 0
    for line in header.decode("ascii").strip().split("\r\n"):
        if line.lower().startswith("content-length:"):
            content_length = int(line.split(":", 1)[1].strip())
            break

    if content_length == 0:
        raise McpConnectionError("Missing Content-Length header")

    body = reader.read_exact(content_length)
    return json.loads(body.decode("utf-8"))


def _read_message_auto(reader: _TimeoutReader) -> tuple[dict[str, Any], str]:
    """Read one message, auto-detecting JSONL vs Content-Length framing.

    Returns ``(parsed_message, mode)`` where *mode* is ``"jsonl"`` or
    ``"framed"``.

    sensitivity_tier: 1
    """
    first = reader.read_byte()

    if first == b"{":
        # JSONL — read rest of line
        buf = first
        while True:
            b = reader.read_byte()
            if b == b"\n":
                break
            buf += b
        return json.loads(buf.decode("utf-8")), "jsonl"

    if first.upper() == b"C":
        # Content-Length framing
        return _read_framed(reader), "framed"

    msg = f"Unexpected first byte from MCP server: {first!r}"
    raise McpConnectionError(msg)


def _is_unknown_tool_error(message: str) -> bool:
    """Detect common "tool not found" error shapes.

    sensitivity_tier: 1
    """
    text = message.lower()
    return (
        "unknown tool" in text
        or "tool not found" in text
        or "method not found" in text
        or "not found" in text and "tool" in text
    )


def _flatten_nested_records(
    records: list[dict[str, Any]],
    keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Flatten wrapper records that contain nested result lists.

    sensitivity_tier: 1
    """
    flat: list[dict[str, Any]] = []
    for record in records:
        # Some MCP servers return mixed lists with raw strings
        # interleaved with dict records. Skip the strings (and any
        # other non-dict types) instead of crashing with
        # ``'str' object has no attribute 'get'``.
        if not isinstance(record, dict):
            continue
        nested: list[dict[str, Any]] | None = None
        for key in keys:
            value = record.get(key)
            if isinstance(value, list):
                nested = [v for v in value if isinstance(v, dict)]
                break
        if nested is not None:
            flat.extend(nested)
        else:
            flat.append(record)
    return flat


def _normalize_legacy_apple_records(
    requested_tool_name: str,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize Apple MCP records into legacy catalog field names.

    sensitivity_tier: 1
    """
    if requested_tool_name == "list_calendar_events":
        events = _flatten_nested_records(
            records,
            ("events", "items", "results", "data"),
        )
        normalized: list[dict[str, Any]] = []

        def _extract_time(raw: Any) -> Any:
            if isinstance(raw, dict):
                return raw.get("dateTime") or raw.get("date")
            return raw

        for event in events:
            row = dict(event)
            raw_text = str(row.get("_raw_text", "")).strip().lower()
            if "no events found" in raw_text or "too slow" in raw_text:
                continue
            row.setdefault(
                "id",
                event.get("eventIdentifier")
                or event.get("uid")
                or event.get("identifier"),
            )
            row.setdefault(
                "title",
                event.get("summary")
                or event.get("name")
                or event.get("subject")
                or "Untitled Event",
            )
            row.setdefault(
                "start_time",
                _extract_time(event.get("startDate"))
                or _extract_time(event.get("start"))
                or _extract_time(event.get("date")),
            )
            row.setdefault(
                "end_time",
                _extract_time(event.get("endDate"))
                or _extract_time(event.get("end"))
                or row.get("start_time"),
            )
            row.setdefault("description", event.get("notes"))
            row.setdefault("is_all_day", event.get("isAllDay"))
            if "attendees" not in row:
                row["attendees"] = event.get("attendees", [])
            title_text = str(row.get("title", "")).strip().lower()
            desc_text = str(row.get("description", "")).strip().lower()
            event_id = str(row.get("id", "")).strip().lower()
            if (
                "calendar operations too slow" in title_text
                or "notoriously slow and unreliable" in desc_text
                or event_id in {"dummy-event-1", "dummy-event"}
            ):
                continue
            # Keep sync resilient: skip records still missing required timestamps.
            if not row.get("start_time") or not row.get("end_time"):
                logger.debug(
                    "Skipping calendar record missing time bounds: %s",
                    row,
                )
                continue
            normalized.append(row)
        return normalized

    if requested_tool_name == "list_reminders":
        reminders = _flatten_nested_records(
            records,
            ("reminders", "tasks", "items", "results", "data"),
        )
        normalized = []
        for reminder in reminders:
            row = dict(reminder)
            raw_text = str(row.get("_raw_text", "")).strip().lower()
            if (
                "found 0 lists and 0 reminders" in raw_text
                or "reminders_query_too_slow" in raw_text
                or "reminder_search_not_implemented_for_performance" in raw_text
                or "reminders_by_id_not_implemented_for_performance" in raw_text
            ):
                continue
            row.setdefault(
                "id",
                reminder.get("identifier")
                or reminder.get("uid")
                or reminder.get("reminderId"),
            )
            row.setdefault(
                "title",
                reminder.get("name")
                or reminder.get("title")
                or reminder.get("summary")
                or "Untitled Reminder",
            )
            row.setdefault(
                "due_date",
                reminder.get("dueDate") or reminder.get("due"),
            )
            row.setdefault(
                "notes",
                reminder.get("body") or reminder.get("notes"),
            )
            row.setdefault(
                "list_name",
                reminder.get("listName") or reminder.get("list"),
            )
            rid = str(row.get("id", "")).strip().lower()
            if rid in {"none", "null"}:
                row["id"] = None
                rid = ""
            title = str(row.get("title", "") or "").strip()
            notes = str(row.get("notes", "") or "").strip()
            due = str(row.get("due_date", "") or "").strip()
            list_name = str(row.get("list_name", "") or "").strip()
            if title.lower() == "untitled reminder" and not any(
                [rid, notes, due, list_name],
            ):
                continue
            if not title:
                continue
            normalized.append(row)
        return normalized

    return records


def _normalize_whatsapp_timestamp(value: Any) -> str | None:
    """Best-effort normalize WhatsApp timestamps to ISO 8601 strings.

    Supports ISO text, unix seconds, unix milliseconds, and numeric strings.

    sensitivity_tier: 1
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        num = float(value)
        if num > 1e12:
            num /= 1000.0
        try:
            return datetime.fromtimestamp(
                num, tz=timezone.utc,
            ).isoformat()
        except (OSError, OverflowError, ValueError):
            return None

    text = str(value).strip()
    if not text:
        return None

    # Numeric text payloads from some Baileys builds.
    try:
        num = float(text)
    except ValueError:
        num = 0.0
    else:
        if num != 0.0:
            if num > 1e12:
                num /= 1000.0
            try:
                return datetime.fromtimestamp(
                    num, tz=timezone.utc,
                ).isoformat()
            except (OSError, OverflowError, ValueError):
                return None

    # Already-ISO payloads.
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return text


def _normalize_legacy_whatsapp_records(
    requested_tool_name: str,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize WhatsApp MCP chat rows into raw_messages-compatible shape.

    The Baileys-based server exposes ``get_recent_chats`` with fields such as
    ``name``, ``lastMessage``, and ``lastMessageTime``. Our catalog expects
    ``sender/content/timestamp/chat_name/is_group`` and raw_messages also needs
    ``recipient`` to be non-null.

    sensitivity_tier: 1
    """
    if requested_tool_name != "list_chats":
        return records

    chats = _flatten_nested_records(
        records,
        ("chats", "items", "results", "data"),
    )
    normalized: list[dict[str, Any]] = []
    for chat in chats:
        raw_text = str(chat.get("_raw_text", "")).strip().lower()
        if raw_text:
            # get_recent_chats can return plain-text error payloads such as
            # "Error: Invalid time value"; drop those rows to keep sync alive.
            if raw_text.startswith("error:") or "invalid time value" in raw_text:
                continue

        chat_id = (
            chat.get("id")
            or chat.get("chat_id")
            or chat.get("chatId")
            or chat.get("jid")
        )
        if not chat_id:
            continue
        chat_id_text = str(chat_id).strip()
        if not chat_id_text:
            continue

        chat_name = (
            chat.get("chat_name")
            or chat.get("name")
            or chat.get("title")
            or chat_id_text
        )
        chat_name_text = str(chat_name).strip() or chat_id_text

        content = (
            chat.get("content")
            or chat.get("lastMessage")
            or chat.get("last_message")
            or chat.get("text")
            or ""
        )

        ts = _normalize_whatsapp_timestamp(
            chat.get("timestamp")
            or chat.get("lastMessageTime")
            or chat.get("last_message_time")
            or chat.get("time"),
        )
        if ts is None:
            # Skip rows with unusable timestamps to avoid violating
            # raw_messages.timestamp NOT NULL constraints.
            continue

        sender = chat.get("sender") or chat_name_text
        recipient = chat.get("recipient") or chat_name_text
        is_group_raw = chat.get("is_group", chat.get("isGroup"))

        normalized.append(
            {
                "id": chat_id_text,
                "sender": str(sender).strip() or chat_name_text,
                "recipient": str(recipient).strip() or chat_name_text,
                "content": str(content),
                "timestamp": ts,
                "chat_name": chat_name_text,
                "is_group": bool(is_group_raw),
            }
        )

    return normalized


def _extract_structured_result_records(
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract records from structured MCP result payload fields.

    Some MCP servers (including apple-mcp) return tool data on
    top-level result keys (for example ``events`` or ``reminders``)
    instead of JSON text content. This helper preserves those rows.

    sensitivity_tier: 1
    """
    records: list[dict[str, Any]] = []

    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        records.append(structured)
    elif isinstance(structured, list):
        records.extend(v for v in structured if isinstance(v, dict))

    # Prefer known list-style payload keys used by popular MCP servers.
    container_keys = (
        "data",
        "items",
        "results",
        "events",
        "reminders",
        "contacts",
        "notes",
        "emails",
        "messages",
        "lists",
    )
    for key in container_keys:
        value = result.get(key)
        if isinstance(value, list):
            dict_items = [v for v in value if isinstance(v, dict)]
            if dict_items:
                records.append({key: dict_items})
        elif isinstance(value, dict):
            records.append(value)

    return records


# ---------------------------------------------------------------------------
# MCP Client
# ---------------------------------------------------------------------------


class McpClient:
    """Minimal MCP client using JSON-RPC 2.0 over stdio.

    Spawns an MCP server as a subprocess, performs the protocol handshake,
    and provides methods to list tools and call them.  Auto-detects whether
    the server uses JSONL or Content-Length framing on the first response.

    sensitivity_tier: 1
    """

    def __init__(
        self,
        command: str,
        args: tuple[str, ...] = (),
        timeout: float = 30.0,
        env: dict[str, str] | None = None,
    ) -> None:
        self._command = command
        self._args = args
        self._timeout = timeout
        # Extra env vars layered on top of the parent process env.
        # Treated as Tier 3 secrets — never logged or persisted to cache.
        self._extra_env = dict(env) if env else None
        self._process: subprocess.Popen[bytes] | None = None
        self._request_id = 0
        self._stderr_thread: threading.Thread | None = None
        self._stderr_output: list[str] = []
        self._mode: str = "jsonl"  # default; overwritten by auto-detect
        self._reader: _TimeoutReader | None = None

    def _next_id(self) -> int:
        """Generate the next JSON-RPC request ID.

        sensitivity_tier: 1
        """
        self._request_id += 1
        return self._request_id

    def _send(self, message: dict[str, Any]) -> None:
        """Send a JSON-RPC message to the MCP server.

        sensitivity_tier: 1
        """
        if self._process is None or self._process.stdin is None:
            msg = "MCP client not connected"
            raise McpConnectionError(msg)
        if self._mode == "framed":
            data = _encode_framed(message)
        else:
            data = _encode_jsonl(message)
        self._process.stdin.write(data)
        self._process.stdin.flush()

    def _recv(self) -> dict[str, Any]:
        """Receive a JSON-RPC message from the MCP server.

        Skips notification messages (no 'id' field) and returns the
        next response.

        sensitivity_tier: 1
        """
        if self._reader is None:
            msg = "MCP client not connected"
            raise McpConnectionError(msg)

        while True:
            if self._mode == "jsonl":
                try:
                    msg = _read_jsonl(self._reader)
                except json.JSONDecodeError:
                    # Some MCP servers occasionally emit non-JSON log
                    # lines on stdout. Skip those and keep reading.
                    continue
            else:
                msg = _read_framed(self._reader)
            # Skip notifications (they have no 'id')
            if "id" in msg:
                return msg

    def _drain_stderr(self) -> None:
        """Read stderr in background thread to prevent pipe deadlock.

        sensitivity_tier: 1
        """
        if self._process is None or self._process.stderr is None:
            return
        for raw_line in self._process.stderr:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\n")
            self._stderr_output.append(line)
            overflow = len(self._stderr_output) - _STDERR_TAIL_MAX_LINES
            if overflow > 0:
                del self._stderr_output[:overflow]

    def _stderr_hint(self, max_lines: int = 10) -> str:
        """Return the tail of captured stderr for error messages.

        sensitivity_tier: 1
        """
        if not self._stderr_output:
            return ""
        tail = self._stderr_output[-max_lines:]
        return "\n".join(tail)

    def _check_process_alive(self) -> None:
        """Raise McpConnectionError if the process has already exited.

        sensitivity_tier: 1
        """
        if self._process is None:
            return
        returncode = self._process.poll()
        if returncode is not None:
            # Give stderr thread a moment to capture output
            if self._stderr_thread:
                self._stderr_thread.join(timeout=1.0)
            hint = self._stderr_hint()
            msg = f"MCP server exited immediately (code {returncode})"
            if hint:
                msg += f"\nServer output:\n{hint}"
            raise McpConnectionError(msg)

    def connect(self) -> None:
        """Start the MCP server subprocess and perform handshake.

        Sends 'initialize' request and waits for the response, then
        sends 'notifications/initialized'.  The wire protocol (JSONL vs
        Content-Length framing) is auto-detected from the first response.

        sensitivity_tier: 1
        """
        cmd = [self._command, *self._args]
        if self._extra_env:
            popen_env = dict(os.environ)
            popen_env.update(self._extra_env)
        else:
            popen_env = None
        try:
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                start_new_session=True,
                env=popen_env,
            )
        except FileNotFoundError as exc:
            msg = f"Command not found: {self._command}"
            raise McpConnectionError(msg) from exc
        except OSError as exc:
            msg = f"Failed to start MCP server: {exc}"
            raise McpConnectionError(msg) from exc

        self._reader = _TimeoutReader(self._process.stdout, self._timeout)

        # Start stderr drain thread
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True,
        )
        self._stderr_thread.start()

        # Brief pause to let the process fail fast if command is invalid
        time.sleep(0.2)
        self._check_process_alive()

        # Send initialize request (try JSONL first — the current default)
        init_id = self._next_id()
        init_msg = {
            "jsonrpc": "2.0",
            "id": init_id,
            "method": "initialize",
            "params": {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": "arandu",
                    "version": "1.0.0",
                },
            },
        }
        try:
            self._send(init_msg)
        except (BrokenPipeError, OSError):
            self._check_process_alive()
            hint = self._stderr_hint()
            msg = "MCP server closed stdin before handshake"
            if hint:
                msg += f"\nServer output:\n{hint}"
            raise McpConnectionError(msg)

        # Wait for initialize response — auto-detect protocol
        try:
            response, detected_mode = _read_message_auto(self._reader)
        except McpConnectionError:
            # Enrich the error with stderr output
            if self._stderr_thread:
                self._stderr_thread.join(timeout=1.0)
            hint = self._stderr_hint()
            msg = "MCP server closed connection during handshake"
            if hint:
                msg += f"\nServer output:\n{hint}"
            raise McpConnectionError(msg)
        except McpTimeoutError:
            if self._stderr_thread:
                self._stderr_thread.join(timeout=1.0)
            hint = self._stderr_hint()
            msg = f"MCP server did not respond within {self._timeout}s"
            if hint:
                msg += f"\nServer output:\n{hint}"
            raise McpTimeoutError(msg)

        self._mode = detected_mode
        logger.info("MCP wire protocol: %s", self._mode)

        if "error" in response:
            error = response["error"]
            msg = f"MCP initialize failed: {error.get('message', error)}"
            raise McpConnectionError(msg)

        # Send initialized notification
        self._send({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        })

        logger.info("MCP client connected to %s", self._command)

    def list_tools(self) -> list[McpToolInfo]:
        """Discover all tools provided by the MCP server.

        sensitivity_tier: 1
        """
        req_id = self._next_id()
        self._send({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "tools/list",
            "params": {},
        })

        response = self._recv()
        if "error" in response:
            error = response["error"]
            msg = f"tools/list failed: {error.get('message', error)}"
            raise McpConnectionError(msg)

        result = response.get("result", {})
        tools_data = result.get("tools", [])

        tools: list[McpToolInfo] = []
        for t in tools_data:
            tools.append(McpToolInfo(
                name=t.get("name", ""),
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", {}),
            ))

        logger.info("Discovered %d tools from %s", len(tools), self._command)
        return tools

    def _call_tool_rpc(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send one tools/call request and return the raw RPC response.

        sensitivity_tier: 1
        """
        req_id = self._next_id()
        self._send({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments or {},
            },
        })
        return self._recv()

    def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Call an MCP tool and return its content items as dicts.

        Text content that looks like JSON is automatically parsed.

        sensitivity_tier: 1
        """
        response = self._call_tool_rpc(tool_name, arguments)
        fallback = _LEGACY_TOOL_FALLBACKS.get(tool_name)

        if "error" in response and fallback is not None:
            error = response["error"]
            err_detail = str(
                error.get("message", error)
                if isinstance(error, dict) else error,
            )
            if _is_unknown_tool_error(err_detail):
                mapped_name, mapped_args = fallback
                merged_args = dict(mapped_args)
                if arguments:
                    merged_args.update(arguments)
                response = self._call_tool_rpc(mapped_name, merged_args)

        if "error" in response:
            error = response["error"]
            err_detail = (
                error.get("message", error)
                if isinstance(error, dict) else error
            )
            msg = f"Tool '{tool_name}' failed: {err_detail}"
            raise McpToolError(msg)

        result = response.get("result", {})

        # Check for tool-level error
        if result.get("isError"):
            content = result.get("content", [])
            error_text = ""
            if content:
                error_text = content[0].get("text", "Unknown error")
            if fallback is not None and _is_unknown_tool_error(error_text):
                mapped_name, mapped_args = fallback
                merged_args = dict(mapped_args)
                if arguments:
                    merged_args.update(arguments)
                response = self._call_tool_rpc(mapped_name, merged_args)
                result = response.get("result", {})
                if result.get("isError"):
                    content = result.get("content", [])
                    error_text = (
                        content[0].get("text", "Unknown error")
                        if content
                        else "Unknown error"
                    )
                    msg = f"Tool '{tool_name}' returned error: {error_text}"
                    raise McpToolError(msg)
            else:
                msg = f"Tool '{tool_name}' returned error: {error_text}"
                raise McpToolError(msg)

        # Extract content items
        records: list[dict[str, Any]] = []
        for item in result.get("content", []):
            if item.get("type") == "text":
                text = item.get("text", "")
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, list):
                        records.extend(
                            r for r in parsed if isinstance(r, dict)
                        )
                    elif isinstance(parsed, dict):
                        records.append(parsed)
                except (json.JSONDecodeError, TypeError):
                    records.append({"_raw_text": text})
            elif item.get("type") == "json":
                payload = item.get("json")
                if isinstance(payload, list):
                    records.extend(
                        r for r in payload if isinstance(r, dict)
                    )
                elif isinstance(payload, dict):
                    records.append(payload)

        structured_records = _extract_structured_result_records(result)
        if structured_records:
            records.extend(structured_records)
            # Drop plain text placeholders when structured rows exist.
            records = [
                r for r in records if set(r.keys()) != {"_raw_text"}
            ]

        if fallback is not None:
            records = _normalize_legacy_apple_records(
                tool_name, records,
            )
            records = _normalize_legacy_whatsapp_records(
                tool_name, records,
            )

        return records

    def close(self) -> None:
        """Shut down the MCP server subprocess and all its children.

        Uses process-group kill to ensure grandchild processes (e.g.
        node spawned by npx/npm) are cleaned up.

        sensitivity_tier: 1
        """
        if self._process is None:
            return

        # Try graceful shutdown via MCP protocol
        try:
            self._send({
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "shutdown",
                "params": {},
            })
            self._send({
                "jsonrpc": "2.0",
                "method": "exit",
            })
        except (McpConnectionError, OSError, BrokenPipeError):
            pass

        pgid: int | None = None
        try:
            pgid = os.getpgid(self._process.pid)
        except (OSError, ProcessLookupError):
            pass

        try:
            # Kill entire process group (npx → npm → node)
            if pgid is not None and pgid != os.getpgid(os.getpid()):
                os.killpg(pgid, signal.SIGTERM)
            else:
                self._process.terminate()
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if pgid is not None and pgid != os.getpgid(os.getpid()):
                os.killpg(pgid, signal.SIGKILL)
            else:
                self._process.kill()
            self._process.wait(timeout=2)
        except (OSError, ProcessLookupError):
            pass

        self._process = None
        self._reader = None
        logger.info("MCP client disconnected from %s", self._command)

    def __enter__(self) -> McpClient:
        """Context manager entry — connects to the MCP server.

        sensitivity_tier: 1
        """
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Context manager exit — closes the MCP server.

        sensitivity_tier: 1
        """
        self.close()
