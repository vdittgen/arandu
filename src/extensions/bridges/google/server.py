"""Local MCP bridge for Google Calendar and Gmail.

Speaks the same hand-rolled JSON-RPC stdio protocol as the Apple bridge
(`src/extensions/bridges/apple/server.py`), so the connector catalog and
`IngestionAdapter` treat it like any other MCP server — no adapter
changes, no sync-engine changes.

**No new dependencies.** The Calendar and Gmail REST APIs are plain
JSON over HTTPS, so this uses `urllib.request` from the stdlib rather
than pulling in `google-api-python-client` and its transitive tree. The
surface we need is two read-only list endpoints; a full API client would
be a lot of weight (and a lot of new supply-chain) for that.

**Auth** reuses the OAuth machinery the connector requirements layer
already has (`src/extensions/connectors/requirements.py`): the user
authorizes in their own browser, the PKCE code exchange stores a JSON
credential (access + refresh token, expiry, token endpoint) in the
macOS Keychain under `arandu-oauth-google_oauth`, and this bridge reads
it back — refreshing the ~1h access token itself, proactively on expiry
and once more on a mid-call 401. Legacy bare-token entries still work,
minus the refresh. No client secret and no token ever lives in the repo
or in settings.json.

**Body content is deliberately not fetched.** Gmail messages are read
with `format=metadata`, which returns headers plus Google's own
`snippet` and *not* the message body. `raw_emails.body_preview` only
ever needs a preview, and requesting full bodies would pull the most
sensitive content on the account into local storage for no downstream
gain — the proactive pillars work from sender/subject/recency.

sensitivity_tier: 3 (reads calendar and mail metadata)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from email.utils import getaddresses, parsedate_to_datetime
from typing import Any

SERVER_NAME = "arandu-google-bridge"
SERVER_VERSION = "1.0.0"
PROTOCOL_VERSION = "2024-11-05"

# Provider string shared with the catalog's `requires_auth`, which
# determines the Keychain service name via
# `RequirementsChecker._oauth_service_name`.
OAUTH_PROVIDER = "google_oauth"
KEYCHAIN_SERVICE = f"arandu-oauth-{OAUTH_PROVIDER}"

# Test/CI seam, mirroring ARANDU_OAUTH_<PROVIDER>_TEST_TOKEN in the
# requirements checker: lets the bridge run without a Keychain entry.
TOKEN_ENV_VAR = "ARANDU_GOOGLE_OAUTH_TOKEN"
# Point the bridge at a local fake during tests.
API_BASE_ENV_VAR = "ARANDU_GOOGLE_API_BASE"

DEFAULT_API_BASE = "https://www.googleapis.com"
HTTP_TIMEOUT_SECONDS = 20
DEFAULT_LIMIT = 200
MAX_LIMIT = 500
# Gmail metadata headers we ask for — everything the canonical
# raw_emails columns need, and nothing more.
_GMAIL_HEADERS = ("Subject", "From", "To", "Date")


class GoogleAuthError(RuntimeError):
    """No usable Google credential, or Google rejected the one we have.

    sensitivity_tier: 1
    """


TOOLS: list[dict[str, Any]] = [
    {
        "name": "list_calendar_events",
        "description": "List Google Calendar events in a date range.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": MAX_LIMIT},
                "fromDate": {"type": "string"},
                "toDate": {"type": "string"},
                "calendarId": {"type": "string"},
            },
            "additionalProperties": True,
        },
    },
    {
        "name": "list_emails",
        "description": "List Gmail messages (metadata and snippet only).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": MAX_LIMIT},
                "query": {"type": "string"},
                "fromDate": {"type": "string"},
            },
            "additionalProperties": True,
        },
    },
]


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Credential:
    """Parsed Google credential, bare-token or refreshable JSON form.

    The OAuth flow stores a JSON blob (access + refresh token, expiry,
    token endpoint, client id/secret) when a token URL is configured;
    older installs may hold a bare access-token string. Env-supplied
    tokens (test seam) are never refreshed or persisted.

    sensitivity_tier: 3 (carries bearer tokens)
    """

    access_token: str
    refresh_token: str = ""
    expires_at: str = ""
    token_url: str = ""
    client_id: str = ""
    client_secret: str = ""
    from_env: bool = False

    @property
    def refreshable(self) -> bool:
        return bool(
            self.refresh_token and self.token_url and not self.from_env,
        )


def _keychain_read() -> str:
    """Read the raw Keychain entry the OAuth flow wrote.

    sensitivity_tier: 3 (returns credential material)
    """
    try:
        result = subprocess.run(
            [
                "security",
                "find-generic-password",
                "-s",
                KEYCHAIN_SERVICE,
                "-w",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        msg = "Could not read the Keychain to load Google credentials"
        raise GoogleAuthError(msg) from exc
    raw = result.stdout.strip()
    if result.returncode != 0 or not raw:
        msg = (
            "Google is not connected. Authorize it from Connectors "
            "(Settings -> Connectors -> Google) and try again."
        )
        raise GoogleAuthError(msg)
    return raw


def _keychain_write(value: str) -> None:
    """Persist a refreshed credential back to the Keychain.

    Best-effort: the in-memory token still works for this run even if
    the write fails; the next run just refreshes again.

    sensitivity_tier: 3 (writes credential material)
    """
    try:
        subprocess.run(
            [
                "security",
                "add-generic-password",
                "-U",
                "-s",
                KEYCHAIN_SERVICE,
                "-a",
                "arandu",
                "-w",
                value,
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass


def _load_credential() -> _Credential:
    """Load the stored Google credential.

    Env var first so tests and CI never touch the Keychain (and never
    need a real Google account); otherwise parse the Keychain entry —
    JSON credential from the PKCE exchange, or a legacy bare token.

    sensitivity_tier: 3
    """
    env_token = os.environ.get(TOKEN_ENV_VAR, "").strip()
    if env_token:
        return _Credential(access_token=env_token, from_env=True)
    raw = _keychain_read()
    if raw.startswith("{"):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            return _Credential(
                access_token=str(data.get("access_token") or ""),
                refresh_token=str(data.get("refresh_token") or ""),
                expires_at=str(data.get("expires_at") or ""),
                token_url=str(data.get("token_url") or ""),
                client_id=str(data.get("client_id") or ""),
                client_secret=str(data.get("client_secret") or ""),
            )
    return _Credential(access_token=raw)


def _expired(credential: _Credential) -> bool:
    """Whether the access token is at (or within 60s of) expiry.

    Unknown/unparseable expiry reads as "not expired" — the 401 retry
    path catches those.

    sensitivity_tier: 1
    """
    if not credential.expires_at:
        return False
    try:
        expires = datetime.fromisoformat(
            credential.expires_at.replace("Z", "+00:00"),
        )
    except ValueError:
        return False
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    now = datetime.now(tz=timezone.utc)
    return now >= expires - timedelta(seconds=60)


def _refresh_credential(credential: _Credential) -> _Credential:
    """Trade the refresh token for a fresh access token and persist it.

    sensitivity_tier: 3
    """
    data = {
        "grant_type": "refresh_token",
        "refresh_token": credential.refresh_token,
    }
    if credential.client_id:
        data["client_id"] = credential.client_id
    if credential.client_secret:
        data["client_secret"] = credential.client_secret
    payload = _post_form(credential.token_url, data)
    token = str(payload.get("access_token") or "")
    if not token:
        msg = (
            "Google token refresh returned no access token. "
            "Re-authorize Google in Connectors."
        )
        raise GoogleAuthError(msg)
    expires_at = ""
    expires_in = payload.get("expires_in")
    if expires_in is not None:
        try:
            expires_at = (
                datetime.now(tz=timezone.utc)
                + timedelta(seconds=int(expires_in))
            ).isoformat()
        except (TypeError, ValueError):
            expires_at = ""
    refreshed = replace(
        credential,
        access_token=token,
        # Providers may rotate the refresh token on use; keep the old
        # one when they don't.
        refresh_token=str(
            payload.get("refresh_token") or credential.refresh_token,
        ),
        expires_at=expires_at,
    )
    stored = {
        "access_token": refreshed.access_token,
        "refresh_token": refreshed.refresh_token,
        "token_url": refreshed.token_url,
    }
    if refreshed.expires_at:
        stored["expires_at"] = refreshed.expires_at
    if refreshed.client_id:
        stored["client_id"] = refreshed.client_id
    if refreshed.client_secret:
        stored["client_secret"] = refreshed.client_secret
    _keychain_write(json.dumps(stored))
    return refreshed


def _post_form(url: str, data: dict[str, str]) -> dict[str, Any]:
    """POST a urlencoded form to the token endpoint, decode JSON.

    Failures surface as `GoogleAuthError` — every caller is on an auth
    path where "re-authorize" is the actionable outcome.

    sensitivity_tier: 3 (carries tokens)
    """
    body = urllib.parse.urlencode(data).encode("ascii")
    request = urllib.request.Request(  # noqa: S310 - operator-set URL
        url,
        data=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(  # noqa: S310
            request, timeout=HTTP_TIMEOUT_SECONDS,
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        msg = (
            f"Google token refresh was rejected (HTTP {exc.code}). "
            "Re-authorize Google in Connectors."
        )
        raise GoogleAuthError(msg) from exc
    except urllib.error.URLError as exc:
        msg = f"Could not reach the Google token endpoint: {exc.reason}"
        raise GoogleAuthError(msg) from exc
    except (OSError, ValueError) as exc:
        msg = f"Google token refresh failed: {exc}"
        raise GoogleAuthError(msg) from exc
    if not isinstance(payload, dict):
        msg = "Google token endpoint response was not a JSON object"
        raise GoogleAuthError(msg)
    return payload


def _access_token() -> str:
    """Return a usable Google access token, refreshing if expired.

    sensitivity_tier: 3 (returns a bearer token)
    """
    credential = _load_credential()
    if _expired(credential) and credential.refreshable:
        credential = _refresh_credential(credential)
    if not credential.access_token:
        msg = (
            "Google is not connected. Authorize it from Connectors "
            "(Settings -> Connectors -> Google) and try again."
        )
        raise GoogleAuthError(msg)
    return credential.access_token


def _try_refresh() -> bool:
    """Refresh the stored credential after a mid-call auth failure.

    Returns True when a refresh happened (so the caller may retry
    once); False when the credential can't refresh itself — env-token,
    legacy bare-token, or a refresh the provider rejected.

    sensitivity_tier: 2
    """
    try:
        credential = _load_credential()
    except GoogleAuthError:
        return False
    if not credential.refreshable:
        return False
    try:
        _refresh_credential(credential)
    except GoogleAuthError:
        return False
    return True


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def _api_base() -> str:
    return os.environ.get(API_BASE_ENV_VAR, DEFAULT_API_BASE).rstrip("/")


def _api_get(path: str, params: dict[str, Any], token: str) -> dict[str, Any]:
    """GET a Google REST endpoint and decode the JSON body.

    sensitivity_tier: 3 (response carries user data)
    """
    query = urllib.parse.urlencode(
        {k: v for k, v in params.items() if v is not None},
        doseq=True,
    )
    url = f"{_api_base()}{path}?{query}" if query else f"{_api_base()}{path}"
    request = urllib.request.Request(  # noqa: S310 - fixed https base
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(  # noqa: S310
            request, timeout=HTTP_TIMEOUT_SECONDS,
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            msg = (
                "Google rejected the stored credential (HTTP "
                f"{exc.code}). Re-authorize Google in Connectors."
            )
            raise GoogleAuthError(msg) from exc
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:200]
        except Exception:  # noqa: BLE001 - best-effort error detail
            detail = ""
        msg = f"Google API error (HTTP {exc.code}) on {path}: {detail}"
        raise RuntimeError(msg) from exc
    except urllib.error.URLError as exc:
        msg = f"Could not reach the Google API: {exc.reason}"
        raise RuntimeError(msg) from exc
    if not isinstance(payload, dict):
        msg = f"Unexpected Google API response shape for {path}"
        raise RuntimeError(msg)
    return payload


def _paginate(
    path: str,
    params: dict[str, Any],
    token: str,
    *,
    items_key: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Follow ``nextPageToken`` until ``limit`` items or pages run out.

    sensitivity_tier: 3
    """
    items: list[dict[str, Any]] = []
    page_token: str | None = None
    while len(items) < limit:
        page_params = dict(params)
        page_params["maxResults"] = min(limit - len(items), 250)
        if page_token:
            page_params["pageToken"] = page_token
        payload = _api_get(path, page_params, token)
        batch = payload.get(items_key) or []
        if not isinstance(batch, list):
            break
        items.extend(item for item in batch if isinstance(item, dict))
        page_token = payload.get("nextPageToken")
        if not page_token:
            break
    return items[:limit]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_limit(arguments: dict[str, Any]) -> int:
    raw = arguments.get("limit")
    try:
        value = int(raw) if raw is not None else DEFAULT_LIMIT
    except (TypeError, ValueError):
        return DEFAULT_LIMIT
    return max(1, min(MAX_LIMIT, value))


def _arg_str(arguments: dict[str, Any], key: str) -> str | None:
    value = arguments.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _rfc3339(value: str | None, *, default_days: int) -> str:
    """Coerce a date-ish argument to RFC 3339, or fall back N days out.

    Negative ``default_days`` reaches into the past.

    sensitivity_tier: 1
    """
    if value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            parsed = None
        if parsed is not None:
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc).isoformat()
    fallback = datetime.now(tz=timezone.utc) + timedelta(days=default_days)
    return fallback.isoformat()


def _event_time(slot: Any) -> tuple[str | None, bool]:
    """Return ``(iso_timestamp, is_all_day)`` for a Calendar time slot.

    All-day events carry ``date`` instead of ``dateTime``; the flag has
    to come from which key is present, since a midnight ``dateTime`` is
    a legitimate timed event.

    sensitivity_tier: 2
    """
    if not isinstance(slot, dict):
        return None, False
    date_time = slot.get("dateTime")
    if isinstance(date_time, str) and date_time:
        return date_time, False
    date_only = slot.get("date")
    if isinstance(date_only, str) and date_only:
        return date_only, True
    return None, False


def _attendee_names(event: dict[str, Any]) -> list[str]:
    """Attendee display names (falling back to email) as a JSON array.

    sensitivity_tier: 3
    """
    out: list[str] = []
    for attendee in event.get("attendees") or []:
        if not isinstance(attendee, dict):
            continue
        label = attendee.get("displayName") or attendee.get("email")
        if label:
            out.append(str(label))
    return out


def _self_response_status(event: dict[str, Any]) -> str:
    """This account's RSVP on the event, for meeting-prep ranking.

    sensitivity_tier: 2
    """
    for attendee in event.get("attendees") or []:
        if isinstance(attendee, dict) and attendee.get("self"):
            return str(attendee.get("responseStatus") or "")
    return ""


def _event_origin(event: dict[str, Any]) -> str:
    """``organizer`` when this account owns the event, else ``invited``.

    Mirrors the Apple bridge's column so downstream consumers don't need
    to know which connector produced a row.

    sensitivity_tier: 1
    """
    organizer = event.get("organizer")
    if isinstance(organizer, dict) and organizer.get("self"):
        return "organizer"
    if event.get("attendees"):
        return "invited"
    return "organizer"


def _split_addresses(raw: str) -> list[str]:
    """Split a ``To`` header into individual addresses.

    The canonical ``raw_emails.to_addresses`` column is a JSON array
    (``json_array`` transform), so a single joined string would land as a
    one-element array and break any per-recipient consumer.

    ``getaddresses`` rather than a comma split: a quoted display name
    may itself contain a comma (``"Doe, John" <jdoe@example.com>``), and
    a naive split shears that into two garbage entries — one of which
    would defeat the pending-replies sweep's address matching.

    sensitivity_tier: 3
    """
    if not raw:
        return []
    return [addr for _, addr in getaddresses([raw]) if addr]


def _folder(label_ids: list[Any]) -> str:
    """Map Gmail labels onto the folder vocabulary downstream reads.

    ``Sent`` matters most: the pending-replies sweep resolves a reply by
    finding a later Sent-folder row addressed to the same correspondent
    (``LOWER(folder) LIKE '%sent%'``, matching what the Apple Mail
    bridge writes). Mapping sent mail to ``ARCHIVE`` would leave Gmail
    pending replies resolving only on the weaker is-read signal.

    sensitivity_tier: 1
    """
    if "SENT" in label_ids:
        return "Sent"
    if "INBOX" in label_ids:
        return "INBOX"
    return "ARCHIVE"


def _headers_map(payload: dict[str, Any]) -> dict[str, str]:
    headers = (payload.get("payload") or {}).get("headers") or []
    out: dict[str, str] = {}
    for header in headers:
        if not isinstance(header, dict):
            continue
        name = str(header.get("name", "")).lower()
        if name:
            out[name] = str(header.get("value", ""))
    return out


def _email_date_iso(raw: str) -> str | None:
    """Parse an RFC 2822 ``Date`` header into an ISO 8601 string.

    sensitivity_tier: 1
    """
    if not raw:
        return None
    try:
        parsed = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def list_calendar_events(arguments: dict[str, Any]) -> list[dict[str, Any]]:
    """List Google Calendar events, shaped for ``raw_calendar_events``.

    ``singleEvents=true`` expands recurring series into instances, which
    is what downstream meeting-prep and daily-plan features expect —
    without it a weekly standup would arrive as one master event with a
    recurrence rule nothing here interprets.

    sensitivity_tier: 3
    """
    token = _access_token()
    limit = _normalize_limit(arguments)
    calendar_id = _arg_str(arguments, "calendarId") or "primary"
    params: dict[str, Any] = {
        "timeMin": _rfc3339(_arg_str(arguments, "fromDate"), default_days=-30),
        "timeMax": _rfc3339(_arg_str(arguments, "toDate"), default_days=90),
        "singleEvents": "true",
        "orderBy": "startTime",
    }
    path = (
        "/calendar/v3/calendars/"
        f"{urllib.parse.quote(calendar_id, safe='')}/events"
    )
    events = _paginate(path, params, token, items_key="items", limit=limit)

    rows: list[dict[str, Any]] = []
    for event in events:
        if event.get("status") == "cancelled":
            continue
        start, start_all_day = _event_time(event.get("start"))
        end, end_all_day = _event_time(event.get("end"))
        if not start:
            continue
        rows.append({
            # No `source` column on raw_calendar_events (unlike
            # raw_emails) — the `gcal:` id prefix carries provenance.
            "id": f"gcal:{event.get('id', '')}",
            "title": event.get("summary") or "(no title)",
            "description": event.get("description") or "",
            "start_time": start,
            "end_time": end or start,
            "location": event.get("location") or "",
            "attendees": _attendee_names(event),
            "is_all_day": start_all_day or end_all_day,
            "calendar_name": calendar_id,
            "self_response_status": _self_response_status(event),
            "event_origin": _event_origin(event),
        })
    return rows


def list_emails(arguments: dict[str, Any]) -> list[dict[str, Any]]:
    """List Gmail messages, shaped for ``raw_emails``.

    Metadata + snippet only — see the module docstring on why bodies are
    not fetched.

    sensitivity_tier: 3
    """
    token = _access_token()
    limit = _normalize_limit(arguments)

    query_parts: list[str] = []
    explicit_query = _arg_str(arguments, "query")
    if explicit_query:
        query_parts.append(explicit_query)
    from_date = _arg_str(arguments, "fromDate")
    if from_date:
        # Gmail's `after:` takes YYYY/MM/DD.
        iso = _rfc3339(from_date, default_days=-30)
        query_parts.append(f"after:{iso[:10].replace('-', '/')}")

    listing = _paginate(
        "/gmail/v1/users/me/messages",
        {"q": " ".join(query_parts) or None},
        token,
        items_key="messages",
        limit=limit,
    )

    rows: list[dict[str, Any]] = []
    for stub in listing:
        message_id = stub.get("id")
        if not message_id:
            continue
        detail = _api_get(
            f"/gmail/v1/users/me/messages/{urllib.parse.quote(str(message_id))}",
            {
                "format": "metadata",
                "metadataHeaders": list(_GMAIL_HEADERS),
            },
            token,
        )
        headers = _headers_map(detail)
        label_ids = detail.get("labelIds") or []
        if not isinstance(label_ids, list):
            label_ids = []
        if "DRAFT" in label_ids:
            # Drafts aren't correspondence — ingesting one creates a
            # phantom email that was never sent or received.
            continue
        rows.append({
            "id": f"gmail:{message_id}",
            "source": "gmail",
            "subject": headers.get("subject", ""),
            "from_address": headers.get("from", ""),
            "to_addresses": _split_addresses(headers.get("to", "")),
            "date": _email_date_iso(headers.get("date", "")) or "",
            "body_preview": detail.get("snippet") or "",
            "folder": _folder(label_ids),
            "is_read": "UNREAD" not in label_ids,
        })
    return rows


# ---------------------------------------------------------------------------
# JSON-RPC plumbing
# ---------------------------------------------------------------------------


def _send(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _error_response(req_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": code, "message": message},
    }


def _tool_error_result(message: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": message}], "isError": True}


def _handle_tool_call(
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    handlers = {
        "list_calendar_events": list_calendar_events,
        "list_emails": list_emails,
    }
    handler = handlers.get(tool_name)
    if handler is None:
        return _tool_error_result(f"Unknown tool: {tool_name}")
    try:
        rows = handler(arguments)
    except GoogleAuthError:
        # A 401/403 mid-call usually means the access token expired
        # between the proactive check and the request. If the stored
        # credential can refresh itself, do it once and retry;
        # otherwise let the re-authorize message surface.
        if not _try_refresh():
            raise
        rows = handler(arguments)
    return {"content": [{"type": "json", "json": rows}], "isError": False}


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
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}

    if method == "tools/call":
        name = str(params.get("name", ""))
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            arguments = {}
        try:
            result = _handle_tool_call(name, arguments)
        except Exception as exc:  # noqa: BLE001 - surfaced as a tool error
            result = _tool_error_result(str(exc))
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

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
            response = _error_response(
                request.get("id"), -32000, f"Bridge error: {exc}",
            )
        if response is not None:
            _send(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
