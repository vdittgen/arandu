"""Tests for the Google Calendar + Gmail MCP bridge.

The bridge is driven against a stub of `_api_get` rather than the live
Google APIs, so these run in CI with no Google account, no OAuth client
and no network. The token is supplied through the documented
`ARANDU_GOOGLE_OAUTH_TOKEN` seam so the Keychain is never touched.

What's worth pinning here is the *mapping*: every row the bridge emits
has to line up with the canonical `raw_calendar_events` / `raw_emails`
columns declared in `catalog_data.json`, because the ingestion adapter
matches on those names and silently drops anything else.

sensitivity_tier: 1 (synthetic fixtures only)
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from src.extensions.bridges.google import server
from src.extensions.connectors.catalog import ConnectorCatalog


def _declared_sources(connector_id: str, tool_name: str) -> set[str]:
    """Field names the catalog declares for one connector tool.

    Read through the catalog API rather than the JSON file so the test
    can't drift from however the catalog is loaded, and so a malformed
    entry fails here too.
    """
    entry = ConnectorCatalog().get(connector_id)
    assert entry is not None, f"{connector_id} missing from the catalog"
    tool = next(t for t in entry.tools if t.tool_name == tool_name)
    # `id` is the dedup key, carried on every row but not a mapped field.
    return {f.source_name for f in tool.fields} | {"id"}


@pytest.fixture(autouse=True)
def _token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Supply a token via the env seam so no Keychain call happens."""
    monkeypatch.setenv(server.TOKEN_ENV_VAR, "test-token")


def _stub_api(
    monkeypatch: pytest.MonkeyPatch,
    responses: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """Route `_api_get` to canned payloads, recording every call.

    Keys are matched as substrings of the request path, longest first, so
    a message-detail path wins over the message-list path.
    """
    calls: list[tuple[str, dict[str, Any]]] = []

    def _fake(path: str, params: dict[str, Any], token: str) -> dict[str, Any]:
        assert token == "test-token"
        calls.append((path, params))
        for key in sorted(responses, key=len, reverse=True):
            if key in path:
                return responses[key]
        raise AssertionError(f"unexpected path: {path}")

    monkeypatch.setattr(server, "_api_get", _fake)
    return calls


# --------------------------------------------------------------- calendar


CAL_EVENT = {
    "id": "evt-1",
    "status": "confirmed",
    "summary": "Design Review",
    "description": "Quarterly review",
    "location": "Room 4",
    "start": {"dateTime": "2026-08-03T14:00:00+00:00"},
    "end": {"dateTime": "2026-08-03T15:00:00+00:00"},
    "organizer": {"email": "boss@example.com"},
    "attendees": [
        {"email": "me@example.com", "self": True, "responseStatus": "accepted"},
        {"email": "ana@example.com", "displayName": "Ana Costa"},
    ],
}


def test_calendar_rows_match_canonical_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Emitted keys must be a subset of the catalog's declared columns.

    The adapter maps by source_name; a typo here would be silently
    dropped rather than raising, so assert the contract directly.
    """
    _stub_api(monkeypatch, {"/calendar/v3/": {"items": [CAL_EVENT]}})
    rows = server.list_calendar_events({"limit": 10})

    assert len(rows) == 1
    declared = _declared_sources("google-calendar", "list_calendar_events")
    assert set(rows[0]) <= declared, set(rows[0]) - declared


def test_calendar_maps_fields_and_rsvp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_api(monkeypatch, {"/calendar/v3/": {"items": [CAL_EVENT]}})
    row = server.list_calendar_events({})[0]

    assert row["id"] == "gcal:evt-1"
    assert row["title"] == "Design Review"
    assert row["start_time"] == "2026-08-03T14:00:00+00:00"
    assert row["location"] == "Room 4"
    # displayName preferred, email as fallback — and a list, since the
    # column applies the json_array transform.
    assert row["attendees"] == ["me@example.com", "Ana Costa"]
    assert row["self_response_status"] == "accepted"
    # organizer.self is absent, and there are attendees → invited.
    assert row["event_origin"] == "invited"
    assert row["is_all_day"] is False


def test_calendar_detects_all_day_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All-day events use `date`; a midnight `dateTime` must not count.

    Inferring all-day from a 00:00 timestamp would mislabel a legitimate
    midnight event, so the flag comes from which key is present.
    """
    all_day = {
        **CAL_EVENT,
        "start": {"date": "2026-08-03"},
        "end": {"date": "2026-08-04"},
    }
    midnight = {
        **CAL_EVENT,
        "id": "evt-2",
        "start": {"dateTime": "2026-08-03T00:00:00+00:00"},
        "end": {"dateTime": "2026-08-03T01:00:00+00:00"},
    }
    _stub_api(monkeypatch, {"/calendar/v3/": {"items": [all_day, midnight]}})
    rows = server.list_calendar_events({})

    assert rows[0]["is_all_day"] is True
    assert rows[1]["is_all_day"] is False


def test_calendar_skips_cancelled_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancelled instances still come back from the API — drop them."""
    cancelled = {**CAL_EVENT, "id": "evt-x", "status": "cancelled"}
    _stub_api(monkeypatch, {"/calendar/v3/": {"items": [cancelled, CAL_EVENT]}})

    rows = server.list_calendar_events({})
    assert [r["id"] for r in rows] == ["gcal:evt-1"]


def test_calendar_expands_recurring_series(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`singleEvents=true` — otherwise a weekly standup is one master row.

    Nothing downstream interprets recurrence rules, so the expansion has
    to happen at the API.
    """
    calls = _stub_api(monkeypatch, {"/calendar/v3/": {"items": []}})
    server.list_calendar_events({})

    _, params = calls[0]
    assert params["singleEvents"] == "true"
    assert params["orderBy"] == "startTime"


# ------------------------------------------------------------------ gmail


def _gmail_detail(msg_id: str = "m1") -> dict[str, Any]:
    return {
        "id": msg_id,
        "snippet": "Can you confirm Thursday?",
        "labelIds": ["INBOX", "UNREAD"],
        "payload": {
            "headers": [
                {"name": "Subject", "value": "Thursday?"},
                {"name": "From", "value": "Ana <ana@example.com>"},
                {"name": "To", "value": "me@example.com, team@example.com"},
                {"name": "Date", "value": "Mon, 3 Aug 2026 09:15:00 +0000"},
            ],
        },
    }


def test_gmail_rows_match_canonical_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_api(
        monkeypatch,
        {
            "/gmail/v1/users/me/messages/": _gmail_detail(),
            "/gmail/v1/users/me/messages": {"messages": [{"id": "m1"}]},
        },
    )
    rows = server.list_emails({})

    declared = _declared_sources("google-gmail", "list_emails")
    assert set(rows[0]) <= declared, set(rows[0]) - declared


def test_gmail_maps_headers_labels_and_recipients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_api(
        monkeypatch,
        {
            "/gmail/v1/users/me/messages/": _gmail_detail(),
            "/gmail/v1/users/me/messages": {"messages": [{"id": "m1"}]},
        },
    )
    row = server.list_emails({})[0]

    assert row["id"] == "gmail:m1"
    assert row["subject"] == "Thursday?"
    assert row["from_address"] == "Ana <ana@example.com>"
    # json_array column, so the To header must be split per recipient.
    assert row["to_addresses"] == ["me@example.com", "team@example.com"]
    # RFC 2822 header normalised to ISO for the iso_to_timestamp transform.
    assert row["date"].startswith("2026-08-03T09:15:00")
    assert row["body_preview"] == "Can you confirm Thursday?"
    assert row["folder"] == "INBOX"
    assert row["is_read"] is False


def test_gmail_sent_mail_lands_in_sent_folder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sent mail must carry a folder the reply sweep recognises.

    `sweep_resolved_pending_replies` resolves a Gmail pending reply by
    finding a later row with ``LOWER(folder) LIKE '%sent%'`` addressed
    to the same correspondent — the convention the Apple Mail bridge
    established. Mapping sent mail to ARCHIVE silently breaks that.
    """
    detail = _gmail_detail()
    detail["labelIds"] = ["SENT"]
    _stub_api(
        monkeypatch,
        {
            "/gmail/v1/users/me/messages/": detail,
            "/gmail/v1/users/me/messages": {"messages": [{"id": "m1"}]},
        },
    )
    row = server.list_emails({})[0]

    assert "sent" in row["folder"].lower()


def test_gmail_drafts_are_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drafts are not correspondence and must not become email rows."""
    draft = _gmail_detail("d1")
    draft["labelIds"] = ["DRAFT"]
    _stub_api(
        monkeypatch,
        {
            "/gmail/v1/users/me/messages/d1": draft,
            "/gmail/v1/users/me/messages/m1": _gmail_detail(),
            "/gmail/v1/users/me/messages": {
                "messages": [{"id": "d1"}, {"id": "m1"}],
            },
        },
    )
    rows = server.list_emails({})

    assert [r["id"] for r in rows] == ["gmail:m1"]


def test_gmail_quoted_display_name_with_comma(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A comma inside a quoted display name is not a recipient split."""
    detail = _gmail_detail()
    detail["payload"]["headers"] = [
        h for h in detail["payload"]["headers"] if h["name"] != "To"
    ] + [{
        "name": "To",
        "value": '"Doe, John" <jdoe@example.com>, ana@example.com',
    }]
    _stub_api(
        monkeypatch,
        {
            "/gmail/v1/users/me/messages/": detail,
            "/gmail/v1/users/me/messages": {"messages": [{"id": "m1"}]},
        },
    )
    row = server.list_emails({})[0]

    assert row["to_addresses"] == ["jdoe@example.com", "ana@example.com"]


def test_gmail_never_requests_message_bodies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Privacy invariant: metadata only.

    `format=full`/`raw` would pull the most sensitive content on the
    account into local storage, and `raw_emails.body_preview` only ever
    needs a preview.
    """
    calls = _stub_api(
        monkeypatch,
        {
            "/gmail/v1/users/me/messages/": _gmail_detail(),
            "/gmail/v1/users/me/messages": {"messages": [{"id": "m1"}]},
        },
    )
    server.list_emails({})

    detail_calls = [
        (p, params) for p, params in calls if p.rstrip("/").endswith("/m1")
    ]
    assert detail_calls, "expected a message-detail fetch"
    for _, params in detail_calls:
        assert params["format"] == "metadata"
        assert set(params["metadataHeaders"]) == set(server._GMAIL_HEADERS)


def test_gmail_from_date_becomes_gmail_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gmail's `after:` wants YYYY/MM/DD, not ISO."""
    calls = _stub_api(
        monkeypatch, {"/gmail/v1/users/me/messages": {"messages": []}},
    )
    server.list_emails({"fromDate": "2026-07-01T00:00:00+00:00"})

    _, params = calls[0]
    assert params["q"] == "after:2026/07/01"


# ------------------------------------------------------------------- auth


def test_legacy_bare_token_still_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-exchange installs hold a bare token — must keep working."""
    monkeypatch.delenv(server.TOKEN_ENV_VAR, raising=False)
    monkeypatch.setattr(server, "_keychain_read", lambda: "legacy-token")

    assert server._access_token() == "legacy-token"


def test_expired_json_credential_refreshes_and_persists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An expired access token refreshes proactively and is written back.

    Google access tokens live ~1h; without this the hourly Gmail sync
    would 401 on every cycle after the first.
    """
    monkeypatch.delenv(server.TOKEN_ENV_VAR, raising=False)
    stored = {
        "access_token": "old-token",
        "refresh_token": "rt-1",
        "expires_at": "2020-01-01T00:00:00+00:00",
        "token_url": "https://oauth2.example/token",
        "client_id": "cid-1",
    }
    monkeypatch.setattr(
        server, "_keychain_read", lambda: json.dumps(stored),
    )
    writes: list[str] = []
    monkeypatch.setattr(server, "_keychain_write", writes.append)
    posts: list[tuple[str, dict[str, str]]] = []

    def _fake_post(url: str, data: dict[str, str]) -> dict[str, Any]:
        posts.append((url, data))
        return {"access_token": "new-token", "expires_in": 3600}

    monkeypatch.setattr(server, "_post_form", _fake_post)

    assert server._access_token() == "new-token"

    url, data = posts[0]
    assert url == stored["token_url"]
    assert data["grant_type"] == "refresh_token"
    assert data["refresh_token"] == "rt-1"
    assert data["client_id"] == "cid-1"

    persisted = json.loads(writes[0])
    assert persisted["access_token"] == "new-token"
    # Google usually omits the refresh token on refresh — keep the old.
    assert persisted["refresh_token"] == "rt-1"
    assert persisted["token_url"] == stored["token_url"]


def test_auth_failure_mid_call_refreshes_once_and_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 401 mid-call refreshes the credential and retries the tool.

    Covers the token expiring between the proactive expiry check and
    the request (or an install whose credential has no expiry stamp).
    """
    monkeypatch.delenv(server.TOKEN_ENV_VAR, raising=False)
    box = {
        "raw": json.dumps({
            "access_token": "old-token",
            "refresh_token": "rt-1",
            "token_url": "https://oauth2.example/token",
        }),
    }
    monkeypatch.setattr(server, "_keychain_read", lambda: box["raw"])
    monkeypatch.setattr(
        server, "_keychain_write", lambda v: box.update(raw=v),
    )
    monkeypatch.setattr(
        server,
        "_post_form",
        lambda url, data: {"access_token": "new-token"},
    )

    tokens_seen: list[str] = []

    def _fake_api(
        path: str, params: dict[str, Any], token: str,
    ) -> dict[str, Any]:
        tokens_seen.append(token)
        if token == "old-token":
            msg = "Google rejected the stored credential (HTTP 401)."
            raise server.GoogleAuthError(msg)
        return {"messages": []}

    monkeypatch.setattr(server, "_api_get", _fake_api)

    result = server._handle_tool_call("list_emails", {})

    assert result["isError"] is False
    assert tokens_seen[0] == "old-token"
    assert tokens_seen[-1] == "new-token"


def test_unrefreshable_credential_surfaces_reauthorize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy bare tokens can't refresh — the 401 must surface as-is."""
    monkeypatch.delenv(server.TOKEN_ENV_VAR, raising=False)
    monkeypatch.setattr(server, "_keychain_read", lambda: "legacy-token")

    def _fake_api(
        path: str, params: dict[str, Any], token: str,
    ) -> dict[str, Any]:
        msg = "Google rejected the stored credential (HTTP 401)."
        raise server.GoogleAuthError(msg)

    monkeypatch.setattr(server, "_api_get", _fake_api)

    with pytest.raises(server.GoogleAuthError, match="401"):
        server._handle_tool_call("list_emails", {})


# ------------------------------------------------------------------- misc


def test_limit_is_clamped() -> None:
    assert server._normalize_limit({"limit": 10_000}) == server.MAX_LIMIT
    assert server._normalize_limit({"limit": 0}) == 1
    assert server._normalize_limit({"limit": "nonsense"}) == server.DEFAULT_LIMIT
    assert server._normalize_limit({}) == server.DEFAULT_LIMIT


def test_missing_token_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A disconnected account must say what to do, not leak a stack trace."""
    monkeypatch.delenv(server.TOKEN_ENV_VAR, raising=False)
    monkeypatch.setattr(
        server.subprocess,
        "run",
        lambda *a, **k: type("R", (), {"returncode": 1, "stdout": ""})(),
    )
    with pytest.raises(server.GoogleAuthError, match="not connected"):
        server.list_emails({})


def test_tool_errors_are_returned_not_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An auth failure must come back as an MCP tool error.

    Raising out of the dispatch loop would kill the bridge process and
    the sync engine would see a dead server instead of a message it can
    surface to the user.
    """
    def _boom(_args: dict[str, Any]) -> list[dict[str, Any]]:
        raise server.GoogleAuthError("re-authorize Google")

    monkeypatch.setattr(server, "list_emails", _boom)
    result = server._handle_request({
        "jsonrpc": "2.0",
        "id": 7,
        "method": "tools/call",
        "params": {"name": "list_emails", "arguments": {}},
    })

    assert result is not None
    assert result["result"]["isError"] is True
    assert "re-authorize" in result["result"]["content"][0]["text"]


def test_tools_list_advertises_both_data_tools() -> None:
    """The catalog's tool_names must exist on the server."""
    response = server._handle_request(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
    )
    assert response is not None
    names = {t["name"] for t in response["result"]["tools"]}

    catalog = ConnectorCatalog()
    for cid in ("google-calendar", "google-gmail"):
        entry = catalog.get(cid)
        assert entry is not None
        for tool in entry.tools:
            assert tool.tool_name in names
