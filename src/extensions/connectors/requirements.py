"""Requirement checking for MCP server connectors.

Validates that system prerequisites (macOS permissions, OAuth tokens,
environment variables, installed apps) are met before enabling a connector.

sensitivity_tier: 1 (checks system state, no user data accessed)
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import logging
import os
import secrets
import subprocess
import threading
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _provider_label(provider: str) -> str:
    """Human-readable name for an OAuth provider key.

    Provider keys carry an ``_oauth`` suffix by convention
    (``spotify_oauth``, ``google_oauth``) because it becomes the Keychain
    service name. Titling the raw key surfaced "Sign in with Google
    Oauth" in the UI, so drop the suffix for display.

    sensitivity_tier: 1
    """
    stem = provider[: -len("_oauth")] if provider.endswith("_oauth") else provider
    return stem.replace("_", " ").title() or provider


@dataclass(frozen=True)
class MissingRequirement:
    """A single unmet requirement for enabling a connector.

    sensitivity_tier: 1
    """

    requirement_type: str  # "permission" | "oauth" | "env" | "app"
    key: str  # e.g. "macOS Calendar", "google_oauth", "OBSIDIAN_VAULT_PATH"
    label: str  # human-readable description
    action: str  # "grant_permission" | "start_oauth" | "provide_value" | "install_app"


@dataclass
class RequirementsStatus:
    """Result of checking all requirements for a connector.

    sensitivity_tier: 1
    """

    all_met: bool = True
    missing: list[MissingRequirement] = field(default_factory=list)

    def add_missing(self, req: MissingRequirement) -> None:
        """Record an unmet requirement.

        sensitivity_tier: 1
        """
        self.missing.append(req)
        self.all_met = False


@dataclass(frozen=True)
class OAuthResult:
    """Result of an OAuth authentication flow.

    sensitivity_tier: 2 (contains provider info but not tokens)
    """

    success: bool
    provider: str
    error: str | None = None


@dataclass
class _OAuthCallbackState:
    """Shared callback state for the temporary local OAuth server."""

    event: threading.Event
    payload: dict[str, str] = field(default_factory=dict)


class _OAuthCallbackHandler(BaseHTTPRequestHandler):
    """Capture OAuth callback query parameters and acknowledge in browser."""

    server: HTTPServer

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        payload: dict[str, str] = {}
        for key, values in urllib.parse.parse_qs(parsed.query).items():
            if values:
                payload[key] = values[0]
        state: _OAuthCallbackState | None = getattr(
            self.server, "callback_state", None,
        )
        if state is not None:
            state.payload = payload
            state.event.set()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        msg = b"Arandu authentication complete. You can close this window."
        self.wfile.write(msg)

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        # Keep OAuth callback server quiet in normal logs.
        return


# Permissions that cannot be requested via a runtime macOS dialog and must
# be granted manually by the user in System Settings. The connection
# manager treats these as hard preconditions — enabling a connector that
# needs one without it granted returns ``needs_setup`` instead of starting
# the MCP server (which would silently fail on the read path).
MANUAL_GRANT_PERMISSIONS: frozenset[str] = frozenset({"Full Disk Access"})


class RequirementChecker:
    """Check and request system prerequisites for connectors.

    sensitivity_tier: 1
    """

    def check_macos_permission(self, permission: str) -> bool:
        """Check if the app has a macOS permission.

        Supported permissions:
        - "Full Disk Access": probes access to FDA-protected directories
          (~/Library/Messages, ~/Library/Mail, ~/Library/Calendars). Needed
          by the apple-* connectors, which read SQLite databases directly.
        - "macOS Calendar" / "macOS Contacts" / "macOS Notes" / "macOS Mail":
          AppleScript Automation access. Used by write paths only — none of
          the apple-* read-only connectors require these today, but the
          checker still supports them for future connectors that script the
          host app instead of reading SQLite.

        sensitivity_tier: 1
        """
        if permission == "Full Disk Access":
            return self._check_full_disk_access()

        automation_targets: dict[str, str] = {
            "macOS Calendar": "Calendar",
            "macOS Contacts": "Contacts",
            "macOS Notes": "Notes",
            "macOS Mail": "Mail",
        }
        app_name = automation_targets.get(permission)
        if app_name is None:
            logger.warning("Unknown permission: %s — assuming not granted", permission)
            return False
        return self._check_automation_access(app_name)

    def _check_full_disk_access(self) -> bool:
        """Probe FDA by trying to list known protected directories.

        Tries multiple paths so the probe survives macOS installs where
        one of them is absent (e.g. iMessage never configured, Mail.app
        never launched). ``PermissionError`` from any probe is a
        definitive "FDA not granted"; ``FileNotFoundError`` just means
        the user doesn't have that app set up — fall through to the
        next candidate.

        sensitivity_tier: 1
        """
        probes = (
            Path.home() / "Library" / "Messages",
            Path.home() / "Library" / "Mail",
            Path.home() / "Library" / "Calendars",
        )
        for probe in probes:
            try:
                next(iter(probe.iterdir()), None)
                return True
            except PermissionError:
                return False
            except (FileNotFoundError, NotADirectoryError, OSError):
                continue
        return False

    def _check_automation_access(self, app_name: str) -> bool:
        """Check AppleScript automation access to a target macOS app.

        sensitivity_tier: 1
        """
        script = f'tell application "{app_name}" to return name'
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def check_oauth(self, provider: str) -> bool:
        """Check if we have a stored OAuth token for the given provider.

        Checks the macOS Keychain for a stored token.

        sensitivity_tier: 1
        """
        service_name = self._oauth_service_name(provider)
        try:
            result = subprocess.run(
                [
                    "security",
                    "find-generic-password",
                    "-s",
                    service_name,
                    "-w",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0 and bool(result.stdout.strip())
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def store_oauth_token(self, provider: str, token: str) -> bool:
        """Store an OAuth token securely in macOS Keychain.

        sensitivity_tier: 2
        """
        token_value = token.strip()
        if not token_value:
            return False

        service_name = self._oauth_service_name(provider)
        try:
            result = subprocess.run(
                [
                    "security",
                    "add-generic-password",
                    "-U",
                    "-s",
                    service_name,
                    "-a",
                    "arandu",
                    "-w",
                    token_value,
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def check_env_var(
        self,
        var_name: str,
        user_inputs: dict[str, Any] | None = None,
    ) -> bool:
        """Check if an env var is provided by the user or exists in system.

        sensitivity_tier: 1
        """
        if user_inputs and var_name in user_inputs:
            value = user_inputs[var_name]
            return bool(value) and str(value).strip() != ""
        return var_name in os.environ and bool(os.environ[var_name].strip())

    def check_app_installed(self, app_name: str) -> bool:
        """Check if a macOS app is installed.

        sensitivity_tier: 1
        """
        app_roots = [
            Path("/Applications"),
            Path.home() / "Applications",
            Path("/System/Applications"),
        ]

        variants = self._app_name_variants(app_name)
        candidates: list[Path] = []
        for root in app_roots:
            for variant in variants:
                candidates.append(root / f"{variant}.app")
                compact = variant.replace(" ", "")
                if compact != variant:
                    candidates.append(root / f"{compact}.app")
        return any(path.is_dir() for path in candidates)

    @staticmethod
    def _app_name_variants(app_name: str) -> list[str]:
        """Return normalized app-name variants for bundle lookup."""
        name = app_name.strip()
        if not name:
            return []

        variants: list[str] = [name]
        for suffix in (" Desktop", " App", " Application"):
            if name.endswith(suffix):
                base = name[: -len(suffix)].strip()
                if base:
                    variants.append(base)

        # Preserve order while removing duplicates.
        seen: set[str] = set()
        deduped: list[str] = []
        for variant in variants:
            if variant not in seen:
                seen.add(variant)
                deduped.append(variant)
        return deduped

    def check_all(
        self,
        *,
        requires_permission: str | None = None,
        requires_auth: str | None = None,
        requires_env: dict[str, str] | None = None,
        requires_app: str | None = None,
        user_inputs: dict[str, Any] | None = None,
    ) -> RequirementsStatus:
        """Check all requirements for a connector template.

        Args:
            requires_permission: macOS permission name.
            requires_auth: OAuth provider name.
            requires_env: Map of env var name to description.
            requires_app: macOS app name.
            user_inputs: User-provided values (env vars, etc.).

        sensitivity_tier: 1
        """
        status = RequirementsStatus()

        if requires_permission:
            if not self.check_macos_permission(requires_permission):
                status.add_missing(
                    MissingRequirement(
                        requirement_type="permission",
                        key=requires_permission,
                        label=f"Grant {requires_permission} access",
                        action="grant_permission",
                    )
                )

        if requires_auth:
            if not self.check_oauth(requires_auth):
                status.add_missing(
                    MissingRequirement(
                        requirement_type="oauth",
                        key=requires_auth,
                        label=f"Sign in with {_provider_label(requires_auth)}",
                        action="start_oauth",
                    )
                )

        if requires_env:
            for var_name, description in requires_env.items():
                if not self.check_env_var(var_name, user_inputs):
                    status.add_missing(
                        MissingRequirement(
                            requirement_type="env",
                            key=var_name,
                            label=description,
                            action="provide_value",
                        )
                    )

        if requires_app:
            if not self.check_app_installed(requires_app):
                status.add_missing(
                    MissingRequirement(
                        requirement_type="app",
                        key=requires_app,
                        label=f"Install {requires_app}",
                        action="install_app",
                    )
                )

        return status

    def request_macos_permission(self, permission: str) -> bool:
        """Trigger the macOS permission dialog.

        Opens System Settings to the relevant pane so the user can
        grant permission manually. Returns True if permission is
        subsequently granted (re-checked after a delay).

        sensitivity_tier: 1
        """
        # Map permissions to System Settings deep links
        pane_map: dict[str, str] = {
            "macOS Calendar": (
                "x-apple.systempreferences:"
                "com.apple.preference.security"
                "?Privacy_Automation"
            ),
            "macOS Contacts": (
                "x-apple.systempreferences:"
                "com.apple.preference.security"
                "?Privacy_Automation"
            ),
            "Full Disk Access": (
                "x-apple.systempreferences:"
                "com.apple.preference.security"
                "?Privacy_AllFiles"
            ),
            "macOS Notes": (
                "x-apple.systempreferences:"
                "com.apple.preference.security"
                "?Privacy_Automation"
            ),
            "macOS Mail": (
                "x-apple.systempreferences:"
                "com.apple.preference.security"
                "?Privacy_Automation"
            ),
        }

        url = pane_map.get(permission)
        if url:
            try:
                subprocess.run(
                    ["open", url],
                    check=False,
                    timeout=5,
                )
            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.warning("Could not open System Settings for %s", permission)
                return False

        # Re-check after opening the dialog
        return self.check_macos_permission(permission)

    def start_oauth_flow(self, provider: str) -> OAuthResult:
        """Run a local OAuth browser callback flow and store the token.

        Provider configuration is expected in environment variables:
        - `ARANDU_OAUTH_<PROVIDER>_AUTH_URL` (required)
        - `ARANDU_OAUTH_<PROVIDER>_TOKEN_URL` (required for real
          authorization-code providers): enables the PKCE code->token
          exchange. Without it, the loopback callback can only succeed
          when the provider hands a token directly in query parameters
          — an authorization code on its own is *not* a credential, and
          storing one produces a connector that fails on every call.
        - `ARANDU_OAUTH_<PROVIDER>_CLIENT_ID` / `_CLIENT_SECRET`
          (optional): sent on the exchange. Native-app client secrets
          (e.g. Google "Desktop app" clients) are not treated as
          confidential by the provider, but still come from the
          operator's environment — never from the repo.
        - `ARANDU_OAUTH_<PROVIDER>_AUTH_PARAMS` (optional JSON map)
        - `ARANDU_OAUTH_<PROVIDER>_TEST_TOKEN` (optional shortcut for tests)

        When the exchange runs, the Keychain entry holds a JSON
        credential (access token, refresh token, expiry, token URL,
        client id/secret) so bridges can refresh expired access tokens
        themselves; otherwise it holds the bare token, as before.

        sensitivity_tier: 2
        """
        logger.info("Starting OAuth flow for provider: %s", provider)
        provider_key = self._provider_env_key(provider)
        auth_url_key = f"ARANDU_OAUTH_{provider_key}_AUTH_URL"
        auth_params_key = f"ARANDU_OAUTH_{provider_key}_AUTH_PARAMS"
        test_token_key = f"ARANDU_OAUTH_{provider_key}_TEST_TOKEN"
        token_url_key = f"ARANDU_OAUTH_{provider_key}_TOKEN_URL"
        client_id_key = f"ARANDU_OAUTH_{provider_key}_CLIENT_ID"
        client_secret_key = f"ARANDU_OAUTH_{provider_key}_CLIENT_SECRET"

        test_token = os.environ.get(test_token_key, "").strip()
        if test_token:
            if self.store_oauth_token(provider, test_token):
                return OAuthResult(success=True, provider=provider)
            return OAuthResult(
                success=False,
                provider=provider,
                error="Failed to store OAuth token in Keychain",
            )

        auth_url = os.environ.get(auth_url_key, "").strip()
        if not auth_url:
            return OAuthResult(
                success=False,
                provider=provider,
                error=f"OAuth provider '{provider}' is not configured",
            )

        token_url = os.environ.get(token_url_key, "").strip()
        client_id = os.environ.get(client_id_key, "").strip()
        client_secret = os.environ.get(client_secret_key, "").strip()

        callback_state = _OAuthCallbackState(event=threading.Event())
        server = HTTPServer(("127.0.0.1", 0), _OAuthCallbackHandler)
        setattr(server, "callback_state", callback_state)
        server.timeout = 0.5

        stop_server = threading.Event()

        def _serve() -> None:
            while not stop_server.is_set() and not callback_state.event.is_set():
                server.handle_request()

        server_thread = threading.Thread(target=_serve, daemon=True)
        server_thread.start()

        redirect_uri = f"http://127.0.0.1:{server.server_port}/callback"
        state_token = secrets.token_urlsafe(24)
        params: dict[str, str] = {
            "redirect_uri": redirect_uri,
            "state": state_token,
            "response_type": "code",
        }
        if client_id:
            params["client_id"] = client_id
        # PKCE (RFC 7636) whenever a token endpoint is configured: lets
        # native apps do the code exchange without a confidential
        # client secret, and binds the code to this flow instance.
        code_verifier = ""
        if token_url:
            code_verifier = secrets.token_urlsafe(64)
            challenge = (
                base64.urlsafe_b64encode(
                    hashlib.sha256(code_verifier.encode("ascii")).digest(),
                )
                .rstrip(b"=")
                .decode("ascii")
            )
            params["code_challenge"] = challenge
            params["code_challenge_method"] = "S256"
        params.update(self._load_auth_params(auth_params_key))
        oauth_url = self._append_query_params(auth_url, params)

        try:
            opened = webbrowser.open(oauth_url)
            if not opened:
                subprocess.run(["open", oauth_url], check=False, timeout=5)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            stop_server.set()
            with contextlib.suppress(Exception):
                server.server_close()
            server_thread.join(timeout=1.0)
            return OAuthResult(
                success=False,
                provider=provider,
                error="Could not open browser for OAuth",
            )

        timeout_s = int(os.environ.get("ARANDU_OAUTH_CALLBACK_TIMEOUT", "180"))
        completed = callback_state.event.wait(timeout=timeout_s)
        stop_server.set()
        with contextlib.suppress(Exception):
            server.server_close()
        server_thread.join(timeout=1.0)

        if not completed:
            return OAuthResult(
                success=False,
                provider=provider,
                error=f"OAuth callback timed out after {timeout_s}s",
            )

        payload = callback_state.payload
        if payload.get("state") != state_token:
            return OAuthResult(
                success=False,
                provider=provider,
                error="OAuth state mismatch",
            )
        if "error" in payload:
            return OAuthResult(
                success=False,
                provider=provider,
                error=payload.get("error_description", payload["error"]),
            )

        code = payload.get("code", "")
        if token_url and code:
            exchanged = self._exchange_code_for_tokens(
                token_url=token_url,
                code=code,
                redirect_uri=redirect_uri,
                code_verifier=code_verifier,
                client_id=client_id,
                client_secret=client_secret,
            )
            error = exchanged.get("error")
            if error:
                return OAuthResult(
                    success=False,
                    provider=provider,
                    error=str(error),
                )
            credential = json.dumps(exchanged["credential"])
            if not self.store_oauth_token(provider, credential):
                return OAuthResult(
                    success=False,
                    provider=provider,
                    error="Failed to store OAuth credential in Keychain",
                )
            return OAuthResult(success=True, provider=provider)

        token = payload.get("access_token") or payload.get("token")
        if not token:
            if code:
                # Storing the bare authorization code used to "succeed"
                # here — and then every API call failed with 401,
                # because a code is not a credential. Fail loudly with
                # the missing piece instead.
                return OAuthResult(
                    success=False,
                    provider=provider,
                    error=(
                        "OAuth returned an authorization code but "
                        f"{token_url_key} is not set, so it cannot be "
                        "exchanged for an access token"
                    ),
                )
            return OAuthResult(
                success=False,
                provider=provider,
                error="OAuth callback missing access token",
            )

        if not self.store_oauth_token(provider, token):
            return OAuthResult(
                success=False,
                provider=provider,
                error="Failed to store OAuth token in Keychain",
            )

        return OAuthResult(
            success=True,
            provider=provider,
        )

    def _exchange_code_for_tokens(
        self,
        *,
        token_url: str,
        code: str,
        redirect_uri: str,
        code_verifier: str,
        client_id: str,
        client_secret: str,
    ) -> dict[str, Any]:
        """Exchange an authorization code for tokens (RFC 6749 §4.1.3).

        Returns ``{"credential": {...}}`` on success or
        ``{"error": str}`` on failure. The credential dict carries
        everything a bridge needs to refresh the access token later.

        sensitivity_tier: 2 (handles tokens)
        """
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "code_verifier": code_verifier,
        }
        if client_id:
            data["client_id"] = client_id
        if client_secret:
            data["client_secret"] = client_secret
        try:
            payload = self._post_form(token_url, data)
        except urllib.error.HTTPError as exc:
            detail = ""
            with contextlib.suppress(Exception):
                detail = exc.read().decode("utf-8", errors="replace")[:200]
            return {
                "error": (
                    f"Token exchange failed (HTTP {exc.code}): {detail}"
                ),
            }
        except (urllib.error.URLError, OSError, ValueError) as exc:
            return {"error": f"Token exchange failed: {exc}"}

        access_token = str(payload.get("access_token") or "")
        if not access_token:
            return {"error": "Token endpoint returned no access_token"}

        credential: dict[str, Any] = {
            "access_token": access_token,
            "token_url": token_url,
        }
        if payload.get("refresh_token"):
            credential["refresh_token"] = str(payload["refresh_token"])
        expires_in = payload.get("expires_in")
        if expires_in is not None:
            with contextlib.suppress(TypeError, ValueError):
                credential["expires_at"] = (
                    datetime.now(tz=timezone.utc)
                    + timedelta(seconds=int(expires_in))
                ).isoformat()
        if client_id:
            credential["client_id"] = client_id
        if client_secret:
            credential["client_secret"] = client_secret
        return {"credential": credential}

    @staticmethod
    def _post_form(url: str, data: dict[str, str]) -> dict[str, Any]:
        """POST a urlencoded form and decode the JSON response.

        sensitivity_tier: 2 (carries tokens)
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
        with urllib.request.urlopen(  # noqa: S310
            request, timeout=30,
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            msg = "Token endpoint response was not a JSON object"
            raise ValueError(msg)
        return payload

    @staticmethod
    def _oauth_service_name(provider: str) -> str:
        """Return Keychain service name for the OAuth provider."""
        return f"arandu-oauth-{provider}"

    @staticmethod
    def _provider_env_key(provider: str) -> str:
        """Normalize provider name for environment variable keys."""
        out = "".join(
            c if c.isalnum() else "_"
            for c in provider.upper()
        ).strip("_")
        return out or "PROVIDER"

    @staticmethod
    def _load_auth_params(env_key: str) -> dict[str, str]:
        """Load optional auth query params from JSON env var."""
        raw = os.environ.get(env_key, "").strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Invalid OAuth params in %s", env_key)
            return {}
        if not isinstance(parsed, dict):
            logger.warning("OAuth params env %s must be a JSON object", env_key)
            return {}
        return {
            str(k): str(v)
            for k, v in parsed.items()
            if v is not None
        }

    @staticmethod
    def _append_query_params(base_url: str, params: dict[str, str]) -> str:
        """Append query parameters to a URL preserving existing values."""
        parsed = urllib.parse.urlparse(base_url)
        existing = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
        for key, value in params.items():
            existing[key] = [value]
        query = urllib.parse.urlencode(existing, doseq=True)
        return urllib.parse.urlunparse(parsed._replace(query=query))
