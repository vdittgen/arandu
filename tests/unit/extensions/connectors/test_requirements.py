"""Unit tests for the RequirementChecker.

Tests use mocks for filesystem and subprocess calls to verify
requirement checking logic without needing actual macOS permissions.
"""

from __future__ import annotations

import base64
import hashlib
import json
import urllib.parse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from src.extensions.connectors.requirements import (
    OAuthResult,
    RequirementChecker,
)


@pytest.fixture()
def checker() -> RequirementChecker:
    return RequirementChecker()


# ---------------------------------------------------------------------------
# check_env_var
# ---------------------------------------------------------------------------


class TestCheckEnvVar:
    def test_env_var_from_user_inputs(
        self, checker: RequirementChecker,
    ) -> None:
        """User-provided values should satisfy env var checks."""
        result = checker.check_env_var(
            "OBSIDIAN_VAULT_PATH",
            {"OBSIDIAN_VAULT_PATH": "/Users/test/vault"},
        )
        assert result is True

    def test_env_var_empty_in_user_inputs(
        self, checker: RequirementChecker,
    ) -> None:
        """Empty string in user inputs should not satisfy."""
        result = checker.check_env_var(
            "OBSIDIAN_VAULT_PATH",
            {"OBSIDIAN_VAULT_PATH": ""},
        )
        assert result is False

    def test_env_var_missing_from_user_inputs(
        self, checker: RequirementChecker,
    ) -> None:
        """Missing key in user inputs falls through to system env."""
        with patch.dict("os.environ", {}, clear=True):
            result = checker.check_env_var("MISSING_VAR", {})
            assert result is False

    def test_env_var_from_system_env(
        self, checker: RequirementChecker,
    ) -> None:
        """System environment variables should satisfy the check."""
        with patch.dict(
            "os.environ", {"MY_VAR": "/some/path"},
        ):
            result = checker.check_env_var("MY_VAR")
            assert result is True

    def test_env_var_none_inputs(
        self, checker: RequirementChecker,
    ) -> None:
        """None user_inputs should check system env only."""
        with patch.dict("os.environ", {}, clear=True):
            result = checker.check_env_var("X", None)
            assert result is False


# ---------------------------------------------------------------------------
# check_app_installed
# ---------------------------------------------------------------------------


class TestCheckAppInstalled:
    def test_app_exists(
        self, checker: RequirementChecker,
    ) -> None:
        """Should return True when the .app directory exists."""
        with patch.object(Path, "is_dir", return_value=True):
            assert checker.check_app_installed("Safari") is True

    def test_app_not_found(
        self, checker: RequirementChecker,
    ) -> None:
        """Should return False when the .app directory is missing."""
        with patch.object(Path, "is_dir", return_value=False):
            assert checker.check_app_installed("FakeApp") is False

    def test_desktop_alias_matches_base_bundle(
        self, checker: RequirementChecker,
    ) -> None:
        """`Foo Desktop` should also match `Foo.app` bundles."""
        expected = Path("/Applications/WhatsApp.app")

        def _fake_is_dir(path: Path) -> bool:
            return path == expected

        with patch("pathlib.Path.is_dir", new=_fake_is_dir):
            assert checker.check_app_installed("WhatsApp Desktop") is True

    def test_app_found_in_user_applications(
        self, checker: RequirementChecker,
    ) -> None:
        """Should also scan ~/Applications for app bundles."""
        expected = Path.home() / "Applications" / "MyTool.app"

        def _fake_is_dir(path: Path) -> bool:
            return path == expected

        with patch("pathlib.Path.is_dir", new=_fake_is_dir):
            assert checker.check_app_installed("MyTool") is True


# ---------------------------------------------------------------------------
# check_macos_permission
# ---------------------------------------------------------------------------


class TestCheckMacosPermission:
    def test_full_disk_access_granted(
        self, checker: RequirementChecker,
    ) -> None:
        """First protected dir listable → FDA granted."""
        with patch.object(Path, "iterdir", return_value=iter([])):
            result = checker.check_macos_permission(
                "Full Disk Access",
            )
            assert result is True

    def test_full_disk_access_denied(
        self, checker: RequirementChecker,
    ) -> None:
        """PermissionError on any probe → FDA denied (definitive)."""
        with patch.object(
            Path, "iterdir", side_effect=PermissionError,
        ):
            result = checker.check_macos_permission(
                "Full Disk Access",
            )
            assert result is False

    def test_full_disk_access_falls_through_missing_paths(
        self, checker: RequirementChecker,
    ) -> None:
        """FileNotFoundError on one probe → try the next.

        Users without iMessage / Mail.app / Calendar.app on this Mac may
        have one or more of the protected directories absent. The probe
        must not interpret "doesn't exist" as "denied".
        """
        attempts: list[None] = []

        def fake_iterdir(self):  # noqa: ANN001, ARG001
            attempts.append(None)
            if len(attempts) == 1:
                raise FileNotFoundError
            return iter([])

        with patch.object(Path, "iterdir", fake_iterdir):
            result = checker.check_macos_permission(
                "Full Disk Access",
            )
            assert result is True
            assert len(attempts) == 2  # first missing, second succeeded

    def test_manual_grant_permissions_contains_full_disk_access(self) -> None:
        """The connection manager keys off this set to decide whether
        to block enable. If FDA ever drops out of it, every apple-*
        connector silently registers as connected without working."""
        from src.extensions.connectors.requirements import (
            MANUAL_GRANT_PERMISSIONS,
        )

        assert "Full Disk Access" in MANUAL_GRANT_PERMISSIONS

    def test_calendar_permission_granted(
        self, checker: RequirementChecker,
    ) -> None:
        """Calendar permission passes when automation probe succeeds."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = checker.check_macos_permission(
                "macOS Calendar",
            )
            assert result is True

    def test_calendar_permission_denied(
        self, checker: RequirementChecker,
    ) -> None:
        """Calendar permission fails when automation probe fails."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            result = checker.check_macos_permission(
                "macOS Calendar",
            )
            assert result is False

    def test_unknown_permission(
        self, checker: RequirementChecker,
    ) -> None:
        """Unknown permission names should return False."""
        result = checker.check_macos_permission(
            "Unknown Permission",
        )
        assert result is False


# ---------------------------------------------------------------------------
# check_oauth
# ---------------------------------------------------------------------------


class TestCheckOAuth:
    def test_oauth_token_exists(
        self, checker: RequirementChecker,
    ) -> None:
        """Should return True when keychain has a token."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="token123\n",
            )
            assert checker.check_oauth("google_oauth") is True

    def test_oauth_token_missing(
        self, checker: RequirementChecker,
    ) -> None:
        """Should return False when keychain has no token."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=44, stdout="",
            )
            assert checker.check_oauth("google_oauth") is False

    def test_oauth_timeout(
        self, checker: RequirementChecker,
    ) -> None:
        """Should return False on subprocess timeout."""
        import subprocess

        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired("cmd", 5),
        ):
            assert checker.check_oauth("google_oauth") is False


# ---------------------------------------------------------------------------
# check_all
# ---------------------------------------------------------------------------


class TestCheckAll:
    def test_no_requirements(
        self, checker: RequirementChecker,
    ) -> None:
        """Connector with no requirements should pass."""
        status = checker.check_all()
        assert status.all_met is True
        assert status.missing == []

    def test_missing_env_var(
        self, checker: RequirementChecker,
    ) -> None:
        """Missing env var should be reported."""
        with patch.dict("os.environ", {}, clear=True):
            status = checker.check_all(
                requires_env={"MY_VAR": "Some description"},
            )
            assert status.all_met is False
            assert len(status.missing) == 1
            assert status.missing[0].requirement_type == "env"
            assert status.missing[0].key == "MY_VAR"

    def test_missing_app(
        self, checker: RequirementChecker,
    ) -> None:
        """Missing app should be reported."""
        with patch.object(Path, "is_dir", return_value=False):
            status = checker.check_all(
                requires_app="FakeApp",
            )
            assert status.all_met is False
            assert status.missing[0].requirement_type == "app"

    def test_multiple_missing(
        self, checker: RequirementChecker,
    ) -> None:
        """Multiple missing requirements should all be reported."""
        with (
            patch.object(
                Path, "is_dir", return_value=False,
            ),
            patch.dict("os.environ", {}, clear=True),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(
                returncode=44, stdout="",
            )
            status = checker.check_all(
                requires_app="FakeApp",
                requires_auth="google_oauth",
                requires_env={"VAR": "desc"},
            )
            assert status.all_met is False
            assert len(status.missing) == 3
            types_found = {
                m.requirement_type for m in status.missing
            }
            assert types_found == {"app", "oauth", "env"}

    def test_all_met_with_user_inputs(
        self, checker: RequirementChecker,
    ) -> None:
        """All requirements met when user provides env vars."""
        with (
            patch.object(
                RequirementChecker,
                "check_macos_permission",
                return_value=True,
            ),
            patch.object(
                RequirementChecker,
                "check_oauth",
                return_value=True,
            ),
        ):
            status = checker.check_all(
                requires_permission="macOS Calendar",
                requires_auth="google_oauth",
                requires_env={"VAR": "desc"},
                user_inputs={"VAR": "/path"},
            )
            assert status.all_met is True


# ---------------------------------------------------------------------------
# OAuth helpers and start_oauth_flow
# ---------------------------------------------------------------------------


class TestOAuthFlow:
    def test_store_oauth_token_success(
        self, checker: RequirementChecker,
    ) -> None:
        """Tokens should be persisted in Keychain via `security`."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert checker.store_oauth_token("google_oauth", "token123") is True

    def test_store_oauth_token_rejects_blank(
        self, checker: RequirementChecker,
    ) -> None:
        """Blank tokens should be rejected before shelling out."""
        with patch("subprocess.run") as mock_run:
            assert checker.store_oauth_token("google_oauth", "   ") is False
            mock_run.assert_not_called()

    def test_oauth_flow_uses_test_token(
        self, checker: RequirementChecker,
    ) -> None:
        """Test token env should bypass browser/callback flow."""
        with (
            patch.dict(
                "os.environ",
                {"ARANDU_OAUTH_GOOGLE_OAUTH_TEST_TOKEN": "test-token"},
                clear=True,
            ),
            patch.object(
                RequirementChecker,
                "store_oauth_token",
                return_value=True,
            ) as mock_store,
        ):
            result = checker.start_oauth_flow("google_oauth")
            assert result.success is True
            mock_store.assert_called_once_with("google_oauth", "test-token")

    def test_oauth_flow_missing_provider_config(
        self, checker: RequirementChecker,
    ) -> None:
        """Missing auth URL configuration should return a clear error."""
        with patch.dict("os.environ", {}, clear=True):
            result = checker.start_oauth_flow("google_oauth")

        assert isinstance(result, OAuthResult)
        assert result.success is False
        assert "not configured" in (result.error or "")

    def test_oauth_flow_callback_success(
        self, checker: RequirementChecker,
    ) -> None:
        """Configured flow should accept callback token and store it."""
        state_token = "state-123"

        class FakeHttpServer:
            def __init__(self, addr, handler) -> None:  # noqa: ANN001
                self.server_port = 8765
                self.timeout = 0.5

            def handle_request(self) -> None:
                state = getattr(self, "callback_state")
                state.payload = {
                    "state": state_token,
                    "access_token": "callback-token",
                }
                state.event.set()

            def server_close(self) -> None:
                return None

        with (
            patch.dict(
                "os.environ",
                {
                    "ARANDU_OAUTH_GOOGLE_OAUTH_AUTH_URL": (
                        "https://example.com/oauth"
                    ),
                },
                clear=True,
            ),
            patch("src.extensions.connectors.requirements.HTTPServer", FakeHttpServer),
            patch(
                "src.extensions.connectors.requirements.secrets.token_urlsafe",
                return_value=state_token,
            ),
            patch(
                "src.extensions.connectors.requirements.webbrowser.open",
                return_value=True,
            ),
            patch.object(
                RequirementChecker,
                "store_oauth_token",
                return_value=True,
            ) as mock_store,
        ):
            result = checker.start_oauth_flow("google_oauth")

        assert isinstance(result, OAuthResult)
        assert result.success is True
        assert result.provider == "google_oauth"
        mock_store.assert_called_once_with("google_oauth", "callback-token")

    def test_oauth_flow_exchanges_code_with_pkce(
        self, checker: RequirementChecker,
    ) -> None:
        """A code callback + token URL must run the PKCE exchange.

        The stored value is a JSON credential (access + refresh token,
        expiry, token endpoint) rather than the raw code — storing the
        code used to "succeed" and then 401 on every API call.
        """
        state_token = "state-123"

        class FakeHttpServer:
            def __init__(self, addr, handler) -> None:  # noqa: ANN001
                self.server_port = 8765
                self.timeout = 0.5

            def handle_request(self) -> None:
                state = getattr(self, "callback_state")
                state.payload = {
                    "state": state_token,
                    "code": "auth-code-1",
                }
                state.event.set()

            def server_close(self) -> None:
                return None

        opened_urls: list[str] = []

        with (
            patch.dict(
                "os.environ",
                {
                    "ARANDU_OAUTH_GOOGLE_OAUTH_AUTH_URL": (
                        "https://example.com/oauth"
                    ),
                    "ARANDU_OAUTH_GOOGLE_OAUTH_TOKEN_URL": (
                        "https://example.com/token"
                    ),
                    "ARANDU_OAUTH_GOOGLE_OAUTH_CLIENT_ID": "cid-1",
                },
                clear=True,
            ),
            patch(
                "src.extensions.connectors.requirements.HTTPServer",
                FakeHttpServer,
            ),
            patch(
                "src.extensions.connectors.requirements.secrets.token_urlsafe",
                return_value=state_token,
            ),
            patch(
                "src.extensions.connectors.requirements.webbrowser.open",
                side_effect=lambda url: opened_urls.append(url) or True,
            ),
            patch.object(
                RequirementChecker,
                "_post_form",
                return_value={
                    "access_token": "at-1",
                    "refresh_token": "rt-1",
                    "expires_in": 3600,
                },
            ) as mock_post,
            patch.object(
                RequirementChecker,
                "store_oauth_token",
                return_value=True,
            ) as mock_store,
        ):
            result = checker.start_oauth_flow("google_oauth")

        assert result.success is True

        # The exchange used the code, the verifier, and the client id.
        (token_url, data), _ = mock_post.call_args
        assert token_url == "https://example.com/token"
        assert data["grant_type"] == "authorization_code"
        assert data["code"] == "auth-code-1"
        assert data["code_verifier"] == state_token
        assert data["client_id"] == "cid-1"

        # The browser URL carried the matching S256 challenge (PKCE).
        query = urllib.parse.parse_qs(
            urllib.parse.urlparse(opened_urls[0]).query,
        )
        expected_challenge = (
            base64.urlsafe_b64encode(
                hashlib.sha256(state_token.encode("ascii")).digest(),
            )
            .rstrip(b"=")
            .decode("ascii")
        )
        assert query["code_challenge"] == [expected_challenge]
        assert query["code_challenge_method"] == ["S256"]
        assert query["client_id"] == ["cid-1"]

        # The Keychain got a refreshable JSON credential, not the code.
        (provider, stored), _ = mock_store.call_args
        assert provider == "google_oauth"
        credential = json.loads(stored)
        assert credential["access_token"] == "at-1"
        assert credential["refresh_token"] == "rt-1"
        assert credential["token_url"] == "https://example.com/token"
        assert credential["client_id"] == "cid-1"
        assert "expires_at" in credential

    def test_oauth_flow_code_without_token_url_fails_loudly(
        self, checker: RequirementChecker,
    ) -> None:
        """An unexchangeable code must not be stored as a credential."""
        state_token = "state-123"

        class FakeHttpServer:
            def __init__(self, addr, handler) -> None:  # noqa: ANN001
                self.server_port = 8765
                self.timeout = 0.5

            def handle_request(self) -> None:
                state = getattr(self, "callback_state")
                state.payload = {
                    "state": state_token,
                    "code": "auth-code-1",
                }
                state.event.set()

            def server_close(self) -> None:
                return None

        with (
            patch.dict(
                "os.environ",
                {
                    "ARANDU_OAUTH_GOOGLE_OAUTH_AUTH_URL": (
                        "https://example.com/oauth"
                    ),
                },
                clear=True,
            ),
            patch(
                "src.extensions.connectors.requirements.HTTPServer",
                FakeHttpServer,
            ),
            patch(
                "src.extensions.connectors.requirements.secrets.token_urlsafe",
                return_value=state_token,
            ),
            patch(
                "src.extensions.connectors.requirements.webbrowser.open",
                return_value=True,
            ),
            patch.object(
                RequirementChecker,
                "store_oauth_token",
            ) as mock_store,
        ):
            result = checker.start_oauth_flow("google_oauth")

        assert result.success is False
        assert "TOKEN_URL" in (result.error or "")
        mock_store.assert_not_called()
