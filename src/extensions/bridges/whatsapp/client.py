"""Python wrapper for the custom Baileys WhatsApp client subprocess.

Manages a Node.js child process (``client.js``) and communicates via JSONL
over stdio.  All user-data transit is text messages at sensitivity tier 2-3.

sensitivity_tier: 1 (protocol wrapper — no user data held in class state)
"""

from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import subprocess
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PACKAGE_DIR = Path(__file__).resolve().parent / "node"
_CLIENT_SCRIPT = _PACKAGE_DIR / "client.js"

# Well-known install dirs probed when the PATH lookup fails. An app
# launched from Finder gets launchd's minimal GUI PATH
# (/usr/bin:/bin:...), which never contains Homebrew or version-manager
# installs — without the fallback the listener dies with "node not
# found" on every tick even though Node is present on the machine.
_TOOL_FALLBACK_DIRS: tuple[Path, ...] = (
    Path("/opt/homebrew/bin"),  # Homebrew (Apple Silicon)
    Path("/usr/local/bin"),  # Homebrew (Intel) / manual installs
    Path.home() / ".asdf" / "shims",
    Path.home() / ".volta" / "bin",
)


def _version_key(name: str) -> tuple[int, ...]:
    """Numeric sort key for an nvm version dir name like ``v20.11.0``.

    Lexicographic order would rank ``v9...`` above ``v20...``.

    sensitivity_tier: 1
    """
    parts: list[int] = []
    for part in name.lstrip("v").split("."):
        try:
            parts.append(int(part))
        except ValueError:
            parts.append(0)
    return tuple(parts)


def _runs(candidate: str) -> bool:
    """True if ``candidate --version`` executes successfully.

    A binary can exist yet be unrunnable — e.g. a Homebrew node linked
    against a since-upgraded icu4c dylib aborts at load time (SIGABRT)
    on every spawn. Candidates must prove they execute before being
    chosen, so a broken install is skipped in favour of the next
    working one.

    sensitivity_tier: 1
    """
    try:
        result = subprocess.run(
            [candidate, "--version"],
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _node_has_webcrypto(candidate: str) -> bool:
    """True if ``candidate`` exposes ``globalThis.crypto.subtle``.

    Baileys >=7 reads ``globalThis.crypto.subtle`` at import time and
    crashes immediately (exit code 1, no usable message) without it.
    That global is unavailable on Node < 19 (< 18 behind a flag). A
    candidate can pass ``_runs()`` — it launches, prints a version —
    while still being too old for this; version-manager shims (asdf,
    nvm) in particular tend to resolve to whatever the project/global
    pin says, which is easy to leave on an old Node for unrelated
    reasons. Check the actual capability instead of trusting a version
    number.

    sensitivity_tier: 1
    """
    try:
        result = subprocess.run(
            [candidate, "-e", "process.exit(globalThis.crypto?.subtle ? 0 : 1)"],
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _tool_candidates(name: str) -> Iterator[str]:
    """Yield existing executable candidates for ``name``, best first.

    sensitivity_tier: 1
    """
    found = shutil.which(name)
    if found:
        yield found
    for d in _TOOL_FALLBACK_DIRS:
        candidate = d / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            yield str(candidate)
    # nvm keeps one dir per installed version; prefer the newest.
    nvm_dir = Path.home() / ".nvm" / "versions" / "node"
    try:
        versions = sorted(
            (p for p in nvm_dir.iterdir() if p.name.startswith("v")),
            key=lambda p: _version_key(p.name),
            reverse=True,
        )
    except OSError:
        versions = []
    for v in versions:
        candidate = v / "bin" / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            yield str(candidate)


def _resolve_tool(
    name: str,
    extra_check: Callable[[str], bool] | None = None,
) -> str | None:
    """Locate a WORKING ``name`` binary via PATH, then known locations.

    Returns the first candidate that actually executes — and, when
    *extra_check* is given, also satisfies it — or ``None`` when none
    does (the caller raises with an actionable message).

    sensitivity_tier: 1
    """
    for candidate in _tool_candidates(name):
        if not _runs(candidate):
            logger.warning(
                "%s found at %s but it does not run; trying next candidate",
                name,
                candidate,
            )
            continue
        if extra_check is not None and not extra_check(candidate):
            logger.warning(
                "%s found at %s but failed a capability check; "
                "trying next candidate",
                name,
                candidate,
            )
            continue
        return candidate
    return None


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class WhatsAppClientError(RuntimeError):
    """Base error for WhatsApp client operations."""


class WhatsAppQRRequiredError(WhatsAppClientError):
    """Raised when the Baileys client needs QR pairing."""

    def __init__(self, qr_data: str) -> None:
        super().__init__("QR pairing required")
        self.qr_data = qr_data


class WhatsAppConnectTimeoutError(WhatsAppClientError):
    """Raised when the client doesn't become ready within the timeout."""


class WhatsAppSendError(WhatsAppClientError):
    """Raised when a send command fails."""


# ---------------------------------------------------------------------------
# Client class
# ---------------------------------------------------------------------------


class WhatsAppClient:
    """Manage the custom Baileys Node.js subprocess via JSONL stdio.

    Usage::

        with WhatsAppClient(auth_dir) as client:
            client.send_message("5511999@s.whatsapp.net", "Hello")
            status = client.get_status()

    Or without context manager::

        client = WhatsAppClient(auth_dir)
        client.connect()
        try:
            ...
        finally:
            client.close()

    sensitivity_tier: 1
    """

    def __init__(
        self,
        auth_dir: Path,
        *,
        timeout: float = 45.0,
        node_script: Path | None = None,
    ) -> None:
        self._auth_dir = Path(auth_dir)
        self._timeout = timeout
        self._script = Path(node_script) if node_script else _CLIENT_SCRIPT

        self._proc: subprocess.Popen[bytes] | None = None
        self._reader_thread: threading.Thread | None = None

        # Request/response tracking
        self._lock = threading.Lock()
        self._pending: dict[str, threading.Event] = {}
        self._responses: dict[str, dict[str, Any]] = {}

        # Event queue for non-response messages
        self._event_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=10_000)

        # WhatsApp delivery acks (status >= 3). Routed here by the reader thread
        # so the listener can update _notification_log.delivered_at without
        # contending with iter_events() consumers.
        self._acks_received: deque[dict[str, Any]] = deque(maxlen=10_000)
        self._acks_lock = threading.Lock()

        # Connection state
        self._connected = threading.Event()
        self._qr_event = threading.Event()
        self._qr_data: str | None = None
        self.jid: str | None = None
        self.lid: str | None = None  # Linked Device ID (for @lid self-chat)

        self._closed = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Start the Node.js subprocess and wait for the ``ready`` event.

        Raises :class:`WhatsAppQRRequiredError` if QR pairing is needed.
        Raises :class:`WhatsAppConnectTimeoutError` if the client doesn't
        become ready within *timeout* seconds.
        """
        if self._proc is not None:
            return

        self._ensure_npm_installed()

        node = _resolve_tool("node", extra_check=_node_has_webcrypto)
        if node is None:
            raise WhatsAppClientError(
                "No Node.js with Web Crypto support (needed by Baileys) "
                "found in PATH or common install locations — install "
                "Node 20+ (e.g. `brew install node`)",
            )

        cmd = [
            node,
            str(self._script),
            "--auth-dir",
            str(self._auth_dir),
        ]

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )

        # Start reader thread
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            daemon=True,
            name="whatsapp-client-reader",
        )
        self._reader_thread.start()

        # Start stderr drain thread (prevents buffer deadlock)
        threading.Thread(
            target=self._stderr_drain,
            daemon=True,
            name="whatsapp-client-stderr",
        ).start()

        # Wait for ready or QR
        deadline = time.monotonic() + self._timeout

        while time.monotonic() < deadline:
            if self._connected.wait(timeout=1.0):
                return
            if self._qr_event.is_set():
                raise WhatsAppQRRequiredError(self._qr_data or "")
            # Check if process died
            if self._proc.poll() is not None:
                raise WhatsAppClientError(
                    f"Client process exited with code {self._proc.returncode}",
                )

        raise WhatsAppConnectTimeoutError(
            f"Client did not become ready within {self._timeout}s",
        )

    def close(self) -> None:
        """Terminate the Node.js subprocess."""
        if self._closed:
            return
        self._closed = True

        proc = self._proc
        if proc is None:
            return

        # Try graceful shutdown via SIGTERM to process group
        try:
            os.killpg(os.getpgid(proc.pid), 15)  # SIGTERM
        except (ProcessLookupError, PermissionError, OSError):
            pass

        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), 9)  # SIGKILL
            except (ProcessLookupError, PermissionError, OSError):
                pass
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                pass

        self._proc = None

    def __enter__(self) -> WhatsAppClient:
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def send_message(
        self,
        to: str,
        text: str,
        message_id: str | None = None,
    ) -> dict[str, Any]:
        """Send a text message and wait for the response.

        Returns the response ``data`` dict (contains ``message_id``).
        Raises :class:`WhatsAppSendError` on failure.

        When *message_id* is provided, it is passed to Baileys as the WhatsApp
        message ID, enabling server-side dedup on re-sends (used by the
        listener outbox to replay leftover requests after a crash).

        sensitivity_tier: 3
        """
        cmd: dict[str, Any] = {
            "cmd": "send",
            "to": to,
            "text": text,
        }
        if message_id:
            cmd["messageId"] = message_id
        resp = self._send_command(cmd)
        if not resp.get("ok"):
            raise WhatsAppSendError(str(resp.get("error", "Send failed")))
        return resp.get("data", {})

    def drain_acks(self) -> list[dict[str, Any]]:
        """Pop all pending delivery acks accumulated by the reader thread.

        Returns a list of ``message_ack`` event dicts in arrival order.

        sensitivity_tier: 2
        """
        with self._acks_lock:
            if not self._acks_received:
                return []
            drained = list(self._acks_received)
            self._acks_received.clear()
            return drained

    def get_status(self) -> dict[str, Any]:
        """Return connection status from the Node.js client.

        sensitivity_tier: 1
        """
        resp = self._send_command({"cmd": "status"})
        return resp.get("data", {})

    def iter_events(
        self,
        timeout: float = 0.1,
    ) -> Iterator[dict[str, Any]]:
        """Yield events from the event queue (non-blocking).

        sensitivity_tier: 2
        """
        while True:
            try:
                event = self._event_queue.get(timeout=timeout)
                yield event
            except queue.Empty:
                return

    @property
    def is_connected(self) -> bool:
        """Whether the client is currently connected."""
        return self._connected.is_set() and self._proc is not None

    # ------------------------------------------------------------------
    # Internal: command/response
    # ------------------------------------------------------------------

    def _send_command(
        self,
        cmd: dict[str, Any],
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Write a JSONL command to stdin, wait for the matching response."""
        if self._proc is None or self._proc.stdin is None:
            raise WhatsAppClientError("Client not connected")

        req_id = uuid.uuid4().hex[:12]
        cmd["id"] = req_id

        event = threading.Event()
        with self._lock:
            self._pending[req_id] = event

        # Write command
        line = json.dumps(cmd, ensure_ascii=False) + "\n"
        try:
            self._proc.stdin.write(line.encode("utf-8"))
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            with self._lock:
                self._pending.pop(req_id, None)
            raise WhatsAppClientError(f"Failed to write command: {exc}") from exc

        # Wait for response
        actual_timeout = timeout or self._timeout
        if not event.wait(timeout=actual_timeout):
            with self._lock:
                self._pending.pop(req_id, None)
            raise WhatsAppClientError(
                f"Command timed out after {actual_timeout}s: {cmd.get('cmd')}",
            )

        with self._lock:
            self._pending.pop(req_id, None)
            return self._responses.pop(req_id, {"ok": False, "error": "No response"})

    # ------------------------------------------------------------------
    # Internal: reader thread
    # ------------------------------------------------------------------

    def _reader_loop(self) -> None:
        """Background thread reading JSONL from the Node.js stdout."""
        proc = self._proc
        if proc is None or proc.stdout is None:
            return

        for raw_line in proc.stdout:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("Non-JSON line from client: %s", line[:200])
                continue

            event_type = event.get("type", "")
            event_id = event.get("id")

            # Match responses to pending commands
            if event_type in ("response", "error") and event_id:
                with self._lock:
                    pending = self._pending.get(event_id)
                    if pending:
                        self._responses[event_id] = event
                        pending.set()
                continue

            # Handle connection lifecycle events
            if event_type == "ready":
                self.jid = event.get("jid")
                self.lid = event.get("lid")
                self._connected.set()

            elif event_type == "qr":
                self._qr_data = event.get("qr")
                self._qr_event.set()

            elif event_type == "connection":
                if event.get("status") == "open":
                    self._connected.set()
                elif event.get("status") == "close":
                    self._connected.clear()

            # Delivery acks live in their own bounded deque so the listener's
            # outbox-ack drain can find them deterministically without racing
            # with iter_events() consumers.
            if event_type == "message_ack":
                with self._acks_lock:
                    self._acks_received.append(event)
                continue

            # Put all events in the queue for the listener
            try:
                self._event_queue.put_nowait(event)
            except queue.Full:
                # Drop oldest event to prevent memory bloat
                try:
                    self._event_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._event_queue.put_nowait(event)
                except queue.Full:
                    pass

        # stdout closed — process is exiting
        self._connected.clear()

    def _stderr_drain(self) -> None:
        """Drain stderr to prevent buffer deadlock and log to debug."""
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        for raw_line in proc.stderr:
            line = raw_line.decode("utf-8", errors="replace").rstrip()
            if line:
                logger.debug("[whatsapp-client] %s", line)

    # ------------------------------------------------------------------
    # Internal: npm install
    # ------------------------------------------------------------------

    def _ensure_npm_installed(self) -> None:
        """Run ``npm install`` if ``node_modules/`` is missing."""
        node_modules = _PACKAGE_DIR / "node_modules"
        if node_modules.exists():
            return

        npm = _resolve_tool("npm")
        if npm is None:
            raise WhatsAppClientError(
                "npm not found in PATH or common install locations — "
                "cannot install WhatsApp client dependencies",
            )

        logger.info("Installing WhatsApp client dependencies...")
        result = subprocess.run(
            [npm, "install", "--omit=dev"],
            cwd=str(_PACKAGE_DIR),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if result.returncode != 0:
            raise WhatsAppClientError(
                f"npm install failed: {result.stderr[:500]}",
            )
        logger.info("WhatsApp client dependencies installed")
