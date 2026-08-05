"""Persistent WhatsApp listener lifecycle and ingestion runtime.

Keeps a long-lived custom Baileys client (``src/extensions/whatsapp/client.js``)
connected so Baileys can receive ``messages.upsert`` continuously, then
ingests store-backed message deltas into DuckDB.

sensitivity_tier: 2 (process + message ingestion orchestration)
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from src.core.db_helpers import utc_now_iso
from src.core.sqlite.engine import DEFAULT_DB_PATH, DatabaseEngine
from src.extensions.bridges.whatsapp.client import (
    WhatsAppClient,
    WhatsAppClientError,
    WhatsAppQRRequiredError,
)
from src.extensions.bridges.whatsapp.paths import (
    resolve_whatsapp_auth_dir,
    resolve_whatsapp_store_path,
)
from src.extensions.connectors.catalog import ConnectorCatalog
from src.extensions.ingestion.adapter import IngestionAdapter
from src.extensions.models import ToolTemplate

logger = logging.getLogger(__name__)

_PIPELINE_LOCK_PATH = Path.home() / ".arandu" / "data" / ".pipeline_running"
_RUNTIME_DIR = Path.home() / ".arandu" / "data" / "whatsapp_listener"
_PID_PATH = _RUNTIME_DIR / "listener.pid.json"
_STATUS_PATH = _RUNTIME_DIR / "status.json"
_LOG_PATH = _RUNTIME_DIR / "listener.log"
_OUTBOX_DIR = _RUNTIME_DIR / "outbox"
_OUTBOX_RESP_DIR = _RUNTIME_DIR / "outbox_responses"
_LOCK_PATH = _RUNTIME_DIR / "listener.lock"


def _is_process_running(
    pid: int,
    expected_cmd_substring: str | None = None,
) -> bool:
    """Check whether a process ID is alive (optionally command-matched)."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    if not expected_cmd_substring:
        return True

    try:
        completed = subprocess.run(  # noqa: S603
            ["ps", "-p", str(pid), "-o", "command="],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return True

    if completed.returncode != 0:
        return False
    cmdline = completed.stdout.strip()
    return expected_cmd_substring in cmdline


def _pipeline_pid_alive() -> bool:
    """Return True if the PID recorded in the pipeline lock file is alive.

    Used before stealing a stale-looking pipeline lock — a slow worker
    that is still running must keep its lock (see staleness protocol in
    ``src/pipeline/worker.py``).

    sensitivity_tier: 1
    """
    try:
        pid = int(_PIPELINE_LOCK_PATH.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False
    return _is_process_running(pid)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically to avoid partial state files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk; return empty dict on failure."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _try_acquire_lock() -> int | None:
    """Try to acquire exclusive listener lock.

    Returns the file descriptor on success (caller must keep it open)
    or ``None`` when another process already holds the lock.
    """
    _RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(_LOCK_PATH), os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except OSError:
        os.close(fd)
        return None


def _release_lock(fd: int) -> None:
    """Release the listener lock."""
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    except OSError:
        pass


def _is_listener_locked() -> bool:
    """Check if another process holds the listener lock.

    Non-blocking probe: opens the lock file, attempts a
    non-blocking exclusive flock, then releases immediately.
    """
    if not _LOCK_PATH.exists():
        return False
    fd = os.open(str(_LOCK_PATH), os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Got the lock → no one else holds it
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    except OSError:
        return True
    finally:
        os.close(fd)


def _pgrep_listener_pid() -> int:
    """Find a running ``whatsapp-listener-run`` subprocess via pgrep.

    Returns the first matching pid, or ``0`` when none is found. Used by
    ``WhatsAppListenerService.status`` as a fallback when ``listener.pid.json``
    is missing — the Rust supervisor is the authoritative writer, but during
    spawn there's a brief window before the file appears, and prior sessions
    may have left a live subprocess with no pid file at all.
    """
    try:
        completed = subprocess.run(  # noqa: S603
            ["pgrep", "-f", "whatsapp-listener-run"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return 0
    if completed.returncode != 0:
        return 0
    own_pid = os.getpid()
    for line in completed.stdout.split():
        try:
            pid = int(line.strip())
        except ValueError:
            continue
        if pid <= 0 or pid == own_pid:
            continue
        return pid
    return 0


def _outbox_request_path(request_id: str) -> Path:
    """Return outbound request path for a request id."""
    return _OUTBOX_DIR / f"{request_id}.json"


def _outbox_response_path(request_id: str) -> Path:
    """Return outbound response path for a request id."""
    return _OUTBOX_RESP_DIR / f"{request_id}.json"


_MAX_OUTBOUND_ATTEMPTS = 3


def _is_self_chat_jid(to: str) -> bool:
    """Whether *to* is a Linked Device ID (self-chat).

    Baileys does not emit ``messages.update`` events with
    ``status >= 3`` (DELIVERY_ACK/READ) for messages addressed to the
    sender's own @lid, so the ack drain in
    :func:`_drain_message_acks` will never fire for these.  Callers
    must therefore short-circuit the ack flow and confirm delivery
    synchronously instead.

    sensitivity_tier: 1
    """
    return to.endswith("@lid")


def _mark_delivered_now(message_id: str) -> None:
    """Stamp ``_notification_log.delivered_at`` for a synchronously-delivered send.

    Used by self-chat sends, where the WhatsApp-side ack never arrives
    but delivery is effectively confirmed when Baileys' ``sendMessage``
    returns successfully (the message is going to the user's own
    linked devices via the locally-connected Baileys session).

    sensitivity_tier: 2 (touches notification metadata, no message body)
    """
    if not message_id:
        return
    try:
        with DatabaseEngine(db_path=DEFAULT_DB_PATH) as db:
            db.execute(
                "UPDATE _notification_log "
                "SET delivered_at = ? "
                "WHERE message_id = ? AND delivered_at IS NULL",
                [utc_now_iso(), message_id],
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to stamp delivered_at for self-chat send %s: %s",
            message_id, exc,
        )


def _process_outbound_requests(
    client: WhatsAppClient,
) -> None:
    """Process queued outbound send requests using active listener client.

    On success, the request file is **kept** on disk until a WhatsApp
    delivery ack arrives (handled by :func:`_drain_message_acks`).  This
    makes deliveries durable across listener/app restarts: if Tauri
    crashes after Baileys' ``sendMessage`` returns but before the message
    actually transmits, the leftover ``{id}.json`` file is re-processed
    on the next tick.  The outbox request id is passed through as the
    WhatsApp ``messageId`` so Baileys' server-side dedup prevents
    double-delivery on replay.

    **Self-chat exception**: messages to the user's own ``@lid`` never
    receive ``messages.update`` delivery acks from Baileys, so the file
    is deleted immediately on send-success and
    ``_notification_log.delivered_at`` is stamped synchronously.
    Otherwise the file would be re-sent on every loop tick forever
    (idempotent due to messageId dedup, but wasteful and noisy).

    On Baileys failure, the ``attempts`` counter is bumped and the file
    is kept for the next tick to retry; after
    ``_MAX_OUTBOUND_ATTEMPTS`` failed attempts the file is renamed to
    ``{id}.failed`` to break the loop.  Validation errors (missing
    fields) delete the file outright — replaying would never succeed.

    sensitivity_tier: 3 (handles message bodies before WhatsApp send)
    """
    _OUTBOX_DIR.mkdir(parents=True, exist_ok=True)
    _OUTBOX_RESP_DIR.mkdir(parents=True, exist_ok=True)

    request_files = sorted(_OUTBOX_DIR.glob("*.json"))
    for req_path in request_files:
        payload = _read_json(req_path)
        request_id = str(payload.get("id") or req_path.stem).strip()
        to = str(payload.get("to") or "").strip()
        message = str(payload.get("message") or "").strip()
        attempts = int(payload.get("attempts") or 0)

        response: dict[str, Any] = {
            "id": request_id,
            "status": "failed",
            "message_id": None,
            "error": None,
            "processed_at": utc_now_iso(),
        }

        validation_error = False
        try:
            if not request_id:
                validation_error = True
                raise ValueError("Missing outbound request id")
            if not to or not message:
                validation_error = True
                raise ValueError(
                    "Missing outbound request fields: to/message",
                )

            result = client.send_message(to, message, message_id=request_id)
            response["status"] = "sent"
            # Baileys returns its own key.id; use it if the caller did not
            # supply messageId. Either way, this is the WhatsApp-side id
            # the ack will reference.
            response["message_id"] = result.get("message_id") or request_id
            # When the node side resolved the JID via onWhatsApp (e.g.
            # Brazilian mobile-9 quirk), surface it so callers can show
            # the actual destination address.
            resolved_jid = result.get("resolved_jid")
            if resolved_jid:
                response["resolved_jid"] = resolved_jid
        except Exception as exc:  # noqa: BLE001
            response["error"] = str(exc)

        _write_json_atomic(
            _outbox_response_path(request_id),
            response,
        )

        if response["status"] == "sent":
            if _is_self_chat_jid(to):
                # No ack will ever come — confirm delivery synchronously
                # and remove the file so we don't re-send forever.
                try:
                    req_path.unlink(missing_ok=True)
                except OSError:
                    pass
                _mark_delivered_now(response["message_id"])
            else:
                # Keep file until delivery ack drains it. Stamp the
                # resolved message_id so the ack handler can match
                # deterministically.
                payload["message_id"] = response["message_id"]
                payload["last_send_at"] = utc_now_iso()
                _write_json_atomic(req_path, payload)
        elif validation_error:
            try:
                req_path.unlink(missing_ok=True)
            except OSError:
                pass
        else:
            attempts += 1
            payload["attempts"] = attempts
            payload["last_error"] = response["error"]
            payload["last_attempt_at"] = utc_now_iso()
            if attempts >= _MAX_OUTBOUND_ATTEMPTS:
                failed_path = req_path.with_suffix(".failed")
                _write_json_atomic(failed_path, payload)
                try:
                    req_path.unlink(missing_ok=True)
                except OSError:
                    pass
                logger.warning(
                    "WhatsApp outbound request %s gave up after %d attempts: %s",
                    request_id,
                    attempts,
                    response["error"],
                )
            else:
                _write_json_atomic(req_path, payload)


def _drain_message_acks(client: WhatsAppClient) -> None:
    """Consume delivery acks from the client and clean up the outbox.

    For each ack pulled via :meth:`WhatsAppClient.drain_acks`, delete the
    matching outbox request file and stamp
    ``_notification_log.delivered_at`` so callers can distinguish
    "queued at Baileys" from "WhatsApp delivered to recipient."

    sensitivity_tier: 2 (touches notification metadata, no message body)
    """
    acks = client.drain_acks()
    if not acks:
        return

    delivered_at = utc_now_iso()
    msg_ids: list[str] = []
    for ack in acks:
        msg_id = str(ack.get("msg_id") or "").strip()
        if not msg_id:
            continue
        msg_ids.append(msg_id)

        # Remove outbox file — file may be absent if the message was sent
        # by something other than the outbox flow (e.g. interactive
        # reply path), which is fine.
        req_path = _OUTBOX_DIR / f"{msg_id}.json"
        try:
            req_path.unlink(missing_ok=True)
        except OSError:
            pass

    if not msg_ids:
        return

    try:
        with DatabaseEngine(db_path=DEFAULT_DB_PATH) as db:
            for msg_id in msg_ids:
                db.execute(
                    "UPDATE _notification_log "
                    "SET delivered_at = ? "
                    "WHERE message_id = ? AND delivered_at IS NULL",
                    [delivered_at, msg_id],
                )
                logger.info(
                    "WhatsApp delivery confirmed: %s", msg_id,
                )
    except Exception as exc:  # noqa: BLE001
        # Don't let a DB hiccup crash the listener loop — acks will be
        # lost for this batch but the message is already delivered.
        logger.warning(
            "Failed to update _notification_log for %d ack(s): %s",
            len(msg_ids),
            exc,
        )


def send_text_via_running_listener(
    to: str,
    message: str,
    timeout_seconds: float = 20.0,
) -> dict[str, Any] | None:
    """Queue an outbound text send to the running listener.

    Returns ``None`` when listener isn't running. Otherwise returns
    response payload from listener (sent/failed).
    """
    service = WhatsAppListenerService()
    status = service.status()
    if not bool(status.get("running")):
        return None

    request_id = uuid.uuid4().hex
    req_payload = {
        "id": request_id,
        "to": str(to).strip(),
        "message": str(message),
        "created_at": utc_now_iso(),
    }

    req_path = _outbox_request_path(request_id)
    res_path = _outbox_response_path(request_id)

    _write_json_atomic(req_path, req_payload)

    deadline = time.monotonic() + max(1.0, timeout_seconds)
    while time.monotonic() < deadline:
        if res_path.exists():
            response = _read_json(res_path)
            try:
                res_path.unlink(missing_ok=True)
            except OSError:
                pass
            return response
        time.sleep(0.2)

    return {
        "id": request_id,
        "status": "failed",
        "message_id": None,
        "error": "Timed out waiting for listener outbound response",
        "processed_at": utc_now_iso(),
    }


class WhatsAppListenerService:
    """Manage the background WhatsApp listener process."""

    def __init__(self) -> None:
        self._runtime_dir = _RUNTIME_DIR
        self._pid_path = _PID_PATH
        self._status_path = _STATUS_PATH
        self._log_path = _LOG_PATH

    def status(self) -> dict[str, Any]:
        """Return current listener status.

        The Rust supervisor writes ``listener.pid.json`` after spawning the
        child, so in steady state we read it directly. If the file is missing
        but the lock is held — happens during the brief spawn window, after
        a supervisor crash, or when a legacy caller bypassed the supervisor —
        we fall back to ``pgrep`` so the notifier doesn't reject sends for
        a listener that's demonstrably alive.
        """
        payload = _read_json(self._pid_path)
        pid = int(payload.get("pid", 0) or 0)
        running = _is_process_running(
            pid,
            expected_cmd_substring="whatsapp-listener-run",
        )

        # Clean stale pid files eagerly.
        if pid and not running:
            try:
                self._pid_path.unlink(missing_ok=True)
            except OSError:
                pass
            pid = 0

        if not running and _is_listener_locked():
            scanned_pid = _pgrep_listener_pid()
            if scanned_pid and _is_process_running(
                scanned_pid,
                expected_cmd_substring="whatsapp-listener-run",
            ):
                pid = scanned_pid
                running = True

        status_data = _read_json(self._status_path)
        return {
            "running": running,
            "locked": _is_listener_locked(),
            "pid": pid if running else None,
            "started_at": payload.get("started_at"),
            "command": payload.get("command"),
            "args": payload.get("args", []),
            "auth_dir": str(resolve_whatsapp_auth_dir()),
            "store_path": str(resolve_whatsapp_store_path()),
            "status_file": status_data,
            "log_path": str(self._log_path),
        }

    def ensure_running(
        self,
        command: str,
        args: tuple[str, ...],
    ) -> dict[str, Any]:
        """Start listener when not running; otherwise return current status."""
        current = self.status()
        if current.get("running"):
            return current
        return self.start(command, args)

    def start(
        self,
        command: str,
        args: tuple[str, ...],
    ) -> dict[str, Any]:
        """Start the background listener process."""
        if _is_listener_locked():
            logger.info(
                "Listener already running (lock held), "
                "skipping spawn",
            )
            return self.status()
        self._runtime_dir.mkdir(parents=True, exist_ok=True)

        run_cmd = [
            sys.executable,
            "-m",
            "src.core.cli",
            "whatsapp-listener-run",
            "--mcp-command",
            command,
        ]
        for arg in args:
            run_cmd.append(f"--mcp-arg={arg}")

        log_file = self._log_path.open("ab")
        try:
            proc = subprocess.Popen(  # noqa: S603
                run_cmd,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            log_file.close()

        payload = {
            "pid": proc.pid,
            "started_at": utc_now_iso(),
            "command": command,
            "args": list(args),
        }
        _write_json_atomic(self._pid_path, payload)

        # Give the process a short window to fail fast.
        time.sleep(0.4)
        return self.status()

    def stop(self, timeout_seconds: float = 8.0) -> dict[str, Any]:
        """Stop the background listener if running."""
        current = self.status()
        pid = int(current.get("pid") or 0)
        if pid <= 0:
            return current

        deadline = time.monotonic() + max(0.1, timeout_seconds)
        signals = [signal.SIGTERM, signal.SIGKILL]

        for sig in signals:
            if not _is_process_running(pid):
                break
            try:
                os.killpg(pid, sig)
            except ProcessLookupError:
                break
            except PermissionError:
                try:
                    os.kill(pid, sig)
                except (ProcessLookupError, PermissionError):
                    break

            while time.monotonic() < deadline:
                if not _is_process_running(pid):
                    break
                time.sleep(0.1)

        try:
            self._pid_path.unlink(missing_ok=True)
        except OSError:
            pass

        return self.status()

    def clear_pid_if_current(self, pid: int) -> None:
        """Remove pid file when it points to the current process."""
        payload = _read_json(self._pid_path)
        file_pid = int(payload.get("pid", 0) or 0)
        if file_pid == pid:
            try:
                self._pid_path.unlink(missing_ok=True)
            except OSError:
                pass

    def write_runtime_status(self, payload: dict[str, Any]) -> None:
        """Persist listener runtime heartbeat/status."""
        data = dict(payload)
        data["updated_at"] = utc_now_iso()
        _write_json_atomic(self._status_path, data)


class _NoopMcpClient:
    """Placeholder client for store-driven ingestion paths."""

    def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        msg = f"Unexpected MCP tool call in store ingestion path: {tool_name}"
        raise RuntimeError(msg)


def _resolve_whatsapp_data_tool(catalog: ConnectorCatalog) -> ToolTemplate:
    """Return the WhatsApp data tool that writes into ``raw_messages``."""
    template = catalog.get("whatsapp")
    if template is None:
        raise RuntimeError("WhatsApp connector not found in catalog")
    for tool in template.tools:
        if tool.tool_type == "data" and tool.tool_name == "list_chats":
            return tool
    raise RuntimeError("WhatsApp list_chats data tool not found in catalog")


def ingest_whatsapp_store_once(
    db_path: Path = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """Ingest one incremental batch from WhatsApp store.json into DuckDB."""
    catalog = ConnectorCatalog()
    tool = _resolve_whatsapp_data_tool(catalog)

    with DatabaseEngine(db_path=db_path) as db:
        adapter = IngestionAdapter(
            connector_id="whatsapp",
            tool=tool,
            mcp_client=_NoopMcpClient(),
            db_engine=db,
        )
        result = adapter.sync()

    return {
        "status": result.status,
        "rows_fetched": result.rows_fetched,
        "rows_new": result.rows_new,
        "rows_updated": result.rows_updated,
        "rows_synced": result.rows_new + result.rows_updated,
        "duration_seconds": result.duration_seconds,
        "error": result.error,
    }


def _poll_self_chat_messages(
    self_jid: str,
    send_phone: str | None = None,
    self_lid: str | None = None,
) -> int:
    """Read self-chat messages from store.json and upsert new ones.

    Baileys stores self-chat messages under MULTIPLE possible keys:

    * The Baileys-normalized JID (e.g. ``"554892011083@s.whatsapp.net"``)
      — historical messages captured via ``messages.upsert``.
    * The full-phone JID (e.g. ``"5548992011083@s.whatsapp.net"``)
      — messages in the thread created when sending to self-chat using
      the user's full international phone number.
    * Linked Device ID JIDs (e.g. ``"161048623628515@lid"``)
      — self-chat messages sent from the phone in multi-device mode.
      These ``@lid`` JIDs are opaque identifiers assigned by WhatsApp.

    *self_jid* is the Baileys JID (from ``creds.json`` ``me.id``).
    *send_phone* is the settings phone stripped of ``+`` (may differ
    from *self_jid* due to country normalization, e.g. Brazil).
    *self_lid* is the Linked Device ID (from ``creds.json`` ``me.lid``).

    Returns the number of new messages inserted.
    Non-fatal: failures are logged and return 0.

    sensitivity_tier: 2
    """
    try:
        import json as _json

        from src.core.sqlite.engine import (
            DEFAULT_DB_PATH,
            DatabaseEngine,
        )
        from src.extensions.ingestion.adapter import (
            IngestionAdapter,
        )

        # Canonical JID used for all raw_messages entries
        phone_jid = f"{self_jid}@s.whatsapp.net"

        store_path = resolve_whatsapp_store_path()
        if not store_path.exists():
            return 0

        data = _json.loads(
            store_path.read_text(encoding="utf-8"),
        )
        store_msgs = data.get("messages", {})

        # Merge messages from all known self-chat JID keys:
        # 1. Baileys phone JID
        # 2. Full-phone JID (if different from Baileys JID)
        # 3. @lid JIDs (linked device IDs for multi-device self-chat)
        raw_items: list[dict] = list(store_msgs.get(phone_jid, []))
        if send_phone and send_phone != self_jid:
            alt_jid = f"{send_phone}@s.whatsapp.net"
            raw_items.extend(store_msgs.get(alt_jid, []))

        # Scan for @lid JIDs — self-chat from the phone uses these.
        # ONLY use the known self-LID from creds.json.  The previous
        # heuristic (>50% fromMe) incorrectly grabbed regular DMs sent
        # from other @lid JIDs, causing replies to strangers' messages.
        if self_lid:
            lid_jid = f"{self_lid}@lid"
            raw_items.extend(store_msgs.get(lid_jid, []))

        if not raw_items:
            return 0

        with DatabaseEngine(db_path=DEFAULT_DB_PATH) as db:
            inserted = 0
            for entry in raw_items:
                if not isinstance(entry, dict):
                    continue
                key = entry.get("key", {})
                msg_id = key.get("id")
                if not msg_id:
                    continue

                full_id = f"{phone_jid}:{msg_id}"
                existing = db.query(
                    "SELECT 1 FROM raw_messages WHERE id = ?",
                    [full_id],
                )
                if existing:
                    continue

                msg_body = entry.get("message", {}) or {}
                text = (
                    msg_body.get("conversation")
                    or msg_body.get(
                        "extendedTextMessage", {},
                    ).get("text")
                    or ""
                )
                # Handle non-text message types
                if not text:
                    if isinstance(msg_body.get("audioMessage"), dict):
                        text = "[audio]"
                    elif isinstance(msg_body.get("imageMessage"), dict):
                        caption = str(
                            msg_body["imageMessage"].get("caption", ""),
                        ).strip()
                        text = caption or "[image]"
                    elif isinstance(msg_body.get("stickerMessage"), dict):
                        text = "[sticker]"
                    elif isinstance(
                        msg_body.get("documentMessage"),
                        dict,
                    ):
                        text = "[document]"
                    else:
                        continue

                ts = entry.get("messageTimestamp", 0)
                if isinstance(ts, dict) and "low" in ts:
                    ts = ts["low"]
                ts_str = (
                    IngestionAdapter._coerce_utc_timestamp(ts)
                    or utc_now_iso()
                )
                from_me = bool(key.get("fromMe", True))

                # Extract quoted message text when the user
                # replies to a specific message in the thread.
                # This is crucial for pronoun resolution (e.g.
                # "dele" → "João Fonseca" from the quoted msg).
                metadata_json: str | None = None
                ext_text_msg = msg_body.get(
                    "extendedTextMessage", {},
                )
                ctx_info = (
                    ext_text_msg.get("contextInfo", {})
                    or msg_body.get("contextInfo", {})
                    or {}
                )
                quoted_msg = ctx_info.get("quotedMessage", {})
                if quoted_msg:
                    quoted_text = (
                        quoted_msg.get("conversation", "")
                        or quoted_msg.get(
                            "extendedTextMessage", {},
                        ).get("text", "")
                    )
                    if quoted_text:
                        metadata_json = _json.dumps(
                            {"quoted_text": quoted_text},
                        )

                db.execute(
                    "INSERT INTO raw_messages "
                    "(id, sender, sender_name, recipient,"
                    " content, timestamp, is_from_me,"
                    " chat_name, metadata, source) "
                    "VALUES (?,?,?,?,?,?,?,?,?,'whatsapp')",
                    [
                        full_id,
                        "me" if from_me else phone_jid,
                        "me" if from_me else self_jid,
                        phone_jid,
                        str(text).strip(),
                        ts_str,
                        from_me,
                        phone_jid,
                        metadata_json,
                    ],
                )
                inserted += 1

            if inserted > 0:
                logger.info(
                    "Polled %d new self-chat messages "
                    "from store.messages",
                    inserted,
                )
            return inserted

    except Exception:  # noqa: BLE001
        logger.debug(
            "Self-chat store poll failed", exc_info=True,
        )
        return 0


def _maybe_process_replies(
    client: WhatsAppClient,
) -> None:
    """Check for and process user replies in the WhatsApp self-chat.

    Non-fatal: failures are logged but never crash the listener loop.
    Opens its own write-mode DuckDB connection since the ingestion
    adapter's connection is already closed at this point.

    Uses the live *client* for sending replies — avoids outbox IPC
    round-trip and session contention.

    sensitivity_tier: 3
    """
    try:
        import json as _json
        from pathlib import Path as _Path

        from src.agents.action_executor import ActionExecutor
        from src.agents.brain import BrainAgentV2
        from src.agents.tool_registry import ToolRegistry
        from src.core.chromadb.engine import VectorEngine
        from src.core.kuzu.engine import GraphEngine
        from src.core.query_engine import QueryEngine
        from src.core.sqlite.engine import DEFAULT_DB_PATH, DatabaseEngine
        from src.extensions.bridges.whatsapp.paths import (
            resolve_self_jid,
            resolve_self_lid,
        )
        from src.extensions.connectors.catalog import ConnectorCatalog
        from src.extensions.connectors.registry import ExtensionRegistry
        from src.models.llm_provider import create_provider_from_settings
        from src.notifications.reply_handler import ReplyHandler

        # Read phone from settings
        settings_file = _Path.home() / ".arandu" / "settings.json"
        if not settings_file.exists():
            return
        settings = _json.loads(settings_file.read_text(encoding="utf-8"))
        if not settings.get("notifications_enabled"):
            return
        phone = settings.get("whatsapp_notification_phone")
        if not phone:
            return

        # Resolve the bare WhatsApp JID from creds.json (e.g. "554892011083").
        # This may differ from the settings phone due to country-specific
        # normalization (e.g. Brazil drops a leading "9" in mobile numbers).
        self_jid = resolve_self_jid() or phone.lstrip("+")
        self_lid = resolve_self_lid()

        # Poll store.messages to capture self-chat msgs Baileys missed.
        # Pass send_phone and self_lid so we also check @lid JID keys
        # (self-chat from phone arrives under @lid in multi-device mode).
        send_phone = phone.lstrip("+")
        _poll_self_chat_messages(
            self_jid, send_phone=send_phone, self_lid=self_lid,
        )

        def _direct_send(to: str, message: str) -> bool:
            """Send via the live listener WhatsApp client.

            Retries up to 3 times with a 60s connection-wait between
            attempts so that Baileys reconnection (exponential backoff)
            has time to complete after a mid-inference disconnect.

            sensitivity_tier: 2
            """
            import time as _t

            _max_send_retries = 3
            for attempt in range(_max_send_retries):
                # Wait for Baileys reconnection if disconnected.
                if not client.is_connected:
                    logger.info(
                        "WhatsApp disconnected, waiting for reconnect "
                        "before send (%d/%d)…",
                        attempt + 1, _max_send_retries,
                    )
                    waited = 0
                    while not client.is_connected and waited < 60:
                        _t.sleep(2)
                        waited += 2
                    if not client.is_connected:
                        logger.warning(
                            "WhatsApp still disconnected after 60s wait",
                        )
                        continue
                try:
                    client.send_message(to, message)
                    return True
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Direct reply send failed (attempt %d/%d): %s",
                        attempt + 1, _max_send_retries, exc,
                    )
                    if attempt < _max_send_retries - 1:
                        _t.sleep(5)
            return False

        data_dir = _Path.home() / ".arandu" / "data"
        with DatabaseEngine(db_path=DEFAULT_DB_PATH) as db:
            # Open Kuzu/ChromaDB with retry — another process (pipeline
            # worker, Tauri background task) may hold the lock briefly.
            # Full GraphRAG context is important for smart replies.
            import time as _time

            kuzu_eng = None
            chroma_eng = None
            _max_retries = 20  # 20 × 5s = up to ~100s wait
            for attempt in range(_max_retries):
                try:
                    # Read-only: the listener only queries Kuzu for
                    # BrainAgent reply context. Sharing read-only
                    # access lets the chat CLI (and other readers)
                    # open the same database concurrently.
                    kuzu_eng = GraphEngine(
                        db_path=data_dir / "kuzu_db",
                        read_only=True,
                    )
                    break
                except Exception:  # noqa: BLE001
                    if attempt < _max_retries - 1:
                        logger.debug(
                            "Kuzu busy, retrying (%d/%d)…",
                            attempt + 1, _max_retries,
                        )
                        _time.sleep(5)
                    else:
                        logger.warning(
                            "Kuzu unavailable after %d retries — "
                            "replies will lack graph context",
                            _max_retries,
                        )
            for attempt in range(_max_retries):
                try:
                    chroma_eng = VectorEngine(db_path=data_dir / "chromadb")
                    break
                except Exception:  # noqa: BLE001
                    if attempt < _max_retries - 1:
                        logger.debug(
                            "ChromaDB busy, retrying (%d/%d)…",
                            attempt + 1, _max_retries,
                        )
                        _time.sleep(5)
                    else:
                        logger.warning(
                            "ChromaDB unavailable after %d retries — "
                            "replies will lack vector context",
                            _max_retries,
                        )
            qe = QueryEngine(
                duckdb=db, kuzu=kuzu_eng, chromadb=chroma_eng,
            )
            # Use interactive priority — replies are user-facing and
            # must never be blocked by background pipeline/sync tasks.
            provider = create_provider_from_settings(background=False)

            # ToolRegistry powers the propose_action pydantic-ai tool
            # on Brain v2 — required for WhatsApp action UX.
            tool_registry = ToolRegistry(
                catalog=ConnectorCatalog(),
                registry=ExtensionRegistry(),
            )
            brain = BrainAgentV2(
                query_engine=qe,
                provider=provider,
                tool_registry=tool_registry,
            )

            executor = ActionExecutor()

            def _sync_connector(connector_id: str) -> None:
                """Re-sync after action so new data appears."""
                try:
                    from src.extensions.connectors.connection_manager import (
                        ConnectionManager,
                    )
                    ConnectionManager(db_engine=db).sync_now(
                        connector_id,
                    )
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "Post-action re-sync failed",
                        exc_info=True,
                    )

            handler = ReplyHandler(
                db_engine=db, brain_agent=brain, phone=phone,
                self_jid=self_jid, self_lid=self_lid,
                send_fn=_direct_send,
                action_executor=executor,
                sync_fn=_sync_connector,
            )
            try:
                count = handler.process_new_replies()
                if count > 0:
                    logger.info("Processed %d self-chat replies", count)
            finally:
                # Explicitly release Kuzu/ChromaDB locks so other
                # processes (pipeline worker, Tauri tasks) can connect.
                if kuzu_eng is not None:
                    try:
                        kuzu_eng.close()
                    except Exception:  # noqa: BLE001
                        pass
                if chroma_eng is not None:
                    try:
                        chroma_eng.close()
                    except Exception:  # noqa: BLE001
                        pass
    except Exception:  # noqa: BLE001
        logger.warning("Reply processing failed", exc_info=True)


def _maybe_transcribe_audio() -> None:
    """Transcribe downloaded WhatsApp audio files and update raw_messages.

    Scans ``~/.arandu/data/audio_cache/`` for OGG files, transcribes each
    using the local Whisper model, updates the corresponding ``raw_messages``
    row from ``[audio]`` to ``[voice note] {text}``, stores audio metadata
    in the ``metadata`` JSON column, and deletes the audio file.

    Non-fatal: failures are logged but never crash the listener loop.

    sensitivity_tier: 3
    """

    def _read_sidecar(ogg_path: Path) -> dict | None:
        """Read sidecar metadata from the .json written by client.js."""
        import json as _j
        meta_path = ogg_path.with_suffix(".json")
        if meta_path.exists():
            try:
                return _j.loads(meta_path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                return None
        return None

    def _cleanup_sidecar(ogg_path: Path) -> None:
        """Remove sidecar .json if it exists."""
        meta_path = ogg_path.with_suffix(".json")
        if meta_path.exists():
            meta_path.unlink(missing_ok=True)

    try:
        import json as _json
        from pathlib import Path as _Path

        from src.models.voice_transcriber import VoiceTranscriber, is_available

        if not is_available():
            return

        # Check settings for voice transcription
        settings_file = _Path.home() / ".arandu" / "settings.json"
        if settings_file.exists():
            try:
                settings = _json.loads(
                    settings_file.read_text(encoding="utf-8"),
                )
                if not settings.get("voice_transcription_enabled", True):
                    return
            except (ValueError, OSError):
                return
        else:
            return

        audio_dir = _Path.home() / ".arandu" / "data" / "audio_cache"
        if not audio_dir.exists():
            return

        audio_files = list(audio_dir.glob("*.ogg"))
        if not audio_files:
            return

        model_size = settings.get("whisper_model_size", "base")
        transcriber = VoiceTranscriber(model_size=model_size)

        from src.core.sqlite.engine import DEFAULT_DB_PATH, DatabaseEngine

        with DatabaseEngine(db_path=DEFAULT_DB_PATH) as db:
            for audio_path in audio_files:
                msg_key = audio_path.stem  # filename is {key_id}.ogg
                try:
                    # Skip if already transcribed (file re-downloaded
                    # after Python deleted it — emittedSelfChatIds is
                    # in-memory and resets on Node.js restart).
                    db.invalidate_cache()
                    already = db.query(
                        "SELECT 1 FROM raw_messages "
                        "WHERE id LIKE ? AND content LIKE '[voice note]%' "
                        "LIMIT 1",
                        [f"%{msg_key}%"],
                    )
                    if already:
                        audio_path.unlink(missing_ok=True)
                        _cleanup_sidecar(audio_path)
                        continue

                    result = transcriber.transcribe(str(audio_path))
                    if not result.text.strip():
                        audio_path.unlink(missing_ok=True)
                        _cleanup_sidecar(audio_path)
                        continue

                    content = f"[voice note] {result.text.strip()}"
                    metadata = _json.dumps({
                        "audio_duration": result.duration,
                        "original_language": result.language,
                        "transcribed": True,
                    })

                    # Match audio to DB rows by key suffix — covers
                    # both @s.whatsapp.net and @lid JID variants.
                    db.execute(
                        """
                        UPDATE raw_messages
                        SET content = ?, metadata = ?
                        WHERE id LIKE ?
                          AND content = '[audio]'
                        """,
                        [content, metadata, f"%{msg_key}%"],
                    )

                    audio_path.unlink(missing_ok=True)
                    _cleanup_sidecar(audio_path)
                    logger.info(
                        "Transcribed audio %s: %s (%.1fs, %s)",
                        msg_key,
                        result.text[:80],
                        result.duration,
                        result.language,
                    )

                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Audio transcription failed for %s: %s",
                        msg_key,
                        exc,
                    )

    except Exception as exc:  # noqa: BLE001
        logger.warning("_maybe_transcribe_audio failed: %s", exc)


def _maybe_evaluate_messages() -> None:
    """Evaluate newly ingested WhatsApp messages for notifications.

    Non-fatal: failures are logged but never crash the listener loop.
    Opens its own short-lived DuckDB write connection.

    sensitivity_tier: 3
    """
    try:
        import json as _json
        from pathlib import Path as _Path

        from src.agents.message_eval import (
            MessageEvaluator,
            format_realtime_notification,
        )
        from src.core.sqlite.engine import DEFAULT_DB_PATH, DatabaseEngine
        from src.notifications.preference_service import PreferenceService

        settings_file = _Path.home() / ".arandu" / "settings.json"
        if not settings_file.exists():
            return
        settings = _json.loads(
            settings_file.read_text(encoding="utf-8"),
        )
        if not settings.get("notifications_enabled"):
            return
        phone = settings.get("whatsapp_notification_phone")
        if not phone:
            return

        with DatabaseEngine(db_path=DEFAULT_DB_PATH) as db:
            prefs = PreferenceService(db_engine=db)
            if prefs.is_muted_globally():
                return

            # Check per-category preferences (topic-driven categories)
            action_enabled = prefs.is_category_enabled(
                "topic_action",
            )
            enrichment_enabled = prefs.is_category_enabled(
                "topic_enrichment",
            )
            if not action_enabled and not enrichment_enabled:
                return

            evaluator = MessageEvaluator(db_engine=db)
            notifications = evaluator.evaluate_new_messages(
                "whatsapp", "raw_messages",
            )

            if not notifications:
                logger.info(
                    "Message evaluation returned 0 notifications",
                )
                return

            logger.info(
                "Message evaluation: %d notifications to send",
                len(notifications),
            )

            actions = [
                n for n in notifications
                if n.notification_type == "topic_action"
            ]
            enrichment = [
                n for n in notifications
                if n.notification_type == "topic_enrichment"
            ]

            from src.extensions.bridges.whatsapp.paths import (
                resolve_self_jid,
                resolve_self_lid,
            )
            from src.notifications.notifier import get_opt_out_text

            # Resolve self-chat JID (prefer @lid for multi-device)
            self_lid = resolve_self_lid()
            if self_lid:
                to_jid = f"{self_lid}@lid"
            else:
                self_jid = resolve_self_jid()
                to_jid = (
                    f"{self_jid}@s.whatsapp.net"
                    if self_jid
                    else phone
                )

            if actions and action_enabled:
                msg = format_realtime_notification(actions)
                opt_out = get_opt_out_text("topic_action")
                full_msg = f"{msg}\n\n---\n{opt_out}"
                send_text_via_running_listener(
                    to=to_jid, message=full_msg,
                    timeout_seconds=20.0,
                )

            if enrichment and enrichment_enabled:
                msg = format_realtime_notification(enrichment)
                opt_out = get_opt_out_text(
                    "topic_enrichment",
                )
                full_msg = f"{msg}\n\n---\n{opt_out}"
                send_text_via_running_listener(
                    to=to_jid, message=full_msg,
                    timeout_seconds=20.0,
                )

    except Exception:  # noqa: BLE001
        logger.warning(
            "Real-time message evaluation failed",
            exc_info=True,
        )



def _wait_for_pairing(
    client: WhatsAppClient,
    service: WhatsAppListenerService,
    stop_event: threading.Event,
    *,
    initial_qr: str | None,
    last_error: str | None,
    last_ingest_rows: int,
    last_ingest_at: str | None,
) -> tuple[bool, str | None]:
    """Block on the live client until Baileys reports pairing success.

    Baileys' QR is only valid while the WebSocket session that produced
    it stays connected. We must keep the Node subprocess running until
    the user scans, rotating ``pending_qr`` in status.json whenever
    Baileys emits a fresh code (~20s cadence).

    Returns ``(paired, latest_qr)``. ``paired`` is True when the client
    transitions to connected; False when the subprocess dies or
    ``stop_event`` fires. ``latest_qr`` is the most recent QR observed.
    """
    current_qr = initial_qr
    proc = client._proc  # noqa: SLF001 — subprocess liveness check
    while not stop_event.is_set():
        if client.is_connected:
            return True, current_qr
        if proc is None or proc.poll() is not None:
            return False, current_qr

        for event in client.iter_events(timeout=1.0):
            etype = event.get("type")
            if etype == "qr":
                new_qr = event.get("qr") or None
                if new_qr and new_qr != current_qr:
                    current_qr = new_qr
                    service.write_runtime_status(
                        {
                            "running": True,
                            "phase": "awaiting_pair",
                            "last_error": last_error,
                            "last_ingest_rows": last_ingest_rows,
                            "last_ingest_at": last_ingest_at,
                            "qr": current_qr,
                        }
                    )
            elif etype == "ready":
                return True, current_qr
            elif etype == "connection":
                status = event.get("status")
                if status == "open":
                    return True, current_qr
                if status == "close":
                    return False, current_qr
    return False, current_qr


def run_whatsapp_listener(
    command: str,
    args: tuple[str, ...],
    mcp_timeout_seconds: float = 45.0,
    scan_interval_seconds: float = 2.0,
    reconnect_backoff_seconds: float = 5.0,
) -> int:
    """Run the foreground listener loop (background subprocess).

    Uses the custom Baileys client (``client.js``). The *command*
    and *args* are accepted for CLI compat but **ignored**.

    Acquires an exclusive ``fcntl.flock`` so only one listener
    process can run at a time.  Returns 1 immediately when
    another instance already holds the lock.
    """
    lock_fd = _try_acquire_lock()
    if lock_fd is None:
        logger.error(
            "Another listener already running (lock held), "
            "exiting",
        )
        return 1

    service = WhatsAppListenerService()
    stop_event = threading.Event()

    def _request_stop(_signum: int, _frame: Any) -> None:  # noqa: ANN401
        stop_event.set()

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    auth_dir = resolve_whatsapp_auth_dir()
    last_store_mtime_ns: int | None = None
    last_ingest_rows = 0
    last_ingest_at: str | None = None
    last_error: str | None = None
    pending_qr: str | None = None
    last_reply_poll_at: float = 0.0
    eval_thread_active = threading.Event()  # prevents concurrent LLM evals

    try:
        while not stop_event.is_set():
            service.write_runtime_status(
                {
                    "running": True,
                    "phase": (
                        "awaiting_pair" if pending_qr else "connecting"
                    ),
                    "last_error": last_error,
                    "last_ingest_rows": last_ingest_rows,
                    "last_ingest_at": last_ingest_at,
                    "qr": pending_qr,
                }
            )

            client: WhatsAppClient | None = None
            try:
                client = WhatsAppClient(
                    auth_dir=auth_dir,
                    timeout=mcp_timeout_seconds,
                )
                try:
                    client.connect()
                except WhatsAppQRRequiredError as exc:
                    pending_qr = exc.qr_data or None
                    last_error = "QR pairing required"
                    logger.info(
                        "WhatsApp QR pairing required — surfacing to UI",
                    )
                    service.write_runtime_status(
                        {
                            "running": True,
                            "phase": "awaiting_pair",
                            "last_error": last_error,
                            "last_ingest_rows": last_ingest_rows,
                            "last_ingest_at": last_ingest_at,
                            "qr": pending_qr,
                        }
                    )
                    paired, pending_qr = _wait_for_pairing(
                        client,
                        service,
                        stop_event,
                        initial_qr=pending_qr,
                        last_error=last_error,
                        last_ingest_rows=last_ingest_rows,
                        last_ingest_at=last_ingest_at,
                    )
                    if not paired:
                        continue
                    last_error = None
                    pending_qr = None

                logger.info(
                    "WhatsApp client connected (jid=%s)",
                    client.jid,
                )
                last_store_mtime_ns = None

                while not stop_event.is_set():
                    store_path = resolve_whatsapp_store_path()
                    mtime_ns: int | None = None
                    if store_path.exists():
                        try:
                            mtime_ns = store_path.stat().st_mtime_ns
                        except OSError:
                            mtime_ns = None

                    try:
                        _process_outbound_requests(client)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "WhatsApp listener outbound processing failed: %s",
                            exc,
                        )

                    try:
                        _drain_message_acks(client)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "WhatsApp listener ack drain failed: %s",
                            exc,
                        )

                    # Skip writes while the pipeline is running to avoid
                    # SQLite write-lock contention (DDL needs exclusive lock).
                    # Staleness protocol (mirrored from pipeline/worker.py —
                    # keep in sync): the worker heartbeats the lock mtime on
                    # every progress event, so the lock is stale only when
                    # the mtime is old (>15 min) AND the recorded PID is
                    # dead — never steal the lock from a live worker before
                    # the 1h hard cap (matches the Tauri-side auto-reset).
                    pipeline_active = False
                    if _PIPELINE_LOCK_PATH.exists():
                        try:
                            age = time.time() - _PIPELINE_LOCK_PATH.stat().st_mtime
                            pipeline_active = age < 900  # 15 min heartbeat window
                            if not pipeline_active:
                                if _pipeline_pid_alive() and age < 3600:
                                    # Slow-but-alive worker — keep skipping
                                    # writes, leave its lock alone.
                                    pipeline_active = True
                                else:
                                    _PIPELINE_LOCK_PATH.unlink(missing_ok=True)
                                    logger.info(
                                        "Removed stale pipeline lock (%.0fs old)",
                                        age,
                                    )
                        except OSError:
                            pass
                    if pipeline_active:
                        logger.debug(
                            "Pipeline lock active — skipping ingestion cycle",
                        )

                    if (
                        not pipeline_active
                        and mtime_ns is not None
                        and mtime_ns != last_store_mtime_ns
                    ):
                        ingest = ingest_whatsapp_store_once()
                        last_store_mtime_ns = mtime_ns
                        last_ingest_rows = int(
                            ingest.get("rows_synced", 0) or 0,
                        )
                        last_ingest_at = utc_now_iso()
                        last_error = (
                            str(ingest["error"])
                            if (
                                ingest.get("status") == "error"
                                and ingest.get("error")
                            )
                            else None
                        )

                    # All write operations below are skipped while the
                    # pipeline holds the exclusive DDL lock.
                    if not pipeline_active:
                        # Transcribe downloaded WhatsApp audio files.
                        # Always check — audio may arrive before ingestion
                        # stores the [audio] placeholder in raw_messages.
                        _maybe_transcribe_audio()

                        # Evaluate new messages in a background thread
                        # so the main loop stays responsive for outbox
                        # sends (proactive notifications, replies).
                        # The LLM call can take 10-60s — must not block
                        # the outbox polling.
                        if (
                            last_ingest_rows > 0
                            and not eval_thread_active.is_set()
                        ):
                            eval_thread_active.set()

                            def _bg_evaluate() -> None:
                                try:
                                    _maybe_evaluate_messages()
                                finally:
                                    eval_thread_active.clear()

                            threading.Thread(
                                target=_bg_evaluate,
                                daemon=True,
                                name="msg-eval",
                            ).start()

                        # Check for user replies in self-chat.
                        # Run after ingestion with new rows, or periodically
                        # (every ~30s) to catch self-chat messages that
                        # Baileys' messages.upsert may have missed.
                        if last_ingest_rows > 0:
                            _maybe_process_replies(client)
                            last_reply_poll_at = time.monotonic()
                        elif time.monotonic() - last_reply_poll_at > 30:
                            last_reply_poll_at = time.monotonic()
                            _maybe_process_replies(client)

                    pending_qr = None
                    service.write_runtime_status(
                        {
                            "running": True,
                            "phase": "connected",
                            "last_error": last_error,
                            "last_ingest_rows": last_ingest_rows,
                            "last_ingest_at": last_ingest_at,
                            "qr": None,
                        }
                    )
                    stop_event.wait(max(0.5, scan_interval_seconds))

            except WhatsAppClientError as exc:
                last_error = str(exc)
                logger.warning("WhatsApp client error: %s", exc)
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                logger.warning("WhatsApp listener loop error: %s", exc)
            finally:
                if client is not None:
                    try:
                        client.close()
                    except Exception:  # noqa: BLE001
                        pass

            if not stop_event.is_set():
                service.write_runtime_status(
                    {
                        "running": True,
                        "phase": (
                            "awaiting_pair" if pending_qr else "reconnecting"
                        ),
                        "last_error": last_error,
                        "last_ingest_rows": last_ingest_rows,
                        "last_ingest_at": last_ingest_at,
                        "qr": pending_qr,
                    }
                )
                stop_event.wait(max(1.0, reconnect_backoff_seconds))
    finally:
        _release_lock(lock_fd)
        service.write_runtime_status(
            {
                "running": False,
                "phase": "stopped",
                "last_error": last_error,
                "last_ingest_rows": last_ingest_rows,
                "last_ingest_at": last_ingest_at,
            }
        )
        service.clear_pid_if_current(os.getpid())

    return 0
