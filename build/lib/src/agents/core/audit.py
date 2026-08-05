"""SHA-256 chained audit log for agent decisions.

Mirrors the on-disk format of ``src-tauri/src/firewall/audit.rs`` so that
Python-side decisions (firewall verdicts, LLM egress events, deep-agent
tool calls) append to the same chain as Rust-side data-access decisions.

The chain is append-only. ``verify_chain()`` walks the file and confirms
every entry's ``previous_hash`` matches the SHA-256 hash of the prior
serialized line. Tampering breaks the chain — there is no delete API.

sensitivity_tier: N/A (records decisions about all tiers but stores no
content; only hashes of prompts/responses).
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

GENESIS_HASH = "0" * 64

DEFAULT_AUDIT_PATH = (
    Path.home() / ".arandu" / "data" / "audit.jsonl"
)


def _hash_line(line: str) -> str:
    """SHA-256 of a single serialized entry, hex-encoded.

    sensitivity_tier: N/A
    """
    return hashlib.sha256(line.encode("utf-8")).hexdigest()


def hash_payload(payload: str | bytes) -> str:
    """Return the hex SHA-256 of an arbitrary payload (prompt, response).

    Callers should hash sensitive content before adding it to an audit
    entry — the chain never stores raw prompts.

    sensitivity_tier: N/A
    """
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class AuditChain:
    """Append-only audit log with SHA-256 chain integrity.

    Thread-safe via an internal lock — multiple agents may share one
    instance. For cross-process use (subprocess agents), each process
    opens its own ``AuditChain`` over the same path; ``append()`` takes
    an exclusive ``fcntl.flock`` over the file across the
    read-last-hash + write critical section so writers across
    processes can't race and produce two entries claiming the same
    ``previous_hash``.

    sensitivity_tier: N/A
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or DEFAULT_AUDIT_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    @staticmethod
    def _last_hash_from_fd(f: Any) -> str:
        """Return the SHA-256 of the last non-empty line in ``f``.

        Reads from the current file position; callers should seek to
        the start first. Returns the genesis hash when the file is
        empty.

        sensitivity_tier: N/A
        """
        last_line: str | None = None
        for raw in f:
            line = raw.strip()
            if line:
                last_line = line
        if last_line is None:
            return GENESIS_HASH
        return _hash_line(last_line)

    def append(
        self,
        *,
        event_type: str,
        agent_id: str,
        decision: str,
        payload_hash: str | None = None,
        tier: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str:
        """Append an audit entry and return its line hash.

        The entry's ``previous_hash`` is the hash of the prior line
        (or the genesis hash for the first entry). Returned hash is the
        SHA-256 of *this* serialized line — callers can persist it for
        cross-reference.

        Holds ``fcntl.flock(LOCK_EX)`` over the read-last-hash + write
        critical section so concurrent appenders across processes
        serialize. The intra-process ``threading.Lock`` is kept as a
        fast path for same-process threads (each ``flock`` acquisition
        opens a fresh fd, so without it two threads in one process
        could deadlock against their own fds).

        sensitivity_tier: N/A
        """
        with self._lock:
            # ``a+`` creates the file if missing, lets us read for the
            # last-hash, and appends our new line — all on one fd so
            # ``flock`` covers both operations.
            with self._path.open("a+", encoding="utf-8") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.seek(0)
                    previous_hash = self._last_hash_from_fd(f)
                    entry = {
                        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                        "event_type": event_type,
                        "agent_id": agent_id,
                        "decision": decision,
                        "tier": tier,
                        "payload_hash": payload_hash,
                        "previous_hash": previous_hash,
                        "extra": extra or {},
                    }
                    line = json.dumps(
                        entry, sort_keys=True, separators=(",", ":"),
                    )
                    f.write(line + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                    return _hash_line(line)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def verify(self) -> bool:
        """Re-walk the chain and confirm hash linkage end-to-end.

        sensitivity_tier: N/A
        """
        with self._lock:
            if not self._path.exists():
                return True
            expected = GENESIS_HASH
            with self._path.open("r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        return False
                    if entry.get("previous_hash") != expected:
                        return False
                    expected = _hash_line(line)
            return True

    def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return up to ``limit`` most-recent entries (newest first).

        sensitivity_tier: N/A
        """
        if not self._path.exists():
            return []
        with self._lock:
            with self._path.open("r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
        out: list[dict[str, Any]] = []
        for raw in reversed(lines):
            try:
                out.append(json.loads(raw))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed audit line")
            if len(out) >= limit:
                break
        return out


_default_chain: AuditChain | None = None
_default_lock = threading.Lock()


def default_chain() -> AuditChain:
    """Return a process-wide default chain at ``~/.arandu/data/audit.jsonl``.

    sensitivity_tier: N/A
    """
    global _default_chain
    if _default_chain is None:
        with _default_lock:
            if _default_chain is None:
                override = os.environ.get("ARANDU_AUDIT_PATH")
                path = Path(override) if override else DEFAULT_AUDIT_PATH
                _default_chain = AuditChain(path=path)
    return _default_chain


def reset_default_chain_for_tests() -> None:
    """Drop the cached default chain.

    Tests that swap ``ARANDU_AUDIT_PATH`` mid-run should call this to
    avoid leaking a chain pointed at the old path.

    sensitivity_tier: N/A
    """
    global _default_chain
    with _default_lock:
        _default_chain = None
