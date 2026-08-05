"""Per-call store of original/redacted message pairs for audit-log drilldown.

The SHA-256 chained audit log at ``~/.arandu/data/audit.jsonl`` records
*that* a redaction happened but never the prompt text — its whole point
is to be tamper-evident metadata. To let the user click an audit row and
see what was actually flagged, we persist the pre/post-redaction payload
in a separate Tier 3 SQLite store, keyed by the same ``payload_hash``
that the ``egress_decision`` and ``egress_redaction`` audit rows already
carry.

Why SQLite and not per-call JSON files:

- Time-based retention is a one-liner (``DELETE WHERE stored_at < ?``)
- Single file with 0600 mode, no filesystem walking to prune
- Lookup by hash is the primary access pattern → primary key fits

Retention: rows older than :data:`DEFAULT_RETENTION_HOURS` are pruned
opportunistically on every write. No background job, no scheduler
dependency — the store self-cleans as it is used. The window is short
on purpose: the realistic use case is "I just sent that prompt, let
me see what was redacted", and that curiosity is satisfied within
hours, not days. Keeping raw Tier 3 prompts on disk past that is
unnecessary exposure.

The store is intentionally **separate** from
``redaction_registry.sqlite``: the registry is a monotonically-growing
entity dictionary (raw → placeholder), while this log is a rotating
per-call audit trail. Mixing them would conflate two different
lifecycles in one file.

sensitivity_tier: 3 (holds raw Tier 3 message content)
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


DEFAULT_PATH = Path.home() / ".arandu" / "data" / "redaction_log.sqlite"
DEFAULT_RETENTION_HOURS = 24

_SCHEMA = """
CREATE TABLE IF NOT EXISTS redaction_log (
    payload_hash       TEXT PRIMARY KEY,
    stored_at          TEXT NOT NULL,
    agent_id           TEXT NOT NULL,
    lane               TEXT NOT NULL,
    original_messages  TEXT NOT NULL,
    redacted_messages  TEXT NOT NULL,
    placeholder_map    TEXT NOT NULL
)
"""

_INDEX = (
    "CREATE INDEX IF NOT EXISTS redaction_log_stored_at "
    "ON redaction_log(stored_at)"
)


def _is_hex_sha256(payload_hash: str) -> bool:
    """Reject hashes that aren't a clean lowercase hex SHA-256.

    sensitivity_tier: 1
    """
    if len(payload_hash) != 64:
        return False
    try:
        int(payload_hash, 16)
    except ValueError:
        return False
    return True


class RedactionStore:
    """SQLite-backed log of per-call redaction details.

    sensitivity_tier: 3
    """

    def __init__(
        self,
        path: Path | None = None,
        *,
        retention_hours: int = DEFAULT_RETENTION_HOURS,
    ) -> None:
        self._path = path or DEFAULT_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._retention_hours = retention_hours
        self._lock = threading.RLock()
        existed = self._path.exists()
        self._conn = sqlite3.connect(
            self._path, isolation_level=None, check_same_thread=False,
        )
        self._conn.execute(_SCHEMA)
        self._conn.execute(_INDEX)
        if not existed:
            try:
                os.chmod(self._path, 0o600)
            except OSError:  # pragma: no cover — non-POSIX fallback
                pass

    @property
    def path(self) -> Path:
        return self._path

    def store(
        self,
        *,
        payload_hash: str,
        agent_id: str,
        lane: str,
        original_messages: list[dict[str, str]],
        redacted_messages: list[dict[str, str]],
        placeholder_map: dict[str, str],
    ) -> bool:
        """Persist a redaction detail row keyed by ``payload_hash``.

        ``placeholder_map`` is ``{placeholder: original}`` so the UI
        can render the substitutions inline. Returns ``True`` if the
        row was written. Opportunistically prunes rows older than the
        retention window before inserting.

        sensitivity_tier: 3
        """
        if not _is_hex_sha256(payload_hash):
            logger.warning("Refusing to store redaction with bad hash")
            return False
        now = datetime.now(tz=timezone.utc)
        cutoff = (now - timedelta(hours=self._retention_hours)).isoformat()
        with self._lock:
            self._conn.execute(
                "DELETE FROM redaction_log WHERE stored_at < ?",
                (cutoff,),
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO redaction_log "
                "(payload_hash, stored_at, agent_id, lane, "
                "original_messages, redacted_messages, placeholder_map) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    payload_hash,
                    now.isoformat(),
                    agent_id,
                    lane,
                    json.dumps(original_messages, ensure_ascii=False),
                    json.dumps(redacted_messages, ensure_ascii=False),
                    json.dumps(placeholder_map, ensure_ascii=False),
                ),
            )
        return True

    def get(self, payload_hash: str) -> dict[str, Any] | None:
        """Load a stored redaction row, or ``None`` if absent/bad.

        sensitivity_tier: 3
        """
        if not _is_hex_sha256(payload_hash):
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT stored_at, agent_id, lane, original_messages, "
                "redacted_messages, placeholder_map "
                "FROM redaction_log WHERE payload_hash = ?",
                (payload_hash,),
            ).fetchone()
        if row is None:
            return None
        stored_at, agent_id, lane, original, redacted, placeholders = row
        try:
            return {
                "payload_hash": payload_hash,
                "stored_at": stored_at,
                "agent_id": agent_id,
                "lane": lane,
                "original_messages": json.loads(original),
                "redacted_messages": json.loads(redacted),
                "placeholder_map": json.loads(placeholders),
            }
        except json.JSONDecodeError as exc:
            logger.warning(
                "Corrupt redaction row for %s: %s", payload_hash, exc,
            )
            return None

    def prune_older_than(self, hours: int) -> int:
        """Delete rows older than ``hours`` hours. Returns rows removed.

        sensitivity_tier: 3
        """
        cutoff = (
            datetime.now(tz=timezone.utc) - timedelta(hours=hours)
        ).isoformat()
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM redaction_log WHERE stored_at < ?",
                (cutoff,),
            )
            return cur.rowcount or 0

    def close(self) -> None:
        """Close the underlying connection. Tests / shutdown only.

        sensitivity_tier: 1
        """
        with self._lock:
            self._conn.close()


_default_store: RedactionStore | None = None
_default_lock = threading.Lock()


def default_redaction_store() -> RedactionStore:
    """Process-wide default store at ``~/.arandu/data/redaction_log.sqlite``.

    sensitivity_tier: 3
    """
    global _default_store
    if _default_store is None:
        with _default_lock:
            if _default_store is None:
                override = os.environ.get("ARANDU_REDACTION_STORE_PATH")
                path = Path(override) if override else DEFAULT_PATH
                _default_store = RedactionStore(path=path)
    return _default_store


def reset_default_store_for_tests() -> None:
    """Drop the cached default store.

    sensitivity_tier: 1
    """
    global _default_store
    with _default_lock:
        if _default_store is not None:
            _default_store.close()
        _default_store = None
