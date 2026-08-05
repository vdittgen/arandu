"""Kuzu embedded graph database engine for Arandu.

Provides a single-connection graph engine with persistent storage at
~/.arandu/data/kuzu_db/.

sensitivity_tier: infrastructure layer — no user data stored here.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import kuzu

from src.core.profiler import timed

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path.home() / ".arandu" / "data" / "kuzu_db"

# Kuzu takes an exclusive cross-process file lock for a read-write handle.
# While the pipeline holds it (writing graph nodes), every other open —
# even read-only — fails with "Could not set lock on file". Those holds are
# usually short, so a read-only opener retries with backoff before giving up
# and letting the caller surface an honest "temporarily unavailable" error
# instead of a misleading empty graph.
_LOCK_RETRY_ATTEMPTS = 5
_LOCK_RETRY_BASE_DELAY_S = 0.2


def _is_lock_contention(exc: Exception) -> bool:
    """Return True if ``exc`` is Kuzu's cross-process lock-contention error."""
    return "set lock on file" in str(exc).lower()


class GraphEngine:
    """Embedded Kuzu graph engine backed by a persistent on-disk database.

    Uses a single long-lived connection. Kuzu allows multiple read-only
    handles to coexist with at most one read-write handle; long-running
    processes that only need to query the graph (the WhatsApp listener
    serving BrainAgent reply context, ``ask``/``ask-stream`` chat) should
    open with ``read_only=True`` so they do not block each other or the
    pipeline writer.

    sensitivity_tier: N/A (infrastructure layer — no user data stored here)
    """

    def __init__(
        self,
        db_path: Path = DEFAULT_DB_PATH,
        *,
        read_only: bool = False,
    ) -> None:
        """Initialize the engine and open (or create) the Kuzu database.

        Args:
            db_path: Directory path for persistent Kuzu storage.
                     Created automatically if it does not exist.
            read_only: When True, open Kuzu in shared read-only mode so
                multiple processes can query concurrently. Required for
                long-running query-only consumers; write callers must
                leave this False.
        """
        self._db_path = db_path
        self._read_only = read_only
        # Kuzu creates the database directory itself; pre-creating it causes a
        # RuntimeError ("path cannot be a directory").  We only ensure the
        # *parent* exists so that the path is reachable.
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db: kuzu.Database = self._open_with_retry(read_only)
        self._conn: kuzu.Connection = kuzu.Connection(self._db)
        logger.info(
            "Kuzu graph DB opened: %s (read_only=%s)",
            self._db_path, read_only,
        )

    def _open_with_retry(self, read_only: bool) -> kuzu.Database:
        """Open the Kuzu database, retrying transient lock contention.

        A concurrent read-write holder (the pipeline) blocks every other
        open until it releases. Those holds are usually brief, so we retry
        with linear backoff and only re-raise once attempts are exhausted —
        letting the caller report the failure rather than silently treating
        an unreachable graph as an empty one.
        """
        last_exc: Exception | None = None
        for attempt in range(_LOCK_RETRY_ATTEMPTS):
            try:
                return kuzu.Database(str(self._db_path), read_only=read_only)
            except RuntimeError as exc:
                if not _is_lock_contention(exc):
                    raise
                last_exc = exc
                if attempt < _LOCK_RETRY_ATTEMPTS - 1:
                    delay = _LOCK_RETRY_BASE_DELAY_S * (attempt + 1)
                    logger.warning(
                        "Kuzu open blocked by lock on %s (attempt %d/%d); "
                        "retrying in %.2fs",
                        self._db_path, attempt + 1, _LOCK_RETRY_ATTEMPTS, delay,
                    )
                    time.sleep(delay)
        assert last_exc is not None  # only reached after a contention failure
        raise last_exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @timed()
    def execute(self, cypher: str, parameters: dict[str, Any] | None = None) -> None:
        """Execute a Cypher statement that returns no meaningful rows (DDL / DML).

        Args:
            cypher: Cypher query to execute.
            parameters: Optional named parameters for the query.
        """
        if parameters:
            self._conn.execute(cypher, parameters)
        else:
            self._conn.execute(cypher)

    @timed()
    def query(
        self, cypher: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query and return all rows as a list of dicts.

        Column names are taken directly from the result set so callers get
        clean, aliased names when the query uses ``RETURN n.id AS id``.

        Args:
            cypher: Cypher query to execute.
            parameters: Optional named parameters for the query.

        Returns:
            List of dicts mapping column name -> value for every result row.
        """
        if parameters:
            result = self._conn.execute(cypher, parameters)
        else:
            result = self._conn.execute(cypher)

        columns = result.get_column_names()
        rows: list[dict[str, Any]] = []
        while result.has_next():
            rows.append(dict(zip(columns, result.get_next())))
        return rows

    def close(self) -> None:
        """Release the connection and database handles."""
        # Kuzu does not expose an explicit close(); releasing references
        # triggers cleanup via the C++ destructor.
        del self._conn
        del self._db
        logger.info("Kuzu graph DB closed: %s", self._db_path)

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> GraphEngine:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
