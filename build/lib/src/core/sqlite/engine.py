"""SQLite database engine for Arandu.

Provides a single-connection embedded database engine with persistent storage
at ~/.arandu/data/arandu.sqlite3.  Uses WAL (Write-Ahead Logging) mode
for concurrent access: readers NEVER block writers, writers NEVER block
readers.  ``busy_timeout`` makes writers queue gracefully instead of failing.

API-compatible with the former DuckDB ``DatabaseEngine`` — callers use the
same ``execute()`` / ``query()`` / ``close()`` interface.

Sensitivity tier: methods operating on raw data inherit the tier of the
underlying table (defined per-table in schemas.py).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any

from src.core.profiler import timed

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path.home() / ".arandu" / "data" / "arandu.sqlite3"

# SQLite PRAGMA settings for optimal performance with WAL mode.
_PRAGMAS = [
    ("journal_mode", "WAL"),        # Write-Ahead Logging — concurrent reads
    ("busy_timeout", "30000"),      # 30s — writers queue instead of failing
    ("synchronous", "NORMAL"),      # Safe with WAL, faster than FULL
    ("foreign_keys", "ON"),         # Enforce FK constraints
    ("cache_size", "-32000"),       # 32 MB page cache
]


def connect_with_pragmas(
    db_path: Path | str,
    *,
    check_same_thread: bool = True,
) -> sqlite3.Connection:
    """Open a raw ``sqlite3`` connection with the engine's pragmas applied.

    For stores that hold their own connection instead of a full
    ``DatabaseEngine`` (agent block store, run log, config stores).
    Keeps every connection to the shared database on the same WAL +
    ``busy_timeout`` discipline — a bare ``sqlite3.connect()`` gets the
    5s default busy timeout instead of 30s and silently drifts from the
    engine's configuration.

    sensitivity_tier: N/A (infrastructure)
    """
    conn = sqlite3.connect(
        str(db_path),
        check_same_thread=check_same_thread,
        isolation_level=None,
    )
    for pragma, value in _PRAGMAS:
        conn.execute(f"PRAGMA {pragma} = {value}")
    return conn


# ------------------------------------------------------------------
# Cache key serialization
# ------------------------------------------------------------------


class _CacheEncoder(json.JSONEncoder):
    """Encode date/datetime objects for cache key generation.

    sensitivity_tier: N/A
    """

    def default(self, o: Any) -> Any:
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        return super().default(o)


# ------------------------------------------------------------------
# Query cache
# ------------------------------------------------------------------


class QueryCache:
    """LRU cache with TTL for SELECT results.

    sensitivity_tier: N/A (infrastructure)
    """

    def __init__(
        self,
        maxsize: int = 100,
        ttl_seconds: float = 300.0,
    ) -> None:
        self._maxsize = maxsize
        self._ttl = ttl_seconds
        self._cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
        self._order: list[str] = []
        self._hits = 0
        self._misses = 0
        # Guards _cache/_order/_hits/_misses: the engine is shared across
        # threads (query_engine runs plan + vector search in a
        # ThreadPoolExecutor) and dict/list mutation is not atomic.
        self._lock = threading.Lock()

    def _make_key(
        self,
        sql: str,
        parameters: list[Any] | None,
    ) -> str:
        """Build a deterministic cache key from SQL and parameters.

        sensitivity_tier: N/A
        """
        params_str = json.dumps(
            parameters or [],
            cls=_CacheEncoder,
            sort_keys=True,
        )
        return f"{sql}|{params_str}"

    def get(
        self,
        sql: str,
        parameters: list[Any] | None,
    ) -> list[dict[str, Any]] | None:
        """Return cached result if present and not expired.

        sensitivity_tier: N/A
        """
        key = self._make_key(sql, parameters)
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None

            ts, result = entry
            if (time.monotonic() - ts) > self._ttl:
                del self._cache[key]
                self._order.remove(key)
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._order.remove(key)
            self._order.append(key)
            self._hits += 1
            return result

    def put(
        self,
        sql: str,
        parameters: list[Any] | None,
        result: list[dict[str, Any]],
    ) -> None:
        """Store a query result in the cache.

        sensitivity_tier: N/A
        """
        key = self._make_key(sql, parameters)

        with self._lock:
            if key in self._cache:
                self._order.remove(key)
            elif len(self._cache) >= self._maxsize:
                oldest = self._order.pop(0)
                del self._cache[oldest]

            self._cache[key] = (time.monotonic(), result)
            self._order.append(key)

    def invalidate(self) -> None:
        """Clear all cached entries.

        sensitivity_tier: N/A
        """
        with self._lock:
            self._cache.clear()
            self._order.clear()

    def stats(self) -> dict[str, Any]:
        """Return cache performance statistics.

        sensitivity_tier: N/A
        """
        with self._lock:
            total = self._hits + self._misses
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._cache),
                "maxsize": self._maxsize,
                "hit_rate": self._hits / total if total > 0 else 0.0,
            }


class DatabaseEngine:
    """Embedded SQLite engine with WAL mode for concurrent access.

    Uses WAL (Write-Ahead Logging) so readers never block writers and
    writers never block readers.  ``busy_timeout`` queues concurrent
    writers for up to 30 seconds instead of failing immediately.

    API-compatible with the former DuckDB ``DatabaseEngine``.

    sensitivity_tier: N/A (infrastructure layer — no user data stored here)
    """

    def __init__(
        self,
        db_path: Path = DEFAULT_DB_PATH,
        read_only: bool = False,
    ) -> None:
        """Initialize the engine and open (or create) the database file.

        Args:
            db_path: Filesystem path for the persistent SQLite database.
                     Parent directories are created automatically.
            read_only: Accepted for API compatibility but ignored.
                       SQLite WAL mode allows concurrent readers without
                       a special mode flag.
        """
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._read_only = read_only
        self._in_transaction = False
        self._conn = self._connect()
        self._cache = QueryCache()

    def _connect(self) -> sqlite3.Connection:
        """Open a SQLite connection with WAL mode and performance pragmas.

        sensitivity_tier: N/A (infrastructure)
        """
        conn = connect_with_pragmas(
            self._db_path,
            check_same_thread=False,
        )

        logger.info(
            "SQLite connected: %s (WAL mode)",
            self._db_path,
        )
        return conn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def _is_transaction_control(sql: str) -> str | None:
        """Return the transaction keyword if *sql* is a control statement.

        Recognised prefixes: ``BEGIN``, ``COMMIT``, ``ROLLBACK``,
        ``END`` (synonym for COMMIT), ``SAVEPOINT``, ``RELEASE``.

        Returns:
            The upper-cased keyword (e.g. ``"BEGIN"``) or ``None``.

        sensitivity_tier: N/A
        """
        first_word = sql.strip().split(None, 1)[0].upper() if sql.strip() else ""
        if first_word in {"BEGIN", "COMMIT", "ROLLBACK", "END", "SAVEPOINT", "RELEASE"}:
            return first_word
        return None

    @timed()
    def execute(
        self, sql: str, parameters: list[Any] | None = None,
    ) -> None:
        """Execute a SQL statement that returns no rows (DDL / DML).

        Invalidates the query cache since DDL/DML may change data.

        Transaction control statements (``BEGIN``, ``COMMIT``,
        ``ROLLBACK``) are passed through without an automatic commit so
        that callers can manage transactions explicitly.  Statements
        executed between ``BEGIN`` and ``COMMIT``/``ROLLBACK`` are also
        not auto-committed — the caller's explicit ``COMMIT`` or
        ``ROLLBACK`` will finalise the transaction.

        Args:
            sql: SQL statement to execute.
            parameters: Optional positional parameters for parameterised
                queries.
        """
        self._cache.invalidate()

        txn_keyword = self._is_transaction_control(sql)

        if parameters:
            self._conn.execute(sql, parameters)
        else:
            self._conn.execute(sql)

        if txn_keyword == "BEGIN":
            # Caller opened an explicit transaction — suppress
            # auto-commit until COMMIT / ROLLBACK.
            self._in_transaction = True
        elif txn_keyword in {"COMMIT", "ROLLBACK", "END"}:
            # Transaction boundary handled by the statement itself.
            self._in_transaction = False
        elif not self._in_transaction:
            # Normal DML/DDL outside an explicit transaction —
            # auto-commit as before.
            self._conn.commit()

    @timed()
    def query(
        self, sql: str, parameters: list[Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a SQL query and return all rows as a list of dicts.

        Results are cached with LRU eviction and TTL expiry.

        Args:
            sql: SELECT statement to execute.
            parameters: Optional positional parameters for parameterised
                queries.

        Returns:
            List of dicts where keys are column names and values are
            row values.
        """
        cached = self._cache.get(sql, parameters)
        if cached is not None:
            return cached

        cursor = self._conn.execute(sql, parameters or [])

        if cursor.description is None:
            return []

        columns = [desc[0] for desc in cursor.description]
        result = [dict(zip(columns, row)) for row in cursor.fetchall()]
        self._cache.put(sql, parameters, result)
        return result

    @contextmanager
    def transaction(self) -> Iterator[DatabaseEngine]:
        """Run a block of statements as a single atomic transaction.

        Commits on normal exit, rolls back if the block raises.  With
        ``isolation_level=None`` each ``execute()`` otherwise
        auto-commits, so multi-statement invariants (e.g. insert a row
        + bump a counter) need this wrapper to survive crashes between
        statements.

        Not reentrant: SQLite does not support nested ``BEGIN``, so a
        ``transaction()`` block must not be opened inside another one.

        sensitivity_tier: N/A (infrastructure)
        """
        self.execute("BEGIN")
        try:
            yield self
        except BaseException:
            self.execute("ROLLBACK")
            raise
        else:
            self.execute("COMMIT")

    def invalidate_cache(self) -> None:
        """Clear the query cache.

        sensitivity_tier: N/A
        """
        self._cache.invalidate()

    def cache_stats(self) -> dict[str, Any]:
        """Return cache performance statistics.

        sensitivity_tier: N/A
        """
        return self._cache.stats()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
        logger.info("SQLite connection closed: %s", self._db_path)

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> DatabaseEngine:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
