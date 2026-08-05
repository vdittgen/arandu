"""Persistent log of every retrieval call.

Stores one row per :meth:`src.core.retrieval.pipeline.HybridSearch.search`
invocation: the query, the IDs returned, the per-result scores, the
latency, and which embedding model produced them.

The log feeds two downstream consumers:

* :mod:`evals.retrieval.seeder` ``--from-log`` mode — surfaces real
  user queries as golden-set seeds, keeping the eval set close to
  what people actually ask.
* manual debugging — ``SELECT query, retrieved_ids FROM
  _retrieval_log ORDER BY ts DESC`` is the fastest way to diagnose
  "why didn't the Brain Agent find X?"

Schema lives in the main SQLite database alongside ``_query_log``;
the leading underscore matches the convention for internal tables
(``_query_log`` / ``_interest_profile``) and keeps it out of the
raw_* / ext_* namespaces enforced by SensitivityGuard.

sensitivity_tier: 2 (query strings can be tier 2; IDs only)
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from src.core.sqlite.engine import DatabaseEngine

logger = logging.getLogger(__name__)

TABLE = "_retrieval_log"


@dataclass(frozen=True)
class LoggedRetrieval:
    """One row from ``_retrieval_log``.

    sensitivity_tier: 2
    """

    id: str
    ts: str
    query: str
    retrieved_ids: list[str]
    scores: list[float]
    latency_ms: float
    mode: str  # vector | bm25 | hybrid | routed
    embedding_model: str
    policy: str  # remote-default | local-only — egress posture at call time
    extra: dict[str, Any]


def init_table(db: DatabaseEngine) -> None:
    """Create ``_retrieval_log`` if absent.

    Idempotent — every caller can pre-init defensively.

    sensitivity_tier: N/A
    """
    db.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            id              TEXT PRIMARY KEY,
            ts              TEXT NOT NULL,
            query           TEXT NOT NULL,
            retrieved_ids   TEXT NOT NULL,
            scores          TEXT NOT NULL,
            latency_ms      REAL NOT NULL,
            mode            TEXT NOT NULL,
            embedding_model TEXT NOT NULL DEFAULT '',
            policy          TEXT NOT NULL DEFAULT '',
            extra           TEXT NOT NULL DEFAULT '{{}}'
        )
        """,
    )
    # Recency-first index — every read query is "newest first".
    db.execute(
        f"CREATE INDEX IF NOT EXISTS "
        f"{TABLE}_ts_idx ON {TABLE}(ts DESC)",
    )


def record(
    db: DatabaseEngine,
    *,
    query: str,
    retrieved_ids: list[str],
    scores: list[float],
    latency_ms: float,
    mode: str,
    embedding_model: str = "",
    policy: str = "",
    extra: dict[str, Any] | None = None,
) -> str:
    """Persist one retrieval call. Returns the row ID for traceability.

    Best-effort — any exception is logged and swallowed so an
    observability outage never breaks live retrieval. We accept the
    one cost (an occasional lost row) in exchange for never blocking
    a user query on log writes.

    sensitivity_tier: 2
    """
    row_id = str(uuid.uuid4())
    try:
        init_table(db)
        db.execute(
            f"INSERT INTO {TABLE} "
            f"(id, ts, query, retrieved_ids, scores, latency_ms, "
            f"mode, embedding_model, policy, extra) "
            f"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                row_id,
                datetime.now(tz=UTC).isoformat(),
                query,
                json.dumps(retrieved_ids),
                json.dumps(scores),
                float(latency_ms),
                mode,
                embedding_model,
                policy,
                json.dumps(extra or {}),
            ],
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("retrieval_log write failed (continuing): %s", exc)
    return row_id


def recent(
    db: DatabaseEngine,
    *,
    limit: int = 100,
    mode: str | None = None,
) -> list[LoggedRetrieval]:
    """Read the most-recent N rows, newest first.

    sensitivity_tier: 2
    """
    init_table(db)
    if mode is None:
        sql = (
            f"SELECT * FROM {TABLE} ORDER BY ts DESC LIMIT ?"
        )
        params: list[Any] = [limit]
    else:
        sql = (
            f"SELECT * FROM {TABLE} WHERE mode = ? "
            f"ORDER BY ts DESC LIMIT ?"
        )
        params = [mode, limit]
    rows = db.query(sql, params)
    return [
        LoggedRetrieval(
            id=str(r["id"]),
            ts=str(r["ts"]),
            query=str(r["query"]),
            retrieved_ids=json.loads(r["retrieved_ids"]),
            scores=json.loads(r["scores"]),
            latency_ms=float(r["latency_ms"]),
            mode=str(r["mode"]),
            embedding_model=str(r.get("embedding_model") or ""),
            policy=str(r.get("policy") or ""),
            extra=json.loads(r.get("extra") or "{}"),
        )
        for r in rows
    ]


@contextmanager
def measure() -> Any:
    """Tiny stopwatch context manager returning ``elapsed_ms`` after exit.

    Usage::

        with measure() as t:
            hits = pipeline.search(query)
        retrieval_log.record(db, ..., latency_ms=t.ms)

    sensitivity_tier: N/A
    """
    class _Timer:
        ms: float = 0.0

    timer = _Timer()
    start = time.perf_counter()
    try:
        yield timer
    finally:
        timer.ms = (time.perf_counter() - start) * 1000.0
