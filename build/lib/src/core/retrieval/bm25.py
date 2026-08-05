"""SQLite-FTS5-backed BM25 index over the embedded chunk corpus.

Sits alongside ChromaDB: every chunk that goes into a vector
collection also gets a row in ``_chunks_fts`` here. Hybrid search
(see :mod:`src.core.retrieval.pipeline`) fuses BM25 scores from
this table with cosine scores from ChromaDB via reciprocal-rank
fusion, recovering proper-noun and exact-phrase recall that pure
cosine misses ("Who is Israel Casa Rosa?" is the canonical case).

Schema::

    CREATE VIRTUAL TABLE _chunks_fts USING fts5(
        id UNINDEXED,         -- ChromaDB doc id (chunk-suffixed)
        record_id UNINDEXED,  -- raw record id (deduplicates chunks at read)
        text,                 -- the embedded text body
        collection UNINDEXED, -- domain collection
        layer UNINDEXED,      -- raw | mart
        sensitivity_tier UNINDEXED,
        tokenize = 'unicode61 remove_diacritics 2'
    )

``UNINDEXED`` skips full-text indexing on filter-only columns so the
index stays small. ``unicode61 remove_diacritics 2`` matches "joao"
to "joão" — important for the multilingual corpus the user has.

sensitivity_tier: 3 (mirrors raw record text)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from src.core.sqlite.engine import DatabaseEngine

logger = logging.getLogger(__name__)

FTS_TABLE = "_chunks_fts"


@dataclass(frozen=True)
class BM25Hit:
    """One hit from the FTS5 BM25 query.

    Score is negated — FTS5 ranks "more relevant" with more-negative
    scores (it sorts ASC by raw bm25() output). We negate so callers
    can sort DESC like every other ranking signal.

    sensitivity_tier: 1
    """

    id: str
    record_id: str
    score: float
    text: str
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------


def init_table(db: DatabaseEngine) -> None:
    """Create the FTS5 virtual table if it doesn't exist.

    Idempotent — safe to call on every indexer start. Uses ``IF NOT
    EXISTS`` so concurrent processes don't race.

    sensitivity_tier: N/A
    """
    db.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS {FTS_TABLE} USING fts5(
            id UNINDEXED,
            record_id UNINDEXED,
            text,
            collection UNINDEXED,
            layer UNINDEXED,
            sensitivity_tier UNINDEXED,
            tokenize = 'unicode61 remove_diacritics 2'
        )
        """,
    )


def clear(db: DatabaseEngine) -> int:
    """Empty the FTS table. Returns the number of rows removed.

    Used by the migration CLI when rebuilding the index under a new
    embedding model — keeps the FTS table in lockstep with the
    vector collections.

    sensitivity_tier: N/A
    """
    before = count(db)
    db.execute(f"DELETE FROM {FTS_TABLE}")
    return before


def count(db: DatabaseEngine) -> int:
    """Diagnostic — row count in the FTS table.

    sensitivity_tier: N/A
    """
    try:
        rows = db.query(f"SELECT count(*) AS n FROM {FTS_TABLE}")
    except Exception:  # noqa: BLE001
        return 0
    return int(rows[0]["n"]) if rows else 0


# ---------------------------------------------------------------------------
# Write path
# ---------------------------------------------------------------------------


def upsert_documents(
    db: DatabaseEngine,
    rows: list[dict[str, Any]],
) -> None:
    """Insert (or replace) FTS rows for a batch of chunks.

    FTS5 doesn't support ``INSERT OR REPLACE`` directly on the
    virtual table — we DELETE matching IDs first, then INSERT.
    Wrapped in a single transaction so the table can't be observed
    in a half-replaced state.

    ``rows`` shape::

        [{
            "id": str,           # chunk id (chunk-suffixed when applicable)
            "record_id": str,    # base record id for dedup
            "text": str,
            "collection": str,
            "layer": str,
            "sensitivity_tier": int,
        }, ...]

    sensitivity_tier: 3
    """
    if not rows:
        return
    conn = db._conn  # noqa: SLF001 — direct executemany for batch perf
    ids = [r["id"] for r in rows]
    placeholders = ",".join("?" for _ in ids)
    try:
        conn.execute("BEGIN")
        conn.execute(
            f"DELETE FROM {FTS_TABLE} WHERE id IN ({placeholders})",
            ids,
        )
        conn.executemany(
            f"INSERT INTO {FTS_TABLE} "
            f"(id, record_id, text, collection, layer, sensitivity_tier) "
            f"VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    r["id"], r["record_id"], r["text"], r["collection"],
                    r["layer"], int(r.get("sensitivity_tier", 2)),
                )
                for r in rows
            ],
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


# ---------------------------------------------------------------------------
# Read path
# ---------------------------------------------------------------------------


# FTS5 raw query characters that would break parsing. We strip
# anything that could be interpreted as query-DSL — we want a
# natural-language match. The list errs on the side of stripping:
# false-negative (a user types `+` literally) is fine, false-
# positive (FTS5 raises syntax error) breaks the whole search.
_FTS_RESERVED = (
    '"', "'", "(", ")", "*", ":", "^", "{", "}",
    "?", "!", ".", ",", ";", "+", "-",
)


def sanitise_query(query: str) -> str:
    """Strip FTS5 syntax characters from a free-text query.

    The brain agent forwards user prompts unmodified — we don't want
    a stray quote or paren to crash the search.

    sensitivity_tier: 1
    """
    out = query
    for ch in _FTS_RESERVED:
        out = out.replace(ch, " ")
    # Collapse repeated whitespace so we can wrap each remaining
    # token in quotes for an OR-of-phrases match below.
    return " ".join(out.split())


def search(
    db: DatabaseEngine,
    query: str,
    *,
    k: int = 50,
    max_tier: int = 3,
    collections: list[str] | None = None,
) -> list[BM25Hit]:
    """Run a BM25 ranked query against the FTS index.

    Splits the user query into tokens and matches as an OR of
    individual terms — FTS5's default conjunction is AND, which
    would miss too much (queries like "Who is Israel Casa Rosa?"
    would require every word to appear). OR matching is the
    standard hybrid-search posture; ranking still favours
    documents that contain more of the query tokens via BM25.

    Filters on ``sensitivity_tier`` and (optionally) ``collection``
    mirror the vector search's ``where`` clause so the two ranking
    signals see the same candidate set.

    sensitivity_tier: 3
    """
    cleaned = sanitise_query(query)
    if not cleaned:
        return []
    tokens = cleaned.split()
    if not tokens:
        return []
    match = " OR ".join(tokens)

    where_clauses = [
        f"{FTS_TABLE} MATCH ?",
        "CAST(sensitivity_tier AS INTEGER) <= ?",
    ]
    params: list[Any] = [match, max_tier]
    if collections:
        placeholders = ",".join("?" for _ in collections)
        where_clauses.append(f"collection IN ({placeholders})")
        params.extend(collections)
    params.append(k)

    sql = (
        f"SELECT id, record_id, text, collection, layer, "
        f"sensitivity_tier, bm25({FTS_TABLE}) AS score "
        f"FROM {FTS_TABLE} "
        f"WHERE {' AND '.join(where_clauses)} "
        f"ORDER BY score ASC LIMIT ?"
    )
    try:
        rows = db.query(sql, params)
    except Exception as exc:  # noqa: BLE001
        logger.warning("bm25 query failed: %s", exc)
        return []

    return [
        BM25Hit(
            id=str(r["id"]),
            record_id=str(r["record_id"]),
            # Negate so higher = better, matching every other signal.
            score=-float(r["score"]),
            text=str(r.get("text") or ""),
            metadata={
                "collection": str(r.get("collection") or ""),
                "layer": str(r.get("layer") or ""),
                "sensitivity_tier": int(r.get("sensitivity_tier") or 0),
            },
        )
        for r in rows
    ]
