"""Rank-fusion primitives.

Phase 4 wires vector + BM25 together through reciprocal-rank fusion
(RRF). The functions here are pure — no engines, no I/O — so they
can be exhaustively unit-tested and reused by future ranking
signals (graph proximity, recency boost, …).

sensitivity_tier: N/A
"""

from __future__ import annotations

from collections import defaultdict


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    *,
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """Combine multiple ranked ID lists via RRF.

    Each list is treated as one ranking; the per-list contribution
    for item ``i`` at rank ``r`` (1-based) is ``weight / (k + r)``.
    Items absent from a list contribute 0 from that list. Returns
    ``[(id, score), ...]`` sorted by score descending.

    The default ``k=60`` is the value Cormack et al. recommended in
    the original RRF paper — robust across very different signal
    magnitudes (cosine distance vs BM25 score) without needing to
    normalise either side first.

    ``weights`` lets you bias one signal over another (e.g.
    ``[1.0, 0.5]`` halves the BM25 contribution). Defaults to 1.0
    per list. Length must match ``ranked_lists`` if provided.

    sensitivity_tier: N/A
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError(
            f"weights length {len(weights)} != ranked_lists length "
            f"{len(ranked_lists)}",
        )

    scores: dict[str, float] = defaultdict(float)
    for ranking, weight in zip(ranked_lists, weights, strict=True):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += weight / (k + rank)

    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def dedupe_by_record(
    ranked: list[tuple[str, float]],
    chunk_to_record: dict[str, str] | None = None,
) -> list[tuple[str, float]]:
    """Keep only the best-scoring chunk per source record.

    Hybrid search frequently surfaces multiple chunks of the same
    document (especially with overlap > 0); the caller usually wants
    distinct *records* in the top-k. Preserves input order for ties.

    ``chunk_to_record`` maps a chunk-suffixed id to its record id.
    When absent, we strip ``-chunk-N`` suffixes ourselves so
    callers can pass through the chunk ids directly.

    sensitivity_tier: N/A
    """
    seen: set[str] = set()
    out: list[tuple[str, float]] = []
    for chunk_id, score in ranked:
        record_id = (
            chunk_to_record.get(chunk_id, chunk_id)
            if chunk_to_record else _strip_chunk(chunk_id)
        )
        if record_id in seen:
            continue
        seen.add(record_id)
        out.append((chunk_id, score))
    return out


def _strip_chunk(doc_id: str) -> str:
    """Drop the ``-chunk-N`` suffix from a doc id.

    Mirrors :func:`evals.retrieval.metrics.normalise_id` so the two
    modules can't drift.

    sensitivity_tier: N/A
    """
    marker = "-chunk-"
    idx = doc_id.find(marker)
    return doc_id if idx < 0 else doc_id[:idx]
