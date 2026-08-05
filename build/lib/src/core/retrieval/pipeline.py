"""Hybrid retrieval pipeline — vector + BM25 → RRF → top-k.

End-to-end query path used by the eval runner (``--mode hybrid``)
and the Brain Agent (Phase 4 wires this into
:class:`src.core.query_engine.QueryEngine` as the default
``_vector_search`` replacement).

Sequence:

1. **Vector search** — :class:`VectorEngine.search` per collection,
   ``per_collection_k`` results each, sensitivity-tier filter.
2. **BM25 search** — :func:`src.core.retrieval.bm25.search` on the
   FTS5 mirror, ``bm25_k`` results total.
3. **RRF fusion** — :func:`src.core.retrieval.fusion.reciprocal_rank_fusion`
   combines the two rankings.
4. **Dedup by record** — chunk-level hits collapse to record-level
   so the caller sees distinct documents.
5. **Truncate to ``top_k``**.

Optional fields enriched on output: ``document`` (text), ``metadata``
dict, and the *origin* (``vector`` / ``bm25`` / ``both``) so the
caller can show provenance.

Reranking is deferred to a follow-up — adding a cross-encoder
(Cohere / bge-reranker) is the natural next step once we have a
hybrid baseline to measure against.

sensitivity_tier: 3
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.core.chromadb.engine import COLLECTION_NAMES, VectorEngine
from src.core.retrieval import bm25, retrieval_log
from src.core.retrieval.fusion import dedupe_by_record, reciprocal_rank_fusion
from src.core.sqlite.engine import DatabaseEngine

logger = logging.getLogger(__name__)

# Pull more candidates than we need so RRF has room to disagree with
# either signal. Vector pulls per-collection; BM25 pulls globally.
DEFAULT_PER_COLLECTION_K = 20
DEFAULT_BM25_K = 50
DEFAULT_TOP_K = 10


@dataclass
class HybridHit:
    """One result row from :meth:`HybridSearch.search`.

    sensitivity_tier: 3
    """

    id: str
    record_id: str
    score: float
    document: str
    metadata: dict[str, Any] = field(default_factory=dict)
    origin: str = "both"  # "vector" | "bm25" | "both"


class HybridSearch:
    """Vector + BM25 + RRF over the shared chunk corpus.

    Stateless aside from the two engine references. Cheap to
    construct per request, but the typical caller reuses one
    instance for the process lifetime.

    sensitivity_tier: 3
    """

    def __init__(
        self,
        chroma: VectorEngine,
        sqlite_db: DatabaseEngine,
        *,
        per_collection_k: int = DEFAULT_PER_COLLECTION_K,
        bm25_k: int = DEFAULT_BM25_K,
        rrf_k: int = 60,
        # BM25 enters as a tiebreaker (~0.5) rather than equal vote
        # so it boosts proper-noun precision without diluting
        # near-perfect semantic matches. Empirically this preserves
        # Phase 3's MRR / NDCG on the LLM-generated eval set while
        # still surfacing BM25-only matches when vector misses.
        bm25_weight: float = 0.5,
        vector_weight: float = 1.0,
        log_retrievals: bool = True,
    ) -> None:
        self._chroma = chroma
        self._db = sqlite_db
        self._per_collection_k = per_collection_k
        self._bm25_k = bm25_k
        self._rrf_k = rrf_k
        self._weights = (vector_weight, bm25_weight)
        self._log_retrievals = log_retrievals

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        max_tier: int = 3,
        collections: list[str] | None = None,
    ) -> list[HybridHit]:
        """Run the hybrid pipeline and return the top-k record-deduped hits.

        Empty query returns an empty list (no fan-out cost).
        Per-engine failures are logged and swallowed — a broken BM25
        index shouldn't kill the vector path, and vice-versa.

        Every non-empty call is appended to ``_retrieval_log`` (best
        effort — observability never blocks live retrieval).

        sensitivity_tier: 3
        """
        query = (query or "").strip()
        if not query:
            return []

        with retrieval_log.measure() as timer:
            targets = collections or list(COLLECTION_NAMES)
            vector_hits = self._vector_search(query, targets, max_tier)
            bm25_hits = self._bm25_search(query, targets, max_tier)

            corpus: dict[str, HybridHit] = {}
            for h in vector_hits:
                corpus[h.id] = h
            for h in bm25_hits:
                if h.id in corpus:
                    corpus[h.id].origin = "both"
                else:
                    corpus[h.id] = h

            fused = reciprocal_rank_fusion(
                [[h.id for h in vector_hits], [h.id for h in bm25_hits]],
                k=self._rrf_k,
                weights=list(self._weights),
            )
            deduped = dedupe_by_record(fused)

            out: list[HybridHit] = []
            for chunk_id, fused_score in deduped[:top_k]:
                hit = corpus.get(chunk_id)
                if hit is None:
                    continue
                hit.score = fused_score
                out.append(hit)

        if self._log_retrievals:
            retrieval_log.record(
                self._db,
                query=query,
                retrieved_ids=[h.id for h in out],
                scores=[h.score for h in out],
                latency_ms=timer.ms,
                mode="hybrid",
                embedding_model=self._embedding_model_name(),
                extra={
                    "vector_n": len(vector_hits),
                    "bm25_n": len(bm25_hits),
                    "fused_n": len(fused),
                    "max_tier": max_tier,
                    "collections": targets,
                },
            )
        return out

    def _embedding_model_name(self) -> str:
        """Best-effort lookup of the model id for log attribution.

        sensitivity_tier: 1
        """
        fn = getattr(self._chroma, "_embedding_fn", None)
        if fn is None:
            return ""
        if hasattr(fn, "provider"):
            return fn.provider.model_name
        return getattr(fn, "_model", type(fn).__name__)

    # ----------------------------------------------------------------
    # Internals — one fan-out per engine
    # ----------------------------------------------------------------

    def _vector_search(
        self,
        query: str,
        collections: list[str],
        max_tier: int,
    ) -> list[HybridHit]:
        where = {"sensitivity_tier": {"$lte": max_tier}}
        hits: list[HybridHit] = []
        for name in collections:
            try:
                results = self._chroma.search(
                    name, query, n_results=self._per_collection_k, where=where,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("vector search failed for %s: %s", name, exc)
                continue
            for r in results:
                cid = str(r["id"])
                hits.append(
                    HybridHit(
                        id=cid,
                        record_id=_strip_chunk(cid),
                        # 1 / (1 + distance) — vectorised by RRF; raw
                        # value here is only used as a fallback when
                        # the hit is later picked from the corpus dict.
                        score=1.0 / (1.0 + float(r.get("distance", 0.0))),
                        document=str(r.get("document") or ""),
                        metadata=dict(r.get("metadata") or {})
                        | {"collection": name},
                        origin="vector",
                    ),
                )
        # Sort within engine by score desc so the RRF rank reflects
        # cosine ordering even when results come from different
        # collections.
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits

    def _bm25_search(
        self,
        query: str,
        collections: list[str],
        max_tier: int,
    ) -> list[HybridHit]:
        rows = bm25.search(
            self._db, query,
            k=self._bm25_k,
            max_tier=max_tier,
            collections=collections,
        )
        return [
            HybridHit(
                id=r.id,
                record_id=r.record_id,
                score=r.score,
                document=r.text,
                metadata=r.metadata,
                origin="bm25",
            )
            for r in rows
        ]


def _strip_chunk(doc_id: str) -> str:
    """Mirror of :func:`fusion._strip_chunk` (avoid private import).

    sensitivity_tier: N/A
    """
    marker = "-chunk-"
    idx = doc_id.find(marker)
    return doc_id if idx < 0 else doc_id[:idx]
