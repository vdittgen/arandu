"""ChromaDB embedded vector store engine for Arandu.

Provides a single persistent ChromaDB client with pre-defined domain
collections.  Uses Ollama's nomic-embed-text model for embeddings by default,
with automatic fallback to all-MiniLM-L6-v2 when Ollama is unavailable.

Persistent storage: ~/.arandu/data/chromadb/

sensitivity_tier: infrastructure layer — no user data stored here directly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb import Collection
from chromadb.api.types import EmbeddingFunction
from chromadb.config import Settings

from src.core.chromadb.embedding import (
    ChromaEmbeddingFunctionAdapter,
    OllamaEmbeddingFunction,
)
from src.core.chromadb.meta import check_compatibility, read_meta
from src.core.profiler import timed
from src.models.embedding_provider import (
    MODEL_DIMENSIONS,
    EmbeddingProvider,
    OllamaEmbeddingProvider,
    OpenAIEmbeddingProvider,
)

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path.home() / ".arandu" / "data" / "chromadb"

# Domain namespaces for the five vector collections.
COLLECTION_NAMES: list[str] = [
    "personal",
    "work",
    "health",
    "social",
    "ideas",
]


class VectorEngine:
    """Embedded ChromaDB vector store with domain-namespaced collections.

    One persistent client is shared across all collections.  ChromaDB manages
    its own connection pool internally, so no manual pooling is needed.

    sensitivity_tier: N/A (infrastructure layer — no user data stored here)
    """

    def __init__(
        self,
        db_path: Path = DEFAULT_DB_PATH,
        embedding_fn: EmbeddingFunction | EmbeddingProvider | None = None,
        *,
        prewarm_collections: bool = False,
    ) -> None:
        """Initialize the engine and open (or create) the ChromaDB store.

        Args:
            db_path: Filesystem directory for ChromaDB's persistent
                     storage.  Created automatically if it does not
                     exist.
            embedding_fn: Either a ChromaDB ``EmbeddingFunction``
                          (legacy) or an ``EmbeddingProvider`` (Phase 1+).
                          When a provider is passed, it's wrapped in
                          :class:`ChromaEmbeddingFunctionAdapter`
                          automatically. Defaults to the legacy
                          ``OllamaEmbeddingFunction`` to preserve the
                          existing index dimension until Phase 2 ships
                          the migration CLI.
            prewarm_collections: If True, create all domain collections
                                 on init.  If False (default),
                                 collections are created on first
                                 access via ``get_or_create_collection``.

        sensitivity_tier: N/A
        """
        self._db_path = db_path
        self._db_path.mkdir(parents=True, exist_ok=True)
        if embedding_fn is None:
            self._embedding_fn: EmbeddingFunction = (
                _default_embedding_fn_from_sentinel(db_path)
            )
        elif isinstance(embedding_fn, EmbeddingProvider):
            self._embedding_fn = ChromaEmbeddingFunctionAdapter(embedding_fn)
        else:
            self._embedding_fn = embedding_fn
        self._client: chromadb.ClientAPI = chromadb.PersistentClient(
            path=str(self._db_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self._check_embedding_compatibility()
        if prewarm_collections:
            self.warm_collections()
        logger.info("ChromaDB opened: %s", self._db_path)

    def _check_embedding_compatibility(self) -> None:
        """Compare the active embedder to the sentinel; warn on mismatch.

        Best-effort: never raises. A mismatch means the stored
        vectors were embedded by a different model and queries will
        return garbage until the user runs the migration CLI.

        sensitivity_tier: N/A
        """
        provider, model, dim = self._describe_active_embedder()
        if not provider:
            return
        try:
            mismatch = check_compatibility(
                self._db_path, provider, model, dim,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("embedding meta check failed: %s", exc)
            return
        if mismatch is None:
            return
        logger.warning(
            "ChromaDB embedding mismatch — stored=%s/%s (dim=%d), "
            "active=%s/%s (dim=%d). Run "
            "`python -m src.core.chromadb.migrate --to-model %s` to rebuild.",
            mismatch.provider, mismatch.model_name, mismatch.dimension,
            provider, model, dim,
            model,
        )

    def embedding_mismatch_message(self) -> str | None:
        """Actionable message when the active embedder differs from the
        stored index meta, else ``None``.

        A dimension/provider mismatch makes every upsert fail and is the
        usual cause of an empty vector store after a model/provider
        switch.  Callers surface this so the user knows to run the
        migration rather than seeing a generic "0 docs".

        sensitivity_tier: N/A
        """
        provider, model, dim = self._describe_active_embedder()
        if not provider:
            return None
        try:
            mismatch = check_compatibility(
                self._db_path, provider, model, dim,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("embedding meta check failed: %s", exc)
            return None
        if mismatch is None:
            return None
        return (
            f"Embedding model changed: index built with "
            f"{mismatch.provider}/{mismatch.model_name} "
            f"(dim {mismatch.dimension}), now using "
            f"{provider}/{model} (dim {dim}). Run "
            f"`python -m src.core.chromadb.migrate --to-model {model}` "
            f"to rebuild."
        )

    def _describe_active_embedder(self) -> tuple[str, str, int]:
        """Best-effort introspection of the embedder's identity.

        Returns ``("", "", 0)`` when the embedder doesn't expose
        enough metadata (e.g. an arbitrary user-supplied
        ``EmbeddingFunction``); the meta check is then skipped.

        sensitivity_tier: N/A
        """
        fn = self._embedding_fn
        # Phase 1+ adapter wraps an EmbeddingProvider.
        if isinstance(fn, ChromaEmbeddingFunctionAdapter):
            p = fn.provider
            return p.provider_name, p.model_name, p.dimension
        # Legacy OllamaEmbeddingFunction.
        if isinstance(fn, OllamaEmbeddingFunction):
            model = getattr(fn, "_model", "")
            dim = MODEL_DIMENSIONS.get(model, 0)
            return "ollama", model, dim
        return "", "", 0

    def warm_collections(self) -> None:
        """Pre-create all domain collections.

        Useful when you know all collections will be accessed and want
        to pay the embedding-function initialization cost up front.

        sensitivity_tier: N/A
        """
        for name in COLLECTION_NAMES:
            self._client.get_or_create_collection(
                name,
                embedding_function=self._embedding_fn,
            )
        logger.info("ChromaDB collections pre-warmed")

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def get_or_create_collection(self, name: str) -> Collection:
        """Return an existing collection or create it if absent.

        Args:
            name: Collection name (ideally one of COLLECTION_NAMES).

        Returns:
            The ChromaDB Collection object.
        """
        return self._client.get_or_create_collection(
            name,
            embedding_function=self._embedding_fn,
        )

    def get_collection_count(self, name: str) -> int:
        """Return the number of documents in a collection, or 0 if absent.

        Args:
            name: Collection name to check.

        Returns:
            Document count, or 0 if the collection does not exist.

        sensitivity_tier: N/A
        """
        try:
            col = self._client.get_collection(name)
            return col.count()
        except Exception:  # noqa: BLE001
            return 0

    # ------------------------------------------------------------------
    # Document operations
    # ------------------------------------------------------------------

    @timed()
    def add_documents(
        self,
        collection_name: str,
        documents: list[str],
        metadatas: list[dict[str, Any]],
        ids: list[str],
    ) -> None:
        """Embed and add documents to the named collection.

        Duplicate IDs are silently upserted (ChromaDB behaviour).

        Args:
            collection_name: Target collection (e.g. "personal", "work").
            documents: Raw text content to embed and store.
            metadatas: Per-document metadata dicts.  Each must include at
                       minimum: ``source``, ``timestamp``, ``sensitivity_tier``,
                       ``domain``.
            ids: Stable unique identifier for each document.

        Raises:
            ValueError: If the three lists are not the same length.
        """
        if not (len(documents) == len(metadatas) == len(ids)):
            raise ValueError("documents, metadatas, and ids must have the same length.")
        col = self.get_or_create_collection(collection_name)
        col.upsert(documents=documents, metadatas=metadatas, ids=ids)

    @timed()
    def search(
        self,
        collection_name: str,
        query: str,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search against a collection.

        Args:
            collection_name: Collection to query.
            query: Free-text query string; embedded on the fly.
            n_results: Maximum number of results to return.
            where: Optional ChromaDB metadata filter (MongoDB-style operators,
                   e.g. ``{"sensitivity_tier": {"$lte": 2}}``).

        Returns:
            List of result dicts, each with keys:
            ``id``, ``document``, ``metadata``, ``distance``.
            Ordered by ascending distance (most similar first).
        """
        col = self.get_or_create_collection(collection_name)

        # Clamp n_results to the number of documents in the collection so
        # ChromaDB does not raise when the collection is smaller than requested.
        count = col.count()
        if count == 0:
            return []
        effective_n = min(n_results, count)

        kwargs: dict[str, Any] = {
            "query_texts": [query],
            "n_results": effective_n,
        }
        if where:
            kwargs["where"] = where

        raw = col.query(**kwargs)

        results: list[dict[str, Any]] = []
        for i in range(len(raw["ids"][0])):
            results.append(
                {
                    "id": raw["ids"][0][i],
                    "document": raw["documents"][0][i],
                    "metadata": raw["metadatas"][0][i],
                    "distance": raw["distances"][0][i],
                }
            )
        return results

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release the ChromaDB client (no-op for the embedded client)."""
        # PersistentClient flushes on GC; explicit reset is a courtesy.
        self._client.clear_system_cache()
        logger.info("ChromaDB client released: %s", self._db_path)

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> VectorEngine:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Sentinel-driven default embedder (Phase 2)
# ---------------------------------------------------------------------------


def _default_embedding_fn_from_sentinel(db_path: Path) -> EmbeddingFunction:
    """Build the embedding function that matches the on-disk sentinel.

    The sentinel at ``<db_path>/.embedding_meta.json`` records the
    model that built the current index. Querying with a different
    model returns garbage (chroma rejects the upsert outright on
    dim mismatch, but queries with the wrong dim raise an error
    that the engine catches and logs — leading to silent zero
    recall). This helper ensures the default embedder always
    matches whatever is on disk.

    Recovery story: after the migration CLI writes a new sentinel,
    the next ``VectorEngine()`` automatically picks up the new
    model — no settings.json mutation, no manual wiring.

    Fallback chain (defensive):

    1. No sentinel → legacy ``OllamaEmbeddingFunction`` with
       ``nomic-embed-text`` (matches pre-Phase-2 behaviour).
    2. Sentinel ``provider == "ollama"`` → ``OllamaEmbeddingProvider``
       wrapped in the Chroma adapter.
    3. Sentinel ``provider == "openai"`` → ``OpenAIEmbeddingProvider``
       if an API key can be found in settings, else falls back to
       legacy (with a loud warning).

    sensitivity_tier: N/A
    """
    meta = read_meta(db_path)
    if meta is None:
        return OllamaEmbeddingFunction()

    if meta.provider == "ollama":
        return ChromaEmbeddingFunctionAdapter(
            OllamaEmbeddingProvider(model=meta.model_name),
        )

    if meta.provider == "openai":
        from src.models.embedding_provider import (
            _load_settings,
            resolve_embedding_base_url,
        )

        settings = _load_settings()
        api_key = (
            settings.get("embedding_api_key")
            or settings.get("llm_api_key")
            or settings.get("openai_api_key")
        )
        if not api_key:
            logger.warning(
                "sentinel asks for openai/%s but no API key in settings — "
                "falling back to legacy nomic-embed-text; queries will "
                "return zero recall until you set embedding_api_key.",
                meta.model_name,
            )
            return OllamaEmbeddingFunction()
        # Honour stored dimension as a truncation hint (matches the
        # text-embedding-3-large `dimensions=` parameter the migration
        # was run with).
        dimensions = meta.dimension if meta.dimension < 3072 else None
        return ChromaEmbeddingFunctionAdapter(
            OpenAIEmbeddingProvider(
                api_key=str(api_key),
                model=meta.model_name,
                base_url=resolve_embedding_base_url(settings),
                dimensions=dimensions,
            ),
        )

    logger.warning(
        "unknown sentinel provider %r; falling back to legacy embedder",
        meta.provider,
    )
    return OllamaEmbeddingFunction()
