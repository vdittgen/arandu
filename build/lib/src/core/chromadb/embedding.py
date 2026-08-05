"""Chroma-side embedding adapters.

Holds two classes:

* :class:`OllamaEmbeddingFunction` — the legacy direct-to-Ollama
  embedding function. Kept intact so existing callers (engine
  default, tests) keep working unchanged. ``nomic-embed-text`` with
  fallback to ChromaDB's default ``all-MiniLM-L6-v2``.
* :class:`ChromaEmbeddingFunctionAdapter` — Phase 1 adapter that
  wraps any :class:`src.models.embedding_provider.EmbeddingProvider`
  in the :class:`EmbeddingFunction` interface ChromaDB expects.
  Opt-in: ``VectorEngine`` only uses it when the caller passes an
  ``EmbeddingProvider`` explicitly. Phase 2 will flip the default
  via settings + the migration CLI.

sensitivity_tier: N/A (infrastructure — processes text for embedding only)
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import ollama
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

if TYPE_CHECKING:
    from src.models.embedding_provider import EmbeddingProvider

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "nomic-embed-text"
DEFAULT_HOST = "http://localhost:11434"
MAX_RETRIES = 3
BASE_DELAY_S = 1.0
OFFLINE_COOLDOWN_S = 60.0


class OllamaEmbeddingFunction(EmbeddingFunction[Documents]):
    """Embed documents using Ollama's nomic-embed-text model.

    Falls back to ChromaDB's default all-MiniLM-L6-v2 embedding function
    when Ollama is unreachable after all retry attempts.

    sensitivity_tier: N/A (infrastructure layer)
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        max_retries: int = MAX_RETRIES,
        base_delay: float = BASE_DELAY_S,
        offline_cooldown: float = OFFLINE_COOLDOWN_S,
    ) -> None:
        """Initialize the embedding function.

        Args:
            model: Ollama embedding model name.
            host: Ollama server URL.
            max_retries: Maximum retry attempts on failure.
            base_delay: Base delay in seconds for exponential backoff.
            offline_cooldown: Seconds to skip Ollama attempts after
                             detecting an unreachable server.
        """
        self._model = model
        self._host = host
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._offline_cooldown = max(0.0, offline_cooldown)
        self._client = ollama.Client(host=host)
        self._fallback: DefaultEmbeddingFunction | None = None
        self._last_used_ollama: bool = False
        self._offline_until: float = 0.0

    def __call__(self, input: Documents) -> Embeddings:
        """Embed a batch of documents.

        Attempts Ollama first with retry+backoff.  On complete failure,
        falls back to the default ChromaDB embedding function.

        Args:
            input: List of text strings to embed.

        Returns:
            List of embedding vectors.

        sensitivity_tier: N/A
        """
        if not input:
            self._last_used_ollama = False
            return []  # type: ignore[return-value]

        if time.monotonic() < self._offline_until:
            self._last_used_ollama = False
            return self._embed_via_fallback(input)

        result = self._embed_via_ollama(input)
        if result is not None:
            self._offline_until = 0.0
            self._last_used_ollama = True
            return result

        logger.warning(
            "Ollama embedding failed after %d attempts — using fallback",
            self._max_retries,
        )
        self._offline_until = time.monotonic() + self._offline_cooldown
        self._last_used_ollama = False
        return self._embed_via_fallback(input)

    def _embed_via_ollama(self, texts: Documents) -> Embeddings | None:
        """Attempt to embed via Ollama with retry logic.

        Returns None if all attempts fail.

        sensitivity_tier: N/A
        """
        for attempt in range(self._max_retries):
            try:
                response = self._client.embed(
                    model=self._model,
                    input=texts,
                )
                return response["embeddings"]

            except (ollama.ResponseError, ollama.RequestError) as exc:
                logger.warning(
                    "Ollama embed failed (attempt %d/%d): %s",
                    attempt + 1,
                    self._max_retries,
                    exc,
                )
                if self._is_unreachable_error(exc):
                    break
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Unexpected error from Ollama embed (attempt %d/%d): %s",
                    attempt + 1,
                    self._max_retries,
                    exc,
                )
                if self._is_unreachable_error(exc):
                    break

            if attempt < self._max_retries - 1:
                delay = self._base_delay * (2**attempt)
                time.sleep(delay)

        return None

    @staticmethod
    def _is_unreachable_error(exc: Exception) -> bool:
        """Heuristic check for network-unreachable Ollama failures.

        sensitivity_tier: N/A
        """
        msg = str(exc).lower()
        markers = (
            "failed to connect",
            "connection refused",
            "max retries exceeded",
            "timed out",
            "name or service not known",
            "temporary failure in name resolution",
            "nodename nor servname provided",
        )
        return any(marker in msg for marker in markers)

    def _embed_via_fallback(self, texts: Documents) -> Embeddings:
        """Embed using ChromaDB's default all-MiniLM-L6-v2.

        sensitivity_tier: N/A
        """
        if self._fallback is None:
            self._fallback = DefaultEmbeddingFunction()
        return self._fallback(texts)

    @property
    def is_using_ollama(self) -> bool:
        """True if the last embedding call used Ollama successfully."""
        return self._last_used_ollama


# ---------------------------------------------------------------------------
# Phase 1 — EmbeddingProvider adapter
# ---------------------------------------------------------------------------


class ChromaEmbeddingFunctionAdapter(EmbeddingFunction[Documents]):
    """Wrap any :class:`EmbeddingProvider` as a ChromaDB embedding fn.

    ChromaDB collections hold a reference to an
    :class:`EmbeddingFunction` and invoke it on every ``upsert`` /
    ``query``. This adapter is the seam between the provider
    abstraction and ChromaDB's interface — pass an
    ``EmbeddingProvider`` to :class:`VectorEngine` and the engine
    wraps it in this adapter.

    On any :class:`EmbeddingUnavailableError` from the provider, the
    adapter falls back to ChromaDB's default ``all-MiniLM-L6-v2``
    embedder so an offline Ollama doesn't crash the indexer — same
    behaviour as the legacy ``OllamaEmbeddingFunction``.

    sensitivity_tier: 2 (forwards text to whatever provider it wraps)
    """

    def __init__(self, provider: EmbeddingProvider) -> None:
        """Hold a reference to the provider.

        sensitivity_tier: 1
        """
        self._provider = provider
        self._fallback: DefaultEmbeddingFunction | None = None
        self._last_used_provider: bool = False

    def __call__(self, input: Documents) -> Embeddings:
        """Embed via the wrapped provider; fall back on failure.

        sensitivity_tier: 2
        """
        if not input:
            self._last_used_provider = False
            return []  # type: ignore[return-value]
        from src.models.embedding_provider import EmbeddingUnavailableError

        try:
            vecs = self._provider.embed_documents(list(input))
            self._last_used_provider = True
            return vecs  # type: ignore[return-value]
        except EmbeddingUnavailableError as exc:
            logger.warning(
                "embedding provider %s unavailable — falling back: %s",
                self._provider.provider_name, exc,
            )
            self._last_used_provider = False
            return self._embed_via_fallback(input)

    def _embed_via_fallback(self, texts: Documents) -> Embeddings:
        """Embed using ChromaDB's default all-MiniLM-L6-v2.

        sensitivity_tier: N/A
        """
        if self._fallback is None:
            self._fallback = DefaultEmbeddingFunction()
        return self._fallback(texts)

    @property
    def provider(self) -> EmbeddingProvider:
        """The wrapped provider (for diagnostics and dimension probes).

        sensitivity_tier: 1
        """
        return self._provider

    @property
    def model_name(self) -> str:
        """Pass-through for the runner's report header.

        sensitivity_tier: 1
        """
        return self._provider.model_name

    @property
    def is_using_provider(self) -> bool:
        """True if the last call hit the provider (not the fallback).

        sensitivity_tier: N/A
        """
        return self._last_used_provider
