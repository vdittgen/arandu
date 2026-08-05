"""Embedding-provider abstraction.

Mirrors :mod:`src.models.llm_provider` for embedding models. The
existing ``OllamaEmbeddingFunction`` at
``src/core/chromadb/embedding.py`` was bolted onto ChromaDB with no
abstraction — this module is the pluggable layer that lets the
firewall and migration CLI route embeddings the same way they route
chat.

Concrete providers shipped in this module:

* :class:`OllamaEmbeddingProvider` — local, default
  ``nomic-embed-text`` (preserves the existing index dimension; swap
  to ``bge-m3`` or ``mxbai-embed-large`` after running the migration
  CLI in Phase 2).
* :class:`OpenAIEmbeddingProvider` — remote ``text-embedding-3-large``
  (3072-dim by default; ``dimensions=`` truncation supported).
* :class:`VoyageEmbeddingProvider` — stub. Raises on use. Ship in a
  follow-up once we decide to take the ``voyageai`` SDK dep.

The factory :func:`create_embedding_provider_from_settings` reads
``~/.arandu/settings.json`` and selects a provider that matches the
chat provider's locality, so a user on local-only chat does not
silently get their embedding text sent to a cloud API.

Phase 1 ships the abstraction with **no behavior change**: defaults
preserve ``nomic-embed-text`` until the user opts into a different
model via settings + the Phase 2 migration CLI.

sensitivity_tier: 2 (handles text destined for embedding; classification
upstream of any remote egress is the firewall's job)
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import ollama

from src.models.llm_provider import SETTINGS_PATH

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_OLLAMA_MODEL = "nomic-embed-text"  # 768-dim. Existing index.
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OPENAI_MODEL = "text-embedding-3-large"  # 3072-dim.

# Known model dimensions. Used as a hint only — providers report the
# real dimension after first call. Helps the migration CLI estimate
# storage requirements before reindex.
MODEL_DIMENSIONS: dict[str, int] = {
    "nomic-embed-text": 768,
    "bge-m3": 1024,
    "mxbai-embed-large": 1024,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "voyage-3-large": 1024,
    "all-MiniLM-L6-v2": 384,
}

MAX_RETRIES = 3
BASE_DELAY_S = 1.0
OFFLINE_COOLDOWN_S = 60.0

# Per-request batch cap for the OpenAI-compatible ``/embeddings`` endpoint.
# Many providers reject 1024+ items; OpenAI's documented limit is 2048
# but real-world reliability degrades beyond a few hundred. 128 is a
# conservative cap that works with every provider we've tested and
# keeps a single failed batch's blast radius small during a full reindex.
DEFAULT_BATCH_CHUNK = 128


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class EmbeddingProvider(ABC):
    """Abstract base for embedding providers.

    Implementations must be safe to call from a single thread. The
    Chroma adapter at :mod:`src.core.chromadb.embedding` is the only
    expected entry point in production code — direct use is for
    migration, eval, and diagnostic tooling.

    sensitivity_tier: 2
    """

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents (storage-side).

        Some models (e.g. ``nomic-embed-text``) distinguish query and
        document modes via prefixes — implementations apply the right
        prefix internally so callers stay agnostic.

        sensitivity_tier: 2
        """

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string (retrieval-side).

        sensitivity_tier: 2
        """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier, e.g. ``"nomic-embed-text"`` or ``"bge-m3"``.

        sensitivity_tier: 1
        """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider identifier: ``"ollama"`` / ``"openai"`` / ``"voyage"``.

        sensitivity_tier: 1
        """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimensionality emitted by the model.

        Implementations may compute lazily on first embed if the
        model isn't in :data:`MODEL_DIMENSIONS`.

        sensitivity_tier: 1
        """


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Local Ollama embedding provider.

    Wraps the ``ollama.Client.embed`` call with retry + backoff and
    an offline-cooldown so a temporarily-unreachable Ollama doesn't
    burn the full retry budget on every subsequent call.

    sensitivity_tier: 2 (text never leaves the device)
    """

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        host: str = DEFAULT_OLLAMA_HOST,
        *,
        max_retries: int = MAX_RETRIES,
        base_delay: float = BASE_DELAY_S,
        offline_cooldown: float = OFFLINE_COOLDOWN_S,
    ) -> None:
        """Initialize.

        sensitivity_tier: 1
        """
        self._model = model
        self._host = host
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._offline_cooldown = max(0.0, offline_cooldown)
        self._client = ollama.Client(host=host)
        self._offline_until: float = 0.0
        self._cached_dim: int | None = MODEL_DIMENSIONS.get(model)

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def dimension(self) -> int:
        if self._cached_dim is not None:
            return self._cached_dim
        # Force a single embed to learn the dimension.
        vec = self.embed_query("dimension probe")
        self._cached_dim = len(vec)
        return self._cached_dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._embed(texts, query=False)

    def embed_query(self, text: str) -> list[float]:
        out = self._embed([text], query=True)
        return out[0] if out else []

    # -- internals -------------------------------------------------

    def _embed(
        self, texts: list[str], *, query: bool,  # noqa: ARG002
    ) -> list[list[float]]:
        """Embed via Ollama with retry, backoff, and offline cooldown.

        ``query`` is reserved for future per-model prefixing (e.g.
        nomic-embed-text uses ``search_query:`` vs ``search_document:``).
        Phase 1 keeps the prompt unchanged so the existing index stays
        valid; Phase 3 turns on prefixing alongside the indexer rewrite.

        sensitivity_tier: 2
        """
        if time.monotonic() < self._offline_until:
            raise EmbeddingUnavailableError(
                f"ollama unreachable; cooldown active for "
                f"{self._offline_until - time.monotonic():.0f}s",
            )
        for attempt in range(self._max_retries):
            try:
                resp = self._client.embed(model=self._model, input=texts)
                vecs = resp["embeddings"]
                if vecs and self._cached_dim is None:
                    self._cached_dim = len(vecs[0])
                return vecs
            except (ollama.ResponseError, ollama.RequestError) as exc:
                logger.warning(
                    "ollama embed failed (%d/%d): %s",
                    attempt + 1, self._max_retries, exc,
                )
                if _is_unreachable(exc):
                    break
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "ollama embed unexpected (%d/%d): %s",
                    attempt + 1, self._max_retries, exc,
                )
                if _is_unreachable(exc):
                    break
            if attempt < self._max_retries - 1:
                time.sleep(self._base_delay * (2**attempt))
        self._offline_until = time.monotonic() + self._offline_cooldown
        raise EmbeddingUnavailableError(
            f"ollama embed failed after {self._max_retries} attempts",
        )


# ---------------------------------------------------------------------------
# OpenAI / OpenAI-compatible
# ---------------------------------------------------------------------------


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Remote OpenAI (or OpenAI-compatible) embedding provider.

    Uses the same ``openai>=1.40.0`` SDK already in the
    ``[remote-llm]`` extra. Honours ``dimensions=`` so callers can
    truncate ``text-embedding-3-large`` from 3072 → 1024 to match
    the storage footprint of a local model.

    Egress posture: every batch is passed through
    :func:`redact_with_registry` before the API call so high-signal
    entities (names, emails, phone, money, dates) are replaced with
    stable placeholders before the request leaves the device. Doc
    embeddings and query embeddings share the same registry, so a
    name redacted to ``__PERSON_3__`` at index time matches the same
    placeholder at query time. Set ``redact=False`` only when the
    caller has already redacted upstream.

    sensitivity_tier: 2 (text egresses to the API, post-redaction)
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_OPENAI_MODEL,
        *,
        base_url: str | None = None,
        dimensions: int | None = None,
        redact: bool = True,
        batch_chunk: int = DEFAULT_BATCH_CHUNK,
    ) -> None:
        """Initialize.

        sensitivity_tier: 1
        """
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise EmbeddingUnavailableError(
                "openai SDK not installed — `pip install arandu[remote-llm]`",
            ) from exc
        self._model = model
        self._dimensions = dimensions
        # Same defenses as the agent / chat clients (see
        # ``src/agents/core/model_factory.py`` and the PR #240 repro):
        # connection pooling + no client-side timeout = the
        # "always hangs in the app, never in curl" bug. Embeddings
        # are on the brain's hot path (recall_context → embed query →
        # vector search) so a hung pool entry here freezes the whole
        # turn. Disable pool reuse + apply httpx split timeouts +
        # disable SDK retries so a single bounded failure beats a
        # stacked 30-min wait.
        import httpx as _httpx

        from src.models.llm_provider import (
            OPENAI_CONNECT_TIMEOUT_S,
            OPENAI_POOL_TIMEOUT_S,
            OPENAI_READ_TIMEOUT_S,
            OPENAI_WRITE_TIMEOUT_S,
        )
        embed_http = _httpx.Client(
            timeout=_httpx.Timeout(
                connect=OPENAI_CONNECT_TIMEOUT_S,
                read=OPENAI_READ_TIMEOUT_S,
                write=OPENAI_WRITE_TIMEOUT_S,
                pool=OPENAI_POOL_TIMEOUT_S,
            ),
            limits=_httpx.Limits(
                max_connections=10,
                max_keepalive_connections=0,
            ),
        )
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=embed_http,
            max_retries=0,
        )
        self._cached_dim: int | None = (
            dimensions or MODEL_DIMENSIONS.get(model)
        )
        self._redact = redact
        self._batch_chunk = max(1, batch_chunk)

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def dimension(self) -> int:
        if self._cached_dim is not None:
            return self._cached_dim
        vec = self.embed_query("dimension probe")
        self._cached_dim = len(vec)
        return self._cached_dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        out = self._embed([text])
        return out[0] if out else []

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts``, splitting into ``self._batch_chunk`` slices.

        Redacts each text through the persistent
        :class:`RedactionRegistry` so personal entities are replaced
        with stable placeholders before egress. The mapping is
        discarded — embeddings are vectors, not text, so there is
        nothing to rehydrate on the way back.

        Slicing matters because OpenAI-compatible providers cap each
        ``/embeddings`` call (often at 1024 or lower in practice). A
        full corpus reindex easily exceeds that cap —
        without chunking the single failed request bubbles up as
        ``EmbeddingUnavailableError`` and the Chroma adapter silently
        falls back to its 384-dim default, quietly corrupting the
        index. The OpenAI SDK already retries idempotent failures
        internally (configurable via ``max_retries=``); we don't add
        a second retry layer here.

        sensitivity_tier: 2
        """
        payload = self._redact_batch(texts) if self._redact else texts
        out: list[list[float]] = []
        for start in range(0, len(payload), self._batch_chunk):
            slice_ = payload[start:start + self._batch_chunk]
            kwargs: dict[str, Any] = {"model": self._model, "input": slice_}
            if self._dimensions is not None:
                kwargs["dimensions"] = self._dimensions
            try:
                resp = self._client.embeddings.create(**kwargs)
            except Exception as exc:  # noqa: BLE001
                raise EmbeddingUnavailableError(
                    f"openai embed failed at batch "
                    f"[{start}:{start + len(slice_)}] of {len(payload)}: "
                    f"{exc}",
                ) from exc
            out.extend(item.embedding for item in resp.data)
        if out and self._cached_dim is None:
            self._cached_dim = len(out[0])
        return out

    @staticmethod
    def _redact_batch(texts: list[str]) -> list[str]:
        """Run each text through the persistent redactor.

        The per-call :class:`RedactionMap` is dropped on the floor —
        embeddings cannot be rehydrated, and the persistent registry
        already records the (raw → placeholder) binding for future
        calls (chat egress + subsequent embed queries).

        sensitivity_tier: 1 (input is 2-3; redacted output is 1)
        """
        from src.models.redactor import redact_with_registry

        return [redact_with_registry(t)[0] for t in texts]


# ---------------------------------------------------------------------------
# Voyage AI (stub)
# ---------------------------------------------------------------------------


class VoyageEmbeddingProvider(EmbeddingProvider):
    """Stub for Voyage AI ``voyage-3-large``.

    Voyage typically tops MTEB-retrieval benchmarks but pulling in
    the SDK is a new vendor dependency. Ship as stub in Phase 1;
    light up in a follow-up after the rest of the overhaul lands.

    sensitivity_tier: 1
    """

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise NotImplementedError(
            "VoyageEmbeddingProvider is not yet implemented. "
            "Use OpenAIEmbeddingProvider until the voyageai dep lands.",
        )

    @property
    def model_name(self) -> str:  # pragma: no cover
        return "voyage-3-large"

    @property
    def provider_name(self) -> str:  # pragma: no cover
        return "voyage"

    @property
    def dimension(self) -> int:  # pragma: no cover
        return MODEL_DIMENSIONS["voyage-3-large"]

    def embed_documents(
        self, texts: list[str],  # noqa: ARG002
    ) -> list[list[float]]:  # pragma: no cover
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:  # pragma: no cover  # noqa: ARG002
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class EmbeddingUnavailableError(RuntimeError):
    """Raised when an embedding provider cannot fulfil a request.

    Distinct from a generic exception so the Chroma adapter can
    decide whether to fall back to the default ChromaDB embedder
    instead of crashing the indexer.

    sensitivity_tier: N/A
    """


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _load_settings() -> dict[str, Any]:
    """Read settings.json (same shape as llm_provider.load_llm_settings).

    Lives here as a tiny local helper so we don't depend on the LLM
    module's caching behaviour for embedding decisions.

    sensitivity_tier: 1
    """
    path = Path(SETTINGS_PATH)
    if not path.exists():
        return {}
    try:
        import json

        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to read settings.json: %s", exc)
        return {}


def resolve_embedding_base_url(settings: dict[str, Any]) -> str | None:
    """Pick the OpenAI-compatible ``base_url`` for embedding calls.

    Precedence:

    1. Explicit ``embedding_base_url`` in settings — always wins; lets
       a user point embeddings at a different OpenAI-compat host than
       their chat backend.
    2. ``llm_host`` when ``llm_provider == "openai_compat"`` — keeps
       embeddings on the same vendor as chat. The historic default of
       falling through to the openai SDK's ``api.openai.com`` was a
       silent footgun for users on third-party OpenAI-compat hosts
       whose ``llm_api_key`` was rejected by the wrong endpoint.
    3. ``None`` — let the openai SDK default (``api.openai.com``)
       win, which is correct for the literal OpenAI case.

    sensitivity_tier: 1
    """
    explicit = settings.get("embedding_base_url")
    if explicit:
        return str(explicit)
    chat_provider = settings.get("llm_provider")
    if chat_provider == "openai_compat":
        host = settings.get("llm_host")
        if host:
            return str(host)
    return None


def create_embedding_provider_from_settings() -> EmbeddingProvider:
    """Pick the right embedding provider based on settings.

    Decision matrix:

    +---------------------------+---------------------------+
    | Chat provider             | Embedding provider        |
    +===========================+===========================+
    | ollama                    | OllamaEmbeddingProvider   |
    +---------------------------+---------------------------+
    | openai_compat / anthropic | OllamaEmbeddingProvider   |
    | + ``embedding_remote...`` | (mirrors chat locality)   |
    | == false (default true)   |                           |
    +---------------------------+---------------------------+
    | openai_compat / anthropic | OpenAIEmbeddingProvider   |
    | + ``embedding_remote...`` | (host follows ``llm_host``|
    | == true                   |  — see                    |
    |                           |  :func:`resolve_embedding_|
    |                           |  base_url`)               |
    +---------------------------+---------------------------+

    Override knobs in settings.json:

    * ``embedding_provider``: explicit ``"ollama" | "openai"``
      override; bypasses the chat-mirroring rule.
    * ``embedding_model_local``: model name for Ollama (default
      ``nomic-embed-text`` so the existing index stays valid).
    * ``embedding_model_remote``: model name for OpenAI (default
      ``text-embedding-3-large``).
    * ``embedding_remote_when_chat_remote``: bool, default ``True``.
    * ``embedding_base_url``: explicit OpenAI-compat host; otherwise
      derived from ``llm_host`` when ``llm_provider == "openai_compat"``
      so embeddings land at the same vendor as chat.
    * ``embedding_dimensions``: optional truncation for OpenAI.

    Falls back to Ollama if a remote provider is selected but its
    API key is missing — same posture as
    :func:`create_provider_from_settings`.

    sensitivity_tier: 1
    """
    settings = _load_settings()

    explicit = settings.get("embedding_provider")
    if explicit == "ollama":
        return _build_ollama(settings)
    if explicit == "openai":
        return _build_openai_or_fallback(settings)

    chat_provider = settings.get("llm_provider", "ollama")
    mirror_remote = bool(
        settings.get("embedding_remote_when_chat_remote", True),
    )
    if chat_provider in ("openai_compat", "anthropic") and mirror_remote:
        return _build_openai_or_fallback(settings)
    return _build_ollama(settings)


def _build_ollama(settings: dict[str, Any]) -> OllamaEmbeddingProvider:
    """Build an Ollama embedder honouring local-host overrides.

    sensitivity_tier: 1
    """
    model = settings.get("embedding_model_local", DEFAULT_OLLAMA_MODEL)
    # Reuse llm_host only when it points at Ollama (localhost).
    host = settings.get("ollama_host") or DEFAULT_OLLAMA_HOST
    return OllamaEmbeddingProvider(model=model, host=host)


def _build_openai_or_fallback(settings: dict[str, Any]) -> EmbeddingProvider:
    """Build an OpenAI embedder; fall back to Ollama if key missing.

    sensitivity_tier: 1
    """
    api_key = (
        settings.get("embedding_api_key")
        or settings.get("llm_api_key")
        or settings.get("openai_api_key")
    )
    if not api_key:
        logger.warning(
            "remote embedding requested but no API key — falling back to Ollama",
        )
        return _build_ollama(settings)
    model = settings.get("embedding_model_remote", DEFAULT_OPENAI_MODEL)
    dimensions = settings.get("embedding_dimensions")
    base_url = resolve_embedding_base_url(settings)
    try:
        return OpenAIEmbeddingProvider(
            api_key=api_key,
            model=model,
            base_url=base_url,
            dimensions=dimensions,
        )
    except EmbeddingUnavailableError:
        logger.warning("openai SDK unavailable — falling back to Ollama")
        return _build_ollama(settings)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


_UNREACHABLE_MARKERS = (
    "failed to connect",
    "connection refused",
    "max retries exceeded",
    "timed out",
    "name or service not known",
    "temporary failure in name resolution",
    "nodename nor servname provided",
)


def _is_unreachable(exc: Exception) -> bool:
    """Heuristic check for network-unreachable failures.

    sensitivity_tier: N/A
    """
    msg = str(exc).lower()
    return any(marker in msg for marker in _UNREACHABLE_MARKERS)
