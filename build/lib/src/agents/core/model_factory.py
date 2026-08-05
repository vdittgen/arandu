"""Build pydantic-ai ``Model`` instances from user settings.

Routes:

- **local** — Ollama on the user's machine, fronted by its
  OpenAI-compatible endpoint. The only route used in Arandu.
- **remote** — A reserved extension-point route. Falls through to
  local here.

The factory is lazy: pydantic-ai imports are deferred until first call
so ``pyproject.toml`` install order doesn't matter for unrelated test
modules.

sensitivity_tier: 1
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SETTINGS_PATH = Path.home() / ".arandu" / "settings.json"

# Defaults are placeholders; the real provider URL + model name come from
# user settings written by the AI Model section of the settings page.
DEFAULT_REMOTE_MODEL = "llama3.1:70b"
DEFAULT_REMOTE_BASE_URL = "http://localhost:11434/v1"
DEFAULT_LOCAL_MODEL = "llama3.1:70b"
DEFAULT_LOCAL_BASE_URL = "http://localhost:11434/v1"


@dataclass(frozen=True)
class ModelEndpoint:
    """Settings needed to instantiate one ``pydantic_ai`` model.

    sensitivity_tier: 1
    """

    route: str  # "remote" | "local"
    base_url: str
    model_name: str
    api_key: str | None


def _load_settings() -> dict[str, Any]:
    if not SETTINGS_PATH.exists():
        return {}
    try:
        with SETTINGS_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read settings: %s", exc)
        return {}


def remote_endpoint() -> ModelEndpoint:
    """Endpoint for the configured remote provider.

    In Arandu the remote route is never taken (the egress firewall
    keeps every call local); it remains as a reserved extension point.

    sensitivity_tier: 1
    """
    settings = _load_settings()
    return ModelEndpoint(
        route="remote",
        base_url=settings.get(
            "llm_remote_base_url",
            settings.get("llm_host", DEFAULT_REMOTE_BASE_URL),
        ),
        model_name=settings.get(
            "llm_remote_model",
            settings.get("llm_model", DEFAULT_REMOTE_MODEL),
        ),
        api_key=settings.get(
            "llm_remote_api_key",
            settings.get("llm_api_key")
            or os.environ.get("ARANDU_REMOTE_API_KEY"),
        ),
    )


def local_endpoint() -> ModelEndpoint:
    """Endpoint for the local Ollama fallback / Tier-3 router target.

    sensitivity_tier: 1
    """
    settings = _load_settings()
    return ModelEndpoint(
        route="local",
        base_url=settings.get(
            "llm_local_base_url", DEFAULT_LOCAL_BASE_URL,
        ),
        # Arandu has a single user-facing model setting (`llm_model`, set by
        # the onboarding wizard / Settings). Agents always take the local
        # route, so honour `llm_model` here; `llm_local_model` stays an
        # optional explicit override for advanced setups.
        model_name=settings.get("llm_local_model")
        or settings.get("llm_model")
        or DEFAULT_LOCAL_MODEL,
        # Ollama's OpenAI-compatible endpoint accepts any bearer token.
        api_key=settings.get("llm_local_api_key", "ollama"),
    )


class ModelFactory:
    """Lazy builder for pydantic-ai models keyed by route + override.

    Cached per (route, model_override) pair. Resolution failures
    (missing pydantic-ai install or bad endpoint settings) raise.

    sensitivity_tier: 1
    """

    def __init__(self) -> None:
        # Each entry pairs the built model with the endpoint it was
        # built for, so `get` can rebuild when the resolved endpoint
        # drifts (rotated credentials, edited settings).
        self._cache: dict[
            tuple[str, str | None], tuple[Any, ModelEndpoint],
        ] = {}

    def endpoint_for(
        self,
        route: str,
        *,
        model_override: str | None = None,
    ) -> ModelEndpoint:
        """Return the endpoint that would be used for ``route``.

        When ``model_override`` is provided, the endpoint's ``model_name``
        is replaced with the override while preserving base_url + api_key.

        sensitivity_tier: 1
        """
        if route not in {"remote", "local"}:
            msg = f"Unknown route: {route!r}"
            raise ValueError(msg)
        base = remote_endpoint() if route == "remote" else local_endpoint()
        if not model_override:
            return base
        return ModelEndpoint(
            route=base.route,
            base_url=base.base_url,
            model_name=model_override,
            api_key=base.api_key,
        )

    def get(self, route: str, *, model_override: str | None = None) -> Any:
        """Return a pydantic-ai ``Model`` for ``route``.

        When ``model_override`` is set, the model name is replaced but
        the route's base_url + api_key are preserved.

        The endpoint is re-resolved on every call and the cached model
        rebuilt when it no longer matches. A long-lived process must
        not keep using credentials or settings that have since changed
        — e.g. a bearer token with an hourly expiry stays pinned in a
        forever-cached model and every call starts failing with 401s
        until the process restarts.

        sensitivity_tier: 1
        """
        key = (route, model_override or None)
        endpoint = self.endpoint_for(route, model_override=model_override)
        cached = self._cache.get(key)
        if cached is not None and cached[1] == endpoint:
            return cached[0]
        model = _build_pydantic_ai_model(endpoint)
        self._cache[key] = (model, endpoint)
        return model

    def reset(self) -> None:
        """Drop cached models — for tests that swap settings.

        sensitivity_tier: 1
        """
        self._cache.clear()


def _build_pydantic_ai_model(endpoint: ModelEndpoint) -> Any:
    """Instantiate the pydantic-ai model corresponding to ``endpoint``.

    Import is deferred to keep core imports cheap and avoid hard-failing
    when ``pydantic-ai-slim`` isn't installed during unrelated tests.

    sensitivity_tier: 1
    """
    try:
        from pydantic_ai.models.openai import OpenAIModel  # type: ignore
        from pydantic_ai.providers.openai import OpenAIProvider  # type: ignore
    except ImportError as exc:  # pragma: no cover
        # Distinguish "absent" from "present but incompatible". The two
        # need opposite fixes, and reporting a version mismatch as "not
        # installed" sends you hunting for a missing package that is
        # sitting right there. pydantic-ai 2.x renamed ``OpenAIModel``
        # to ``OpenAIChatModel``, so an unpinned install lands here.
        import importlib.util

        if importlib.util.find_spec("pydantic_ai") is None:
            msg = (
                "pydantic-ai-slim[openai] is not installed; "
                "agent base classes are unavailable"
            )
        else:
            installed = "unknown"
            try:
                from importlib.metadata import version

                installed = version("pydantic-ai-slim")
            except Exception:  # noqa: BLE001 - best-effort diagnostics
                pass
            msg = (
                "pydantic-ai-slim is installed "
                f"(version {installed}) but does not expose the API this "
                "build targets — Arandu requires the 1.x line "
                "(pydantic-ai-slim>=1.104.0,<2). Reinstall with "
                "'pip install \"pydantic-ai-slim[openai]>=1.104.0,<2\"'."
            )
        raise RuntimeError(msg) from exc

    from src.models._openai_schema_compat import (
        install_schema_compat_patch,
    )
    install_schema_compat_patch()

    # The agent loop (chat / brain / sub-agents) runs through this
    # pydantic-ai model. Without an explicit http client, the default
    # has no read timeout on the openai SDK call — a silently-stalled
    # upstream pins the UI on ask_brain forever. Mirror the
    # ``OpenAICompatibleProvider`` timeout (10s connect / 120s read /
    # 10s write / 10s pool) so an interactive turn fails in ≤2 min
    # instead of hanging indefinitely.
    import httpx as _httpx
    import openai as _openai

    from src.models.llm_provider import (
        OPENAI_CONNECT_TIMEOUT_S,
        OPENAI_POOL_TIMEOUT_S,
        OPENAI_READ_TIMEOUT_S,
        OPENAI_WRITE_TIMEOUT_S,
    )

    timeout = _httpx.Timeout(
        connect=OPENAI_CONNECT_TIMEOUT_S,
        read=OPENAI_READ_TIMEOUT_S,
        write=OPENAI_WRITE_TIMEOUT_S,
        pool=OPENAI_POOL_TIMEOUT_S,
    )

    # Connection pooling is the real culprit behind the "always
    # hangs in the app, never in curl" reports. The openai SDK
    # caches HTTPS connections; when one sits idle past the local
    # NAT / firewall's connection-tracking timeout, the entry is
    # silently evicted with no TCP RST sent to us. The next call
    # grabs the pooled fd, writes the request into a black hole,
    # and waits for a response that never arrives — only the
    # read-timeout breaks the spell.
    # Disable keep-alive entirely on the agent client: every chat
    # completion gets a fresh TCP+TLS handshake. The extra ~150ms
    # is invisible against multi-second LLM latency and turns a
    # silently-dead pool entry into an impossible failure mode.
    limits = _httpx.Limits(
        max_connections=10,
        max_keepalive_connections=0,
    )
    async_http = _httpx.AsyncClient(timeout=timeout, limits=limits)

    # The openai SDK retries ``APITimeoutError`` up to
    # ``max_retries=2`` by default (3 total attempts). On a stalled
    # upstream that stacks to 3 × read-timeout ≈ 6 min before
    # surfacing — exactly the symptom this change is fixing. Disable
    # SDK-level retries: the user is already watching an interactive
    # turn, and one bounded failure beats a quietly tripled wait.
    # OpenAICompatibleProvider has its own retry loop for non-timeout
    # errors; this path is the agent loop where the timeout is a hard
    # ceiling.
    async_openai = _openai.AsyncOpenAI(
        base_url=endpoint.base_url,
        api_key=endpoint.api_key or "missing",
        timeout=timeout,
        max_retries=0,
        http_client=async_http,
    )
    provider = OpenAIProvider(openai_client=async_openai)
    return OpenAIModel(endpoint.model_name, provider=provider)


@lru_cache(maxsize=1)
def default_factory() -> ModelFactory:
    """Return the process-wide ``ModelFactory``.

    sensitivity_tier: 1
    """
    return ModelFactory()


# Prefixes treated as "chat-likely" — sorted to the top of the model
# list so the override dropdown's first results are the relevant ones.
_CHAT_PREFIXES: tuple[str, ...] = (
    "deepseek-ai/",
    "Qwen/",
    "meta-llama/",
    "mistralai/",
    "google/gemma",
    "google/gemini",
    "microsoft/",
    "01-ai/",
    "nvidia/",
    "claude-",
    "gpt-",
    "o1-",
    "o3-",
    "o4-",
)

# Substrings that strongly suggest a non-chat modality. The OpenAI
# /models endpoint doesn't expose task type, so we infer from the id.
# Case-insensitive substring match in :func:`_chat_rank`.
_NON_CHAT_HINTS: tuple[str, ...] = (
    "image",
    "embedding",
    "whisper",
    "tts",
    "kokoro",
    "xtts",
    "reranker",
    "rerank",
    "ocr",
    "flux",
    "sdxl",
    "stable-diffusion",
    "clip",
)


def _chat_rank(model_id: str) -> int:
    """Return a sort bucket: 0 chat-likely, 1 unknown, 2 likely non-chat.

    Used as the primary sort key so the most useful models float to
    the top of the dropdown. Two-pass:

    - First check for an unambiguous non-chat substring in the id
      (image / embedding / whisper / etc.). Demote to bucket 2.
    - Otherwise, check a known chat-family prefix. Bucket 0.
    - Otherwise, bucket 1 (unknown — keep visible but below chat).

    sensitivity_tier: 1
    """
    lowered = model_id.lower()
    if any(hint in lowered for hint in _NON_CHAT_HINTS):
        return 2
    if model_id.startswith(_CHAT_PREFIXES):
        return 0
    return 1


def list_models(route: str) -> list[str]:
    """Return all model ids the ``route``'s endpoint exposes.

    Hits ``{base_url}/models`` via the OpenAI SDK (every supported
    backend speaks this standard endpoint). The list is sorted with
    chat-likely families first so dropdowns surface relevant models
    without the user scrolling past image / audio / embedding entries.

    Accepts ``"remote"``, ``"local"``, or ``"inherit"`` (falls back to
    ``"remote"``). Raises ``RuntimeError`` if the underlying call fails
    so callers can surface a useful error to the UI.

    sensitivity_tier: 1
    """
    resolved_route = "remote" if route == "inherit" else route
    if resolved_route not in {"remote", "local"}:
        msg = f"Unknown route: {route!r}"
        raise ValueError(msg)
    endpoint = (
        remote_endpoint() if resolved_route == "remote"
        else local_endpoint()
    )
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        msg = "openai SDK not installed; cannot list models"
        raise RuntimeError(msg) from exc
    try:
        client = OpenAI(
            base_url=endpoint.base_url,
            api_key=endpoint.api_key or "missing",
        )
        response = client.models.list()
    except Exception as exc:  # noqa: BLE001
        msg = f"failed to list models at {endpoint.base_url}: {exc}"
        raise RuntimeError(msg) from exc
    ids = [
        m.id for m in response.data
        if getattr(m, "id", None)
    ]
    return sorted(ids, key=lambda mid: (_chat_rank(mid), mid))


__all__ = [
    "DEFAULT_LOCAL_MODEL",
    "DEFAULT_REMOTE_MODEL",
    "ModelEndpoint",
    "ModelFactory",
    "default_factory",
    "list_models",
    "local_endpoint",
    "remote_endpoint",
]
