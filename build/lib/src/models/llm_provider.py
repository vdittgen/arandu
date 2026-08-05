"""Local LLM provider abstraction (Ollama).

Factory + provider class that normalizes the chat API. Components use
``create_provider_from_settings()`` instead of importing ``ollama``
directly. All inference runs locally on Ollama.

Embeddings are NOT handled here — see :mod:`src.models.embedding_provider`.
The embedding provider mirrors chat locality: embeddings run on the
same local Ollama backend (the migration CLI records the model in the
ChromaDB sentinel so the indexer reloads the right embedder on
restart).

sensitivity_tier: 1 (only model names and provider type, no user data)
"""

from __future__ import annotations

import json
import logging
import re as _re
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Retry defaults (consolidating logic from BrainAgent / EmotionalLabeler)
MAX_RETRIES = 3
BASE_DELAY_S = 1.0

# Hard wall-clock timeout per Ollama request (seconds).
# httpx read-timeout resets on each data chunk, so a slowly-streaming
# stuck response can hang forever.  This wraps the call in a thread
# with a non-resettable deadline.
OLLAMA_WALL_TIMEOUT_S = 120

# Fine-grained timeouts for the OpenAI-compatible (remote) provider.
# A scalar timeout of e.g. 600s blocks the UI for 10 minutes when the
# upstream silently hangs — connection alive, no response chunks. Use
# httpx-style split timeouts to fail fast on the dead-connection case
# while still allowing genuinely long reasoning to finish.
#   - connect: hostname → TCP handshake
#   - read:    per-chunk wait once the request is in flight
#   - write:   uploading the request body
#   - pool:    waiting for a free connection in the pool
OPENAI_CONNECT_TIMEOUT_S = 10.0
OPENAI_READ_TIMEOUT_S = 120.0
OPENAI_WRITE_TIMEOUT_S = 10.0
OPENAI_POOL_TIMEOUT_S = 10.0

# Hard wall-clock timeout for the OpenAI-compatible provider, mirroring
# OLLAMA_WALL_TIMEOUT_S.  httpx's read timeout resets on every chunk,
# so a trickling-but-never-ending response (or a stalled non-streaming
# call past headers) can keep the worker alive indefinitely.  This
# bounds the whole call.
OPENAI_WALL_TIMEOUT_S = 120

# Anthropic response length cap
DEFAULT_MAX_TOKENS = 4096

SETTINGS_PATH = Path.home() / ".arandu" / "settings.json"

_THINKING_RE = _re.compile(r"<think>.*?</think>", _re.DOTALL)
_FENCE_RE = _re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", _re.DOTALL)


def _strip_json_wrapper(text: str) -> str:
    """Strip thinking tags and markdown fences from LLM output.

    sensitivity_tier: 1
    """
    text = _THINKING_RE.sub("", text).strip()
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PreemptedError(Exception):
    """Raised when a background LLM call is preempted by interactive."""


class LLMWallTimeoutError(TimeoutError):
    """Raised when an LLM call exceeds its wall-clock deadline.

    Distinct subclass so callers can differentiate a hard hang
    (we cut the call) from a generic ``TimeoutError`` bubbling up
    from the underlying HTTP layer.
    """


def _run_with_deadline(
    fn: Any,
    *,
    timeout: float,
    on_abort: Any,
    poll_interval_s: float = 2.0,
) -> Any:
    """Run *fn* in a worker thread under a hard wall-clock deadline.

    Polls with ``time.time()`` (survives macOS sleep, unlike a blocking
    ``future.result(timeout=)`` which sits on ``pthread_cond_timedwait``
    and stalls through a sleep/wake cycle).  On timeout, calls
    ``on_abort(pool)`` so the caller can close the HTTP transport
    before raising — otherwise the worker thread keeps running and the
    next call inherits a wedged client.

    ``on_abort`` MUST be idempotent and non-blocking: it should not
    wait on the worker thread.  The pool is shut down with
    ``wait=False`` regardless.

    sensitivity_tier: 1
    """
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(fn)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            result = future.result(timeout=poll_interval_s)
            pool.shutdown(wait=False)
            return result
        except FuturesTimeout:
            continue
    on_abort(pool)
    raise LLMWallTimeoutError(
        f"LLM call exceeded {timeout}s wall-clock timeout",
    )


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TokenUsage:
    """Normalized token usage for spend accounting.

    ``cached_tokens`` is the count of input tokens served from the
    provider's prompt cache. Provider-specific field names (the
    standard ``prompt_tokens_details.cached_tokens`` plus alternates
    like ``prompt_cache_hit_tokens``) are normalized to this single
    field so downstream spend tracking has one shape to consume.

    sensitivity_tier: 1
    """

    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class LLMResponse:
    """Normalized response from any LLM provider.

    ``usage`` is populated by providers that surface token counts
    (most OpenAI-compatible cloud providers) and left ``None`` by
    providers that don't (Ollama).

    sensitivity_tier: varies (depends on prompt content)
    """

    content: str
    model: str
    usage: TokenUsage | None = None


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class LLMProvider(ABC):
    """Abstract base for LLM providers.

    sensitivity_tier: 1
    """

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
    ) -> LLMResponse:
        """Send a chat completion request and return the full response.

        sensitivity_tier: varies
        """

    @abstractmethod
    def chat_stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
    ) -> Iterator[str]:
        """Stream a chat completion, yielding text tokens.

        sensitivity_tier: varies
        """

    @abstractmethod
    def chat_json(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
    ) -> dict[str, Any]:
        """Chat completion expecting a JSON response.

        sensitivity_tier: varies
        """

    @abstractmethod
    def check_health(self) -> dict[str, Any]:
        """Return provider health status as a JSON-serializable dict.

        sensitivity_tier: 1
        """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider identifier: ``"ollama"`` or ``"anthropic"``.

        sensitivity_tier: 1
        """

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Currently configured model name.

        sensitivity_tier: 1
        """


# ---------------------------------------------------------------------------
# Ollama implementation
# ---------------------------------------------------------------------------


class OllamaProvider(LLMProvider):
    """Ollama backend via the ``ollama`` Python SDK.

    Includes built-in retry with exponential backoff.

    Three lock priorities — interactive, background, proactive — run
    on independent lock files so they never block each other.
    Background callers inject ``/no_think`` for qwen models to skip
    the expensive thinking step on classification tasks.

    sensitivity_tier: 1
    """

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "llama3.3:70b",
        max_retries: int = MAX_RETRIES,
        base_delay: float = BASE_DELAY_S,
        *,
        background: bool = False,
        proactive: bool = False,
    ) -> None:
        import ollama as _ollama

        # HTTP timeout must be >= wall_timeout so the wall-clock timer
        # is the controlling timeout, not the HTTP layer.
        self._client = _ollama.Client(host=host, timeout=600.0)
        self._model = model
        self._host = host
        self._max_retries = max_retries
        self._base_delay = base_delay
        # Store module reference for exception handling
        self._ollama = _ollama

        # Lock tiers and timeouts.
        # Priority hierarchy: interactive > proactive > background.
        # Interactive and proactive signal preemption; background
        # checks for preemption and yields (aborts in-flight call).
        if proactive:
            self._lock_priority = "proactive"
            self._wall_timeout = 600  # digest prompts are large
        elif background:
            self._lock_priority = "background"
            self._wall_timeout = 600  # generous per-call timeout
        else:
            self._lock_priority = "interactive"
            self._wall_timeout = OLLAMA_WALL_TIMEOUT_S

        # Background callers can be preempted mid-call by
        # interactive/proactive callers.
        self._preemptable = self._lock_priority == "background"
        # Background tasks discourage thinking via system prompt.
        self._disable_thinking = background

    @staticmethod
    def _resolve_keep_alive() -> str | None:
        """Decide keep_alive based on lock contention.

        If another process is waiting for the Ollama lock, keep the
        model loaded so the next caller doesn't pay reload cost.
        If nobody's waiting, return None (Ollama default — unloads
        after its configured idle timeout).

        sensitivity_tier: 1
        """
        from src.models.ollama_lock import has_lock_contention

        if has_lock_contention():
            return "5m"
        return None

    def _abort_in_flight(self, pool: ThreadPoolExecutor) -> None:
        """Abort the current Ollama call and recreate the client.

        Closes the httpx transport so Ollama stops generating,
        then immediately shuts down the thread pool without waiting.

        sensitivity_tier: 1
        """
        try:
            self._client._client.close()  # noqa: SLF001
        except Exception:  # noqa: BLE001
            pass
        pool.shutdown(wait=False, cancel_futures=True)
        import ollama as _ollama

        self._client = _ollama.Client(
            host=self._host, timeout=600.0,
        )

    def _call_with_wall_timeout(
        self, fn: Any, timeout: int = OLLAMA_WALL_TIMEOUT_S,
    ) -> Any:
        """Run *fn* in a thread with a hard wall-clock timeout.

        httpx read-timeout resets on each received chunk, so a slowly
        streaming response can hang forever.  This enforces a fixed
        deadline regardless of data flow.

        Uses a polling loop with ``time.time()`` instead of
        ``future.result(timeout=)`` because macOS pauses
        ``pthread_cond_timedwait`` during laptop sleep — the blocking
        ``result()`` call never wakes up even after the deadline.
        Polling with real wall-clock checks survives sleep/wake.

        **Preemption**: if ``self._preemptable`` is True (background
        tier), the polling loop also checks for preemption signals
        from interactive/proactive callers.  On preemption the
        in-flight call is aborted and ``PreemptedError`` raised,
        releasing the Ollama lock immediately.

        sensitivity_tier: 1
        """
        from src.models.ollama_preempt import check_preempted

        # Do NOT use `with` — ThreadPoolExecutor.__exit__ calls
        # shutdown(wait=True), which blocks forever if the thread is
        # stuck in an Ollama call that didn't abort.  This would hold
        # the Ollama lock indefinitely, starving all other callers.
        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(fn)

        # Poll with real wall-clock time (survives macOS sleep).
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                result = future.result(timeout=2)
                pool.shutdown(wait=False)
                return result
            except FuturesTimeout:
                # Check if a higher-priority caller wants the lock.
                if self._preemptable and check_preempted():
                    logger.info(
                        "Preempted mid-call — aborting background "
                        "Ollama request",
                    )
                    self._abort_in_flight(pool)
                    raise PreemptedError from None
                continue

        # Timeout reached — abort the in-flight request.
        self._abort_in_flight(pool)
        raise TimeoutError(
            f"Ollama call exceeded {timeout}s wall-clock timeout",
        )

    def _prepare_messages(
        self,
        messages: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Prepend a conciseness system prompt for background callers.

        Keeps background responses short and focused on the requested
        format.  ``_strip_json_wrapper`` still strips any markdown
        fences or tags from output as a safety net.

        sensitivity_tier: 1
        """
        if not self._disable_thinking:
            return messages
        # Prepend system instruction to skip reasoning steps.
        system_msg = {
            "role": "system",
            "content": (
                "Respond directly and concisely. "
                "Do not include reasoning or thinking steps. "
                "Output only the requested format."
            ),
        }
        return [system_msg, *messages]

    def _preempt_guard(self) -> None:
        """Signal or wait depending on caller tier.

        Interactive and proactive callers signal preemption so
        background work yields (aborts in-flight call within ~2s).
        Background callers wait for a quiet window and check for
        preemption before each LLM call.

        Priority: interactive > proactive > background.

        sensitivity_tier: 1
        """
        from src.models.ollama_preempt import (
            check_preempted,
            signal_preempt,
            wait_for_quiet_window,
        )

        if self._lock_priority in ("interactive", "proactive"):
            # Signal so background callers abort their in-flight call.
            signal_preempt()
        if self._preemptable:
            # Wait for interactive/proactive to stop before starting.
            if not wait_for_quiet_window(timeout=self._wall_timeout):
                raise TimeoutError(
                    "Timed out waiting for quiet window",
                )
            # Check if preempted right before acquiring lock.
            if check_preempted():
                raise PreemptedError

    def _interactive_cleanup(self) -> None:
        """Clear preemption signal after interactive/proactive completes.

        sensitivity_tier: 1
        """
        if self._lock_priority in ("interactive", "proactive"):
            from src.models.ollama_preempt import clear_preempt

            clear_preempt()

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
    ) -> LLMResponse:
        """Send a chat request with retry.

        Interactive/proactive callers signal preemption so background
        callers abort.  Background callers wait for a quiet window
        and re-wait after being preempted.

        The lock is acquired per attempt so other callers can get the
        lock between retries (prevents lock starvation).

        sensitivity_tier: varies
        """
        from src.models.ollama_lock import ollama_lock

        target = model or self._model
        prepared = self._prepare_messages(messages)
        try:
            self._preempt_guard()
            for attempt in range(self._max_retries):
                try:
                    with ollama_lock(priority=self._lock_priority):
                        chat_kwargs: dict[str, Any] = {
                            "model": target,
                            "messages": prepared,
                        }
                        keep_alive = self._resolve_keep_alive()
                        if keep_alive is not None:
                            chat_kwargs["keep_alive"] = keep_alive
                        resp = self._call_with_wall_timeout(
                            lambda: self._client.chat(**chat_kwargs),
                            timeout=self._wall_timeout,
                        )
                        return LLMResponse(
                            content=resp.message.content,
                            model=target,
                        )
                except PreemptedError:
                    logger.info("Preempted — waiting for quiet window")
                    self._preempt_guard()  # re-wait for quiet window
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Ollama chat failed (attempt %d/%d): %s",
                        attempt + 1,
                        self._max_retries,
                        exc,
                    )
                    if attempt < self._max_retries - 1:
                        delay = self._base_delay * (2**attempt)
                        time.sleep(delay)
        finally:
            self._interactive_cleanup()

        return LLMResponse(content="", model=target)

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
    ) -> Iterator[str]:
        """Stream tokens from Ollama.

        Interactive callers signal preemption before streaming.
        Holds the Ollama lock for the entire stream duration.

        sensitivity_tier: varies
        """
        from src.models.ollama_lock import ollama_lock

        target = model or self._model
        prepared = self._prepare_messages(messages)
        try:
            self._preempt_guard()
            with ollama_lock(priority=self._lock_priority):
                chat_kwargs: dict[str, Any] = {
                    "model": target,
                    "messages": prepared,
                    "stream": True,
                }
                keep_alive = self._resolve_keep_alive()
                if keep_alive is not None:
                    chat_kwargs["keep_alive"] = keep_alive
                stream = self._client.chat(**chat_kwargs)
                for chunk in stream:
                    token = chunk.message.content
                    if token:
                        yield token
        finally:
            self._interactive_cleanup()

    def chat_json(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
    ) -> dict[str, Any]:
        """Chat requesting JSON output via Ollama ``format="json"``.

        Uses Ollama's native JSON constraint for reliable structured
        output.  ``_strip_json_wrapper`` still strips markdown fences
        as a safety net.

        Interactive callers signal preemption; background callers
        wait for a quiet window and yield on preemption.

        sensitivity_tier: varies
        """
        from src.models.ollama_lock import ollama_lock

        target = model or self._model
        try:
            self._preempt_guard()
            for attempt in range(self._max_retries):
                try:
                    logger.info(
                        "Ollama lock acquiring (%s, attempt %d)…",
                        self._lock_priority, attempt + 1,
                    )
                    with ollama_lock(priority=self._lock_priority):
                        prepared = self._prepare_messages(messages)
                        chat_kwargs: dict[str, Any] = {
                            "model": target,
                            "messages": prepared,
                            "format": "json",
                        }
                        keep_alive = self._resolve_keep_alive()
                        if keep_alive is not None:
                            chat_kwargs["keep_alive"] = keep_alive
                        logger.info(
                            "Ollama calling %s (%d msgs)…",
                            target, len(prepared),
                        )
                        resp = self._call_with_wall_timeout(
                            lambda: self._client.chat(**chat_kwargs),
                            timeout=self._wall_timeout,
                        )
                        text = resp.message.content
                        logger.info(
                            "Ollama response: %d chars → %s",
                            len(text),
                            repr(text[:200]),
                        )
                        # Strip markdown fences and thinking tags
                        text = _strip_json_wrapper(text)
                        return json.loads(text)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "Ollama JSON parse failed: %s", exc,
                    )
                except PreemptedError:
                    logger.info("Preempted — waiting for quiet window")
                    self._preempt_guard()  # re-wait for quiet window
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Ollama chat_json failed (attempt %d/%d): %s",
                        attempt + 1,
                        self._max_retries,
                        exc,
                    )
                    if attempt < self._max_retries - 1:
                        delay = self._base_delay * (2**attempt)
                        time.sleep(delay)
        finally:
            self._interactive_cleanup()

        return {}

    def check_health(self) -> dict[str, Any]:
        """Check Ollama server and model availability.

        sensitivity_tier: 1
        """
        try:
            model_list = self._client.list()
            available: set[str] = set()
            for m in model_list.models:
                available.add(m.model)
            model_ok = self._model in available or any(
                a.startswith(self._model.split(":")[0] + ":")
                for a in available
            )
            return {
                "provider": "ollama",
                "server_reachable": True,
                "chat_model": self._model,
                "chat_model_status": "available" if model_ok else "not_found",
            }
        except Exception:  # noqa: BLE001
            return {
                "provider": "ollama",
                "server_reachable": False,
                "chat_model": self._model,
                "chat_model_status": "offline",
            }

    @property
    def provider_name(self) -> str:
        """sensitivity_tier: 1"""
        return "ollama"

    @property
    def default_model(self) -> str:
        """sensitivity_tier: 1"""
        return self._model


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def load_llm_settings() -> dict[str, Any]:
    """Load LLM-related settings from ``~/.arandu/settings.json``.

    Returns an empty dict if the file doesn't exist.

    sensitivity_tier: 1
    """
    if SETTINGS_PATH.exists():
        try:
            with open(SETTINGS_PATH) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read settings: %s", exc)
    return {}


def create_provider_from_settings(
    *,
    background: bool = False,
    proactive: bool = False,
) -> LLMProvider:
    """Create the local Ollama provider from user settings.

    Arandu runs all inference locally on Ollama. Three caller
    tiers:

    - **interactive** (default) — chat / streaming calls.
    - **background** — pipeline, sync, classification tasks.
      Disables thinking mode (``/no_think``) for speed.
    - **proactive** — 2-hour digest. Thinking enabled, generous
      timeouts.

    sensitivity_tier: 1
    """
    settings = load_llm_settings()
    return _create_ollama_provider(
        settings, background=background, proactive=proactive,
    )


def _create_ollama_provider(
    settings: dict[str, Any],
    *,
    background: bool = False,
    proactive: bool = False,
) -> OllamaProvider:
    """Create an OllamaProvider from settings dict.

    sensitivity_tier: 1
    """
    return OllamaProvider(
        host=settings.get("llm_host", "http://localhost:11434"),
        model=settings.get("llm_model", "llama3.1:70b"),
        background=background,
        proactive=proactive,
    )


