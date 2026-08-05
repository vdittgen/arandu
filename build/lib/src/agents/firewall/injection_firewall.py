"""Prompt-injection firewall.

Runs on every agent prompt before it leaves the process. Two-pass
strategy:

1. **Heuristic pass** (no LLM) — fast regex / keyword check for known
   injection patterns. Catches the obvious cases at zero cost.
2. **Semantic pass** (LLM) — only when the heuristic pass is
   inconclusive. The firewall delegates to its own
   ``SBAgent[InjectionVerdict]`` instance. The check itself runs at
   :data:`Tier.SYSTEM` priority so a busy background queue can't
   delay a user chat turn.

The firewall is **not editable** by users. Its system prompt and
heuristics live in this file; ``register_agent`` flags it
``editable=False``.

On rejection the firewall raises :class:`InjectionRejected`, which is
caught by ``AgentContext.ask_llm`` and translated to an
``AgentAccessDeniedError`` for the calling agent.

sensitivity_tier: 1
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass

from src.agents.core.audit import default_chain, hash_payload
from src.agents.core.output_types import InjectionVerdict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Heuristic patterns
# ---------------------------------------------------------------------------

# Each entry: (compiled regex, category, confidence, reason).
#
# Order is significant: more *specific* signals come first because the
# first match wins. Chat-template token injection is the most concrete
# attack signature, so it precedes the broader persona/role checks
# (an im_start blob will usually contain a "you are unrestricted"
# payload, but the template injection is the more informative
# diagnosis).
_HEURISTICS: tuple[tuple[re.Pattern[str], str, float, str], ...] = (
    (
        re.compile(
            r"<\s*\|?\s*(?:system|im_start|im_end|assistant)\s*\|?\s*>",
            re.IGNORECASE,
        ),
        "injection",
        0.85,
        "chat-template token injection",
    ),
    (
        re.compile(
            r"\b(ignore|disregard|forget)\s+(all\s+)?(previous|prior|"
            r"earlier|above)\s+(instructions?|messages?|context)\b",
            re.IGNORECASE,
        ),
        "role_override",
        0.95,
        "instructs the model to ignore prior context",
    ),
    (
        # ``you are X`` where X is a privileged role, with optional
        # filler words ("now", "in", "an", etc.) in between. The
        # ``(?:\w+\s+){0,3}`` window catches "you are now in god mode".
        re.compile(
            r"\byou\s+are\s+(?:\w+\s+){0,3}(?:admin|root|developer|"
            r"system|sudo|god[\s\-]?mode|jailbroken|unrestricted)\b",
            re.IGNORECASE,
        ),
        "jailbreak",
        0.95,
        "attempts to elevate model persona",
    ),
    (
        re.compile(
            r"\bsystem\s*:\s*(?:you\s+are|act\s+as|pretend)",
            re.IGNORECASE,
        ),
        "role_override",
        0.9,
        "fake system message",
    ),
    (
        re.compile(
            r"\b(reveal|print|leak|exfiltrate|expose)\s+(?:the\s+)?"
            r"(system\s+prompt|instructions|api\s*key|secret)",
            re.IGNORECASE,
        ),
        "data_bleed",
        0.95,
        "asks the model to leak privileged content",
    ),
    (
        # The literal jailbreak handle ``DAN`` is conventionally written
        # in uppercase; allowing case-insensitive matching catches the
        # common personal name ``Dan`` and fires on every WhatsApp
        # message sender called Dan. Use ``(?-i:...)`` to pin DAN to
        # uppercase while keeping the long-form phrase tolerant.
        re.compile(
            r"(?-i:\bDAN\b)|\b(?:do\s+anything\s+now)\b",
            re.IGNORECASE,
        ),
        "jailbreak",
        0.9,
        "named jailbreak attempt",
    ),
)

# Suspiciously long base64 blobs can hide payloads. ~512 chars of base64
# is ~384 bytes — well above any legitimate inline content.
_BASE64_BLOB = re.compile(r"[A-Za-z0-9+/]{512,}={0,2}")


# Internal sub-agents whose prompt is always a curated JSON-formatted
# batch produced by the orchestrator (messages from connectors,
# discovered schemas, etc.). The semantic LLM scan reliably false-
# positives on such inputs because the structured wrapper *looks*
# like an indirect-injection payload to the judge — "JSON arrays with
# placeholder tokens designed for downstream parsing". The heuristic
# regex pass still runs for these agents, catching obvious attack
# tokens embedded inside the data (``DAN``, ``ignore previous``,
# template injections, oversized base64); only the LLM judge is
# skipped. Add an agent here only when its prompt format is bounded
# by the orchestrator and never echoes free-form attacker text as
# the top-level instruction surface.
_BATCH_AGENT_IDS: frozenset[str] = frozenset({
    "triage",
    "message_evaluator",
    "pending_reply",
    "contact_context",
    "actionable_events",
    "topic_extractor",
    "schema_discovery",
    "model_generator",
    # Internal pipeline agents whose prompts are system-generated
    # (query routing, action param extraction, WHERE clause gen).
    # The user's original question is already scanned at the
    # orchestrator entry point (brain.ask / chat.ask); re-scanning
    # every internal LLM call wastes 2-5s per call.
    "brain",
    "brain.actions.where",
    "brain.actions.params",
    "brain.actions.judge",
    "query_router",
    "sensitivity",
    "labeler",
    "insight",
    "fact_extractor",
    "task_curator",
})


@dataclass(frozen=True)
class _CachedVerdict:
    """One cached injection verdict.

    sensitivity_tier: 1
    """

    verdict: InjectionVerdict
    expires_at: float


class InjectionRejected(Exception):  # noqa: N818
    """Raised when the firewall blocks a prompt.

    sensitivity_tier: 1
    """

    def __init__(self, verdict: InjectionVerdict) -> None:
        super().__init__(f"injection rejected: {verdict.reason}")
        self.verdict = verdict


def _reconcile_verdict(
    verdict: InjectionVerdict, calling_agent_id: str,
) -> InjectionVerdict:
    """Repair internally inconsistent verdicts before they take effect.

    The semantic LLM judge sometimes emits a self-contradictory
    verdict — ``category="safe"`` with high confidence yet
    ``allowed=False`` — which blocks an entirely benign prompt. This is
    pronounced with cloud models under native structured output, where
    ``allowed`` and ``category`` can be populated inconsistently. The
    ``category`` carries the actual attack classification, so a "safe"
    category must never block: trust it over the boolean.

    Logged at WARNING so the over-trigger stays visible even though the
    prompt is no longer wrongly rejected.

    sensitivity_tier: 1
    """
    if verdict.category == "safe" and not verdict.allowed:
        logger.warning(
            "Injection verdict reconciled for %s: category=safe but "
            "allowed=False (confidence=%.2f) — treating as allowed",
            calling_agent_id,
            verdict.confidence,
        )
        return verdict.model_copy(update={"allowed": True})
    return verdict


class InjectionFirewall:
    """Non-editable prompt-injection guard.

    Thread-safe. Phase 1 ships with the heuristic pass + caching. The
    semantic pass is wired but defaults to "allow with low confidence"
    when the underlying LLM agent isn't constructible (tests, offline).
    Phase 2 lands the real ``SBAgent[InjectionVerdict]`` implementation.

    sensitivity_tier: 1
    """

    AGENT_ID = "firewall.injection"
    cache_ttl_s: float = 600.0

    def __init__(self) -> None:
        self._cache: dict[str, _CachedVerdict] = {}
        self._lock = threading.Lock()

    # ----- public API -------------------------------------------------

    def scan(
        self,
        prompt: str,
        *,
        calling_agent_id: str = "unknown",
        context_data: str = "",
    ) -> InjectionVerdict:
        """Return a verdict for ``prompt`` (+ optional retrieved context).

        Audit entry is appended on every call. Cache key combines the
        full payload + the calling agent so an injection accepted for
        one agent isn't silently reused for another with different
        privileges.

        sensitivity_tier: 1
        """
        payload = self._payload(prompt, context_data, calling_agent_id)
        key = hash_payload(payload)
        cached = self._lookup(key)
        if cached is not None:
            return cached

        verdict = self._heuristic_scan(prompt, context_data)
        if verdict is None:
            if calling_agent_id in _BATCH_AGENT_IDS:
                verdict = InjectionVerdict(
                    allowed=True,
                    category="safe",
                    confidence=0.5,
                    reason=(
                        "heuristic pass clean; semantic scan skipped "
                        f"for internal batch agent {calling_agent_id!r}"
                    ),
                )
            else:
                verdict = self._semantic_scan(prompt, context_data)

        verdict = _reconcile_verdict(verdict, calling_agent_id)
        self._store(key, verdict)
        # Persist the scanned prompt + context under the same
        # payload_hash so the audit row is clickable. Imported lazily
        # to avoid pulling the model layer into firewall startup.
        try:
            from src.models.redaction_store import default_redaction_store

            messages = [{"role": "user", "content": prompt}]
            if context_data:
                messages.append(
                    {"role": "context", "content": context_data},
                )
            default_redaction_store().store(
                payload_hash=key,
                agent_id=calling_agent_id,
                lane="injection_scan",
                original_messages=messages,
                redacted_messages=messages,
                placeholder_map={},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to persist scan detail: %s", exc)
        default_chain().append(
            event_type="prompt_scan",
            agent_id=calling_agent_id,
            decision="allow" if verdict.allowed else "block",
            payload_hash=key,
            extra={
                "firewall": "injection",
                "category": verdict.category,
                "confidence": verdict.confidence,
            },
        )
        return verdict

    def assert_allowed(
        self,
        prompt: str,
        *,
        calling_agent_id: str = "unknown",
        context_data: str = "",
    ) -> InjectionVerdict:
        """Scan and raise :class:`InjectionRejected` if blocked.

        sensitivity_tier: 1
        """
        verdict = self.scan(
            prompt,
            calling_agent_id=calling_agent_id,
            context_data=context_data,
        )
        if not verdict.allowed:
            raise InjectionRejected(verdict)
        return verdict

    # ----- internals --------------------------------------------------

    def _payload(self, prompt: str, ctx: str, agent_id: str) -> str:
        # Agent-independent: injection risk depends on content, not caller.
        # This lets nested calls (chat→brain on the same question) hit cache.
        return f"scan\0{prompt}\0{ctx}"

    def _lookup(self, key: str) -> InjectionVerdict | None:
        now = time.monotonic()
        with self._lock:
            cached = self._cache.get(key)
            if cached is None:
                return None
            if cached.expires_at < now:
                del self._cache[key]
                return None
            return cached.verdict

    def _store(self, key: str, verdict: InjectionVerdict) -> None:
        expires = time.monotonic() + self.cache_ttl_s
        with self._lock:
            self._cache[key] = _CachedVerdict(
                verdict=verdict, expires_at=expires,
            )
            if len(self._cache) > 2048:
                # Drop the oldest 256 — cheap pruning.
                oldest = sorted(
                    self._cache.items(),
                    key=lambda kv: kv[1].expires_at,
                )[:256]
                for k, _ in oldest:
                    self._cache.pop(k, None)

    def _heuristic_scan(
        self, prompt: str, ctx: str,
    ) -> InjectionVerdict | None:
        text = f"{prompt}\n{ctx}"
        for regex, category, confidence, reason in _HEURISTICS:
            if regex.search(text):
                return InjectionVerdict(
                    allowed=False,
                    category=category,  # type: ignore[arg-type]
                    confidence=confidence,
                    reason=reason,
                )
        if _BASE64_BLOB.search(text):
            return InjectionVerdict(
                allowed=False,
                category="injection",
                confidence=0.7,
                reason="oversized base64 payload",
            )
        # Heuristics didn't trip — defer to semantic check (or pass-through).
        return None

    def _semantic_scan(self, prompt: str, ctx: str) -> InjectionVerdict:
        """Delegate to :class:`InjectionScanAgent` for an LLM judgement.

        The semantic pass runs only when the heuristic pass is clean.
        We pin the call to the local Ollama backend — sending a
        suspected-injection prompt to the third-party provider is the
        very thing we're trying to avoid — and tag the call at
        ``Tier.SYSTEM`` priority so a busy background queue can't delay
        a user chat turn.

        Fail behaviour: when the local LLM stack is unavailable the
        scanner returns a ``safe`` verdict with low confidence rather
        than fail-closed. The audit chain still records the decision,
        so any after-the-fact review can flag prompts that bypassed
        the LLM check. Set ``ARANDU_FIREWALL_FAIL_CLOSED=1`` in the
        environment to flip this to "block on classifier outage".

        sensitivity_tier: 1
        """
        import os
        if os.environ.get("ARANDU_FIREWALL_DISABLE_SEMANTIC_SCAN") == "1":
            return InjectionVerdict(
                allowed=True,
                category="safe",
                confidence=0.5,
                reason="heuristic pass clean; semantic check disabled",
            )

        try:
            from src.agents.firewall.injection_scan_agent import (
                run_injection_scan,
            )
            verdict = run_injection_scan(prompt=prompt, context=ctx)
        except Exception:  # noqa: BLE001
            logger.debug("injection semantic scan failed", exc_info=True)
            verdict = None

        if verdict is not None:
            return verdict

        if os.environ.get("ARANDU_FIREWALL_FAIL_CLOSED") == "1":
            return InjectionVerdict(
                allowed=False,
                category="injection",
                confidence=0.5,
                reason=(
                    "heuristic pass clean; semantic check failed and "
                    "ARANDU_FIREWALL_FAIL_CLOSED is set"
                ),
            )
        return InjectionVerdict(
            allowed=True,
            category="safe",
            confidence=0.5,
            reason="heuristic pass clean; semantic check unavailable",
        )


_default_injection_firewall: InjectionFirewall | None = None
_default_injection_lock = threading.Lock()


def default_injection_firewall() -> InjectionFirewall:
    """Return the process-wide injection firewall instance.

    sensitivity_tier: 1
    """
    global _default_injection_firewall
    if _default_injection_firewall is None:
        with _default_injection_lock:
            if _default_injection_firewall is None:
                _default_injection_firewall = InjectionFirewall()
    return _default_injection_firewall


def reset_injection_firewall_for_tests() -> InjectionFirewall:
    """Drop the cached firewall instance — for test isolation.

    sensitivity_tier: 1
    """
    global _default_injection_firewall
    with _default_injection_lock:
        _default_injection_firewall = InjectionFirewall()
    return _default_injection_firewall


def known_patterns() -> Iterable[str]:
    """Return human-readable descriptions of the heuristics.

    Used by the Agents page to render the firewall card's "what we
    catch" list.

    sensitivity_tier: 1
    """
    return [reason for (_, _, _, reason) in _HEURISTICS]


__all__ = [
    "InjectionFirewall",
    "InjectionRejected",
    "default_injection_firewall",
    "known_patterns",
    "reset_injection_firewall_for_tests",
]
