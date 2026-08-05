"""Egress firewall — decides which model an LLM call may reach.

In Arandu every call resolves to the local Ollama backend. The
firewall still classifies the prompt's sensitivity tier (Tier 1 / 2 /
3) and emits an audit-chain entry per request — those signals matter
even when the destination is local — but it never routes off-device.

The legacy ``RoutingPolicy`` field on disk (``remote-default`` /
``local-only``) is preserved as a forward-compatible extension
point. Here both values behave identically.

Tier classification is *upper-bound*: the firewall takes the maximum
of (agent's ``max_sensitivity_tier``, the explicit tier passed in by
the caller, and a quick keyword pre-classification of the prompt
text).

sensitivity_tier: 1
"""

from __future__ import annotations

import enum
import json
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.agents.core.audit import default_chain, hash_payload
from src.agents.core.output_types import EgressDecision

logger = logging.getLogger(__name__)

# Routing policies stored in settings.json. Arandu routes every
# call locally regardless of which of these is selected; the
# ``remote-default`` value is a reserved extension point.
RoutingPolicy = Literal["remote-default", "local-only"]
ComplexityTier = Literal["fast", "balanced", "deep"]

SETTINGS_PATH = Path.home() / ".arandu" / "settings.json"


class Lane(enum.Enum):
    """Traffic lane a request is attributed to.

    The lane categorises each request by its origin (background work,
    an interactive turn, the sensitivity classifier, an escalation, or
    coding) so the gateway can pick a scheduler tier and the audit
    chain can attribute the call. See ``_tier_to_scheduler`` in
    ``llm_gateway``.

    sensitivity_tier: 1
    """

    BACKGROUND = "background"
    INTERACTIVE = "interactive"
    CLASSIFIER = "classifier"
    ESCALATION = "escalation"
    CODING = "coding"


class EgressFirewallError(Exception):
    """Raised when an egress decision can't be made.

    sensitivity_tier: 1
    """


# ---------------------------------------------------------------------------
# Keyword pre-classification — coarse but useful as a floor
# ---------------------------------------------------------------------------

_TIER3_KEYWORDS = re.compile(
    r"\b(?:"
    r"depression|anxiety|trauma|abuse|suicide|self[\s\-]harm|"
    r"diagnos|medication|prescription|symptom|"
    r"bank\s*account|routing\s*number|credit\s*card|ssn|"
    r"social\s*security|tax\s*id|password|2fa|seed\s*phrase"
    r")\b",
    re.IGNORECASE,
)
_TIER2_KEYWORDS = re.compile(
    r"\b(?:"
    r"phone\s*number|address|email\s*address|"
    r"sister|brother|mother|father|partner|spouse|colleague|"
    r"meeting|appointment|calendar"
    r")\b",
    re.IGNORECASE,
)


def keyword_tier_floor(text: str) -> int:
    """Return a minimum tier based on a quick text scan.

    sensitivity_tier: 1
    """
    if _TIER3_KEYWORDS.search(text):
        return 3
    if _TIER2_KEYWORDS.search(text):
        return 2
    return 1


# ---------------------------------------------------------------------------
# Settings + policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EgressPolicy:
    """Resolved policy from settings.

    In Arandu the ``routing`` field does not affect where calls
    go — every route resolves to local Ollama. The field is retained
    as a reserved extension point (and recorded in the audit chain).
    Derived from the single user-facing setting
    ``local_inference_for_sensitive``.

    sensitivity_tier: 1
    """

    routing: RoutingPolicy = "remote-default"
    local_inference_for_sensitive: bool = False


def _settings() -> dict[str, object]:
    """Read settings.json, returning ``{}`` on any error.

    sensitivity_tier: 1
    """
    if not SETTINGS_PATH.exists():
        return {}
    try:
        with SETTINGS_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read settings: %s", exc)
        return {}


def _load_policy() -> EgressPolicy:
    """Synthesize the :class:`EgressPolicy` from settings.json.

    The new setting ``local_inference_for_sensitive`` is the single
    source of truth. For one release we also honour the legacy
    ``llm_routing_policy="privacy-strict"`` value as an alias for
    ``local_inference_for_sensitive=true`` so users mid-migration
    don't lose their privacy posture on first launch.

    sensitivity_tier: 1
    """
    settings = _settings()
    local = bool(settings.get("local_inference_for_sensitive", False))
    if not local:
        legacy = settings.get("llm_routing_policy")
        if legacy == "privacy-strict":
            local = True
    routing: RoutingPolicy = "local-only" if local else "remote-default"
    return EgressPolicy(
        routing=routing,
        local_inference_for_sensitive=local,
    )


# Internal agent ids that must not recursively invoke the LLM-driven
# sensitivity classifier. The classifier itself runs an LLM call; if the
# egress firewall reclassified that call's prompt it would loop forever.
_CLASSIFIER_SAFE_LIST: frozenset[str] = frozenset({
    "firewall.injection",
    "firewall.egress",
    "firewall.injection.scan",
    "llm_classifier",
    "sensitivity_classifier",
})


def _local_only_classifier() -> object | None:
    """Build a :class:`SensitivityClassifier` that always runs locally.

    Used only under ``local-only`` mode — under ``remote-default`` we
    don't want to put Ollama on every prompt's hot path just to
    compute a tier that never changes the (always-local) route.

    sensitivity_tier: 1
    """
    import os
    if os.environ.get("ARANDU_FIREWALL_DISABLE_LLM_TIER") == "1":
        return None
    try:
        from src.models.llm_provider import (
            OllamaProvider,
            load_llm_settings,
        )
        from src.models.sensitivity_classifier import SensitivityClassifier

        settings = load_llm_settings()
        local_provider = OllamaProvider(
            host=settings.get(
                "llm_local_host",
                settings.get("ollama_host", "http://localhost:11434"),
            ),
            model=settings.get(
                "llm_local_model",
                settings.get("llm_classifier_model", "gemma4:e2b"),
            ),
            background=True,
        )
        return SensitivityClassifier(llm_provider=local_provider)
    except Exception:  # noqa: BLE001
        logger.debug("local sensitivity classifier unavailable", exc_info=True)
        return None


def _llm_classify_tier(text: str) -> int | None:
    """Best-effort LLM-driven tier classification.

    sensitivity_tier: 1
    """
    classifier = _local_only_classifier()
    if classifier is None:
        return None
    try:
        tier = classifier.classify(text)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        logger.debug("LLM tier classify failed", exc_info=True)
        return None
    return tier if tier in (1, 2, 3) else None


# ---------------------------------------------------------------------------
# Firewall
# ---------------------------------------------------------------------------


class EgressFirewall:
    """Non-editable router for outbound LLM calls.

    Stateless except for a cached policy snapshot loaded from settings
    on construction. Long-running processes should call
    :meth:`reload_policy` when the user updates their routing choice in
    the UI.

    sensitivity_tier: 1
    """

    AGENT_ID = "firewall.egress"

    def __init__(
        self,
        *,
        policy: EgressPolicy | None = None,
    ) -> None:
        self._policy = policy or _load_policy()
        self._lock = threading.Lock()
        # Hash-keyed cache of (max_tier, route, reason) so repeated
        # prompts (the labeler frequently retries the same text) skip
        # the LLM-driven tier classifier on the hot path.
        self._tier_cache: dict[str, int] = {}

    def reload_policy(self) -> None:
        """Re-read settings.json and refresh the policy snapshot.

        sensitivity_tier: 1
        """
        with self._lock:
            self._policy = _load_policy()

    @property
    def policy(self) -> EgressPolicy:
        with self._lock:
            return self._policy

    def classify(
        self,
        prompt: str,
        *,
        calling_agent_id: str = "unknown",
        agent_max_tier: int = 1,
        explicit_tier: int | None = None,
        context_data: str = "",
    ) -> EgressDecision:
        """Compute the egress decision for a single prompt.

        ``agent_max_tier`` is the agent manifest's
        ``max_sensitivity_tier``. ``explicit_tier`` is an upstream
        classifier's tier (for example, the sensitivity sub-agent's
        verdict on the prompt content).

        The chosen tier is the maximum of those two and the keyword
        floor. The local-LLM tier classifier only runs under
        ``local-only`` mode — it exists for audit-chain visibility
        into how sensitive local traffic is, not to gate routing,
        since every route already resolves locally.

        sensitivity_tier: 1
        """
        text = f"{prompt}\n{context_data}".strip()
        keyword_tier = keyword_tier_floor(text)
        policy = self.policy

        llm_tier: int | None = None
        if (
            policy.routing == "local-only"
            and explicit_tier is None
            and calling_agent_id not in _CLASSIFIER_SAFE_LIST
        ):
            cache_key = hash_payload(text)
            cached = self._tier_cache.get(cache_key)
            if cached is not None:
                llm_tier = cached
            else:
                llm_tier = _llm_classify_tier(text)
                if llm_tier is not None:
                    self._tier_cache[cache_key] = llm_tier

        max_tier = max(
            keyword_tier,
            agent_max_tier,
            explicit_tier or 0,
            llm_tier or 0,
        )
        max_tier = max(1, min(3, max_tier))
        # OSS: every tier stays local regardless of policy.
        route = "local"
        reason = f"tier {max_tier} stays local (Arandu)"
        decision = EgressDecision(
            route=route,  # type: ignore[arg-type]
            max_tier=max_tier,
            reason=reason,
        )
        default_chain().append(
            event_type="egress_decision",
            agent_id=calling_agent_id,
            decision=route,
            payload_hash=hash_payload(text),
            tier=max_tier,
            extra={
                "policy": policy.routing,
                "keyword_tier": keyword_tier,
                "agent_max_tier": agent_max_tier,
                "explicit_tier": explicit_tier,
                "llm_tier": llm_tier,
            },
        )
        return decision


_default_egress_firewall: EgressFirewall | None = None
_default_egress_lock = threading.Lock()


def default_egress_firewall() -> EgressFirewall:
    """Return the process-wide egress firewall instance.

    sensitivity_tier: 1
    """
    global _default_egress_firewall
    if _default_egress_firewall is None:
        with _default_egress_lock:
            if _default_egress_firewall is None:
                _default_egress_firewall = EgressFirewall()
    return _default_egress_firewall


def reset_egress_firewall_for_tests(
    *, policy: EgressPolicy | None = None,
) -> EgressFirewall:
    """Drop the cached firewall — for test isolation.

    sensitivity_tier: 1
    """
    global _default_egress_firewall
    with _default_egress_lock:
        _default_egress_firewall = EgressFirewall(policy=policy)
    return _default_egress_firewall


__all__ = [
    "ComplexityTier",
    "EgressFirewall",
    "EgressFirewallError",
    "EgressPolicy",
    "Lane",
    "RoutingPolicy",
    "default_egress_firewall",
    "keyword_tier_floor",
    "reset_egress_firewall_for_tests",
]
