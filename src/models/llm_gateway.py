"""Single chokepoint for outbound LLM calls.

Every component that wants to call the LLM should go through
:func:`chat_via_firewalls`. The gateway:

1. Checks the per-agent block table (set when local-only mode is on
   and the agent's eval suite failed).
2. Runs the prompt-injection firewall.
3. Resolves the egress decision (route + tier) and persists the
   prompt detail for audit drill-down.
4. Acquires a scheduler permit at the right tier.
5. Builds the right :class:`LLMProvider` for the resolved route
   — callers no longer pick the provider themselves. Arandu
   always resolves to local Ollama.
6. Calls the provider and returns a :class:`LLMResponse`.

CLAUDE.md pitfall #11 explicitly forbids skipping the firewall "for
convenience": this module exists so call sites don't need to choose
between convenience and correctness.

sensitivity_tier: 1
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from src.agents.core.audit import hash_payload
from src.agents.core.scheduler import Tier, default_scheduler
from src.agents.firewall.egress_firewall import (
    ComplexityTier,
    Lane,
    default_egress_firewall,
)
from src.agents.firewall.injection_firewall import (
    InjectionRejected,
    default_injection_firewall,
)
from src.agents.firewall.lane_context import lane_scope
from src.models.llm_provider import LLMProvider, LLMResponse
from src.models.redaction_store import default_redaction_store

logger = logging.getLogger(__name__)


class GatewayBlocked(Exception):  # noqa: N818 — paired with AgentAccessDeniedError name shape
    """Raised when a firewall blocks an LLM call at the gateway.

    Callers should treat this as the gateway's equivalent of
    :class:`AgentAccessDeniedError` and surface it appropriately
    to the user (refuse the call, show the consent dialog, etc.).
    The :class:`AgentContext.ask_llm` shim catches this and re-raises
    as :class:`AgentAccessDeniedError` so existing call sites don't
    need to change exception handling.

    sensitivity_tier: 1
    """


def _tier_to_scheduler(lane: Lane) -> Tier:
    """Map a routing :class:`Lane` to a scheduler :class:`Tier`.

    sensitivity_tier: 1
    """
    if lane in (Lane.INTERACTIVE,):
        return Tier.INTERACTIVE
    if lane in (Lane.CLASSIFIER, Lane.ESCALATION):
        return Tier.PROACTIVE
    return Tier.BACKGROUND


_provider_factory_override: Callable[[str], LLMProvider] | None = None


def set_provider_factory_for_tests(
    factory: Callable[[str], LLMProvider] | None,
) -> None:
    """Install or clear a provider-building override.

    Tests call this with a factory that returns a stub
    :class:`LLMProvider` (typically a :class:`MagicMock`) so the
    gateway exercises the firewall flow without hitting a real
    LLM endpoint.

    sensitivity_tier: 1
    """
    global _provider_factory_override
    _provider_factory_override = factory


def _build_provider(*, route: str) -> LLMProvider:
    """Construct the :class:`LLMProvider` for a resolved egress route.

    Always builds a *fresh* provider — the configured default in
    settings may point at a remote endpoint, and we must not reuse
    that when the egress firewall has chosen ``"local"``.

    sensitivity_tier: 1
    """
    if _provider_factory_override is not None:
        return _provider_factory_override(route)

    from src.agents.core.model_factory import (
        local_endpoint,
        remote_endpoint,
    )
    from src.models.llm_provider import (
        OllamaProvider,
        OpenAICompatibleProvider,
    )

    endpoint = remote_endpoint() if route == "remote" else local_endpoint()

    if route == "local":
        return OllamaProvider(
            host=endpoint.base_url.removesuffix("/v1"),
            model=endpoint.model_name,
            background=True,
        )
    return OpenAICompatibleProvider(
        host=endpoint.base_url,
        model=endpoint.model_name,
        api_key=endpoint.api_key,
        background=True,
    )


def _last_user_text(messages: list[dict[str, str]]) -> str:
    """Extract the most recent user-content text for firewall scanning.

    The injection / egress firewalls only need the user-supplied
    portion of the call. System-role messages come from trusted
    code paths and are excluded.

    sensitivity_tier: varies
    """
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    return "\n".join(
        str(m.get("content", "")) for m in messages
        if m.get("role") != "system"
    )


def chat_via_firewalls(
    messages: list[dict[str, str]],
    *,
    agent_id: str,
    lane: Lane,
    agent_max_tier: int = 1,
    complexity: ComplexityTier = "balanced",
    explicit_tier: int | None = None,
    model_override: str | None = None,
) -> LLMResponse:
    """Send ``messages`` to the LLM via both firewalls.

    Arguments:
        messages: OpenAI-style chat messages. The most recent user
            message is what the injection / egress firewalls inspect.
        agent_id: Stable id of the caller. Recorded in the audit chain
            and consulted by :class:`EgressFirewall` for the
            per-agent Tier 3 allow list.
        lane: Routing lane the call belongs to. Drives both lane
            accounting and the scheduler tier.
        agent_max_tier: Maximum sensitivity tier the caller is
            authorised to handle (from the agent manifest).
        complexity: ``"fast" | "balanced" | "deep"``. Accepted for
            forward compatibility with callers; currently inert, since
            every route resolves to the same local model regardless.
        explicit_tier: Optional pre-computed tier (e.g. from an
            upstream classifier). When ``None`` the egress firewall
            runs its own classifier.
        model_override: Optional model id to substitute for the
            provider's default.

    Raises:
        GatewayBlocked: When the injection firewall blocks the prompt
            or the egress firewall returns ``route="blocked"``.
            :class:`AgentContext.ask_llm` translates this to the
            existing :class:`AgentAccessDeniedError` for sandboxed
            agents.

    sensitivity_tier: varies
    """
    user_text = _last_user_text(messages)

    # Eval-failure gate (only ever populated under local-only mode).
    # Imported lazily so the gateway works in tests / minimal installs
    # that don't import the agents store at module load time.
    try:
        from src.agents.core.agent_block_store import (
            default_agent_block_store,
        )
        block = default_agent_block_store().get_block(agent_id)
    except Exception:  # noqa: BLE001
        block = None
    if block is not None:
        raise GatewayBlocked(
            f"Agent '{agent_id}' is blocked: {block}",
        )

    try:
        default_injection_firewall().assert_allowed(
            user_text, calling_agent_id=agent_id,
        )
    except InjectionRejected as exc:
        raise GatewayBlocked(
            f"Prompt rejected by injection firewall: {exc.verdict.reason}",
        ) from exc

    egress = default_egress_firewall().classify(
        user_text,
        calling_agent_id=agent_id,
        agent_max_tier=agent_max_tier,
        explicit_tier=explicit_tier,
    )
    if egress.route == "blocked":
        raise GatewayBlocked(
            f"Egress firewall blocked the call: {egress.reason}",
        )

    # Same hash the egress_decision row carries, so the UI can match a
    # clicked audit row to its stored detail blob.
    user_text_hash = hash_payload(user_text)

    # Persist the prompt detail for every call so every egress_decision
    # row in the audit log is clickable. 24h retention applies.
    try:
        default_redaction_store().store(
            payload_hash=user_text_hash,
            agent_id=agent_id,
            lane=lane.value,
            original_messages=[dict(m) for m in messages],
            redacted_messages=[dict(m) for m in messages],
            placeholder_map={},
        )
    except Exception as exc:  # noqa: BLE001
        # Detail persistence is non-critical — the LLM call must
        # still proceed even if disk writes fail.
        logger.warning("Failed to persist prompt detail: %s", exc)

    resolved_route = egress.route
    provider = _build_provider(route=resolved_route)
    scheduler_tier = _tier_to_scheduler(lane)
    with scheduler.acquire_context(scheduler_tier, resolved_route, agent_id):
        with lane_scope(lane):
            response = provider.chat(messages, model=model_override)

    return response


class _Scheduler:
    """Tiny shim that exposes :func:`default_scheduler` as a context.

    The gateway calls into the process-wide scheduler the same way
    every other LLM caller does; this helper just wraps the
    boilerplate so the main flow above reads top-to-bottom.

    sensitivity_tier: 1
    """

    @staticmethod
    def acquire_context(
        tier: Tier, route: str, agent_id: str,
    ) -> Any:
        return default_scheduler().acquire(
            tier, route=route, agent_id=agent_id,  # type: ignore[arg-type]
        )


scheduler = _Scheduler()


__all__ = [
    "GatewayBlocked",
    "chat_via_firewalls",
    "set_provider_factory_for_tests",
]
