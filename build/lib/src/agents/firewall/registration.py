"""Registry entries for the locked firewall agents.

InjectionFirewall and EgressFirewall aren't ``SBAgent`` subclasses
— they're rule-driven and live outside the LLM hot path. But the
Agents page needs registry rows for them so the UI can render the
locked cards + the manual "Run eval" button. This module supplies
those entries.

Both are marked ``editable=False``; the config store refuses writes
to either. They appear in :data:`AGENT_SUITE_MAP` so manual evals
work; they are excluded from auto-trigger by :data:`MANUAL_ONLY_AGENTS`.

sensitivity_tier: 1
"""

from __future__ import annotations

from src.agents.core.config_store import AgentConfig
from src.agents.core.registry import (
    AgentDefinition,
    get_agent,
    register_agent,
)
from src.agents.core.scheduler import Tier


def register_injection_firewall_agent() -> None:
    """Register the injection-firewall as a locked registry agent.

    Also registers :class:`InjectionScanAgent` (the LLM-judge sub-agent)
    as a child so the Agents page surfaces both under the locked
    Injection Firewall card. Each entry is registered independently so
    upgrades that add the sub-agent to an install that already has the
    outer card still wire the new child up.

    sensitivity_tier: 1
    """
    from src.agents.firewall.injection_firewall import (
        InjectionFirewall,
    )
    from src.agents.firewall.injection_scan_agent import (
        InjectionScanAgent,
    )

    if get_agent("firewall.injection") is None:
        default = AgentConfig(
            agent_id="firewall.injection",
            system_prompt=(
                "Heuristic prompt-injection scanner. Rejects role "
                "overrides, data-bleed attempts, chat-template token "
                "injections, and oversized base64 blobs before the "
                "prompt reaches the model."
            ),
            model_route="inherit",
            model_override=None,
            enabled_tools=(),
            enabled_skills=(),
            editable=False,
        )
        register_agent(AgentDefinition(
            agent_id="firewall.injection",
            name="Injection Firewall",
            description=(
                "Non-editable prompt-injection guard. Heuristic-first, "
                "with an LLM-judge semantic pass for prompts the regex "
                "layer is least sure about. Runs on every agent prompt "
                "before egress."
            ),
            category="firewall",
            parent_agent=None,
            tier=Tier.SYSTEM,
            max_sensitivity_tier=3,
            editable=False,
            default_config=default,
            available_tools=(),
            available_skills=(),
            output_schema="InjectionVerdict",
            pattern="single",
            factory=InjectionFirewall,
            tags=("locked", "firewall"),
        ))

    if get_agent("firewall.injection.scan") is None:
        scan_default = AgentConfig(
            agent_id="firewall.injection.scan",
            system_prompt=InjectionScanAgent.system_prompt,
            # Runs on the local Ollama model, same as every other agent.
            model_route="local",
            model_override=None,
            enabled_tools=(),
            enabled_skills=(),
            editable=False,
        )
        register_agent(AgentDefinition(
            agent_id="firewall.injection.scan",
            name="Injection Semantic Scanner",
            description=(
                "LLM-judge sub-agent owned by the injection firewall. "
                "Runs only when the heuristic pass is clean. Judges "
                "the prompt against the configured LLM (local Ollama "
                "in Arandu) to catch what the heuristic layer "
                "missed."
            ),
            category="firewall",
            parent_agent="firewall.injection",
            tier=Tier.SYSTEM,
            max_sensitivity_tier=3,
            editable=False,
            default_config=scan_default,
            available_tools=(),
            available_skills=(),
            output_schema="InjectionVerdict",
            pattern="single",
            factory=InjectionScanAgent,
            tags=("locked", "firewall"),
        ))


def register_egress_firewall_agent() -> None:
    """Register the egress-firewall as a locked registry agent.

    sensitivity_tier: 1
    """
    if get_agent("firewall.egress") is not None:
        return
    from src.agents.firewall.egress_firewall import EgressFirewall

    default = AgentConfig(
        agent_id="firewall.egress",
        system_prompt=(
            "Routes every outbound LLM call to the configured provider "
            "based on the routing policy and the prompt's maximum "
            "sensitivity tier. In Arandu all traffic is local. "
            "See docs/PRIVACY.md."
        ),
        model_route="inherit",
        model_override=None,
        enabled_tools=(),
        enabled_skills=(),
        editable=False,
    )
    register_agent(AgentDefinition(
        agent_id="firewall.egress",
        name="Egress Firewall",
        description=(
            "Non-editable egress router. Decides whether each prompt "
            "stays on the local LLM, per the active routing "
            "policy."
        ),
        category="firewall",
        parent_agent=None,
        tier=Tier.SYSTEM,
        max_sensitivity_tier=3,
        editable=False,
        default_config=default,
        available_tools=(),
        available_skills=(),
        output_schema="EgressDecision",
        pattern="single",
        factory=EgressFirewall,
        tags=("locked", "firewall"),
    ))


__all__ = [
    "register_egress_firewall_agent",
    "register_injection_firewall_agent",
]
