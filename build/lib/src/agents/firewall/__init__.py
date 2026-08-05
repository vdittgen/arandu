"""Prompt firewalls — non-editable agents that gate every LLM call.

- :class:`InjectionFirewall` — rejects prompt-injection / jailbreak / role
  override attempts before the prompt reaches the model.
- :class:`EgressFirewall` — decides whether the prompt may egress to the
  remote provider, or must stay on the local Ollama fallback, based on
  the maximum sensitivity tier of the data it carries.

Both are designed to be configuration-free from the user's perspective:
they live in code, not in ``agent_configs``, and the Agents page renders
them as locked cards.

sensitivity_tier: 1
"""

from src.agents.firewall.egress_firewall import (
    EgressFirewall,
    EgressFirewallError,
    RoutingPolicy,
)
from src.agents.firewall.injection_firewall import (
    InjectionFirewall,
    InjectionRejected,
)

__all__ = [
    "EgressFirewall",
    "EgressFirewallError",
    "InjectionFirewall",
    "InjectionRejected",
    "RoutingPolicy",
]
