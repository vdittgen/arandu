"""Cost-tiered model assignment for built-in agents.

In Arandu this is a no-op: all inference runs against the single
local Ollama model the user configured in Settings, so per-agent tier
overrides serve no purpose. The function stays in the codebase as an
extension point for downstream builds that map agents to multiple
model tiers.

sensitivity_tier: 1
"""

from __future__ import annotations

AGENT_TIER_MAP: dict[str, str] = {}


def tier_model_for(agent_id: str) -> str | None:
    """Return the tier-default model for ``agent_id``, or None.

    Always returns None in OSS — every agent falls through to the
    user's :data:`llm_model` setting.

    sensitivity_tier: 1
    """
    return AGENT_TIER_MAP.get(agent_id)


__all__ = ["AGENT_TIER_MAP", "tier_model_for"]
