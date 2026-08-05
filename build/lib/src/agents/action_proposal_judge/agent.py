"""Action-proposal judge — an independent verifier over the primary extractor.

The primary path in :mod:`src.agents.brain.actions` produces an
:class:`ActionProposal` from a free-form user request via a mix of
deterministic regex extraction (``user_value_extractor``) and an
LLM-driven JSON extraction. Both have failed in practice — the LLM
hallucinated titles, the deterministic pass can't cover every shape.

This agent runs *after* the primary extractor, sees the same user
message plus the structured proposal, and emits an
:class:`ActionProposalVerdict`. The integration in
``build_action_proposal`` applies the judge's patches before the
confirmation card is rendered.

In Arandu the judge runs against the same local model as the
primary extractor (one Ollama model). Where multiple model families
are available, the judge can be pinned to a different one via the
Agents page so a hallucination shared by both extractor and judge is
harder to slip through.

sensitivity_tier: 2
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date as _date
from typing import Any

from src.agents.core.agent_base import SBAgent
from src.agents.core.output_types import ActionProposalVerdict
from src.agents.core.scheduler import Tier
from src.models.prompt_cache import PROMPTS_DIR, FrozenPromptTemplate

logger = logging.getLogger(__name__)

_TEMPLATE = FrozenPromptTemplate(
    PROMPTS_DIR / "action_proposal_judge_v1.txt",
)
DEFAULT_SYSTEM_PROMPT = _TEMPLATE.prefix


@dataclass
class JudgeDeps:
    """Per-call inputs for :class:`ActionProposalJudge`.

    sensitivity_tier: 2
    """

    user_message: str
    tool_name: str
    proposed_arguments: dict[str, Any]
    tool_schema: dict[str, Any]
    today_iso: str = ""


class ActionProposalJudge(SBAgent[JudgeDeps, ActionProposalVerdict]):
    """Independent verifier for action proposals.

    Mirrors the surface of other locked single-purpose classifiers in
    this codebase (Sensitivity, Labeler, Triage) — typed deps, typed
    output, default model lives in :data:`AGENT_TIER_MAP`.

    sensitivity_tier: 2
    """

    agent_id = "action_proposal_judge"
    output_type = ActionProposalVerdict
    tier = Tier.INTERACTIVE
    system_prompt = DEFAULT_SYSTEM_PROMPT

    def build_prompt(self, deps: JudgeDeps) -> str:
        """Render the dynamic portion the judge sees on top of the
        frozen system prompt.

        Schema and arguments are serialized with the most-compact JSON
        form (no indent, no sort_keys, no whitespace) — pretty-printing
        was ~30% of the input-token budget and the judge LLM doesn't
        care about formatting.

        sensitivity_tier: 2
        """
        today = deps.today_iso or _date.today().isoformat()
        schema_json = json.dumps(deps.tool_schema, separators=(",", ":"))
        args_json = json.dumps(
            deps.proposed_arguments, separators=(",", ":"),
        )
        return (
            f"USER_MESSAGE:\n{deps.user_message}\n\n"
            f"TOOL_NAME: {deps.tool_name}\n\n"
            f"TOOL_SCHEMA: {schema_json}\n\n"
            f"PROPOSED_ARGUMENTS: {args_json}\n\n"
            f"TODAY: {today}\n"
        )


def judge_action_proposal(
    *,
    user_message: str,
    tool_name: str,
    tool_schema: dict[str, Any],
    proposed_arguments: dict[str, Any],
    today_iso: str | None = None,
) -> ActionProposalVerdict | None:
    """Run the judge and return its verdict, or ``None`` on failure.

    Failure modes that yield ``None``:

    - The agent isn't registered (boot order issue / unit-test
      environment).
    - The underlying model isn't reachable (offline, missing keys).
    - The LLM returned no parseable output.

    A ``None`` result means "no opinion" — the caller should ship the
    primary's proposal as-is rather than block on a missing safeguard.
    This is deliberate: a judge that crashes the create flow when its
    model is down would be worse than no judge at all.

    sensitivity_tier: 2
    """
    from src.agents.core.registry import get_agent

    definition = get_agent("action_proposal_judge")
    if definition is None or definition.factory is None:
        logger.debug(
            "action_proposal_judge not registered — skipping judge step",
        )
        return None
    instance: ActionProposalJudge = definition.factory()
    deps = JudgeDeps(
        user_message=user_message,
        tool_name=tool_name,
        proposed_arguments=proposed_arguments,
        tool_schema=tool_schema,
        today_iso=today_iso or _date.today().isoformat(),
    )
    try:
        record = instance.run(deps)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "action_proposal_judge failed for %s: %s", tool_name, exc,
        )
        return None
    if record.output is None or record.error is not None:
        if record.error:
            logger.info(
                "action_proposal_judge error for %s: %s",
                tool_name, record.error,
            )
        return None
    return record.output


def register_action_proposal_judge() -> None:
    """Register the judge in the global agent registry.

    Registration is idempotent and locked (``editable=False``) — the
    judge's job is part of the action-proposal contract; user edits
    to its prompt would silently weaken the safeguard.

    sensitivity_tier: 1
    """
    from src.agents.core.config_store import AgentConfig
    from src.agents.core.registry import (
        AgentDefinition,
        get_agent,
        register_agent,
    )

    if get_agent("action_proposal_judge") is not None:
        return

    default = AgentConfig(
        agent_id="action_proposal_judge",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        model_route="inherit",
        model_override=None,
        enabled_tools=(),
        enabled_skills=(),
        editable=False,
    )
    register_agent(AgentDefinition(
        agent_id="action_proposal_judge",
        name="Action Proposal Judge",
        description=(
            "Independent verifier that inspects a proposed MCP "
            "action against the user's literal request before the "
            "confirmation card is rendered. Runs on a different LLM "
            "family from the primary extractor so the two are not "
            "correlated."
        ),
        category="judge",
        parent_agent="brain",
        tier=Tier.INTERACTIVE,
        max_sensitivity_tier=2,
        editable=False,
        default_config=default,
        available_tools=(),
        available_skills=(),
        output_schema="ActionProposalVerdict",
        pattern="single",
        factory=ActionProposalJudge,
        tags=("judge", "locked"),
    ))


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "ActionProposalJudge",
    "JudgeDeps",
    "judge_action_proposal",
    "register_action_proposal_judge",
]
