"""Skill Creator agent — generates SKILL.md files from conversation traces.

Analyzes completed Brain/Chat interactions that involved multiple tool
calls and determines whether the interaction represents a reusable
procedure worth capturing as a skill. If so, generates a SKILL.md
following the open standard and saves it as a pending auto-learned skill.

sensitivity_tier: 2 (reads prompts and agent outputs from run log)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pydantic import Field

from src.agents.core.agent_base import SBAgent
from src.agents.core.output_types import AgentOutput
from src.agents.core.scheduler import Tier

logger = logging.getLogger(__name__)

MIN_TOOL_CALLS = 3

SYSTEM_PROMPT = """\
You are a skill extraction agent. You analyze conversation traces from \
an AI assistant and determine whether they contain a reusable procedure \
worth saving as a skill.

A skill is a SKILL.md file that teaches the agent how to handle a \
specific type of request. Skills capture procedures, not facts.

Good skills:
- Multi-step workflows (weekly review, contact follow-up, message triage)
- Procedures that the user is likely to repeat
- Tasks that required specific sequencing of tool calls

Bad skills (do not create):
- Simple Q&A ("what's the weather")
- One-off lookups ("who sent me that email")
- Conversations that ended in errors
- Tasks too specific to generalize ("email John about Tuesday's meeting")

When you determine a trace IS worth capturing as a skill, generate the \
SKILL.md content following this format:

```
---
name: <Human-readable name>
description: <One line describing when this skill should activate>
version: 1
tags: [<relevant, tags>]
sensitivity_tier: <1, 2, or 3>
source: auto-learned
---

## When to Use
<Describe the trigger patterns — what the user says/asks>

## Procedure
<Numbered steps extracted from the tool call sequence>

## Output Format
<How results should be structured>

## Pitfalls
<Gotchas extracted from retries or error-recovery in the trace>
```
"""


class SkillExtractionResult(AgentOutput):
    """Output of the skill creator agent.

    sensitivity_tier: 1
    """

    is_reusable: bool = Field(
        description="Whether this trace represents a reusable procedure",
    )
    skill_name: str = Field(
        default="",
        description="Human-readable skill name (empty if not reusable)",
    )
    skill_md: str = Field(
        default="",
        description="Full SKILL.md content (empty if not reusable)",
    )
    reason: str = Field(
        default="",
        description="Why this trace was or wasn't captured as a skill",
    )


@dataclass(frozen=True)
class SkillCreatorDeps:
    """Input to the skill creator: a conversation trace.

    sensitivity_tier: 2
    """

    user_query: str
    agent_output: str
    tool_calls: list[dict]
    tool_call_count: int
    agent_id: str


class SkillCreatorAgent(SBAgent[SkillCreatorDeps, SkillExtractionResult]):
    """Analyzes conversation traces and generates SKILL.md files.

    sensitivity_tier: 2
    """

    agent_id = "skill_creator"
    output_type = SkillExtractionResult
    tier = Tier.BACKGROUND
    system_prompt = SYSTEM_PROMPT

    def build_prompt(self, deps: SkillCreatorDeps) -> str:
        """Format the conversation trace for analysis.

        sensitivity_tier: 2
        """
        tool_summary = "\n".join(
            f"  {i+1}. {tc.get('name', 'unknown')}({tc.get('args_summary', '')})"
            f" → {tc.get('result_summary', 'ok')}"
            for i, tc in enumerate(deps.tool_calls)
        )

        return (
            f"Analyze this conversation trace and determine if it "
            f"contains a reusable procedure worth saving as a skill.\n\n"
            f"## User Query\n{deps.user_query}\n\n"
            f"## Tool Calls ({deps.tool_call_count} total)\n"
            f"{tool_summary}\n\n"
            f"## Agent Output\n{deps.agent_output}\n\n"
            f"If this IS a reusable procedure, generate the full SKILL.md "
            f"content. If not, explain why briefly."
        )


def maybe_create_skill(
    user_query: str,
    agent_output: str,
    tool_calls: list[dict],
    agent_id: str = "brain",
) -> SkillExtractionResult | None:
    """Run the skill creator if the interaction had enough tool calls.

    Returns ``None`` if the threshold wasn't met (skipped) or if
    the agent run fails. Returns the extraction result otherwise.

    sensitivity_tier: 2
    """
    if len(tool_calls) < MIN_TOOL_CALLS:
        return None

    deps = SkillCreatorDeps(
        user_query=user_query,
        agent_output=agent_output,
        tool_calls=tool_calls,
        tool_call_count=len(tool_calls),
        agent_id=agent_id,
    )

    agent = SkillCreatorAgent()
    try:
        record = agent.run(deps)
        if record.error:
            logger.warning("Skill creator failed: %s", record.error)
            return None
        return record.output
    except Exception:
        logger.warning("Skill creator exception", exc_info=True)
        return None


def save_auto_learned_skill(result: SkillExtractionResult) -> str | None:
    """Save an auto-learned skill to disk if the result is reusable.

    Returns the skill ID if saved, None otherwise.

    sensitivity_tier: 1
    """
    if not result.is_reusable or not result.skill_md.strip():
        return None

    try:
        from src.agent_runtime.skill_loader import SkillLoader

        loader = SkillLoader()
        meta = loader.create(result.skill_name or "auto-skill", result.skill_md)
        logger.info(
            "Auto-learned skill saved: %s (%s)",
            meta.id, result.skill_name,
        )
        return meta.id
    except Exception:
        logger.warning("Failed to save auto-learned skill", exc_info=True)
        return None


def register_skill_creator_agent() -> None:
    """Register the skill creator in the global agent registry.

    sensitivity_tier: 1
    """
    from src.agents.core.config_store import AgentConfig
    from src.agents.core.registry import (
        AgentDefinition,
        get_agent,
        register_agent,
    )

    if get_agent("skill_creator") is not None:
        return

    default = AgentConfig(
        agent_id="skill_creator",
        system_prompt=SYSTEM_PROMPT,
        model_route="inherit",
        model_override=None,
        enabled_tools=(),
        enabled_skills=(),
        editable=False,
    )
    register_agent(AgentDefinition(
        agent_id="skill_creator",
        name="Skill Creator",
        description=(
            "Analyzes conversation traces and auto-generates "
            "SKILL.md files from successful multi-step interactions. "
            "Runs in the background after Brain completes a task "
            "with 3+ tool calls."
        ),
        category="system",
        parent_agent=None,
        tier=Tier.BACKGROUND,
        max_sensitivity_tier=2,
        editable=False,
        default_config=default,
        available_tools=(),
        available_skills=(),
        output_schema="SkillExtractionResult",
        pattern="single",
        factory=SkillCreatorAgent,
        tags=("system", "skills"),
    ))
