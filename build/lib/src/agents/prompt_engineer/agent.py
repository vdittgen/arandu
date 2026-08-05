"""Pydantic AI prompt engineer.

Reads the spec of a user agent (saved or unsaved) and proposes a
:class:`PromptSuggestion` — either a full rewrite of the system prompt
plus description with categorised :class:`Improvement` edits, or a
refusal with the manual edits the user should apply first. Also emits
``system_prompt_additions`` — short imperative lines the user may
append verbatim without taking the full rewrite.

One LLM call per suggestion. After the call a deterministic post-check
runs locally:

- Hallucinated ``original_snippet`` values (text that does not appear
  verbatim in the input prompt or description) are pruned from the
  improvements list.
- When ``can_improve=True``, both rewrite fields must be non-empty and
  the rewritten system prompt must differ from the input; otherwise
  the suggestion is downgraded to a refusal.
- When ``can_improve=False``, ``reason_if_not`` must be non-empty.

No LLM retry. The cost stays predictable; the user can always click
"Improve prompts" again for a second pass.

sensitivity_tier: 1
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from src.agents.core.agent_base import SBAgent
from src.agents.core.output_types import Improvement, PromptSuggestion
from src.agents.core.scheduler import Tier
from src.models.prompt_cache import PROMPTS_DIR, FrozenPromptTemplate

logger = logging.getLogger(__name__)

_TEMPLATE = FrozenPromptTemplate(PROMPTS_DIR / "prompt_engineer_v1.txt")
DEFAULT_SYSTEM_PROMPT = _TEMPLATE.prefix


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalFailure:
    """One failed eval case fed into the next prompt-engineer round.

    Same shape as :class:`src.agents.model_picker.FailedCase` but kept
    separate so the two agents can evolve independently. ``reason``
    feeds the language / format / refusal-semantics detection in the
    surgical-additions rubric.

    sensitivity_tier: 1
    """

    name: str
    evaluator: str
    reason: str


@dataclass(frozen=True)
class PromptEngineerInput:
    """Input deps for :class:`PromptEngineerAgent`.

    ``agent_id`` is ``None`` when the user is iterating in the
    create-agent wizard (no DB row exists yet). Built-in agents have
    ids that do NOT start with ``user.``; the prompt's refusal
    criteria use that distinction to guard against rewrites of locked
    agents.

    ``prior_eval_failures`` carries failed cases from the last eval
    run when available. The agent uses them to target the actual gaps
    the eval suite uncovered rather than guessing from the prompt
    alone. ``prior_attempts`` carries reasons earlier suggestions were
    rejected by the deterministic post-check (currently unused by the
    CLI handler — the input shape leaves room for a future iterative
    loop).

    sensitivity_tier: 1
    """

    name: str
    description: str
    system_prompt: str
    max_sensitivity_tier: int
    agent_id: str | None = None
    output_schema: str | None = None
    available_tools: tuple[str, ...] = ()
    available_skills: tuple[str, ...] = ()
    enabled_mcp_tools: tuple[str, ...] = ()
    has_dataset: bool = False
    prior_eval_failures: tuple[EvalFailure, ...] = ()
    prior_attempts: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class PromptEngineerAgent(SBAgent[PromptEngineerInput, PromptSuggestion]):
    """Rewrite a user agent's system prompt + description.

    Deps is a :class:`PromptEngineerInput`. The agent emits a
    :class:`PromptSuggestion`; :meth:`suggest` wraps :meth:`run` with
    a deterministic post-check that prunes hallucinated snippets and
    downgrades vacuous rewrites to refusals.

    sensitivity_tier: 1
    """

    agent_id = "prompt_engineer"
    output_type = PromptSuggestion
    tier = Tier.BACKGROUND
    system_prompt = DEFAULT_SYSTEM_PROMPT

    def build_prompt(self, deps: PromptEngineerInput) -> str:
        """Render the variable user message.

        The improvement rubric lives in the frozen system prompt; only
        per-call data goes here.

        sensitivity_tier: 1
        """
        body: dict[str, object] = {
            "agent_id": deps.agent_id,
            "name": deps.name,
            "description": deps.description,
            "system_prompt": deps.system_prompt,
            "max_sensitivity_tier": deps.max_sensitivity_tier,
            "output_schema": deps.output_schema,
            "available_tools": list(deps.available_tools),
            "available_skills": list(deps.available_skills),
            "enabled_mcp_tools": list(deps.enabled_mcp_tools),
            "has_dataset": deps.has_dataset,
            "prior_eval_failures": [
                {
                    "name": f.name,
                    "evaluator": f.evaluator,
                    "reason": f.reason,
                }
                for f in deps.prior_eval_failures
            ],
        }
        if deps.prior_attempts:
            body["prior_attempts"] = list(deps.prior_attempts)
        return (
            "Improve this agent's prompts:\n\n"
            f"{json.dumps(body, indent=2, sort_keys=True)}"
        )

    def suggest(self, deps: PromptEngineerInput) -> PromptSuggestion:
        """Run the LLM once, then validate the result locally.

        The deterministic checks here are intentionally narrow — they
        catch outright hallucinations and empty rewrites without
        spending a second LLM call. Semantic critique (does the
        rewrite actually make the agent better?) is the user's job
        via the modal.

        sensitivity_tier: 1
        """
        result = self._run_once(deps)
        return _post_check(result, deps)

    # ----- internals --------------------------------------------------

    def _run_once(self, deps: PromptEngineerInput) -> PromptSuggestion:
        """One LLM round-trip, with graceful fallback on failure.

        sensitivity_tier: 1
        """
        try:
            record = self.run(deps)
        except Exception as exc:  # noqa: BLE001
            logger.exception("PromptEngineerAgent.run failed")
            return PromptSuggestion(
                can_improve=False,
                reason_if_not=f"model error: {exc}",
                improvements=[
                    Improvement(
                        category="clarity",
                        suggested_replacement=(
                            "Verify the remote LLM endpoint is configured "
                            "and reachable, then try again."
                        ),
                        rationale=(
                            "The prompt engineer could not reach the "
                            "model."
                        ),
                    ),
                    Improvement(
                        category="clarity",
                        suggested_replacement=(
                            "If the endpoint is healthy, re-open this "
                            "modal — transient errors usually clear on "
                            "retry."
                        ),
                        rationale=(
                            "A second attempt often succeeds when the "
                            "first was a network blip."
                        ),
                    ),
                ],
            )
        if record is None or record.output is None:
            reason = record.error if record else "no model output"
            return PromptSuggestion(
                can_improve=False,
                reason_if_not=reason,
                improvements=[
                    Improvement(
                        category="clarity",
                        suggested_replacement=(
                            "Try again — the model returned no output."
                        ),
                        rationale=(
                            "Empty model responses are usually transient "
                            "and clear on retry."
                        ),
                    ),
                    Improvement(
                        category="clarity",
                        suggested_replacement=(
                            "If the issue persists, switch the agent's "
                            "model override to a known-good model."
                        ),
                        rationale=(
                            "Persistent empty outputs typically indicate "
                            "a misconfigured model."
                        ),
                    ),
                ],
            )
        return record.output


# ---------------------------------------------------------------------------
# Deterministic post-check
# ---------------------------------------------------------------------------


def _post_check(
    suggestion: PromptSuggestion, deps: PromptEngineerInput,
) -> PromptSuggestion:
    """Prune hallucinations and downgrade vacuous rewrites.

    Rules (see the agent's module docstring for the rationale):

    - When ``can_improve=True``:
      * Hallucinated ``original_snippet`` values are dropped silently.
      * If the rewritten system prompt is empty OR byte-equal to the
        input, downgrade to a refusal.
      * If the rewritten description is empty, downgrade to a refusal.
    - When ``can_improve=False``:
      * ``reason_if_not`` must be non-empty; otherwise stamp a
        generic reason.

    sensitivity_tier: 1
    """
    if not suggestion.can_improve:
        if not (suggestion.reason_if_not or "").strip():
            return suggestion.model_copy(update={
                "reason_if_not": "no improvement returned by the model",
            })
        return suggestion

    haystack = (deps.system_prompt or "") + "\n" + (deps.description or "")
    pruned: list[Improvement] = [
        item for item in suggestion.improvements
        if not item.original_snippet
        or item.original_snippet in haystack
    ]

    rewrite = suggestion.improved_system_prompt.strip()
    description_rewrite = suggestion.improved_description.strip()
    if not rewrite or not description_rewrite:
        return _downgrade(
            suggestion,
            reason=(
                "model claimed an improvement but returned an empty "
                "rewrite field"
            ),
        )
    if suggestion.improved_system_prompt == deps.system_prompt:
        return _downgrade(
            suggestion,
            reason="model returned the original prompt unchanged",
        )

    return suggestion.model_copy(update={"improvements": pruned})


def _downgrade(
    original: PromptSuggestion, *, reason: str,
) -> PromptSuggestion:
    """Stamp a structured refusal preserving the model's notes.

    sensitivity_tier: 1
    """
    return PromptSuggestion(
        can_improve=False,
        reason_if_not=reason,
        improvements=list(original.improvements) or [
            Improvement(
                category="clarity",
                suggested_replacement=(
                    "Re-open the modal — the previous attempt did not "
                    "produce a usable rewrite."
                ),
                rationale=(
                    "A second attempt usually succeeds when the first "
                    "returned empty fields."
                ),
            ),
            Improvement(
                category="clarity",
                suggested_replacement=(
                    "If the issue persists, tighten the description "
                    "first — the engineer leans on it heavily."
                ),
                rationale=(
                    "A clearer description gives the engineer more to "
                    "anchor the rewrite on."
                ),
            ),
        ],
        notes=list(original.notes),
        confidence=original.confidence,
    )


# ---------------------------------------------------------------------------
# Registry hook
# ---------------------------------------------------------------------------


def register_prompt_engineer_agent() -> None:
    """Register the prompt engineer as a non-editable system agent.

    Idempotent. Mirrors :func:`register_model_picker_agent` — the
    agent appears as a locked card on the Agents page and refuses
    config patches.

    sensitivity_tier: 1
    """
    from src.agents.core.config_store import AgentConfig
    from src.agents.core.registry import (
        AgentDefinition,
        get_agent,
        register_agent,
    )

    if get_agent("prompt_engineer") is not None:
        return

    default = AgentConfig(
        agent_id="prompt_engineer",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        model_route="inherit",
        model_override=None,
        enabled_tools=(),
        enabled_skills=(),
        editable=False,
    )
    register_agent(AgentDefinition(
        agent_id="prompt_engineer",
        name="Prompt Engineer",
        description=(
            "Rewrites a user agent's system prompt and description for "
            "clarity, expected-output strictness, language pinning, "
            "format strictness, scope, and safety. Returns either a "
            "full rewrite plus surgical additions, or a refusal with "
            "concrete manual edits when the agent's purpose is too "
            "vague to improve."
        ),
        category="advisor",
        parent_agent="brain",
        tier=Tier.BACKGROUND,
        max_sensitivity_tier=1,
        editable=False,
        default_config=default,
        available_tools=(),
        available_skills=(),
        output_schema="PromptSuggestion",
        pattern="single",
        factory=PromptEngineerAgent,
        tags=("locked", "advisor", "builtin"),
    ))


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "EvalFailure",
    "PromptEngineerAgent",
    "PromptEngineerInput",
    "register_prompt_engineer_agent",
]
