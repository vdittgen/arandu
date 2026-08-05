"""Pydantic AI model picker.

Reads the spec of a (saved or unsaved) user agent and proposes two
:class:`ModelOption` picks — a ``best_overall`` capability fit and a
``cost_effective`` cheaper pick that still clears the agent's
complexity bar. Returns a :class:`ModelRecommendation` whose
``can_recommend`` flag mirrors :class:`DatasetSuggestion`'s
``can_create`` refusal pattern.

One LLM call per recommendation. The CLI handler validates that both
picks reference ids the configured endpoints actually expose (live
``/models`` enumeration) and downgrades to a refusal when the model
hallucinated an id — keeping the cost bounded.

sensitivity_tier: 1
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from src.agents.core.agent_base import SBAgent
from src.agents.core.output_types import ModelRecommendation
from src.agents.core.scheduler import Tier
from src.models.prompt_cache import PROMPTS_DIR, FrozenPromptTemplate

logger = logging.getLogger(__name__)

_TEMPLATE = FrozenPromptTemplate(PROMPTS_DIR / "model_picker_v1.txt")
DEFAULT_SYSTEM_PROMPT = _TEMPLATE.prefix


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FailedCase:
    """One failed eval case fed back into the next picker round.

    The LLM uses ``name`` + ``evaluator`` + ``reason`` to infer which
    capability gap of the tested model caused the failure (e.g.
    "model_id refused to emit a JSON object, so pick a stronger
    instruction-following family next").

    sensitivity_tier: 1
    """

    name: str
    evaluator: str
    reason: str


@dataclass(frozen=True)
class PriorAttempt:
    """A previously-tested model and the eval cases it failed on.

    Used to teach the picker which directions NOT to repeat. Multiple
    attempts can accumulate across iterations of the suggest → use →
    eval loop.

    sensitivity_tier: 1
    """

    model_id: str
    route: str
    failed_cases: tuple[FailedCase, ...] = ()


@dataclass(frozen=True)
class ModelPickerInput:
    """Input deps for :class:`ModelPickerAgent`.

    The two ``available_*`` tuples are the live id lists returned by
    :func:`src.agents.core.model_factory.list_models` for the
    ``remote`` and ``local`` routes respectively. The CLI handler
    fetches them at call time so the LLM only picks among real ids.

    ``excluded_models`` and ``prior_attempts`` carry feedback from
    earlier iterations of the same suggestion loop:

    - ``excluded_models`` lists ids the picker must NOT propose again
      (typically because the user already tested them and got failures
      or eval errors).
    - ``prior_attempts`` describes *why* each excluded model fell short
      so the LLM can reason about which capability gap the next pick
      should close (e.g. weak JSON-mode → suggest a stronger
      instruction-following model).

    sensitivity_tier: 1
    """

    name: str
    description: str
    system_prompt: str
    max_sensitivity_tier: int
    available_remote_models: tuple[str, ...] = ()
    available_local_models: tuple[str, ...] = ()
    output_schema: str | None = None
    enabled_skills: tuple[str, ...] = ()
    enabled_mcp_tools: tuple[str, ...] = ()
    agent_id: str | None = None
    excluded_models: tuple[str, ...] = ()
    prior_attempts: tuple[PriorAttempt, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class ModelPickerAgent(SBAgent[ModelPickerInput, ModelRecommendation]):
    """Recommend ``best_overall`` + ``cost_effective`` models for an agent.

    Deps is a :class:`ModelPickerInput`. The agent emits a
    :class:`ModelRecommendation`; :meth:`recommend` wraps :meth:`run`
    with a catalog-membership check that downgrades to a refusal when
    a returned id is not in the live lists.

    sensitivity_tier: 1
    """

    agent_id = "model_picker"
    output_type = ModelRecommendation
    tier = Tier.BACKGROUND
    system_prompt = DEFAULT_SYSTEM_PROMPT

    def build_prompt(self, deps: ModelPickerInput) -> str:
        """Render the variable user message.

        The instructional rubric lives in the frozen system prompt;
        per-call data (the agent spec + the live catalogs + iteration
        feedback) goes here.

        sensitivity_tier: 1
        """
        body = {
            "agent_id": deps.agent_id,
            "name": deps.name,
            "description": deps.description,
            "system_prompt": deps.system_prompt,
            "max_sensitivity_tier": deps.max_sensitivity_tier,
            "output_schema": deps.output_schema,
            "enabled_skills": list(deps.enabled_skills),
            "enabled_mcp_tools": list(deps.enabled_mcp_tools),
            "available_remote_models": list(deps.available_remote_models),
            "available_local_models": list(deps.available_local_models),
        }
        if deps.excluded_models:
            body["excluded_models"] = list(deps.excluded_models)
        if deps.prior_attempts:
            body["prior_attempts"] = [
                {
                    "model_id": a.model_id,
                    "route": a.route,
                    "failed_cases": [
                        {
                            "name": fc.name,
                            "evaluator": fc.evaluator,
                            "reason": fc.reason,
                        }
                        for fc in a.failed_cases
                    ],
                }
                for a in deps.prior_attempts
            ]
        return (
            "Recommend two models for this agent spec:\n\n"
            f"{json.dumps(body, indent=2, sort_keys=True)}"
        )

    def recommend(self, deps: ModelPickerInput) -> ModelRecommendation:
        """Run the LLM once, validate against the live catalog.

        When the catalogs are both empty, return a refusal without
        spending an LLM call. When the LLM returns a recommendation
        whose ids are not present in the live lists, downgrade to a
        refusal — never surface a model the user cannot pick.

        sensitivity_tier: 1
        """
        if not deps.available_remote_models and not deps.available_local_models:
            return ModelRecommendation(
                can_recommend=False,
                reason_if_not="no models available from the configured endpoints",
                improvement_hints=[
                    "Configure the LLM endpoint in Settings → AI Model "
                    "and verify it is reachable.",
                ],
            )

        result = self._run_once(deps)
        if not result.can_recommend:
            return result

        catalog = _catalog_map(deps)
        downgrade = _validate_catalog(result, catalog)
        if downgrade is not None:
            return downgrade

        excluded = _validate_exclusions(result, deps.excluded_models)
        if excluded is not None:
            return excluded
        return result

    # ----- internals --------------------------------------------------

    def _run_once(self, deps: ModelPickerInput) -> ModelRecommendation:
        """One LLM round-trip, with graceful fallback on failure.

        sensitivity_tier: 1
        """
        try:
            record = self.run(deps)
        except Exception as exc:  # noqa: BLE001
            logger.exception("ModelPickerAgent.run failed")
            return ModelRecommendation(
                can_recommend=False,
                reason_if_not=f"model error: {exc}",
                improvement_hints=[
                    "Verify the remote LLM endpoint is configured and "
                    "reachable, then try again.",
                ],
            )
        if record is None or record.output is None:
            return ModelRecommendation(
                can_recommend=False,
                reason_if_not=record.error if record else "no model output",
                improvement_hints=[
                    "Try again — the model returned no output.",
                ],
            )
        return record.output


# ---------------------------------------------------------------------------
# Catalog validation
# ---------------------------------------------------------------------------


def _catalog_map(deps: ModelPickerInput) -> dict[str, str]:
    """Return ``{model_id: route}`` from the live remote + local lists.

    When the same id appears in both routes (rare) ``remote`` wins so
    the route field on the returned option still validates.

    sensitivity_tier: 1
    """
    catalog: dict[str, str] = {}
    for mid in deps.available_local_models:
        catalog[mid] = "local"
    for mid in deps.available_remote_models:
        catalog[mid] = "remote"
    return catalog


def _validate_exclusions(
    rec: ModelRecommendation,
    excluded: tuple[str, ...],
) -> ModelRecommendation | None:
    """Return a refusal when the picker re-suggested an excluded model.

    The exclusion list is feedback from prior iterations of the same
    suggestion loop: the user already tested those ids and at least
    one of them failed evaluation. Re-suggesting the same id wastes
    the user's time, so we treat it as a hallucination — same
    downgrade pattern :func:`_validate_catalog` uses.

    sensitivity_tier: 1
    """
    if not excluded:
        return None
    excluded_set = set(excluded)
    repeated: list[str] = []
    for option in (rec.best_overall, rec.cost_effective):
        if option is not None and option.model_id in excluded_set:
            repeated.append(option.model_id)
    if not repeated:
        return None
    return ModelRecommendation(
        can_recommend=False,
        reason_if_not=(
            "model re-suggested an id that the user already rejected: "
            + ", ".join(repeated)
        ),
        improvement_hints=[
            "Try again — the picker should pick a model outside the "
            "excluded list.",
        ],
    )


def _validate_catalog(
    rec: ModelRecommendation,
    catalog: dict[str, str],
) -> ModelRecommendation | None:
    """Return a refusal when any picked id is unknown, else ``None``.

    Also checks that each option's ``route`` matches the route the id
    is canonically drawn from — a mismatch is just as fatal as an
    unknown id (the UI would set the wrong route field).

    sensitivity_tier: 1
    """
    unknown: list[str] = []
    for label, option in (
        ("best_overall", rec.best_overall),
        ("cost_effective", rec.cost_effective),
    ):
        if option is None:
            unknown.append(f"{label} was null")
            continue
        canonical_route = catalog.get(option.model_id)
        if canonical_route is None:
            unknown.append(
                f"{label}.model_id={option.model_id!r} not in live catalog",
            )
            continue
        if option.route != canonical_route:
            unknown.append(
                f"{label}.route={option.route!r} does not match catalog "
                f"({canonical_route!r}) for {option.model_id!r}",
            )

    if not unknown:
        return None

    return ModelRecommendation(
        can_recommend=False,
        reason_if_not=(
            "model returned an id not present in the live catalog: "
            + "; ".join(unknown)
        ),
        improvement_hints=[
            "Try again — the model hallucinated a model id.",
        ],
    )


# ---------------------------------------------------------------------------
# Registry hook
# ---------------------------------------------------------------------------


def register_model_picker_agent() -> None:
    """Register the model picker as a non-editable system agent.

    Idempotent. Mirrors :func:`register_dataset_creator_agent` — the
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

    if get_agent("model_picker") is not None:
        return

    default = AgentConfig(
        agent_id="model_picker",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        model_route="inherit",
        model_override=None,
        enabled_tools=(),
        enabled_skills=(),
        editable=False,
    )
    register_agent(AgentDefinition(
        agent_id="model_picker",
        name="Model Picker",
        description=(
            "Recommends a best-overall and a cost-effective model for "
            "any user agent based on its name, description, system "
            "prompt, skills, and MCP tools. Refuses with concrete "
            "improvement hints when the purpose is too vague to map "
            "to a capability profile."
        ),
        category="advisor",
        parent_agent="brain",
        tier=Tier.BACKGROUND,
        max_sensitivity_tier=1,
        editable=False,
        default_config=default,
        available_tools=(),
        available_skills=(),
        output_schema="ModelRecommendation",
        pattern="single",
        factory=ModelPickerAgent,
        tags=("locked", "advisor", "builtin"),
    ))


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "FailedCase",
    "ModelPickerAgent",
    "ModelPickerInput",
    "PriorAttempt",
    "register_model_picker_agent",
]
