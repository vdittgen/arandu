"""Pydantic AI dataset creator.

Reads the spec of a new (or existing) user agent and proposes a
starter :class:`DatasetSuggestion` — either a YAML dataset of 6-12
cases plus per-case evaluators, or a refusal with concrete
``improvement_hints`` when the agent's purpose is not yet evaluable.

The agent runs one LLM call, then runs the deterministic structural
check from :mod:`src.agents.dataset_validator` against the generated
YAML. On failure it retries once with the validator errors fed back
into the prompt. The single retry keeps cost predictable; semantic
critique is the user's job via the UI modal.

When an existing user dataset is on disk, :meth:`suggest` operates in
append mode: the new cases are merged into the existing YAML, only
keeping new cases whose ``name`` does not collide.

sensitivity_tier: 1
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.agents.core.agent_base import SBAgent
from src.agents.core.output_types import DatasetSuggestion
from src.agents.core.scheduler import Tier
from src.models.prompt_cache import PROMPTS_DIR, FrozenPromptTemplate

logger = logging.getLogger(__name__)

_TEMPLATE = FrozenPromptTemplate(PROMPTS_DIR / "dataset_creator_v1.txt")
DEFAULT_SYSTEM_PROMPT = _TEMPLATE.prefix


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetCreatorInput:
    """Input deps for :class:`DatasetCreatorAgent`.

    Not a pydantic ``AgentOutput`` — this is the agent's input, not its
    output. ``agent_id`` is ``None`` when the user is previewing during
    the create-agent flow (the agent row does not exist yet).

    sensitivity_tier: 1
    """

    name: str
    description: str
    system_prompt: str
    max_sensitivity_tier: int
    agent_id: str | None = None
    output_schema: str | None = None
    available_tools: tuple[str, ...] = ()
    existing_case_names: tuple[str, ...] = ()
    prior_errors: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class DatasetCreatorAgent(SBAgent[DatasetCreatorInput, DatasetSuggestion]):
    """Propose a starter eval dataset for a user agent.

    Deps is a :class:`DatasetCreatorInput`. The agent emits a
    :class:`DatasetSuggestion`; :meth:`suggest` wraps :meth:`run` with
    a structural validation step + at most one retry on failure.

    sensitivity_tier: 1
    """

    agent_id = "dataset_creator"
    output_type = DatasetSuggestion
    tier = Tier.BACKGROUND
    system_prompt = DEFAULT_SYSTEM_PROMPT

    def build_prompt(self, deps: DatasetCreatorInput) -> str:
        """Render the variable user message.

        The evaluator catalog menu lives in the frozen system prompt
        (it changes rarely — extending it is a deliberate change that
        also bumps the prompt-cache hash). Only per-call data goes in
        this body: the agent spec, any existing case names, and prior
        retry errors when present.

        sensitivity_tier: 1
        """
        body: dict[str, Any] = {
            "agent_id": deps.agent_id,
            "name": deps.name,
            "description": deps.description,
            "system_prompt": deps.system_prompt,
            "max_sensitivity_tier": deps.max_sensitivity_tier,
            "output_schema": deps.output_schema,
            "available_tools": list(deps.available_tools),
            "existing_case_names": list(deps.existing_case_names),
        }
        if deps.prior_errors:
            body["prior_errors"] = list(deps.prior_errors)
        return (
            "Generate a DatasetSuggestion for this agent spec:\n\n"
            f"{json.dumps(body, indent=2, sort_keys=True)}"
        )

    def suggest(
        self,
        deps: DatasetCreatorInput,
        *,
        existing_yaml: str | None = None,
    ) -> DatasetSuggestion:
        """Generate, validate, optionally merge, and return the proposal.

        ``existing_yaml`` (when provided) is the YAML on disk for the
        target agent. New cases are merged into it; the returned
        ``dataset_yaml`` is the fully-merged YAML so the UI shows the
        final state the user is about to accept.

        sensitivity_tier: 1
        """
        first = self._run_once(deps)
        if not first.can_create:
            return first

        suggestion = first
        # Structural validation gate. Import lazily so the test suite
        # can construct this agent without pulling pydantic-ai.
        from src.agents.dataset_validator.agent import structural_check

        result = structural_check(suggestion.dataset_yaml)
        if not result.valid:
            retry_deps = _deps_with_errors(deps, result.errors)
            second = self._run_once(retry_deps)
            if not second.can_create:
                return _refusal_from_errors(result.errors, notes=second.notes)
            result2 = structural_check(second.dataset_yaml)
            if not result2.valid:
                return _refusal_from_errors(result2.errors, notes=second.notes)
            suggestion = second

        if existing_yaml:
            merged_yaml, merged_count = merge_user_dataset_yaml(
                existing_yaml, suggestion.dataset_yaml,
            )
            suggestion = suggestion.model_copy(update={
                "dataset_yaml": merged_yaml,
                "case_count": merged_count,
            })
        return suggestion

    # ----- internals --------------------------------------------------

    def _run_once(self, deps: DatasetCreatorInput) -> DatasetSuggestion:
        """One LLM round-trip, with graceful fallback on failure.

        sensitivity_tier: 1
        """
        try:
            record = self.run(deps)
        except Exception as exc:  # noqa: BLE001
            logger.exception("DatasetCreatorAgent.run failed")
            return DatasetSuggestion(
                can_create=False,
                reason_if_not=f"model error: {exc}",
                improvement_hints=[
                    "Verify the remote LLM endpoint is configured and reachable.",
                ],
            )
        if record is None or record.output is None:
            return DatasetSuggestion(
                can_create=False,
                reason_if_not=record.error if record else "no model output",
                improvement_hints=[
                    "Try again — the model returned no output.",
                ],
            )
        return record.output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _deps_with_errors(
    deps: DatasetCreatorInput, errors: tuple[str, ...],
) -> DatasetCreatorInput:
    """Return a copy of ``deps`` with ``prior_errors`` populated.

    sensitivity_tier: 1
    """
    return DatasetCreatorInput(
        name=deps.name,
        description=deps.description,
        system_prompt=deps.system_prompt,
        max_sensitivity_tier=deps.max_sensitivity_tier,
        agent_id=deps.agent_id,
        output_schema=deps.output_schema,
        available_tools=deps.available_tools,
        existing_case_names=deps.existing_case_names,
        prior_errors=tuple(errors),
    )


def _refusal_from_errors(
    errors: tuple[str, ...], *, notes: list[str] | None = None,
) -> DatasetSuggestion:
    """Build a refusal :class:`DatasetSuggestion` after retry exhaustion.

    sensitivity_tier: 1
    """
    summary = "generated YAML failed structural validation after retry"
    return DatasetSuggestion(
        can_create=False,
        reason_if_not=summary + ": " + "; ".join(errors),
        notes=list(notes or []),
        improvement_hints=[
            "Open the generated YAML in the modal, fix the structural issues, "
            "and save manually.",
        ],
    )


def merge_user_dataset_yaml(
    existing_yaml: str, new_yaml: str,
) -> tuple[str, int]:
    """Merge ``new_yaml``'s cases into ``existing_yaml``.

    Keeps every existing case and appends only those new cases whose
    ``name`` does not already appear. Returns ``(merged_yaml,
    case_count)``. Both inputs are assumed to have passed
    :func:`src.agents.dataset_validator.structural_check`.

    sensitivity_tier: 1
    """
    existing = yaml.safe_load(existing_yaml) or {}
    new = yaml.safe_load(new_yaml) or {}
    if not isinstance(existing, dict):
        existing = {}
    if not isinstance(new, dict):
        new = {}
    existing_cases = list(existing.get("cases") or [])
    new_cases = list(new.get("cases") or [])
    seen = {
        c.get("name") for c in existing_cases
        if isinstance(c, dict) and isinstance(c.get("name"), str)
    }
    appended = [
        c for c in new_cases
        if isinstance(c, dict) and c.get("name") not in seen
    ]
    merged = dict(existing)
    merged["cases"] = existing_cases + appended
    rendered = yaml.safe_dump(
        merged, sort_keys=False, allow_unicode=True,
    )
    return rendered, len(merged["cases"])


def existing_case_names_from_yaml(content: str | None) -> tuple[str, ...]:
    """Return the case names already present in ``content``.

    Tolerant of empty / invalid YAML — returns ``()`` in that case so
    the caller can still proceed.

    sensitivity_tier: 1
    """
    if not content:
        return ()
    try:
        parsed = yaml.safe_load(content) or {}
    except yaml.YAMLError:
        return ()
    if not isinstance(parsed, dict):
        return ()
    names: list[str] = []
    for case in parsed.get("cases") or []:
        if isinstance(case, dict):
            name = case.get("name")
            if isinstance(name, str):
                names.append(name)
    return tuple(names)


def read_existing_user_dataset(agent_id: str) -> str | None:
    """Return the on-disk YAML for ``agent_id`` or ``None`` when absent.

    sensitivity_tier: 1
    """
    if not agent_id:
        return None
    path = (
        Path.home() / ".arandu" / "user_eval_datasets" / f"{agent_id}.yaml"
    )
    try:
        return path.read_text(encoding="utf-8") if path.is_file() else None
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Registry hook
# ---------------------------------------------------------------------------


def register_dataset_creator_agent() -> None:
    """Register the dataset creator as a non-editable system agent.

    Idempotent. Mirrors :func:`register_dataset_validator_agent` — the
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

    if get_agent("dataset_creator") is not None:
        return

    default = AgentConfig(
        agent_id="dataset_creator",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        model_route="inherit",
        model_override=None,
        enabled_tools=(),
        enabled_skills=(),
        editable=False,
    )
    register_agent(AgentDefinition(
        agent_id="dataset_creator",
        name="Dataset Creator",
        description=(
            "Proposes a starter eval dataset for a newly-created user "
            "agent. Reads the agent's name, description, and system "
            "prompt, infers its purpose, and either generates 6-12 "
            "cases with evaluators or refuses with concrete edits the "
            "user can apply to clarify the agent."
        ),
        category="validator",
        parent_agent="brain",
        tier=Tier.BACKGROUND,
        max_sensitivity_tier=1,
        editable=False,
        default_config=default,
        available_tools=(),
        available_skills=(),
        output_schema="DatasetSuggestion",
        pattern="single",
        factory=DatasetCreatorAgent,
        tags=("validator", "builtin"),
    ))


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "DatasetCreatorAgent",
    "DatasetCreatorInput",
    "existing_case_names_from_yaml",
    "merge_user_dataset_yaml",
    "read_existing_user_dataset",
    "register_dataset_creator_agent",
]
