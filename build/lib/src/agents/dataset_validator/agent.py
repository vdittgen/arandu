"""Pydantic AI dataset validator.

Inspects a user-uploaded eval dataset YAML and returns a
:class:`DatasetValidationReport`. The agent does two things:

1. A deterministic structural pass (:func:`structural_check`) that
   parses the YAML, verifies the ``cases`` shape, and confirms every
   referenced evaluator is in :mod:`evals.evaluators`. This is the
   authority for ``valid`` / ``errors``.
2. An LLM pass that produces ``proposals`` — short prose suggestions
   for tightening the dataset (more edge cases, sharper bounds, etc.).

``firewall_verdict`` is filled in by the caller after running the
content through :class:`InjectionFirewall`.

sensitivity_tier: 1
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import yaml

from src.agents.core.agent_base import SBAgent
from src.agents.core.output_types import DatasetValidationReport
from src.agents.core.scheduler import Tier

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """\
You review YAML eval datasets for an AI assistant. Each dataset is a \
dict with a `cases` list; every case has `name`, `inputs`, and \
typically `expected_output` plus an evaluator list.

Your job is to PROPOSE improvements in prose — never overwrite the \
dataset, never invent new evaluator classes. Focus on:

- coverage gaps (missing tier-3 cases, missing failure modes)
- ambiguous inputs that the model could legitimately interpret two ways
- evaluator bounds that are too loose or too tight
- duplicate cases that don't add signal

Return a `DatasetValidationReport`. Set `valid=true` always — the \
deterministic structural pass will overwrite this field. Leave \
`errors` empty (also overwritten). Fill `proposals` with 0-6 short \
sentences. Leave `firewall_verdict` as the default `\"allow\"`.\
"""


# ---------------------------------------------------------------------------
# Deterministic structural pass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _StructuralResult:
    """Result of the YAML structural pass.

    sensitivity_tier: 1
    """

    valid: bool
    errors: tuple[str, ...]


def _known_evaluator_names() -> frozenset[str]:
    """Return the set of evaluator class names exported by ``evals``.

    Imported lazily so this module loads even when the optional
    ``evals`` extras are missing.

    sensitivity_tier: 1
    """
    try:
        from evals import evaluators as ev
    except Exception:  # noqa: BLE001
        return frozenset()
    return frozenset(getattr(ev, "__all__", ()))


def _evaluator_name(entry: Any) -> str | None:
    """Extract the evaluator class name from a YAML entry.

    Accepts both forms pydantic-evals supports:
    - long: ``{name: TierEquals, ...}``
    - shorthand: ``{TierEquals: {...args}}`` (single-key dict)

    sensitivity_tier: 1
    """
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        name = entry.get("name")
        if isinstance(name, str) and name:
            return name
        if len(entry) == 1:
            key = next(iter(entry))
            if isinstance(key, str) and key:
                return key
    return None


def structural_check(content: str) -> _StructuralResult:
    """Validate the shape of an eval-dataset YAML payload.

    Returns ``valid=True`` only when:

    - the YAML parses to a mapping with a ``cases`` list,
    - every case is a mapping with a non-empty ``name`` and ``inputs``,
    - every evaluator referenced under ``evaluators:`` is in the
      :mod:`evals.evaluators` allowlist.

    sensitivity_tier: 1
    """
    errors: list[str] = []
    try:
        parsed: Any = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        return _StructuralResult(valid=False, errors=(f"yaml parse error: {exc}",))

    if not isinstance(parsed, dict):
        return _StructuralResult(
            valid=False,
            errors=("top-level YAML must be a mapping with a `cases` key",),
        )

    cases = parsed.get("cases")
    if not isinstance(cases, list) or not cases:
        return _StructuralResult(
            valid=False,
            errors=("`cases` must be a non-empty list",),
        )

    known = _known_evaluator_names()
    seen_names: set[str] = set()
    for idx, case in enumerate(cases):
        prefix = f"case[{idx}]"
        if not isinstance(case, dict):
            errors.append(f"{prefix}: must be a mapping")
            continue
        name = case.get("name")
        if not isinstance(name, str) or not name.strip():
            errors.append(f"{prefix}: missing or empty `name`")
        elif name in seen_names:
            errors.append(f"{prefix}: duplicate name {name!r}")
        else:
            seen_names.add(name)
        if "inputs" not in case:
            errors.append(f"{prefix} ({name}): missing `inputs`")
        elif not isinstance(case["inputs"], str):
            errors.append(
                f"{prefix} ({name}): `inputs` must be a string "
                f"(got {type(case['inputs']).__name__}); user agents "
                f"take a single string at runtime — serialise structured "
                f"payloads to a JSON string instead",
            )

        evaluators = case.get("evaluators", [])
        if evaluators and not isinstance(evaluators, list):
            errors.append(
                f"{prefix} ({name}): `evaluators` must be a list",
            )
            evaluators = []
        for ev_entry in evaluators or []:
            ev_name = _evaluator_name(ev_entry)
            if not ev_name:
                errors.append(
                    f"{prefix} ({name}): evaluator entry missing name",
                )
                continue
            if known and ev_name not in known:
                errors.append(
                    f"{prefix} ({name}): unknown evaluator {ev_name!r}",
                )

    return _StructuralResult(valid=not errors, errors=tuple(errors))


# ---------------------------------------------------------------------------
# Schema canonicalisation
# ---------------------------------------------------------------------------


# Per-evaluator keyword aliases. The LLM frequently picks reasonable-
# but-wrong synonyms ("expected" instead of FieldEquals.value, "values"
# instead of FieldIn.choices). Renaming the aliases at the boundary lets
# the dataset load through pydantic-evals' dataclass-based constructor
# without forcing the user to hand-edit YAML.
#
# The map is keyed by evaluator class name, then `wrong → canonical`.
# When the canonical key is already present we leave the alias alone
# so an explicit choice survives.
_EVALUATOR_KEYWORD_ALIASES: dict[str, dict[str, str]] = {
    "FieldEquals": {"expected": "value"},
    "FieldIn": {
        "expected": "choices",
        "values": "choices",
        "acceptable_values": "choices",
        "options": "choices",
    },
    "FieldContains": {"expected": "substring", "text": "substring"},
}


def canonicalize_dataset_yaml(content: str) -> tuple[str, bool]:
    """Normalise evaluator entries to match pydantic-evals' schema.

    Two rewrites apply:

    1. ``args:`` → ``arguments:`` on every evaluator entry — pydantic-
       evals' ``_DatasetModel`` requires the long-form key. Earlier
       versions of the dataset_creator prompt emitted ``args:``, which
       passes our structural check but fails when ``Dataset.from_file``
       runs.

    2. Per-evaluator keyword aliases inside ``arguments`` (e.g.
       ``FieldEquals: expected → value``, ``FieldIn: expected → choices``).
       The dataclass constructors reject unknown kwargs, and the LLM
       reliably picks intuitive-but-wrong names. The alias map is
       narrow and only fires when the canonical key isn't already set.

    Returns ``(canonical_yaml, changed)``. ``changed=False`` when the
    input parsed cleanly and required no rewrites — callers can use
    this to skip rewriting the file. When the YAML fails to parse we
    return ``(content, False)`` so the caller can still surface the
    underlying error in its normal path.

    sensitivity_tier: 1
    """
    try:
        parsed = yaml.safe_load(content)
    except yaml.YAMLError:
        return content, False
    if not isinstance(parsed, dict):
        return content, False

    cases = parsed.get("cases")
    if not isinstance(cases, list):
        return content, False

    changed = False
    for case in cases:
        if not isinstance(case, dict):
            continue
        evaluators = case.get("evaluators")
        if not isinstance(evaluators, list):
            continue
        for entry in evaluators:
            if not isinstance(entry, dict):
                continue
            if "args" in entry and "arguments" not in entry:
                entry["arguments"] = entry.pop("args")
                changed = True
            arguments = entry.get("arguments")
            ev_name = entry.get("name")
            aliases = _EVALUATOR_KEYWORD_ALIASES.get(
                ev_name if isinstance(ev_name, str) else "",
            )
            if not aliases or not isinstance(arguments, dict):
                continue
            for wrong, canonical in aliases.items():
                if wrong in arguments and canonical not in arguments:
                    arguments[canonical] = arguments.pop(wrong)
                    changed = True

    if not changed:
        return content, False
    return yaml.safe_dump(parsed, sort_keys=False, allow_unicode=True), True


# ---------------------------------------------------------------------------
# LLM agent
# ---------------------------------------------------------------------------


class DatasetValidatorAgent(SBAgent[str, DatasetValidationReport]):
    """Validate + propose improvements for an eval dataset YAML.

    Deps: the raw YAML string. The agent runs an LLM call for prose
    proposals; the deterministic structural pass is the authority for
    ``valid`` / ``errors`` and is enforced by the CLI handler after the
    LLM returns.

    sensitivity_tier: 1
    """

    agent_id = "dataset_validator"
    output_type = DatasetValidationReport
    tier = Tier.BACKGROUND
    system_prompt = DEFAULT_SYSTEM_PROMPT

    def build_prompt(self, deps: str) -> str:
        return (
            "Review this eval dataset YAML and suggest improvements:\n\n"
            f"```yaml\n{deps}\n```"
        )

    def validate(
        self,
        content: str,
        *,
        firewall_verdict: str = "allow",
    ) -> DatasetValidationReport:
        """Run the full structural + LLM pipeline.

        Always returns a report. ``valid`` and ``errors`` come from the
        deterministic structural pass; ``proposals`` come from the LLM
        (best-effort — falls back to an empty list when the model is
        unavailable). ``firewall_verdict`` is plumbed through from the
        caller so the report shows the firewall's decision alongside
        the structural verdict.

        sensitivity_tier: 1
        """
        structural = structural_check(content)
        proposals: list[str] = []
        if structural.valid:
            try:
                record = self.run(content)
            except Exception:  # noqa: BLE001
                record = None  # type: ignore[assignment]
            if record is not None and record.output is not None:
                proposals = list(record.output.proposals)
        return DatasetValidationReport(
            valid=structural.valid,
            errors=list(structural.errors),
            proposals=proposals,
            firewall_verdict=firewall_verdict,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Registry hook
# ---------------------------------------------------------------------------


def register_dataset_validator_agent() -> None:
    """Register the dataset validator agent. Idempotent.

    sensitivity_tier: 1
    """
    from src.agents.core.config_store import AgentConfig
    from src.agents.core.registry import (
        AgentDefinition,
        get_agent,
        register_agent,
    )

    if get_agent("dataset_validator") is not None:
        return

    default = AgentConfig(
        agent_id="dataset_validator",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        model_route="inherit",
        model_override=None,
        enabled_tools=(),
        enabled_skills=(),
        editable=False,
    )
    register_agent(AgentDefinition(
        agent_id="dataset_validator",
        name="Dataset Validator",
        description=(
            "Reviews user-uploaded eval datasets — checks structure "
            "against the shared evaluator catalog and proposes "
            "improvements. Called when a user uploads a dataset for "
            "one of their own agents."
        ),
        category="validator",
        parent_agent="brain",
        tier=Tier.BACKGROUND,
        max_sensitivity_tier=1,
        editable=False,
        default_config=default,
        available_tools=(),
        available_skills=(),
        output_schema="DatasetValidationReport",
        pattern="single",
        factory=DatasetValidatorAgent,
        tags=("validator", "builtin"),
    ))


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "DatasetValidatorAgent",
    "canonicalize_dataset_yaml",
    "register_dataset_validator_agent",
    "structural_check",
]
