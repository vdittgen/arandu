"""DatasetCreatorAgent behaviour.

Covers the deterministic surface of :class:`DatasetCreatorAgent` and
its merge helpers. The LLM call itself is stubbed so the tests run
without a pydantic-ai model — exactly the path the runner takes when
the remote endpoint is offline.

sensitivity_tier: 1
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import yaml
from src.agents.core.audit import reset_default_chain_for_tests
from src.agents.core.output_types import DatasetSuggestion
from src.agents.core.registry import (
    get_agent,
    reset_registry_for_tests,
)
from src.agents.core.scheduler import (
    SchedulerConfig,
    reset_default_scheduler_for_tests,
)
from src.agents.dataset_creator import (
    DatasetCreatorAgent,
    DatasetCreatorInput,
    merge_user_dataset_yaml,
    register_dataset_creator_agent,
)
from src.agents.firewall.egress_firewall import (
    EgressPolicy,
    reset_egress_firewall_for_tests,
)
from src.agents.firewall.injection_firewall import (
    reset_injection_firewall_for_tests,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "ARANDU_AUDIT_PATH", str(tmp_path / "audit.jsonl"),
    )
    reset_default_chain_for_tests()
    reset_injection_firewall_for_tests()
    reset_egress_firewall_for_tests(
        policy=EgressPolicy(
            routing="balanced",
        ),
    )
    reset_default_scheduler_for_tests(SchedulerConfig())
    reset_registry_for_tests()


def _stub(outputs: list[DatasetSuggestion]) -> MagicMock:
    """Build a pydantic-ai stub that pops one output per ``run_sync`` call."""
    fake = MagicMock()
    sequence = list(outputs)

    def _run(_prompt):
        res = MagicMock()
        res.output = sequence.pop(0) if sequence else None
        return res

    fake.run_sync.side_effect = _run
    return fake


def _good_yaml(case_names: tuple[str, ...] = ("case-a", "case-b")) -> str:
    cases = [
        {
            "name": n,
            "inputs": "hello",
            "evaluators": [{"name": "FieldNotEmpty", "field": "answer"}],
        }
        for n in case_names
    ]
    return yaml.safe_dump({"cases": cases}, sort_keys=False)


def _input(**overrides) -> DatasetCreatorInput:
    base = {
        "name": "Summarizer",
        "description": "Summarize a paragraph in one sentence.",
        "system_prompt": (
            "You are a summarizer. Return a single sentence answer."
        ),
        "max_sensitivity_tier": 1,
        "agent_id": "user.summarizer",
        "output_schema": None,
    }
    base.update(overrides)
    return DatasetCreatorInput(**base)


# ---------------------------------------------------------------------------
# Refusal path
# ---------------------------------------------------------------------------


def test_refusal_short_circuits_without_validation(monkeypatch) -> None:
    """When the LLM refuses, suggest() does NOT touch structural_check."""
    agent = DatasetCreatorAgent()
    refusal = DatasetSuggestion(
        can_create=False,
        reason_if_not="description is one word",
        improvement_hints=[
            "Replace the one-word description with a sentence.",
            "Specify the output format in the system prompt.",
        ],
    )
    monkeypatch.setattr(
        agent, "_get_pa_agent", lambda *, route: _stub([refusal]),
    )
    sentinel: dict[str, int] = {"calls": 0}

    def _spy(_content):
        sentinel["calls"] += 1
        raise AssertionError("structural_check must not run on refusal")

    monkeypatch.setattr(
        "src.agents.dataset_validator.agent.structural_check", _spy,
    )

    result = agent.suggest(_input(description="x"))
    assert result.can_create is False
    assert "one word" in (result.reason_if_not or "")
    assert sentinel["calls"] == 0


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_valid_first_response_returns_unchanged(monkeypatch) -> None:
    agent = DatasetCreatorAgent()
    proposal = DatasetSuggestion(
        can_create=True,
        purpose_summary="Summarize paragraphs.",
        output_shape="prose",
        eval_strategy="llm_judge",
        dataset_yaml=_good_yaml(),
        case_count=2,
        confidence=0.7,
    )
    monkeypatch.setattr(
        agent, "_get_pa_agent", lambda *, route: _stub([proposal]),
    )
    result = agent.suggest(_input())
    assert result.can_create is True
    assert result.case_count == 2
    assert "case-a" in result.dataset_yaml


# ---------------------------------------------------------------------------
# Retry on invalid YAML
# ---------------------------------------------------------------------------


def test_invalid_yaml_retries_and_succeeds(monkeypatch) -> None:
    agent = DatasetCreatorAgent()
    bad = DatasetSuggestion(
        can_create=True,
        dataset_yaml="cases: []\n",
        case_count=0,
        confidence=0.1,
    )
    good = DatasetSuggestion(
        can_create=True,
        purpose_summary="Summarize.",
        dataset_yaml=_good_yaml(),
        case_count=2,
        confidence=0.6,
    )
    stub = _stub([bad, good])
    monkeypatch.setattr(agent, "_get_pa_agent", lambda *, route: stub)

    captured: list[str] = []
    original_build = agent.build_prompt

    def _capture(deps: DatasetCreatorInput) -> str:
        body = original_build(deps)
        captured.append(body)
        return body

    monkeypatch.setattr(agent, "build_prompt", _capture)

    result = agent.suggest(_input())
    assert result.can_create is True
    assert result.case_count == 2
    # Second prompt body must carry the structural errors fed back in.
    assert any("prior_errors" in c for c in captured[1:])


def test_invalid_yaml_retry_exhausts_to_refusal(monkeypatch) -> None:
    agent = DatasetCreatorAgent()
    bad1 = DatasetSuggestion(
        can_create=True, dataset_yaml="cases: []\n", case_count=0,
    )
    bad2 = DatasetSuggestion(
        can_create=True, dataset_yaml="cases: []\n", case_count=0,
    )
    stub = _stub([bad1, bad2])
    monkeypatch.setattr(agent, "_get_pa_agent", lambda *, route: stub)
    result = agent.suggest(_input())
    assert result.can_create is False
    assert result.reason_if_not is not None
    assert "structural validation" in result.reason_if_not


# ---------------------------------------------------------------------------
# Unknown evaluator triggers retry
# ---------------------------------------------------------------------------


def test_unknown_evaluator_triggers_retry(monkeypatch) -> None:
    agent = DatasetCreatorAgent()
    bogus_yaml = yaml.safe_dump({
        "cases": [
            {
                "name": "a",
                "inputs": "x",
                "evaluators": [{"name": "FooBarEvaluator"}],
            },
        ],
    }, sort_keys=False)
    bad = DatasetSuggestion(
        can_create=True, dataset_yaml=bogus_yaml, case_count=1,
    )
    good = DatasetSuggestion(
        can_create=True, dataset_yaml=_good_yaml(), case_count=2,
    )
    stub = _stub([bad, good])
    monkeypatch.setattr(agent, "_get_pa_agent", lambda *, route: stub)
    result = agent.suggest(_input())
    assert result.can_create is True
    assert "FooBarEvaluator" not in result.dataset_yaml


# ---------------------------------------------------------------------------
# Append mode
# ---------------------------------------------------------------------------


def test_append_merges_without_duplicating_names(monkeypatch) -> None:
    agent = DatasetCreatorAgent()
    # The model proposes one colliding and one fresh case.
    proposed = yaml.safe_dump({
        "cases": [
            {
                "name": "case-a",  # colliding
                "inputs": "dup",
                "evaluators": [{"name": "FieldNotEmpty", "field": "answer"}],
            },
            {
                "name": "case-c",  # net-new
                "inputs": "new",
                "evaluators": [{"name": "FieldNotEmpty", "field": "answer"}],
            },
        ],
    }, sort_keys=False)
    proposal = DatasetSuggestion(
        can_create=True, dataset_yaml=proposed, case_count=2,
    )
    monkeypatch.setattr(
        agent, "_get_pa_agent",
        lambda *, route: _stub([proposal]),
    )
    existing = _good_yaml(("case-a", "case-b"))

    result = agent.suggest(
        _input(existing_case_names=("case-a", "case-b")),
        existing_yaml=existing,
    )
    assert result.can_create is True
    merged = yaml.safe_load(result.dataset_yaml)
    names = [c["name"] for c in merged["cases"]]
    assert names == ["case-a", "case-b", "case-c"]
    assert result.case_count == 3


# ---------------------------------------------------------------------------
# Merge helper unit test
# ---------------------------------------------------------------------------


def test_merge_helper_dedupes_by_name() -> None:
    existing = _good_yaml(("alpha", "beta"))
    new = yaml.safe_dump({
        "cases": [
            {"name": "alpha", "inputs": "dup"},
            {"name": "gamma", "inputs": "new"},
        ],
    }, sort_keys=False)
    merged_yaml, count = merge_user_dataset_yaml(existing, new)
    merged = yaml.safe_load(merged_yaml)
    names = [c["name"] for c in merged["cases"]]
    assert names == ["alpha", "beta", "gamma"]
    assert count == 3


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_registers_as_non_editable_system_agent() -> None:
    register_dataset_creator_agent()
    definition = get_agent("dataset_creator")
    assert definition is not None
    assert definition.editable is False
    assert definition.output_schema == "DatasetSuggestion"
    assert "builtin" in definition.tags
