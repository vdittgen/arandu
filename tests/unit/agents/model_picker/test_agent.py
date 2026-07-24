"""ModelPickerAgent behaviour.

Covers the deterministic surface of :class:`ModelPickerAgent`: the
empty-catalog short-circuit, refusal pass-through, catalog-membership
validation, and registration. The LLM call itself is stubbed.

sensitivity_tier: 1
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from src.agents.core.audit import reset_default_chain_for_tests
from src.agents.core.output_types import (
    ModelOption,
    ModelRecommendation,
)
from src.agents.core.registry import (
    get_agent,
    reset_registry_for_tests,
)
from src.agents.core.scheduler import (
    SchedulerConfig,
    reset_default_scheduler_for_tests,
)
from src.agents.firewall.egress_firewall import (
    EgressPolicy,
    reset_egress_firewall_for_tests,
)
from src.agents.firewall.injection_firewall import (
    reset_injection_firewall_for_tests,
)
from src.agents.model_picker import (
    ModelPickerAgent,
    ModelPickerInput,
    register_model_picker_agent,
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


def _stub(outputs: list[ModelRecommendation]) -> MagicMock:
    """Build a pydantic-ai stub that pops one output per ``run_sync`` call."""
    fake = MagicMock()
    sequence = list(outputs)

    def _run(_prompt):
        res = MagicMock()
        res.output = sequence.pop(0) if sequence else None
        return res

    fake.run_sync.side_effect = _run
    return fake


def _input(**overrides) -> ModelPickerInput:
    base = {
        "name": "Summarizer",
        "description": "Summarize a paragraph in one sentence.",
        "system_prompt": (
            "You are a summarizer. Return a single sentence answer."
        ),
        "max_sensitivity_tier": 1,
        "available_remote_models": (
            "Qwen/Qwen2.5-7B-Instruct",
            "deepseek-ai/DeepSeek-V3.1",
        ),
        "available_local_models": ("llama3.1:8b",),
    }
    base.update(overrides)
    return ModelPickerInput(**base)


# ---------------------------------------------------------------------------
# Empty-catalog short-circuit (no LLM call)
# ---------------------------------------------------------------------------


def test_empty_catalog_refuses_without_llm_call(monkeypatch) -> None:
    agent = ModelPickerAgent()
    sentinel: dict[str, int] = {"llm_calls": 0}

    def _build(*, route):
        sentinel["llm_calls"] += 1
        raise AssertionError("LLM must not be called with empty catalog")

    monkeypatch.setattr(agent, "_get_pa_agent", _build)
    result = agent.recommend(
        _input(available_remote_models=(), available_local_models=()),
    )
    assert result.can_recommend is False
    assert "no models" in (result.reason_if_not or "").lower()
    assert result.improvement_hints
    assert sentinel["llm_calls"] == 0


# ---------------------------------------------------------------------------
# Refusal pass-through
# ---------------------------------------------------------------------------


def test_llm_refusal_passes_through(monkeypatch) -> None:
    agent = ModelPickerAgent()
    refusal = ModelRecommendation(
        can_recommend=False,
        reason_if_not="description is one word",
        improvement_hints=[
            "Replace the one-word description with a sentence.",
            "State the output format in the system prompt.",
        ],
    )
    monkeypatch.setattr(
        agent, "_get_pa_agent", lambda *, route: _stub([refusal]),
    )
    result = agent.recommend(_input(description="x"))
    assert result.can_recommend is False
    assert "one word" in (result.reason_if_not or "")
    assert len(result.improvement_hints) == 2


# ---------------------------------------------------------------------------
# Happy path with both picks in the catalog
# ---------------------------------------------------------------------------


def test_picks_in_catalog_returned_unchanged(monkeypatch) -> None:
    agent = ModelPickerAgent()
    good = ModelRecommendation(
        can_recommend=True,
        purpose_summary="Summarizes paragraphs.",
        best_overall=ModelOption(
            model_id="deepseek-ai/DeepSeek-V3.1",
            route="remote",
            rationale="strong instruction-following.",
        ),
        cost_effective=ModelOption(
            model_id="Qwen/Qwen2.5-7B-Instruct",
            route="remote",
            rationale="cheap; sufficient for single-sentence prose.",
        ),
        confidence=0.7,
    )
    monkeypatch.setattr(
        agent, "_get_pa_agent", lambda *, route: _stub([good]),
    )
    result = agent.recommend(_input())
    assert result.can_recommend is True
    assert result.best_overall is not None
    assert result.best_overall.model_id == "deepseek-ai/DeepSeek-V3.1"
    assert result.cost_effective is not None
    assert result.cost_effective.model_id == "Qwen/Qwen2.5-7B-Instruct"


# ---------------------------------------------------------------------------
# Hallucinated model id is downgraded to refusal
# ---------------------------------------------------------------------------


def test_hallucinated_id_downgrades_to_refusal(monkeypatch) -> None:
    agent = ModelPickerAgent()
    hallucinated = ModelRecommendation(
        can_recommend=True,
        purpose_summary="Summarizes paragraphs.",
        best_overall=ModelOption(
            model_id="anthropic/claude-fantasy-9000",  # not in catalog
            route="remote",
            rationale="invented.",
        ),
        cost_effective=ModelOption(
            model_id="Qwen/Qwen2.5-7B-Instruct",
            route="remote",
            rationale="real id.",
        ),
        confidence=0.5,
    )
    monkeypatch.setattr(
        agent, "_get_pa_agent", lambda *, route: _stub([hallucinated]),
    )
    result = agent.recommend(_input())
    assert result.can_recommend is False
    assert "live catalog" in (result.reason_if_not or "")
    assert result.improvement_hints


# ---------------------------------------------------------------------------
# Route mismatch is downgraded too
# ---------------------------------------------------------------------------


def test_route_mismatch_downgrades_to_refusal(monkeypatch) -> None:
    agent = ModelPickerAgent()
    # Pick is in the local catalog but the LLM claims it's remote.
    mismatched = ModelRecommendation(
        can_recommend=True,
        purpose_summary="Summarizes paragraphs.",
        best_overall=ModelOption(
            model_id="llama3.1:8b",
            route="remote",  # wrong — llama3.1:8b is local
            rationale="confused route.",
        ),
        cost_effective=ModelOption(
            model_id="Qwen/Qwen2.5-7B-Instruct",
            route="remote",
            rationale="ok.",
        ),
        confidence=0.4,
    )
    monkeypatch.setattr(
        agent, "_get_pa_agent", lambda *, route: _stub([mismatched]),
    )
    result = agent.recommend(_input())
    assert result.can_recommend is False
    assert "route" in (result.reason_if_not or "").lower()


# ---------------------------------------------------------------------------
# LLM error returns a graceful refusal (no exception escapes)
# ---------------------------------------------------------------------------


def test_llm_failure_returns_graceful_refusal(monkeypatch) -> None:
    agent = ModelPickerAgent()
    fake = MagicMock()
    fake.run_sync.side_effect = RuntimeError("endpoint unreachable")
    monkeypatch.setattr(agent, "_get_pa_agent", lambda *, route: fake)

    result = agent.recommend(_input())
    assert result.can_recommend is False
    assert "endpoint unreachable" in (result.reason_if_not or "")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_registers_as_non_editable_system_agent() -> None:
    register_model_picker_agent()
    definition = get_agent("model_picker")
    assert definition is not None
    assert definition.editable is False
    assert definition.output_schema == "ModelRecommendation"
    assert "locked" in definition.tags
    assert "builtin" in definition.tags
