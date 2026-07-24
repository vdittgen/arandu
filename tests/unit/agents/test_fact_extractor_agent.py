"""FactExtractorAgent SBAgent behaviour.

sensitivity_tier: N/A
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from src.agents.core.audit import reset_default_chain_for_tests
from src.agents.core.output_types import (
    LearnedFactBatch,
    LearnedFactDraft,
)
from src.agents.core.registry import (
    get_agent,
    reset_registry_for_tests,
)
from src.agents.core.scheduler import (
    SchedulerConfig,
    reset_default_scheduler_for_tests,
)
from src.agents.fact_extractor import (
    FactExtractorAgent,
    register_fact_extractor_agent,
)
from src.agents.firewall.egress_firewall import (
    EgressPolicy,
    reset_egress_firewall_for_tests,
)
from src.agents.firewall.injection_firewall import (
    reset_injection_firewall_for_tests,
)


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


def _stub(batch: LearnedFactBatch) -> MagicMock:
    fake = MagicMock()
    res = MagicMock()
    res.output = batch
    fake.run_sync.return_value = res
    return fake


def test_extract_returns_batch(monkeypatch) -> None:
    agent = FactExtractorAgent()
    expected = LearnedFactBatch(facts=[
        LearnedFactDraft(
            category="preference",
            subject="self",
            predicate="favorite_food",
            content="User's favorite food is sushi.",
            sensitivity_tier=1,
        ),
        LearnedFactDraft(
            category="relationship",
            subject="Alice",
            predicate="sister",
            content="Alice is the user's sister.",
            sensitivity_tier=2,
        ),
    ])
    monkeypatch.setattr(
        agent, "_get_pa_agent", lambda *, route: _stub(expected),
    )
    batch = agent.extract("user said i love sushi and my sister Alice...")
    assert batch == expected


def test_extract_empty_returns_empty_batch() -> None:
    out = FactExtractorAgent().extract("")
    assert out is not None
    assert out.facts == []


def test_extract_failure_returns_none(monkeypatch) -> None:
    agent = FactExtractorAgent()
    def boom(*, route):
        raise RuntimeError("model down")
    monkeypatch.setattr(agent, "_get_pa_agent", boom)
    assert agent.extract("some conversation") is None


def test_register_marks_child_of_brain() -> None:
    register_fact_extractor_agent()
    d = get_agent("fact_extractor")
    assert d is not None
    assert d.parent_agent == "brain"
    assert d.editable is False
    assert d.output_schema == "LearnedFactBatch"
