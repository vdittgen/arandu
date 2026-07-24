"""GoalExtractorAgent SBAgent behaviour.

sensitivity_tier: N/A
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from src.agents.core.audit import reset_default_chain_for_tests
from src.agents.core.output_types import GoalBatch, GoalDraft
from src.agents.core.registry import (
    get_agent,
    reset_registry_for_tests,
)
from src.agents.core.scheduler import (
    SchedulerConfig,
    Tier,
    reset_default_scheduler_for_tests,
)
from src.agents.firewall.egress_firewall import (
    EgressPolicy,
    reset_egress_firewall_for_tests,
)
from src.agents.firewall.injection_firewall import (
    reset_injection_firewall_for_tests,
)
from src.agents.goal_extractor import (
    GoalExtractorAgent,
    register_goal_extractor_agent,
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


def _stub(batch: GoalBatch) -> MagicMock:
    fake = MagicMock()
    res = MagicMock()
    res.output = batch
    fake.run_sync.return_value = res
    return fake


def _draft(title: str = "Ship v1") -> GoalDraft:
    return GoalDraft(
        title=title,
        description="...",
        category="work",
        horizon="medium",
        importance=8,
        why="to validate the OS",
        source_kind="message",
        source_ref="m1",
    )


def test_extract_returns_batch(monkeypatch) -> None:
    agent = GoalExtractorAgent()
    expected = GoalBatch(goals=[_draft()])
    monkeypatch.setattr(
        agent, "_get_pa_agent", lambda *, route: _stub(expected),
    )
    out = agent.extract(messages=[{"id": "m1", "content": "we should ship v1"}])
    assert out == expected


def test_extract_caps_at_eight(monkeypatch) -> None:
    agent = GoalExtractorAgent()
    expected = GoalBatch(goals=[_draft(f"goal-{i}") for i in range(12)])
    monkeypatch.setattr(
        agent, "_get_pa_agent", lambda *, route: _stub(expected),
    )
    out = agent.extract(messages=[{"id": "m1", "content": "x"}])
    assert out is not None
    assert len(out.goals) == 8


def test_extract_no_evidence_returns_empty() -> None:
    out = GoalExtractorAgent().extract()
    assert out is not None
    assert out.goals == []


def test_register_proactive_tier() -> None:
    register_goal_extractor_agent()
    d = get_agent("goal_extractor")
    assert d is not None
    assert d.tier == Tier.PROACTIVE
    assert d.output_schema == "GoalBatch"
