"""TaskProposerAgent SBAgent behaviour.

sensitivity_tier: N/A
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from src.agents.core.audit import reset_default_chain_for_tests
from src.agents.core.output_types import (
    TaskProposalBatch,
    TaskProposalDraft,
)
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
from src.agents.task_proposer import (
    TaskProposerAgent,
    TaskProposerDeps,
    register_task_proposer_agent,
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


def _stub(batch: TaskProposalBatch) -> MagicMock:
    fake = MagicMock()
    res = MagicMock()
    res.output = batch
    fake.run_sync.return_value = res
    return fake


def _draft(title: str = "Send Maria the deck") -> TaskProposalDraft:
    return TaskProposalDraft(
        title=title,
        category="work",
        importance=7,
        source_message_ids=["m1"],
        reason="explicit ask",
    )


def test_propose_returns_batch(monkeypatch) -> None:
    agent = TaskProposerAgent()
    expected = TaskProposalBatch(tasks=[_draft()])
    monkeypatch.setattr(
        agent, "_get_pa_agent", lambda *, route: _stub(expected),
    )
    out = agent.propose(messages=[{"id": "m1", "content": "send the deck"}])
    assert out == expected


def test_propose_empty_returns_empty_batch() -> None:
    out = TaskProposerAgent().propose(messages=[])
    assert out is not None
    assert out.tasks == []


def test_propose_caps_at_six(monkeypatch) -> None:
    agent = TaskProposerAgent()
    expected = TaskProposalBatch(tasks=[
        _draft(f"task-{i}") for i in range(10)
    ])
    monkeypatch.setattr(
        agent, "_get_pa_agent", lambda *, route: _stub(expected),
    )
    out = agent.propose(messages=[{"id": "m1", "content": "x"}])
    assert out is not None
    assert len(out.tasks) == 6


def test_build_prompt_includes_topics_and_goals() -> None:
    agent = TaskProposerAgent()
    deps = TaskProposerDeps(
        messages=({"id": "m1", "content": "send"},),
        topics=({"topic": "hiring"},),
        goals=({"title": "Staff the clinic"},),
    )
    prompt = agent.build_prompt(deps)
    assert "hiring" in prompt
    assert "Staff the clinic" in prompt
    assert "send" in prompt


def test_register_proactive_tier_and_brain_parent() -> None:
    register_task_proposer_agent()
    d = get_agent("task_proposer")
    assert d is not None
    assert d.parent_agent == "brain"
    assert d.tier == Tier.PROACTIVE
    assert d.output_schema == "TaskProposalBatch"
