"""ActionableEventsAgent SBAgent behaviour.

sensitivity_tier: N/A
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from src.agents.actionable_events import (
    ActionableEventsAgent,
    ActionableEventsDeps,
    register_actionable_events_agent,
)
from src.agents.core.audit import reset_default_chain_for_tests
from src.agents.core.output_types import (
    ActionableEventBatch,
    ActionableEventDraft,
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


def _stub(batch: ActionableEventBatch) -> MagicMock:
    fake = MagicMock()
    res = MagicMock()
    res.output = batch
    fake.run_sync.return_value = res
    return fake


def _draft(event_id: str, importance: int = 8) -> ActionableEventDraft:
    return ActionableEventDraft(
        event_id=event_id,
        action_needed="Prepare quarterly report.",
        importance=importance,
    )


def test_detect_returns_batch(monkeypatch) -> None:
    agent = ActionableEventsAgent()
    expected = ActionableEventBatch(events=[
        _draft("e1"), _draft("e2", importance=10),
    ])
    monkeypatch.setattr(
        agent, "_get_pa_agent", lambda *, route: _stub(expected),
    )
    out = agent.detect(events=[
        {"id": "e1", "title": "Q-report review"},
        {"id": "e2", "title": "Sam's birthday"},
    ])
    assert out == expected


def test_detect_empty_returns_empty_batch() -> None:
    out = ActionableEventsAgent().detect(events=[])
    assert out is not None
    assert out.events == []


def test_detect_failure_returns_none(monkeypatch) -> None:
    agent = ActionableEventsAgent()
    def boom(*, route):
        raise RuntimeError("model down")
    monkeypatch.setattr(agent, "_get_pa_agent", boom)
    assert agent.detect(events=[{"id": "e1"}]) is None


def test_build_prompt_renders_events_json() -> None:
    agent = ActionableEventsAgent()
    deps = ActionableEventsDeps(events=(
        {"id": "e1", "title": "Standup"},
    ))
    prompt = agent.build_prompt(deps)
    assert "e1" in prompt
    assert "Standup" in prompt


def test_register_proactive_tier_and_brain_parent() -> None:
    register_actionable_events_agent()
    d = get_agent("actionable_events")
    assert d is not None
    assert d.parent_agent == "brain"
    assert d.tier == Tier.PROACTIVE
    assert d.output_schema == "ActionableEventBatch"


def test_importance_validation_rejects_out_of_range() -> None:
    # ActionableEventDraft.importance is 1-10.
    with pytest.raises(Exception):  # noqa: B017
        ActionableEventDraft(
            event_id="e1",
            action_needed="x",
            importance=11,
        )
