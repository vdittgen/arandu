"""MessageEvaluatorAgent SBAgent behaviour.

sensitivity_tier: N/A
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from src.agents.core.audit import reset_default_chain_for_tests
from src.agents.core.output_types import (
    MessageNotificationBatch,
    MessageNotificationDraft,
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
from src.agents.message_eval import (
    MessageEvalDeps,
    MessageEvaluatorAgent,
    register_message_evaluator_agent,
)
from src.agents.message_eval.agent import MAX_NOTIFICATIONS


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


def _stub(batch: MessageNotificationBatch) -> MagicMock:
    fake = MagicMock()
    res = MagicMock()
    res.output = batch
    fake.run_sync.return_value = res
    return fake


def _draft(message_id: str, importance: int = 8) -> MessageNotificationDraft:
    return MessageNotificationDraft(
        message_id=message_id,
        notification_type="topic_action",
        importance=importance,
        domain="work",
        summary="Sam asks for status",
        related_to="construction-project",
    )


def test_evaluate_returns_batch(monkeypatch) -> None:
    agent = MessageEvaluatorAgent()
    expected = MessageNotificationBatch(
        notifications=[_draft("m1"), _draft("m2", importance=9)],
    )
    monkeypatch.setattr(
        agent, "_get_pa_agent", lambda *, route: _stub(expected),
    )
    out = agent.evaluate(
        messages=[{"id": "m1", "content": "hi"}],
        topics={"sam": {"topics": ["construction-project"]}},
    )
    assert out == expected


def test_evaluate_empty_returns_empty_batch() -> None:
    out = MessageEvaluatorAgent().evaluate(messages=[])
    assert out is not None
    assert out.notifications == []


def test_evaluate_caps_at_max(monkeypatch) -> None:
    agent = MessageEvaluatorAgent()
    drafts = [_draft(f"m{i}", importance=10) for i in range(6)]
    monkeypatch.setattr(
        agent, "_get_pa_agent",
        lambda *, route: _stub(
            MessageNotificationBatch(notifications=drafts),
        ),
    )
    out = agent.evaluate(messages=[{"id": "m0"}])
    assert out is not None
    assert len(out.notifications) == MAX_NOTIFICATIONS


def test_build_prompt_contains_json_blocks() -> None:
    agent = MessageEvaluatorAgent()
    deps = MessageEvalDeps(
        messages=({"id": "m1", "content": "hello"},),
        topics={"alice": {"topics": ["health"]}},
        today_events=({"id": "e1"},),
        existing_pending_ids=("p1",),
    )
    prompt = agent.build_prompt(deps)
    assert "alice" in prompt
    assert "m1" in prompt
    # Plain JSON of empty/missing pieces would still parse.
    pieces = prompt.split("\n")
    json_segments = [p for p in pieces if p.startswith(("[", "{"))]
    for seg in json_segments:
        json.loads(seg)


def test_register_proactive_tier() -> None:
    register_message_evaluator_agent()
    d = get_agent("message_evaluator")
    assert d is not None
    assert d.tier == Tier.PROACTIVE
    assert d.output_schema == "MessageNotificationBatch"
