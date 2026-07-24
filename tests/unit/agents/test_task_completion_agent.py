"""TaskCompletionAgent behaviour.

sensitivity_tier: N/A
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from src.agents.core.audit import reset_default_chain_for_tests
from src.agents.core.output_types import (
    TaskCompletionBatch,
    TaskCompletionDraft,
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
from src.agents.task_completion import (
    TaskCompletionAgent,
    register_task_completion_agent,
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


def _stub(batch: TaskCompletionBatch) -> MagicMock:
    fake = MagicMock()
    res = MagicMock()
    res.output = batch
    fake.run_sync.return_value = res
    return fake


def test_detect_no_inputs_returns_empty() -> None:
    out = TaskCompletionAgent().detect(open_tasks=[], evidence=[])
    assert out is not None
    assert out.completions == []


def test_detect_returns_batch(monkeypatch) -> None:
    agent = TaskCompletionAgent()
    expected = TaskCompletionBatch(completions=[
        TaskCompletionDraft(
            task_id="t1",
            evidence_message_id="m1",
            evidence_summary="confirmed",
            confidence=0.9,
        ),
    ])
    monkeypatch.setattr(
        agent, "_get_pa_agent", lambda *, route: _stub(expected),
    )
    out = agent.detect(
        open_tasks=[{"id": "t1", "title": "send deck"}],
        evidence=[{"id": "m1", "content": "got the deck"}],
    )
    assert out == expected


def test_register_proactive() -> None:
    register_task_completion_agent()
    d = get_agent("task_completion")
    assert d is not None
    assert d.tier == Tier.PROACTIVE
    assert d.output_schema == "TaskCompletionBatch"


def test_confidence_validation_rejects_out_of_range() -> None:
    with pytest.raises(Exception):  # noqa: B017
        TaskCompletionDraft(
            task_id="t",
            evidence_message_id="m",
            evidence_summary="x",
            confidence=1.5,
        )
