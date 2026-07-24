"""TriageAgent SBAgent behaviour.

sensitivity_tier: N/A
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from src.agents.core.audit import reset_default_chain_for_tests
from src.agents.core.output_types import TriageBatch, TriageDecision
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
from src.agents.triage import (
    TriageAgent,
    TriageMessage,
    register_triage_agent,
)
from src.agents.triage.agent import TriageDeps


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


def _stub_pa_agent(batch: TriageBatch) -> MagicMock:
    fake = MagicMock()
    result = MagicMock()
    result.output = batch
    fake.run_sync.return_value = result
    return fake


def test_triage_returns_batch(monkeypatch) -> None:
    agent = TriageAgent()
    expected = TriageBatch(decisions=[
        TriageDecision(message_id="m1", keep=True, reason="direct question"),
        TriageDecision(
            message_id="m2", keep=False, reason="promo", is_promo=True,
        ),
    ])
    monkeypatch.setattr(
        agent, "_get_pa_agent",
        lambda *, route: _stub_pa_agent(expected),
    )
    batch = agent.triage([
        TriageMessage(message_id="m1", content="Are you free Tuesday?"),
        TriageMessage(message_id="m2", content="EXCLUSIVE OFFER!!!"),
    ])
    assert batch == expected


def test_triage_empty_returns_empty_batch() -> None:
    agent = TriageAgent()
    out = agent.triage([])
    assert out is not None
    assert out.decisions == []


def test_build_prompt_handles_typed_deps() -> None:
    agent = TriageAgent()
    deps = TriageDeps(messages=(
        TriageMessage(
            message_id="m1", content="Hello world", sender_name="Bob",
        ),
    ))
    text = agent.build_prompt(deps)
    assert "id=m1" in text
    assert "Hello world" in text


def test_build_prompt_handles_raw_string() -> None:
    agent = TriageAgent()
    text = agent.build_prompt("Plain message")
    assert "id=msg_1" in text
    assert "Plain message" in text


def test_build_prompt_truncates_long_content() -> None:
    agent = TriageAgent()
    long_content = "x" * 500
    deps = TriageDeps(messages=(
        TriageMessage(message_id="m1", content=long_content),
    ))
    text = agent.build_prompt(deps)
    # Body is truncated to 240 chars + ellipsis.
    assert "…" in text
    assert text.count("x") < 500


def test_register_marks_child_of_brain() -> None:
    register_triage_agent()
    definition = get_agent("triage")
    assert definition is not None
    assert definition.parent_agent == "brain"
    assert definition.editable is False
    assert definition.output_schema == "TriageBatch"
    assert "batch" in definition.tags
