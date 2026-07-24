"""LabelerAgent SBAgent behaviour.

sensitivity_tier: N/A
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from src.agents.core.audit import reset_default_chain_for_tests
from src.agents.core.output_types import EmotionalLabel
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
from src.agents.labeler import (
    LabelerAgent,
    register_labeler_agent,
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


def _stub_pa_agent(label: EmotionalLabel) -> MagicMock:
    fake = MagicMock()
    result = MagicMock()
    result.output = label
    fake.run_sync.return_value = result
    return fake


def test_label_returns_emotional_label(monkeypatch) -> None:
    agent = LabelerAgent()
    expected = EmotionalLabel(
        primary_emotion="joy",
        intensity=0.7,
        feelings=["excited"],
        desires=["share news"],
        actors=["Alice"],
        environment="text message",
        domain="personal",
    )
    monkeypatch.setattr(
        agent, "_get_pa_agent",
        lambda *, route: _stub_pa_agent(expected),
    )
    out = agent.label("Great news!")
    assert out == expected


def test_label_empty_text_returns_none() -> None:
    assert LabelerAgent().label("") is None


def test_label_failure_returns_none(monkeypatch) -> None:
    agent = LabelerAgent()

    def boom(*, route):
        raise RuntimeError("model down")

    monkeypatch.setattr(agent, "_get_pa_agent", boom)
    assert agent.label("some text") is None


def test_register_marks_child_of_brain() -> None:
    register_labeler_agent()
    definition = get_agent("labeler")
    assert definition is not None
    assert definition.parent_agent == "brain"
    assert definition.editable is False
    assert definition.output_schema == "EmotionalLabel"
