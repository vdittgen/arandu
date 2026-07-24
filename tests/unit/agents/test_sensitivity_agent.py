"""SensitivityAgent SBAgent behaviour.

The underlying ``pydantic_ai.Agent`` is mocked so the test exercises:
- prompt routing through the scheduler + firewalls
- classify_tier fail-safe behaviour
- registry contract (parent_agent=brain, editable=False)

sensitivity_tier: N/A
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from src.agents.core.audit import reset_default_chain_for_tests
from src.agents.core.output_types import SensitivityVerdict
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
from src.agents.sensitivity import (
    SensitivityAgent,
    register_sensitivity_agent,
)
from src.agents.sensitivity.agent import FAIL_SAFE_TIER


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


def _stub_pa_agent(tier: int, reason: str = "stub") -> MagicMock:
    fake = MagicMock()
    result = MagicMock()
    result.output = SensitivityVerdict(tier=tier, reason=reason)
    fake.run_sync.return_value = result
    return fake


def test_classify_tier_returns_int(monkeypatch) -> None:
    agent = SensitivityAgent()
    monkeypatch.setattr(
        agent, "_get_pa_agent",
        lambda *, route: _stub_pa_agent(2, "people names"),
    )
    assert agent.classify_tier("Meeting with Alice at 5pm") == 2


def test_classify_tier_empty_text_returns_one() -> None:
    agent = SensitivityAgent()
    assert agent.classify_tier("") == 1


def test_classify_tier_fail_safe_on_error(monkeypatch) -> None:
    agent = SensitivityAgent()

    def boom(*, route):
        raise RuntimeError("model down")

    monkeypatch.setattr(agent, "_get_pa_agent", boom)
    assert agent.classify_tier("some text") == FAIL_SAFE_TIER


def test_run_returns_verdict(monkeypatch) -> None:
    agent = SensitivityAgent()
    monkeypatch.setattr(
        agent, "_get_pa_agent",
        lambda *, route: _stub_pa_agent(3, "health"),
    )
    record = agent.run("My medication is...")
    assert record.output is not None
    assert record.output.tier == 3
    assert record.error is None


def test_register_marks_child_of_brain() -> None:
    register_sensitivity_agent()
    definition = get_agent("sensitivity")
    assert definition is not None
    assert definition.parent_agent == "brain"
    assert definition.editable is False
    assert definition.output_schema == "SensitivityVerdict"
    assert definition.pattern == "single"


def test_register_idempotent() -> None:
    register_sensitivity_agent()
    register_sensitivity_agent()
    assert get_agent("sensitivity") is not None
