"""WeeklyDigestAgent SBAgent behaviour.

sensitivity_tier: N/A
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from src.agents.core.audit import reset_default_chain_for_tests
from src.agents.core.output_types import DigestSummary
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
from src.agents.weekly_digest import (
    WeeklyDigestAgent,
    register_weekly_digest_agent,
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


def _stub(digest: DigestSummary) -> MagicMock:
    fake = MagicMock()
    res = MagicMock()
    res.output = digest
    fake.run_sync.return_value = res
    return fake


def test_author_returns_digest(monkeypatch) -> None:
    agent = WeeklyDigestAgent()
    expected = DigestSummary(
        highlight="Three client decisions landed.",
        communication="- Sam confirmed the contract\n- Alice booked Tue",
        schedule="- Board meeting Wed 10am",
        notes="- Jot on the Garopaba renovation",
    )
    monkeypatch.setattr(
        agent, "_get_pa_agent", lambda *, route: _stub(expected),
    )
    out = agent.author(
        "Messages: 42; Events: 7; Notes: 3 ... (data summary)",
    )
    assert out == expected


def test_author_empty_summary_returns_none() -> None:
    assert WeeklyDigestAgent().author("") is None


def test_author_failure_returns_none(monkeypatch) -> None:
    agent = WeeklyDigestAgent()
    def boom(*, route):
        raise RuntimeError("model down")
    monkeypatch.setattr(agent, "_get_pa_agent", boom)
    assert agent.author("summary") is None


def test_register_builtin_indirect_author() -> None:
    register_weekly_digest_agent()
    d = get_agent("weekly_digest")
    assert d is not None
    assert d.tier == Tier.BACKGROUND
    assert d.output_schema == "DigestSummary"
    assert "builtin" in d.tags
    assert "indirect" in d.tags
