"""InsightAgent SBAgent behaviour.

sensitivity_tier: N/A
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from src.agents.core.audit import reset_default_chain_for_tests
from src.agents.core.output_types import InsightDraft
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
from src.agents.insight import (
    InsightAgent,
    register_insight_agent,
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


def _stub(draft: InsightDraft) -> MagicMock:
    fake = MagicMock()
    res = MagicMock()
    res.output = draft
    fake.run_sync.return_value = res
    return fake


def test_author_returns_draft(monkeypatch) -> None:
    agent = InsightAgent()
    expected = InsightDraft(
        title="Three pending replies",
        content=(
            "You have three messages awaiting a response: "
            "Alice (Tue), Bob (Wed), Carol (today)."
        ),
        suggested_followup="Who am I most behind with?",
    )
    monkeypatch.setattr(
        agent, "_get_pa_agent", lambda *, route: _stub(expected),
    )
    draft = agent.author(
        "Surface pending replies based on the last 7 days of messages.",
    )
    assert draft == expected


def test_author_empty_returns_none() -> None:
    assert InsightAgent().author("") is None


def test_author_failure_returns_none(monkeypatch) -> None:
    agent = InsightAgent()
    def boom(*, route):
        raise RuntimeError("model down")
    monkeypatch.setattr(agent, "_get_pa_agent", boom)
    assert agent.author("any prompt") is None


def test_register_proactive_tier_and_brain_parent() -> None:
    register_insight_agent()
    d = get_agent("insight")
    assert d is not None
    assert d.parent_agent == "brain"
    assert d.tier == Tier.PROACTIVE
    assert d.output_schema == "InsightDraft"
