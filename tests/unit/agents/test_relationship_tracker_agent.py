"""RelationshipTrackerAgent SBAgent behaviour.

sensitivity_tier: N/A
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from src.agents.core.audit import reset_default_chain_for_tests
from src.agents.core.output_types import RelationshipNudge
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
from src.agents.relationship_tracker import (
    RelationshipTrackerAgent,
    register_relationship_tracker_agent,
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


def _stub(nudge: RelationshipNudge) -> MagicMock:
    fake = MagicMock()
    res = MagicMock()
    res.output = nudge
    fake.run_sync.return_value = res
    return fake


def test_author_returns_nudge(monkeypatch) -> None:
    agent = RelationshipTrackerAgent()
    expected = RelationshipNudge(
        contact_name="Alice",
        nudge="It has been six weeks since you last spoke with Alice.",
        suggested_topic="the Garopaba trip you both planned",
    )
    monkeypatch.setattr(
        agent, "_get_pa_agent", lambda *, route: _stub(expected),
    )
    out = agent.author(
        "Contact: Alice. Last contact: 6 weeks ago. Shared topics: "
        "Garopaba trip.",
    )
    assert out == expected


def test_author_empty_returns_none() -> None:
    assert RelationshipTrackerAgent().author("") is None


def test_author_failure_returns_none(monkeypatch) -> None:
    agent = RelationshipTrackerAgent()
    def boom(*, route):
        raise RuntimeError("model down")
    monkeypatch.setattr(agent, "_get_pa_agent", boom)
    assert agent.author("any context") is None


def test_register_builtin_indirect_author() -> None:
    register_relationship_tracker_agent()
    d = get_agent("relationship_tracker")
    assert d is not None
    assert d.tier == Tier.BACKGROUND
    assert d.output_schema == "RelationshipNudge"
    assert "builtin" in d.tags
    assert "indirect" in d.tags
