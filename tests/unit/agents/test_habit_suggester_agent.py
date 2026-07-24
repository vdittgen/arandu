"""HabitSuggesterAgent behaviour — atomic-habits coupling.

sensitivity_tier: N/A
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from src.agents.core.audit import reset_default_chain_for_tests
from src.agents.core.output_types import HabitBatch, HabitDraft
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
from src.agents.habit_suggester import (
    HabitSuggesterAgent,
    register_habit_suggester_agent,
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


def _stub(batch: HabitBatch) -> MagicMock:
    fake = MagicMock()
    res = MagicMock()
    res.output = batch
    fake.run_sync.return_value = res
    return fake


def _draft(goal_id: str = "g1") -> HabitDraft:
    return HabitDraft(
        title="Skim résumés 10 min",
        cadence="daily",
        preferred_window="morning",
        goal_id=goal_id,
        why="to validate",
        reason="moves the goal",
    )


def test_suggest_no_goals_returns_empty() -> None:
    out = HabitSuggesterAgent().suggest(goals=[])
    assert out is not None
    assert out.habits == []


def test_suggest_drops_habits_for_unknown_goal(monkeypatch) -> None:
    agent = HabitSuggesterAgent()
    expected = HabitBatch(habits=[_draft("g1"), _draft("bogus")])
    monkeypatch.setattr(
        agent, "_get_pa_agent", lambda *, route: _stub(expected),
    )
    out = agent.suggest(goals=[{"id": "g1", "title": "Ship v1"}])
    assert out is not None
    ids = {h.goal_id for h in out.habits}
    assert ids == {"g1"}


def test_register_background_tier() -> None:
    register_habit_suggester_agent()
    d = get_agent("habit_suggester")
    assert d is not None
    assert d.tier == Tier.BACKGROUND
    assert d.output_schema == "HabitBatch"
