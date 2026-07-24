"""DailySchedulerAgent behaviour.

sensitivity_tier: N/A
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from src.agents.core.audit import reset_default_chain_for_tests
from src.agents.core.output_types import DailySchedule, ScheduleSlot
from src.agents.core.registry import (
    get_agent,
    reset_registry_for_tests,
)
from src.agents.core.scheduler import (
    SchedulerConfig,
    Tier,
    reset_default_scheduler_for_tests,
)
from src.agents.daily_scheduler import (
    DailySchedulerAgent,
    register_daily_scheduler_agent,
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


def _stub(schedule: DailySchedule) -> MagicMock:
    fake = MagicMock()
    res = MagicMock()
    res.output = schedule
    fake.run_sync.return_value = res
    return fake


def test_plan_returns_schedule(monkeypatch) -> None:
    agent = DailySchedulerAgent()
    expected = DailySchedule(
        schedule_date="2026-05-21",
        slots=[
            ScheduleSlot(
                kind="event",
                ref_id="e1",
                title="Standup",
                start="2026-05-21T09:00:00",
                end="2026-05-21T09:30:00",
                why="fixed",
            ),
            ScheduleSlot(
                kind="task",
                ref_id="t1",
                title="Write spec",
                start="2026-05-21T10:00:00",
                end="2026-05-21T11:00:00",
                category="work",
                goal_id="g1",
                why="due tomorrow",
            ),
        ],
        unscheduled_overflow=["t99"],
        rationale="Deep work block before meetings",
        category_balance={"work": 90, "personal": 0, "life": 0},
    )
    monkeypatch.setattr(
        agent, "_get_pa_agent", lambda *, route: _stub(expected),
    )
    out = agent.plan(
        schedule_date="2026-05-21",
        events=[{"id": "e1", "title": "Standup"}],
        tasks=[{"id": "t1", "title": "Write spec"}],
        habits=[],
        goals=[{"id": "g1", "title": "Ship v1"}],
    )
    assert out == expected


def test_register_background_tier() -> None:
    register_daily_scheduler_agent()
    d = get_agent("daily_scheduler")
    assert d is not None
    assert d.tier == Tier.BACKGROUND
    assert d.output_schema == "DailySchedule"
