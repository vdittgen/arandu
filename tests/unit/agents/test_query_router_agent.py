"""QueryRouterAgent SBAgent behaviour.

sensitivity_tier: N/A
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from src.agents.core.audit import reset_default_chain_for_tests
from src.agents.core.output_types import (
    DuckDBQuerySpec,
    RetrievalPlan,
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
from src.agents.query_router import (
    QueryRouterAgent,
    register_query_router_agent,
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


def _stub(plan: RetrievalPlan) -> MagicMock:
    fake = MagicMock()
    res = MagicMock()
    res.output = plan
    fake.run_sync.return_value = res
    return fake


def test_plan_returns_retrieval_plan(monkeypatch) -> None:
    agent = QueryRouterAgent()
    expected = RetrievalPlan(
        duckdb_queries=[
            DuckDBQuerySpec(
                table="raw_calendar_events",
                columns=["title", "start_time"],
                where="start_time >= current_date",
                limit=5,
            ),
        ],
        chromadb_collections=["personal"],
        use_graph=False,
        reasoning="Today's schedule lookup",
    )
    monkeypatch.setattr(
        agent, "_get_pa_agent", lambda *, route: _stub(expected),
    )
    plan = agent.plan("What's on my schedule today?")
    assert plan == expected


def test_plan_empty_question_returns_none() -> None:
    assert QueryRouterAgent().plan("") is None


def test_plan_failure_returns_none(monkeypatch) -> None:
    agent = QueryRouterAgent()
    def boom(*, route):
        raise RuntimeError("model down")
    monkeypatch.setattr(agent, "_get_pa_agent", boom)
    assert agent.plan("anything") is None


def test_register_interactive_tier() -> None:
    register_query_router_agent()
    d = get_agent("query_router")
    assert d is not None
    assert d.tier == Tier.INTERACTIVE
    assert d.output_schema == "RetrievalPlan"
    assert "indirect" in d.tags


def test_duckdb_query_spec_validates_columns() -> None:
    # columns defaults to []; limit defaults to 10.
    spec = DuckDBQuerySpec(table="raw_messages")
    assert spec.columns == []
    assert spec.limit == 10
