"""SchemaDiscoveryAgent SBAgent behaviour.

sensitivity_tier: N/A
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from src.agents.core.audit import reset_default_chain_for_tests
from src.agents.core.output_types import (
    FieldMappingDraft,
    SchemaDiscoveryDraft,
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
from src.agents.schema_discovery import (
    SchemaDiscoveryAgent,
    SchemaDiscoveryDeps,
    register_schema_discovery_agent,
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


def _stub(draft: SchemaDiscoveryDraft) -> MagicMock:
    fake = MagicMock()
    res = MagicMock()
    res.output = draft
    fake.run_sync.return_value = res
    return fake


def test_discover_returns_draft(monkeypatch) -> None:
    agent = SchemaDiscoveryAgent()
    expected = SchemaDiscoveryDraft(
        target_table="ext_strava_runs",
        is_new_table=True,
        domain="health",
        fields=[
            FieldMappingDraft(
                source_name="id", target_column="id",
                target_type="TEXT", sensitivity_tier=1,
            ),
            FieldMappingDraft(
                source_name="hr_avg", target_column="hr_avg",
                target_type="DOUBLE", sensitivity_tier=3,
            ),
        ],
        dedup_key=["id"],
    )
    monkeypatch.setattr(
        agent, "_get_pa_agent", lambda *, route: _stub(expected),
    )
    out = agent.discover(
        tool_name="strava-list-activities",
        sample_records=[{"id": "1", "hr_avg": 152.0}],
    )
    assert out == expected


def test_discover_empty_records_returns_none() -> None:
    assert SchemaDiscoveryAgent().discover(
        tool_name="x", sample_records=[],
    ) is None


def test_discover_failure_returns_none(monkeypatch) -> None:
    agent = SchemaDiscoveryAgent()
    def boom(*, route):
        raise RuntimeError("model down")
    monkeypatch.setattr(agent, "_get_pa_agent", boom)
    assert agent.discover(
        tool_name="x", sample_records=[{"a": 1}],
    ) is None


def test_build_prompt_includes_tool_and_sample() -> None:
    agent = SchemaDiscoveryAgent()
    deps = SchemaDiscoveryDeps(
        tool_name="strava",
        sample_records=({"id": "1", "x": 2},),
        known_tables=("raw_workouts",),
    )
    prompt = agent.build_prompt(deps)
    assert "strava" in prompt
    assert "raw_workouts" in prompt
    assert '"id"' in prompt


def test_register_indirect_ingestion() -> None:
    register_schema_discovery_agent()
    d = get_agent("schema_discovery")
    assert d is not None
    assert d.tier == Tier.BACKGROUND
    assert d.output_schema == "SchemaDiscoveryDraft"
    assert "indirect" in d.tags
    assert "ingestion" in d.tags
