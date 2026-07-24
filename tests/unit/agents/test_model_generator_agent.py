"""ModelGeneratorAgent SBAgent behaviour.

sensitivity_tier: N/A
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from src.agents.core.audit import reset_default_chain_for_tests
from src.agents.core.output_types import GeneratedSQLModel
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
from src.agents.model_generator import (
    ModelGeneratorAgent,
    ModelGeneratorDeps,
    register_model_generator_agent,
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


def _stub(model: GeneratedSQLModel) -> MagicMock:
    fake = MagicMock()
    res = MagicMock()
    res.output = model
    fake.run_sync.return_value = res
    return fake


def test_generate_returns_sql_model(monkeypatch) -> None:
    agent = ModelGeneratorAgent()
    expected = GeneratedSQLModel(
        name="ext_stg_strava_runs",
        layer="staging",
        sql="MODEL (name ext_stg_strava_runs, kind FULL);\nSELECT ...",
        sensitivity_summary="Tier 3 — heart rate is health data.",
    )
    monkeypatch.setattr(
        agent, "_get_pa_agent", lambda *, route: _stub(expected),
    )
    out = agent.generate(
        schema={"table": "ext_strava_runs"},
        layer="staging",
        connector_id="strava",
    )
    assert out == expected


def test_generate_empty_schema_returns_none() -> None:
    assert ModelGeneratorAgent().generate(schema={}) is None


def test_generate_failure_returns_none(monkeypatch) -> None:
    agent = ModelGeneratorAgent()
    def boom(*, route):
        raise RuntimeError("model down")
    monkeypatch.setattr(agent, "_get_pa_agent", boom)
    assert agent.generate(schema={"x": 1}) is None


def test_build_prompt_includes_connector_and_layer() -> None:
    agent = ModelGeneratorAgent()
    deps = ModelGeneratorDeps(
        schema={"foo": "bar"},
        layer="intermediate",
        connector_id="my-conn",
    )
    prompt = agent.build_prompt(deps)
    assert "my-conn" in prompt
    assert "intermediate" in prompt


def test_register_indirect_ingestion() -> None:
    register_model_generator_agent()
    d = get_agent("model_generator")
    assert d is not None
    assert d.tier == Tier.BACKGROUND
    assert d.output_schema == "GeneratedSQLModel"
    assert "indirect" in d.tags
    assert "ingestion" in d.tags


def test_generated_sql_model_layer_literal_validated() -> None:
    # GeneratedSQLModel.layer is a Literal — invalid value rejected.
    with pytest.raises(Exception):  # noqa: B017
        GeneratedSQLModel(
            name="x", layer="snapshot", sql="select 1",  # type: ignore[arg-type]
        )
