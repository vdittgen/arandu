"""ContactContextAgent SBAgent behaviour.

sensitivity_tier: N/A
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from src.agents.contact_context import (
    ContactContextAgent,
    ContactContextDeps,
    register_contact_context_agent,
)
from src.agents.core.audit import reset_default_chain_for_tests
from src.agents.core.output_types import (
    ContactContextBatch,
    ContactContextDraft,
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


def _stub(batch: ContactContextBatch) -> MagicMock:
    fake = MagicMock()
    res = MagicMock()
    res.output = batch
    fake.run_sync.return_value = res
    return fake


def _draft(
    contact_id: str = "c1", priority: int = 2,
) -> ContactContextDraft:
    return ContactContextDraft(
        contact_id=contact_id,
        active_context="Discussing project deadline.",
        context_domains=["work"],
        context_priority=priority,
    )


def test_summarize_returns_batch(monkeypatch) -> None:
    agent = ContactContextAgent()
    expected = ContactContextBatch(contexts=[_draft("c1"), _draft("c2", 3)])
    monkeypatch.setattr(
        agent, "_get_pa_agent", lambda *, route: _stub(expected),
    )
    out = agent.summarize(
        contacts=[
            {"id": "c1", "name": "Alice", "messages_7d": 12},
            {"id": "c2", "name": "Bob", "messages_7d": 3},
        ],
    )
    assert out == expected


def test_summarize_empty_returns_empty_batch() -> None:
    out = ContactContextAgent().summarize(contacts=[])
    assert out is not None
    assert out.contexts == []


def test_summarize_failure_returns_none(monkeypatch) -> None:
    agent = ContactContextAgent()
    def boom(*, route):
        raise RuntimeError("model down")
    monkeypatch.setattr(agent, "_get_pa_agent", boom)
    assert agent.summarize(contacts=[{"id": "c1"}]) is None


def test_build_prompt_renders_contacts() -> None:
    agent = ContactContextAgent()
    deps = ContactContextDeps(
        contacts=({"id": "c1", "name": "Alice"},),
        topics={"c1": {"topics": ["work"]}},
    )
    prompt = agent.build_prompt(deps)
    assert "Alice" in prompt
    assert "c1" in prompt


def test_register_proactive_tier_and_brain_parent() -> None:
    register_contact_context_agent()
    d = get_agent("contact_context")
    assert d is not None
    assert d.parent_agent == "brain"
    assert d.tier == Tier.PROACTIVE
    assert d.output_schema == "ContactContextBatch"


def test_priority_validation_rejects_out_of_range() -> None:
    # ContactContextDraft.context_priority must be 0-3.
    with pytest.raises(Exception):  # noqa: B017
        ContactContextDraft(
            contact_id="c1",
            active_context="x",
            context_priority=5,
        )
