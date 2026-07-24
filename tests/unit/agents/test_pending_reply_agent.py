"""PendingReplyAgent SBAgent behaviour.

sensitivity_tier: N/A
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from src.agents.core.audit import reset_default_chain_for_tests
from src.agents.core.output_types import (
    PendingReplyBatch,
    PendingReplyDraft,
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
from src.agents.pending_reply import (
    PendingReplyAgent,
    PendingReplyDeps,
    register_pending_reply_agent,
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


def _stub(batch: PendingReplyBatch) -> MagicMock:
    fake = MagicMock()
    res = MagicMock()
    res.output = batch
    fake.run_sync.return_value = res
    return fake


def _draft(message_id: str, *, needs_reply: bool = True) -> PendingReplyDraft:
    return PendingReplyDraft(
        message_id=message_id,
        needs_reply=needs_reply,
        importance=7,
        domain="work",
        reason="direct question awaiting answer",
    )


def test_detect_returns_batch(monkeypatch) -> None:
    agent = PendingReplyAgent()
    expected = PendingReplyBatch(replies=[_draft("m1"), _draft("m2")])
    monkeypatch.setattr(
        agent, "_get_pa_agent", lambda *, route: _stub(expected),
    )
    out = agent.detect(messages=[{"id": "m1"}, {"id": "m2"}])
    assert out == expected


def test_detect_filters_no_reply_items(monkeypatch) -> None:
    agent = PendingReplyAgent()
    mixed = PendingReplyBatch(replies=[
        _draft("m1", needs_reply=True),
        _draft("m2", needs_reply=False),
        _draft("m3", needs_reply=True),
    ])
    monkeypatch.setattr(
        agent, "_get_pa_agent", lambda *, route: _stub(mixed),
    )
    out = agent.detect(messages=[{"id": "m1"}])
    assert out is not None
    assert {r.message_id for r in out.replies} == {"m1", "m3"}


def test_detect_empty_returns_empty_batch() -> None:
    out = PendingReplyAgent().detect(messages=[])
    assert out is not None
    assert out.replies == []


def test_build_prompt_string_passthrough() -> None:
    agent = PendingReplyAgent()
    assert agent.build_prompt("raw prompt") == "raw prompt"


def test_build_prompt_typed_deps_renders_messages() -> None:
    agent = PendingReplyAgent()
    deps = PendingReplyDeps(
        messages=({"id": "m1", "content": "Are you free?"},),
        topics={},
    )
    prompt = agent.build_prompt(deps)
    assert "m1" in prompt
    assert "Are you free?" in prompt


def test_register_proactive_tier() -> None:
    register_pending_reply_agent()
    d = get_agent("pending_reply")
    assert d is not None
    assert d.tier == Tier.PROACTIVE
    assert d.output_schema == "PendingReplyBatch"
    assert d.max_sensitivity_tier == 2
