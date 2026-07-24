"""TopicExtractorAgent SBAgent behaviour.

sensitivity_tier: N/A
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from src.agents.core.audit import reset_default_chain_for_tests
from src.agents.core.output_types import Topic, TopicBatch
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
from src.agents.topic_extractor import (
    TopicExtractorAgent,
    TopicExtractorDeps,
    register_topic_extractor_agent,
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


def _stub(batch: TopicBatch) -> MagicMock:
    fake = MagicMock()
    res = MagicMock()
    res.output = batch
    fake.run_sync.return_value = res
    return fake


def _topic(name: str = "construction", importance: int = 8) -> Topic:
    return Topic(
        topic=name,
        description="Tracking the contractor's progress on the kitchen.",
        importance=importance,
        status="active",
    )


def test_extract_returns_batch(monkeypatch) -> None:
    agent = TopicExtractorAgent()
    expected = TopicBatch(topics=[_topic(), _topic("vacation", 5)])
    monkeypatch.setattr(
        agent, "_get_pa_agent", lambda *, route: _stub(expected),
    )
    out = agent.extract(
        contact_name="Sam",
        messages_block="2026-05-01 Sam: kitchen done by Friday\n...",
    )
    assert out == expected


def test_extract_empty_block_returns_empty_batch() -> None:
    out = TopicExtractorAgent().extract(
        contact_name="Sam", messages_block="",
    )
    assert out is not None
    assert out.topics == []


def test_extract_caps_topics_at_five(monkeypatch) -> None:
    agent = TopicExtractorAgent()
    big = TopicBatch(topics=[_topic(f"t{i}") for i in range(8)])
    monkeypatch.setattr(
        agent, "_get_pa_agent", lambda *, route: _stub(big),
    )
    out = agent.extract(
        contact_name="Sam", messages_block="msg",
    )
    assert out is not None
    assert len(out.topics) == 5


def test_build_prompt_renders_contact_and_messages() -> None:
    agent = TopicExtractorAgent()
    deps = TopicExtractorDeps(
        contact_name="Alice", messages_block="hello\nworld",
    )
    prompt = agent.build_prompt(deps)
    assert "Alice" in prompt
    assert "hello" in prompt


def test_build_prompt_truncates_long_block() -> None:
    agent = TopicExtractorAgent()
    long_block = "x" * 20000
    deps = TopicExtractorDeps(
        contact_name="Alice", messages_block=long_block,
    )
    prompt = agent.build_prompt(deps)
    assert "[truncated]" in prompt


def test_register_background_tier() -> None:
    register_topic_extractor_agent()
    d = get_agent("topic_extractor")
    assert d is not None
    assert d.tier == Tier.BACKGROUND
    assert d.output_schema == "TopicBatch"
    assert "indirect" in d.tags
