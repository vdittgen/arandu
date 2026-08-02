"""Egress firewall routing tests for Arandu.

OSS routes every call to local Ollama regardless of policy / tier /
lane / complexity. These tests assert that invariant plus the
keyword-floor classification and the safe-list short-circuit.

sensitivity_tier: N/A
"""

from __future__ import annotations

import pytest
from src.agents.core.audit import reset_default_chain_for_tests
from src.agents.firewall.egress_firewall import (
    EgressFirewall,
    EgressPolicy,
    keyword_tier_floor,
)


@pytest.fixture(autouse=True)
def _isolate_audit(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "ARANDU_AUDIT_PATH", str(tmp_path / "audit.jsonl"),
    )
    reset_default_chain_for_tests()


def _fw(*, local: bool = False) -> EgressFirewall:
    return EgressFirewall(
        policy=EgressPolicy(
            routing="local-only" if local else "remote-default",
            local_inference_for_sensitive=local,
        ),
    )


# ---------------------------------------------------------------------------
# Keyword floor
# ---------------------------------------------------------------------------


def test_keyword_floor_detects_tier3() -> None:
    assert keyword_tier_floor("I think I have depression") == 3
    assert keyword_tier_floor("Routing number is 123") == 3


def test_keyword_floor_detects_tier2() -> None:
    assert keyword_tier_floor("Meeting with my sister at 5") == 2


def test_keyword_floor_defaults_to_tier1() -> None:
    assert keyword_tier_floor("Summarize today's weather") == 1


# ---------------------------------------------------------------------------
# OSS routing — always local
# ---------------------------------------------------------------------------


def test_classify_tier1_routes_local() -> None:
    d = _fw().classify("Plain prompt", agent_max_tier=1)
    assert d.route == "local"
    assert d.max_tier == 1


def test_classify_tier2_routes_local() -> None:
    d = _fw().classify(
        "Schedule meeting with my sister", agent_max_tier=2,
    )
    assert d.route == "local"
    assert d.max_tier == 2


def test_classify_tier3_routes_local() -> None:
    d = _fw().classify(
        "I have depression and need help", agent_max_tier=2,
    )
    assert d.route == "local"
    assert d.max_tier == 3


def test_local_only_mode_still_routes_local() -> None:
    """OSS ignores the remote-default vs local-only distinction —
    every call is local either way."""
    d = _fw(local=True).classify("Plain prompt", agent_max_tier=1)
    assert d.route == "local"


def test_local_only_skips_llm_classifier_for_safelist(monkeypatch) -> None:
    """Firewall agents must not recurse into the LLM-driven classifier.

    The classifier is itself an LLM caller; if the egress firewall asked
    it to classify the classifier's own prompt, the call would loop.
    """
    monkeypatch.delenv(
        "ARANDU_FIREWALL_DISABLE_LLM_TIER", raising=False,
    )
    calls: list[str] = []

    def fake(text: str) -> int:
        calls.append(text)
        return 3

    monkeypatch.setattr(
        "src.agents.firewall.egress_firewall._llm_classify_tier", fake,
    )
    fw = _fw(local=True)
    fw.classify(
        "Routine query",
        calling_agent_id="firewall.injection.scan",
        agent_max_tier=1,
    )
    assert calls == [], "safe-list caller invoked the LLM classifier"


# ---------------------------------------------------------------------------
# Explicit tier floor
# ---------------------------------------------------------------------------


def test_explicit_tier_floor_honoured() -> None:
    d = _fw().classify(
        "Innocuous text", agent_max_tier=1, explicit_tier=3,
    )
    assert d.max_tier == 3
    assert d.route == "local"
