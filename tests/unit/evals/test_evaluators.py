"""Custom-evaluator behaviour tests.

We synthesise lightweight ``EvaluatorContext`` instances to drive each
evaluator and assert on the returned :class:`EvaluationReason`.

sensitivity_tier: N/A
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from evals.evaluators import (
    ConfidenceInRange,
    ContainsIds,
    DelegatedTo,
    EmotionalLabelStructural,
    FactSetMatches,
    FirewallAllowedMatches,
    LLMJudgeOnField,
    LLMJudgeOnReason,
    TierEquals,
    TriageDecisionAccuracy,
    _resolve_attr,
)


def _ctx(*, output: Any, expected: Any = None, inputs: Any = None) -> Any:
    """Build a minimal EvaluatorContext stand-in.

    The pydantic-evals ``EvaluatorContext`` has a long signature with
    fields the evaluators here don't touch; a SimpleNamespace with the
    two attributes the evaluators read is sufficient.
    """
    return SimpleNamespace(
        output=output,
        expected_output=expected,
        inputs=inputs,
        metadata=None,
    )


# ---------------------------------------------------------------------------
# TierEquals
# ---------------------------------------------------------------------------


def test_tier_equals_passes_on_match() -> None:
    out = SimpleNamespace(tier=2)
    r = TierEquals().evaluate(_ctx(output=out, expected={"tier": 2}))
    assert r.value is True


def test_tier_equals_fails_on_mismatch() -> None:
    out = SimpleNamespace(tier=1)
    r = TierEquals().evaluate(_ctx(output=out, expected={"tier": 3}))
    assert r.value is False
    assert "tier=1" in r.reason


def test_tier_equals_accepts_dict_output() -> None:
    r = TierEquals().evaluate(
        _ctx(output={"tier": 3}, expected={"tier": 3}),
    )
    assert r.value is True


# ---------------------------------------------------------------------------
# TriageDecisionAccuracy
# ---------------------------------------------------------------------------


def test_triage_accuracy_passes_on_match() -> None:
    output = SimpleNamespace(decisions=[
        SimpleNamespace(
            message_id="m1", keep=True,
            is_promo=False, is_automated=False, is_ack_only=False,
        ),
    ])
    expected = {"decisions": [{"message_id": "m1", "keep": True}]}
    r = TriageDecisionAccuracy().evaluate(
        _ctx(output=output, expected=expected),
    )
    assert r.value is True


def test_triage_accuracy_catches_flag_mismatch() -> None:
    output = SimpleNamespace(decisions=[
        SimpleNamespace(
            message_id="m1", keep=False,
            is_promo=False, is_automated=True, is_ack_only=False,
        ),
    ])
    expected = {"decisions": [{
        "message_id": "m1", "keep": False, "is_promo": True,
    }]}
    r = TriageDecisionAccuracy().evaluate(
        _ctx(output=output, expected=expected),
    )
    assert r.value is False
    assert "is_promo" in r.reason


def test_triage_accuracy_flags_missing_id() -> None:
    output = SimpleNamespace(decisions=[])
    expected = {"decisions": [{"message_id": "m1", "keep": True}]}
    r = TriageDecisionAccuracy().evaluate(
        _ctx(output=output, expected=expected),
    )
    assert r.value is False
    assert "missing" in r.reason


# ---------------------------------------------------------------------------
# EmotionalLabelStructural
# ---------------------------------------------------------------------------


def test_emotional_label_passes_within_tolerance() -> None:
    output = SimpleNamespace(
        primary_emotion="joy",
        intensity=0.55,
        domain="personal",
    )
    expected = {
        "primary_emotion": "joy",
        "intensity": 0.5,
        "domain": "personal",
    }
    r = EmotionalLabelStructural().evaluate(
        _ctx(output=output, expected=expected),
    )
    assert r.value is True


def test_emotional_label_rejects_emotion_mismatch() -> None:
    output = SimpleNamespace(
        primary_emotion="sadness",
        intensity=0.5,
        domain="personal",
    )
    expected = {"primary_emotion": "joy"}
    r = EmotionalLabelStructural().evaluate(
        _ctx(output=output, expected=expected),
    )
    assert r.value is False
    assert "emotion" in r.reason


def test_emotional_label_rejects_intensity_drift() -> None:
    output = SimpleNamespace(
        primary_emotion="joy", intensity=0.9, domain="personal",
    )
    expected = {"primary_emotion": "joy", "intensity": 0.2}
    r = EmotionalLabelStructural(intensity_tolerance=0.25).evaluate(
        _ctx(output=output, expected=expected),
    )
    assert r.value is False
    assert "intensity" in r.reason


# ---------------------------------------------------------------------------
# FirewallAllowedMatches
# ---------------------------------------------------------------------------


def test_firewall_passes_on_allowed_match() -> None:
    output = SimpleNamespace(allowed=True, category="safe")
    r = FirewallAllowedMatches().evaluate(
        _ctx(output=output, expected={"allowed": True}),
    )
    assert r.value is True


def test_firewall_passes_with_category_match() -> None:
    output = SimpleNamespace(allowed=False, category="jailbreak")
    r = FirewallAllowedMatches().evaluate(
        _ctx(
            output=output,
            expected={"allowed": False, "category": "jailbreak"},
        ),
    )
    assert r.value is True


def test_firewall_fails_on_category_mismatch() -> None:
    output = SimpleNamespace(allowed=False, category="role_override")
    r = FirewallAllowedMatches().evaluate(
        _ctx(
            output=output,
            expected={"allowed": False, "category": "injection"},
        ),
    )
    assert r.value is False
    assert "category" in r.reason


# ---------------------------------------------------------------------------
# ConfidenceInRange + LLMJudgeOnReason
# ---------------------------------------------------------------------------


def test_confidence_in_range_passes() -> None:
    output = SimpleNamespace(confidence=0.85)
    r = ConfidenceInRange(lo=0.5, hi=1.0).evaluate(_ctx(output=output))
    assert r.value is True


def test_confidence_in_range_fails_below_floor() -> None:
    output = SimpleNamespace(confidence=0.1)
    r = ConfidenceInRange(lo=0.5, hi=1.0).evaluate(_ctx(output=output))
    assert r.value is False


# ---------------------------------------------------------------------------
# _resolve_attr — dotted paths with list indexing
# ---------------------------------------------------------------------------


def test_resolve_attr_walks_into_list() -> None:
    obj = SimpleNamespace(
        notifications=[
            {"reason": "alice asked you", "importance": 9},
            {"reason": "bob is waiting", "importance": 7},
        ],
    )
    assert _resolve_attr(obj, "notifications.0.reason") == "alice asked you"
    assert _resolve_attr(obj, "notifications.1.importance") == 7
    assert _resolve_attr(obj, "notifications.99.reason") is None
    assert _resolve_attr(obj, "notifications.0.missing") is None


def test_resolve_attr_wildcard_flattens_list() -> None:
    obj = SimpleNamespace(
        replies=[
            SimpleNamespace(reason="alice asked you"),
            SimpleNamespace(reason="bob is waiting"),
        ],
    )
    assert _resolve_attr(obj, "replies.*.reason") == [
        "alice asked you",
        "bob is waiting",
    ]


def test_resolve_attr_wildcard_drops_missing_items() -> None:
    obj = SimpleNamespace(
        replies=[
            SimpleNamespace(reason="alice asked you"),
            SimpleNamespace(other="no reason here"),
        ],
    )
    assert _resolve_attr(obj, "replies.*.reason") == ["alice asked you"]


def test_resolve_attr_wildcard_on_non_list_returns_none() -> None:
    obj = SimpleNamespace(replies="not a list")
    assert _resolve_attr(obj, "replies.*.reason") is None


# ---------------------------------------------------------------------------
# ContainsIds — set-shaped batch checks
# ---------------------------------------------------------------------------


def test_contains_ids_passes_when_subset() -> None:
    output = SimpleNamespace(
        replies=[{"message_id": "m1"}, {"message_id": "m3"}],
    )
    r = ContainsIds(
        field="replies", id_key="message_id", expected_key="ids",
    ).evaluate(_ctx(output=output, expected={"ids": ["m1"]}))
    assert r.value is True


def test_contains_ids_fails_when_missing() -> None:
    output = SimpleNamespace(replies=[{"message_id": "m3"}])
    r = ContainsIds(
        field="replies", id_key="message_id", expected_key="ids",
    ).evaluate(_ctx(output=output, expected={"ids": ["m1", "m2"]}))
    assert r.value is False
    assert "m1" in r.reason
    assert "m2" in r.reason


def test_contains_ids_exact_rejects_extras() -> None:
    output = SimpleNamespace(
        replies=[{"message_id": "m1"}, {"message_id": "extra"}],
    )
    r = ContainsIds(
        field="replies", id_key="message_id",
        expected_key="ids", exact=True,
    ).evaluate(_ctx(output=output, expected={"ids": ["m1"]}))
    assert r.value is False
    assert "extra" in r.reason


def test_contains_ids_missing_field_fails() -> None:
    r = ContainsIds(
        field="replies", id_key="message_id", expected_key="ids",
    ).evaluate(_ctx(output=SimpleNamespace(), expected={"ids": ["m1"]}))
    assert r.value is False


# ---------------------------------------------------------------------------
# FactSetMatches — triple membership
# ---------------------------------------------------------------------------


def test_fact_set_matches_passes_subset() -> None:
    facts = [
        SimpleNamespace(
            category="preference", subject="self",
            predicate="favorite_food",
        ),
    ]
    output = SimpleNamespace(facts=facts)
    expected = {
        "facts": [
            {
                "category": "Preference", "subject": "Self",
                "predicate": "favorite_food",
            },
        ],
    }
    r = FactSetMatches().evaluate(_ctx(output=output, expected=expected))
    assert r.value is True


def test_fact_set_matches_fails_when_predicate_differs() -> None:
    facts = [
        SimpleNamespace(
            category="relationship", subject="alice", predicate="friend",
        ),
    ]
    expected = {
        "facts": [
            {
                "category": "relationship", "subject": "alice",
                "predicate": "sister",
            },
        ],
    }
    r = FactSetMatches().evaluate(
        _ctx(output=SimpleNamespace(facts=facts), expected=expected),
    )
    assert r.value is False
    assert "sister" in r.reason


def test_fact_set_matches_accepts_predicate_alternative() -> None:
    facts = [
        SimpleNamespace(
            category="health", subject="self",
            predicate="takes_medication",
        ),
    ]
    expected = {
        "facts": [
            {"category": "health", "subject": "self", "predicate": "medication"},
        ],
    }
    r = FactSetMatches(
        predicate_alternatives={
            "medication": ["takes_medication", "prescribed_medication"],
        },
    ).evaluate(_ctx(output=SimpleNamespace(facts=facts), expected=expected))
    assert r.value is True


def test_fact_set_matches_alternative_requires_same_category_subject() -> None:
    facts = [
        SimpleNamespace(
            category="opinion", subject="self",
            predicate="takes_medication",
        ),
    ]
    expected = {
        "facts": [
            {"category": "health", "subject": "self", "predicate": "medication"},
        ],
    }
    r = FactSetMatches(
        predicate_alternatives={"medication": ["takes_medication"]},
    ).evaluate(_ctx(output=SimpleNamespace(facts=facts), expected=expected))
    assert r.value is False
    assert "health/self/medication" in r.reason


def test_fact_set_matches_alternative_does_not_count_as_extra() -> None:
    facts = [
        SimpleNamespace(
            category="health", subject="self",
            predicate="takes_medication",
        ),
    ]
    expected = {
        "facts": [
            {"category": "health", "subject": "self", "predicate": "medication"},
        ],
    }
    r = FactSetMatches(
        extras_allowed=False,
        predicate_alternatives={"medication": ["takes_medication"]},
    ).evaluate(_ctx(output=SimpleNamespace(facts=facts), expected=expected))
    assert r.value is True


def test_fact_set_matches_no_extras_mode() -> None:
    facts = [
        SimpleNamespace(category="a", subject="b", predicate="c"),
        SimpleNamespace(category="x", subject="y", predicate="z"),
    ]
    expected = {
        "facts": [{"category": "a", "subject": "b", "predicate": "c"}],
    }
    r = FactSetMatches(extras_allowed=False).evaluate(
        _ctx(output=SimpleNamespace(facts=facts), expected=expected),
    )
    assert r.value is False


# ---------------------------------------------------------------------------
# LLM judge — both evaluators degrade gracefully + pass on score >= threshold
# ---------------------------------------------------------------------------


def test_llm_judge_on_reason_skipped_when_unavailable(monkeypatch) -> None:
    import evals.evaluators as ev_mod

    monkeypatch.setattr(
        "evals.judge.grade",
        lambda **_: None,
    )
    # Re-route attribute resolution if evaluators imported `grade`
    # eagerly (it doesn't — import is inside .evaluate).
    output = SimpleNamespace(reason="anything goes when judge is offline")
    r = LLMJudgeOnReason(rubric="anything", threshold=7).evaluate(
        _ctx(output=output, inputs="something"),
    )
    assert r.value is True
    assert "skipped" in r.reason
    del ev_mod  # silence unused-import warning


def test_llm_judge_on_reason_passes_above_threshold(monkeypatch) -> None:
    from evals.judge import JudgeVerdict

    def fake_grade(**_kw: Any) -> JudgeVerdict:
        return JudgeVerdict(score=9, passed=True, reason="great")

    monkeypatch.setattr("evals.judge.grade", fake_grade)
    output = SimpleNamespace(reason="any prose")
    r = LLMJudgeOnReason(rubric="r", threshold=7).evaluate(
        _ctx(output=output, inputs="x"),
    )
    assert r.value is True
    assert "9/10" in r.reason


def test_llm_judge_on_reason_fails_below_threshold(monkeypatch) -> None:
    from evals.judge import JudgeVerdict

    def fake_grade(**_kw: Any) -> JudgeVerdict:
        return JudgeVerdict(score=3, passed=False, reason="generic")

    monkeypatch.setattr("evals.judge.grade", fake_grade)
    output = SimpleNamespace(reason="generic blah")
    r = LLMJudgeOnReason(rubric="r", threshold=7).evaluate(
        _ctx(output=output, inputs="x"),
    )
    assert r.value is False
    assert "3/10" in r.reason


def test_llm_judge_on_field_walks_dotted_path(monkeypatch) -> None:
    from evals.judge import JudgeVerdict

    captured: dict[str, Any] = {}

    def fake_grade(**kw: Any) -> JudgeVerdict:
        captured.update(kw)
        return JudgeVerdict(score=8, passed=True, reason="ok")

    monkeypatch.setattr("evals.judge.grade", fake_grade)
    output = SimpleNamespace(
        notifications=[SimpleNamespace(reason="careful prose")],
    )
    r = LLMJudgeOnField(
        field="notifications.0.reason",
        rubric="grounded?", threshold=6,
    ).evaluate(_ctx(output=output, inputs={"foo": "bar"}))
    assert r.value is True
    assert captured["output_text"] == "careful prose"
    assert captured["threshold"] == 6


def test_llm_judge_on_field_missing_target_fails(monkeypatch) -> None:
    monkeypatch.setattr("evals.judge.grade", lambda **_: None)
    r = LLMJudgeOnField(
        field="notifications.0.reason", rubric="r",
    ).evaluate(_ctx(output=SimpleNamespace(notifications=[])))
    assert r.value is False
    assert "missing" in r.reason


def test_llm_judge_on_field_wildcard_renders_numbered_payload(
    monkeypatch,
) -> None:
    from evals.judge import JudgeVerdict
    captured: dict = {}

    def fake_grade(**kw) -> JudgeVerdict:
        captured.update(kw)
        return JudgeVerdict(score=9, passed=True, reason="ok")

    monkeypatch.setattr("evals.judge.grade", fake_grade)
    output = SimpleNamespace(
        replies=[
            SimpleNamespace(reason="Carol asked you to sign the contract"),
            SimpleNamespace(reason="Sam needs wire approval"),
        ],
    )
    r = LLMJudgeOnField(
        field="replies.*.reason",
        rubric="grounded?", threshold=6,
    ).evaluate(_ctx(output=output, inputs={"foo": "bar"}))
    assert r.value is True
    assert captured["output_text"] == (
        "[1] Carol asked you to sign the contract\n"
        "[2] Sam needs wire approval"
    )


def test_llm_judge_on_field_wildcard_empty_list_is_missing(monkeypatch) -> None:
    monkeypatch.setattr("evals.judge.grade", lambda **_: None)
    r = LLMJudgeOnField(
        field="replies.*.reason", rubric="r",
    ).evaluate(_ctx(output=SimpleNamespace(replies=[])))
    assert r.value is False
    assert "missing" in r.reason


# `MagicMock` is still used above
_ = MagicMock


# ---------------------------------------------------------------------------
# DelegatedTo
# ---------------------------------------------------------------------------


def test_delegated_to_passes_on_exact_match() -> None:
    ev = DelegatedTo(expected=["goal_extractor"])
    out = ev.evaluate(
        _ctx(output={"answer": "...", "delegated": ["goal_extractor"]}),
    )
    assert out.value is True


def test_delegated_to_fails_on_wrong_child() -> None:
    ev = DelegatedTo(expected=["task_proposer"])
    out = ev.evaluate(
        _ctx(output={"answer": "...", "delegated": ["daily_scheduler"]}),
    )
    assert out.value is False
    assert "daily_scheduler" in out.reason


def test_delegated_to_fails_when_nothing_was_called() -> None:
    """The native-structured-output failure mode: router answers direct."""
    ev = DelegatedTo(expected=["goal_extractor"])
    out = ev.evaluate(_ctx(output={"answer": "...", "delegated": []}))
    assert out.value is False
    assert "nobody" in out.reason


def test_delegated_to_rejects_over_delegation() -> None:
    """Right child + a speculative extra is still a miss."""
    ev = DelegatedTo(expected=["habit_suggester"])
    out = ev.evaluate(
        _ctx(
            output={
                "answer": "...",
                "delegated": ["habit_suggester", "goal_extractor"],
            },
        ),
    )
    assert out.value is False


def test_delegated_to_expects_no_delegation() -> None:
    ev = DelegatedTo(expected=[])
    out = ev.evaluate(_ctx(output={"answer": "...", "delegated": []}))
    assert out.value is True


def test_delegated_to_handles_non_orchestrator_output() -> None:
    ev = DelegatedTo(expected=["goal_extractor"])
    out = ev.evaluate(_ctx(output=SimpleNamespace(answer="hi")))
    assert out.value is False
    assert "delegated" in out.reason


def test_contains_ids_flattens_list_valued_key() -> None:
    """``source_message_ids`` is a list per record, not a scalar."""
    ev = ContainsIds(field="tasks", id_key="source_message_ids")
    out = ev.evaluate(
        _ctx(
            output={"tasks": [{"source_message_ids": ["m1", "m2"]}]},
            expected={"ids": ["m1"]},
        ),
    )
    assert out.value is True


def test_contains_ids_still_handles_scalar_key() -> None:
    # NB: not "items" as the field name — _resolve_attr tries hasattr()
    # first, which on a dict output returns the bound dict.items method.
    ev = ContainsIds(field="records", id_key="message_id")
    out = ev.evaluate(
        _ctx(
            output={"records": [{"message_id": "m1"}]},
            expected={"ids": ["m1"]},
        ),
    )
    assert out.value is True


def test_contains_ids_reports_missing_from_list_key() -> None:
    ev = ContainsIds(field="tasks", id_key="source_message_ids")
    out = ev.evaluate(
        _ctx(
            output={"tasks": [{"source_message_ids": ["m9"]}]},
            expected={"ids": ["m1"]},
        ),
    )
    assert out.value is False
    assert "m1" in out.reason
