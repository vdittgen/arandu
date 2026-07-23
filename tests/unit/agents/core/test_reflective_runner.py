"""Unit tests for the reflective runner in :mod:`src.agents.core.agent_base`.

We stub pydantic-ai's ``Agent.iter`` with a hand-written async context
manager that yields a deterministic sequence of nodes so we can assert
exactly when the runner fires reviews, injects STOP_REQUEST messages,
and emits stream events.

sensitivity_tier: 1
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Fake pydantic-ai shape: just enough for the runner to walk nodes.
# ---------------------------------------------------------------------------
# The runner imports these names lazily; we have to import the same
# concrete classes the runner uses so isinstance checks work.
from pydantic_ai.messages import ModelRequest, UserPromptPart  # noqa: E402
from pydantic_graph import End  # noqa: E402
from src.agents.core.agent_base import _run_pa_with_reflection_async
from src.agents.core.output_types import ReflectionVerdict
from src.agents.core.task_budget import TaskBudget, TaskClass


@dataclass
class _FakeModelRequestNode:
    """Bare-bones stand-in matching the runner's attribute access."""

    request: ModelRequest


@dataclass
class _FakeCallToolsNode:
    """Stand-in for a tool-execution node — no ``request`` attribute."""


@dataclass
class _FakeAgentRun:
    """Iterates a pre-baked node sequence; ``result`` is the last value.

    Mirrors pydantic-ai's ``AgentRun`` surface: ``next_node``, async
    ``next(node)``, and a ``result`` attribute populated when we step
    past the ``End``.
    """

    nodes: list[Any]
    cursor: int = 0
    result: Any = "FINAL_OUTPUT"
    # Records of nodes that were *executed* (passed through ``next``).
    executed: list[Any] = field(default_factory=list)
    # Tick wall-clock forward by this many seconds per ``next()`` call.
    seconds_per_step: float = 0.0
    _now: float = 0.0

    @property
    def next_node(self) -> Any:
        return self.nodes[self.cursor]

    async def next(self, node: Any) -> Any:
        self.executed.append(node)
        self.cursor += 1
        self._now += self.seconds_per_step
        return self.nodes[self.cursor] if self.cursor < len(self.nodes) else (
            End(data=self.result)
        )


class _FakeAgent:
    def __init__(self, run: _FakeAgentRun) -> None:
        self._run = run

    def iter(self, prompt: str) -> _FakeIterCM:
        return _FakeIterCM(self._run)


class _FakeIterCM:
    def __init__(self, run: _FakeAgentRun) -> None:
        self._run = run

    async def __aenter__(self) -> _FakeAgentRun:
        return self._run

    async def __aexit__(self, *_exc: Any) -> None:
        return None


class _StubReflector:
    """Returns a canned verdict per call; records the calls."""

    def __init__(self, verdicts: list[ReflectionVerdict]) -> None:
        self._verdicts = list(verdicts)
        self.calls: list[tuple[float, TaskClass]] = []

    def review(
        self,
        original_question: str,
        tool_log: Any,
        elapsed_s: float,
        current_class: TaskClass,
    ) -> ReflectionVerdict:
        self.calls.append((elapsed_s, current_class))
        if self._verdicts:
            return self._verdicts.pop(0)
        return ReflectionVerdict(
            continue_research=False,
            reason="default stop",
            suggested_class="interactive_fast",
        )


def _make_mr_node() -> _FakeModelRequestNode:
    return _FakeModelRequestNode(
        request=ModelRequest(parts=[UserPromptPart(content="q")]),
    )


def _drive(
    fake_run: _FakeAgentRun,
    reflector: _StubReflector,
    *,
    budget: TaskBudget,
    events: list[dict[str, Any]] | None = None,
    run_id: str = "test-run",
    monkeypatch: pytest.MonkeyPatch | None = None,
) -> Any:
    """Run the async runner to completion and return the result.

    When ``monkeypatch`` is provided, replace the runner's wall-clock
    with the fake run's deterministic ``_now`` so reflection cadences
    fire at exact (test-controlled) elapsed times.

    sensitivity_tier: 1
    """
    sink_events: list[dict[str, Any]] = events if events is not None else []
    if monkeypatch is not None:
        from src.agents.core import agent_base

        def _fake_monotonic() -> float:
            return fake_run._now

        monkeypatch.setattr(agent_base.time, "monotonic", _fake_monotonic)
    return asyncio.run(
        _run_pa_with_reflection_async(
            _FakeAgent(fake_run),
            "What's the question?",
            budget=budget,
            run_id=run_id,
            reflector=reflector,
            event_sink=sink_events.append,
            tool_log=[],
        ),
    )


def test_runner_returns_result_without_review_under_budget(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Short runs never trigger self-review.

    With ``seconds_per_step=0`` the wall clock never advances past
    the 10s first-reflect threshold.

    sensitivity_tier: 1
    """
    monkeypatch.setenv("ARANDU_DATA_DIR", str(tmp_path))
    fake_run = _FakeAgentRun(
        nodes=[_make_mr_node(), _FakeCallToolsNode()],
    )
    refl = _StubReflector(verdicts=[])
    events: list[dict[str, Any]] = []

    result = _drive(
        fake_run, refl,
        budget=TaskBudget.interactive_fast(),
        events=events,
    )
    assert result == "FINAL_OUTPUT"
    assert refl.calls == []  # no reviews
    types = [e["type"] for e in events]
    assert "self_review_start" not in types


def test_runner_injects_stop_when_review_returns_false(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If review says stop, the next ModelRequestNode gets STOP_REQUEST.

    sensitivity_tier: 1
    """
    monkeypatch.setenv("ARANDU_DATA_DIR", str(tmp_path))
    target_node = _make_mr_node()
    fake_run = _FakeAgentRun(
        nodes=[
            _FakeCallToolsNode(),  # advances clock past reflect threshold
            target_node,            # this should receive the STOP_REQUEST
        ],
        seconds_per_step=11.0,
    )
    refl = _StubReflector(verdicts=[
        ReflectionVerdict(
            continue_research=False,
            reason="enough already",
            suggested_class="interactive_fast",
        ),
    ])
    events: list[dict[str, Any]] = []
    _drive(
        fake_run, refl,
        budget=TaskBudget.interactive_fast(),
        events=events,
        monkeypatch=monkeypatch,
    )

    # Review was triggered once.
    assert len(refl.calls) == 1
    # The target ModelRequestNode picked up the STOP_REQUEST.
    stop_parts = [
        p for p in target_node.request.parts
        if isinstance(p, UserPromptPart)
        and "STOP_REQUEST" in p.content  # type: ignore[arg-type]
    ]
    assert len(stop_parts) == 1
    types = [e["type"] for e in events]
    assert "self_review_start" in types
    assert "self_review_done" in types


def test_runner_promotes_class_on_continue_verdict(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``continue_research=True`` with deep suggestion emits the banner event.

    sensitivity_tier: 1
    """
    monkeypatch.setenv("ARANDU_DATA_DIR", str(tmp_path))
    fake_run = _FakeAgentRun(
        nodes=[
            _FakeCallToolsNode(),    # advances clock past 10s
            _make_mr_node(),         # next ModelRequestNode (not stopped)
            _FakeCallToolsNode(),
        ],
        seconds_per_step=11.0,
    )
    refl = _StubReflector(verdicts=[
        ReflectionVerdict(
            continue_research=True,
            reason="needs synthesis across many sources",
            suggested_class="interactive_deep",
        ),
        # No second review fires within this short fake run, but
        # provide a fallback verdict in case the cadence-interval
        # math triggers a second checkpoint.
        ReflectionVerdict(
            continue_research=False,
            reason="enough",
            suggested_class="interactive_fast",
        ),
    ])
    events: list[dict[str, Any]] = []
    _drive(
        fake_run, refl,
        budget=TaskBudget.interactive_fast(),
        events=events,
        monkeypatch=monkeypatch,
    )

    promo_events = [
        e for e in events
        if e.get("type") == "extended_research_announced"
    ]
    assert len(promo_events) == 1
    assert promo_events[0]["task_class"] == TaskClass.INTERACTIVE_DEEP.value
    assert "synthesis" in promo_events[0]["reason"]


def test_runner_honors_user_cancel(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancel-registry flag set mid-run triggers a STOP_REQUEST.

    sensitivity_tier: 1
    """
    monkeypatch.setenv("ARANDU_DATA_DIR", str(tmp_path))
    from src.agents.core import cancel_registry

    cancel_registry._reset_for_tests()
    cancel_registry.request_cancel("user-cancel-run")

    target_node = _make_mr_node()
    fake_run = _FakeAgentRun(
        nodes=[target_node, _FakeCallToolsNode()],
    )
    refl = _StubReflector(verdicts=[])
    events: list[dict[str, Any]] = []
    _drive(
        fake_run, refl,
        budget=TaskBudget.interactive_fast(),
        events=events,
        run_id="user-cancel-run",
        monkeypatch=monkeypatch,
    )

    types = [e["type"] for e in events]
    assert "user_stopped_research" in types
    # Target node got a STOP_REQUEST appended.
    stop_parts = [
        p for p in target_node.request.parts
        if isinstance(p, UserPromptPart)
        and "STOP_REQUEST" in p.content  # type: ignore[arg-type]
    ]
    assert len(stop_parts) == 1
    cancel_registry._reset_for_tests()
