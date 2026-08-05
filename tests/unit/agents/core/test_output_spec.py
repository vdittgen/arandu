"""Structured-output mode selection per agent pattern.

Regression cover for the cloud delegation failure: agents that
delegate must use tool-based structured output, or a native-mode
profile makes them answer directly and never call a ``delegate_*``
tool.

sensitivity_tier: N/A
"""

from __future__ import annotations

import pytest
from src.agents.core.agent_base import (
    SBAgent,
    SBDeepAgent,
    SBOrchestrator,
    _output_spec_for,
)
from src.agents.core.output_types import BrainResponse

ToolOutput = pytest.importorskip("pydantic_ai").ToolOutput


class _Single(SBAgent):
    agent_id = "single"
    output_type = BrainResponse


class _Orchestrator(SBOrchestrator):
    agent_id = "orchestrator"
    output_type = BrainResponse


class _Deep(SBDeepAgent):
    agent_id = "deep"
    output_type = BrainResponse


def test_single_agent_keeps_the_profile_default() -> None:
    """Non-delegating agents must stay on the model's own default.

    The Pro build points cloud models at native structured output
    because the gateway ignores ``tool_choice="required"``; forcing
    tool-mode here would undo that for the agents it was added for.
    """
    assert _output_spec_for(_Single()) is BrainResponse


def test_orchestrator_is_pinned_to_tool_output() -> None:
    spec = _output_spec_for(_Orchestrator())
    assert isinstance(spec, ToolOutput)


def test_deep_agent_is_pinned_to_tool_output() -> None:
    spec = _output_spec_for(_Deep())
    assert isinstance(spec, ToolOutput)


def test_delegating_classes_are_marked() -> None:
    """The marker drives the choice, so assert it directly."""
    assert SBOrchestrator.delegates is True
    assert SBDeepAgent.delegates is True
    assert getattr(SBAgent, "delegates", False) is False


def test_unknown_agent_shape_falls_back_to_raw_output_type() -> None:
    """A duck-typed agent without the marker must not crash."""

    class _Bare:
        output_type = BrainResponse

    assert _output_spec_for(_Bare()) is BrainResponse
