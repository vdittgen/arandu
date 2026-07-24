"""Generic user-agent task adapter for evals.

Covers the boundary between the eval runner and the user-agent
registry: that :func:`evals.tasks.user_agent_task` resolves a
registered agent, raises ``ModelUnavailableError`` for missing
factories or non-string inputs, and pipes the agent's output through
unchanged on success.

sensitivity_tier: N/A
"""

from __future__ import annotations

import pytest
from evals.tasks import (
    ModelUnavailableError,
    _expand_brain_response,
    user_agent_task,
)
from src.agents.core.audit import reset_default_chain_for_tests
from src.agents.core.config_store import AgentConfig
from src.agents.core.output_types import BrainResponse
from src.agents.core.registry import (
    AgentDefinition,
    register_agent,
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


class _FakeRecord:
    def __init__(self, output, error=None):
        self.output = output
        self.error = error


class _FakeAgent:
    """Minimal SBAgent-shaped stub that returns a canned response."""

    agent_id = "user.fake"

    def __init__(self, response: BrainResponse | None,
                 error: str | None = None):
        self._response = response
        self._error = error

    def run(self, _inputs):
        if self._error is not None:
            raise RuntimeError(self._error)
        return _FakeRecord(self._response)


def _register(agent_id: str, factory) -> None:
    register_agent(AgentDefinition(
        agent_id=agent_id,
        name="Fake user agent",
        description="",
        category="user",
        parent_agent="brain",
        tier=Tier.INTERACTIVE,
        max_sensitivity_tier=1,
        editable=True,
        default_config=AgentConfig(
            agent_id=agent_id,
            system_prompt="",
            model_route="inherit",
            model_override=None,
            enabled_tools=(),
            enabled_skills=(),
            editable=True,
        ),
        available_tools=(),
        available_skills=(),
        output_schema="BrainResponse",
        pattern="single",
        factory=factory,
        tags=("user",),
    ))


def test_unknown_agent_raises_unavailable() -> None:
    with pytest.raises(ModelUnavailableError) as exc:
        user_agent_task("user.never_registered")
    assert "not registered" in str(exc.value)


def test_factory_missing_raises_unavailable() -> None:
    _register("user.no_factory", factory=None)
    with pytest.raises(ModelUnavailableError) as exc:
        user_agent_task("user.no_factory")
    assert "factory" in str(exc.value)


def test_resolved_task_runs_string_input() -> None:
    response = BrainResponse(
        answer="ok", model="stub", sources=[], context_summary="",
    )
    _register("user.fake", factory=lambda: _FakeAgent(response))
    task = user_agent_task("user.fake")
    # Plain prose in `answer` → expander returns the raw response.
    assert task("hello world") is response


def test_task_expands_json_in_answer_into_top_level_fields() -> None:
    response = BrainResponse(
        answer='{"purchase_id": "12345", "status": "enviada"}',
        model="stub",
        sources=[],
        context_summary="",
    )
    _register("user.fake", factory=lambda: _FakeAgent(response))
    task = user_agent_task("user.fake")
    out = task("Email: Sua compra #12345 foi enviada.")
    assert isinstance(out, dict)
    # JSON keys surface at the top level so FieldEquals on
    # `output.purchase_id` resolves correctly.
    assert out["purchase_id"] == "12345"
    assert out["status"] == "enviada"
    # Original BrainResponse fields remain available for evaluators
    # that test the raw answer string.
    assert out["answer"] == '{"purchase_id": "12345", "status": "enviada"}'
    assert out["model"] == "stub"


def test_non_string_inputs_raise_unavailable() -> None:
    response = BrainResponse(
        answer="ok", model="stub", sources=[], context_summary="",
    )
    _register("user.fake", factory=lambda: _FakeAgent(response))
    task = user_agent_task("user.fake")
    with pytest.raises(ModelUnavailableError) as exc:
        task({"not": "a string"})
    assert "must be a string" in str(exc.value)


def test_pydantic_ai_runtime_error_maps_to_unavailable() -> None:
    _register(
        "user.fake",
        factory=lambda: _FakeAgent(None, error="pydantic-ai not installed"),
    )
    task = user_agent_task("user.fake")
    with pytest.raises(ModelUnavailableError):
        task("hello")


def test_other_runtime_error_propagates() -> None:
    _register(
        "user.fake",
        factory=lambda: _FakeAgent(None, error="bad input"),
    )
    task = user_agent_task("user.fake")
    with pytest.raises(RuntimeError) as exc:
        task("hello")
    assert "bad input" in str(exc.value)


def test_no_output_raises_unavailable() -> None:
    _register("user.fake", factory=lambda: _FakeAgent(None))
    task = user_agent_task("user.fake")
    with pytest.raises(ModelUnavailableError):
        task("hello")


# ---------------------------------------------------------------------------
# _expand_brain_response — direct unit coverage
# ---------------------------------------------------------------------------


def test_expand_returns_raw_when_no_answer_attr() -> None:
    class _NoAnswer:
        pass

    obj = _NoAnswer()
    assert _expand_brain_response(obj) is obj


def test_expand_returns_raw_when_answer_is_prose() -> None:
    resp = BrainResponse(
        answer="just a sentence, no JSON here.",
        model="m", sources=[], context_summary="",
    )
    assert _expand_brain_response(resp) is resp


def test_expand_handles_markdown_fenced_json() -> None:
    fenced = '```json\n{"purchase_id": "abc", "status": "ok"}\n```'
    resp = BrainResponse(
        answer=fenced, model="m", sources=[], context_summary="",
    )
    out = _expand_brain_response(resp)
    assert isinstance(out, dict)
    assert out["purchase_id"] == "abc"
    assert out["status"] == "ok"


def test_expand_handles_prose_around_json_object() -> None:
    resp = BrainResponse(
        answer='Here you go: {"purchase_id": "99"} hope it helps.',
        model="m", sources=[], context_summary="",
    )
    out = _expand_brain_response(resp)
    assert isinstance(out, dict)
    assert out["purchase_id"] == "99"


def test_expand_returns_raw_when_answer_is_non_object_json() -> None:
    # A bare string or list parses as JSON but isn't a dict — fall
    # through to the raw response so evaluators don't see a confusing
    # shape change.
    resp = BrainResponse(
        answer='"just a string"',
        model="m", sources=[], context_summary="",
    )
    assert _expand_brain_response(resp) is resp


def test_expand_parsed_keys_override_brain_response_keys() -> None:
    # If the JSON happens to use a BrainResponse field name (e.g.
    # `model`), the parsed value wins because the dataset is the
    # ground truth of what the user wants to test.
    resp = BrainResponse(
        answer='{"model": "user-picked", "purchase_id": "1"}',
        model="actual-model", sources=[], context_summary="",
    )
    out = _expand_brain_response(resp)
    assert out["model"] == "user-picked"
    assert out["purchase_id"] == "1"
