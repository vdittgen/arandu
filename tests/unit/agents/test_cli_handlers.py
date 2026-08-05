"""CLI handler smoke tests for the Agents page IPC surface.

The handlers shell-print JSON to stdout. We capture stdout, parse the
result, and assert on the shape — that's the contract Rust speaks.

sensitivity_tier: N/A
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from src.agents.brain import bootstrap_agents
from src.agents.cli_handlers import (
    cmd_agents_create,
    cmd_agents_get,
    cmd_agents_list,
    cmd_agents_reset,
    cmd_agents_update,
    cmd_agents_user_update,
)
from src.agents.core.registry import reset_registry_for_tests


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    # Redirect the agent_configs SQLite path so each test starts fresh.
    monkeypatch.setattr(
        "src.agents.cli_handlers._agents_db_path",
        lambda: tmp_path / "test.sqlite3",
    )
    # ``_UserAgent`` resolves its system_prompt from agent_configs via
    # the same path; point its DEFAULT_DB_PATH at the same tmp file so
    # writes-from-cli-handlers and reads-from-registration agree.
    from src.agents.core import config_store as _cfg_store
    monkeypatch.setattr(
        _cfg_store, "DEFAULT_DB_PATH", tmp_path / "test.sqlite3",
    )
    # User-agents store lives in the same SQLite DB; point it at the
    # same tmp path so the test process sees a fresh table.
    from src.agents.user_agents import store as _ua_store
    monkeypatch.setattr(_ua_store, "DEFAULT_DB_PATH", tmp_path / "user.sqlite3")
    from src.agents.user_agents import skill_store as _sk_store
    monkeypatch.setattr(_sk_store, "DEFAULT_DB_PATH", tmp_path / "user.sqlite3")
    reset_registry_for_tests()
    bootstrap_agents()


def _make_editable_agent() -> str:
    """Register a user-authored agent and return its id."""
    from src.agents.user_agents.registration import register_one_user_agent
    from src.agents.user_agents.store import UserAgentStore, UserAgentUpsert

    store = UserAgentStore()
    try:
        row = store.insert(UserAgentUpsert(
            name="Test User Agent",
            description="for unit tests",
            system_prompt="default user prompt",
            model_route="inherit",
        ))
    finally:
        store.close()
    register_one_user_agent(row)
    return row.agent_id


def _capture(func, *args) -> tuple[int, dict]:
    """Run a handler and parse the printed JSON."""
    import io
    import sys

    buf = io.StringIO()
    real_stdout = sys.stdout
    sys.stdout = buf
    try:
        code = func(*args)
    finally:
        sys.stdout = real_stdout
    payload = buf.getvalue().strip()
    return code, json.loads(payload) if payload else {}


def test_list_returns_all_registered_agents() -> None:
    code, payload = _capture(cmd_agents_list)
    assert code == 0
    ids = {a["agent_id"] for a in payload["agents"]}
    # 1 brain + 14 sub-agents migrated so far.
    assert "brain" in ids
    for sub_id in (
        "sensitivity", "labeler", "triage",
        "fact_extractor",
        "message_evaluator", "pending_reply",
        "contact_context", "actionable_events",
        "query_router", "topic_extractor",
        "schema_discovery", "model_generator",
        "relationship_tracker",
    ):
        assert sub_id in ids, f"missing {sub_id}"


def test_get_returns_brain_with_locked_flag() -> None:
    code, payload = _capture(cmd_agents_get, "brain")
    assert code == 0
    assert payload["agent"]["agent_id"] == "brain"
    assert payload["agent"]["editable"] is False
    assert "locked" in payload["agent"]["tags"]


def test_get_unknown_agent_errors() -> None:
    code, payload = _capture(cmd_agents_get, "no_such_agent")
    assert code == 1
    assert "unknown" in payload["error"]


def test_update_persists_prompt(tmp_path: Path) -> None:
    agent_id = _make_editable_agent()
    code, payload = _capture(
        cmd_agents_update,
        agent_id,
        json.dumps({"system_prompt": "custom prompt"}),
    )
    assert code == 0
    assert payload["agent"]["config"]["system_prompt"] == "custom prompt"
    assert payload["agent"]["config"]["version"] == 2

    # Re-reading via get reflects the persisted override.
    code2, again = _capture(cmd_agents_get, agent_id)
    assert code2 == 0
    assert again["agent"]["config"]["system_prompt"] == "custom prompt"


def test_update_user_agent_override_takes_effect_at_runtime(
    tmp_path: Path,
) -> None:
    """Regression: a textarea save on a ``user.*`` agent must take
    effect at runtime, not just in the UI display.

    Previously ``cmd_agents_update`` wrote to ``agent_configs`` while
    ``_UserAgent.__init__`` read only from ``user_agents`` — so the
    UI displayed the new prompt while the runtime kept executing the
    old one. The fix: ``_UserAgent`` resolves with ``agent_configs``
    taking precedence, and ``cmd_agents_update`` re-registers the
    dynamic class so freshly-saved overrides are picked up without
    a process restart.
    """
    from src.agents.core.registry import get_agent
    from src.agents.user_agents.store import UserAgentStore

    agent_id = _make_editable_agent()

    # agent_configs is empty before the update — baseline reigns.
    definition = get_agent(agent_id)
    assert definition is not None
    assert definition.factory is not None
    assert definition.factory().system_prompt == "default user prompt"

    code, _ = _capture(
        cmd_agents_update,
        agent_id,
        json.dumps({"system_prompt": "fresh runtime prompt"}),
    )
    assert code == 0

    # The in-memory _UserAgent class (what SBAgent.run uses) now
    # picks up the override.
    instance = get_agent(agent_id).factory()
    assert instance.system_prompt == "fresh runtime prompt"

    # ``user_agents`` row stays at the baseline — that's the
    # "saved" version that cmd_agents_reset restores when the
    # override is cleared.
    store = UserAgentStore()
    try:
        row = store.get(agent_id)
    finally:
        store.close()
    assert row is not None
    assert row.system_prompt == "default user prompt"


def test_update_rejects_bogus_tool_id_for_user_agent() -> None:
    """User-agent tool ids go through catalog validation, not the
    symbolic ``definition.available_tools`` allowlist.

    The old behavior silently dropped any unknown id; this caused
    real catalog tool ids like ``apple-mail:list_emails`` to be
    dropped on save because user agents' ``available_tools`` is a
    capability surface (``run_mcp_tool``, ``deliver:...``), not a
    list of raw connector tool ids. Now bogus ids surface as an
    explicit error.
    """
    agent_id = _make_editable_agent()
    code, payload = _capture(
        cmd_agents_update,
        agent_id,
        json.dumps({"enabled_tools": ["bogus_tool"]}),
    )
    assert code == 1
    assert "bogus_tool" in payload["error"]


def test_update_persists_data_tool_source_binding_for_user_agent() -> None:
    """Regression: binding a catalog data tool as a source on a user
    agent must round-trip through both the overlay and the
    ``user_agents`` row.

    Previously ``filter_tools_for_agent`` intersected the patch's
    ``enabled_tools`` against the user agent's symbolic
    ``available_tools`` (``run_mcp_tool``, ``deliver:...``, ...),
    which never contains raw ``connector_id:tool_name`` ids — so a
    "List Emails" source binding was silently stripped on save, and
    the scheduler strip kept reading "no source" after refresh.
    """
    from src.agents.user_agents.store import UserAgentStore

    agent_id = _make_editable_agent()
    tool_id = "apple-mail:list_emails"

    code, payload = _capture(
        cmd_agents_update,
        agent_id,
        json.dumps({"enabled_tools": [tool_id]}),
    )
    assert code == 0, payload
    assert tool_id in payload["agent"]["config"]["enabled_tools"]

    # The runner reads from the user_agents row, not the overlay,
    # so verify both surfaces agree.
    store = UserAgentStore()
    try:
        row = store.get(agent_id)
    finally:
        store.close()
    assert row is not None
    assert tool_id in row.enabled_mcp_tools

    # And the persisted overlay survives a re-read.
    code2, again = _capture(cmd_agents_get, agent_id)
    assert code2 == 0
    assert tool_id in again["agent"]["config"]["enabled_tools"]


def test_update_rejects_system_prompt_on_locked_agent() -> None:
    code, payload = _capture(
        cmd_agents_update,
        "brain",
        json.dumps({"system_prompt": "x"}),
    )
    assert code == 1
    assert "locked" in payload["error"]
    assert "system_prompt" in payload["error"]


def test_update_rejects_system_prompt_on_builtin_agent() -> None:
    code, payload = _capture(
        cmd_agents_update,
        "triage",
        json.dumps({"system_prompt": "x"}),
    )
    assert code == 1
    assert "locked" in payload["error"]
    assert "system_prompt" in payload["error"]


def test_update_accepts_model_override_on_locked_agent() -> None:
    code, payload = _capture(
        cmd_agents_update,
        "brain",
        json.dumps({"model_override": "qwen3:8b"}),
    )
    assert code == 0
    assert payload["agent"]["config"]["model_override"] == "qwen3:8b"


def test_update_accepts_model_override_on_builtin_agent() -> None:
    code, payload = _capture(
        cmd_agents_update,
        "triage",
        json.dumps({"model_override": "qwen3:8b"}),
    )
    assert code == 0
    assert payload["agent"]["config"]["model_override"] == "qwen3:8b"


def test_update_bad_json_returns_error() -> None:
    agent_id = _make_editable_agent()
    code, payload = _capture(
        cmd_agents_update,
        agent_id,
        "not-json",
    )
    assert code == 1
    assert "bad patch JSON" in payload["error"]


def test_reset_restores_default() -> None:
    agent_id = _make_editable_agent()
    _capture(
        cmd_agents_update,
        agent_id,
        json.dumps({"system_prompt": "custom prompt"}),
    )
    code, payload = _capture(cmd_agents_reset, agent_id)
    assert code == 0
    assert payload["agent"]["config"]["system_prompt"] == "default user prompt"


def test_reset_rejects_locked_agent() -> None:
    code, payload = _capture(cmd_agents_reset, "brain")
    assert code == 1
    assert "not editable" in payload["error"]


# ---------------------------------------------------------------------------
# Phase 5b — eval status / run
# ---------------------------------------------------------------------------


def test_run_eval_status_round_trip(monkeypatch, tmp_path) -> None:
    # Use a separate DB for the eval runner so the test is isolated.
    from src.agents import cli_handlers, eval_runner
    monkeypatch.setattr(
        eval_runner, "DEFAULT_DB_PATH", tmp_path / "evals.sqlite3",
    )
    # Drive a deterministic suite end-to-end.
    from src.agents.cli_handlers import (
        cmd_agents_eval_status as _status,
    )
    from src.agents.cli_handlers import (
        cmd_agents_run_eval as _run,
    )

    code, payload = _capture(_run, "firewall.injection", "manual")
    assert code == 0
    assert payload["run"]["status"] == "passed"
    assert payload["run"]["cases_passed"] > 0

    code, status = _capture(_status, "firewall.injection", 1)
    assert code == 0
    assert status["latest"]["status"] == "passed"
    # cli_handlers re-export check
    assert callable(cli_handlers.cmd_agents_run_eval)


def test_run_eval_unknown_agent_errors(monkeypatch, tmp_path) -> None:
    from src.agents import eval_runner
    from src.agents.cli_handlers import cmd_agents_run_eval as _run
    monkeypatch.setattr(
        eval_runner, "DEFAULT_DB_PATH", tmp_path / "evals.sqlite3",
    )
    code, payload = _capture(_run, "no_such", "manual")
    assert code == 1
    assert "unknown agent" in payload["error"]


def test_update_does_not_trigger_auto_eval(monkeypatch) -> None:
    """Edits must NOT spawn an eval subprocess.

    Auto-eval was removed in 0.5.0; evals run on explicit user
    action. The Agents page surfaces a "Run eval" button that the
    user clicks when they want to run judging.
    """
    import subprocess

    spawned: list[Any] = []
    monkeypatch.setattr(
        subprocess, "Popen",
        lambda *a, **kw: spawned.append((a, kw)) or object(),
    )

    agent_id = _make_editable_agent()
    code, _payload = _capture(
        cmd_agents_update, agent_id,
        '{"system_prompt": "no-auto-eval"}',
    )
    assert code == 0
    assert spawned == []


# ---------------------------------------------------------------------------
# Orchestrator validation (Phase 2)
# ---------------------------------------------------------------------------


def _create_payload(
    *,
    name: str = "Research Lead",
    pattern: str = "single",
    subagents: tuple[str, ...] = (),
) -> str:
    return json.dumps({
        "name": name,
        "description": "delegates research questions",
        "system_prompt": "Route the question to the right specialist.",
        "model_route": "inherit",
        "model_override": None,
        "enabled_skills": [],
        "enabled_mcp_tools": [],
        "brain_access": True,
        "max_sensitivity_tier": 2,
        "schedule_cron": None,
        "schedule_enabled": False,
        "pattern": pattern,
        "subagents": list(subagents),
    })


def test_create_orchestrator_succeeds_with_valid_subagents() -> None:
    sub_a = _make_editable_agent()
    sub_b = _make_editable_agent()
    code, payload = _capture(
        cmd_agents_create,
        _create_payload(
            pattern="orchestrator",
            subagents=(sub_a, sub_b),
        ),
    )
    assert code == 0, payload
    assert payload["agent"]["pattern"] == "orchestrator"
    assert payload["user_row"]["pattern"] == "orchestrator"
    assert payload["user_row"]["subagents"] == [sub_a, sub_b]
    # Delegation targets surface in available_tools for the UI.
    assert f"delegate:{sub_a}" in payload["agent"]["available_tools"]
    assert f"delegate:{sub_b}" in payload["agent"]["available_tools"]


def test_create_leaves_model_override_unset_by_default() -> None:
    """OSS: created agents inherit the user's configured Ollama model
    instead of pinning to a tier default (the tier map is empty)."""
    code, payload = _capture(
        cmd_agents_create,
        _create_payload(name="Default Model Watcher"),
    )
    assert code == 0, payload
    assert payload["user_row"]["model_override"] is None


def test_create_preserves_explicit_model_override() -> None:
    explicit = "anthropic/claude-haiku-4-5"
    raw = json.loads(_create_payload(name="Explicit Model"))
    raw["model_override"] = explicit
    code, payload = _capture(cmd_agents_create, json.dumps(raw))
    assert code == 0, payload
    assert payload["user_row"]["model_override"] == explicit


def test_create_rejects_unknown_subagent() -> None:
    code, payload = _capture(
        cmd_agents_create,
        _create_payload(
            pattern="orchestrator",
            subagents=("user.does_not_exist",),
        ),
    )
    assert code == 1
    assert "not registered" in payload["error"]


def test_create_rejects_orchestrator_as_subagent() -> None:
    # Brain ships as a registered orchestrator — picking it must fail
    # because v1 forbids orchestrator-of-orchestrator graphs.
    code, payload = _capture(
        cmd_agents_create,
        _create_payload(
            pattern="orchestrator",
            subagents=("brain",),
        ),
    )
    assert code == 1
    assert "only single-pattern" in payload["error"]


def test_create_rejects_deep_pattern() -> None:
    code, payload = _capture(
        cmd_agents_create,
        _create_payload(pattern="deep"),
    )
    assert code == 1
    assert "invalid pattern" in payload["error"]


def test_create_rejects_orchestrator_without_subagents() -> None:
    code, payload = _capture(
        cmd_agents_create,
        _create_payload(pattern="orchestrator", subagents=()),
    )
    assert code == 1
    assert "at least one subagent" in payload["error"]


def test_create_rejects_duplicate_subagents() -> None:
    sub_a = _make_editable_agent()
    code, payload = _capture(
        cmd_agents_create,
        _create_payload(
            pattern="orchestrator",
            subagents=(sub_a, sub_a),
        ),
    )
    assert code == 1
    assert "more than once" in payload["error"]


def test_create_rejects_subagents_on_single_pattern() -> None:
    sub_a = _make_editable_agent()
    code, payload = _capture(
        cmd_agents_create,
        _create_payload(
            pattern="single",
            subagents=(sub_a,),
        ),
    )
    assert code == 1
    assert "only be set when pattern is 'orchestrator'" in payload["error"]


def test_user_update_rejects_self_as_subagent() -> None:
    sub_a = _make_editable_agent()
    # Create an orchestrator that delegates to sub_a.
    code, payload = _capture(
        cmd_agents_create,
        _create_payload(
            pattern="orchestrator",
            subagents=(sub_a,),
        ),
    )
    assert code == 0
    orchestrator_id = payload["agent"]["agent_id"]

    # Try to update it so it delegates to itself.
    code, payload = _capture(
        cmd_agents_user_update,
        orchestrator_id,
        _create_payload(
            pattern="orchestrator",
            subagents=(orchestrator_id,),
        ),
    )
    assert code == 1
    assert "may not delegate to itself" in payload["error"]


def test_brain_serialization_carries_subagents() -> None:
    """Brain's runtime delegation list must surface in agents-list."""
    code, payload = _capture(cmd_agents_list)
    assert code == 0
    brain = next(a for a in payload["agents"] if a["agent_id"] == "brain")
    assert brain["pattern"] == "orchestrator"
    assert brain["subagents"], "Brain should expose its subagents list"
    # A few well-known sub-agents the orchestrator delegates to.
    for known in ("sensitivity", "labeler", "triage"):
        assert known in brain["subagents"], f"missing {known}"


def test_user_orchestrator_subagents_round_trip_through_get() -> None:
    sub_a = _make_editable_agent()
    sub_b = _make_editable_agent()
    code, payload = _capture(
        cmd_agents_create,
        _create_payload(
            pattern="orchestrator",
            subagents=(sub_a, sub_b),
        ),
    )
    assert code == 0
    orchestrator_id = payload["agent"]["agent_id"]
    assert payload["agent"]["subagents"] == [sub_a, sub_b]

    code, again = _capture(cmd_agents_get, orchestrator_id)
    assert code == 0
    assert again["agent"]["subagents"] == [sub_a, sub_b]


def test_create_rejects_unknown_mcp_tool_id() -> None:
    payload_json = json.dumps({
        "name": "Bogus Tools Agent",
        "description": "d",
        "system_prompt": "p",
        "model_route": "inherit",
        "enabled_mcp_tools": ["fake:nonsense"],
        "brain_access": True,
        "max_sensitivity_tier": 2,
        "schedule_cron": None,
        "schedule_enabled": False,
    })
    code, payload = _capture(cmd_agents_create, payload_json)
    assert code == 1
    assert "fake:nonsense" in payload["error"]


def test_create_rejects_unknown_delivery_tool_id() -> None:
    payload_json = json.dumps({
        "name": "Bad Delivery Agent",
        "description": "d",
        "system_prompt": "p",
        "model_route": "inherit",
        "enabled_mcp_tools": [],
        "delivery_tools": ["fake:nonsense"],
        "brain_access": True,
        "max_sensitivity_tier": 2,
        "schedule_cron": None,
        "schedule_enabled": False,
    })
    code, payload = _capture(cmd_agents_create, payload_json)
    assert code == 1
    assert "delivery tool" in payload["error"]
    assert "fake:nonsense" in payload["error"]


def test_create_rejects_data_tool_as_delivery() -> None:
    """Delivery requires an action tool — passing a data tool errors."""
    payload_json = json.dumps({
        "name": "Wrong Delivery Type",
        "description": "d",
        "system_prompt": "p",
        "model_route": "inherit",
        "enabled_mcp_tools": ["apple-mail:list_emails"],
        "delivery_tools": ["apple-mail:list_emails"],
        "brain_access": True,
        "max_sensitivity_tier": 2,
        "schedule_cron": None,
        "schedule_enabled": False,
    })
    code, payload = _capture(cmd_agents_create, payload_json)
    assert code == 1
    assert "must be an action tool" in payload["error"]


def test_create_allows_delivery_tool_outside_enabled_mcp_tools() -> None:
    """Delivery is independent of enabled_mcp_tools by design."""
    payload_json = json.dumps({
        "name": "Pure Delivery",
        "description": "d",
        "system_prompt": "p",
        "model_route": "inherit",
        "enabled_mcp_tools": ["apple-mail:list_emails"],
        "delivery_tools": ["whatsapp:send_message"],
        "brain_access": True,
        "max_sensitivity_tier": 2,
        "schedule_cron": None,
        "schedule_enabled": False,
    })
    code, payload = _capture(cmd_agents_create, payload_json)
    assert code == 0, payload
    assert payload["user_row"]["delivery_tools"] == ["whatsapp:send_message"]
    # The LLM should NOT see whatsapp:send_message under run_mcp_tool
    # since it's delivery-only, but it's present in delivery_tools
    # (verified via the available_tools render below).
    assert "deliver:whatsapp:send_message" in payload["agent"]["available_tools"]


def test_create_defaults_pattern_to_single_when_omitted() -> None:
    """Missing `pattern` field falls back to ``"single"`` (back-compat)."""
    payload_json = json.dumps({
        "name": "Plain User Agent",
        "description": "no orchestration",
        "system_prompt": "Be helpful.",
        "model_route": "inherit",
        "model_override": None,
        "enabled_skills": [],
        "enabled_mcp_tools": [],
        "brain_access": True,
        "max_sensitivity_tier": 2,
        "schedule_cron": None,
        "schedule_enabled": False,
        # No "pattern", no "subagents".
    })
    code, payload = _capture(cmd_agents_create, payload_json)
    assert code == 0, payload
    assert payload["agent"]["pattern"] == "single"
    assert payload["user_row"]["pattern"] == "single"
    assert payload["user_row"]["subagents"] == []


