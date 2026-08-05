"""CLI handlers for the agent management surface.

These are invoked from ``src/core/cli.py`` by the Tauri shell to back
the Agents page. Output is always a single JSON document written to
stdout; errors are returned as ``{"error": str}`` for the frontend.

Surface:

- ``agents-list``                 → list every registered agent
- ``agents-get  <agent_id>``     → resolved config + registry info
- ``agents-update <agent_id> <patch_json>`` → apply override
- ``agents-reset <agent_id>``    → drop override and return default

sensitivity_tier: 1
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Resolve the built-in eval datasets directory without importing
# ``evals.run_evals`` — that module imports ``pydantic_evals``, which
# lives in the ``dev`` optional-dependency group and is absent in
# regular installs. The Agents page only needs to *read* the YAML to
# display it, so we don't need the runner machinery here.
_BUILTIN_EVAL_DATASETS_DIR = (
    Path(__file__).resolve().parents[2] / "evals" / "datasets"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _emit(payload: dict[str, Any]) -> int:
    """Write a JSON payload to stdout, return ``0`` for success."""
    sys.stdout.write(json.dumps(payload, default=_json_default))
    sys.stdout.write("\n")
    sys.stdout.flush()
    return 0


def _emit_error(message: str) -> int:
    """Write an ``{"error": ...}`` payload and return non-zero."""
    sys.stdout.write(json.dumps({"error": message}))
    sys.stdout.write("\n")
    sys.stdout.flush()
    return 1


def _json_default(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "name"):  # Enum
        return obj.name
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"unserializable: {type(obj)!r}")


def _ensure_bootstrap() -> None:
    """Populate the agent registry once per process.

    Importing ``bootstrap_agents`` and calling it has the side effect of
    registering all Phase 1-3g agents. The call is idempotent.
    """
    from src.agents.brain import bootstrap_agents

    bootstrap_agents()


def _agents_db_path() -> Path:
    """Return the SQLite path the agent_configs table lives in."""
    return Path.home() / ".arandu" / "data" / "arandu.sqlite3"


def _config_store() -> Any:
    """Open a fresh :class:`AgentConfigStore` over the SQLite DB.

    The schema is created by the central migration runner; we call
    ``initialize()`` defensively so the page works on a fresh install
    that hasn't run a pipeline migration yet.
    """
    from src.agents.core.config_store import AgentConfigStore
    from src.core.sqlite.engine import connect_with_pragmas

    path = _agents_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # ``connect_with_pragmas`` opens in autocommit mode — essential here
    # because each CLI invocation opens a fresh connection that exits
    # immediately after the handler returns; without autocommit the
    # override write is rolled back.  It also applies the engine's WAL +
    # busy_timeout pragmas so this connection queues on contention like
    # every other writer instead of failing after the 5s default.
    conn = connect_with_pragmas(path)
    store = AgentConfigStore(conn)
    store.initialize()
    return store


def _resolved_model_name(resolved: Any) -> str:
    """Return the concrete LLM model name that would actually be used.

    Honours ``model_override`` first, then falls back to the route's
    configured model. Reports ``"inherit"`` when the agent uses the
    global default and no override is set, so the UI can show the
    abstraction without misleading the user.

    sensitivity_tier: 1
    """
    if resolved.model_override:
        return str(resolved.model_override)
    route = (resolved.model_route or "inherit").lower()
    try:
        from src.agents.core.model_factory import (
            local_endpoint,
            remote_endpoint,
        )
        if route == "remote":
            return remote_endpoint().model_name
        if route == "local":
            return local_endpoint().model_name
    except Exception:  # noqa: BLE001
        pass
    # "inherit" or unknown route — the firewall picks at call time.
    try:
        from src.agents.core.model_factory import remote_endpoint
        return remote_endpoint().model_name
    except Exception:  # noqa: BLE001
        return "default"


def _serialize_definition(
    definition: Any,
    *,
    store: Any,
    user_row_index: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Project an ``AgentDefinition`` + resolved config into the IPC shape.

    The frontend gets one object per agent with the registry metadata
    plus the merged config currently in effect.

    For user-authored agents, ``pre_ai_system_prompt`` and
    ``pre_ai_description`` carry the snapshot captured at the most
    recent prompt-engineer apply. They are ``None`` for built-in
    agents and for user agents that have no pending revert. Pass
    ``user_row_index`` to avoid re-opening the SQLite store per call
    in bulk paths like ``cmd_agents_list``.
    """
    resolved = store.resolve(
        definition.agent_id, default=definition.default_config,
    )
    pre_ai_system_prompt: str | None = None
    pre_ai_description: str | None = None
    delivery_tools: list[str] = []
    if definition.agent_id.startswith("user."):
        user_row = None
        if user_row_index is not None:
            user_row = user_row_index.get(definition.agent_id)
        else:
            from src.agents.user_agents.store import UserAgentStore
            ua_store = UserAgentStore()
            try:
                user_row = ua_store.get(definition.agent_id)
            finally:
                ua_store.close()
        if user_row is not None:
            pre_ai_system_prompt = user_row.pre_ai_system_prompt
            pre_ai_description = user_row.pre_ai_description
            delivery_tools = list(user_row.delivery_tools)
    return {
        "agent_id": definition.agent_id,
        "name": definition.name,
        "description": definition.description,
        "category": definition.category,
        "parent_agent": definition.parent_agent,
        "tier": definition.tier.name,
        "max_sensitivity_tier": definition.max_sensitivity_tier,
        "editable": definition.editable,
        "pattern": definition.pattern,
        "output_schema": definition.output_schema,
        "available_tools": list(definition.available_tools),
        "available_skills": list(definition.available_skills),
        "tags": list(definition.tags),
        "subagents": list(definition.subagents),
        "config": {
            "system_prompt": resolved.system_prompt,
            "model_route": resolved.model_route,
            "model_override": resolved.model_override,
            "resolved_model": _resolved_model_name(resolved),
            "enabled_tools": list(resolved.enabled_tools),
            "enabled_skills": list(resolved.enabled_skills),
            "version": resolved.version,
            "delivery_tools": delivery_tools,
        },
        "pre_ai_system_prompt": pre_ai_system_prompt,
        "pre_ai_description": pre_ai_description,
    }


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_agents_list() -> int:
    """List every registered agent as a JSON array.

    sensitivity_tier: 1
    """
    try:
        _ensure_bootstrap()
        from src.agents.core.registry import all_agents
        from src.agents.user_agents.store import UserAgentStore

        store = _config_store()
        ua_store = UserAgentStore()
        try:
            user_index = {r.agent_id: r for r in ua_store.list_all()}
        finally:
            ua_store.close()
        rows = [
            _serialize_definition(d, store=store, user_row_index=user_index)
            for d in all_agents()
        ]
        return _emit({"agents": rows})
    except Exception as exc:  # noqa: BLE001
        logger.exception("agents-list failed")
        return _emit_error(str(exc))


def cmd_agents_list_models(route: str) -> int:
    """Enumerate the models exposed by ``route``'s endpoint.

    Calls the OpenAI-compatible ``/models`` endpoint via the SDK.
    Returns ``{"route": str, "models": [str, ...]}`` sorted with
    chat-family ids first so the dropdown surfaces relevant entries.
    Errors return ``{"route": str, "models": [], "error": str}`` so
    the UI can fall back to a free-text input without crashing.

    sensitivity_tier: 1
    """
    try:
        from src.agents.core.model_factory import list_models
        ids = list_models(route)
        return _emit({"route": route, "models": ids})
    except Exception as exc:  # noqa: BLE001
        logger.exception("agents-list-models failed")
        return _emit({
            "route": route,
            "models": [],
            "error": str(exc),
        })


def cmd_agents_get(agent_id: str) -> int:
    """Return one agent's definition + resolved config.

    sensitivity_tier: 1
    """
    try:
        _ensure_bootstrap()
        from src.agents.core.registry import get_agent

        definition = get_agent(agent_id)
        if definition is None:
            return _emit_error(f"unknown agent: {agent_id}")
        store = _config_store()
        return _emit({"agent": _serialize_definition(definition, store=store)})
    except Exception as exc:  # noqa: BLE001
        logger.exception("agents-get failed")
        return _emit_error(str(exc))


def cmd_agents_update(agent_id: str, patch_json: str) -> int:
    """Apply a patch to one agent's config; refuses non-editable agents.

    ``patch_json`` is a JSON object containing any subset of
    ``system_prompt``, ``model_route``, ``model_override``,
    ``enabled_tools``, ``enabled_skills``, ``delivery_tools``.

    For user-authored agents, ``enabled_tools`` and ``delivery_tools``
    are also synced to the ``user_agents`` row so the runner (which
    reads from the row, not the overlay) picks them up on the next
    tick. ``delivery_tools`` is ignored for built-in agents.

    Evals are no longer triggered automatically — running them costs
    money on the judge tier, so the user kicks them off explicitly
    from the Agents page or via ``python -m evals.run_evals``.

    sensitivity_tier: 1
    """
    try:
        _ensure_bootstrap()
        from src.agents.core.config_store import (
            AgentConfigStoreError,
        )
        from src.agents.core.registry import (
            filter_tools_for_agent,
            get_agent,
        )
        definition = get_agent(agent_id)
        if definition is None:
            return _emit_error(f"unknown agent: {agent_id}")
        try:
            patch = json.loads(patch_json) if patch_json else {}
        except json.JSONDecodeError as exc:
            return _emit_error(f"bad patch JSON: {exc}")
        if not isinstance(patch, dict):
            return _emit_error("patch must be a JSON object")
        # ``filter_tools_for_agent`` intersects against
        # ``definition.available_tools`` — a real allowlist for built-in
        # agents, but a symbolic capability surface for user agents
        # (``run_mcp_tool``, ``deliver:...``, ``delegate:...``) that
        # never contains raw ``connector_id:tool_name`` ids. Applying
        # it to a user-agent patch silently drops every binding the
        # user just selected. User-agent tool ids are validated
        # against the catalog below via
        # ``_validate_user_agent_tool_patch`` instead.
        if (
            "enabled_tools" in patch
            and patch["enabled_tools"] is not None
            and not agent_id.startswith("user.")
        ):
            patch["enabled_tools"] = list(
                filter_tools_for_agent(definition, patch["enabled_tools"]),
            )
        # ``delivery_tools`` lives on the user_agents row, not the
        # config_store overlay; pop it before the overlay write and
        # validate up front so the row + overlay stay consistent.
        delivery_patch: list[str] | None = None
        if "delivery_tools" in patch:
            value = patch.pop("delivery_tools")
            if value is not None and not agent_id.startswith("user."):
                return _emit_error(
                    "delivery_tools is only valid for user-authored agents",
                )
            if value is not None:
                delivery_patch = [str(v) for v in value]
        if (
            agent_id.startswith("user.")
            and (
                "enabled_tools" in patch
                or delivery_patch is not None
            )
        ):
            err = _validate_user_agent_tool_patch(
                enabled_tools=patch.get("enabled_tools"),
                delivery_tools=delivery_patch,
            )
            if err is not None:
                return _emit_error(err)
        store = _config_store()
        try:
            store.update(
                agent_id,
                default=definition.default_config,
                patch=patch,
            )
        except AgentConfigStoreError as exc:
            return _emit_error(str(exc))
        # Sync tool changes to the user_agents row so the runner reads
        # them on the next tick. Re-register the agent so the in-memory
        # class picks up new tool + prompt wiring without a restart.
        if agent_id.startswith("user."):
            try:
                from src.agents.user_agents.registration import (
                    register_one_user_agent,
                )
                from src.agents.user_agents.store import (
                    UserAgentStore,
                    UserAgentUpsert,
                )
                ua_store = UserAgentStore()
                try:
                    row = ua_store.get(agent_id)
                    if row is not None and (
                        ("enabled_tools" in patch)
                        or delivery_patch is not None
                    ):
                        new_tools = (
                            tuple(patch["enabled_tools"])
                            if "enabled_tools" in patch
                            else row.enabled_mcp_tools
                        )
                        new_delivery = (
                            tuple(delivery_patch)
                            if delivery_patch is not None
                            else row.delivery_tools
                        )
                        row = ua_store.update(agent_id, UserAgentUpsert(
                            name=row.name,
                            description=row.description,
                            system_prompt=row.system_prompt,
                            model_route=row.model_route,
                            model_override=row.model_override,
                            enabled_skills=row.enabled_skills,
                            enabled_mcp_tools=new_tools,
                            brain_access=row.brain_access,
                            max_sensitivity_tier=row.max_sensitivity_tier,
                            schedule_cron=row.schedule_cron,
                            schedule_enabled=row.schedule_enabled,
                            pattern=row.pattern,
                            subagents=row.subagents,
                            delivery_tools=new_delivery,
                        ))
                finally:
                    ua_store.close()
                if row is not None:
                    register_one_user_agent(row)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "user-agent re-registration failed for %s — "
                    "runtime may keep the old prompt until restart",
                    agent_id,
                )
        return _emit({"agent": _serialize_definition(definition, store=store)})
    except Exception as exc:  # noqa: BLE001
        logger.exception("agents-update failed")
        return _emit_error(str(exc))


def _validate_user_agent_tool_patch(
    *,
    enabled_tools: list[str] | None,
    delivery_tools: list[str] | None,
) -> str | None:
    """Reuse the catalog-id check from ``_validate_user_agent_upsert``.

    The patch path doesn't go through that function because it doesn't
    construct a full upsert object; this helper applies the same
    catalog-presence + type checks against the patch fields.

    sensitivity_tier: 1
    """
    tool_index = _catalog_tool_index()
    if not tool_index:
        return None
    if enabled_tools is not None:
        for tool_id in enabled_tools:
            if ":" not in str(tool_id):
                return (
                    f"invalid mcp tool id {tool_id!r}; "
                    "expected 'connector_id:tool_name'"
                )
            if tool_id not in tool_index:
                return (
                    f"mcp tool {tool_id!r} not found in the connector "
                    "catalog"
                )
    if delivery_tools is not None:
        for tool_id in delivery_tools:
            if ":" not in str(tool_id):
                return (
                    f"invalid delivery tool id {tool_id!r}; "
                    "expected 'connector_id:tool_name'"
                )
            kind = tool_index.get(tool_id)
            if kind is None:
                return (
                    f"delivery tool {tool_id!r} not found in the "
                    "connector catalog"
                )
            if kind != "action":
                return (
                    f"delivery tool {tool_id!r} must be an action tool "
                    f"(catalog type is {kind!r})"
                )
    return None


def cmd_agents_reset(agent_id: str) -> int:
    """Drop the override row for one agent and return the default config.

    Evals are no longer triggered automatically on reset; the user
    re-runs them explicitly from the Agents page or via
    ``python -m evals.run_evals`` when they want to spend on judging.

    sensitivity_tier: 1
    """
    try:
        _ensure_bootstrap()
        from src.agents.core.config_store import (
            AgentConfigStoreError,
        )
        from src.agents.core.registry import get_agent

        definition = get_agent(agent_id)
        if definition is None:
            return _emit_error(f"unknown agent: {agent_id}")
        if not definition.editable:
            return _emit_error(f"agent not editable: {agent_id}")
        store = _config_store()
        try:
            store.reset(agent_id, default=definition.default_config)
        except AgentConfigStoreError as exc:
            return _emit_error(str(exc))
        return _emit({"agent": _serialize_definition(definition, store=store)})
    except Exception as exc:  # noqa: BLE001
        logger.exception("agents-reset failed")
        return _emit_error(str(exc))


def cmd_agents_run_eval(agent_id: str, trigger: str = "manual") -> int:
    """Run the agent's eval suite synchronously and emit the resulting row.

    Used both by the "Run eval" button (manual, blocking) and by the
    background subprocess spawned after ``agents-update`` /
    ``agents-reset``.

    Under local-only privacy mode the result also updates the
    per-agent block table: a ``passed`` run clears any existing block;
    a ``failed`` run installs one. This keeps the gateway-side block
    in sync with the most recent eval verdict so a user who fixes
    their local model can lift the block by re-running the suite.

    sensitivity_tier: 1
    """
    try:
        _ensure_bootstrap()
        from src.agents.core.agent_block_store import (
            default_agent_block_store,
        )
        from src.agents.core.registry import get_agent
        from src.agents.eval_runner import EvalRunStore, run_agent_eval
        from src.agents.firewall.egress_firewall import (
            default_egress_firewall,
        )

        definition = get_agent(agent_id)
        if definition is None:
            return _emit_error(f"unknown agent: {agent_id}")
        store = EvalRunStore()
        run = run_agent_eval(agent_id, trigger=trigger, store=store)

        policy = default_egress_firewall().policy
        if policy.routing == "local-only":
            block_store = default_agent_block_store()
            if run.status == "passed":
                block_store.unblock(agent_id)
            elif run.status == "failed":
                block_store.block(
                    agent_id,
                    reason="local model failed eval suite",
                )
        return _emit({"run": run.to_dict()})
    except Exception as exc:  # noqa: BLE001
        logger.exception("agents-run-eval failed")
        return _emit_error(str(exc))


def cmd_agents_run_eval_proposal(
    agent_id: str, proposed_override: str,
) -> int:
    """Run the agent's eval suite against a proposed ``model_override``.

    Wraps :func:`run_agent_eval` in a :func:`proposed_model_override`
    scope so every agent instantiated during the eval uses the
    candidate model instead of the saved row. The persisted row carries
    a distinct trigger (``"model_change_proposal"``) and the candidate
    model name in its error/note field so the history page can
    distinguish proposal runs from real ones.

    The frontend uses the return value to decide whether to call
    ``update_agent_config`` and persist the change — only if the
    run's status is ``"passed"``.

    sensitivity_tier: 1
    """
    try:
        _ensure_bootstrap()
        from src.agents.core.config_store import proposed_model_override
        from src.agents.core.registry import get_agent
        from src.agents.eval_runner import EvalRunStore, run_agent_eval

        definition = get_agent(agent_id)
        if definition is None:
            return _emit_error(f"unknown agent: {agent_id}")
        candidate = (proposed_override or "").strip()
        if not candidate:
            return _emit_error("proposed_override cannot be empty")
        store = EvalRunStore()
        with proposed_model_override(agent_id, candidate):
            run = run_agent_eval(
                agent_id,
                trigger="model_change_proposal",
                store=store,
            )
        return _emit({
            "run": run.to_dict(),
            "proposed_override": candidate,
        })
    except Exception as exc:  # noqa: BLE001
        logger.exception("agents-run-eval-proposal failed")
        return _emit_error(str(exc))


def cmd_agents_activity(agent_id: str, limit: int = 100) -> int:
    """Emit the most-recent input/output entries for one agent.

    Backs the "Recent runs" panel on the Agents page. ``limit`` is
    clamped to ``[1, MAX_PER_AGENT]`` server-side. Newest entries are
    returned first. Returns ``{"agent_id": ..., "entries": [...]}``.

    sensitivity_tier: varies
    """
    try:
        from src.agents.core.run_log import default_run_log

        entries = default_run_log().recent(agent_id, limit=int(limit))
        return _emit({
            "agent_id": agent_id,
            "entries": [e.to_dict() for e in entries],
        })
    except Exception as exc:  # noqa: BLE001
        logger.exception("agents-activity failed")
        return _emit_error(str(exc))


def cmd_agents_eval_status(agent_id: str, limit: int = 1) -> int:
    """Emit the latest (or N most recent) eval rows for one agent.

    sensitivity_tier: 1
    """
    try:
        _ensure_bootstrap()
        from src.agents.eval_runner import EvalRunStore

        store = EvalRunStore()
        if int(limit) <= 1:
            run = store.latest(agent_id)
            return _emit({"latest": run.to_dict() if run else None})
        history = store.history(agent_id, limit=int(limit))
        return _emit({
            "latest": history[0].to_dict() if history else None,
            "history": [r.to_dict() for r in history],
        })
    except Exception as exc:  # noqa: BLE001
        logger.exception("agents-eval-status failed")
        return _emit_error(str(exc))


# ---------------------------------------------------------------------------
# Eval dataset surface
# ---------------------------------------------------------------------------


def _parse_dataset_cases(content: str) -> list[dict[str, Any]]:
    """Best-effort projection of a YAML dataset into UI rows.

    Returns an empty list on parse error — the caller surfaces the
    raw ``content`` separately so the UI can still render it.

    sensitivity_tier: 1
    """
    try:
        import yaml
        parsed = yaml.safe_load(content) or {}
    except Exception:  # noqa: BLE001
        return []
    cases = parsed.get("cases") if isinstance(parsed, dict) else None
    if not isinstance(cases, list):
        return []
    rows: list[dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, dict):
            continue
        evaluators = case.get("evaluators") or []
        ev_names: list[str] = []
        for ev_entry in evaluators if isinstance(evaluators, list) else []:
            if isinstance(ev_entry, str):
                ev_names.append(ev_entry)
            elif isinstance(ev_entry, dict) and ev_entry.get("name"):
                ev_names.append(str(ev_entry["name"]))
        expected = case.get("expected_output")
        rows.append({
            "name": str(case.get("name") or ""),
            "inputs": json.dumps(case.get("inputs"), default=str)
            if not isinstance(case.get("inputs"), str)
            else str(case.get("inputs") or ""),
            "expected_output": (
                None if expected is None
                else json.dumps(expected, default=str)
            ),
            "evaluators": ev_names,
        })
    return rows


def cmd_agents_eval_dataset(agent_id: str) -> int:
    """Return the eval dataset YAML for ``agent_id``.

    Built-in datasets live under ``evals/datasets/<suite>.yaml`` and
    are read-only on the UI. User-agent datasets live under
    ``~/.arandu/user_eval_datasets/<agent_id>.yaml`` and are
    editable (subject to validation + firewall on upload).

    sensitivity_tier: 1
    """
    try:
        _ensure_bootstrap()
        from src.agents.core.registry import get_agent
        from src.agents.eval_runner import (
            AGENT_SUITE_MAP,
            _user_dataset_path,
        )

        definition = get_agent(agent_id)
        if definition is None:
            return _emit_error(f"unknown agent: {agent_id}")
        suite = AGENT_SUITE_MAP.get(agent_id)
        source = "builtin"
        path: Path | None = None
        if suite is not None:
            path = _BUILTIN_EVAL_DATASETS_DIR / f"{suite}.yaml"
        else:
            source = "user"
            path = _user_dataset_path(agent_id)
        content: str | None = None
        exists = path is not None and path.exists()
        if exists and path is not None:
            try:
                content = path.read_text(encoding="utf-8")
            except OSError as exc:
                return _emit_error(f"could not read dataset: {exc}")
        if content is None:
            source = "none"
        return _emit({
            "agent_id": agent_id,
            "suite": suite,
            "source": source,
            "path": str(path) if path is not None else None,
            "content": content,
            "parsed_cases": _parse_dataset_cases(content or ""),
            "exists": bool(exists),
        })
    except Exception as exc:  # noqa: BLE001
        logger.exception("agents-eval-dataset failed")
        return _emit_error(str(exc))


def cmd_agents_validate_dataset(agent_id: str, content: str) -> int:
    """Validate a user-uploaded eval YAML and persist it on success.

    Pipeline:

    1. Reject unknown agents and shipped (non-user) agents.
    2. Run the deterministic structural check.
    3. Run the LLM proposal step (best-effort).
    4. Scan the content with the injection firewall.
    5. Write the file under ``~/.arandu/user_eval_datasets/`` only
       when structural validation passed AND the firewall did not
       block.

    Always returns a :class:`DatasetValidationReport` payload, even
    when persistence is skipped.

    sensitivity_tier: 1
    """
    try:
        _ensure_bootstrap()
        from src.agents.core.registry import get_agent
        from src.agents.dataset_validator import DatasetValidatorAgent
        from src.agents.eval_runner import _user_dataset_path

        definition = get_agent(agent_id)
        if definition is None:
            return _emit_error(f"unknown agent: {agent_id}")
        if not agent_id.startswith("user."):
            return _emit_error(
                "only user-authored agents accept uploaded datasets",
            )

        # Canonicalise evaluator key (`args` → `arguments`) before
        # validating + writing so the persisted file loads cleanly
        # through pydantic-evals' Dataset model at eval-run time.
        from src.agents.dataset_validator import canonicalize_dataset_yaml

        canonical, _ = canonicalize_dataset_yaml(content)
        firewall_verdict = _scan_dataset_with_firewall(canonical)
        report = DatasetValidatorAgent().validate(
            canonical, firewall_verdict=firewall_verdict,
        )

        persisted = False
        if report.valid and firewall_verdict != "block":
            path = _user_dataset_path(agent_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(canonical, encoding="utf-8")
            persisted = True

        return _emit({
            "report": report.model_dump(),
            "persisted": persisted,
        })
    except Exception as exc:  # noqa: BLE001
        logger.exception("agents-validate-dataset failed")
        return _emit_error(str(exc))


def _scan_dataset_with_firewall(content: str) -> str:
    """Return the firewall verdict (``allow`` / ``warn`` / ``block``).

    Falls back to ``allow`` when the firewall is unavailable so the
    UI can still proceed in development — production wiring sets the
    firewall up at bootstrap.

    sensitivity_tier: 1
    """
    try:
        from src.agents.firewall.injection_firewall import (
            default_injection_firewall,
        )
    except ImportError:
        return "allow"
    try:
        firewall = default_injection_firewall()
        verdict = firewall.scan(content)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Firewall scan failed: %s", exc)
        return "allow"
    if verdict is None:
        return "allow"
    allowed = getattr(verdict, "allowed", True)
    category = str(getattr(verdict, "category", "safe") or "safe")
    if not allowed:
        return "block"
    if category != "safe":
        return "warn"
    return "allow"


# ---------------------------------------------------------------------------
# Dataset creator
# ---------------------------------------------------------------------------


def cmd_agents_suggest_dataset(
    agent_id: str | None,
    unsaved_spec: str | None,
) -> int:
    """Propose an eval dataset for a user agent.

    Exactly one of ``agent_id`` (a saved user agent — the row provides
    the spec) or ``unsaved_spec`` (a JSON object carrying the in-flight
    create-modal fields) must be set.

    Emits the :class:`DatasetSuggestion` payload. When the agent is
    already on disk with an existing dataset, the suggestion's
    ``dataset_yaml`` is the merged YAML (existing cases preserved +
    new non-colliding cases appended).

    sensitivity_tier: 1
    """
    try:
        _ensure_bootstrap()
        if (not agent_id and not unsaved_spec) or (agent_id and unsaved_spec):
            return _emit_error(
                "exactly one of --agent-id or --unsaved-spec is required",
            )

        from src.agents.core.registry import get_agent
        from src.agents.dataset_creator import (
            DatasetCreatorAgent,
            DatasetCreatorInput,
            existing_case_names_from_yaml,
            read_existing_user_dataset,
        )

        existing_yaml: str | None = None
        existing_names: tuple[str, ...] = ()

        if agent_id is not None:
            if not agent_id.startswith("user."):
                return _emit_error(
                    "only user-authored agents accept dataset suggestions",
                )
            definition = get_agent(agent_id)
            if definition is None:
                return _emit_error(f"unknown agent: {agent_id}")
            default = definition.default_config
            deps = DatasetCreatorInput(
                name=definition.name,
                description=definition.description,
                system_prompt=default.system_prompt,
                max_sensitivity_tier=definition.max_sensitivity_tier,
                agent_id=agent_id,
                output_schema=definition.output_schema,
                available_tools=tuple(definition.available_tools),
            )
            existing_yaml = read_existing_user_dataset(agent_id)
            existing_names = existing_case_names_from_yaml(existing_yaml)
            deps = _deps_with_existing_names(deps, existing_names)
        else:
            try:
                payload = json.loads(unsaved_spec or "")
            except json.JSONDecodeError as exc:
                return _emit_error(f"bad unsaved-spec JSON: {exc}")
            if not isinstance(payload, dict):
                return _emit_error("unsaved-spec must be a JSON object")
            name = str(payload.get("name") or "").strip()
            description = str(payload.get("description") or "")
            system_prompt = str(payload.get("system_prompt") or "")
            if not name:
                return _emit_error("unsaved-spec: name is required")
            if not system_prompt:
                return _emit_error(
                    "unsaved-spec: system_prompt is required",
                )
            deps = DatasetCreatorInput(
                name=name,
                description=description,
                system_prompt=system_prompt,
                max_sensitivity_tier=int(
                    payload.get("max_sensitivity_tier") or 2,
                ),
                agent_id=None,
                output_schema=payload.get("output_schema") or None,
                available_tools=tuple(payload.get("available_tools") or ()),
            )

        suggestion = DatasetCreatorAgent().suggest(
            deps, existing_yaml=existing_yaml,
        )

        return _emit({
            "suggestion": suggestion.model_dump(),
            "existing_case_names": list(existing_names),
            "has_existing_dataset": existing_yaml is not None,
        })
    except Exception as exc:  # noqa: BLE001
        logger.exception("agents-suggest-dataset failed")
        return _emit_error(str(exc))


def _deps_with_existing_names(deps: Any, names: tuple[str, ...]) -> Any:
    """Return a copy of ``deps`` with ``existing_case_names`` populated.

    Kept as a tiny helper to avoid leaking the dataclass-copy idiom
    into the handler body.

    sensitivity_tier: 1
    """
    from src.agents.dataset_creator import DatasetCreatorInput

    return DatasetCreatorInput(
        name=deps.name,
        description=deps.description,
        system_prompt=deps.system_prompt,
        max_sensitivity_tier=deps.max_sensitivity_tier,
        agent_id=deps.agent_id,
        output_schema=deps.output_schema,
        available_tools=deps.available_tools,
        existing_case_names=names,
    )


# ---------------------------------------------------------------------------
# Model picker
# ---------------------------------------------------------------------------


def cmd_agents_suggest_model(unsaved_spec: str) -> int:
    """Recommend best-overall + cost-effective models for an agent spec.

    Always takes the unsaved spec from the wizard / edit row — the
    recommendation depends on live form values, not the saved row.
    Fetches the live ``/models`` lists for both ``remote`` and
    ``local`` routes so the LLM only picks ids the user can actually
    use. A failure on either endpoint is non-fatal (empty list) so the
    other route can still be recommended.

    The payload may also carry iteration feedback from earlier
    suggest → use → eval loops:

    - ``excluded_models`` — ids the picker must skip (already tested)
    - ``prior_attempts`` — list of ``{model_id, route, failed_cases:
      [{name, evaluator, reason}, ...]}`` records that teach the
      picker which capability gap to close on the next round.

    Emits the :class:`ModelRecommendation` payload.

    sensitivity_tier: 1
    """
    try:
        _ensure_bootstrap()
        from src.agents.core.model_factory import list_models
        from src.agents.model_picker import (
            FailedCase,
            ModelPickerAgent,
            ModelPickerInput,
            PriorAttempt,
        )

        try:
            payload = json.loads(unsaved_spec or "")
        except json.JSONDecodeError as exc:
            return _emit_error(f"bad unsaved-spec JSON: {exc}")
        if not isinstance(payload, dict):
            return _emit_error("unsaved-spec must be a JSON object")
        name = str(payload.get("name") or "").strip()
        description = str(payload.get("description") or "")
        system_prompt = str(payload.get("system_prompt") or "")
        if not name:
            return _emit_error("unsaved-spec: name is required")
        if not system_prompt:
            return _emit_error("unsaved-spec: system_prompt is required")

        remote_models: tuple[str, ...] = ()
        local_models: tuple[str, ...] = ()
        try:
            remote_models = tuple(list_models("remote"))
        except Exception:  # noqa: BLE001
            logger.exception("list_models(remote) failed; continuing")
        try:
            local_models = tuple(list_models("local"))
        except Exception:  # noqa: BLE001
            logger.exception("list_models(local) failed; continuing")

        excluded_models = tuple(
            str(m) for m in (payload.get("excluded_models") or ())
            if isinstance(m, str) and m
        )
        prior_attempts: list[PriorAttempt] = []
        for raw in payload.get("prior_attempts") or ():
            if not isinstance(raw, dict):
                continue
            mid = raw.get("model_id")
            route = raw.get("route")
            if not isinstance(mid, str) or not isinstance(route, str):
                continue
            failed_cases = tuple(
                FailedCase(
                    name=str(fc.get("name") or ""),
                    evaluator=str(fc.get("evaluator") or ""),
                    reason=str(fc.get("reason") or ""),
                )
                for fc in (raw.get("failed_cases") or ())
                if isinstance(fc, dict)
            )
            prior_attempts.append(PriorAttempt(
                model_id=mid,
                route=route,
                failed_cases=failed_cases,
            ))

        deps = ModelPickerInput(
            name=name,
            description=description,
            system_prompt=system_prompt,
            max_sensitivity_tier=int(
                payload.get("max_sensitivity_tier") or 2,
            ),
            available_remote_models=remote_models,
            available_local_models=local_models,
            output_schema=payload.get("output_schema") or None,
            enabled_skills=tuple(payload.get("enabled_skills") or ()),
            enabled_mcp_tools=tuple(payload.get("enabled_mcp_tools") or ()),
            agent_id=payload.get("agent_id") or None,
            excluded_models=excluded_models,
            prior_attempts=tuple(prior_attempts),
        )

        recommendation = ModelPickerAgent().recommend(deps)
        return _emit({
            "recommendation": recommendation.model_dump(),
            "available_remote_models": list(remote_models),
            "available_local_models": list(local_models),
        })
    except Exception as exc:  # noqa: BLE001
        logger.exception("agents-suggest-model failed")
        return _emit_error(str(exc))


def cmd_agents_suggest_prompt_improvements(unsaved_spec: str) -> int:
    """Rewrite a user agent's system prompt + description.

    Always takes the unsaved spec from the wizard / edit row — the
    recommendation depends on live form values, not the saved row.
    Emits the :class:`PromptSuggestion` payload (full rewrite +
    surgical additions + categorised improvements + change summary).

    sensitivity_tier: 1
    """
    try:
        _ensure_bootstrap()
        from src.agents.prompt_engineer import (
            EvalFailure,
            PromptEngineerAgent,
            PromptEngineerInput,
        )

        try:
            payload = json.loads(unsaved_spec or "")
        except json.JSONDecodeError as exc:
            return _emit_error(f"bad unsaved-spec JSON: {exc}")
        if not isinstance(payload, dict):
            return _emit_error("unsaved-spec must be a JSON object")
        name = str(payload.get("name") or "").strip()
        description = str(payload.get("description") or "")
        system_prompt = str(payload.get("system_prompt") or "")
        if not name:
            return _emit_error("unsaved-spec: name is required")
        if not system_prompt:
            return _emit_error("unsaved-spec: system_prompt is required")

        prior_eval_failures: list[EvalFailure] = []
        for raw in payload.get("prior_eval_failures") or ():
            if not isinstance(raw, dict):
                continue
            prior_eval_failures.append(EvalFailure(
                name=str(raw.get("name") or ""),
                evaluator=str(raw.get("evaluator") or ""),
                reason=str(raw.get("reason") or ""),
            ))

        deps = PromptEngineerInput(
            name=name,
            description=description,
            system_prompt=system_prompt,
            max_sensitivity_tier=int(
                payload.get("max_sensitivity_tier") or 2,
            ),
            agent_id=payload.get("agent_id") or None,
            output_schema=payload.get("output_schema") or None,
            available_tools=tuple(payload.get("available_tools") or ()),
            available_skills=tuple(payload.get("available_skills") or ()),
            enabled_mcp_tools=tuple(payload.get("enabled_mcp_tools") or ()),
            has_dataset=bool(payload.get("has_dataset", False)),
            prior_eval_failures=tuple(prior_eval_failures),
        )

        suggestion = PromptEngineerAgent().suggest(deps)
        return _emit({"suggestion": suggestion.model_dump()})
    except Exception as exc:  # noqa: BLE001
        logger.exception("agents-suggest-prompt-improvements failed")
        return _emit_error(str(exc))


# ---------------------------------------------------------------------------
# User agents
# ---------------------------------------------------------------------------


def _user_agent_upsert_from_payload(
    payload: dict[str, Any],
) -> Any:
    from src.agents.user_agents.store import UserAgentUpsert

    return UserAgentUpsert(
        name=str(payload.get("name") or "").strip(),
        description=str(payload.get("description") or ""),
        system_prompt=str(payload.get("system_prompt") or ""),
        model_route=str(payload.get("model_route") or "inherit"),
        model_override=(
            str(payload["model_override"])
            if payload.get("model_override")
            else None
        ),
        enabled_skills=tuple(payload.get("enabled_skills") or []),
        enabled_mcp_tools=tuple(payload.get("enabled_mcp_tools") or []),
        brain_access=bool(payload.get("brain_access", True)),
        max_sensitivity_tier=int(payload.get("max_sensitivity_tier") or 2),
        schedule_cron=(
            str(payload["schedule_cron"])
            if payload.get("schedule_cron")
            else None
        ),
        schedule_enabled=bool(payload.get("schedule_enabled", False)),
        pattern=str(payload.get("pattern") or "single"),
        subagents=tuple(payload.get("subagents") or []),
        delivery_tools=tuple(payload.get("delivery_tools") or []),
    )


def _catalog_tool_index() -> dict[str, str]:
    """Map ``"connector_id:tool_name"`` → ``tool_type`` for every catalog tool.

    Returns ``{}`` when the catalog cannot be loaded so validation
    short-circuits to "no checks" rather than failing the caller.

    sensitivity_tier: 1
    """
    try:
        from src.extensions.connectors.catalog import ConnectorCatalog
    except Exception:  # noqa: BLE001
        return {}
    try:
        catalog = ConnectorCatalog()
    except Exception:  # noqa: BLE001
        return {}
    index: dict[str, str] = {}
    for template in catalog.all:
        for tool in template.tools:
            if tool.tool_type in ("action", "data"):
                index[f"{template.id}:{tool.tool_name}"] = tool.tool_type
    return index


def _validate_user_agent_upsert(
    upsert: Any,
    *,
    self_agent_id: str | None = None,
) -> str | None:
    """Return an error message if the upsert is invalid, else ``None``.

    Enforces:
    * pattern is ``"single"`` or ``"orchestrator"``;
    * orchestrators require at least one subagent and may not delegate
      to themselves; every subagent must be a registered single agent.
    * every ``enabled_mcp_tools`` entry is a known
      ``connector_id:tool_name`` whose catalog type is ``"data"`` or
      ``"action"``.
    * every ``delivery_tools`` entry is a known catalog ``"action"``
      tool. Delivery tools may live outside ``enabled_mcp_tools`` —
      delivery is independent of LLM-callability by design.

    sensitivity_tier: 1
    """
    tool_index = _catalog_tool_index()
    if tool_index:
        for tool_id in upsert.enabled_mcp_tools:
            if ":" not in str(tool_id):
                return (
                    f"invalid mcp tool id {tool_id!r}; "
                    "expected 'connector_id:tool_name'"
                )
            if tool_id not in tool_index:
                return (
                    f"mcp tool {tool_id!r} not found in the connector "
                    "catalog"
                )
        for tool_id in upsert.delivery_tools:
            if ":" not in str(tool_id):
                return (
                    f"invalid delivery tool id {tool_id!r}; "
                    "expected 'connector_id:tool_name'"
                )
            kind = tool_index.get(tool_id)
            if kind is None:
                return (
                    f"delivery tool {tool_id!r} not found in the "
                    "connector catalog"
                )
            if kind != "action":
                return (
                    f"delivery tool {tool_id!r} must be an action tool "
                    f"(catalog type is {kind!r})"
                )
    pattern = upsert.pattern
    if pattern not in ("single", "orchestrator"):
        return (
            f"invalid pattern {pattern!r}; "
            "supported values are 'single' and 'orchestrator'"
        )
    if pattern == "single":
        if upsert.subagents:
            return (
                "subagents may only be set when pattern is 'orchestrator'"
            )
        return None
    if not upsert.subagents:
        return "orchestrator requires at least one subagent"
    if self_agent_id is not None and self_agent_id in upsert.subagents:
        return (
            f"orchestrator may not delegate to itself ({self_agent_id})"
        )
    from src.agents.core.registry import get_agent

    seen: set[str] = set()
    for sub_id in upsert.subagents:
        if sub_id in seen:
            return f"subagent {sub_id!r} listed more than once"
        seen.add(sub_id)
        definition = get_agent(sub_id)
        if definition is None:
            return f"subagent {sub_id!r} is not registered"
        if definition.pattern != "single":
            return (
                f"subagent {sub_id!r} has pattern "
                f"{definition.pattern!r}; only single-pattern agents "
                "may be delegated to in v1"
            )
    return None


def cmd_agents_create(payload_json: str) -> int:
    """Create a new user-authored agent and mount it in the registry.

    Refuses payloads with empty ``name`` or ``system_prompt``.

    sensitivity_tier: 1
    """
    try:
        _ensure_bootstrap()
        from src.agents.user_agents.registration import (
            register_one_user_agent,
        )
        from src.agents.user_agents.store import UserAgentStore

        try:
            payload = json.loads(payload_json)
        except json.JSONDecodeError as exc:
            return _emit_error(f"bad payload JSON: {exc}")
        if not isinstance(payload, dict):
            return _emit_error("payload must be a JSON object")
        upsert = _user_agent_upsert_from_payload(payload)
        if not upsert.name:
            return _emit_error("name is required")
        if not upsert.system_prompt:
            return _emit_error("system_prompt is required")
        # OSS: user agents inherit the user's configured Ollama model
        # (settings.json `llm_model`). Pro can attach a per-agent tier
        # default by setting `model_override` before calling here.
        validation_err = _validate_user_agent_upsert(upsert)
        if validation_err:
            return _emit_error(validation_err)

        store = UserAgentStore()
        try:
            row = store.insert(upsert)
        finally:
            store.close()
        register_one_user_agent(row)

        if payload.get("skip_backfill"):
            from src.agents.user_agents.runner import (
                data_tool_ids_for_row,
                seed_existing_items,
            )
            from src.core.data_layer import DataLayer

            data_tools = data_tool_ids_for_row(row)
            if data_tools:
                layer = DataLayer()
                seed_existing_items(layer.duckdb, row.agent_id, data_tools)

        from src.agents.core.registry import get_agent

        definition = get_agent(row.agent_id)
        cfg_store = _config_store()
        return _emit({
            "agent": _serialize_definition(definition, store=cfg_store),
            "user_row": row.to_dict(),
        })
    except Exception as exc:  # noqa: BLE001
        logger.exception("agents-create failed")
        return _emit_error(str(exc))


def cmd_agents_user_update(agent_id: str, payload_json: str) -> int:
    """Update a user-authored agent in the SQLite store.

    Unlike :func:`cmd_agents_update` (which only patches the
    ``agent_configs`` override), this writes the full row so the
    schedule + Brain-access toggles + skill / MCP tool selection
    survive a process restart.

    sensitivity_tier: 1
    """
    try:
        _ensure_bootstrap()
        from src.agents.user_agents.registration import (
            register_one_user_agent,
        )
        from src.agents.user_agents.store import UserAgentStore

        if not agent_id.startswith("user."):
            return _emit_error("only user-authored agents are editable here")
        try:
            payload = json.loads(payload_json)
        except json.JSONDecodeError as exc:
            return _emit_error(f"bad payload JSON: {exc}")
        if not isinstance(payload, dict):
            return _emit_error("payload must be a JSON object")
        upsert = _user_agent_upsert_from_payload(payload)
        if not upsert.name:
            return _emit_error("name is required")
        if not upsert.system_prompt:
            return _emit_error("system_prompt is required")
        validation_err = _validate_user_agent_upsert(
            upsert, self_agent_id=agent_id,
        )
        if validation_err:
            return _emit_error(validation_err)
        store = UserAgentStore()
        try:
            row = store.update(agent_id, upsert)
        except KeyError as exc:
            store.close()
            return _emit_error(str(exc))
        finally:
            store.close()
        register_one_user_agent(row)
        from src.agents.core.registry import get_agent

        definition = get_agent(agent_id)
        cfg_store = _config_store()
        return _emit({
            "agent": _serialize_definition(definition, store=cfg_store),
            "user_row": row.to_dict(),
        })
    except Exception as exc:  # noqa: BLE001
        logger.exception("agents-user-update failed")
        return _emit_error(str(exc))


def cmd_agents_user_apply_prompt_edit(
    agent_id: str, payload_json: str,
) -> int:
    """Apply a prompt-engineer rewrite to a user-authored agent.

    Snapshots the current ``system_prompt`` + ``description`` into
    the ``pre_ai_*`` columns, writes the new values to
    ``user_agents``, and mirrors the new ``system_prompt`` into the
    ``agent_configs`` overlay so the editor reads the same value the
    registry will serve at runtime. Re-registers the agent.

    ``payload_json`` is a JSON object with ``system_prompt`` and
    ``description``. Both are required and must be non-empty.

    Refuses for built-in agents (any id not starting with ``user.``).

    sensitivity_tier: 1
    """
    try:
        _ensure_bootstrap()
        from src.agents.core.config_store import AgentConfigStoreError
        from src.agents.user_agents.registration import (
            register_one_user_agent,
        )
        from src.agents.user_agents.store import (
            UserAgentStore,
            UserAgentUpsert,
        )

        if not agent_id.startswith("user."):
            return _emit_error(
                "only user-authored agents can be edited by the "
                "prompt engineer",
            )
        try:
            payload = json.loads(payload_json or "")
        except json.JSONDecodeError as exc:
            return _emit_error(f"bad payload JSON: {exc}")
        if not isinstance(payload, dict):
            return _emit_error("payload must be a JSON object")
        new_prompt = str(payload.get("system_prompt") or "").strip()
        new_description = str(payload.get("description") or "").strip()
        if not new_prompt:
            return _emit_error("system_prompt is required")
        if not new_description:
            return _emit_error("description is required")

        store = UserAgentStore()
        try:
            existing = store.get(agent_id)
            if existing is None:
                return _emit_error(f"unknown user agent: {agent_id!r}")
            store.snapshot_pre_ai_edit(
                agent_id,
                prev_system_prompt=existing.system_prompt,
                prev_description=existing.description,
            )
            row = store.update(agent_id, UserAgentUpsert(
                name=existing.name,
                description=new_description,
                system_prompt=new_prompt,
                model_route=existing.model_route,
                model_override=existing.model_override,
                enabled_skills=existing.enabled_skills,
                enabled_mcp_tools=existing.enabled_mcp_tools,
                brain_access=existing.brain_access,
                max_sensitivity_tier=existing.max_sensitivity_tier,
                schedule_cron=existing.schedule_cron,
                schedule_enabled=existing.schedule_enabled,
                pattern=existing.pattern,
                subagents=existing.subagents,
                delivery_tools=existing.delivery_tools,
            ))
        finally:
            store.close()
        register_one_user_agent(row)
        from src.agents.core.registry import get_agent

        definition = get_agent(agent_id)
        cfg_store = _config_store()
        # Mirror the new prompt into the override overlay so the editor
        # and the registry serve identical values. Skip silently if the
        # config store refuses (e.g. when the agent is somehow locked).
        try:
            cfg_store.update(
                agent_id,
                default=definition.default_config,
                patch={"system_prompt": new_prompt},
            )
        except AgentConfigStoreError:
            logger.warning(
                "apply-prompt-edit: could not sync overlay for %s",
                agent_id,
            )
        return _emit({
            "agent": _serialize_definition(definition, store=cfg_store),
            "user_row": row.to_dict(),
        })
    except Exception as exc:  # noqa: BLE001
        logger.exception("agents-user-apply-prompt-edit failed")
        return _emit_error(str(exc))


def cmd_agents_user_revert_ai_edit(agent_id: str) -> int:
    """Restore the pre-AI snapshot for a user-authored agent.

    Reads the snapshot stored at the most recent prompt-engineer apply,
    swaps it back into the live ``system_prompt`` + ``description``
    columns, and clears the snapshot slot. Also mirrors the restored
    ``system_prompt`` into the ``agent_configs`` overlay so the editor
    reads the reverted value. Returns the updated row so the UI can
    reset its local state.

    Refuses for built-in agents (any id not starting with ``user.``)
    and emits a friendly error when no snapshot is on file.

    sensitivity_tier: 1
    """
    try:
        _ensure_bootstrap()
        from src.agents.core.config_store import AgentConfigStoreError
        from src.agents.user_agents.registration import (
            register_one_user_agent,
        )
        from src.agents.user_agents.store import UserAgentStore

        if not agent_id.startswith("user."):
            return _emit_error("only user-authored agents are revertable")
        store = UserAgentStore()
        try:
            try:
                row = store.revert_pre_ai_snapshot(agent_id)
            except LookupError as exc:
                return _emit_error(str(exc))
            except KeyError as exc:
                return _emit_error(str(exc))
        finally:
            store.close()
        register_one_user_agent(row)
        from src.agents.core.registry import get_agent

        definition = get_agent(agent_id)
        cfg_store = _config_store()
        try:
            cfg_store.update(
                agent_id,
                default=definition.default_config,
                patch={"system_prompt": row.system_prompt},
            )
        except AgentConfigStoreError:
            logger.warning(
                "revert-ai-edit: could not sync overlay for %s",
                agent_id,
            )
        return _emit({
            "agent": _serialize_definition(definition, store=cfg_store),
            "user_row": row.to_dict(),
        })
    except Exception as exc:  # noqa: BLE001
        logger.exception("agents-user-revert-ai-edit failed")
        return _emit_error(str(exc))


def cmd_agents_delete(agent_id: str) -> int:
    """Remove a user-authored agent from SQLite and the registry.

    sensitivity_tier: 1
    """
    try:
        _ensure_bootstrap()
        from src.agents.user_agents.registration import (
            unregister_user_agent,
        )
        from src.agents.user_agents.store import UserAgentStore

        if not agent_id.startswith("user."):
            return _emit_error("only user-authored agents can be deleted")
        store = UserAgentStore()
        try:
            removed = store.delete(agent_id)
        finally:
            store.close()
        unregister_user_agent(agent_id)
        return _emit({"deleted": removed, "agent_id": agent_id})
    except Exception as exc:  # noqa: BLE001
        logger.exception("agents-delete failed")
        return _emit_error(str(exc))


def cmd_agents_set_schedule(
    agent_id: str,
    cron: str | None,
    enabled: bool,
) -> int:
    """Persist the schedule cron + enabled flag for a user agent.

    Source / callable / delivery tool selection has moved into the
    main row update (``enabled_mcp_tools`` + ``delivery_tools``); this
    handler is now a pure cron+enabled write.

    sensitivity_tier: 1
    """
    try:
        _ensure_bootstrap()
        from src.agents.user_agents.store import UserAgentStore

        if not agent_id.startswith("user."):
            return _emit_error(
                "only user-authored agents can be scheduled here",
            )
        store = UserAgentStore()
        try:
            row = store.set_schedule(
                agent_id, cron=cron, enabled=enabled,
            )
        except KeyError as exc:
            store.close()
            return _emit_error(str(exc))
        finally:
            store.close()
        return _emit({"user_row": row.to_dict()})
    except Exception as exc:  # noqa: BLE001
        logger.exception("agents-set-schedule failed")
        return _emit_error(str(exc))


def cmd_agents_run_now(agent_id: str) -> int:
    """Invoke a user agent immediately, regardless of schedule.

    Agents whose ``enabled_mcp_tools`` includes at least one catalog
    ``data`` tool take the batch path (one LLM call per unread item);
    others fall through to the generic Portuguese trigger.

    sensitivity_tier: 2
    """
    try:
        _ensure_bootstrap()
        from src.agents.user_agents.runner import (
            data_tool_ids_for_row,
            run_user_agent_batch,
            run_user_agent_generic,
        )
        from src.agents.user_agents.store import UserAgentStore
        from src.core.data_layer import DataLayer

        if not agent_id.startswith("user."):
            return _emit_error(
                "only user-authored agents can be run here",
            )

        store = UserAgentStore()
        try:
            row = store.get(agent_id)
        finally:
            store.close()
        if row is None:
            return _emit_error(f"unknown user agent: {agent_id!r}")

        layer = DataLayer()
        if data_tool_ids_for_row(row):
            summary = run_user_agent_batch(layer, agent_id)
        else:
            summary = run_user_agent_generic(layer, agent_id)

        # Stamp the schedule-state file so the "Last run" surface in
        # the UI moves immediately — keeps run-now and the cron tick on
        # the same source of truth.
        try:
            _stamp_last_run(agent_id)
        except Exception:  # noqa: BLE001
            logger.debug("Failed to stamp last_run after run-now",
                         exc_info=True)

        return _emit({"summary": summary.to_dict()})
    except Exception as exc:  # noqa: BLE001
        logger.exception("agents-run-now failed")
        return _emit_error(str(exc))


def cmd_agents_user_status(agent_id: str) -> int:
    """Emit the scheduling/runtime status for the Agents page strip.

    sensitivity_tier: 1
    """
    try:
        _ensure_bootstrap()
        from src.agents.user_agents.runner import get_user_agent_status
        from src.core.data_layer import DataLayer

        if not agent_id.startswith("user."):
            return _emit_error(
                "only user-authored agents are queryable here",
            )

        layer = DataLayer()
        status = get_user_agent_status(layer, agent_id)
        if status.get("error"):
            return _emit_error(str(status["error"]))
        return _emit({"status": status})
    except Exception as exc:  # noqa: BLE001
        logger.exception("agents-user-status failed")
        return _emit_error(str(exc))


def _stamp_last_run(agent_id: str) -> None:
    """Write ``now`` into ``agent_schedule_state.json`` for ``agent_id``.

    Mirrors the persistence side of ``cmd_run_scheduled_agents`` so a
    manual Run-now click updates the same UI-facing "last run" surface.

    sensitivity_tier: 1
    """
    from datetime import datetime, timezone

    state_path = (
        Path.home() / ".arandu" / "data" / "agent_schedule_state.json"
    )
    state: dict[str, str] = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            state = {}
    state[agent_id] = datetime.now(tz=timezone.utc).isoformat()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def cmd_agents_list_mcp_tools() -> int:
    """List every data + action tool exposed by enabled connectors.

    The single picker on the Agents page groups these by ``connector_id``
    and uses ``tool_type`` to decide which row (sources / callable /
    delivery) each chip belongs to. ``target_table`` is surfaced for
    data tools so the runner can map a source binding to a SQLite
    table without consulting a second source of truth.

    sensitivity_tier: 1
    """
    try:
        from src.agents.tool_registry import ToolRegistry
        from src.extensions.connectors.catalog import ConnectorCatalog
        from src.extensions.connectors.registry import ExtensionRegistry

        catalog = ConnectorCatalog()
        registry = ExtensionRegistry()
        tools = ToolRegistry(catalog, registry).get_available_tools()
        return _emit({
            "tools": [
                {
                    "connector_id": t.connector_id,
                    "connector_name": t.connector_name,
                    "tool_name": t.tool_name,
                    "display_name": t.display_name,
                    "description": t.description,
                    "tool_type": t.tool_type,
                    "target_table": t.target_table,
                    "input_schema": t.input_schema or {},
                }
                for t in tools
            ],
        })
    except Exception as exc:  # noqa: BLE001
        logger.exception("agents-list-mcp-tools failed")
        return _emit_error(str(exc))


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------


def _user_skill_upsert_from_payload(
    payload: dict[str, Any],
) -> Any:
    from src.agents.user_agents.skill_store import UserSkillUpsert

    params = payload.get("parameters") or {}
    if not isinstance(params, dict):
        params = {}
    return UserSkillUpsert(
        name=str(payload.get("name") or "").strip(),
        description=str(payload.get("description") or ""),
        category=str(payload.get("category") or "general"),
        prompt_template=str(payload.get("prompt_template") or ""),
        parameters={str(k): str(v) for k, v in params.items()},
        uses_llm=bool(payload.get("uses_llm", True)),
    )


def cmd_skills_create(payload_json: str) -> int:
    """Create a new user-authored skill.

    sensitivity_tier: 1
    """
    try:
        from src.agents.user_agents.skill_store import UserSkillStore

        try:
            payload = json.loads(payload_json)
        except json.JSONDecodeError as exc:
            return _emit_error(f"bad payload JSON: {exc}")
        if not isinstance(payload, dict):
            return _emit_error("payload must be a JSON object")
        upsert = _user_skill_upsert_from_payload(payload)
        if not upsert.name:
            return _emit_error("name is required")
        if not upsert.prompt_template:
            return _emit_error("prompt_template is required")
        store = UserSkillStore()
        try:
            row = store.insert(upsert)
        finally:
            store.close()
        return _emit({"skill": row.to_dict()})
    except Exception as exc:  # noqa: BLE001
        logger.exception("skills-create failed")
        return _emit_error(str(exc))


def cmd_skills_update(skill_id: str, payload_json: str) -> int:
    """Update an existing user-authored skill.

    sensitivity_tier: 1
    """
    try:
        from src.agents.user_agents.skill_store import UserSkillStore

        if not skill_id.startswith("user."):
            return _emit_error("only user-authored skills are editable")
        try:
            payload = json.loads(payload_json)
        except json.JSONDecodeError as exc:
            return _emit_error(f"bad payload JSON: {exc}")
        if not isinstance(payload, dict):
            return _emit_error("payload must be a JSON object")
        upsert = _user_skill_upsert_from_payload(payload)
        if not upsert.name:
            return _emit_error("name is required")
        if not upsert.prompt_template:
            return _emit_error("prompt_template is required")
        store = UserSkillStore()
        try:
            row = store.update(skill_id, upsert)
        except KeyError as exc:
            store.close()
            return _emit_error(str(exc))
        finally:
            store.close()
        return _emit({"skill": row.to_dict()})
    except Exception as exc:  # noqa: BLE001
        logger.exception("skills-update failed")
        return _emit_error(str(exc))


def cmd_skills_delete(skill_id: str) -> int:
    """Remove a user-authored skill.

    sensitivity_tier: 1
    """
    try:
        from src.agents.user_agents.skill_store import UserSkillStore

        if not skill_id.startswith("user."):
            return _emit_error("only user-authored skills can be deleted")
        store = UserSkillStore()
        try:
            removed = store.delete(skill_id)
        finally:
            store.close()
        return _emit({"deleted": removed, "skill_id": skill_id})
    except Exception as exc:  # noqa: BLE001
        logger.exception("skills-delete failed")
        return _emit_error(str(exc))


def cmd_skills_get(skill_id: str) -> int:
    """Return one skill's metadata + template.

    Built-in skills expose their (constant) ``execute_fn`` docstring as
    the ``prompt_template`` so the UI can render a read-only inspect
    panel. User skills expose the stored template directly.

    sensitivity_tier: 1
    """
    try:
        from src.agent_runtime.skills import SkillRegistry
        from src.agents.user_agents.skill_store import UserSkillStore

        if skill_id.startswith("user."):
            store = UserSkillStore()
            try:
                row = store.get(skill_id)
            finally:
                store.close()
            if row is None:
                return _emit_error(f"unknown skill: {skill_id}")
            return _emit({"skill": {**row.to_dict(), "builtin": False}})

        registry = SkillRegistry()
        registry.register_builtin_skills()
        skill = registry.get(skill_id)
        if skill is None:
            return _emit_error(f"unknown skill: {skill_id}")
        return _emit({
            "skill": {
                "skill_id": skill.id,
                "name": skill.name,
                "description": skill.description,
                "category": skill.category,
                "prompt_template": (skill.execute_fn.__doc__ or "").strip(),
                "parameters": dict(skill.parameters),
                "uses_llm": skill.uses_llm,
                "builtin": True,
            },
        })
    except Exception as exc:  # noqa: BLE001
        logger.exception("skills-get failed")
        return _emit_error(str(exc))


# ---------------------------------------------------------------------------
# Skills v2 — SKILL.md-based commands
# ---------------------------------------------------------------------------


def cmd_skills_list_v2() -> int:
    """List all SKILL.md-based skills (L1 metadata).

    sensitivity_tier: 1
    """
    try:
        from src.agent_runtime.skill_loader import SkillLoader

        loader = SkillLoader()
        skills = loader.discover()
        return _emit([s.to_dict() for s in skills])
    except Exception as exc:  # noqa: BLE001
        logger.exception("skills-list-v2 failed")
        return _emit_error(str(exc))


def cmd_skills_get_v2(skill_id: str) -> int:
    """Return one skill's full SKILL.md content (L2).

    sensitivity_tier: 1
    """
    try:
        from src.agent_runtime.skill_loader import SkillLoader

        loader = SkillLoader()
        doc = loader.load(skill_id)
        if doc is None:
            return _emit_error(f"unknown skill: {skill_id}")
        return _emit(doc.to_dict())
    except Exception as exc:  # noqa: BLE001
        logger.exception("skills-get-v2 failed")
        return _emit_error(str(exc))


def cmd_skills_create_v2(name: str, content: str) -> int:
    """Create a new SKILL.md-based skill.

    sensitivity_tier: 1
    """
    try:
        from src.agent_runtime.skill_loader import SkillLoader

        loader = SkillLoader()
        meta = loader.create(name, content)
        return _emit(meta.to_dict())
    except Exception as exc:  # noqa: BLE001
        logger.exception("skills-create-v2 failed")
        return _emit_error(str(exc))


def cmd_skills_update_v2(skill_id: str, content: str) -> int:
    """Update a SKILL.md-based skill's content.

    sensitivity_tier: 1
    """
    try:
        from src.agent_runtime.skill_loader import SkillLoader

        loader = SkillLoader()
        meta = loader.update(skill_id, content)
        return _emit(meta.to_dict())
    except KeyError:
        return _emit_error(f"unknown skill: {skill_id}")
    except Exception as exc:  # noqa: BLE001
        logger.exception("skills-update-v2 failed")
        return _emit_error(str(exc))


def cmd_skills_delete_v2(skill_id: str) -> int:
    """Delete a SKILL.md-based skill.

    sensitivity_tier: 1
    """
    try:
        from src.agent_runtime.skill_loader import SkillLoader

        loader = SkillLoader()
        removed = loader.delete(skill_id)
        return _emit({"deleted": removed, "skill_id": skill_id})
    except Exception as exc:  # noqa: BLE001
        logger.exception("skills-delete-v2 failed")
        return _emit_error(str(exc))


# ---------------------------------------------------------------------------
# Privacy mode toggle (local inference opt-in)
# ---------------------------------------------------------------------------


_SETTINGS_PATH = Path.home() / ".arandu" / "settings.json"


def _read_settings_dict() -> dict[str, Any]:
    """Read settings.json, returning ``{}`` on missing / unreadable file.

    sensitivity_tier: 1
    """
    if not _SETTINGS_PATH.exists():
        return {}
    try:
        with _SETTINGS_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_settings_dict(data: dict[str, Any]) -> None:
    """Persist ``data`` back to settings.json.

    sensitivity_tier: 1
    """
    _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _SETTINGS_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def cmd_set_local_inference_for_sensitive(enabled_flag: str) -> int:
    """Flip the ``local_inference_for_sensitive`` privacy mode.

    ``enabled_flag`` is the string ``"true"`` or ``"false"`` (the
    Tauri shell stringifies booleans).

    Behaviour:

    - ``enabled=false`` — persist the flag immediately, clear every
      row from ``agent_blocked``, reload the egress firewall, audit
      the toggle, and return ``{"status": "ok"}``.
    - ``enabled=true`` — run :func:`run_agent_eval` for every agent
      in ``AGENT_SUITE_MAP``. If any agent's status is not
      ``"passed"``, abort: the flag stays ``false`` and the response
      carries ``{"status": "eval_failed", "failures": [...]}``. On
      all-pass, persist the flag, reload the firewall, audit the
      toggle, and return ``{"status": "ok"}``.

    sensitivity_tier: 1
    """
    try:
        _ensure_bootstrap()
        from src.agents.core.agent_block_store import (
            default_agent_block_store,
        )
        from src.agents.core.audit import default_chain, hash_payload
        from src.agents.eval_runner import (
            AGENT_SUITE_MAP,
            EvalRunStore,
            run_agent_eval,
        )
        from src.agents.firewall.egress_firewall import (
            default_egress_firewall,
        )

        enabled = str(enabled_flag).strip().lower() in ("1", "true", "yes")
        settings = _read_settings_dict()

        if not enabled:
            settings["local_inference_for_sensitive"] = False
            _write_settings_dict(settings)
            default_agent_block_store().clear()
            default_egress_firewall().reload_policy()
            default_chain().append(
                event_type="local_inference_toggle",
                agent_id="firewall.egress",
                decision="disabled",
                payload_hash=hash_payload("disabled"),
                tier=1,
                extra={"enabled": False},
            )
            return _emit({"status": "ok", "enabled": False})

        # enabled = true — run the full eval suite first.
        store = EvalRunStore()
        failures: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []
        for agent_id in AGENT_SUITE_MAP:
            run = run_agent_eval(
                agent_id,
                trigger="local_inference_gate",
                store=store,
            )
            results.append({
                "agent_id": agent_id,
                "status": run.status,
                "cases_total": run.cases_total,
                "cases_passed": run.cases_passed,
                "cases_failed": run.cases_failed,
            })
            if run.status != "passed":
                failures.append({
                    "agent_id": agent_id,
                    "status": run.status,
                    "failed_cases": run.failed_cases,
                    "error": run.error,
                })

        if failures:
            return _emit({
                "status": "eval_failed",
                "enabled": False,
                "failures": failures,
                "results": results,
            })

        settings["local_inference_for_sensitive"] = True
        _write_settings_dict(settings)
        # Fresh start: nothing should be blocked when we just observed
        # every agent pass.
        default_agent_block_store().clear()
        default_egress_firewall().reload_policy()
        default_chain().append(
            event_type="local_inference_toggle",
            agent_id="firewall.egress",
            decision="enabled",
            payload_hash=hash_payload(json.dumps(results, sort_keys=True)),
            tier=1,
            extra={
                "enabled": True,
                "evals_run": len(results),
            },
        )
        return _emit({
            "status": "ok",
            "enabled": True,
            "results": results,
        })
    except Exception as exc:  # noqa: BLE001
        logger.exception("set-local-inference-for-sensitive failed")
        return _emit_error(str(exc))


__all__ = [
    "cmd_agents_activity",
    "cmd_agents_create",
    "cmd_agents_delete",
    "cmd_agents_eval_dataset",
    "cmd_agents_eval_status",
    "cmd_agents_get",
    "cmd_agents_list",
    "cmd_agents_list_mcp_tools",
    "cmd_agents_reset",
    "cmd_agents_run_eval",
    "cmd_agents_set_schedule",
    "cmd_agents_suggest_prompt_improvements",
    "cmd_agents_update",
    "cmd_agents_user_apply_prompt_edit",
    "cmd_agents_user_revert_ai_edit",
    "cmd_agents_user_update",
    "cmd_agents_validate_dataset",
    "cmd_set_local_inference_for_sensitive",
    "cmd_skills_create",
    "cmd_skills_delete",
    "cmd_skills_get",
    "cmd_skills_update",
]
