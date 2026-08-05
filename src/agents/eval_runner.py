"""Per-agent eval orchestration.

This module is the bridge between :mod:`evals` (datasets, evaluators)
and the agent registry / SQLite store. It owns:

- ``AGENT_SUITE_MAP`` — agent_id → dataset stem for every registered
  agent. Agents without a dataset are absent from the map; the runner
  records a ``skipped`` row when asked to evaluate them.
- :class:`EvalRunStore` — persists every run to
  ``agent_eval_runs`` with the failed-case detail the UI renders.
- :func:`run_agent_eval` — synchronous entry point used by the CLI
  ``agents-run-eval`` handler.

Evals run only on explicit user action: the "Run eval" button in
the Agents page (manual trigger) or the ``evals.run_evals`` batch
CLI. Automatic evals on edit/save were removed in 0.5.0 so the judge
does not run on every settings tweak.

sensitivity_tier: 1
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent id → dataset stem
# ---------------------------------------------------------------------------

# Datasets live in ``evals/datasets/<stem>.yaml``. Every agent in the
# registry that ships a dataset must appear here. Agents whose work
# requires a real LLM still appear — the runner records ``skipped``
# when no model is reachable.
AGENT_SUITE_MAP: dict[str, str] = {
    # Locked, manual-only
    "brain": "brain_qa",
    "chat": "chat_qa",
    "firewall.injection": "firewall_prompts",
    "firewall.injection.scan": "injection_scan",
    "firewall.egress": "egress_routing",
    # Direct sub-agents
    "sensitivity": "sensitivity",
    "labeler": "labeler",
    "triage": "triage",
    "fact_extractor": "fact_extractor",
    "message_evaluator": "message_eval",
    "pending_reply": "pending_reply",
    "contact_context": "contact_context",
    "actionable_events": "actionable_events",
    # Indirect sub-agents
    "query_router": "query_router",
    "topic_extractor": "topic_extractor",
    "schema_discovery": "schema_discovery",
    "model_generator": "model_generator",
    "relationship_tracker": "relationship_tracker",
    "dataset_validator": "dataset_validator",
    "dataset_creator": "dataset_creator",
    "prompt_engineer": "prompt_engineer",
    # Goals + habits planner
    "goal_extractor": "goal_extractor",
    "habit_suggester": "habit_suggester",
    # Skill system
    "skill_creator": "skill_creator",
    # Action proposal + judging
    "action_proposal_judge": "action_proposal_judge",
    "event_categorizer": "event_categorizer",
    # Reflection + scheduling + task lifecycle
    "reflector": "reflector",
    "daily_scheduler": "daily_scheduler",
    "model_picker": "model_picker",
    "task_proposer": "task_proposer",
    "task_curator": "task_curator",
    "task_completion": "task_completion",
}

# Agents that should not auto-run on edit (brain + firewalls). They
# also can't be edited via the IPC, but ``run_agent_eval`` still
# accepts them when triggered manually from the Agents page.
MANUAL_ONLY_AGENTS: frozenset[str] = frozenset({
    "brain", "chat",
    "firewall.injection", "firewall.injection.scan", "firewall.egress",
})


# ---------------------------------------------------------------------------
# Row + store
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalRun:
    """One persisted eval row, as the UI consumes it.

    sensitivity_tier: 1
    """

    run_id: str
    agent_id: str
    suite: str | None
    trigger: str
    started_at: str
    finished_at: str | None
    status: str
    cases_total: int
    cases_passed: int
    cases_failed: int
    failed_cases: list[dict[str, Any]]
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "agent_id": self.agent_id,
            "suite": self.suite,
            "trigger": self.trigger,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "cases_total": self.cases_total,
            "cases_passed": self.cases_passed,
            "cases_failed": self.cases_failed,
            "failed_cases": self.failed_cases,
            "error": self.error,
        }


DEFAULT_DB_PATH = (
    Path.home() / ".arandu" / "data" / "arandu.sqlite3"
)

SUITE_TO_AGENT: dict[str, str] = {
    suite: agent for agent, suite in AGENT_SUITE_MAP.items()
}


_REGISTRY_READY = False


def _ensure_registry() -> None:
    """Populate the agent registry once per process.

    A bare ``python -m evals.run_evals`` imports nothing that registers
    agents, so the registry is empty and every fingerprint would come
    back ``None`` — the gate would never fire. ``bootstrap_agents`` is
    idempotent and makes no LLM calls.

    sensitivity_tier: 1
    """
    global _REGISTRY_READY
    if _REGISTRY_READY:
        return
    try:
        from src.agents.brain.v2 import bootstrap_agents

        bootstrap_agents()
    except Exception:  # noqa: BLE001 - fingerprinting is best-effort
        logger.debug("agent registry bootstrap failed", exc_info=True)
    _REGISTRY_READY = True


def suite_fingerprint(suite: str, dataset_path: Path) -> str | None:
    """Hash the three inputs that can change a suite's result.

    Those are the **model** serving the agent, its **system prompt**,
    and the **dataset**. If none moved, re-running spends tokens to
    reproduce a known answer — a full run is ~480 LLM calls, and the
    suite agents are exactly the ones on the premium tier.

    Returns ``None`` when any input can't be resolved (no registry, no
    config store, unreadable dataset). Callers must treat ``None`` as
    "cannot prove it's unchanged" and run the suite.

    sensitivity_tier: 1
    """
    agent_id = SUITE_TO_AGENT.get(suite)
    if agent_id is None:
        return None
    try:
        dataset = dataset_path.read_bytes()
    except OSError:
        return None
    try:
        from src.agents.core.config_store import current_model_override
        from src.agents.core.registry import get_agent
    except Exception:  # noqa: BLE001
        return None

    _ensure_registry()
    definition = get_agent(agent_id)
    if definition is None:
        return None
    prompt = definition.default_config.system_prompt or ""
    try:
        model = current_model_override(agent_id) or ""
    except Exception:  # noqa: BLE001
        model = ""
    if not model:
        # No per-agent override: the global model decides. Read it the
        # same way the factory does so a provider switch invalidates.
        try:
            from src.agents.core.model_factory import _load_settings

            settings = _load_settings()
            model = "{}:{}".format(
                settings.get("llm_provider", ""),
                settings.get("llm_model", ""),
            )
        except Exception:  # noqa: BLE001
            return None

    digest = hashlib.sha256()
    digest.update(model.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(prompt.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(dataset)
    return digest.hexdigest()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_eval_runs (
    run_id              TEXT PRIMARY KEY,
    agent_id            TEXT NOT NULL,
    suite               TEXT,
    trigger             TEXT NOT NULL DEFAULT 'manual',
    started_at          TEXT NOT NULL,
    finished_at         TEXT,
    status              TEXT NOT NULL DEFAULT 'pending',
    cases_total         INTEGER NOT NULL DEFAULT 0,
    cases_passed        INTEGER NOT NULL DEFAULT 0,
    cases_failed        INTEGER NOT NULL DEFAULT 0,
    failed_cases_json   TEXT,
    error               TEXT,
    fingerprint         TEXT
)
"""

_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_agent_eval_runs_agent_started "
    "ON agent_eval_runs(agent_id, started_at DESC)"
)


class EvalRunStore:
    """Read/write helper for ``agent_eval_runs``.

    Always opens the connection in autocommit mode — the runner is
    invoked from short-lived subprocesses (auto-trigger) and from the
    CLI handlers; rolling back at exit would lose the row we just
    persisted.

    sensitivity_tier: 1
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or DEFAULT_DB_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, isolation_level=None)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self._conn.execute(_SCHEMA)
        self._conn.execute(_INDEX)
        # ``fingerprint`` post-dates the original table; add it in place
        # for installs created before the skip-if-unchanged gate.
        cols = {
            r[1] for r in self._conn.execute(
                "PRAGMA table_info(agent_eval_runs)",
            )
        }
        if "fingerprint" not in cols:
            self._conn.execute(
                "ALTER TABLE agent_eval_runs ADD COLUMN fingerprint TEXT",
            )

    def close(self) -> None:
        self._conn.close()

    # ---- writers ----

    def insert_pending(
        self,
        *,
        agent_id: str,
        suite: str | None,
        trigger: str,
        fingerprint: str | None = None,
    ) -> str:
        run_id = uuid.uuid4().hex
        started = _now_iso()
        self._conn.execute(
            """
            INSERT INTO agent_eval_runs (
                run_id, agent_id, suite, trigger, started_at, status,
                fingerprint
            ) VALUES (?, ?, ?, ?, ?, 'running', ?)
            """,
            (run_id, agent_id, suite, trigger, started, fingerprint),
        )
        return run_id

    def has_passed_with(self, suite: str, fingerprint: str) -> bool:
        """True when ``suite`` has ever passed on this exact fingerprint.

        "Ever" rather than "most recently": the fingerprint pins the
        model, the prompt and the dataset, so a suite that passed on
        these inputs will pass on them again. Matching only the latest
        run would re-spend on an A -> B -> A edit, where reverting is
        the one case we can be surest about.

        Only ``ok`` counts. A failed or errored run must never suppress
        a re-run — seeing whether the failure persists is the point.

        sensitivity_tier: 1
        """
        cur = self._conn.execute(
            """
            SELECT 1 FROM agent_eval_runs
            WHERE suite = ? AND status = 'ok' AND fingerprint = ?
            LIMIT 1
            """,
            (suite, fingerprint),
        )
        return cur.fetchone() is not None

    def finalize(
        self,
        run_id: str,
        *,
        status: str,
        cases_total: int = 0,
        cases_passed: int = 0,
        cases_failed: int = 0,
        failed_cases: list[dict[str, Any]] | None = None,
        error: str | None = None,
    ) -> None:
        self._conn.execute(
            """
            UPDATE agent_eval_runs SET
                finished_at = ?,
                status = ?,
                cases_total = ?,
                cases_passed = ?,
                cases_failed = ?,
                failed_cases_json = ?,
                error = ?
            WHERE run_id = ?
            """,
            (
                _now_iso(),
                status,
                cases_total,
                cases_passed,
                cases_failed,
                json.dumps(failed_cases or []),
                error,
                run_id,
            ),
        )

    # ---- readers ----

    def latest(self, agent_id: str) -> EvalRun | None:
        cur = self._conn.execute(
            """
            SELECT run_id, agent_id, suite, trigger, started_at,
                   finished_at, status, cases_total, cases_passed,
                   cases_failed, failed_cases_json, error
            FROM agent_eval_runs
            WHERE agent_id = ?
            ORDER BY started_at DESC LIMIT 1
            """,
            (agent_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return _row_to_run(row)

    def history(self, agent_id: str, limit: int = 20) -> list[EvalRun]:
        cur = self._conn.execute(
            """
            SELECT run_id, agent_id, suite, trigger, started_at,
                   finished_at, status, cases_total, cases_passed,
                   cases_failed, failed_cases_json, error
            FROM agent_eval_runs
            WHERE agent_id = ?
            ORDER BY started_at DESC LIMIT ?
            """,
            (agent_id, int(limit)),
        )
        return [_row_to_run(r) for r in cur.fetchall()]


def _row_to_run(row: tuple) -> EvalRun:
    failed = json.loads(row[10]) if row[10] else []
    return EvalRun(
        run_id=row[0],
        agent_id=row[1],
        suite=row[2],
        trigger=row[3],
        started_at=row[4],
        finished_at=row[5],
        status=row[6],
        cases_total=int(row[7] or 0),
        cases_passed=int(row[8] or 0),
        cases_failed=int(row[9] or 0),
        failed_cases=failed,
        error=row[11],
    )


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def _user_dataset_path(agent_id: str) -> Path:
    """Where a user-uploaded eval dataset lives on disk.

    sensitivity_tier: 1
    """
    return (
        Path.home()
        / ".arandu"
        / "user_eval_datasets"
        / f"{agent_id}.yaml"
    )


def run_agent_eval(
    agent_id: str,
    *,
    trigger: str = "manual",
    store: EvalRunStore | None = None,
) -> EvalRun:
    """Run the eval suite mapped to ``agent_id`` and persist a row.

    Statuses:

    - ``passed``  — at least one case, every evaluator returned True.
    - ``failed``  — at least one evaluator returned False.
    - ``skipped`` — no dataset mapped, dataset file missing, or the
      agent's model isn't reachable. The Agents page renders this as
      a neutral state, not a failure.
    - ``error``   — unexpected exception. Recorded but the row is
      kept so the editor can show the message.

    sensitivity_tier: 1
    """
    store = store or EvalRunStore()
    suite = AGENT_SUITE_MAP.get(agent_id)
    user_path = _user_dataset_path(agent_id)
    if suite is None and not user_path.exists():
        # Pre-insert + finalize so the row reflects the run attempt.
        run_id = store.insert_pending(
            agent_id=agent_id, suite=None, trigger=trigger,
        )
        store.finalize(
            run_id,
            status="skipped",
            error="no dataset mapped for this agent",
        )
        return store.latest(agent_id)  # type: ignore[return-value]

    if suite is None:
        # User agent with an uploaded dataset. The structural validator
        # already vetted the YAML at upload time; here we resolve the
        # registered user-agent factory and run each case through it,
        # sharing the run_dataset_detailed core with built-in suites.
        run_id = store.insert_pending(
            agent_id=agent_id, suite=None, trigger=trigger,
        )
        try:
            from evals.run_evals import run_dataset_detailed
            from evals.tasks import ModelUnavailableError, user_agent_task
            try:
                task = user_agent_task(agent_id)
            except ModelUnavailableError as exc:
                store.finalize(run_id, status="skipped", error=str(exc))
                return store.latest(agent_id)  # type: ignore[return-value]
            try:
                total, passed, failed, failed_cases, _ = (
                    run_dataset_detailed(user_path, task)
                )
            except ModelUnavailableError as exc:
                store.finalize(run_id, status="skipped", error=str(exc))
                return store.latest(agent_id)  # type: ignore[return-value]
        except Exception as exc:  # noqa: BLE001
            logger.exception("user-agent eval run failed for %s", agent_id)
            store.finalize(run_id, status="error", error=str(exc))
            return store.latest(agent_id)  # type: ignore[return-value]

        status = "passed" if failed == 0 and total > 0 else (
            "failed" if failed > 0 else "skipped"
        )
        store.finalize(
            run_id,
            status=status,
            cases_total=total,
            cases_passed=passed,
            cases_failed=failed,
            failed_cases=failed_cases,
        )
        return store.latest(agent_id)  # type: ignore[return-value]

    run_id = store.insert_pending(
        agent_id=agent_id, suite=suite, trigger=trigger,
    )

    try:
        from evals.run_evals import DATASETS_DIR, run_suite_detailed
        from evals.tasks import ModelUnavailableError
        dataset_path = DATASETS_DIR / f"{suite}.yaml"
        if not dataset_path.exists():
            store.finalize(
                run_id,
                status="skipped",
                error=f"dataset not found: {suite}.yaml",
            )
            return store.latest(agent_id)  # type: ignore[return-value]
        try:
            total, passed, failed, failed_cases = run_suite_detailed(suite)
        except ModelUnavailableError as exc:
            store.finalize(run_id, status="skipped", error=str(exc))
            return store.latest(agent_id)  # type: ignore[return-value]
    except Exception as exc:  # noqa: BLE001
        logger.exception("eval run failed for %s", agent_id)
        store.finalize(
            run_id, status="error", error=str(exc),
        )
        return store.latest(agent_id)  # type: ignore[return-value]

    status = "passed" if failed == 0 and total > 0 else (
        "failed" if failed > 0 else "skipped"
    )
    store.finalize(
        run_id,
        status=status,
        cases_total=total,
        cases_passed=passed,
        cases_failed=failed,
        failed_cases=failed_cases,
    )
    return store.latest(agent_id)  # type: ignore[return-value]


class _ModelUnavailable(RuntimeError):  # noqa: N818
    """Raised when the eval task can't construct its agent's model.

    Used to translate a runtime construction failure into a ``skipped``
    row rather than ``error``.
    """


__all__ = [
    "AGENT_SUITE_MAP",
    "SUITE_TO_AGENT",
    "suite_fingerprint",
    "DEFAULT_DB_PATH",
    "EvalRun",
    "EvalRunStore",
    "MANUAL_ONLY_AGENTS",
    "run_agent_eval",
]
