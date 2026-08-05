"""EvalRunStore + run_agent_eval behaviour tests.

These exercise the persistence layer (autocommit + row shape) and the
status-decision branches (passed / failed / skipped / error) without
hitting the real eval framework. Where a full eval run is needed,
we drive the firewall_prompts suite which is deterministic and runs
offline.

sensitivity_tier: N/A
"""

from __future__ import annotations

from pathlib import Path

import pytest
from src.agents.eval_runner import (
    AGENT_SUITE_MAP,
    MANUAL_ONLY_AGENTS,
    EvalRunStore,
    run_agent_eval,
)


@pytest.fixture()
def store(tmp_path: Path) -> EvalRunStore:
    db = tmp_path / "evals.sqlite3"
    return EvalRunStore(path=db)


def test_insert_and_finalize_round_trip(store: EvalRunStore) -> None:
    run_id = store.insert_pending(
        agent_id="triage", suite="triage", trigger="manual",
    )
    store.finalize(
        run_id,
        status="passed",
        cases_total=5, cases_passed=5, cases_failed=0,
        failed_cases=[],
    )
    row = store.latest("triage")
    assert row is not None
    assert row.run_id == run_id
    assert row.status == "passed"
    assert row.cases_total == 5
    assert row.cases_passed == 5
    assert row.failed_cases == []


def test_finalize_persists_failed_cases(store: EvalRunStore) -> None:
    run_id = store.insert_pending(
        agent_id="triage", suite="triage", trigger="auto",
    )
    failed = [
        {"case": "promo_dropped", "evaluator": "TriageDecisionAccuracy",
         "reason": "is_promo: got False, expected True"},
    ]
    store.finalize(
        run_id, status="failed",
        cases_total=2, cases_passed=1, cases_failed=1,
        failed_cases=failed,
    )
    row = store.latest("triage")
    assert row is not None
    assert row.status == "failed"
    assert row.failed_cases == failed


def test_history_orders_newest_first(store: EvalRunStore) -> None:
    import time
    for i in range(3):
        rid = store.insert_pending(
            agent_id="labeler", suite="labeler", trigger="manual",
        )
        store.finalize(rid, status="passed", cases_total=i + 1)
        time.sleep(0.01)  # ensure distinct started_at timestamps
    rows = store.history("labeler", limit=10)
    assert len(rows) == 3
    # Newest (highest cases_total) first.
    assert rows[0].cases_total == 3
    assert rows[-1].cases_total == 1


def test_agent_suite_map_covers_every_registered_agent() -> None:
    from src.agents.brain import bootstrap_agents
    from src.agents.core.registry import (
        all_agents,
        reset_registry_for_tests,
    )

    reset_registry_for_tests()
    bootstrap_agents()
    missing = [
        d.agent_id for d in all_agents()
        if d.agent_id not in AGENT_SUITE_MAP
    ]
    assert missing == [], f"agents without an eval suite: {missing}"


def test_manual_only_agents_are_locked() -> None:
    # The manual-only set should match the locked agents in the
    # registry to keep auto-trigger / locked-card behaviour aligned.
    assert "brain" in MANUAL_ONLY_AGENTS
    assert "firewall.injection" in MANUAL_ONLY_AGENTS
    assert "firewall.egress" in MANUAL_ONLY_AGENTS


def test_unknown_agent_records_skipped(tmp_path: Path) -> None:
    store = EvalRunStore(path=tmp_path / "evals.sqlite3")
    run = run_agent_eval(
        "no_such_agent", trigger="manual", store=store,
    )
    assert run.status == "skipped"
    assert "no dataset" in (run.error or "")


def test_firewall_injection_eval_passes(tmp_path: Path) -> None:
    # Deterministic suite — must always pass end-to-end.
    store = EvalRunStore(path=tmp_path / "evals.sqlite3")
    run = run_agent_eval(
        "firewall.injection", trigger="manual", store=store,
    )
    assert run.status == "passed"
    assert run.cases_failed == 0
    assert run.cases_passed > 0
    assert run.suite == "firewall_prompts"


def test_firewall_egress_eval_passes(tmp_path: Path) -> None:
    store = EvalRunStore(path=tmp_path / "evals.sqlite3")
    run = run_agent_eval(
        "firewall.egress", trigger="manual", store=store,
    )
    assert run.status == "passed"
    assert run.suite == "egress_routing"


def test_spawn_auto_eval_no_longer_exported() -> None:
    """Auto-eval was removed in 0.5.0; evals run on explicit user action.

    The Agents page now exposes a "Run eval" button, and ``make
    evals`` / ``python -m evals.run_evals`` runs the full batch from
    the CLI. Nothing else may trigger judge calls.
    """
    import src.agents.eval_runner as er

    assert not hasattr(er, "spawn_auto_eval")


# ---------------------------------------------------------------------------
# Fingerprint gate — skip a suite when nothing that affects it changed
# ---------------------------------------------------------------------------


def test_has_passed_with_matches_exact_fingerprint(
    store: EvalRunStore,
) -> None:
    rid = store.insert_pending(
        agent_id="firewall.egress", suite="egress_routing",
        trigger="cli", fingerprint="abc123",
    )
    store.finalize(rid, status="ok", cases_total=4, cases_passed=4)
    assert store.has_passed_with("egress_routing", "abc123") is True
    assert store.has_passed_with("egress_routing", "different") is False
    assert store.has_passed_with("other_suite", "abc123") is False


def test_has_passed_with_ignores_failed_runs(store: EvalRunStore) -> None:
    """A failure must never suppress the re-run that would confirm it."""
    rid = store.insert_pending(
        agent_id="firewall.egress", suite="egress_routing",
        trigger="cli", fingerprint="abc123",
    )
    store.finalize(rid, status="failed", cases_total=4, cases_failed=1)
    assert store.has_passed_with("egress_routing", "abc123") is False


def test_has_passed_with_matches_any_earlier_pass(
    store: EvalRunStore,
) -> None:
    """A -> B -> A must skip on the return to A, not re-spend."""
    for fp in ("fp-A", "fp-B"):
        rid = store.insert_pending(
            agent_id="firewall.egress", suite="egress_routing",
            trigger="cli", fingerprint=fp,
        )
        store.finalize(rid, status="ok", cases_total=4, cases_passed=4)
    assert store.has_passed_with("egress_routing", "fp-A") is True


def test_fingerprint_column_added_to_legacy_table(tmp_path: Path) -> None:
    """Installs predating the gate must migrate in place, not crash."""
    import sqlite3

    db = tmp_path / "legacy.sqlite3"
    conn = sqlite3.connect(db, isolation_level=None)
    conn.execute(
        """
        CREATE TABLE agent_eval_runs (
            run_id TEXT PRIMARY KEY, agent_id TEXT NOT NULL, suite TEXT,
            trigger TEXT NOT NULL DEFAULT 'manual',
            started_at TEXT NOT NULL, finished_at TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            cases_total INTEGER NOT NULL DEFAULT 0,
            cases_passed INTEGER NOT NULL DEFAULT 0,
            cases_failed INTEGER NOT NULL DEFAULT 0,
            failed_cases_json TEXT, error TEXT
        )
        """,
    )
    conn.close()

    store = EvalRunStore(path=db)
    cols = {
        r[1] for r in store._conn.execute(
            "PRAGMA table_info(agent_eval_runs)",
        )
    }
    assert "fingerprint" in cols
    rid = store.insert_pending(
        agent_id="firewall.egress", suite="egress_routing",
        trigger="cli", fingerprint="fp",
    )
    store.finalize(rid, status="ok")
    assert store.has_passed_with("egress_routing", "fp") is True


def test_suite_fingerprint_tracks_dataset_and_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hash must move when the dataset or the model moves."""
    from src.agents.eval_runner import suite_fingerprint

    ds = tmp_path / "egress_routing.yaml"
    ds.write_text("cases: []\n")

    monkeypatch.setattr(
        "src.agents.core.config_store.current_model_override",
        lambda _agent_id: "arandu-pro",
    )
    base = suite_fingerprint("egress_routing", ds)
    assert base is not None, "registry should bootstrap for a known suite"

    # Same inputs -> same hash.
    assert suite_fingerprint("egress_routing", ds) == base

    # Dataset moved.
    ds.write_text("cases: []\n# edited\n")
    assert suite_fingerprint("egress_routing", ds) != base

    # Model moved.
    ds.write_text("cases: []\n")
    monkeypatch.setattr(
        "src.agents.core.config_store.current_model_override",
        lambda _agent_id: "arandu-reasoning",
    )
    assert suite_fingerprint("egress_routing", ds) != base


def test_suite_fingerprint_none_for_unknown_suite(tmp_path: Path) -> None:
    """``None`` means 'cannot prove unchanged' — callers must run."""
    from src.agents.eval_runner import suite_fingerprint

    ds = tmp_path / "nope.yaml"
    ds.write_text("cases: []\n")
    assert suite_fingerprint("no_such_suite", ds) is None
    assert suite_fingerprint("egress_routing", tmp_path / "missing") is None
