"""Eval runner CLI.

Loads one or more YAML datasets from ``evals/datasets/``, dispatches
each to the matching task in :mod:`evals.tasks`, runs every case
through the corresponding Pydantic AI agent, and prints a summary
report.

Usage::

    python -m evals.run_evals --suite all
    python -m evals.run_evals --suite sensitivity
    python -m evals.run_evals --suite sensitivity,triage --json

Phase 5 ships the structural evaluators only. The LLM-judge
evaluator stub passes unconditionally with a ``skipped`` reason —
Phase 5b lights up the semantic checks once a judge model is wired.

sensitivity_tier: N/A
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic_evals import Dataset

from evals.evaluators import (
    ConfidenceInRange,
    ContainsIds,
    DelegatedTo,
    EmotionalLabelStructural,
    FactSetMatches,
    FieldContains,
    FieldEquals,
    FieldIn,
    FieldNotEmpty,
    FirewallAllowedMatches,
    IntInRange,
    ListLengthInRange,
    LLMJudgeOnField,
    LLMJudgeOnReason,
    TierEquals,
    TriageDecisionAccuracy,
)
from evals.tasks import TASK_REGISTRY

logger = logging.getLogger(__name__)

DATASETS_DIR = Path(__file__).parent / "datasets"

CUSTOM_EVALUATORS = (
    ConfidenceInRange,
    ContainsIds,
    DelegatedTo,
    EmotionalLabelStructural,
    FactSetMatches,
    FieldContains,
    FieldEquals,
    FieldIn,
    FieldNotEmpty,
    FirewallAllowedMatches,
    IntInRange,
    LLMJudgeOnField,
    LLMJudgeOnReason,
    ListLengthInRange,
    TierEquals,
    TriageDecisionAccuracy,
)


# ---------------------------------------------------------------------------
# Suite resolution
# ---------------------------------------------------------------------------


def available_suites() -> list[str]:
    """Return suite names — the basenames of YAML files in ``datasets/``.

    sensitivity_tier: N/A
    """
    if not DATASETS_DIR.exists():
        return []
    return sorted(
        p.stem for p in DATASETS_DIR.glob("*.yaml")
    )


def resolve_suites(spec: str) -> list[str]:
    """Translate a ``--suite`` argument into the list of suite stems.

    sensitivity_tier: N/A
    """
    available = available_suites()
    if spec == "all" or not spec:
        return available
    requested = [s.strip() for s in spec.split(",") if s.strip()]
    missing = [s for s in requested if s not in available]
    if missing:
        msg = (
            f"unknown suite(s): {sorted(missing)}; "
            f"available: {available}"
        )
        raise SystemExit(msg)
    return requested


# ---------------------------------------------------------------------------
# Run + summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SuiteResult:
    """Summary of one suite's eval run.

    sensitivity_tier: N/A
    """

    suite: str
    cases: int
    passed: int
    failed: int
    duration_s: float

    @property
    def pass_rate(self) -> float:
        return self.passed / self.cases if self.cases else 0.0


def run_suite(suite: str) -> SuiteResult:
    """Run one suite and return the aggregate summary.

    sensitivity_tier: N/A
    """
    total, passed, failed, _failed_cases, duration = _run_suite_internal(suite)
    return SuiteResult(
        suite=suite, cases=total, passed=passed,
        failed=failed, duration_s=duration,
    )


def run_suite_detailed(
    suite: str,
) -> tuple[int, int, int, list[dict[str, Any]]]:
    """Run one suite and return per-case failure detail.

    Returns ``(cases_total, cases_passed, cases_failed, failed_cases)``
    where ``failed_cases`` is a list of dicts the Agents page renders
    directly: ``{"case": str, "evaluator": str, "reason": str}``.

    sensitivity_tier: N/A
    """
    total, passed, failed, failed_cases, _ = _run_suite_internal(suite)
    return total, passed, failed, failed_cases


def _run_suite_internal(
    suite: str,
) -> tuple[int, int, int, list[dict[str, Any]], float]:
    path = DATASETS_DIR / f"{suite}.yaml"
    if not path.exists():
        msg = f"dataset not found: {path}"
        raise FileNotFoundError(msg)
    task_factory = TASK_REGISTRY.get(path.name)
    if task_factory is None:
        msg = f"no task registered for {path.name}"
        raise KeyError(msg)
    return run_dataset_detailed(path, task_factory())


def run_dataset_detailed(
    path: Path,
    task: Callable[[Any], Any],
) -> tuple[int, int, int, list[dict[str, Any]], float]:
    """Run an arbitrary dataset YAML through a caller-supplied task.

    Shared core used by built-in suites (via the dataset filename →
    factory map in :data:`TASK_REGISTRY`) and by user-agent runs
    (dataset path under ``~/.arandu/user_eval_datasets/`` + task
    constructed by :func:`evals.tasks.user_agent_task`). Returns
    ``(total, passed, failed, failed_cases, duration_s)``.

    sensitivity_tier: 1
    """
    from evals.tasks import ModelUnavailableError

    # Auto-canonicalise legacy `args:` → `arguments:` in evaluator
    # entries. pydantic-evals' _DatasetModel requires `arguments:`;
    # earlier dataset_creator versions emitted `args:` which passed
    # the structural check but blew up here. Rewrite in place so the
    # file lands on the canonical schema and subsequent loads are no-ops.
    try:
        from src.agents.dataset_validator import canonicalize_dataset_yaml

        original = path.read_text(encoding="utf-8")
        canonical, changed = canonicalize_dataset_yaml(original)
        if changed:
            path.write_text(canonical, encoding="utf-8")
    except OSError:
        pass

    dataset = Dataset.from_file(
        path, custom_evaluator_types=CUSTOM_EVALUATORS,
    )
    report = dataset.evaluate_sync(task, progress=False)

    # Per-case task exceptions surface in report.failures, NOT
    # report.cases. If every case errored with ModelUnavailableError
    # we treat the whole suite as "no model available" so the UI can
    # show a clean skipped row instead of a flood of fake failures.
    task_failures = list(getattr(report, "failures", []) or [])
    if task_failures and not report.cases:
        all_unavailable = all(
            (f.error_message or "").startswith("ModelUnavailableError")
            for f in task_failures
        )
        if all_unavailable:
            # Pull the first error message as the human-readable cause.
            first = task_failures[0].error_message or "model unavailable"
            cleaned = first.replace("ModelUnavailableError: ", "", 1)
            raise ModelUnavailableError(cleaned)

    passed = 0
    failed = 0
    failed_cases: list[dict[str, Any]] = []
    for case_report in report.cases:
        case_ok = True
        for ev_name, ev in case_report.assertions.items():
            if not bool(ev.value):
                case_ok = False
                failed_cases.append({
                    "case": case_report.name or "(unnamed)",
                    "evaluator": ev_name,
                    "reason": ev.reason or "",
                })
        if case_ok:
            passed += 1
        else:
            failed += 1
    # Record task-level errors as failed cases so the UI shows them.
    for f in task_failures:
        failed += 1
        failed_cases.append({
            "case": f.name or "(unnamed)",
            "evaluator": "task",
            "reason": (f.error_message or "")[:300],
        })
    duration = sum(
        case_report.task_duration for case_report in report.cases
    )
    return (
        len(report.cases) + len(task_failures),
        passed,
        failed,
        failed_cases,
        duration,
    )


def run_suites(suites: list[str]) -> list[SuiteResult]:
    """Run multiple suites sequentially.

    sensitivity_tier: N/A
    """
    return [run_suite(s) for s in suites]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _render_text(results: list[SuiteResult]) -> str:
    if not results:
        return "(no suites)\n"
    width = max(len(r.suite) for r in results) + 2
    lines = [
        f"{'suite'.ljust(width)} cases  pass  fail   rate    time",
        "-" * (width + 36),
    ]
    total_cases = total_pass = total_fail = 0
    total_time = 0.0
    for r in results:
        total_cases += r.cases
        total_pass += r.passed
        total_fail += r.failed
        total_time += r.duration_s
        lines.append(
            f"{r.suite.ljust(width)} "
            f"{r.cases:>5}  {r.passed:>4}  {r.failed:>4}  "
            f"{r.pass_rate*100:>5.1f}%  {r.duration_s:>6.2f}s",
        )
    lines.append("-" * (width + 36))
    overall = total_pass / total_cases if total_cases else 0.0
    lines.append(
        f"{'TOTAL'.ljust(width)} "
        f"{total_cases:>5}  {total_pass:>4}  {total_fail:>4}  "
        f"{overall*100:>5.1f}%  {total_time:>6.2f}s",
    )
    return "\n".join(lines) + "\n"


def _render_json(results: list[SuiteResult]) -> str:
    payload: dict[str, Any] = {
        "suites": [
            {
                "suite": r.suite,
                "cases": r.cases,
                "passed": r.passed,
                "failed": r.failed,
                "pass_rate": r.pass_rate,
                "duration_s": r.duration_s,
            }
            for r in results
        ],
    }
    return json.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Entrypoint used by ``python -m evals.run_evals``.

    sensitivity_tier: N/A
    """
    parser = argparse.ArgumentParser(
        description="Run Arandu agent eval suites.",
    )
    parser.add_argument(
        "--suite",
        default="all",
        help="Suite stem, comma-separated list, or 'all' (default).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the available suites and exit.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON report instead of the text table.",
    )
    args = parser.parse_args(argv)
    if args.list:
        for s in available_suites():
            print(s)
        return 0
    suites = resolve_suites(args.suite)
    results = run_suites(suites)
    output = _render_json(results) if args.json else _render_text(results)
    sys.stdout.write(output)
    sys.stdout.flush()
    # Exit non-zero if any case failed — useful for CI gating.
    failed_total = sum(r.failed for r in results)
    return 0 if failed_total == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
