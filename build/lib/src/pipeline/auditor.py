"""Pipeline auditor — post-transform data quality checks.

Runs not_null, unique, and accepted_values audits against the output
tables of executed models.

sensitivity_tier: 1 (infrastructure — checks data quality, no user data stored)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.core.sqlite.engine import DatabaseEngine
from src.pipeline.manifest import ModelSpec

logger = logging.getLogger(__name__)


@dataclass
class AuditResult:
    """Result of running audits on a single model.

    sensitivity_tier: N/A
    """

    model_name: str
    passed: bool = True
    failures: list[str] = field(default_factory=list)


def audit_not_null(
    db: DatabaseEngine,
    table: str,
    columns: list[str],
) -> list[str]:
    """Check that specified columns have no NULL values.

    Returns list of failure messages (empty if all pass).

    sensitivity_tier: 1
    """
    failures: list[str] = []
    for col in columns:
        try:
            rows = db.query(
                f"SELECT COUNT(*) AS n FROM {table} WHERE {col} IS NULL",
            )
            null_count = rows[0]["n"] if rows else 0
            if null_count > 0:
                failures.append(
                    f"not_null({col}): {null_count} NULL values in {table}",
                )
        except Exception as exc:  # noqa: BLE001
            failures.append(f"not_null({col}): query failed — {exc}")
    return failures


def audit_unique(
    db: DatabaseEngine,
    table: str,
    columns: list[str],
) -> list[str]:
    """Check that specified columns have unique values.

    Returns list of failure messages (empty if all pass).

    sensitivity_tier: 1
    """
    failures: list[str] = []
    for col in columns:
        try:
            rows = db.query(
                f"SELECT {col}, COUNT(*) AS cnt FROM {table} "
                f"GROUP BY {col} HAVING cnt > 1 LIMIT 1",
            )
            if rows:
                failures.append(
                    f"unique({col}): duplicate values found in {table}",
                )
        except Exception as exc:  # noqa: BLE001
            failures.append(f"unique({col}): query failed — {exc}")
    return failures


def audit_accepted_values(
    db: DatabaseEngine,
    table: str,
    checks: dict[str, list[Any]],
) -> list[str]:
    """Check that column values are within accepted sets.

    Returns list of failure messages (empty if all pass).

    sensitivity_tier: 1
    """
    failures: list[str] = []
    for col, accepted in checks.items():
        try:
            placeholders = ", ".join("?" for _ in accepted)
            rows = db.query(
                f"SELECT DISTINCT {col} FROM {table} "
                f"WHERE {col} NOT IN ({placeholders}) "
                f"AND {col} IS NOT NULL",
                list(accepted),
            )
            if rows:
                bad_values = [r[col] for r in rows[:5]]
                failures.append(
                    f"accepted_values({col}): unexpected values "
                    f"{bad_values} in {table}",
                )
        except Exception as exc:  # noqa: BLE001
            failures.append(
                f"accepted_values({col}): query failed — {exc}",
            )
    return failures


def audit_model(
    db: DatabaseEngine,
    model: ModelSpec,
) -> AuditResult:
    """Run all audits for a single model.

    Args:
        db: The SQLite database engine.
        model: Model specification with audit definitions.

    Returns:
        AuditResult with pass/fail status and failure messages.

    sensitivity_tier: 1
    """
    result = AuditResult(model_name=model.name)
    table = model.name
    spec = model.audits

    # Check table exists
    try:
        rows = db.query(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name = ?",
            [table],
        )
        if not rows:
            result.passed = False
            result.failures.append(f"Table {table} does not exist")
            return result
    except Exception as exc:  # noqa: BLE001
        result.passed = False
        result.failures.append(f"Cannot check table existence: {exc}")
        return result

    # Run audits
    result.failures.extend(audit_not_null(db, table, spec.not_null))
    result.failures.extend(audit_unique(db, table, spec.unique))
    result.failures.extend(
        audit_accepted_values(db, table, spec.accepted_values),
    )

    result.passed = len(result.failures) == 0

    if result.passed:
        logger.debug("Audit passed: %s", model.name)
    else:
        logger.warning(
            "Audit failed for %s: %s", model.name, result.failures,
        )

    return result


def audit_pipeline(
    db: DatabaseEngine,
    models: list[ModelSpec],
) -> list[AuditResult]:
    """Run audits for all models.

    Args:
        db: The SQLite database engine.
        models: List of model specs to audit.

    Returns:
        List of AuditResult, one per model.

    sensitivity_tier: 1
    """
    results: list[AuditResult] = []
    for model in models:
        results.append(audit_model(db, model))
    return results
