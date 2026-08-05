"""Pipeline executor — runs SQL/Python transforms against SQLite.

Executes each model as ``DROP TABLE IF EXISTS t; CREATE TABLE t AS <sql>``.
Python models are imported and called with the DatabaseEngine directly.

sensitivity_tier: 1 (infrastructure — executes transforms, no user data stored)
"""

from __future__ import annotations

import importlib
import logging
import time
from pathlib import Path
from typing import Any

from src.core.sqlite.engine import DatabaseEngine
from src.pipeline.manifest import ModelSpec

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read_sql_file(sql_file: str) -> str:
    """Read a SQL file relative to the project root.

    Strips the MODEL() header if present (legacy SQLMesh format).

    sensitivity_tier: N/A
    """
    path = PROJECT_ROOT / sql_file
    content = path.read_text(encoding="utf-8")

    # Strip SQLMesh MODEL() header if present
    # Pattern: MODEL (...); followed by the actual SELECT
    import re

    model_pattern = re.compile(
        r"^\s*MODEL\s*\(.*?\)\s*;?\s*",
        re.DOTALL | re.IGNORECASE,
    )
    content = model_pattern.sub("", content).strip()

    return content


def execute_sql_model(
    db: DatabaseEngine,
    model: ModelSpec,
) -> int:
    """Execute a SQL model: drop + create table from SELECT.

    Args:
        db: The SQLite database engine.
        model: Model specification with sql_file path.

    Returns:
        Row count of the newly created table.

    Raises:
        FileNotFoundError: If the SQL file doesn't exist.

    sensitivity_tier: 1
    """
    if model.sql_file is None:
        msg = f"Model {model.name!r} has no sql_file"
        raise ValueError(msg)

    sql = _read_sql_file(model.sql_file)
    table_name = model.name

    # Full refresh: drop and recreate
    db.execute(f"DROP TABLE IF EXISTS {table_name}")
    db.execute(f"CREATE TABLE {table_name} AS {sql}")

    # Count rows
    rows = db.query(f"SELECT COUNT(*) AS n FROM {table_name}")
    count = rows[0]["n"] if rows else 0

    logger.info(
        "Executed SQL model %s → %d rows", model.name, count,
    )
    return count


def execute_python_model(
    db: DatabaseEngine,
    model: ModelSpec,
) -> int:
    """Execute a Python model by importing and calling its function.

    The function signature must be:
        execute(db: DatabaseEngine) -> list[dict[str, Any]]

    The returned rows are inserted into a table named after the model.

    Args:
        db: The SQLite database engine.
        model: Model specification with python_module and python_function.

    Returns:
        Row count of the newly created table.

    sensitivity_tier: 1
    """
    if model.python_module is None or model.python_function is None:
        msg = (
            f"Python model {model.name!r} missing "
            f"python_module or python_function"
        )
        raise ValueError(msg)

    module = importlib.import_module(model.python_module)
    func = getattr(module, model.python_function)

    rows: list[dict[str, Any]] = func(db)

    table_name = model.name

    if not rows:
        db.execute(f"DROP TABLE IF EXISTS {table_name}")
        # Create empty table with no columns — will be recreated next run
        logger.info("Python model %s produced 0 rows", model.name)
        return 0

    # Build table from first row's keys
    columns = list(rows[0].keys())
    col_defs = ", ".join(f"{col} TEXT" for col in columns)
    placeholders = ", ".join("?" for _ in columns)

    db.execute(f"DROP TABLE IF EXISTS {table_name}")
    db.execute(f"CREATE TABLE {table_name} ({col_defs})")

    for row in rows:
        values = [row.get(col) for col in columns]
        db.execute(
            f"INSERT INTO {table_name} ({', '.join(columns)}) "
            f"VALUES ({placeholders})",
            values,
        )

    logger.info(
        "Executed Python model %s → %d rows", model.name, len(rows),
    )
    return len(rows)


def execute_model(
    db: DatabaseEngine,
    model: ModelSpec,
) -> int:
    """Execute a single model (SQL or Python).

    Args:
        db: The SQLite database engine.
        model: Model specification.

    Returns:
        Row count of the resulting table.

    sensitivity_tier: 1
    """
    if model.model_type == "python":
        return execute_python_model(db, model)
    return execute_sql_model(db, model)


def execute_pipeline(
    db: DatabaseEngine,
    models: list[ModelSpec],
    on_progress: Any | None = None,
    cancel_check: Any | None = None,
) -> dict[str, int]:
    """Execute all models in order, returning per-model row counts.

    Args:
        db: The SQLite database engine.
        models: Models in execution order (topologically sorted).
        on_progress: Optional callback(dict) for progress events.
        cancel_check: Optional callable returning True to cancel.

    Returns:
        Dict mapping model name to row count (-1 on error).

    sensitivity_tier: 1
    """
    counts: dict[str, int] = {}
    total = len(models)
    start_time = time.monotonic()

    for idx, model in enumerate(models):
        if cancel_check is not None and cancel_check():
            logger.info("Pipeline cancelled at model %s", model.name)
            break

        if on_progress is not None:
            on_progress({
                "type": "model_start",
                "model_name": model.name,
                "step_index": idx,
                "total_steps": total,
                "elapsed_seconds": round(
                    time.monotonic() - start_time, 2,
                ),
            })

        try:
            count = execute_model(db, model)
            counts[model.name] = count
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Failed to execute model %s: %s", model.name, exc,
            )
            counts[model.name] = -1

        if on_progress is not None:
            on_progress({
                "type": "model_complete",
                "model_name": model.name,
                "step_index": idx + 1,
                "total_steps": total,
                "rows_processed": counts.get(model.name, -1),
                "elapsed_seconds": round(
                    time.monotonic() - start_time, 2,
                ),
            })

    return counts
