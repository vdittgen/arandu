"""Named field transform functions for MCP data ingestion.

Each transform converts a raw MCP tool response value into a
DuckDB-compatible value.  Every function handles ``None`` gracefully
(returns ``None``) and never raises — invalid input is logged as a
warning and ``None`` is returned.

sensitivity_tier: 1 (data structure transforms, no user data stored)
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

TransformFn = Callable[[Any], Any]


# ------------------------------------------------------------------
# Individual transform functions
# ------------------------------------------------------------------


def iso_to_timestamp(value: Any) -> str | None:
    """Parse ISO 8601 string and return it for DuckDB CAST.

    Accepts formats like ``2025-06-02T10:30:00Z``,
    ``2025-06-02T10:30:00+05:00``, and ``2025-06-02 10:30:00``.
    Returns the string as-is if it parses successfully.

    sensitivity_tier: 1
    """
    if value is None:
        return None
    try:
        s = str(value)
        datetime.fromisoformat(s.replace("Z", "+00:00"))
        return s
    except (ValueError, TypeError) as exc:
        logger.warning(
            "iso_to_timestamp failed for %r: %s", value, exc,
        )
        return None


def unix_to_timestamp(value: Any) -> str | None:
    """Convert Unix epoch (seconds or milliseconds) to ISO 8601 string.

    Auto-detects milliseconds when the numeric value exceeds 1e12.

    sensitivity_tier: 1
    """
    if value is None:
        return None
    try:
        num = float(value)
        if num > 1e12:
            num /= 1000.0
        return datetime.fromtimestamp(num, tz=timezone.utc).isoformat()
    except (ValueError, TypeError, OSError) as exc:
        logger.warning(
            "unix_to_timestamp failed for %r: %s", value, exc,
        )
        return None


def json_serialize(value: Any) -> str | None:
    """Serialize any value to a JSON string.

    sensitivity_tier: 1
    """
    if value is None:
        return None
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        logger.warning(
            "json_serialize failed for %r: %s", value, exc,
        )
        return None


def json_array(value: Any) -> str | None:
    """Ensure the value is a JSON array string.

    Lists are serialized directly.  A JSON-array string is returned
    as-is.  A single scalar is wrapped in an array.

    sensitivity_tier: 1
    """
    if value is None:
        return None
    try:
        if isinstance(value, list):
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return value
                return json.dumps([parsed], ensure_ascii=False)
            except (json.JSONDecodeError, ValueError):
                return json.dumps([value], ensure_ascii=False)
        return json.dumps([value], ensure_ascii=False)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.warning(
            "json_array failed for %r: %s", value, exc,
        )
        return None


def flatten_object(value: Any) -> str | None:
    """Flatten a nested dict/object to a JSON string.

    sensitivity_tier: 1
    """
    if value is None:
        return None
    try:
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        logger.warning(
            "flatten_object failed for %r: %s", value, exc,
        )
        return None


def to_int(value: Any) -> int | None:
    """Convert to integer.  Floats are truncated.

    sensitivity_tier: 1
    """
    if value is None:
        return None
    try:
        return int(float(value))
    except (ValueError, TypeError) as exc:
        logger.warning("to_int failed for %r: %s", value, exc)
        return None


def to_float(value: Any) -> float | None:
    """Convert to float.

    sensitivity_tier: 1
    """
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError) as exc:
        logger.warning("to_float failed for %r: %s", value, exc)
        return None


def to_bool(value: Any) -> bool | None:
    """Convert to boolean.

    Handles ``"true"``/``"false"`` strings, ``0``/``1``, and Python
    booleans.

    sensitivity_tier: 1
    """
    if value is None:
        return None
    try:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            low = value.strip().lower()
            if low in ("true", "1", "yes"):
                return True
            if low in ("false", "0", "no"):
                return False
            return None
        return bool(value)
    except (ValueError, TypeError) as exc:
        logger.warning("to_bool failed for %r: %s", value, exc)
        return None


def array_to_json(value: Any) -> str | None:
    """Alias for :func:`json_array`.

    sensitivity_tier: 1
    """
    return json_array(value)


def trim(value: Any) -> str | None:
    """Strip leading/trailing whitespace.

    sensitivity_tier: 1
    """
    if value is None:
        return None
    try:
        if isinstance(value, str):
            return value.strip()
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("trim failed for %r: %s", value, exc)
        return None


def lowercase(value: Any) -> str | None:
    """Convert string to lowercase.

    sensitivity_tier: 1
    """
    if value is None:
        return None
    try:
        if isinstance(value, str):
            return value.lower()
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("lowercase failed for %r: %s", value, exc)
        return None


# ------------------------------------------------------------------
# Transform registry
# ------------------------------------------------------------------

TRANSFORMS: dict[str, TransformFn] = {
    "iso_to_timestamp": iso_to_timestamp,
    "unix_to_timestamp": unix_to_timestamp,
    "json_serialize": json_serialize,
    "json_array": json_array,
    "flatten_object": flatten_object,
    "to_int": to_int,
    "to_float": to_float,
    "to_bool": to_bool,
    "array_to_json": array_to_json,
    "trim": trim,
    "lowercase": lowercase,
}


def apply_transform(transform_name: str | None, value: Any) -> Any:
    """Look up and apply a named transform.

    Returns *value* unchanged when *transform_name* is ``None``.
    Logs a warning and returns ``None`` for unknown transform names.

    sensitivity_tier: 1
    """
    if transform_name is None:
        return value
    fn = TRANSFORMS.get(transform_name)
    if fn is None:
        logger.warning("Unknown transform: %r", transform_name)
        return None
    return fn(value)
