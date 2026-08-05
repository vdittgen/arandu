"""Shared LLM response parsing utilities.

Provides reusable functions for parsing LLM JSON responses that handle
common quirks: markdown fences, wrapper keys, and malformed output.

Used by: proactive_intelligence, message_evaluator, fact_learner — all
need the same "best-effort parse LLM JSON" logic.

sensitivity_tier: 1 (pure parsing, no user data access)
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def parse_llm_json_array(raw: Any) -> list[dict[str, Any]]:
    """Parse LLM output that should be a JSON array.

    Handles common LLM quirks:
    - ``chat_json()`` returns a dict with wrapper keys (``facts``,
      ``results``, ``items``, ``evaluations``, ``events``)
    - ``chat_json()`` returns a raw list directly
    - Response wrapped in markdown code fences
    - Response is a single dict that should be wrapped in a list

    Args:
        raw: The LLM response — typically a dict from ``chat_json()``,
            or a string that needs JSON parsing.

    Returns:
        List of dicts, or empty list on failure.

    sensitivity_tier: 1
    """
    # Already a list
    if isinstance(raw, list):
        return raw

    # Dict with common wrapper keys
    if isinstance(raw, dict):
        for key in (
            "facts", "results", "items", "evaluations",
            "events", "messages", "contacts", "data",
            "needs_reply",
        ):
            if key in raw and isinstance(raw[key], list):
                return raw[key]
        # Single-item dict that looks like a result row
        if len(raw) > 1:
            return [raw]
        return []

    # String — try to extract JSON array
    if isinstance(raw, str):
        return _parse_json_array_from_string(raw)

    return []


def parse_llm_json_dict(raw: Any) -> dict[str, Any]:
    """Parse LLM output that should be a JSON dict.

    Handles markdown fences and string responses.

    Args:
        raw: The LLM response.

    Returns:
        Dict, or empty dict on failure.

    sensitivity_tier: 1
    """
    if isinstance(raw, dict):
        return raw

    if isinstance(raw, str):
        text = _strip_markdown_fences(raw)
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

    return {}


def safe_chat_json(
    llm_provider: Any,
    messages: list[dict[str, str]],
) -> dict[str, Any] | list[Any]:
    """Non-fatal ``chat_json()`` call.

    Wraps ``llm_provider.chat_json(messages)`` in a try/except and
    returns an empty dict on any failure.

    Args:
        llm_provider: LLM provider with ``chat_json()`` method.
        messages: Chat messages to send.

    Returns:
        The LLM response (dict or list), or empty dict on failure.

    sensitivity_tier: varies (depends on message content)
    """
    if llm_provider is None:
        return {}
    try:
        return llm_provider.chat_json(messages)
    except Exception:
        logger.debug("safe_chat_json failed", exc_info=True)
        return {}


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences from LLM output.

    sensitivity_tier: 1
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [
            ln for ln in lines
            if not ln.strip().startswith("```")
        ]
        text = "\n".join(lines).strip()
    return text


def _parse_json_array_from_string(raw: str) -> list[dict[str, Any]]:
    """Extract a JSON array from a string, handling markdown fences.

    sensitivity_tier: 1
    """
    text = _strip_markdown_fences(raw)
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        result = json.loads(text[start : end + 1])
        if isinstance(result, list):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    return []
