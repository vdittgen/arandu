"""OpenAI-compatible backend JSON-schema compatibility shim.

Many non-OpenAI OpenAI-compatible backends reject
``tools[*].function.parameters`` payloads that contain ``$defs`` /
``$ref`` — they return an opaque HTTP 500 instead of a 4xx schema
error. Pydantic emits ``$ref`` whenever a structured output type has
nested ``BaseModel`` fields, so any agent whose ``output_type``
contains a nested model (e.g. ``ChatResponse.parts: list[MessagePart]``)
silently breaks against these backends.

The fix: walk every tool's parameter schema and inline ``$ref``
nodes against the local ``$defs``, then strip ``$defs``. We patch the
``openai`` SDK's ``AsyncCompletions.create`` / ``Completions.create``
at the call boundary because ``pydantic_ai`` constructs its own
``openai`` client internally and bypasses our :class:`LLMProvider`
wrappers, so an in-provider fix wouldn't catch agent calls.

The patch is a no-op for requests targeting ``api.openai.com`` (real
OpenAI supports ``$ref``); we only flatten when the base URL is
something else.

sensitivity_tier: 1
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

logger = logging.getLogger(__name__)

_PATCHED = False
_MAX_INLINE_DEPTH = 32


def inline_json_schema_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``schema`` with ``$ref`` resolved against ``$defs``.

    Walks the schema recursively. Each ``{"$ref": "#/$defs/X"}`` node is
    replaced by a deep copy of ``schema["$defs"]["X"]``. Recursive refs
    are bounded by ``_MAX_INLINE_DEPTH`` to avoid infinite expansion;
    after that depth the ``$ref`` is preserved verbatim.

    Returns a new dict; the input is not mutated.

    sensitivity_tier: 1
    """
    if not isinstance(schema, dict):
        return schema
    defs = schema.get("$defs") or schema.get("definitions") or {}
    if not defs:
        return schema
    inlined = _walk(deepcopy(schema), defs, depth=0)
    if isinstance(inlined, dict):
        inlined.pop("$defs", None)
        inlined.pop("definitions", None)
    return inlined


def _walk(node: Any, defs: dict[str, Any], depth: int) -> Any:
    if depth > _MAX_INLINE_DEPTH:
        return node
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str):
            target = _resolve_ref(ref, defs)
            if target is not None:
                resolved = _walk(deepcopy(target), defs, depth + 1)
                # Preserve any sibling keys (e.g. "description") that
                # appeared alongside the $ref by merging them on top.
                if isinstance(resolved, dict):
                    siblings = {
                        k: v for k, v in node.items() if k != "$ref"
                    }
                    if siblings:
                        merged = {**resolved, **siblings}
                        return _walk(merged, defs, depth + 1)
                return resolved
        return {k: _walk(v, defs, depth + 1) for k, v in node.items()}
    if isinstance(node, list):
        return [_walk(item, defs, depth + 1) for item in node]
    return node


def _resolve_ref(ref: str, defs: dict[str, Any]) -> dict[str, Any] | None:
    # Accept both "#/$defs/X" and "#/definitions/X".
    prefixes = ("#/$defs/", "#/definitions/")
    for prefix in prefixes:
        if ref.startswith(prefix):
            name = ref[len(prefix):]
            return defs.get(name)
    return None


def _flatten_tools_kwargs(kwargs: dict[str, Any]) -> None:
    tools = kwargs.get("tools")
    if not isinstance(tools, list):
        return
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function")
        if not isinstance(fn, dict):
            continue
        params = fn.get("parameters")
        if isinstance(params, dict) and (
            "$defs" in params or "definitions" in params
        ):
            fn["parameters"] = inline_json_schema_refs(params)


def _should_flatten(client: Any) -> bool:
    """True when the client's base URL is NOT api.openai.com.

    sensitivity_tier: 1
    """
    base_url = getattr(client, "base_url", None)
    if base_url is None:
        return False
    host = str(base_url)
    return "api.openai.com" not in host


def install_schema_compat_patch() -> None:
    """Monkey-patch openai's chat completions create to inline ``$ref``.

    Idempotent. Safe to call from any provider's ``__init__``.

    sensitivity_tier: 1
    """
    global _PATCHED
    if _PATCHED:
        return
    try:
        from openai.resources.chat.completions import (  # type: ignore
            AsyncCompletions,
            Completions,
        )
    except ImportError:
        logger.debug(
            "openai package not importable — schema compat patch skipped",
        )
        return

    _patch_method(Completions, "create", is_async=False)
    _patch_method(AsyncCompletions, "create", is_async=True)
    _PATCHED = True
    logger.debug("Installed openai schema compat patch")


def _patch_method(cls: type, name: str, *, is_async: bool) -> None:
    original = getattr(cls, name)

    if is_async:
        async def wrapper(self: Any, *args: Any, **kw: Any) -> Any:
            try:
                client = getattr(self, "_client", None)
                if client is not None and _should_flatten(client):
                    _flatten_tools_kwargs(kw)
            except Exception:  # noqa: BLE001
                logger.debug(
                    "schema compat patch raised — passing through",
                    exc_info=True,
                )
            return await original(self, *args, **kw)
    else:
        def wrapper(self: Any, *args: Any, **kw: Any) -> Any:  # type: ignore[misc]
            try:
                client = getattr(self, "_client", None)
                if client is not None and _should_flatten(client):
                    _flatten_tools_kwargs(kw)
            except Exception:  # noqa: BLE001
                logger.debug(
                    "schema compat patch raised — passing through",
                    exc_info=True,
                )
            return original(self, *args, **kw)

    setattr(cls, name, wrapper)
