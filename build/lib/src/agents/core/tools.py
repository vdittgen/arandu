"""Central tool catalog for agents.

Agents do not import tool implementations directly — they declare the
tool ids they want (in their manifest or via user override) and the
catalog wires the callable into the underlying ``pydantic_ai.Agent``.

Each tool descriptor carries:

- ``id`` — stable identifier used in manifests and the editor UI
- ``description`` — surfaced as the tool's docstring to the LLM
- ``requires_tier`` — maximum sensitivity tier the tool may produce
  data from. Agents whose ``max_sensitivity_tier`` is below this value
  cannot enable the tool.
- ``categories`` — used to group tools in the UI ("data", "web",
  "files", "compute", "delegation").

This module is import-light: it registers descriptors but does not bind
``pydantic_ai`` tool callables until ``attach_to_agent()`` is called.

sensitivity_tier: 1
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolDescriptor:
    """One tool that an agent may enable.

    ``factory`` returns the callable that pydantic-ai will wrap. Factory
    pattern lets us inject per-run dependencies (DB engine, sandbox)
    without making the descriptor itself stateful.

    sensitivity_tier: 1
    """

    id: str
    description: str
    category: str
    requires_tier: int
    factory: Callable[..., Callable[..., object]]
    tags: tuple[str, ...] = field(default_factory=tuple)


class _ToolCatalog:
    """Thread-safe registry of tool descriptors.

    sensitivity_tier: 1
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tools: dict[str, ToolDescriptor] = {}

    def register(self, descriptor: ToolDescriptor) -> None:
        with self._lock:
            if descriptor.id in self._tools:
                msg = f"Tool already registered: {descriptor.id!r}"
                raise ValueError(msg)
            self._tools[descriptor.id] = descriptor

    def get(self, tool_id: str) -> ToolDescriptor | None:
        with self._lock:
            return self._tools.get(tool_id)

    def list_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._tools))

    def list_for_tier(self, max_tier: int) -> tuple[ToolDescriptor, ...]:
        with self._lock:
            return tuple(
                sorted(
                    (t for t in self._tools.values()
                     if t.requires_tier <= max_tier),
                    key=lambda t: t.id,
                ),
            )

    def reset_for_tests(self) -> None:
        with self._lock:
            self._tools.clear()


_catalog = _ToolCatalog()


def register_tool(descriptor: ToolDescriptor) -> None:
    """Register a new tool descriptor with the global catalog.

    sensitivity_tier: 1
    """
    _catalog.register(descriptor)


def get_tool(tool_id: str) -> ToolDescriptor | None:
    """Look up a tool descriptor by id.

    sensitivity_tier: 1
    """
    return _catalog.get(tool_id)


def list_tools(*, max_tier: int = 3) -> tuple[ToolDescriptor, ...]:
    """Return tool descriptors available up to ``max_tier``.

    sensitivity_tier: 1
    """
    return _catalog.list_for_tier(max_tier)


def reset_catalog_for_tests() -> None:
    """Drop all registered tools — for test isolation.

    sensitivity_tier: 1
    """
    _catalog.reset_for_tests()


__all__ = [
    "ToolDescriptor",
    "get_tool",
    "list_tools",
    "register_tool",
    "reset_catalog_for_tests",
]
