"""ContextVar-based lane attribution for outbound LLM calls.

Agent entry points (e.g. :class:`AgentContext`, brain action handlers)
open a :func:`lane_scope` that sets the current lane in a
:class:`contextvars.ContextVar`. The :class:`MeteredLLMProvider`
decorator reads that value after each call to attribute it to the
right lane in ``spend_calls.lane``.

A ContextVar (rather than a thread-local) survives across asyncio
tasks correctly and inherits into ``contextvars.copy_context()``
worker callbacks, so background pipeline tasks that route their
work through helper threads still see the right lane.

sensitivity_tier: 1
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from src.agents.firewall.egress_firewall import Lane

_CURRENT_LANE: ContextVar[Lane | None] = ContextVar(
    "arandu.current_lane", default=None,
)


def current_lane() -> Lane | None:
    """Return the lane bound to the current execution context, if any.

    sensitivity_tier: 1
    """
    return _CURRENT_LANE.get()


@contextmanager
def lane_scope(lane: Lane) -> Iterator[None]:
    """Bind ``lane`` to the current context for the duration of the block.

    Restores the previous value on exit, even if the block raises.

    sensitivity_tier: 1
    """
    token = _CURRENT_LANE.set(lane)
    try:
        yield
    finally:
        _CURRENT_LANE.reset(token)


__all__ = ["current_lane", "lane_scope"]
