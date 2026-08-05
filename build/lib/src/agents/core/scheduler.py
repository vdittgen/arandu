"""LLM admission control with priority tiers and destination routing.

Every agent LLM call passes through the scheduler, which decides:

1. **When** the call may proceed (tier priority + concurrency cap).
2. **Where** the call goes — in Arandu every call routes to the
   local Ollama backend regardless of tier. The routing seam is a
   reserved extension point for alternate destinations.

Tiers (queue priority, lowest number wins):

- ``system`` — Firewall agents only. Bypasses queues; reserved capacity.
- ``interactive`` — Brain, reply handler, "Try" panel.
- ``proactive`` — message evaluator, notification orchestrator,
  insight generator on event.
- ``background`` — fact_learner, triage, labeler, deep agents, pipeline.

The scheduler is sync-first because most call sites today are sync.
A future async wrapper can sit on top once we move to pydantic-ai's
``run_stream`` in Phase 2.

sensitivity_tier: N/A (infrastructure; only stores tier metadata)
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Literal

logger = logging.getLogger(__name__)


class Tier(IntEnum):
    """Priority tier for an LLM request.

    Lower numeric value = higher priority. ``system`` requests skip
    queueing entirely (firewall must never be blocked by a background
    deep-agent run).

    sensitivity_tier: 1
    """

    SYSTEM = 0
    INTERACTIVE = 1
    PROACTIVE = 2
    BACKGROUND = 3


Route = Literal["remote", "local", "blocked"]


@dataclass(frozen=True)
class SchedulerConfig:
    """Tunable caps and timeouts per tier.

    Defaults are conservative and intended to be overridden via
    ``~/.arandu/settings.json`` in Phase 4.

    sensitivity_tier: 1
    """

    interactive_concurrency: int = 8
    proactive_concurrency: int = 4
    background_concurrency: int = 2
    interactive_timeout_s: float = 60.0
    proactive_timeout_s: float = 120.0
    background_timeout_s: float = 600.0
    system_timeout_s: float = 10.0
    # When True, refuse to start new background work while interactive
    # requests are waiting in queue.
    starve_background_on_interactive: bool = True


@dataclass
class _Permit:
    """Issued by ``LLMScheduler.acquire``.

    Carries the chosen route plus the timeout the caller should apply.

    sensitivity_tier: 1
    """

    tier: Tier
    route: Route
    timeout_s: float
    wait_ms: float
    _release: callable[..., None] = field(repr=False)

    def release(self) -> None:
        self._release()


class LLMScheduler:
    """Process-wide LLM admission controller.

    Thread-safe. Use as a singleton via ``default_scheduler()`` unless
    a test needs an isolated instance.

    sensitivity_tier: 1
    """

    def __init__(self, config: SchedulerConfig | None = None) -> None:
        self._cfg = config or SchedulerConfig()
        self._sems = {
            Tier.INTERACTIVE: threading.Semaphore(
                self._cfg.interactive_concurrency,
            ),
            Tier.PROACTIVE: threading.Semaphore(
                self._cfg.proactive_concurrency,
            ),
            Tier.BACKGROUND: threading.Semaphore(
                self._cfg.background_concurrency,
            ),
        }
        # Counter of interactive requests currently waiting for a slot.
        # Used to starve background admission when set.
        self._interactive_waiting = 0
        self._interactive_waiting_lock = threading.Lock()
        self._metrics: list[dict[str, float | int | str]] = []
        self._metrics_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @contextmanager
    def acquire(
        self,
        tier: Tier,
        *,
        route: Route = "remote",
        agent_id: str = "unknown",
    ) -> Iterator[_Permit]:
        """Block until a slot is available for ``tier``, then yield a permit.

        ``route`` is the destination decided upstream by ``EgressFirewall``;
        the scheduler does not classify content — it only enforces tier
        concurrency and records the chosen route in metrics.

        sensitivity_tier: 1
        """
        if route == "blocked":
            msg = "Cannot acquire scheduler permit for blocked egress"
            raise ValueError(msg)

        start = time.monotonic()

        if tier == Tier.SYSTEM:
            # Bypass queues entirely. System tier is exclusively the
            # firewall; it must never be deferred.
            yield _Permit(
                tier=tier,
                route=route,
                timeout_s=self._cfg.system_timeout_s,
                wait_ms=0.0,
                _release=lambda: None,
            )
            self._record(tier, route, agent_id, 0.0, time.monotonic() - start)
            return

        if tier == Tier.INTERACTIVE:
            with self._interactive_waiting_lock:
                self._interactive_waiting += 1
            try:
                self._sems[tier].acquire()
            finally:
                with self._interactive_waiting_lock:
                    self._interactive_waiting -= 1
        elif tier == Tier.BACKGROUND:
            self._wait_for_background_slot()
        else:
            self._sems[tier].acquire()

        wait_ms = (time.monotonic() - start) * 1000.0
        released = {"flag": False}

        def _release() -> None:
            if released["flag"]:
                return
            released["flag"] = True
            self._sems[tier].release()

        permit = _Permit(
            tier=tier,
            route=route,
            timeout_s=self._timeout_for(tier),
            wait_ms=wait_ms,
            _release=_release,
        )
        try:
            yield permit
        finally:
            _release()
            self._record(
                tier, route, agent_id, wait_ms, time.monotonic() - start,
            )

    def _wait_for_background_slot(self) -> None:
        """Acquire a background slot, deferring while interactive waits.

        sensitivity_tier: 1
        """
        sem = self._sems[Tier.BACKGROUND]
        if not self._cfg.starve_background_on_interactive:
            sem.acquire()
            return
        # Poll-then-acquire pattern: only attempt the semaphore when no
        # interactive callers are queued. This avoids holding a
        # background slot that an interactive request needs.
        while True:
            with self._interactive_waiting_lock:
                interactive_pending = self._interactive_waiting
            if interactive_pending == 0:
                if sem.acquire(timeout=0.1):
                    return
            else:
                time.sleep(0.05)

    def _timeout_for(self, tier: Tier) -> float:
        return {
            Tier.SYSTEM: self._cfg.system_timeout_s,
            Tier.INTERACTIVE: self._cfg.interactive_timeout_s,
            Tier.PROACTIVE: self._cfg.proactive_timeout_s,
            Tier.BACKGROUND: self._cfg.background_timeout_s,
        }[tier]

    def _record(
        self,
        tier: Tier,
        route: Route,
        agent_id: str,
        wait_ms: float,
        run_s: float,
    ) -> None:
        with self._metrics_lock:
            self._metrics.append({
                "tier": tier.name,
                "route": route,
                "agent_id": agent_id,
                "wait_ms": wait_ms,
                "run_ms": run_s * 1000.0,
                "ts": time.time(),
            })
            # Bounded in-memory ring; persistent storage lands in Phase 4.
            if len(self._metrics) > 1000:
                self._metrics = self._metrics[-1000:]

    def metrics_snapshot(self) -> list[dict[str, float | int | str]]:
        """Return a copy of the in-memory metrics buffer.

        sensitivity_tier: 1
        """
        with self._metrics_lock:
            return list(self._metrics)


_default_scheduler: LLMScheduler | None = None
_default_scheduler_lock = threading.Lock()


def default_scheduler() -> LLMScheduler:
    """Return the process-wide scheduler instance.

    sensitivity_tier: 1
    """
    global _default_scheduler
    if _default_scheduler is None:
        with _default_scheduler_lock:
            if _default_scheduler is None:
                _default_scheduler = LLMScheduler()
    return _default_scheduler


def reset_default_scheduler_for_tests(
    config: SchedulerConfig | None = None,
) -> LLMScheduler:
    """Replace the default scheduler — for tests that need isolation.

    sensitivity_tier: 1
    """
    global _default_scheduler
    with _default_scheduler_lock:
        _default_scheduler = LLMScheduler(config)
    return _default_scheduler


__all__ = [
    "LLMScheduler",
    "Route",
    "SchedulerConfig",
    "Tier",
    "default_scheduler",
    "reset_default_scheduler_for_tests",
]
