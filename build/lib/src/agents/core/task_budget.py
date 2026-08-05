"""Wall-clock budgets and self-review cadence for orchestrator runs.

Orchestrators (Brain, Chat) used to enforce a hard ``tool_calls_limit``
that raised ``UsageLimitExceeded`` inside pydantic-ai. That cap was too
coarse: the same value gated a 1-second chat answer and a deep daily
brief. This module replaces the cap with a *time-based* budget plus a
reflection cadence; the actual stop conditions live in
:mod:`src.agents.core.reflection` (self-review verdict) and
:mod:`src.agents.core.cancel_registry` (user cancel).

Task classes
------------
- :data:`TaskClass.INTERACTIVE_FAST` — chat/brain defaults. First
  reflection at 10s, follow-ups every 15s if the run is promoted.
- :data:`TaskClass.INTERACTIVE_DEEP` — promoted from FAST by the
  Reflector when the question is judged complex. Same cadence but a
  larger ``expected_total_s`` so the UI shows a longer progress hint.
- :data:`TaskClass.BACKGROUND_DEEP` — explicit caller opt-in for tasks
  the user expects to take time (daily brief, weekly digest, deep
  research). First reflection at 30s, then every 60s.

No hard ceiling
---------------
There is no time after which a run is *forced* to abort. The only
ways a run terminates are:

1. The model produces a final answer (the natural pydantic-ai
   ``End`` node).
2. Self-review returns ``continue_=False`` and the reflective runner
   injects a ``STOP_REQUEST`` user message; the model finalizes in
   the next ``ModelRequestNode``.
3. The user calls ``stop_research`` via the IPC channel, which sets
   the cancel event; the runner injects the same ``STOP_REQUEST`` at
   the next checkpoint.

sensitivity_tier: 1
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class TaskClass(StrEnum):
    """Categorical label that drives reflection cadence and UI hints.

    Strings are stable wire values — emitted in stream events and
    serialized in audit-chain entries — so don't rename without a
    migration.

    sensitivity_tier: 1
    """

    INTERACTIVE_FAST = "interactive_fast"
    INTERACTIVE_DEEP = "interactive_deep"
    BACKGROUND_DEEP = "background_deep"


@dataclass(frozen=True)
class TaskBudget:
    """Wall-clock budget for one orchestrator run.

    Attributes
    ----------
    task_class:
        Current class. May be promoted mid-run by the Reflector — the
        runner tracks the *active* class separately from the
        ``TaskBudget`` it was constructed with, since promotion swaps
        in the deeper class's cadence.
    first_reflect_at_s:
        Seconds elapsed before the first self-review fires.
    reflect_interval_s:
        Seconds between subsequent reviews after the first one (or
        after a promotion).
    expected_total_s:
        UI progress hint, not a deadline. ``Researching... ~30s``.

    sensitivity_tier: 1
    """

    task_class: TaskClass
    first_reflect_at_s: float
    reflect_interval_s: float
    expected_total_s: float

    @classmethod
    def interactive_fast(cls) -> TaskBudget:
        """Default for user-facing chat/brain questions.

        sensitivity_tier: 1
        """
        return cls(
            task_class=TaskClass.INTERACTIVE_FAST,
            first_reflect_at_s=10.0,
            reflect_interval_s=10.0,
            expected_total_s=10.0,
        )

    @classmethod
    def interactive_deep(cls) -> TaskBudget:
        """Promoted class for questions reflection flags as complex.

        sensitivity_tier: 1
        """
        return cls(
            task_class=TaskClass.INTERACTIVE_DEEP,
            first_reflect_at_s=10.0,
            reflect_interval_s=15.0,
            expected_total_s=60.0,
        )

    @classmethod
    def background_deep(cls) -> TaskBudget:
        """Explicit deep-research budget (daily brief, weekly digest).

        sensitivity_tier: 1
        """
        return cls(
            task_class=TaskClass.BACKGROUND_DEEP,
            first_reflect_at_s=30.0,
            reflect_interval_s=60.0,
            expected_total_s=180.0,
        )

    @classmethod
    def for_class(cls, task_class: TaskClass) -> TaskBudget:
        """Return the canonical budget for ``task_class``.

        Used by the reflective runner when a verdict promotes the
        current class — it swaps the active budget to the promoted
        one's cadence/hint.

        sensitivity_tier: 1
        """
        if task_class is TaskClass.INTERACTIVE_FAST:
            return cls.interactive_fast()
        if task_class is TaskClass.INTERACTIVE_DEEP:
            return cls.interactive_deep()
        if task_class is TaskClass.BACKGROUND_DEEP:
            return cls.background_deep()
        msg = f"unknown TaskClass: {task_class!r}"
        raise ValueError(msg)


__all__ = ["TaskBudget", "TaskClass"]
