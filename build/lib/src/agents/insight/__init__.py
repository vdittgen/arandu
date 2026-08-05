"""Insight — DB persistence for proactive insight cards.

:class:`InsightGenerator` owns the orchestration logic — when to
surface which insight, which domain, which trigger — and the
``_insights`` table. It authors prose through :class:`BrainAgentV2`.

The former ``InsightAgent`` SBAgent wrapper was removed: nothing but
the eval adapter ever constructed it, and ``InsightGenerator`` never
delegated to it.

sensitivity_tier: 2
"""

from src.agents.insight.persistence import (
    Insight,
    InsightGenerator,
)

__all__ = [
    "Insight",
    "InsightGenerator",
]
