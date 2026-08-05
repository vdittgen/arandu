"""Event categoriser as an SBAgent.

Replaces the keyword-based ``CASE`` in ``int_events_enriched.sql``
with an LLM primitive. Given a calendar event (title, description,
location, attendees, time-of-day), return one of
``meeting``/``social``/``health``/``travel``/``other`` so the
domain marts (``mart_work``/``mart_personal``) can route the row.

sensitivity_tier: 2
"""

from src.agents.event_categorizer.agent import (
    DEFAULT_SYSTEM_PROMPT,
    EventCategorizerAgent,
    EventCategorizerDeps,
    register_event_categorizer_agent,
)

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "EventCategorizerAgent",
    "EventCategorizerDeps",
    "register_event_categorizer_agent",
]
