"""Pydantic AI single-event categoriser.

Replaces the keyword-based ``CASE`` expression that previously lived
in ``int_events_enriched.sql``. Given one calendar event's metadata,
the agent returns an :class:`EventCategoryDecision` whose
``category`` field is one of the five values that the downstream
marts (``mart_work``, ``mart_personal``) expect.

Pipeline-only: the orchestration layer
(``src/pipeline/intermediate/int_events_enriched.py``) batches calls,
caches per-event verdicts, and falls back to ``'other'`` when the
LLM is unavailable.

sensitivity_tier: 2
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.agents.core.agent_base import SBAgent
from src.agents.core.output_types import EventCategoryDecision
from src.agents.core.scheduler import Tier

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """\
You categorise ONE calendar event into a single bucket from a fixed list.

Return an EventCategoryDecision with ``category`` chosen from:

- "meeting" — anything work-related: standups, syncs, 1-on-1s, reviews,
  product/UX discussions, refinements, customer calls, internal calls
  with colleagues, demos, planning sessions, working sessions. If the
  event looks like it could plausibly be on a working person's
  professional calendar, choose "meeting".
- "social" — personal gatherings: dinners with friends, parties,
  concerts, weddings, drinks, casual lunches with non-colleagues.
- "health" — therapy, doctor, dentist, hospital, physical exam,
  medical appointments, gym classes that are clearly health-driven.
- "travel" — flights, train rides, road trips, the act of going
  somewhere (NOT a destination meeting once you're there).
- "other" — only when none of the above plausibly applies. Examples:
  birthdays you're observing, public holidays, all-day reminders.

Rules:
- Default toward "meeting" when the title looks professional but is
  ambiguous (e.g. "Refinement", "Sync", "Blocked", names you don't
  recognise). Most calendar entries on a working person's calendar
  are work meetings; only pull them out when there's a clear signal
  otherwise.
- Use ``reason`` for ONE short sentence explaining the choice. Keep
  it under 120 characters.\
"""


@dataclass(frozen=True)
class EventCategorizerDeps:
    """Typed input bundle for :class:`EventCategorizerAgent`.

    Only the fields that influence the verdict are surfaced — we
    deliberately omit the event id so the LLM cannot anchor its
    answer to identifiers.

    sensitivity_tier: 2
    """

    title: str
    description: str = ""
    location: str = ""
    start_time: str = ""
    attendees: str = ""
    attendee_names: str = ""


def _truncate(value: str, limit: int) -> str:
    """Cap a free-text field so prompts stay bounded.

    sensitivity_tier: 2
    """
    text = (value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


class EventCategorizerAgent(
    SBAgent[EventCategorizerDeps | str, EventCategoryDecision],
):
    """Classify one calendar event into the marts' category vocabulary.

    sensitivity_tier: 2
    """

    agent_id = "event_categorizer"
    output_type = EventCategoryDecision
    tier = Tier.BACKGROUND
    system_prompt = DEFAULT_SYSTEM_PROMPT

    def build_prompt(
        self, deps: EventCategorizerDeps | str,
    ) -> str:
        """Render deps into a short natural-language description.

        Phrased as prose rather than colon-separated key/value lines
        so the injection firewall doesn't mistake it for template
        injection — the firewall's heuristic flags structured
        ``Field: value`` patterns as suspicious.

        sensitivity_tier: 2
        """
        if isinstance(deps, str):
            return deps
        title = _truncate(deps.title, 200) or "an untitled event"
        sentence = f"A calendar event called {title!r}"
        if deps.start_time:
            sentence += f" starting at {_truncate(deps.start_time, 40)}"
        if deps.location:
            sentence += f" at {_truncate(deps.location, 200)}"
        names = (
            deps.attendee_names or deps.attendees or ""
        )
        if names:
            sentence += (
                f" with attendees {_truncate(names, 400)}"
            )
        sentence += "."
        if deps.description:
            sentence += (
                f" Description: {_truncate(deps.description, 600)}"
            )
        sentence += (
            " Pick the best matching category for this event."
        )
        return sentence

    def categorize(
        self,
        *,
        title: str,
        description: str = "",
        location: str = "",
        start_time: str = "",
        attendees: str = "",
        attendee_names: str = "",
    ) -> EventCategoryDecision | None:
        """Convenience entrypoint mirroring the call shape callers want.

        Returns ``None`` on agent failure so the orchestrator can apply
        its own fallback (typically ``'other'``).

        sensitivity_tier: 2
        """
        deps = EventCategorizerDeps(
            title=title,
            description=description,
            location=location,
            start_time=start_time,
            attendees=attendees,
            attendee_names=attendee_names,
        )
        record = self.run(deps)
        if record.output is None or record.error is not None:
            return None
        return record.output


def register_event_categorizer_agent() -> None:
    """Register the event categoriser in the global agent registry.

    sensitivity_tier: 1
    """
    from src.agents.core.config_store import AgentConfig
    from src.agents.core.registry import (
        AgentDefinition,
        get_agent,
        register_agent,
    )

    if get_agent("event_categorizer") is not None:
        return

    default = AgentConfig(
        agent_id="event_categorizer",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        model_route="inherit",
        model_override=None,
        enabled_tools=(),
        enabled_skills=(),
        editable=False,
    )
    register_agent(AgentDefinition(
        agent_id="event_categorizer",
        name="Event Categorizer",
        description=(
            "Classifies one calendar event into the closed "
            "vocabulary used by the domain marts "
            "(meeting/social/health/travel/other). Runs inside the "
            "pipeline; not directly delegated by Brain."
        ),
        category="classifier",
        parent_agent="brain",
        tier=Tier.BACKGROUND,
        max_sensitivity_tier=2,
        editable=False,
        default_config=default,
        available_tools=(),
        available_skills=(),
        output_schema="EventCategoryDecision",
        pattern="single",
        factory=EventCategorizerAgent,
        tags=("classifier", "indirect"),
    ))


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "EventCategorizerAgent",
    "EventCategorizerDeps",
    "register_event_categorizer_agent",
]
