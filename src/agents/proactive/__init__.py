"""Proactive intelligence — DB-backed orchestrator.

:class:`ProactiveIntelligence` (re-exported from :mod:`.persistence`)
owns the four proactive-evaluation pillars: pending replies, contact
contexts, actionable events, and topic digests. Persistence lives in
``_pending_replies``, ``_contact_contexts``, ``_actionable_events``,
and ``_proactive_state`` (data fingerprint cache).

Relocated from ``src/agents/proactive_intelligence.py`` in Phase E.

sensitivity_tier: 3
"""

from src.agents.proactive.persistence import (
    ActionableEvent,
    ContactContext,
    MeetingPrepAttendee,
    MeetingPrepBrief,
    PendingReply,
    ProactiveIntelligence,
    ProactiveResult,
    TopicDigestEntry,
)

__all__ = [
    "ActionableEvent",
    "ContactContext",
    "MeetingPrepAttendee",
    "MeetingPrepBrief",
    "PendingReply",
    "ProactiveIntelligence",
    "ProactiveResult",
    "TopicDigestEntry",
]
