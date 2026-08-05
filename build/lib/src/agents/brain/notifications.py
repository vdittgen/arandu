"""Brain notification-preference helpers.

Pure functions that wrap :class:`PreferenceService` so Brain v2 can
expose them as a single pydantic-ai tool. The LLM decides when the user
is asking about notifications and calls
``update_notification_preferences`` with a structured action; the body
here translates that into the right ``PreferenceService`` mutation.

Replaces the legacy keyword-based ``_detect_notification_intent`` flow
that lived in ``src/agents/brain_agent.py`` (per CLAUDE.md "No keyword
filters — use LLM evals").

sensitivity_tier: 1 (preferences only — no Tier 2/3 data flows through)
"""

from __future__ import annotations

import logging
from typing import Any, Literal

logger = logging.getLogger(__name__)

NotificationAction = Literal[
    "show", "mute_all", "unmute", "enable", "disable",
]

_KNOWN_CATEGORIES: frozenset[str] = frozenset({
    "calendar_conflicts",
    "health_alerts",
    "action_results",
    "pipeline_summary",
})


def _format_preferences(prefs: Any) -> str:
    """Format the user's notification preferences for chat display.

    sensitivity_tier: 1
    """
    pref_list = prefs.get_preferences()
    if not pref_list:
        return (
            "You haven't set any notification preferences yet. "
            "Try asking me to notify you about calendar conflicts "
            "to get started."
        )
    lines = ["Your notification preferences:"]
    for pref in pref_list:
        label = pref.category.replace("_", " ")
        state = "on" if pref.enabled else "off"
        lines.append(f"  - {label}: {state}")
    return "\n".join(lines)


def apply_notification_action(
    prefs: Any,
    action: NotificationAction,
    category: str | None = None,
) -> str:
    """Apply ``action`` against ``prefs`` and return a user-facing message.

    Args:
        prefs: A :class:`PreferenceService` instance.
        action: One of ``show``, ``mute_all``, ``unmute``, ``enable``,
            ``disable``. The LLM is expected to pick the right value
            from the user's request.
        category: Required for ``enable`` / ``disable``. One of the
            keys in :data:`_KNOWN_CATEGORIES`. ``None`` returns a
            help string asking the user which category.

    sensitivity_tier: 1
    """
    if action == "show":
        return _format_preferences(prefs)
    if action == "mute_all":
        prefs.mute_all()
        return (
            "Done — all notifications muted for 24 hours. Ask me to "
            "unmute notifications when you want them back on."
        )
    if action == "unmute":
        prefs.unmute_all()
        return (
            "Notifications are back on. I'll send you alerts as "
            "configured."
        )
    if action in ("enable", "disable"):
        if not category:
            return (
                "Which notification category? I support: "
                "calendar conflicts, health alerts, action results, "
                "and pipeline summary."
            )
        if category not in _KNOWN_CATEGORIES:
            return (
                f"Unknown notification category: {category}. "
                "Supported: "
                + ", ".join(sorted(_KNOWN_CATEGORIES))
                + "."
            )
        enabled = action == "enable"
        prefs.update_preference(category, enabled=enabled)
        label = category.replace("_", " ")
        if enabled:
            return (
                f"Got it — I'll notify you about {label}. Ask me to "
                f"stop {label} notifications to turn them off."
            )
        return (
            f"Done — {label} notifications turned off. Ask me to "
            f"notify you about {label} to re-enable them."
        )
    return (
        "I can help with notification preferences. You can ask me to:\n"
        "- notify you about calendar conflicts / health alerts\n"
        "- stop health notifications\n"
        "- mute all notifications\n"
        "- show your notification preferences"
    )
