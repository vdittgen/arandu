"""Relationship Tracker agent — identifies contacts you haven't
interacted with recently and generates follow-up nudges.

sensitivity_tier: 2 (reads contact and message data)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from src.agent_runtime.base import BrainAgent
from src.agent_runtime.context import AgentContext
from src.agent_runtime.models import AgentResult

# Contacts not interacted with for this many days trigger a nudge.
STALE_THRESHOLD_DAYS = 30


class RelationshipTrackerAgent(BrainAgent):
    """Finds contacts the user hasn't interacted with recently
    and generates gentle follow-up nudges.

    sensitivity_tier: 2
    """

    manifest = None  # Loaded by AgentRunner from manifest.yaml

    def run(self, context: AgentContext) -> AgentResult:
        """Query contacts with stale interactions, generate nudges.

        sensitivity_tier: 2
        """
        context.log("Starting relationship tracker")

        cutoff = (
            datetime.now(tz=timezone.utc) - timedelta(days=STALE_THRESHOLD_DAYS)
        ).isoformat()

        # Find contacts with old last_contact.
        stale_contacts = context.query(
            f"SELECT id, name, relationship, last_contact "
            f"FROM raw_contacts "
            f"WHERE last_contact < '{cutoff}' OR last_contact IS NULL "
            f"ORDER BY last_contact ASC LIMIT 20",
        )
        context.log(f"Found {len(stale_contacts)} stale contacts")

        if not stale_contacts:
            context.log("No stale contacts found")
            return AgentResult(
                agent_id="relationship-tracker",
                status="success",
                output="No follow-ups needed.",
            )

        # Generate nudges.
        nudges: list[dict] = []
        for contact in stale_contacts:
            name = contact.get("name", "Someone")
            relationship = contact.get("relationship", "contact")
            last_contact = contact.get("last_contact")

            days_since = _days_since(last_contact)
            nudge_text = _generate_nudge(name, relationship, days_since)

            # Try LLM for a more personal nudge.
            llm_nudge = context.ask_llm(
                f"Write a brief, warm reminder (1-2 sentences) to reach out to "
                f"{name} ({relationship}). It's been {days_since} days since "
                f"last contact. Be friendly, not pushy.",
            )
            if llm_nudge:
                nudge_text = llm_nudge

            nudges.append({
                "id": str(uuid.uuid4()),
                "contact_id": contact.get("id", ""),
                "contact_name": name,
                "relationship": relationship,
                "days_since_contact": days_since,
                "nudge": nudge_text,
            })

        context.write("ext_relationship_tracker_nudges", nudges)
        context.log(f"Generated {len(nudges)} nudges")

        return AgentResult(
            agent_id="relationship-tracker",
            status="success",
            output=f"Generated {len(nudges)} follow-up nudges.",
        )


def _days_since(last_contact: str | None) -> int:
    """Calculate days since last contact.

    sensitivity_tier: 1
    """
    if not last_contact:
        return 999
    try:
        dt = datetime.fromisoformat(str(last_contact))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = datetime.now(tz=timezone.utc) - dt
        return max(0, delta.days)
    except (ValueError, TypeError):
        return 999


def _generate_nudge(name: str, relationship: str, days_since: int) -> str:
    """Generate a simple fallback nudge without LLM.

    sensitivity_tier: 1
    """
    if days_since > 90:
        return (
            f"It's been over {days_since} days since you "
            f"talked to {name}. Maybe send a quick hello?"
        )
    return (
        f"You haven't connected with {name} "
        f"in {days_since} days. Time for a catch-up?"
    )
