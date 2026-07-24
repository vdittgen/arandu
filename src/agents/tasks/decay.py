"""Goal decay — fade and archive brain-mined goals that stop being
confirmed by fresh evidence.

The goal miner bumps ``last_confirmed_at`` every time recent evidence
(messages, notes, facts, chats) re-surfaces a goal. A brain-mined goal
whose evidence dries up keeps an increasingly stale timestamp; reading
that lets the app fade the goal in ranking and, past a longer
threshold, auto-archive it — reversibly.

User-entered goals never decay: the user owns them, and the absence of
message/note evidence is not a signal to fade a goal someone set by
hand. Only ``source == "brain"`` goals are subject to decay.

This module is pure (no DB) so the thresholds and the ranking penalty
are testable in isolation; :class:`~src.agents.tasks.curator.TaskCurator`
applies it against the ``_goals`` table.

sensitivity_tier: 1 (timestamps + status only)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

# A brain-mined goal starts fading once its evidence is this old, and is
# auto-archived once it crosses the longer threshold. Days, not "cycles":
# the mining cadence varies (manual runs, proactive schedule), so wall-
# clock staleness is the robust proxy for "unconfirmed for N cycles".
FADE_AFTER_DAYS = 14
ARCHIVE_AFTER_DAYS = 30

# Urgency points subtracted at full decay, so a fading goal ranks below
# an equally-scored fresh one and drops steadily toward the archive line.
MAX_PENALTY = 10

# Decay states, ordered by increasing staleness.
FRESH = "fresh"
FADING = "fading"
STALE = "stale"


@dataclass(frozen=True)
class DecayState:
    """The decay verdict for a single goal.

    ``state`` is one of ``fresh`` / ``fading`` / ``stale``;
    ``days_since_confirmed`` is the age of the evidence anchor; and
    ``penalty`` is the non-negative urgency deduction to apply when
    ranking (0 while fresh, ramping to :data:`MAX_PENALTY` at the
    archive threshold).

    sensitivity_tier: 1
    """

    state: str
    days_since_confirmed: int
    penalty: int


def _parse_iso(ts: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp to an aware UTC datetime, or None.

    sensitivity_tier: 1
    """
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def goal_decay(
    *,
    source: str,
    last_confirmed_at: str | None,
    created_at: str | None,
    now: datetime | None = None,
    fade_after_days: int = FADE_AFTER_DAYS,
    archive_after_days: int = ARCHIVE_AFTER_DAYS,
) -> DecayState:
    """Compute the decay state for a goal.

    Anchors on ``last_confirmed_at`` (falling back to ``created_at`` for
    a goal never re-confirmed). User goals — and goals with no parseable
    anchor — are always :data:`FRESH` with no penalty.

    sensitivity_tier: 1
    """
    if source != "brain":
        return DecayState(FRESH, 0, 0)

    now = now or datetime.now(timezone.utc)
    anchor = _parse_iso(last_confirmed_at) or _parse_iso(created_at)
    if anchor is None:
        return DecayState(FRESH, 0, 0)

    days = max(0, (now - anchor).days)

    if days >= archive_after_days:
        state = STALE
    elif days >= fade_after_days:
        state = FADING
    else:
        state = FRESH

    if days < fade_after_days:
        penalty = 0
    else:
        span = max(1, archive_after_days - fade_after_days)
        frac = min(1.0, (days - fade_after_days) / span)
        penalty = round(frac * MAX_PENALTY)

    return DecayState(state, days, penalty)


def should_archive(decay: DecayState) -> bool:
    """Whether a goal in this decay state should be auto-archived.

    sensitivity_tier: 1
    """
    return decay.state == STALE
