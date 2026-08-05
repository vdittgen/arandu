"""Deterministic extractor for user-given values in action requests.

The LLM-based parameter extractor is fundamentally unreliable for
values the user *literally typed* — quoted titles, explicit times,
named locations. Feeding it adjacent personal context made it worse
("Play Tennis with Tiago" → "Coffee chat with Sarah"). This module
sidesteps the LLM for the parts of an action request that can be
parsed deterministically.

The extractor returns a :class:`UserGivenValues` dataclass with the
fields the LLM extractor commonly hallucinates over. The caller is
expected to use these as *hard overrides* on the LLM's output — if
the user quoted a title, the proposal's title must be that quoted
string, regardless of what the LLM emitted.

Scope is deliberately narrow: this is not a general NLU layer. We
parse:

* Quoted strings (straight + smart quotes) as title candidates.
* "called X" / "titled X" / "named X" / "entitled X" patterns.
* Relative date / time phrases (today, tomorrow, weekday names) plus
  a 12 / 24-hour clock — no dateparser dependency, just enough to
  cover the calendar-creation surface.

Anything more ambitious belongs in a real NLU library; until we add
one, every additional pattern here needs an accompanying unit test.

sensitivity_tier: 1
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date as _date
from datetime import datetime, time, timedelta


@dataclass(frozen=True)
class UserGivenValues:
    """Values the user typed *verbatim* in the request.

    Each field is the user's literal text (or a deterministic
    derivation, in the case of dates). ``None`` means we couldn't
    extract that piece — the LLM (or a downstream default) handles it.

    sensitivity_tier: 2
    """

    title: str | None = None
    start_time: str | None = None  # ISO 8601, local timezone
    end_time: str | None = None    # ISO 8601, local timezone


# ----- quoted strings + "called X" -----------------------------------------

# Match straight (") and smart (" "), single (') and smart (' '), backticks.
_QUOTE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r'"([^"\n]+)"'),
    re.compile(r"“([^”\n]+)”"),   # “ ”
    re.compile(r"'([^'\n]+)'"),
    re.compile(r"‘([^’\n]+)’"),   # ‘ ’
    re.compile(r"`([^`\n]+)`"),
)


# Verbs that introduce a title-like phrase up to a sentence boundary.
_CALLED_PATTERN = re.compile(
    r"\b(?:called|titled|named|entitled)\s+(?P<title>[^.,!?\n]+)",
    re.IGNORECASE,
)

# Words that signal "the title ended here and a date/time clause is
# starting" — used to truncate a ``called X`` capture.
_TITLE_TERMINATORS = re.compile(
    r"\b("
    r"today|tomorrow|amanh[ãa]|hoje"
    r"|this|next"
    r"|on|at|by|for|from|until"
    r"|monday|tuesday|wednesday|thursday|friday|saturday|sunday"
    r"|mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun"
    r"|noon|midnight"
    r"|\d{1,2}(?::\d{2})?\s*(?:am|pm)?"
    r"|\d{4}-\d{2}-\d{2}"
    r")\b",
    re.IGNORECASE,
)


def _extract_title(question: str) -> str | None:
    """Return the most-explicit user-supplied title, or ``None``.

    Priority: quoted strings beat ``called X`` phrases because a user
    who typed quotes around a title is being maximally explicit.

    sensitivity_tier: 1
    """
    for pat in _QUOTE_PATTERNS:
        match = pat.search(question)
        if match:
            value = match.group(1).strip()
            if value:
                return value
    match = _CALLED_PATTERN.search(question)
    if match:
        candidate = match.group("title").strip()
        # Truncate at the first date/time terminator so phrases like
        # ``called Quarterly review next Monday 3pm`` reduce to just
        # ``Quarterly review``. We probe each word from left to right
        # and stop at the first one that matches a terminator.
        words = candidate.split()
        keep: list[str] = []
        for word in words:
            if _TITLE_TERMINATORS.fullmatch(word):
                break
            keep.append(word)
        candidate = " ".join(keep) if keep else candidate
        return candidate.strip(" \"'`") or None
    return None


# ----- date / time ----------------------------------------------------------

_WEEKDAYS: dict[str, int] = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
    # Common abbreviations
    "mon": 0, "tue": 1, "tues": 1, "wed": 2, "thu": 3, "thur": 3, "thurs": 3,
    "fri": 4, "sat": 5, "sun": 6,
}


def _next_weekday(today: _date, target: int, *, this_week: bool) -> _date:
    """Return the next occurrence of ``target`` weekday.

    ``this_week=True`` means "this Monday" (could be today); ``False``
    means "next Monday" — always at least 7 days out if today is the
    target weekday, otherwise the upcoming one.

    sensitivity_tier: 1
    """
    days_ahead = (target - today.weekday()) % 7
    if days_ahead == 0 and not this_week:
        days_ahead = 7
    if not this_week and days_ahead < 7:
        # "next Friday" said on a Wednesday → upcoming Friday is
        # ambiguous in English; default to the closer one (matches
        # the most common reading).
        pass
    return today + timedelta(days=days_ahead)


def _parse_day_reference(
    question: str, today: _date,
) -> _date | None:
    """Return the date the user referred to, or ``None``.

    Handles: today, tomorrow, day after tomorrow, weekday names,
    "this <weekday>", "next <weekday>", and explicit ISO dates.

    sensitivity_tier: 1
    """
    q = question.lower()

    # Explicit ISO date wins over relative words ("2026-05-23").
    iso_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", question)
    if iso_match:
        try:
            return _date.fromisoformat(iso_match.group(1))
        except ValueError:
            pass

    if re.search(r"\bday after tomorrow\b", q):
        return today + timedelta(days=2)
    if re.search(r"\btomorrow\b|\bamanh[ãa]\b", q):
        return today + timedelta(days=1)
    if re.search(r"\btoday\b|\bhoje\b", q):
        return today

    # "next monday" / "this friday" — most specific first.
    next_match = re.search(
        r"\bnext\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday"
        r"|mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)\b",
        q,
    )
    if next_match:
        return _next_weekday(
            today, _WEEKDAYS[next_match.group(1)], this_week=False,
        )
    this_match = re.search(
        r"\bthis\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday"
        r"|mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)\b",
        q,
    )
    if this_match:
        return _next_weekday(
            today, _WEEKDAYS[this_match.group(1)], this_week=True,
        )
    # Bare weekday — interpret as "the next one", which matches
    # English usage ("Let's meet Monday" said midweek).
    bare = re.search(
        r"\bon\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        q,
    )
    if bare:
        return _next_weekday(
            today, _WEEKDAYS[bare.group(1)], this_week=True,
        )
    return None


# Matches "7am", "7 am", "7:30 pm", "19:00", "noon", "midnight".
_TIME_PATTERN = re.compile(
    r"""
    \b(?:
        (?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?  \s*(?P<period>am|pm)\b
        |
        (?P<hour24>\d{1,2}):(?P<minute24>\d{2})\b
        |
        (?P<noon>noon|midnight)\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _parse_time_reference(question: str) -> time | None:
    """Return the first time-of-day mentioned, or ``None``.

    Picks the first match so "tomorrow 7am" returns 07:00 even when a
    later "until 9am" appears.

    sensitivity_tier: 1
    """
    match = _TIME_PATTERN.search(question)
    if not match:
        return None
    if match.group("noon"):
        return time(12, 0) if match.group("noon").lower() == "noon" else time(0, 0)
    if match.group("hour"):
        hour = int(match.group("hour"))
        minute = int(match.group("minute") or 0)
        period = (match.group("period") or "").lower()
        if period == "pm" and hour < 12:
            hour += 12
        elif period == "am" and hour == 12:
            hour = 0
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return time(hour, minute)
    if match.group("hour24"):
        hour = int(match.group("hour24"))
        minute = int(match.group("minute24"))
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return time(hour, minute)
    return None


def _iso_local(dt: datetime) -> str:
    """Format ``dt`` as an ISO 8601 string in its current tzinfo.

    Drops microseconds for readability. Calendar APIs only care about
    second-level precision.

    sensitivity_tier: 1
    """
    return dt.replace(microsecond=0).isoformat()


# ----- public surface -------------------------------------------------------


def extract_user_given_values(
    question: str,
    *,
    today: _date | None = None,
    default_duration_minutes: int = 60,
) -> UserGivenValues:
    """Pull literal user values out of an action request.

    ``today`` defaults to the system date; tests inject a fixed value
    so the date-math assertions stay stable.

    sensitivity_tier: 2
    """
    if not question:
        return UserGivenValues()
    today = today or _date.today()

    title = _extract_title(question)
    day = _parse_day_reference(question, today)
    clock = _parse_time_reference(question)

    start_iso: str | None = None
    end_iso: str | None = None
    if day is not None or clock is not None:
        # Combine whatever we have. If only a clock was found, default
        # to today; if only a day, default to 9 AM (a sensible "we
        # don't know" for a meeting). The downstream LLM may refine
        # this but our override only fires when we have something
        # actually useful.
        anchor_day = day or today
        anchor_clock = clock or time(9, 0)
        start_dt = datetime.combine(anchor_day, anchor_clock)
        end_dt = start_dt + timedelta(minutes=default_duration_minutes)
        start_iso = _iso_local(start_dt)
        end_iso = _iso_local(end_dt)

    return UserGivenValues(
        title=title,
        start_time=start_iso,
        end_time=end_iso,
    )


__all__ = ["UserGivenValues", "extract_user_given_values"]
