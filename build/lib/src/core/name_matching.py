"""Whole-word name matching for contact/entity resolution.

Contact names must never be matched as bare substrings. ``"ana"`` is a
substring of *Mariana*, *Susana* and *banana*, so a substring test
silently resolves a mention of one person to a different contact — the
"wrong contact on a task" class of bug. Everything that maps free text
onto a known contact goes through :func:`contains_name` so the rule is
enforced in one place.

Boundaries use ``(?<!\\w)`` / ``(?!\\w)`` rather than ``\\b`` because a
name may begin or end with a non-word character (``O'Brien``,
``Ana-Paula``), where ``\\b`` asserts the opposite of what's wanted.
Matching is Unicode-aware, so accented names (``José``, ``Conceição``)
get real boundaries instead of splitting mid-word.

sensitivity_tier: 1 (pure functions on text; no I/O)
"""

from __future__ import annotations

import re
from functools import lru_cache

__all__ = ["build_unique_index", "contains_name", "find_names"]


@lru_cache(maxsize=4096)
def _word_pattern(name: str) -> re.Pattern[str]:
    """Compile (and cache) a whole-word matcher for ``name``.

    Cached because the same contact names are re-matched on every query
    — recompiling per call showed up as avoidable work in retrieval.

    sensitivity_tier: 1
    """
    return re.compile(
        r"(?<!\w)" + re.escape(name) + r"(?!\w)",
        re.IGNORECASE,
    )


def contains_name(name: str, text: str) -> bool:
    """True when ``name`` occurs in ``text`` as a whole word.

    Case-insensitive. Empty names never match — an empty pattern would
    otherwise match at every position.

    sensitivity_tier: 1
    """
    if not name or not text:
        return False
    return _word_pattern(name).search(text) is not None


def find_names(names: list[str], text: str) -> list[str]:
    """Return the ``names`` occurring in ``text``, longest first.

    Longest-first ordering means a full name (``"ana paula"``) is
    reported before a bare first name (``"ana"``) that overlaps it, so
    callers resolving one entity per mention pick the more specific
    contact.

    sensitivity_tier: 1
    """
    ordered = sorted(set(names), key=lambda n: (-len(n), n))
    return [n for n in ordered if contains_name(n, text)]


def build_unique_index(
    variants: dict[str, set[str]],
) -> dict[str, str]:
    """Collapse ``variant -> {owner ids}`` to only unambiguous variants.

    A variant claimed by two or more people (two contacts named *Ana*,
    both contributing the token ``"ana"``) is dropped rather than
    awarded to whichever was indexed first. Resolving such a mention to
    *an* Ana is worse than not resolving it: the caller can degrade to
    no contact, but cannot detect a confidently wrong one.

    sensitivity_tier: 1
    """
    return {
        variant: next(iter(owners))
        for variant, owners in variants.items()
        if len(owners) == 1
    }
