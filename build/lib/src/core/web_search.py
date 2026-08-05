"""Web search fallback for BrainAgent.

Uses DuckDuckGo (free, no API key) to search the web when personal
context is insufficient to answer a question.  Results are ephemeral —
injected into the LLM prompt only, never stored in any database.

PRIVACY:
- Only triggers when personal context has < 2 results (general questions).
- ``is_personal_question()`` blocks search for personal pronouns.
- DuckDuckGo has no tracking, no cookies, no API key.

sensitivity_tier: 1 (search queries are general knowledge, not personal)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Maximum search results to retrieve.
MAX_RESULTS = 5

# Maximum characters of web context to inject into the LLM prompt.
MAX_WEB_CONTEXT_CHARS = 4000

# Patterns suggesting a personal question (should NOT trigger web search).
_PERSONAL_PATTERNS = re.compile(
    r"\b(my|mine|i have|i am|i was|i will|i'm|i've|"
    r"me|myself|our|we|us)\b",
    re.IGNORECASE,
)


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------


@dataclass(frozen=True)
class WebSearchResult:
    """A single web search result.

    sensitivity_tier: 1
    """

    title: str
    body: str
    url: str


@dataclass(frozen=True)
class WebSearchResponse:
    """Collection of web search results.

    sensitivity_tier: 1
    """

    query: str
    results: list[WebSearchResult] = field(default_factory=list)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def is_personal_question(question: str) -> bool:
    """Check whether *question* appears to be about personal data.

    Returns ``True`` when the question contains personal pronouns or
    references that suggest it's about the user's own data rather
    than general knowledge.

    sensitivity_tier: 1
    """
    return bool(_PERSONAL_PATTERNS.search(question))


def search_web(
    query: str,
    max_results: int = MAX_RESULTS,
) -> WebSearchResponse:
    """Search the web via DuckDuckGo.

    Appends the current year to queries and uses a month-based time
    filter to prioritise recent results (avoids stale articles for
    sports scores, news, events, etc.).

    Returns an empty response on **any** failure (network, rate-limit,
    missing package) so that search problems never degrade the
    BrainAgent experience.

    sensitivity_tier: 1
    """
    try:
        try:
            from ddgs import DDGS  # preferred (renamed package)
        except ImportError:
            from duckduckgo_search import DDGS  # fallback

        from datetime import datetime as _dt
        from datetime import timezone as _tz

        current_year = str(_dt.now(tz=_tz.utc).year)

        # Normalize smart/curly quotes and remove problematic
        # apostrophes — DuckDuckGo returns empty results for queries
        # containing certain quote characters.
        normalized = (
            query.replace("\u2018", "")
            .replace("\u2019", "")
            .replace("\u201c", "")
            .replace("\u201d", "")
            .replace("'", "")
        )

        # Append current year to bias DuckDuckGo toward recent results
        # (prevents stale articles for time-sensitive queries like
        # sports, news, events).
        if current_year not in normalized:
            normalized = f"{normalized} {current_year}"

        with DDGS() as ddgs:
            # Try month-limited search first for freshness.
            raw_results = list(
                ddgs.text(
                    normalized,
                    max_results=max_results,
                    timelimit="m",
                ),
            )
            # Fall back to unlimited if no recent results found.
            if not raw_results:
                raw_results = list(
                    ddgs.text(normalized, max_results=max_results),
                )

        results = [
            WebSearchResult(
                title=r.get("title", ""),
                body=r.get("body", ""),
                url=r.get("href", ""),
            )
            for r in raw_results
        ]
        return WebSearchResponse(query=query, results=results)

    except ImportError:
        logger.warning(
            "duckduckgo-search not installed — web search disabled",
        )
        return WebSearchResponse(query=query)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Web search failed: %s", exc)
        return WebSearchResponse(query=query)


def format_web_results(response: WebSearchResponse) -> str:
    """Format web search results into context text for the LLM prompt.

    sensitivity_tier: 1
    """
    if not response.results:
        return ""

    lines = ["--- Web Search Results ---"]
    for r in response.results:
        lines.append(f"[WEB] {r.title}")
        lines.append(f"  {r.body}")
        lines.append(f"  Source: {r.url}")
        lines.append("")

    text = "\n".join(lines)
    if len(text) > MAX_WEB_CONTEXT_CHARS:
        text = text[:MAX_WEB_CONTEXT_CHARS] + "\n[... truncated]"
    return text


def web_results_to_sources(
    response: WebSearchResponse,
) -> list[dict[str, Any]]:
    """Convert web search results into BrainAgent source dicts.

    Follows the same structure as vector/graph/structured sources so
    the frontend ``SourcesSection`` can display them.

    sensitivity_tier: 1
    """
    return [
        {
            "id": f"web-{i}",
            "type": "web",
            "content": r.title,
            "url": r.url,
            "sensitivity_tier": 1,
        }
        for i, r in enumerate(response.results)
    ]
