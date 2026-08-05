"""Learning-loop insight generator.

Generates proactive insights based on question patterns detected in
the user's query history.  Uses :class:`BrainAgent` for LLM-powered
generation and stores results in the ``_insights`` DuckDB table.

Designed for periodic background generation (every ~4 hours).  The
Dashboard reads stored insights — no LLM call on page load.

sensitivity_tier: 3 (generates insights from personal data)
"""

from __future__ import annotations

import json
import logging
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from src.agents.brain import BrainAgentV2
from src.core.db_helpers import ensure_tables, utc_now_iso
from src.core.query_tracker import QueryTracker
from src.core.question_patterns import (
    PatternMatch,
    QuestionPatternDetector,
)
from src.core.sqlite.engine import DatabaseEngine
from src.core.topic_loader import load_topic_contacts

logger = logging.getLogger(__name__)


_DEFAULT_INSIGHT_WINDOW_DAYS = 7
_DEFAULT_INSIGHT_QUESTION_LIMIT = 200


def _insight_window_days() -> int:
    """Days of query history to scan for pattern insights.

    Reads ``insight_window_days`` from settings.json.  Used on a fresh
    install to shrink the first pass (no week of history exists yet).

    sensitivity_tier: 1
    """
    try:
        from src.models.llm_provider import load_llm_settings
    except Exception:  # noqa: BLE001
        return _DEFAULT_INSIGHT_WINDOW_DAYS
    raw = load_llm_settings().get("insight_window_days")
    try:
        days = int(raw) if raw is not None else _DEFAULT_INSIGHT_WINDOW_DAYS
    except (TypeError, ValueError):
        return _DEFAULT_INSIGHT_WINDOW_DAYS
    return days if days > 0 else _DEFAULT_INSIGHT_WINDOW_DAYS


def _insight_question_limit() -> int:
    """Question fetch cap, scaled with the window.

    ``200`` for the default 7-day window; falls to roughly ``30`` for a
    1-day window without overriding via settings.  Override explicitly
    with ``insight_question_limit`` in settings.json.

    sensitivity_tier: 1
    """
    try:
        from src.models.llm_provider import load_llm_settings
    except Exception:  # noqa: BLE001
        return _DEFAULT_INSIGHT_QUESTION_LIMIT
    settings = load_llm_settings()
    raw = settings.get("insight_question_limit")
    try:
        if raw is not None:
            limit = int(raw)
            if limit > 0:
                return limit
    except (TypeError, ValueError):
        pass
    days = _insight_window_days()
    scaled = max(
        30,
        int(
            _DEFAULT_INSIGHT_QUESTION_LIMIT
            * days
            / _DEFAULT_INSIGHT_WINDOW_DAYS,
        ),
    )
    return scaled


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------

# Phrases that indicate the LLM had nothing useful to say.
_LOW_QUALITY_PHRASES = frozenset(
    {
        "i don't have",
        "no data available",
        "i cannot",
        "not enough information",
        "no clear pattern",
        "no notable pattern",
        "i'm unable",
        "unable to",
        "no information",
        "there is no data",
    }
)


@dataclass(frozen=True)
class Insight:
    """A proactive insight generated from question patterns.

    sensitivity_tier: 3
    """

    id: str
    domain: str
    title: str
    content: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    trigger: str = "frequent_pattern"
    pattern: str | None = None
    generated_at: str = ""
    sensitivity_tier: int = 3
    suggested_followup: str | None = None

    def has_substance(self) -> bool:
        """Filter low-quality / fallback LLM responses.

        Returns ``False`` if the content is too short or contains
        common "I don't have data" phrases.

        sensitivity_tier: 1
        """
        if len(self.content) < 20:
            return False
        lower = self.content.lower()
        return not any(p in lower for p in _LOW_QUALITY_PHRASES)


# ------------------------------------------------------------------
# InsightGenerator
# ------------------------------------------------------------------


class InsightGenerator:
    """Generate and manage proactive insights.

    Lifecycle:
        1. ``generate_daily_insights()`` — runs periodically in background
        2. ``get_active_insights()`` — read by Dashboard (no LLM call)
        3. ``dismiss_insight()`` / ``follow_up_insight()`` — user feedback

    sensitivity_tier: 3
    """

    def __init__(
        self,
        db_engine: DatabaseEngine,
        query_tracker: QueryTracker,
        brain_agent: BrainAgentV2,
        pattern_detector: QuestionPatternDetector | None = None,
    ) -> None:
        self._db = db_engine
        self._tracker = query_tracker
        self._brain = brain_agent
        self._detector = pattern_detector or QuestionPatternDetector(
            llm_provider=getattr(query_tracker, "_llm", None),
            db_engine=db_engine,
        )
        self._ensure_table()

    # ----------------------------------------------------------
    # Table setup
    # ----------------------------------------------------------

    def _ensure_table(self) -> None:
        """Create the ``_insights`` table if it doesn't exist.

        sensitivity_tier: 1
        """
        ensure_tables(self._db, [
            """
            CREATE TABLE IF NOT EXISTS _insights (
                id                  VARCHAR PRIMARY KEY,
                domain              VARCHAR NOT NULL,
                title               VARCHAR NOT NULL,
                content             VARCHAR NOT NULL,
                sources             VARCHAR DEFAULT '[]',
                trigger             VARCHAR DEFAULT 'frequent_pattern',
                pattern             VARCHAR,
                generated_at        TEXT DEFAULT CURRENT_TIMESTAMP,
                sensitivity_tier    INTEGER DEFAULT 3,
                suggested_followup  VARCHAR,
                shown_at            TEXT,
                dismissed_at        TEXT,
                followed_up         INTEGER DEFAULT 0
            )
            """,
        ])

    # ----------------------------------------------------------
    # Generation
    # ----------------------------------------------------------

    def generate_daily_insights(
        self,
        max_insights: int = 3,
    ) -> list[Insight]:
        """Generate insights from question patterns + active topics.

        Steps:
            1. Generate topic-based insight (from active topics)
            2. Fetch recent questions (last 7 days)
            3. Detect patterns across all questions
            4. For the top patterns, ask BrainAgent
            5. Filter low-quality responses
            6. Store and return

        Handles Ollama failure gracefully — returns empty list.

        Args:
            max_insights: Maximum number of insights to generate.

        Returns:
            List of newly generated insights.

        sensitivity_tier: 3
        """
        insights: list[Insight] = []

        # Step 1: topic-based insight (highest priority)
        topic_insight = self._generate_topic_insight()
        if topic_insight is not None:
            self._store_insight(topic_insight)
            insights.append(topic_insight)

        # Step 2-6: pattern-based insights (fill remaining)
        remaining = max_insights - len(insights)
        if remaining <= 0:
            return insights

        recent = self._tracker.get_recent_questions(
            limit=_insight_question_limit(),
            days=_insight_window_days(),
        )
        if not recent:
            logger.info("No recent questions for pattern insights")
            return insights

        # Detect patterns across all questions
        pattern_counts: Counter[str] = Counter()
        best_match: dict[str, PatternMatch] = {}

        for row in recent:
            question = row.get("question", "")
            if not question:
                continue
            match = self._detector.detect(question)
            if match is None:
                continue
            name = match.pattern_name
            pattern_counts[name] += 1
            if name not in best_match or (
                match.confidence > best_match[name].confidence
            ):
                best_match[name] = match

        if not pattern_counts:
            logger.info("No patterns detected in recent questions")
            return insights

        # Deduplicate: skip patterns with fresh insight
        existing = self._get_recent_pattern_names(days=1)

        for name, _count in pattern_counts.most_common(remaining + 2):
            if len(insights) >= max_insights:
                break
            if name in existing:
                continue

            match = best_match[name]
            insight = self._generate_single(match)
            if insight is not None and insight.has_substance():
                self._store_insight(insight)
                insights.append(insight)

        return insights

    def _generate_topic_insight(self) -> Insight | None:
        """Generate an insight from active important topics.

        Asks BrainAgent about the user's most important ongoing
        conversations to surface actionable insights.

        Returns ``None`` if no topics or LLM fails.

        sensitivity_tier: 3
        """
        topic_contacts = load_topic_contacts(
            self._db, min_importance=5, limit=5,
        )
        if not topic_contacts:
            return None

        # Check if we already generated a topic insight today
        existing = self._get_recent_pattern_names(days=1)
        if "active_topics" in existing:
            return None

        # Build topic summary for prompt
        topic_lines = []
        for tc in sorted(
            topic_contacts.values(),
            key=lambda t: t.get("importance", 0),
            reverse=True,
        ):
            topics = [
                t.get("topic", "")
                for t in tc.get("topics", [])
                if t.get("topic")
            ]
            if topics:
                topic_lines.append(
                    f"- {tc['name']}: {', '.join(topics)}"
                )

        if not topic_lines:
            return None

        prompt = (
            "Based on my most important ongoing "
            "conversations this week:\n"
            + "\n".join(topic_lines)
            + "\n\nWhat should I be aware of or act on "
            "today? Give 2-3 brief, actionable points."
        )

        try:
            resp = self._brain.ask(
                prompt, max_sensitivity_tier=2,
            )
        except Exception:
            logger.warning(
                "BrainAgent failed for topic insight",
                exc_info=True,
            )
            return None

        insight = Insight(
            id=str(uuid.uuid4()),
            domain="personal",
            title="Active topic priorities",
            content=resp.answer,
            sources=resp.sources,
            trigger="active_topics",
            pattern="active_topics",
            generated_at=utc_now_iso(),
            suggested_followup=(
                "Tell me more about these conversations"
            ),
        )
        return insight if insight.has_substance() else None

    def detect_cross_domain_patterns(
        self,
        max_insights: int = 2,
    ) -> list[Insight]:
        """Detect correlations across active domains.

        Looks at the interest profile for domains with weight > 0.3
        and asks the BrainAgent to find cross-domain patterns.

        Args:
            max_insights: Maximum cross-domain insights.

        Returns:
            List of cross-domain insights.

        sensitivity_tier: 3
        """
        profile = self._tracker.get_interest_profile()
        active = [a for a in profile if a.weight > 0.3]

        if len(active) < 2:
            logger.info(
                "Fewer than 2 active domains — "
                "skipping cross-domain detection"
            )
            return []

        domain_names = [a.domain for a in active[:5]]
        prompt = (
            f"Looking at my activity across {', '.join(domain_names)}"
            ", are there any interesting cross-domain patterns "
            "or correlations you notice?"
        )

        try:
            resp = self._brain.ask(prompt, max_sensitivity_tier=2)
        except Exception:
            logger.warning(
                "BrainAgent failed for cross-domain detection",
                exc_info=True,
            )
            return []

        insight = Insight(
            id=str(uuid.uuid4()),
            domain="cross_domain",
            title="Cross-domain patterns",
            content=resp.answer,
            sources=resp.sources,
            trigger="cross_domain",
            pattern=None,
            generated_at=utc_now_iso(),
            suggested_followup=(
                "Tell me more about how these areas connect"
            ),
        )

        if not insight.has_substance():
            return []

        self._store_insight(insight)
        return [insight]

    # ----------------------------------------------------------
    # Read / feedback
    # ----------------------------------------------------------

    def get_active_insights(
        self,
        limit: int = 3,
    ) -> list[Insight]:
        """Return non-dismissed insights, newest first.

        No LLM call — reads from stored data only.

        Args:
            limit: Maximum number of insights to return.

        Returns:
            List of active (non-dismissed) insights.

        sensitivity_tier: 1 (reads stored data)
        """
        rows = self._db.query(
            "SELECT id, domain, title, content, sources, "
            "\"trigger\", pattern, generated_at, "
            "sensitivity_tier, suggested_followup "
            "FROM _insights "
            "WHERE dismissed_at IS NULL "
            "ORDER BY generated_at DESC "
            f"LIMIT {int(limit)}"
        )
        return [self._row_to_insight(r) for r in rows]

    def dismiss_insight(self, insight_id: str) -> None:
        """Mark an insight as dismissed.

        Args:
            insight_id: The insight UUID to dismiss.

        sensitivity_tier: 1
        """
        now_ts = utc_now_iso()
        self._db.execute(
            "UPDATE _insights SET dismissed_at = ? WHERE id = ?",
            [now_ts, insight_id],
        )

    def follow_up_insight(self, insight_id: str) -> None:
        """Mark an insight as followed-up and boost its domain.

        Args:
            insight_id: The insight UUID.

        sensitivity_tier: 1
        """
        # Mark as followed up
        self._db.execute(
            "UPDATE _insights SET followed_up = true WHERE id = ?",
            [insight_id],
        )

        # Boost the domain weight
        rows = self._db.query(
            "SELECT domain FROM _insights WHERE id = ?",
            [insight_id],
        )
        if rows:
            domain = rows[0]["domain"]
            self._tracker.boost_domain(domain, boost_count=2)

    # ----------------------------------------------------------
    # Private helpers
    # ----------------------------------------------------------

    def _generate_single(
        self, match: PatternMatch,
    ) -> Insight | None:
        """Ask BrainAgent for a single pattern-based insight.

        Returns ``None`` if the LLM call fails.

        sensitivity_tier: 3
        """
        try:
            resp = self._brain.ask(
                match.insight_prompt,
                max_sensitivity_tier=2,
            )
        except Exception:
            logger.warning(
                "BrainAgent failed for pattern %s",
                match.pattern_name,
                exc_info=True,
            )
            return None

        # Derive domain from pattern name
        domain = self._pattern_to_domain(match.pattern_name)

        return Insight(
            id=str(uuid.uuid4()),
            domain=domain,
            title=match.description,
            content=resp.answer,
            sources=resp.sources,
            trigger="frequent_pattern",
            pattern=match.pattern_name,
            generated_at=utc_now_iso(),
            suggested_followup=match.suggested_followup,
        )

    def _store_insight(self, insight: Insight) -> None:
        """Persist an insight to the ``_insights`` table.

        sensitivity_tier: 3
        """
        self._db.execute(
            "INSERT INTO _insights "
            "(id, domain, title, content, sources, "
            "\"trigger\", pattern, generated_at, "
            "sensitivity_tier, suggested_followup) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                insight.id,
                insight.domain,
                insight.title,
                insight.content,
                json.dumps(insight.sources),
                insight.trigger,
                insight.pattern,
                insight.generated_at,
                insight.sensitivity_tier,
                insight.suggested_followup,
            ],
        )

    def _get_recent_pattern_names(
        self, days: int = 1,
    ) -> set[str]:
        """Return pattern names that already have recent insights.

        sensitivity_tier: 1
        """
        rows = self._db.query(
            "SELECT DISTINCT pattern FROM _insights "
            "WHERE pattern IS NOT NULL "
            "AND dismissed_at IS NULL "
            "AND generated_at >= "
            f"datetime('now', '-{int(days)} days')"
        )
        return {r["pattern"] for r in rows}

    @staticmethod
    def _pattern_to_domain(pattern_name: str) -> str:
        """Map a pattern name to its closest domain.

        sensitivity_tier: 1
        """
        mapping = {
            "schedule_today": "calendar",
            "schedule_week": "calendar",
            "person_inquiry": "contacts",
            "health_check": "health",
            "message_search": "messages",
            "note_recall": "notes",
            "general_summary": "general",
            "mood_check": "health",
            "relationship_status": "contacts",
            "work_productivity": "work",
        }
        return mapping.get(pattern_name, "general")

    @staticmethod
    def _row_to_insight(row: dict[str, Any]) -> Insight:
        """Convert a DB row dict to an Insight dataclass.

        sensitivity_tier: 1
        """
        sources_raw = row.get("sources", "[]")
        if isinstance(sources_raw, str):
            try:
                sources = json.loads(sources_raw)
            except (json.JSONDecodeError, TypeError):
                sources = []
        else:
            sources = sources_raw if sources_raw else []

        return Insight(
            id=row["id"],
            domain=row["domain"],
            title=row["title"],
            content=row["content"],
            sources=sources,
            trigger=row.get("trigger", "frequent_pattern"),
            pattern=row.get("pattern"),
            generated_at=str(row.get("generated_at", "")),
            sensitivity_tier=row.get("sensitivity_tier", 3),
            suggested_followup=row.get("suggested_followup"),
        )
