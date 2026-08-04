"""Question pattern detection for the learning loop.

Classifies raw user questions into reusable intent patterns
(e.g. "schedule_today", "health_check") via keyword scoring.
Orthogonal to ``QueryTracker.classify_question_domain()`` which
classifies by domain (calendar, health, etc.) — patterns classify
by *what the user wants to know*.

sensitivity_tier: 1 (pure functions on question text)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.core.name_matching import find_names

# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------


@dataclass(frozen=True)
class PatternMatch:
    """A matched question pattern with confidence score.

    sensitivity_tier: 1
    """

    pattern_name: str
    confidence: float  # 0.0–1.0
    description: str
    insight_prompt: str
    suggested_followup: str


# ------------------------------------------------------------------
# Pattern definitions
# ------------------------------------------------------------------

PATTERNS: dict[str, dict[str, Any]] = {
    "schedule_today": {
        "keywords": [
            "today",
            "schedule",
            "calendar",
            "agenda",
            "planned",
            "happening today",
            "plan for today",
        ],
        "description": "Daily schedule overview",
        "insight_prompt": (
            "Give me a brief summary of my day: key meetings, "
            "deadlines, and anything I should know about."
        ),
        "suggested_followup": "What should I prepare for today?",
    },
    "schedule_week": {
        "keywords": [
            "this week",
            "next week",
            "week ahead",
            "upcoming",
            "next few days",
        ],
        "description": "Weekly schedule overview",
        "insight_prompt": (
            "What are the key things on my calendar this week? "
            "Any patterns in my meeting load?"
        ),
        "suggested_followup": (
            "How does this week compare to last week?"
        ),
    },
    "person_inquiry": {
        "keywords": [
            "about",
            "who is",
            "tell me about",
            "know about",
            "talked to",
            "met with",
            "heard from",
            "messaged",
        ],
        "description": "Information about a person",
        "insight_prompt": (
            "Brief update: who have I interacted with most "
            "recently and what are the patterns?"
        ),
        "suggested_followup": (
            "Who should I follow up with this week?"
        ),
    },
    "health_check": {
        "keywords": [
            "health",
            "sleep",
            "exercise",
            "heart rate",
            "steps",
            "workout",
            "trending",
            "compared",
            "blood pressure",
        ],
        "description": "Health metric trends",
        "insight_prompt": (
            "Quick health summary: any notable trends in my "
            "recent health data compared to my baseline?"
        ),
        "suggested_followup": (
            "What health trends should I pay attention to?"
        ),
    },
    "message_search": {
        "keywords": [
            "unread",
            "new messages",
            "missed",
            "inbox",
            "email",
            "message from",
            "text from",
        ],
        "description": "Unread or recent messages",
        "insight_prompt": (
            "What communication patterns do you notice in my "
            "recent messages?"
        ),
        "suggested_followup": (
            "Are there any messages I should respond to?"
        ),
    },
    "note_recall": {
        "keywords": [
            "wrote",
            "noted",
            "journaled",
            "journal",
            "last note",
            "idea",
            "brainstorm",
            "remember writing",
        ],
        "description": "Recent notes and journal entries",
        "insight_prompt": (
            "What themes keep appearing in my recent notes "
            "and ideas?"
        ),
        "suggested_followup": (
            "What ideas have I been working on lately?"
        ),
    },
    "general_summary": {
        "keywords": [
            "summarize",
            "recap",
            "update",
            "what's new",
            "catch me up",
            "overview",
            "what happened",
        ],
        "description": "General life summary",
        "insight_prompt": (
            "Give me a high-level summary of notable patterns "
            "across all my data this week."
        ),
        "suggested_followup": (
            "What's the most important thing I should know "
            "right now?"
        ),
    },
    "mood_check": {
        "keywords": [
            "mood",
            "feeling",
            "emotional",
            "stressed",
            "happy",
            "anxious",
            "energy",
        ],
        "description": "Emotional state check-in",
        "insight_prompt": (
            "What emotional patterns do you notice from my "
            "messages and activities?"
        ),
        "suggested_followup": (
            "How has my mood been trending this week?"
        ),
    },
    "relationship_status": {
        "keywords": [
            "haven't talked",
            "out of touch",
            "neglecting",
            "should reach out",
            "friend",
            "family",
            "colleague",
        ],
        "description": "Relationship maintenance",
        "insight_prompt": (
            "Looking at my interactions, are there people I "
            "haven't connected with recently?"
        ),
        "suggested_followup": "Who should I reach out to?",
    },
    "work_productivity": {
        "keywords": [
            "work",
            "project",
            "deadline",
            "task",
            "productive",
            "productivity",
            "meeting notes",
            "busy",
        ],
        "description": "Work productivity patterns",
        "insight_prompt": (
            "Based on my calendar and messages, how has my "
            "work pattern been this week?"
        ),
        "suggested_followup": (
            "How can I be more productive this week?"
        ),
    },
}


# ------------------------------------------------------------------
# Detector
# ------------------------------------------------------------------

# Pre-compiled regex for capitalized-word entity extraction
_CAPITALIZED_WORD_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")


_PATTERN_SCHEMA: dict[str, Any] = {
    "pattern": " | ".join(PATTERNS.keys()) + " | none",
    "confidence": "float between 0 and 1",
}

_PATTERN_INSTRUCTIONS = (
    "Classify the user's question into ONE of the listed intent "
    "patterns, or 'none' if none fits.\n"
    "Patterns mean:\n"
    + "\n".join(
        f"- {name}: {spec['description']}"
        for name, spec in PATTERNS.items()
    )
)


class QuestionPatternDetector:
    """Classify questions into reusable intent patterns via the LLM.

    The deterministic keyword fallback was removed; production code
    now delegates to :class:`LLMClassifier`.  When the LLM is
    unavailable, :meth:`detect` returns ``None`` and callers degrade
    gracefully (the insight generator simply skips proactive insight
    creation for that question, which is the safer default).

    sensitivity_tier: 1 (stores fingerprint + verdict, not text)
    """

    _CONFIDENCE_THRESHOLD = 0.3

    def __init__(
        self,
        llm_provider: Any | None = None,
        db_engine: Any | None = None,
    ) -> None:
        self._provider = llm_provider
        self._db = db_engine
        self._classifier: Any | None = None

    def _resolve_classifier(self) -> Any | None:
        """sensitivity_tier: 1"""
        if self._classifier is not None:
            return self._classifier
        from src.core.llm_classifier import LLMClassifier

        provider = self._provider
        if provider is None:
            try:
                from src.models.llm_provider import (
                    create_provider_from_settings,
                )
                provider = create_provider_from_settings(background=True)
            except Exception:  # noqa: BLE001
                return None
        self._classifier = LLMClassifier(
            llm_provider=provider, db_engine=self._db,
        )
        return self._classifier

    def detect(self, question: str) -> PatternMatch | None:
        """Return the best matching pattern, or ``None``.

        sensitivity_tier: 1
        """
        matches = self.detect_all(question)
        return matches[0] if matches else None

    def detect_all(self, question: str) -> list[PatternMatch]:
        """Return all matching patterns sorted by confidence.

        Currently returns at most one match: the LLM's top pick.

        sensitivity_tier: 1
        """
        if not question or not question.strip():
            return []
        classifier = self._resolve_classifier()
        if classifier is None:
            return []
        result = classifier.classify(
            kind="question_pattern",
            text=question,
            schema=_PATTERN_SCHEMA,
            instructions=_PATTERN_INSTRUCTIONS,
        )
        if not result:
            return []
        pattern = str(result.get("pattern", "none")).strip().lower()
        if pattern == "none" or pattern not in PATTERNS:
            return []
        try:
            confidence = float(result.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))
        if confidence < self._CONFIDENCE_THRESHOLD:
            return []
        spec = PATTERNS[pattern]
        return [
            PatternMatch(
                pattern_name=pattern,
                confidence=round(confidence, 3),
                description=spec["description"],
                insight_prompt=spec["insight_prompt"],
                suggested_followup=spec["suggested_followup"],
            ),
        ]

    def extract_entities(
        self,
        question: str,
        known_contacts: list[str] | None = None,
    ) -> list[str]:
        """Extract person names from a question.

        If *known_contacts* is provided, matches question text
        against the contact list on whole-word boundaries, longest
        name first. Otherwise, falls back to extracting capitalized
        words that are at least 3 characters long.

        Args:
            question: Natural-language question text.
            known_contacts: Optional list of known contact names.

        Returns:
            List of matched entity names (may be empty), most
            specific match first.

        sensitivity_tier: 1
        """
        if known_contacts:
            # Whole-word only: a plain substring test matches "Ana"
            # inside "Mariana" and returns the wrong contact.
            return find_names(list(known_contacts), question)

        # Fallback: extract capitalized words (skip sentence starts
        # by requiring at least 2 chars before the match or being
        # not at position 0 after a sentence-ender).
        words = _CAPITALIZED_WORD_RE.findall(question)
        # Filter common non-name words
        skip = {
            "What",
            "When",
            "Where",
            "Who",
            "How",
            "Why",
            "Can",
            "Could",
            "Would",
            "Should",
            "Tell",
            "Give",
            "Show",
            "Does",
            "Did",
            "The",
            "Any",
            "Have",
            "Has",
        }
        return [w for w in words if w not in skip]
