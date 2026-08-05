"""LLM-driven sensitivity tier classifier.

Replaces the previous keyword/regex implementation.  Per
``CLAUDE.md`` the firewall's tier defaults must err on the safe side,
so when the LLM is unavailable or returns garbage the classifier
returns Tier 3 (the most restrictive) — matching the "unknown fields
default to Tier 3" rule documented in the security model.

Tiers:
- 1 (low):    general preferences, interests
- 2 (medium): habits, routines, people names, locations, dates
- 3 (high):   health, finances, emotions, traumas

sensitivity_tier: N/A (classifier itself holds no user data)
"""

from __future__ import annotations

import logging
from typing import Any

from src.core.llm_classifier import LLMClassifier
from src.models.llm_provider import LLMProvider

logger = logging.getLogger(__name__)


_SCHEMA: dict[str, Any] = {
    "tier": "1 | 2 | 3 (integer)",
    "reason": "<short explanation>",
}

_INSTRUCTIONS = (
    "You classify the data sensitivity tier of a single piece of text "
    "for a privacy-first OS.  Use the strictest applicable tier.\n"
    "- tier 3: health conditions, medications, mental health, trauma, "
    "abuse, suicidal ideation, salary/income, bank account, debt, "
    "social security, tax filings, transactions with explicit amounts, "
    "addiction.\n"
    "- tier 2: people's names, family members, locations, addresses, "
    "phone numbers, routines, schedules, dates, work meetings.\n"
    "- tier 1: generic preferences, public categories, abstract topics."
)

# Fail-safe tier when the LLM is unavailable or returns garbage.
# Matches the security model's "unknown fields default to Tier 3" rule.
_FAIL_SAFE_TIER = 3


class SensitivityClassifier:
    """Classify text into a sensitivity tier (1-3) via LLM.

    sensitivity_tier: N/A (infrastructure — no user data stored)
    """

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        db_engine: Any | None = None,
    ) -> None:
        """Initialize the classifier.

        Args:
            llm_provider: Optional LLMProvider.  When ``None``, the
                classifier resolves one lazily via the factory the
                first time it's used; if no provider is configured
                the classifier falls back to Tier 3 for every input.
            db_engine: Optional DB engine for fingerprint caching.
                Without a DB, every call hits the LLM.
        """
        self._provider = llm_provider
        self._db = db_engine
        self._classifier: LLMClassifier | None = None
        self._tried_resolve = False

    def _resolve_classifier(self) -> LLMClassifier | None:
        """Lazily resolve the underlying LLMClassifier.

        sensitivity_tier: 1
        """
        if self._classifier is not None:
            return self._classifier
        if self._tried_resolve:
            return None
        self._tried_resolve = True

        provider = self._provider
        if provider is None:
            try:
                from src.models.llm_provider import (
                    create_provider_from_settings,
                )
                provider = create_provider_from_settings(background=True)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "No LLM provider for sensitivity classification "
                    "— falling back to Tier %d",
                    _FAIL_SAFE_TIER,
                )
                return None

        self._classifier = LLMClassifier(
            llm_provider=provider, db_engine=self._db,
        )
        return self._classifier

    def classify(self, text: str) -> int:
        """Classify a single text into a sensitivity tier.

        Args:
            text: The text to classify.

        Returns:
            Sensitivity tier: 1 (low), 2 (medium), or 3 (high).
            Returns 3 (the safest tier) when the LLM is unavailable
            or returns an invalid response.

        sensitivity_tier: N/A
        """
        if not text:
            return 1
        classifier = self._resolve_classifier()
        if classifier is None:
            return _FAIL_SAFE_TIER
        result = classifier.classify(
            kind="sensitivity_tier",
            text=text,
            schema=_SCHEMA,
            instructions=_INSTRUCTIONS,
        )
        if not result:
            return _FAIL_SAFE_TIER
        try:
            tier = int(result.get("tier", _FAIL_SAFE_TIER))
        except (TypeError, ValueError):
            return _FAIL_SAFE_TIER
        if tier not in (1, 2, 3):
            return _FAIL_SAFE_TIER
        return tier

    def classify_batch(self, texts: list[str]) -> list[int]:
        """Classify multiple texts.

        sensitivity_tier: N/A
        """
        return [self.classify(t) for t in texts]
