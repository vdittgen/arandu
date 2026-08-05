"""Emotional labeling — DB-friendly wrapper around :class:`LabelerAgent`.

Pipelines (currently :mod:`src.pipeline.intermediate.int_labeled_messages`)
pass message text through here to get structured emotion / domain
labels. The LLM step is delegated to :class:`LabelerAgent` (pydantic-ai),
which enforces the literals on ``primary_emotion`` / ``domain`` and the
[0.0, 1.0] range on ``intensity`` directly — so the brittle dict
validation lives in the schema, not in this module.

Legacy callers expect ``dict`` results; we project the pydantic model
back into a dict so downstream code (DuckDB column writes, pipeline
joins) doesn't need to change.

The original implementation batched 10 messages per LLM call for
throughput. The SBAgent works per-text; we keep ``batch_label`` for API
compat by looping, accepting the throughput tradeoff in favour of the
unified agent stack. A future ``LabelerBatchAgent`` can restore batched
inference without touching this wrapper's API.

sensitivity_tier: 3 (processes message content which may contain
health, financial, or emotional data)
"""

from __future__ import annotations

import logging
from typing import Any

from src.core.profiler import timed

logger = logging.getLogger(__name__)

VALID_EMOTIONS = frozenset(
    {
        "joy",
        "sadness",
        "anger",
        "fear",
        "surprise",
        "disgust",
        "trust",
        "anticipation",
    }
)

VALID_DOMAINS = frozenset(
    {
        "personal",
        "work",
        "health",
        "social",
        "spiritual",
    }
)


def _label_to_dict(label: Any) -> dict[str, Any]:
    """Project an :class:`EmotionalLabel` pydantic model into a plain dict.

    Downstream callers (pipeline writes, tests) operate on dicts; keep
    that contract intact while the LLM layer is fully on pydantic-ai.
    """
    return {
        "primary_emotion": label.primary_emotion,
        "intensity": label.intensity,
        "feelings": list(label.feelings),
        "desires": list(label.desires),
        "actors": list(label.actors),
        "environment": label.environment,
        "domain": label.domain,
    }


class EmotionalLabeler:
    """Classify text into structured emotional dimensions via the SBAgent.

    Thin coordinator over :class:`LabelerAgent`. Preserves the legacy
    ``.label(text)`` / ``.batch_label(texts)`` API so the pipeline
    doesn't need to know about pydantic-ai.

    sensitivity_tier: 3 (processes raw message content)
    """

    def __init__(self) -> None:
        # Lazily import the SBAgent so unit tests that monkeypatch the
        # class don't pay an import cost for unrelated paths.
        from src.agents.labeler.agent import LabelerAgent

        self._agent_cls = LabelerAgent

    @timed()
    def label(self, text: str) -> dict[str, Any] | None:
        """Classify a single text into emotional labels.

        Returns ``None`` when the agent reports a failure (no LLM
        available, validation rejected the model's output, etc).

        sensitivity_tier: 3
        """
        if not text:
            return None
        try:
            result = self._agent_cls().label(text)
        except Exception as exc:  # noqa: BLE001
            logger.error("LabelerAgent failed: %s", exc)
            return None
        if result is None:
            return None
        return _label_to_dict(result)

    @timed()
    def batch_label(
        self, texts: list[str],
    ) -> list[dict[str, Any] | None]:
        """Classify a batch of texts.

        Loops over the SBAgent per text. A future ``LabelerBatchAgent``
        can plug in here to restore batched LLM throughput without
        changing the caller surface.

        Bails out early after 3 consecutive failures to avoid hammering
        a known-down LLM provider.

        sensitivity_tier: 3
        """
        if not texts:
            return []

        agent = self._agent_cls()
        results: list[dict[str, Any] | None] = []
        consecutive_failures = 0
        for i, text in enumerate(texts):
            try:
                label = agent.label(text) if text else None
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "LabelerAgent batch item %d failed: %s", i, exc,
                )
                label = None

            if label is None:
                results.append(None)
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    remaining = len(texts) - len(results)
                    logger.warning(
                        "LabelerAgent unavailable — 3 consecutive "
                        "failures, falling back to 'unlabeled' for "
                        "remaining %d messages",
                        remaining,
                    )
                    results.extend([None] * remaining)
                    return results
            else:
                consecutive_failures = 0
                results.append(_label_to_dict(label))
        return results
