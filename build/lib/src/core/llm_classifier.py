"""Shared LLM-driven classifier used everywhere keyword scoring was.

``CLAUDE.md`` mandates AI-driven classification for non-trivial
decisions.  This helper centralizes the wiring: callers pass a *kind*,
a piece of text, and the schema they want back, and the helper handles
prompt assembly, JSON parsing, fingerprint-keyed caching, and
graceful fail-open behaviour when the LLM is unavailable.

Cache: ``_llm_class_cache(fingerprint TEXT PRIMARY KEY, kind TEXT,
result_json TEXT, decided_at TEXT)``.  Fingerprint = sha256(kind +
text)[:16] so the same text reused for different *kinds* doesn't
collide and so cached entries cannot be reversed to the original text.

sensitivity_tier: 1 (only stores fingerprints + verdicts, never raw text)
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from src.agents.firewall.egress_firewall import Lane
from src.agents.firewall.lane_context import lane_scope
from src.core.db_helpers import ensure_tables
from src.core.sqlite.engine import DatabaseEngine
from src.models.llm_provider import LLMProvider
from src.models.prompt_cache import PROMPTS_DIR, FrozenPromptTemplate

logger = logging.getLogger(__name__)


# Frozen prefix (cacheable). The variable portion (schema_json, text,
# trailing instruction) is built per-call below.
_TEMPLATE = FrozenPromptTemplate(PROMPTS_DIR / "llm_classifier_v1.txt")
_SUFFIX_TEMPLATE = (
    "{schema_json}\n\nText:\n{text}\n\nRespond with ONLY a JSON object "
    "matching the schema (no markdown, no explanation).\n"
)


def _fingerprint(kind: str, text: str) -> str:
    """Stable per-(kind, text) hash for cache lookup.

    sensitivity_tier: 1
    """
    payload = f"{kind}\0{text}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]


class LLMClassifier:
    """Cache-aware LLM classifier shared by domain, pattern, urgency,
    sensitivity, and similar subjective decisions.

    sensitivity_tier: 1
    """

    def __init__(
        self,
        llm_provider: LLMProvider | None,
        db_engine: DatabaseEngine | None = None,
    ) -> None:
        self._provider = llm_provider
        self._db = db_engine
        if self._db is not None:
            self._ensure_tables()

    def _ensure_tables(self) -> None:
        """Create ``_llm_class_cache`` if needed.

        sensitivity_tier: 1
        """
        assert self._db is not None
        ensure_tables(self._db, [
            """
            CREATE TABLE IF NOT EXISTS _llm_class_cache (
                fingerprint  TEXT PRIMARY KEY,
                kind         TEXT NOT NULL,
                result_json  TEXT NOT NULL,
                decided_at   TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """,
        ])

    # ----------------------------------------------------------
    # Cache
    # ----------------------------------------------------------

    def _read_cache(self, fp: str) -> dict[str, Any] | None:
        """sensitivity_tier: 1"""
        if self._db is None:
            return None
        try:
            rows = self._db.query(
                "SELECT result_json FROM _llm_class_cache "
                "WHERE fingerprint = ?",
                [fp],
            )
            if not rows:
                return None
            return json.loads(rows[0]["result_json"])
        except Exception:  # noqa: BLE001
            logger.debug("Classifier cache read failed", exc_info=True)
            return None

    def _write_cache(
        self, fp: str, kind: str, result: dict[str, Any],
    ) -> None:
        """sensitivity_tier: 1"""
        if self._db is None:
            return
        try:
            self._db.execute(
                """INSERT OR REPLACE INTO _llm_class_cache
                   (fingerprint, kind, result_json, decided_at)
                   VALUES (?, ?, ?, CURRENT_TIMESTAMP)""",
                [fp, kind, json.dumps(result)],
            )
        except Exception:  # noqa: BLE001
            logger.debug("Classifier cache write failed", exc_info=True)

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def classify(
        self,
        *,
        kind: str,
        text: str,
        schema: dict[str, Any],
        instructions: str = "",
    ) -> dict[str, Any] | None:
        """Classify *text* per *schema*.

        Args:
            kind: Stable classifier identifier (e.g. ``"question_domain"``,
                ``"sensitivity_tier"``).  Part of the cache key.
            text: The text to classify.
            schema: A JSON-serializable schema describing the expected
                output shape (e.g. ``{"domain": "<enum>"}``).
            instructions: Extra prompt instructions appended to the
                default classifier prompt.

        Returns:
            Parsed JSON dict matching the schema, or ``None`` if the
            LLM is unavailable or returned unparsable output.

        sensitivity_tier: 1
        """
        if not text:
            return None
        fp = _fingerprint(kind, text)
        cached = self._read_cache(fp)
        if cached is not None:
            return cached

        if self._provider is None:
            return None

        prompt = _TEMPLATE.prefix + _SUFFIX_TEMPLATE.format(
            schema_json=json.dumps(schema, ensure_ascii=False),
            text=text,
        )
        if instructions:
            prompt = f"{instructions.strip()}\n\n{prompt}"

        try:
            with lane_scope(Lane.CLASSIFIER):
                raw = self._provider.chat_json(
                    [{"role": "user", "content": prompt}],
                )
        except Exception:  # noqa: BLE001
            logger.warning(
                "LLMClassifier %s LLM call failed", kind,
                exc_info=True,
            )
            return None

        if not isinstance(raw, dict) or not raw:
            return None

        self._write_cache(fp, kind, raw)
        return raw
