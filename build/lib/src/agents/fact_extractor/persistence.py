"""Progressive learning — extract and manage personal facts.

Extracts personal facts from conversations and stores them in the
``_learned_facts`` DuckDB table.  Facts are injected into BrainAgent
system prompts so the agent progressively "learns" about the user.

Follows the same lifecycle as :class:`InsightGenerator`:
    1. ``extract_facts_from_conversation()`` — post-conversation LLM pass
    2. ``get_active_facts()`` — read for context injection (no LLM)
    3. ``confirm_fact()`` / ``dismiss_fact()`` — user feedback

sensitivity_tier: 3 (extracts personal data from conversations)
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any

from src.core.db_helpers import ensure_tables, utc_now_iso
from src.core.sqlite.engine import DatabaseEngine

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Fact categories
# ------------------------------------------------------------------

CATEGORIES = frozenset({
    "preference",
    "relationship",
    "biographical",
    "habit",
    "opinion",
    "health",
    "work",
    "location",
})


# ------------------------------------------------------------------
# Data class
# ------------------------------------------------------------------


@dataclass(frozen=True)
class LearnedFact:
    """A learned personal fact.

    sensitivity_tier: 2
    """

    id: str
    category: str
    subject: str
    predicate: str
    content: str
    confidence: float = 0.8
    source_type: str = "conversation"
    source_id: str | None = None
    extracted_at: str = ""
    confirmed_at: str | None = None
    dismissed_at: str | None = None
    superseded_by: str | None = None
    sensitivity_tier: int = 2
    times_used: int = 0


# ------------------------------------------------------------------
# FactLearner
# ------------------------------------------------------------------


class FactLearner:
    """Extract and manage personal facts from conversations.

    Lifecycle:
        1. ``extract_facts_from_conversation()`` — post-conversation
        2. ``get_active_facts()`` — read for context injection (no LLM)
        3. ``confirm_fact()`` / ``dismiss_fact()`` — user feedback
        4. ``get_facts_for_review()`` — frontend review UI

    sensitivity_tier: 3
    """

    def __init__(
        self,
        db_engine: DatabaseEngine,
    ) -> None:
        self._db = db_engine
        self._ensure_table()

    # ----------------------------------------------------------
    # Table setup
    # ----------------------------------------------------------

    def _ensure_table(self) -> None:
        """Create ``_learned_facts`` if it doesn't exist.

        sensitivity_tier: 1
        """
        ensure_tables(self._db, [
            """
            CREATE TABLE IF NOT EXISTS _learned_facts (
                id                  VARCHAR PRIMARY KEY,
                category            VARCHAR NOT NULL,
                subject             VARCHAR NOT NULL,
                predicate           VARCHAR NOT NULL,
                content             VARCHAR NOT NULL,
                confidence          DOUBLE DEFAULT 0.8,
                source_type         VARCHAR DEFAULT 'conversation',
                source_id           VARCHAR,
                extracted_at        TEXT DEFAULT CURRENT_TIMESTAMP,
                confirmed_at        TEXT,
                dismissed_at        TEXT,
                superseded_by       VARCHAR,
                superseded_at       TEXT,
                sensitivity_tier    INTEGER DEFAULT 2,
                times_used          INTEGER DEFAULT 0,
                last_used_at        TEXT
            )
            """,
        ])

    # ----------------------------------------------------------
    # Extraction
    # ----------------------------------------------------------

    def extract_facts_from_conversation(
        self,
        user_messages: list[str],
        assistant_messages: list[str],
        max_facts: int = 5,
    ) -> list[LearnedFact]:
        """Extract personal facts from a conversation turn.

        Delegates the LLM step to :class:`FactExtractorAgent` (pydantic-ai
        SBAgent). The orchestrator persists each draft into
        ``_learned_facts`` with automatic contradiction resolution.

        Args:
            user_messages: User's messages in the conversation.
            assistant_messages: Assistant's responses.
            max_facts: Maximum facts to extract per call.

        Returns:
            List of newly stored facts.

        sensitivity_tier: 3
        """
        from src.agents.fact_extractor.agent import FactExtractorAgent

        turns: list[str] = []
        for i, msg in enumerate(user_messages):
            turns.append(f"User: {msg}")
            if i < len(assistant_messages):
                turns.append(f"Assistant: {assistant_messages[i]}")
        conversation = "\n".join(turns)

        if len(conversation.strip()) < 20:
            return []

        try:
            batch = FactExtractorAgent().extract(conversation)
        except Exception:  # noqa: BLE001
            logger.debug("FactExtractorAgent failed", exc_info=True)
            return []
        if batch is None or not batch.facts:
            return []

        stored: list[LearnedFact] = []
        for draft in batch.facts[:max_facts]:
            fact = self._store_fact({
                "category": draft.category,
                "subject": draft.subject,
                "predicate": draft.predicate,
                "content": draft.content,
                "sensitivity_tier": draft.sensitivity_tier,
            })
            if fact is not None:
                stored.append(fact)

        return stored

    # ----------------------------------------------------------
    # Read (no LLM)
    # ----------------------------------------------------------

    def get_active_facts(
        self,
        limit: int = 20,
        category: str | None = None,
        subject: str | None = None,
        min_confidence: float = 0.5,
    ) -> list[LearnedFact]:
        """Return active (non-dismissed, non-superseded) facts.

        Args:
            limit: Maximum facts to return.
            category: Optional category filter.
            subject: Optional subject filter ("self" or person name).
            min_confidence: Minimum confidence threshold.

        Returns:
            List of active facts, ordered by confidence then recency.

        sensitivity_tier: 2
        """
        conditions = [
            "dismissed_at IS NULL",
            "superseded_by IS NULL",
            "confidence >= ?",
        ]
        params: list[Any] = [min_confidence]

        if category:
            conditions.append("category = ?")
            params.append(category)
        if subject:
            conditions.append("subject = ?")
            params.append(subject)

        where = " AND ".join(conditions)
        params.append(limit)

        rows = self._db.query(
            f"SELECT * FROM _learned_facts "  # noqa: S608
            f"WHERE {where} "
            "ORDER BY "
            "  CASE WHEN confirmed_at IS NOT NULL THEN 0 ELSE 1 END, "
            "  confidence DESC, "
            "  extracted_at DESC "
            "LIMIT ?",
            params,
        )
        return [_row_to_fact(r) for r in rows]

    def get_facts_for_review(
        self,
        limit: int = 50,
    ) -> list[LearnedFact]:
        """Return facts pending user review.

        Includes unconfirmed, non-dismissed facts.

        sensitivity_tier: 2
        """
        rows = self._db.query(
            "SELECT * FROM _learned_facts "
            "WHERE dismissed_at IS NULL "
            "AND superseded_by IS NULL "
            "AND confirmed_at IS NULL "
            "ORDER BY extracted_at DESC "
            "LIMIT ?",
            [limit],
        )
        return [_row_to_fact(r) for r in rows]

    def get_fact_count(self) -> dict[str, int]:
        """Return fact counts by status.

        sensitivity_tier: 1
        """
        rows = self._db.query(
            "SELECT "
            "  COUNT(*) AS total, "
            "  SUM(CASE WHEN confirmed_at IS NOT NULL "
            "    AND dismissed_at IS NULL THEN 1 ELSE 0 END) AS confirmed, "
            "  SUM(CASE WHEN confirmed_at IS NULL "
            "    AND dismissed_at IS NULL "
            "    AND superseded_by IS NULL THEN 1 ELSE 0 END) AS pending_review "
            "FROM _learned_facts"
        )
        if not rows:
            return {"total": 0, "confirmed": 0, "pending_review": 0}

        row = rows[0]
        result = {
            "total": int(row["total"] or 0),
            "confirmed": int(row["confirmed"] or 0),
            "pending_review": int(row["pending_review"] or 0),
        }

        # Per-category counts
        cat_rows = self._db.query(
            "SELECT category, COUNT(*) AS cnt "
            "FROM _learned_facts "
            "WHERE dismissed_at IS NULL AND superseded_by IS NULL "
            "GROUP BY category"
        )
        by_category = {
            str(r["category"]): int(r["cnt"]) for r in cat_rows
        }
        result["by_category"] = by_category  # type: ignore[assignment]

        return result

    # ----------------------------------------------------------
    # Feedback
    # ----------------------------------------------------------

    def confirm_fact(self, fact_id: str) -> None:
        """Mark a fact as confirmed by the user.

        Sets confidence to 1.0 and records confirmation time.

        sensitivity_tier: 2
        """
        now = utc_now_iso()
        self._db.execute(
            "UPDATE _learned_facts "
            "SET confirmed_at = ?, confidence = 1.0 "
            "WHERE id = ?",
            [now, fact_id],
        )

    def dismiss_fact(self, fact_id: str) -> None:
        """Dismiss a fact (user says it's wrong).

        sensitivity_tier: 1
        """
        now = utc_now_iso()
        self._db.execute(
            "UPDATE _learned_facts "
            "SET dismissed_at = ? "
            "WHERE id = ?",
            [now, fact_id],
        )

    def edit_fact(self, fact_id: str, new_content: str) -> None:
        """Edit a fact's content and mark as confirmed.

        sensitivity_tier: 2
        """
        now = utc_now_iso()
        self._db.execute(
            "UPDATE _learned_facts "
            "SET content = ?, confirmed_at = ?, confidence = 1.0 "
            "WHERE id = ?",
            [new_content, now, fact_id],
        )

    # ----------------------------------------------------------
    # Context injection helper
    # ----------------------------------------------------------

    def increment_usage(self, fact_ids: list[str]) -> None:
        """Increment times_used for injected facts.

        sensitivity_tier: 1
        """
        if not fact_ids:
            return
        now = utc_now_iso()
        placeholders = ", ".join(["?"] * len(fact_ids))
        self._db.execute(
            f"UPDATE _learned_facts "  # noqa: S608
            f"SET times_used = times_used + 1, last_used_at = ? "
            f"WHERE id IN ({placeholders})",
            [now, *fact_ids],
        )

    # ----------------------------------------------------------
    # Private helpers
    # ----------------------------------------------------------

    def _store_fact(self, raw: dict[str, Any]) -> LearnedFact | None:
        """Validate, resolve contradictions, and store a fact.

        sensitivity_tier: 2
        """
        category = str(raw.get("category", "")).lower()
        if category not in CATEGORIES:
            category = "preference"

        subject = str(raw.get("subject", "self")).strip()
        predicate = str(raw.get("predicate", "")).strip()
        content = str(raw.get("content", "")).strip()
        tier = int(raw.get("sensitivity_tier", 2))

        if not content or not predicate:
            return None
        if len(content) < 5:
            return None

        tier = max(1, min(3, tier))

        # Resolve contradictions
        self._resolve_contradictions(subject, predicate)

        fact_id = str(uuid.uuid4())
        now = utc_now_iso()

        self._db.execute(
            "INSERT INTO _learned_facts "
            "(id, category, subject, predicate, content, confidence, "
            " source_type, extracted_at, sensitivity_tier) "
            "VALUES (?, ?, ?, ?, ?, 0.8, 'conversation', ?, ?)",
            [fact_id, category, subject, predicate, content, now, tier],
        )

        return LearnedFact(
            id=fact_id,
            category=category,
            subject=subject,
            predicate=predicate,
            content=content,
            confidence=0.8,
            source_type="conversation",
            extracted_at=now,
            sensitivity_tier=tier,
        )

    def _resolve_contradictions(
        self, subject: str, predicate: str,
    ) -> None:
        """Supersede older facts with same (subject, predicate).

        sensitivity_tier: 2
        """
        existing = self._db.query(
            "SELECT id FROM _learned_facts "
            "WHERE subject = ? AND predicate = ? "
            "AND dismissed_at IS NULL "
            "AND superseded_by IS NULL",
            [subject, predicate],
        )
        if not existing:
            return

        now = utc_now_iso()
        ids = [str(r["id"]) for r in existing]
        placeholders = ", ".join(["?"] * len(ids))
        self._db.execute(
            f"UPDATE _learned_facts "  # noqa: S608
            f"SET confidence = 0.3, superseded_at = ? "
            f"WHERE id IN ({placeholders})",
            [now, *ids],
        )


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _row_to_fact(row: dict[str, Any]) -> LearnedFact:
    """Convert a DuckDB row dict to a LearnedFact.

    sensitivity_tier: 2
    """
    return LearnedFact(
        id=str(row["id"]),
        category=str(row["category"]),
        subject=str(row["subject"]),
        predicate=str(row["predicate"]),
        content=str(row["content"]),
        confidence=float(row["confidence"])
        if row.get("confidence") is not None
        else 0.8,
        source_type=str(row["source_type"])
        if row.get("source_type")
        else "conversation",
        source_id=str(row["source_id"])
        if row.get("source_id")
        else None,
        extracted_at=str(row["extracted_at"])
        if row.get("extracted_at")
        else "",
        confirmed_at=str(row["confirmed_at"])
        if row.get("confirmed_at")
        else None,
        dismissed_at=str(row["dismissed_at"])
        if row.get("dismissed_at")
        else None,
        superseded_by=str(row["superseded_by"])
        if row.get("superseded_by")
        else None,
        sensitivity_tier=int(row["sensitivity_tier"])
        if row.get("sensitivity_tier") is not None
        else 2,
        times_used=int(row["times_used"])
        if row.get("times_used") is not None
        else 0,
    )
