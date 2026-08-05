"""DB cache + pydantic-ai triage runner.

Delegates the LLM keep/drop decision to :class:`TriageAgent` and
persists each verdict to ``_triage_log`` so the next pipeline cycle can
short-circuit instead of paying the LLM cost again. Verdicts are keyed
by ``message_id`` and never expire.

sensitivity_tier: 3 (sends message content to the LLM)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from src.core.db_helpers import ensure_tables, safe_str
from src.core.sqlite.engine import DatabaseEngine

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------


@dataclass(frozen=True)
class TriageDecision:
    """LLM verdict for a single candidate message.

    ``keep`` is the only field consumers must respect.  The boolean
    flags exist for auditing and downstream filtering.

    sensitivity_tier: 2
    """

    message_id: str
    keep: bool
    reason: str = ""
    is_promo: bool = False
    is_automated: bool = False
    is_ack_only: bool = False


# ------------------------------------------------------------------
# Prompt
# ------------------------------------------------------------------


# 30 messages per LLM call keeps prompt sizes reasonable while batching
# enough to amortise round-trip cost.
_TRIAGE_BATCH = 30
_CONTENT_TRUNCATE = 240
_NOISE_FLAG_KEYS = ("is_promo", "is_automated", "is_ack_only")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _candidate_id(msg: dict[str, Any]) -> str:
    """Best-effort ID extraction supporting message_id or id."""
    raw = msg.get("message_id") or msg.get("id") or ""
    return str(raw)


def _candidate_content(msg: dict[str, Any]) -> str:
    """Best-effort content extraction supporting content or subject+body."""
    if msg.get("content"):
        return str(msg["content"])
    parts = [
        str(msg.get("subject", "")).strip(),
        str(msg.get("body_preview", "")).strip(),
    ]
    joined = "\n".join(p for p in parts if p)
    return joined


def _candidate_sender(msg: dict[str, Any]) -> str:
    """Best-effort sender extraction for TriageMessage."""
    return safe_str(
        msg.get("sender_name") or msg.get("sender", "Unknown"), 80,
    )


# ------------------------------------------------------------------
# Triager
# ------------------------------------------------------------------


class MessageTriager:
    """DB-cached keep/drop classifier backed by :class:`TriageAgent`.

    Usage::

        triager = MessageTriager(db)
        decisions = triager.triage(candidates)
        kept = [c for c, d in zip(candidates, decisions) if d.keep]

    sensitivity_tier: 3 (sees message content)
    """

    def __init__(
        self,
        db_engine: DatabaseEngine,
    ) -> None:
        self._db = db_engine
        self._ensure_tables()

    # ----------------------------------------------------------
    # Table setup
    # ----------------------------------------------------------

    def _ensure_tables(self) -> None:
        """Create ``_triage_log`` if needed.

        sensitivity_tier: 1
        """
        ensure_tables(self._db, [
            """
            CREATE TABLE IF NOT EXISTS _triage_log (
                message_id   VARCHAR PRIMARY KEY,
                keep         INTEGER NOT NULL,
                reason       TEXT,
                flags_json   TEXT,
                decided_at   TEXT DEFAULT CURRENT_TIMESTAMP,
                sensitivity_tier INTEGER DEFAULT 3
            )
            """,
        ])

    # ----------------------------------------------------------
    # Cache
    # ----------------------------------------------------------

    def _load_cached(
        self,
        message_ids: list[str],
    ) -> dict[str, TriageDecision]:
        """Return decisions already in ``_triage_log`` for the given IDs.

        sensitivity_tier: 2
        """
        if not message_ids:
            return {}
        placeholders = ",".join("?" for _ in message_ids)
        try:
            rows = self._db.query(
                f"SELECT message_id, keep, reason, flags_json "
                f"FROM _triage_log WHERE message_id IN ({placeholders})",
                list(message_ids),
            )
        except Exception:  # noqa: BLE001
            logger.debug("Triage cache lookup failed", exc_info=True)
            return {}
        cached: dict[str, TriageDecision] = {}
        for r in rows:
            flags_raw = r.get("flags_json")
            flags: dict[str, Any] = {}
            if flags_raw:
                try:
                    flags = json.loads(flags_raw)
                except (json.JSONDecodeError, TypeError):
                    pass
            mid = str(r["message_id"])
            cached[mid] = TriageDecision(
                message_id=mid,
                keep=bool(r["keep"]),
                reason=str(r.get("reason", "")),
                is_promo=bool(flags.get("is_promo", False)),
                is_automated=bool(flags.get("is_automated", False)),
                is_ack_only=bool(flags.get("is_ack_only", False)),
            )
        return cached

    def _persist(self, decisions: list[TriageDecision]) -> None:
        """Store fresh decisions in ``_triage_log``.

        sensitivity_tier: 2
        """
        for d in decisions:
            try:
                flags_json = json.dumps({
                    k: bool(getattr(d, k))
                    for k in _NOISE_FLAG_KEYS
                })
                self._db.execute(
                    """INSERT OR REPLACE INTO _triage_log
                       (message_id, keep, reason, flags_json,
                        decided_at, sensitivity_tier)
                       VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, 3)""",
                    [d.message_id, int(d.keep), d.reason, flags_json],
                )
            except Exception:  # noqa: BLE001
                logger.debug(
                    "Triage log write failed for %s",
                    d.message_id, exc_info=True,
                )

    # ----------------------------------------------------------
    # Main entry
    # ----------------------------------------------------------

    def triage(
        self,
        candidates: list[dict[str, Any]],
    ) -> list[TriageDecision]:
        """Return one ``TriageDecision`` per input message in order.

        When the LLM is unavailable, every candidate is kept (fail-open
        so the downstream evaluator still sees the message).  When the
        LLM is available, results are persisted to ``_triage_log`` so
        repeat candidates short-circuit.

        sensitivity_tier: 3
        """
        if not candidates:
            return []

        # Build placeholder decisions so the output is always
        # 1:1 with the input order.
        ids = [_candidate_id(c) for c in candidates]
        # Always-keep default for callers that pass items without IDs.
        decisions: dict[int, TriageDecision] = {}
        cached = self._load_cached([i for i in ids if i])
        needs_llm: list[tuple[int, dict[str, Any]]] = []

        for idx, (msg_id, msg) in enumerate(zip(ids, candidates)):
            if msg_id and msg_id in cached:
                decisions[idx] = cached[msg_id]
                continue
            if not _candidate_content(msg).strip():
                # Empty content → drop without paying for an LLM call.
                decisions[idx] = TriageDecision(
                    message_id=msg_id or f"idx_{idx}",
                    keep=False,
                    reason="empty content",
                    is_automated=True,
                )
                continue
            needs_llm.append((idx, msg))

        if needs_llm:
            self._llm_triage_into(needs_llm, decisions)

        ordered = [decisions[i] for i in range(len(candidates))]

        # Persist only verdicts that came from the LLM (cached entries
        # are already on disk).  Empty-content drops are persisted too,
        # so we don't re-evaluate them next cycle.
        fresh = [
            ordered[idx] for idx, mid in enumerate(ids)
            if mid and (
                idx in {i for i, _ in needs_llm}
                or mid not in cached
            )
        ]
        if fresh:
            self._persist(fresh)

        logger.info(
            "Triage: %d kept / %d total (cached=%d, llm=%d)",
            sum(1 for d in ordered if d.keep),
            len(ordered),
            len(cached),
            len(needs_llm),
        )
        return ordered

    # ----------------------------------------------------------
    # LLM batching
    # ----------------------------------------------------------

    def _llm_triage_into(
        self,
        needs_llm: list[tuple[int, dict[str, Any]]],
        decisions: dict[int, TriageDecision],
    ) -> None:
        """Run :class:`TriageAgent` batches and fill ``decisions``.

        sensitivity_tier: 3
        """
        from src.agents.triage.agent import TriageAgent, TriageMessage

        agent = TriageAgent()
        for batch_start in range(0, len(needs_llm), _TRIAGE_BATCH):
            chunk = needs_llm[batch_start:batch_start + _TRIAGE_BATCH]
            triage_msgs = [
                TriageMessage(
                    message_id=_candidate_id(msg) or f"idx_{idx}",
                    content=_candidate_content(msg),
                    sender_name=_candidate_sender(msg),
                    source=str(msg.get("source", "")),
                )
                for idx, msg in chunk
            ]
            try:
                batch = agent.triage(triage_msgs)
            except Exception:  # noqa: BLE001
                batch = None
                logger.warning(
                    "TriageAgent call failed — falling open for %d msgs",
                    len(chunk),
                    exc_info=True,
                )

            by_id: dict[str, Any] = {}
            ordered: list[Any] = []
            if batch is not None:
                for verdict in batch.decisions:
                    ordered.append(verdict)
                    if verdict.message_id:
                        by_id[verdict.message_id] = verdict

            for offset, (idx, msg) in enumerate(chunk):
                msg_id = _candidate_id(msg) or f"idx_{idx}"
                verdict = by_id.get(msg_id) or (
                    ordered[offset] if offset < len(ordered) else None
                )
                if verdict is None:
                    decisions[idx] = TriageDecision(
                        message_id=msg_id,
                        keep=True,
                        reason="agent missing verdict",
                    )
                else:
                    decisions[idx] = TriageDecision(
                        message_id=msg_id,
                        keep=bool(verdict.keep),
                        reason=str(verdict.reason or "")[:200],
                        is_promo=bool(verdict.is_promo),
                        is_automated=bool(verdict.is_automated),
                        is_ack_only=bool(verdict.is_ack_only),
                    )
