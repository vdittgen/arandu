"""Rule-based notification decision engine.

Evaluates pipeline results, action completions, and generated insights
to decide whether a WhatsApp notification should be sent. Uses
rule-based logic for decisions + a single targeted LLM call to craft
enrichment notifications about important ongoing topics.

Notification philosophy:
- Only notify for actions the user should take or enrichment info
  for important ongoing topics.
- Never notify about routine pipeline completions or generic messages.
- Respect category preferences, global mute, and dedup.
- Use mart_contact_summary for topic-aware decisions (pre-computed scores).

sensitivity_tier: 2 (reads event context and mart data, optional LLM)
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import date
from typing import TYPE_CHECKING, Any

from src.agents.firewall.egress_firewall import Lane
from src.models.llm_gateway import GatewayBlocked, chat_via_firewalls
from src.notifications.models import NotificationDecision

if TYPE_CHECKING:
    from src.core.sqlite.engine import DatabaseEngine
    from src.models.llm_provider import LLMProvider
    from src.notifications.preference_service import PreferenceService

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Shared no-notify default
# ------------------------------------------------------------------

_NO_NOTIFY = NotificationDecision(
    should_notify=False,
    category="",
    importance_score=0.0,
    message="",
    reason="",
    dedupe_key="",
    event_context={},
)


# ------------------------------------------------------------------
# Category keywords for classification
# ------------------------------------------------------------------

_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "calendar_conflicts": ["calendar", "meeting", "schedule", "conflict"],
    "health_alerts": ["health", "fitness", "anomaly", "sleep"],
    "action_results": ["action"],
    "pipeline_summary": ["pipeline", "data refresh"],
}


# ------------------------------------------------------------------
# Insight scoring schema (LLM-driven path)
# ------------------------------------------------------------------

_INSIGHT_SCORE_SCHEMA: dict[str, Any] = {
    "category": "calendar_conflicts | health_alerts | action_results "
                "| pipeline_summary",
    "importance": "0-10 float",
    "summary": "1 sentence",
}

_INSIGHT_SCORE_INSTRUCTIONS = (
    "Rate a generated insight's urgency for the user.  Health "
    "anomalies, calendar conflicts and impending deadlines score "
    "high; recurring summaries and routine completions score low.  "
    "Only output the JSON dict described in the schema."
)

# Fallback cues retained ONLY for the case when no LLM is available
# (Ollama offline, no remote endpoint configured).  Marked deprecated;
# the LLM path is the source of truth.
_DEPRECATED_HIGH_IMPORTANCE_KEYWORDS: frozenset[str] = frozenset({
    "health", "anomaly", "conflict", "urgent", "deadline", "overdue",
    "birthday", "appointment", "doctor", "medication", "emergency",
    "cancer", "treatment", "surgery", "hospital",
})


class BrainNotificationOrchestrator:
    """Rule-based notification decision engine.

    Evaluates pipeline results, action completions, and generated
    insights using deterministic rules. Zero LLM calls.

    sensitivity_tier: 2
    """

    def __init__(
        self,
        preference_service: PreferenceService,
        db_engine: DatabaseEngine | None = None,
        llm_provider: LLMProvider | None = None,
        **_kwargs: Any,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            preference_service: User notification preferences.
            db_engine: Optional DB for topic-aware decisions.
            llm_provider: Optional LLM provider used to rate insight
                urgency.  When absent, falls back to a deprecated
                keyword-based score (kept only for offline operation).
            **_kwargs: Ignored (backwards compat with old BrainAgent param).
        """
        self._prefs = preference_service
        self._db = db_engine
        self._llm = llm_provider

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def evaluate_pipeline_result(
        self,
        run_result: dict[str, Any],
        stats: dict[str, Any],
    ) -> NotificationDecision:
        """Evaluate a pipeline completion for notification worthiness.

        Only notifies on failure or calendar conflicts.
        Routine completions are silently ignored.

        sensitivity_tier: 2
        """
        event_context = {"run_result": run_result, "stats": stats}
        source_id = run_result.get("run_id", "unknown")
        return self._eval_pipeline(source_id, event_context)

    def evaluate_action_result(
        self,
        action_result: dict[str, Any],
        proposal: dict[str, Any],
    ) -> NotificationDecision:
        """Evaluate an action completion for notification worthiness.

        Always notifies — user explicitly requested the action.

        sensitivity_tier: 2
        """
        event_context = {"action_result": action_result, "proposal": proposal}
        source_id = proposal.get("proposal_id", "unknown")
        return self._eval_action(source_id, event_context)

    def evaluate_insight_result(
        self,
        insights: list[dict[str, Any]],
    ) -> NotificationDecision:
        """Evaluate newly generated insights for notification worthiness.

        Only notifies for high-importance topics (health anomalies,
        calendar conflicts, important contact situations).

        sensitivity_tier: 2
        """
        if not insights:
            return _NO_NOTIFY
        event_context = {"insights": insights}
        source_id = insights[0].get("id", "unknown")
        return self._eval_insight(source_id, event_context)

    # ----------------------------------------------------------
    # Rule-based evaluation
    # ----------------------------------------------------------

    def _eval_pipeline(
        self,
        source_id: str,
        event_context: dict[str, Any],
    ) -> NotificationDecision:
        """Pipeline rule: only notify on failure.

        sensitivity_tier: 1
        """
        run_result = event_context.get("run_result", {})
        status = str(run_result.get("status", "")).lower()

        if status == "failed":
            return self._make_decision(
                should_notify=True,
                category="pipeline_summary",
                importance=8.0,
                message=(
                    f"Pipeline failed: "
                    f"{run_result.get('error', 'unknown error')}"
                ),
                reason="Pipeline run failed",
                source_type="pipeline",
                source_id=source_id,
                event_context=event_context,
            )

        # Routine completion — no notification
        return _NO_NOTIFY

    def _eval_action(
        self,
        source_id: str,
        event_context: dict[str, Any],
    ) -> NotificationDecision:
        """Action rule: always notify on completion (user requested it).

        sensitivity_tier: 2
        """
        action_result = event_context.get("action_result", {})
        proposal = event_context.get("proposal", {})
        tool_name = proposal.get("tool_name", "action")
        status = str(action_result.get("status", "")).lower()

        if status in ("error", "failed"):
            return self._make_decision(
                should_notify=True,
                category="action_results",
                importance=8.0,
                message=f"Action failed: {tool_name}",
                reason="User-requested action failed",
                source_type="action",
                source_id=source_id,
                event_context=event_context,
            )

        return self._make_decision(
            should_notify=True,
            category="action_results",
            importance=6.0,
            message=f"Action completed: {tool_name}",
            reason="User-requested action completed successfully",
            source_type="action",
            source_id=source_id,
            event_context=event_context,
        )

    def _eval_insight(
        self,
        source_id: str,
        event_context: dict[str, Any],
    ) -> NotificationDecision:
        """Insight rule: only high-importance actionable insights.

        Notifies when insight relates to health anomalies, calendar
        conflicts, or important contact situations (topics with
        importance >= 7).

        sensitivity_tier: 2
        """
        insights = event_context.get("insights", [])
        if not insights:
            return _NO_NOTIFY

        # Find the highest importance insight
        best_insight: dict[str, Any] | None = None
        best_score = 0.0
        best_category: str | None = None

        # Load active topics from DB for cross-referencing
        active_topics = self._load_active_topics()

        for insight in insights:
            text = json.dumps(insight, default=str).lower()
            score = 0.0

            verdict: dict[str, Any] | None = (
                self._llm_score_insight(text)
                if self._llm is not None else None
            )

            if verdict:
                try:
                    score += float(verdict.get("importance", 0.0))
                except (TypeError, ValueError):
                    pass
            else:
                # Deprecated fallback used only when no LLM is available.
                keyword_hits = sum(
                    1 for kw in _DEPRECATED_HIGH_IMPORTANCE_KEYWORDS
                    if kw in text
                )
                score += keyword_hits * 2.0

            # Score based on explicit importance field (additive).
            score += float(insight.get("importance", 0))

            # Cross-reference with active topics from DB.
            score += self._topic_match_score(text, active_topics)

            if score > best_score:
                best_score = score
                best_insight = insight
                best_category = (
                    str(verdict.get("category"))
                    if verdict and verdict.get("category")
                    else None
                )

        # Only notify if score crosses threshold
        # (roughly: 1 high keyword + some importance)
        if best_score < 7.0 or best_insight is None:
            return _NO_NOTIFY

        # Determine category — prefer the LLM's verdict, else use the
        # keyword fallback for offline operation.
        category = (
            best_category
            if best_category in _CATEGORY_KEYWORDS
            else "health_alerts"
        )
        if best_category is None:
            text_lower = json.dumps(best_insight, default=str).lower()
            for cat, keywords in _CATEGORY_KEYWORDS.items():
                if any(kw in text_lower for kw in keywords):
                    category = cat
                    break

        summary = str(
            best_insight.get("summary")
            or best_insight.get("title")
            or best_insight.get("question", "New insight available")
        )

        return self._make_decision(
            should_notify=True,
            category=category,
            importance=min(10.0, best_score),
            message=summary[:200],
            reason=(
                f"High-importance insight (score={best_score:.1f})"
            ),
            source_type="insight",
            source_id=source_id,
            event_context=event_context,
        )

    # ----------------------------------------------------------
    # LLM classifier
    # ----------------------------------------------------------

    def _llm_score_insight(self, text: str) -> dict[str, Any] | None:
        """Route the insight-urgency LLM call through the firewall.

        The classification text is a JSON blob built from raw insight
        rows that can contain Tier 2-3 content (contact names, health
        keywords, financial figures). Routing through the gateway
        ensures the egress firewall sees the prompt before it leaves
        the device.

        Returns ``None`` when the LLM is blocked, fails, or returns
        non-JSON output — the caller then falls back to the
        deprecated keyword-cue scoring path.

        sensitivity_tier: 2
        """
        prompt = (
            f"{_INSIGHT_SCORE_INSTRUCTIONS}\n\n"
            f"Schema: {json.dumps(_INSIGHT_SCORE_SCHEMA)}\n\n"
            f"Text:\n{text}\n\n"
            "Respond with ONLY a JSON object matching the schema."
        )
        try:
            resp = chat_via_firewalls(
                [{"role": "user", "content": prompt}],
                agent_id="notifications.insight_urgency",
                lane=Lane.CLASSIFIER,
                agent_max_tier=2,
            )
        except GatewayBlocked:
            logger.info(
                "insight urgency classification blocked by firewall",
                exc_info=True,
            )
            return None
        except Exception:  # noqa: BLE001
            logger.debug(
                "insight urgency classification failed",
                exc_info=True,
            )
            return None
        try:
            verdict = json.loads(resp.content)
        except (TypeError, ValueError):
            return None
        return verdict if isinstance(verdict, dict) else None

    # ----------------------------------------------------------
    # DB query helpers
    # ----------------------------------------------------------

    def _load_active_topics(self) -> list[dict[str, Any]]:
        """Load active topics with importance >= 5 for cross-referencing.

        Returns list of {contact_name, top_topic, max_topic_importance}.

        sensitivity_tier: 2
        """
        if self._db is None:
            return []
        try:
            return self._db.query(
                "SELECT contact_name, top_topic, "
                "max_topic_importance "
                "FROM mart_contact_summary "
                "WHERE max_topic_importance >= 5 "
                "  AND top_topic IS NOT NULL"
            )
        except Exception:  # noqa: BLE001
            logger.debug(
                "Could not load active topics", exc_info=True,
            )
            return []

    @staticmethod
    def _topic_match_score(
        text_lower: str,
        active_topics: list[dict[str, Any]],
    ) -> float:
        """Score an insight against active topics from DB.

        Returns max_topic_importance of best matching topic, or 0.

        sensitivity_tier: 1
        """
        best = 0.0
        for topic in active_topics:
            contact = str(topic.get("contact_name", "")).lower()
            topic_name = str(topic.get("top_topic", "")).lower()
            importance = float(topic.get("max_topic_importance", 0))

            if not contact or not topic_name:
                continue

            # Check if insight mentions the contact or their topic
            if contact in text_lower or topic_name in text_lower:
                best = max(best, importance)
        return best

    # ----------------------------------------------------------
    # Decision helpers
    # ----------------------------------------------------------

    def _make_decision(
        self,
        *,
        should_notify: bool,
        category: str,
        importance: float,
        message: str,
        reason: str,
        source_type: str,
        source_id: str,
        event_context: dict[str, Any],
    ) -> NotificationDecision:
        """Build a decision with preference and dedup checks.

        sensitivity_tier: 1
        """
        # Global mute check
        if should_notify and self._prefs.is_muted_globally():
            return NotificationDecision(
                should_notify=False,
                category=category,
                importance_score=importance,
                message=message,
                reason="All notifications are globally muted",
                dedupe_key="",
                event_context=event_context,
            )

        dedupe_key = self._compute_dedupe_key(source_type, source_id, category)

        # Category disabled check
        if should_notify and not self._prefs.is_category_enabled(category):
            return NotificationDecision(
                should_notify=False,
                category=category,
                importance_score=importance,
                message=message,
                reason=f"Category '{category}' is disabled",
                dedupe_key=dedupe_key,
                event_context=event_context,
            )

        # Dedup check
        if should_notify and self._prefs.has_recent_dedup(dedupe_key):
            return NotificationDecision(
                should_notify=False,
                category=category,
                importance_score=importance,
                message=message,
                reason="Duplicate notification within 24h",
                dedupe_key=dedupe_key,
                event_context=event_context,
            )

        return NotificationDecision(
            should_notify=should_notify,
            category=category,
            importance_score=importance,
            message=message,
            reason=reason,
            dedupe_key=dedupe_key,
            event_context=event_context,
        )

    @staticmethod
    def _compute_dedupe_key(
        source_type: str,
        source_id: str,
        category: str,
    ) -> str:
        """Compute a dedup key from source+category+date.

        Allows at most one notification per event per category per day.

        sensitivity_tier: 1
        """
        today = date.today().isoformat()
        raw = f"{source_type}:{source_id}:{category}:{today}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
