"""Pipeline Brain — interest-based smart refresh orchestrator.

Decides which pipeline models to run on each refresh cycle based on user
interest profile (from :class:`QueryTracker`) and data freshness signals
(from :class:`PipelineRunner`).  Not every mart needs to rebuild every time.

Priority tiers:
    1. **Critical** — Dashboard core models with new source data.
    2. **High** — Top-3 user interest domains with new data.
    3. **Medium** — Recently queried domains (last 7 days) with new data.
    4. **Low** — New data arrived but user hasn't asked about this domain.
    5. **Skipped** — No new source data; don't waste CPU.

sensitivity_tier: 1 (infrastructure metrics only)
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.core.query_tracker import QueryTracker
    from src.extensions.ingestion.model_generator import ModelGenerator
    from src.extensions.ingestion.review_flow import ReviewFlow
    from src.pipeline.runner import PipelineRunner

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlannedModel:
    """A model selected for execution with its priority and reason.

    sensitivity_tier: 1
    """

    name: str
    priority: str  # "critical" | "high" | "medium" | "low"
    reason: str


@dataclass(frozen=True)
class SkippedModel:
    """A model excluded from execution with the reason.

    sensitivity_tier: 1
    """

    name: str
    reason: str


@dataclass
class RefreshPlan:
    """Prioritized execution plan produced by :class:`PipelineBrain`.

    sensitivity_tier: 1
    """

    models: list[PlannedModel] = field(default_factory=list)
    skipped: list[SkippedModel] = field(default_factory=list)
    estimated_duration_seconds: float = 0.0
    full_duration_seconds: float = 0.0

    def add(self, model: str, priority: str, reason: str) -> None:
        """Add a model to the execution plan.

        sensitivity_tier: 1
        """
        self.models.append(
            PlannedModel(name=model, priority=priority, reason=reason),
        )

    def skip(self, model: str, reason: str) -> None:
        """Mark a model as skipped.

        sensitivity_tier: 1
        """
        self.skipped.append(SkippedModel(name=model, reason=reason))

    def get_ordered(self) -> list[str]:
        """Return model names sorted by priority (critical first).

        sensitivity_tier: 1
        """
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        sorted_models = sorted(
            self.models,
            key=lambda m: priority_order.get(m.priority, 99),
        )
        return [m.name for m in sorted_models]

    def summary(self) -> str:
        """Human-readable summary for logging and UI.

        sensitivity_tier: 1
        """
        counts = Counter(m.priority for m in self.models)
        return (
            f"Plan: {counts.get('critical', 0)} critical, "
            f"{counts.get('high', 0)} high, "
            f"{counts.get('medium', 0)} medium, "
            f"{counts.get('low', 0)} low, "
            f"{len(self.skipped)} skipped"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the plan to a JSON-compatible dict.

        sensitivity_tier: 1
        """
        return {
            "models": [
                {"name": m.name, "priority": m.priority, "reason": m.reason}
                for m in self.models
            ],
            "skipped": [
                {"name": s.name, "reason": s.reason}
                for s in self.skipped
            ],
            "estimated_duration_seconds": self.estimated_duration_seconds,
            "full_duration_seconds": self.full_duration_seconds,
            "summary": self.summary(),
        }


# ---------------------------------------------------------------------------
# Constants — Model → raw table dependencies (transitive closure)
# ---------------------------------------------------------------------------

MODEL_SOURCE_TABLES: dict[str, set[str]] = {
    # Staging — each reads from one raw table
    "stg_messages": {"raw_messages"},
    "stg_notes": {"raw_notes"},
    "stg_contacts": {"raw_contacts"},
    "stg_calendar_events": {"raw_calendar_events"},
    "stg_health_metrics": {"raw_health_metrics"},
    "stg_emails": {"raw_emails"},
    "stg_reminders": {"raw_reminders"},
    # Intermediate — transitive raw dependencies
    "int_personal_enriched": {"raw_messages", "raw_contacts"},
    "int_events_enriched": {
        "raw_calendar_events",
        "raw_contacts",
    },
    "int_daily_summary": {
        "raw_messages",
        "raw_calendar_events",
        "raw_notes",
    },
    "int_labeled_messages": {"raw_messages"},
    "int_contact_topics": {"raw_messages"},
    "int_communications_enriched": {"raw_messages", "raw_emails"},
    # Marts — transitive raw dependencies
    "mart_today": {
        "raw_calendar_events",
        "raw_messages",
        "raw_notes",
    },
    "mart_personal": {"raw_messages", "raw_contacts"},
    "mart_work": {
        "raw_calendar_events",
        "raw_messages",
        "raw_notes",
    },
    "mart_health": {"raw_health_metrics"},
    "mart_communications": {"raw_messages", "raw_emails"},
    "mart_contact_summary": {"raw_messages", "raw_contacts", "int_contact_topics"},
}


# ---------------------------------------------------------------------------
# Constants — Interest domain → pipeline models
# ---------------------------------------------------------------------------

DOMAIN_MODELS: dict[str, list[str]] = {
    "calendar": [
        "stg_calendar_events",
        "stg_reminders",
        "int_events_enriched",
        "mart_today",
    ],
    "contacts": [
        "stg_contacts",
        "int_personal_enriched",
        "int_contact_topics",
        "mart_personal",
        "mart_contact_summary",
    ],
    "work": [
        "stg_calendar_events",
        "stg_messages",
        "stg_notes",
        "int_events_enriched",
        "mart_work",
    ],
    "health": [
        "stg_health_metrics",
        "mart_health",
    ],
    "messages": [
        "stg_messages",
        "stg_emails",
        "int_labeled_messages",
        "int_communications_enriched",
        "mart_communications",
    ],
    "notes": [
        "stg_notes",
    ],
    "files": [],  # No pipeline models yet
    "music": [],  # Dynamically generated extension models only
    "social": [
        "stg_calendar_events",
        "stg_messages",
        "int_personal_enriched",
        "int_contact_topics",
        "mart_personal",
        "mart_contact_summary",
    ],
    "finance": [],  # No pipeline models yet
}


# ---------------------------------------------------------------------------
# Constants — Tier 1 (always run if stale)
# ---------------------------------------------------------------------------

ALWAYS_RUN: list[str] = [
    "stg_messages",
    "stg_calendar_events",
    "stg_contacts",
    "int_daily_summary",
    "mart_today",
]

# On-demand mart generation thresholds.
NEW_MART_INTEREST_THRESHOLD = 0.55
NEW_MART_MIN_ROWS = 50


# ---------------------------------------------------------------------------
# PipelineBrain
# ---------------------------------------------------------------------------


class PipelineBrain:
    """Decides which pipeline models to run on each refresh cycle.

    Uses the user's interest profile and data freshness signals to produce
    a prioritized :class:`RefreshPlan`.  Models with no new source data are
    skipped entirely.

    Args:
        query_tracker: Interest profile and query log provider.
        pipeline_runner: Pipeline runner for freshness and model listing.

    sensitivity_tier: 1
    """

    def __init__(
        self,
        query_tracker: QueryTracker,
        pipeline_runner: PipelineRunner,
        model_generator: ModelGenerator | None = None,
        review_flow: ReviewFlow | None = None,
    ) -> None:
        self._tracker = query_tracker
        self._runner = pipeline_runner
        self._model_gen = model_generator
        self._review = review_flow

    def plan_refresh(self) -> RefreshPlan:
        """Generate a prioritized refresh plan.

        This is the core intelligence of the system.  Gathers interest
        profile, data freshness, and available models, then assigns each
        model to a priority tier.

        Returns:
            A :class:`RefreshPlan` with models in execution order.

        sensitivity_tier: 1
        """
        # Gather inputs
        interests = self._tracker.get_interest_profile()
        domain_stats = {
            s.domain: s for s in self._tracker.get_domain_stats()
        }
        pending = self._runner.get_pending_changes()
        stale_tables = {t for t, n in pending.items() if n > 0}
        available_models = set(self._runner._all_model_names)

        plan = RefreshPlan()
        planned_set: set[str] = set()

        # ── TIER 1: ALWAYS RUN ──────────────────────────────────
        # Dashboard core models — run if ANY source tables have new data.
        for model in ALWAYS_RUN:
            if model in available_models and _has_stale_sources(
                model, stale_tables,
            ):
                plan.add(
                    model,
                    priority="critical",
                    reason="Dashboard dependency with new data",
                )
                planned_set.add(model)

        # ── TIER 2: HIGH INTEREST ───────────────────────────────
        # Top 3 interest domains with weight > 0.5 and new data.
        top_interests = [i for i in interests if i.weight > 0.5][:3]
        top_domains = {i.domain for i in top_interests}
        for interest in top_interests:
            domain_models = _get_models_for_domain(
                interest.domain, available_models,
            )
            for model in domain_models:
                if model not in planned_set and _has_stale_sources(
                    model, stale_tables,
                ):
                    plan.add(
                        model,
                        priority="high",
                        reason=f"Top interest '{interest.label}' + new data",
                    )
                    planned_set.add(model)

        # ── TIER 3: ACTIVE WITH NEW DATA ────────────────────────
        # Domains queried in the last 7 days that also have fresh data.
        for interest in interests:
            if interest.domain in top_domains:
                continue
            ds = domain_stats.get(interest.domain)
            if ds is None or ds.queries_last_7d < 1:
                continue
            if interest.weight <= 0.1:
                continue
            domain_models = _get_models_for_domain(
                interest.domain, available_models,
            )
            for model in domain_models:
                if model not in planned_set and _has_stale_sources(
                    model, stale_tables,
                ):
                    plan.add(
                        model,
                        priority="medium",
                        reason="Recent interest + new data",
                    )
                    planned_set.add(model)

        # ── TIER 4: NEW DATA, LOW INTEREST ──────────────────────
        # Data arrived but user hasn't asked about this domain recently.
        # Still process staging (cheap) but skip mart (expensive).
        for table in sorted(stale_tables):
            stg_name = table.replace("raw_", "")
            stg_model = f"stg_{stg_name}"
            if stg_model in available_models and stg_model not in planned_set:
                plan.add(
                    stg_model,
                    priority="low",
                    reason="New data, staging only (low interest)",
                )
                planned_set.add(stg_model)

        # ── SKIP ────────────────────────────────────────────────
        # Models whose source tables have NO new data.
        for model in sorted(available_models):
            if model not in planned_set:
                plan.skip(model, reason="No new source data")

        # ── DURATION ESTIMATES ──────────────────────────────────
        try:
            full_est = self._runner.dry_run().estimated_duration_seconds
        except Exception:  # noqa: BLE001
            logger.debug("Duration estimation failed", exc_info=True)
            full_est = 60.0

        total_models = max(len(available_models), 1)
        ratio = len(plan.models) / total_models
        plan.estimated_duration_seconds = round(full_est * ratio, 1)
        plan.full_duration_seconds = full_est

        logger.info("Pipeline plan: %s", plan.summary())
        return plan

    def check_demand_for_new_marts(self) -> list[str]:
        """Stage new generated marts for high-interest domains without one.

        Domains qualify when:
        - Interest weight exceeds ``NEW_MART_INTEREST_THRESHOLD``
        - No core mart is assigned in the interest profile
        - Existing raw data exceeds ``NEW_MART_MIN_ROWS``

        Generated models are staged via :class:`ReviewFlow` for human review.

        Returns:
            List of domains for which new marts were staged.

        sensitivity_tier: 1
        """
        try:
            from src.extensions.ingestion.model_generator import ModelGenerator
            from src.extensions.ingestion.review_flow import ReviewFlow
            from src.extensions.ingestion.schema_discovery import (
                DiscoveredMapping,
                FieldMapping,
            )
        except Exception:  # noqa: BLE001
            logger.warning("Model generation dependencies unavailable", exc_info=True)
            return []

        if self._model_gen is None:
            self._model_gen = ModelGenerator()
        if self._review is None:
            self._review = ReviewFlow(db_engine=getattr(self._runner, "_db", None))

        interests = self._tracker.get_interest_profile()
        available_models = set(self._runner._all_model_names)
        generated_domains: list[str] = []

        for interest in interests:
            if interest.mart is not None:
                continue
            if interest.weight < NEW_MART_INTEREST_THRESHOLD:
                continue
            if not interest.raw_tables:
                continue

            connector_id = f"auto-{interest.domain}"
            if self._review.get_staged(connector_id) is not None:
                continue

            # Skip if any existing extension mart already covers this domain.
            ext_mart_prefix = f"ext_mart_{interest.domain}_"
            if any(m.startswith(ext_mart_prefix) for m in available_models):
                continue

            table_counts = {
                table: self._count_table_rows(table)
                for table in interest.raw_tables
            }
            total_rows = sum(max(0, n) for n in table_counts.values())
            if total_rows < NEW_MART_MIN_ROWS:
                continue

            # Seed generation from the table with the highest observed volume.
            source_table = max(table_counts, key=table_counts.get)
            fields = self._build_field_mappings(source_table, FieldMapping)
            if not fields:
                continue

            dedup_key = (
                ("id",)
                if any(f.target_column == "id" for f in fields)
                else (fields[0].target_column,)
            )
            mapping = DiscoveredMapping(
                tool_name=f"brain_auto_{interest.domain}",
                target_table=source_table,
                is_new_table=True,
                domain=interest.domain,
                confidence=1.0,
                analysis_method="brain_auto",
                fields=tuple(fields),
                dedup_key=dedup_key,
                suggested_schedule="manual",
            )

            try:
                preview = self._model_gen.generate(
                    mapping=mapping,
                    connector_id=connector_id,
                    force_full_pipeline=True,
                )
                self._review.stage(preview)
                generated_domains.append(interest.domain)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Auto mart generation failed for domain %s",
                    interest.domain,
                    exc_info=True,
                )

        return generated_domains

    def _count_table_rows(self, table_name: str) -> int:
        """Return row count for a raw table, or 0 on query failure.

        sensitivity_tier: 1
        """
        try:
            rows = self._runner._db.query(  # noqa: SLF001
                f"SELECT COUNT(*) AS n FROM {table_name}",
            )
            return int(rows[0]["n"]) if rows else 0
        except Exception:  # noqa: BLE001
            return 0

    def _build_field_mappings(self, table_name: str, field_cls: Any) -> list[Any]:
        """Build minimal ``FieldMapping`` entries from PRAGMA table_info.

        sensitivity_tier: 1
        """
        try:
            rows = self._runner._db.query(  # noqa: SLF001
                f"PRAGMA table_info({table_name})",
            )
        except Exception:  # noqa: BLE001
            return []

        fields: list[Any] = []
        for row in rows:
            column = str(row.get("name", ""))
            if not column:
                continue
            raw_type = str(row.get("type", "TEXT"))
            target_type = _map_sqlite_type(raw_type)
            tier = 1 if column in {"id", "source", "created_at", "updated_at"} else 2
            fields.append(
                field_cls(
                    source_name=column,
                    target_column=column,
                    source_type="string",
                    target_type=target_type,
                    sensitivity_tier=tier,
                    confidence=1.0,
                    tier_source="brain_auto",
                    transform=None,
                    is_new_column=False,
                )
            )
        return fields


# ---------------------------------------------------------------------------
# Module-level helpers (for testability)
# ---------------------------------------------------------------------------


def _has_stale_sources(model_name: str, stale_tables: set[str]) -> bool:
    """Check if any of a model's source raw tables have new data.

    For extension models (not in ``MODEL_SOURCE_TABLES``), assumes stale
    if *any* table has new data.

    sensitivity_tier: 1
    """
    sources = MODEL_SOURCE_TABLES.get(model_name)
    if sources is None:
        # Extension model — assume stale if anything is stale.
        return bool(stale_tables)
    return bool(sources & stale_tables)


def _get_models_for_domain(
    domain: str,
    available_models: set[str],
) -> list[str]:
    """Map an interest domain to its pipeline models.

    Returns both the hard-coded domain models and any ``ext_*`` extension
    models whose name contains the domain.

    sensitivity_tier: 1
    """
    base = [m for m in DOMAIN_MODELS.get(domain, []) if m in available_models]
    ext = [
        m
        for m in sorted(available_models)
        if m.startswith("ext_") and domain in m
    ]
    return base + ext


def _map_sqlite_type(raw_type: str) -> str:
    """Map SQLite ``PRAGMA table_info`` types to model target types.

    sensitivity_tier: 1
    """
    upper = raw_type.upper()
    if upper in {"INTEGER", "INT", "BIGINT", "SMALLINT"}:
        return "INTEGER"
    if upper in {"REAL", "DOUBLE", "FLOAT", "DECIMAL", "NUMERIC"}:
        return "REAL"
    if upper in {"BLOB", "BYTEA"}:
        return "BLOB"
    return "TEXT"
