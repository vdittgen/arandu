"""Query tracking and interest profiling.

Logs every question the user asks, classifies it by domain, and
computes a weighted interest profile over time.

Data is stored in DuckDB internal tables (``_query_log`` and
``_interest_profile``), keeping it alongside the rest of the analytical
data without conflicting with ``raw_*`` or ``ext_*`` table namespaces.

sensitivity_tier: 1 (aggregated metadata only)
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from src.core.llm_classifier import LLMClassifier
from src.core.sqlite.engine import DatabaseEngine
from src.models.llm_provider import LLMProvider

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------


@dataclass(frozen=True)
class InterestArea:
    """A single interest area with computed weight.

    sensitivity_tier: 1
    """

    domain: str
    label: str
    description: str
    weight: float  # 0.0–1.0 normalized
    query_count: int
    queries_per_week: float
    trending: bool  # True if last 7d > previous 7d
    explicit: bool  # True if from DEFAULT_INTERESTS with explicit=True
    raw_tables: list[str] = field(default_factory=list)
    mart: str | None = None


@dataclass(frozen=True)
class DomainStats:
    """Per-domain query statistics.

    sensitivity_tier: 1
    """

    domain: str
    total_queries: int
    last_queried_at: str | None  # ISO 8601
    trend: str  # "rising" | "stable" | "declining" | "new"
    queries_last_7d: int
    queries_last_30d: int


# ------------------------------------------------------------------
# Default interest areas
# ------------------------------------------------------------------

DEFAULT_INTERESTS: list[dict[str, Any]] = [
    {
        "domain": "calendar",
        "label": "Schedule & Events",
        "description": "Your calendar, meetings, and upcoming events",
        "explicit": True,
        "raw_tables": ["raw_calendar_events", "raw_reminders"],
        "mart": "mart_today",
    },
    {
        "domain": "contacts",
        "label": "People & Relationships",
        "description": "Your contacts and who you interact with",
        "explicit": True,
        "raw_tables": ["raw_contacts"],
        "mart": "mart_personal",
    },
    {
        "domain": "work",
        "label": "Work & Productivity",
        "description": "Work meetings, tasks, and projects",
        "explicit": True,
        "raw_tables": ["raw_calendar_events", "raw_messages", "raw_notes"],
        "mart": "mart_work",
    },
    {
        "domain": "health",
        "label": "Health & Wellness",
        "description": "Exercise, sleep, mood, and health metrics",
        "explicit": False,
        "raw_tables": ["raw_health_metrics"],
        "mart": "mart_health",
    },
    {
        "domain": "messages",
        "label": "Communications",
        "description": "Messages, emails, and conversations",
        "explicit": True,
        "raw_tables": ["raw_messages", "raw_emails"],
        "mart": "mart_communications",
    },
    {
        "domain": "notes",
        "label": "Notes & Ideas",
        "description": "Your notes, journal entries, and ideas",
        "explicit": False,
        "raw_tables": ["raw_notes"],
        "mart": None,
    },
    {
        "domain": "music",
        "label": "Music & Media",
        "description": "Listening history and media consumption",
        "explicit": False,
        "raw_tables": ["raw_listening_history"],
        "mart": None,
    },
    {
        "domain": "files",
        "label": "Documents & Files",
        "description": "Your local documents and files",
        "explicit": False,
        "raw_tables": ["raw_files"],
        "mart": None,
    },
    {
        "domain": "social",
        "label": "Social Life",
        "description": "Social events, friends, plans",
        "explicit": False,
        "raw_tables": ["raw_calendar_events", "raw_messages"],
        "mart": None,
    },
    {
        "domain": "finance",
        "label": "Finance",
        "description": "Spending, budgets, financial transactions",
        "explicit": False,
        "raw_tables": [],
        "mart": None,
    },
]

_DEFAULT_MAP: dict[str, dict[str, Any]] = {
    d["domain"]: d for d in DEFAULT_INTERESTS
}


# ------------------------------------------------------------------
# Domain classification (LLM-driven)
# ------------------------------------------------------------------

# Legacy cue lists kept ONLY for the structural fast-path in
# ``query_engine._fast_route`` — used before falling through to the
# LLM router.  ``classify_question_domain`` no longer consults these.
DOMAIN_KEYWORDS: dict[str, frozenset[str]] = {
    "calendar": frozenset({
        "meeting", "meetings", "calendar", "event", "events",
        "today", "tomorrow", "schedule", "appointment", "agenda",
        "upcoming", "next week", "this week", "busy", "free",
        "hoje", "amanhã", "amanha", "reunião", "reuniao",
        "compromisso", "compromissos", "semana",
    }),
    "contacts": frozenset({
        "who", "person", "friend", "family", "colleague",
        "relationship", "contact", "talked to", "met with",
        "quem", "amigo", "família", "familia", "colega",
        "contato",
    }),
    "health": frozenset({
        "health", "sleep", "exercise", "heart rate", "steps",
        "workout", "weight", "feeling", "mood", "energy",
        "medication", "doctor", "calories", "bpm",
        "blood pressure", "fitness", "walk", "walked", "running",
        "saúde", "saude", "sono", "exercício", "exercicio",
        "peso", "humor", "energia", "médico", "medico",
    }),
    "work": frozenset({
        "project", "deadline", "task", "work", "presentation",
        "report", "client", "team", "office", "professional",
        "meeting notes",
        "projeto", "prazo", "tarefa", "trabalho", "relatório",
        "relatorio", "cliente", "equipe",
    }),
    "messages": frozenset({
        "message", "messages", "email", "emails", "text", "said",
        "wrote", "conversation", "conversations", "reply", "sent",
        "thread", "chat",
        "mensagem", "mensagens", "conversa", "conversas",
        "respondeu", "enviou", "falou", "disse",
    }),
    "notes": frozenset({
        "note", "notes", "wrote down", "idea", "journal",
        "thought", "remember", "reminded", "brainstorm",
        "nota", "notas", "ideia", "lembrar", "anotação",
        "anotacao",
    }),
    "files": frozenset({
        "file", "files", "document", "pdf", "photo", "download",
        "folder", "saved",
        "arquivo", "arquivos", "documento", "foto", "pasta",
    }),
    "music": frozenset({
        "song", "songs", "music", "listen", "played", "track",
        "artist", "playlist", "spotify", "album",
        "música", "musica", "ouvir",
    }),
    "social": frozenset({
        "going out", "dinner", "party", "birthday", "hangout",
        "weekend plan", "brunch", "concert", "game night",
        "jantar", "festa", "aniversário", "aniversario",
        "fim de semana", "churrasco",
    }),
    "finance": frozenset({
        "money", "spent", "spend", "budget", "cost", "payment",
        "bill", "subscription", "income", "salary", "expense",
        "financial",
        "dinheiro", "gasto", "orçamento", "orcamento", "custo",
        "pagamento", "conta", "salário", "salario", "despesa",
    }),
}

_DOMAIN_ENUM: tuple[str, ...] = (
    "calendar", "contacts", "health", "work", "messages",
    "notes", "files", "music", "social", "finance",
    "general",
)

_DOMAIN_SCHEMA: dict[str, Any] = {
    "domain": " | ".join(_DOMAIN_ENUM),
}

_DOMAIN_INSTRUCTIONS = (
    "Classify the user's question into ONE of the listed domains.\n"
    "- calendar: scheduling, meetings, agenda, today/tomorrow plans\n"
    "- contacts: people, relationships, who/what about a person\n"
    "- health: sleep, exercise, mood, body metrics, doctors\n"
    "- work: projects, deadlines, productivity, work meetings\n"
    "- messages: emails, chats, replies, conversation threads\n"
    "- notes: notes, journals, ideas the user wrote down\n"
    "- files: documents, downloads, attachments\n"
    "- music: songs, playlists, listening habits\n"
    "- social: dinners, parties, hangouts, weekend plans\n"
    "- finance: money, spending, salary, subscriptions, budgeting\n"
    "- general: doesn't fit any other domain."
)


# ------------------------------------------------------------------
# QueryTracker
# ------------------------------------------------------------------


class QueryTracker:
    """Track user queries and build an interest profile.

    Stores data in DuckDB tables ``_query_log`` and
    ``_interest_profile``.  The tables are created automatically on
    first use (idempotent).

    sensitivity_tier: 1 (aggregated metadata only)
    """

    def __init__(
        self,
        db_engine: DatabaseEngine,
        llm_provider: LLMProvider | None = None,
    ) -> None:
        self._db = db_engine
        self._llm = llm_provider
        self._classifier: LLMClassifier | None = None
        self._ensure_tables()

    # ----------------------------------------------------------
    # Table setup
    # ----------------------------------------------------------

    def _ensure_tables(self) -> None:
        """Create internal tracking tables if they don't exist.

        sensitivity_tier: 1
        """
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS _query_log (
                id              VARCHAR PRIMARY KEY,
                question        VARCHAR NOT NULL,
                domain          VARCHAR NOT NULL DEFAULT 'general',
                sub_topics      VARCHAR DEFAULT '[]',
                entities        VARCHAR DEFAULT '[]',
                sources_used    VARCHAR DEFAULT '[]',
                latency_ms      DOUBLE DEFAULT 0.0,
                asked_at        TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS _interest_profile (
                domain          VARCHAR PRIMARY KEY,
                query_count     INTEGER DEFAULT 0,
                last_queried_at TEXT,
                updated_at      TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

    # ----------------------------------------------------------
    # Domain classification
    # ----------------------------------------------------------

    def classify_question_domain(self, question: str) -> str:
        """Classify a question into a known domain via the LLM.

        Falls back to ``"general"`` when the LLM is unavailable or
        returns an unrecognised domain.  Cached per question text so
        repeats are free.

        sensitivity_tier: 1 (stores fingerprint + verdict, not text)
        """
        if not question or not question.strip():
            return "general"
        if self._classifier is None:
            provider = self._llm
            if provider is None:
                try:
                    from src.models.llm_provider import (
                        create_provider_from_settings,
                    )
                    provider = create_provider_from_settings(
                        background=True,
                    )
                except Exception:  # noqa: BLE001
                    provider = None
            self._classifier = LLMClassifier(
                llm_provider=provider, db_engine=self._db,
            )

        result = self._classifier.classify(
            kind="question_domain",
            text=question,
            schema=_DOMAIN_SCHEMA,
            instructions=_DOMAIN_INSTRUCTIONS,
        )
        if not result:
            return "general"
        domain = str(result.get("domain", "general")).strip().lower()
        if domain not in _DOMAIN_ENUM:
            return "general"
        return domain

    # ----------------------------------------------------------
    # Query logging
    # ----------------------------------------------------------

    def log_query(
        self,
        question: str,
        domain: str,
        sub_topics: list[str] | None = None,
        entities: list[str] | None = None,
        sources_used: list[str] | None = None,
        latency_ms: float = 0.0,
    ) -> None:
        """Log a user query and update the interest profile.

        Args:
            question: The original question text.
            domain: Classified domain (from ``classify_question_domain``).
            sub_topics: Optional list of detected intents/topics.
            entities: Optional list of extracted entity names.
            sources_used: Optional list of data sources that contributed.
            latency_ms: Response latency in milliseconds.

        sensitivity_tier: 2 (stores question text)
        """
        query_id = str(uuid.uuid4())
        self._db.execute(
            "INSERT INTO _query_log "
            "(id, question, domain, sub_topics, entities, "
            "sources_used, latency_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                query_id,
                question,
                domain,
                json.dumps(sub_topics or []),
                json.dumps(entities or []),
                json.dumps(sources_used or []),
                latency_ms,
            ],
        )
        now_ts = datetime.now(timezone.utc).isoformat()
        self._db.execute(
            """
            INSERT INTO _interest_profile
                (domain, query_count, last_queried_at, updated_at)
            VALUES (?, 1, ?, ?)
            ON CONFLICT (domain) DO UPDATE SET
                query_count     = _interest_profile.query_count + 1,
                last_queried_at = excluded.last_queried_at,
                updated_at      = excluded.updated_at
            """,
            [domain, now_ts, now_ts],
        )

    # ----------------------------------------------------------
    # Interest profile
    # ----------------------------------------------------------

    def get_interest_profile(
        self,
        overrides: dict[str, int] | None = None,
    ) -> list[InterestArea]:
        """Compute the full interest profile, sorted by weight.

        Weight formula per domain::

            recency_factor = 1.0 if last_query < 7d
                             0.7 if last_query < 30d
                             0.3 otherwise
            raw = log2(query_count + 1) * recency_factor
            if explicit: raw += 2.0       # dashboard baseline
            if overridden: raw += boost   # manual priority
            weight = raw / max_raw        # normalize to 0–1

        Args:
            overrides: Optional ``{domain: rank}`` mapping.  Lower
                rank numbers get a higher weight boost.

        Returns:
            Interest areas sorted by weight descending.

        sensitivity_tier: 1
        """
        now = datetime.now(timezone.utc)
        seven_days_ago = now - timedelta(days=7)
        thirty_days_ago = now - timedelta(days=30)

        # Fetch profile rows
        profile_rows = self._db.query(
            "SELECT domain, query_count, last_queried_at "
            "FROM _interest_profile"
        )
        profile_map: dict[str, dict[str, Any]] = {
            row["domain"]: row for row in profile_rows
        }

        # Fetch 7-day and previous-7-day counts for trending
        recent_counts = self._query_counts_since(seven_days_ago)
        prev_counts = self._query_counts_between(
            thirty_days_ago, seven_days_ago,
        )

        # Compute queries_per_week (based on last 30 days)
        thirty_day_counts = self._query_counts_since(thirty_days_ago)

        # Build raw weight for every domain (defaults + any extra)
        all_domains = set(_DEFAULT_MAP.keys()) | set(profile_map.keys())
        raw_weights: dict[str, float] = {}
        areas_data: dict[str, dict[str, Any]] = {}

        for domain in all_domains:
            defaults = _DEFAULT_MAP.get(domain, {})
            profile = profile_map.get(domain, {})

            query_count = profile.get("query_count", 0)
            last_queried = profile.get("last_queried_at")
            explicit = defaults.get("explicit", False)

            # Recency factor
            recency = 0.3
            if last_queried is not None:
                if isinstance(last_queried, str):
                    try:
                        last_queried = datetime.fromisoformat(last_queried)
                    except (ValueError, TypeError):
                        last_queried = None
                if last_queried is not None:
                    no_tz = (
                        hasattr(last_queried, "tzinfo")
                        and last_queried.tzinfo is None
                    )
                    if no_tz:
                        last_queried = last_queried.replace(
                            tzinfo=timezone.utc,
                        )
                    age = now - last_queried
                    if age < timedelta(days=7):
                        recency = 1.0
                    elif age < timedelta(days=30):
                        recency = 0.7

            raw = math.log2(query_count + 1) * recency
            if explicit:
                raw += 2.0

            # Apply manual override boost
            if overrides and domain in overrides:
                rank = overrides[domain]
                # Lower rank = higher boost (rank 1 = +10, rank 2 = +9, ...)
                raw += max(0, 11 - rank)

            count_7d = recent_counts.get(domain, 0)
            count_prev_7d = prev_counts.get(domain, 0)
            count_30d = thirty_day_counts.get(domain, 0)

            # Trending
            if query_count < 3:
                trend_flag = False
            elif count_7d > count_prev_7d and count_7d >= 3:
                trend_flag = True
            else:
                trend_flag = False

            # Queries per week (avg over last 30 days)
            qpw = count_30d / (30 / 7) if count_30d > 0 else 0.0

            raw_weights[domain] = raw
            areas_data[domain] = {
                "domain": domain,
                "label": defaults.get("label", domain.title()),
                "description": defaults.get("description", ""),
                "query_count": query_count,
                "queries_per_week": round(qpw, 1),
                "trending": trend_flag,
                "explicit": explicit,
                "raw_tables": defaults.get("raw_tables", []),
                "mart": defaults.get("mart"),
            }

        # Normalize weights to 0.0–1.0
        max_raw = max(raw_weights.values()) if raw_weights else 1.0
        if max_raw == 0:
            max_raw = 1.0

        results: list[InterestArea] = []
        for domain, data in areas_data.items():
            weight = raw_weights[domain] / max_raw
            results.append(
                InterestArea(
                    weight=round(weight, 3),
                    **data,
                )
            )

        results.sort(key=lambda a: a.weight, reverse=True)
        return results

    # ----------------------------------------------------------
    # Domain stats
    # ----------------------------------------------------------

    def get_domain_stats(self) -> list[DomainStats]:
        """Per-domain query statistics with trend detection.

        Trend detection::

            rising:    7d_count > prev_7d_count and 7d_count >= 3
            declining: 7d_count < prev_7d_count * 0.5 and prev >= 3
            new:       total_queries < 3
            stable:    everything else

        sensitivity_tier: 1
        """
        now = datetime.now(timezone.utc)
        seven_days_ago = now - timedelta(days=7)
        thirty_days_ago = now - timedelta(days=30)

        profile_rows = self._db.query(
            "SELECT domain, query_count, last_queried_at "
            "FROM _interest_profile "
            "ORDER BY query_count DESC"
        )

        recent_counts = self._query_counts_since(seven_days_ago)
        prev_counts = self._query_counts_between(
            thirty_days_ago, seven_days_ago,
        )
        thirty_day_counts = self._query_counts_since(thirty_days_ago)

        results: list[DomainStats] = []
        for row in profile_rows:
            domain = row["domain"]
            total = row["query_count"]
            last_q = row["last_queried_at"]
            count_7d = recent_counts.get(domain, 0)
            count_prev = prev_counts.get(domain, 0)
            count_30d = thirty_day_counts.get(domain, 0)

            if total < 3:
                trend = "new"
            elif count_7d > count_prev and count_7d >= 3:
                trend = "rising"
            elif count_prev >= 3 and count_7d < count_prev * 0.5:
                trend = "declining"
            else:
                trend = "stable"

            last_str: str | None = None
            if last_q is not None:
                if isinstance(last_q, str):
                    last_str = last_q
                else:
                    last_str = last_q.isoformat()

            results.append(
                DomainStats(
                    domain=domain,
                    total_queries=total,
                    last_queried_at=last_str,
                    trend=trend,
                    queries_last_7d=count_7d,
                    queries_last_30d=count_30d,
                )
            )

        return results

    # ----------------------------------------------------------
    # Top question patterns
    # ----------------------------------------------------------

    def get_top_questions(
        self, limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Return the most frequent domain patterns.

        Groups queries by domain and returns count per domain,
        ordered by frequency descending.

        sensitivity_tier: 1 (aggregated counts only)
        """
        rows = self._db.query(
            "SELECT domain, COUNT(*) AS count "
            "FROM _query_log "
            "GROUP BY domain "
            "ORDER BY count DESC "
            f"LIMIT {int(limit)}"
        )
        return [
            {"domain": r["domain"], "count": r["count"]}
            for r in rows
        ]

    # ----------------------------------------------------------
    # Recent questions (for pattern detection)
    # ----------------------------------------------------------

    def get_recent_questions(
        self,
        limit: int = 100,
        days: int = 7,
    ) -> list[dict[str, Any]]:
        """Return recent questions with metadata from _query_log.

        Used by :class:`InsightGenerator` to detect question patterns
        over recent history.

        Args:
            limit: Maximum number of questions to return.
            days: How many days back to look.

        Returns:
            List of dicts with ``question``, ``domain``, ``entities``,
            ``asked_at`` keys, newest first.

        sensitivity_tier: 2 (contains question text)
        """
        since = (
            datetime.now(timezone.utc) - timedelta(days=days)
        ).isoformat()
        rows = self._db.query(
            "SELECT question, domain, entities, asked_at "
            "FROM _query_log "
            "WHERE asked_at >= ? "
            "ORDER BY asked_at DESC "
            f"LIMIT {int(limit)}",
            [since],
        )
        return [dict(r) for r in rows]

    # ----------------------------------------------------------
    # Domain boost (feedback loop)
    # ----------------------------------------------------------

    def boost_domain(
        self,
        domain: str,
        boost_count: int = 2,
    ) -> None:
        """Manually boost a domain's query count.

        Used by the feedback loop: when a user follows up on an
        insight, the associated domain gets a boost so that future
        interest profile weights reflect that engagement.

        Uses UPSERT on ``_interest_profile``.

        Args:
            domain: The domain to boost (e.g. ``"calendar"``).
            boost_count: Number of synthetic queries to add.

        sensitivity_tier: 1
        """
        now_ts = datetime.now(timezone.utc).isoformat()
        self._db.execute(
            """
            INSERT INTO _interest_profile
                (domain, query_count, last_queried_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (domain) DO UPDATE SET
                query_count = _interest_profile.query_count
                    + excluded.query_count,
                updated_at  = excluded.updated_at
            """,
            [domain, boost_count, now_ts, now_ts],
        )

    # ----------------------------------------------------------
    # Private helpers
    # ----------------------------------------------------------

    def _query_counts_since(
        self, since: datetime,
    ) -> dict[str, int]:
        """Count queries per domain since a given timestamp.

        sensitivity_tier: 1
        """
        rows = self._db.query(
            "SELECT domain, COUNT(*) AS cnt "
            "FROM _query_log "
            "WHERE asked_at >= ? "
            "GROUP BY domain",
            [since.isoformat()],
        )
        return {r["domain"]: r["cnt"] for r in rows}

    def _query_counts_between(
        self,
        start: datetime,
        end: datetime,
    ) -> dict[str, int]:
        """Count queries per domain in a time window.

        sensitivity_tier: 1
        """
        rows = self._db.query(
            "SELECT domain, COUNT(*) AS cnt "
            "FROM _query_log "
            "WHERE asked_at >= ? AND asked_at < ? "
            "GROUP BY domain",
            [start.isoformat(), end.isoformat()],
        )
        return {r["domain"]: r["cnt"] for r in rows}
