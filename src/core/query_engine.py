"""Hybrid GraphRAG query engine for Arandu.

Combines vector search (ChromaDB), graph traversal (Kuzu), and structured
SQL queries (DuckDB) into a unified QueryContext.  An LLM-driven router
decides which databases to query based on the user's question.

Retrieval pipeline:

  1. LLM Routing — LLM produces a RetrievalPlan (tables, collections, graph)
  2. Structured Queries — execute DuckDB queries from the plan
  3. Vector Search — semantic similarity on planned ChromaDB collections
  4. Entity Extraction — lightweight NER from vector results + contact index
  5. Graph Traversal — optional Kuzu traversals when the plan requests it
  6. Context Assembly — merge, deduplicate, cap, and tag with sensitivity

sensitivity_tier: 3 (processes all raw user data during retrieval)
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from src.core.chromadb.engine import COLLECTION_NAMES, VectorEngine
from src.core.kuzu.engine import GraphEngine
from src.core.name_matching import build_unique_index, find_names
from src.core.profiler import timed
from src.core.sqlite.engine import DatabaseEngine
from src.core.topic_loader import load_topic_contacts
from src.models.llm_provider import LLMProvider

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------


@dataclass(frozen=True)
class ContextItem:
    """A single piece of context from any of the three databases.

    sensitivity_tier: inherits from source record
    """

    source_id: str
    source_type: str  # "vector", "graph", "structured"
    source_db: str  # "chromadb", "kuzu", "duckdb"
    content: str
    relevance: float  # 0.0 to 1.0
    sensitivity_tier: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryContext:
    """Assembled context from a hybrid retrieval query.

    sensitivity_tier: 3 (may contain data from any tier)
    """

    question: str
    vector_results: list[dict[str, Any]] = field(
        default_factory=list,
    )
    graph_context: list[dict[str, Any]] = field(
        default_factory=list,
    )
    structured_data: list[dict[str, Any]] = field(
        default_factory=list,
    )
    metadata: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------
# LLM-driven retrieval plan
# ------------------------------------------------------------------


@dataclass(frozen=True)
class DuckDBQuerySpec:
    """A single DuckDB query to execute as part of the retrieval plan.

    sensitivity_tier: 1
    """

    table: str
    columns: list[str]
    where: str | None = None
    order_by: str | None = None
    limit: int = 10


@dataclass(frozen=True)
class RetrievalPlan:
    """LLM-generated plan for which databases to query.

    sensitivity_tier: 1
    """

    duckdb_queries: list[DuckDBQuerySpec]
    chromadb_collections: list[str]
    use_graph: bool
    reasoning: str


# ------------------------------------------------------------------
# Table/column whitelist for SQL injection prevention
# ------------------------------------------------------------------

_ALLOWED_TABLES: dict[str, frozenset[str]] = {
    "raw_messages": frozenset({
        "id", "source", "sender", "recipient", "content",
        "timestamp", "metadata", "sensitivity_tier", "created_at",
        "is_from_me", "chat_name", "is_group", "sender_name",
    }),
    "raw_calendar_events": frozenset({
        "id", "title", "description", "start_time", "end_time",
        "location", "attendees", "sensitivity_tier", "created_at",
        "is_all_day", "event_origin", "self_response_status",
    }),
    "raw_notes": frozenset({
        "id", "title", "content", "source", "created_at",
        "updated_at", "tags", "sensitivity_tier", "filepath",
    }),
    "raw_health_metrics": frozenset({
        "id", "metric_type", "value", "unit", "recorded_at",
        "source", "sensitivity_tier", "created_at",
    }),
    "raw_contacts": frozenset({
        "id", "name", "email", "phone", "relationship", "notes",
        "last_contact", "sensitivity_tier", "created_at",
        "birthday", "address",
    }),
    "raw_files": frozenset({
        "id", "filepath", "filename", "filetype", "size_bytes",
        "created_at", "modified_at", "content_preview",
        "sensitivity_tier",
    }),
    "raw_emails": frozenset({
        "id", "source", "message_id", "subject", "from_address",
        "to_addresses", "date", "body_preview", "is_read",
        "folder", "labels", "sensitivity_tier", "created_at",
    }),
    "raw_reminders": frozenset({
        "id", "source", "title", "due_date", "notes", "completed",
        "list_name", "sensitivity_tier", "created_at",
    }),
    "raw_workouts": frozenset({
        "id", "source", "workout_type", "duration_min", "calories",
        "heart_rate_avg", "date", "sensitivity_tier", "created_at",
    }),
    "raw_listening_history": frozenset({
        "id", "source", "track_name", "artist", "album",
        "played_at", "duration_ms", "context_type",
        "sensitivity_tier", "created_at",
    }),
    "raw_voice_memos": frozenset({
        "id", "source", "title", "duration_seconds", "recorded_at",
        "transcript", "sensitivity_tier", "created_at",
    }),
}

# Dangerous SQL patterns to reject in WHERE clauses
_DANGEROUS_PATTERNS = re.compile(
    r"(;|--|/\*|\*/|DROP\s|DELETE\s|INSERT\s|UPDATE\s|ALTER\s|CREATE\s"
    r"|TRUNCATE\s|EXEC\s|EXECUTE\s|UNION\s)",
    re.IGNORECASE,
)


# ------------------------------------------------------------------
# Rule-based router (fast, no LLM call)
# ------------------------------------------------------------------

# Reuse domain keywords from query_tracker for consistency
from src.core.query_tracker import DOMAIN_KEYWORDS  # noqa: E402

# Map domains → DuckDB tables to query with sensible defaults
# Calendar event tables retrieval must scope to entries the user
# actually owns. Team-awareness and subscribed-calendar events live in
# ``raw_calendar_events`` alongside personal ones (so the dashboard's
# awareness panels can surface them) but Brain answers should treat
# only ``event_origin = 'personal'`` rows as "my events". Mirrors the
# discipline of ``src.core.calendar_filters.personal_events_for_date``.
_CALENDAR_TABLES: frozenset[str] = frozenset({
    "raw_calendar_events",
    "int_events_enriched",
})


_DOMAIN_TO_TABLES: dict[str, list[DuckDBQuerySpec]] = {
    "calendar": [DuckDBQuerySpec(
        table="raw_calendar_events",
        columns=["id", "title", "start_time", "end_time", "location",
                 "attendees", "is_all_day"],
        order_by="start_time DESC",
        limit=10,
    )],
    "health": [DuckDBQuerySpec(
        table="raw_health_metrics",
        columns=["id", "metric_type", "value", "unit", "recorded_at"],
        order_by="recorded_at DESC",
        limit=15,
    )],
    "contacts": [DuckDBQuerySpec(
        table="raw_contacts",
        columns=["id", "name", "email", "phone", "relationship",
                 "birthday"],
        limit=10,
    )],
    "messages": [
        DuckDBQuerySpec(
            table="raw_messages",
            columns=["id", "sender_name", "content", "timestamp",
                     "chat_name", "is_from_me"],
            order_by="timestamp DESC",
            limit=10,
        ),
        DuckDBQuerySpec(
            table="raw_emails",
            columns=["id", "subject", "from_address", "date",
                     "body_preview", "folder"],
            order_by="date DESC",
            limit=10,
        ),
    ],
    "notes": [DuckDBQuerySpec(
        table="raw_notes",
        columns=["id", "title", "content", "created_at", "tags"],
        order_by="created_at DESC",
        limit=10,
    )],
    "files": [DuckDBQuerySpec(
        table="raw_files",
        columns=["id", "filepath", "filename", "filetype", "modified_at"],
        order_by="modified_at DESC",
        limit=10,
    )],
    "music": [DuckDBQuerySpec(
        table="raw_listening_history",
        columns=["id", "track_name", "artist", "album", "played_at"],
        order_by="played_at DESC",
        limit=10,
    )],
}

_DOMAIN_TO_COLLECTIONS: dict[str, list[str]] = {
    "health": ["health"],
    "contacts": ["social"],
    "notes": ["personal", "ideas"],
    "work": ["work"],
    "social": ["social"],
}

# Domains that benefit from graph traversal
_GRAPH_DOMAINS: frozenset[str] = frozenset({"contacts", "social"})

# Patterns that signal semantic/fuzzy search (go to ChromaDB, not rules)
_SEMANTIC_STARTERS: tuple[str, ...] = (
    "tell me about", "what do i know about", "what do you know about",
    "summarize", "how do i feel about", "what are my thoughts on",
    "me fale sobre", "o que eu sei sobre", "resuma",
)


class RuleBasedRouter:
    """Fast keyword-driven query router. No LLM call needed.

    Matches user questions against domain keywords to produce a
    RetrievalPlan. Returns (plan, confidence) where confidence
    indicates match quality.

    sensitivity_tier: 1 (only processes question text)
    """

    def plan(
        self,
        question: str,
        reference_date: date | None = None,
    ) -> tuple[RetrievalPlan | None, float]:
        """Attempt rule-based routing.

        Returns (plan, confidence). confidence >= 0.7 means high enough
        to skip the LLM router.

        sensitivity_tier: 1
        """
        q_lower = question.lower().strip()

        # Semantic/vague queries always go to LLM router
        if any(q_lower.startswith(s) for s in _SEMANTIC_STARTERS):
            return None, 0.0

        # Score each domain by keyword hits
        domain_scores: dict[str, int] = {}
        for domain, keywords in DOMAIN_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in q_lower)
            if hits > 0:
                domain_scores[domain] = hits

        if not domain_scores:
            return None, 0.0

        # Pick the top domain
        best_domain = max(
            domain_scores, key=domain_scores.get,  # type: ignore[arg-type]
        )
        best_score = domain_scores[best_domain]

        # Confidence: 1 hit = 0.6, 2 hits = 0.8, 3+ = 0.9.
        # A single strong keyword is enough for unambiguous domains
        # (calendar, health, messages) — the old threshold of 0.7
        # forced an LLM routing call for "meetings today" (1 hit).
        confidence = min(0.9, 0.4 + best_score * 0.2)

        if confidence < 0.6:
            return None, confidence

        # Build the plan
        today = (reference_date or date.today()).isoformat()
        queries = list(_DOMAIN_TO_TABLES.get(best_domain, []))

        # Inject temporal WHERE for calendar queries with "today"/"tomorrow"
        if best_domain == "calendar" and queries:
            base = queries[0]
            if "today" in q_lower or "hoje" in q_lower:
                queries[0] = DuckDBQuerySpec(
                    table=base.table,
                    columns=base.columns,
                    where=f"DATE(start_time) = DATE('{today}')",
                    order_by="start_time ASC",
                    limit=base.limit,
                )
            elif "tomorrow" in q_lower or "amanhã" in q_lower:
                queries[0] = DuckDBQuerySpec(
                    table=base.table,
                    columns=base.columns,
                    where=f"DATE(start_time) = DATE('{today}', '+1 day')",
                    order_by="start_time ASC",
                    limit=base.limit,
                )
            elif "this week" in q_lower or "next week" in q_lower:
                queries[0] = DuckDBQuerySpec(
                    table=base.table,
                    columns=base.columns,
                    where=(
                        f"DATE(start_time) >= DATE('{today}')"
                        f" AND DATE(start_time) <= "
                        f"DATE('{today}', '+7 days')"
                    ),
                    order_by="start_time ASC",
                    limit=20,
                )

        collections = _DOMAIN_TO_COLLECTIONS.get(best_domain, [])
        use_graph = best_domain in _GRAPH_DOMAINS

        return RetrievalPlan(
            duckdb_queries=queries,
            chromadb_collections=collections,
            use_graph=use_graph,
            reasoning=f"rule-based: domain={best_domain}, score={best_score}",
        ), confidence


# ------------------------------------------------------------------
# LLMRouter
# ------------------------------------------------------------------


class LLMRouter:
    """Routes a question to a :class:`RetrievalPlan`.

    Delegates the LLM step to :class:`QueryRouterAgent` (pydantic-ai)
    and enforces the table/column whitelist on the agent's output so
    the QueryEngine can render parameterised SQL safely. On agent
    failure (LLM down, validation rejects everything) returns a safe
    default plan that searches all ChromaDB collections.

    sensitivity_tier: 1 (only question text sent to LLM, no user data)
    """

    def __init__(self, llm_provider: LLMProvider | None = None) -> None:
        # ``llm_provider`` is accepted for backward-compat with callers
        # that still pass it (e.g. test fixtures). It is unused — the
        # underlying ``QueryRouterAgent`` resolves its own provider via
        # the agent registry / scheduler.
        self._provider = llm_provider

    def plan(
        self,
        question: str,
        reference_date: date | None = None,
    ) -> RetrievalPlan:
        """Generate a retrieval plan for the given question.

        Args:
            question: The user's natural language question.
            reference_date: Today's date for temporal queries.

        Returns:
            A validated RetrievalPlan.

        sensitivity_tier: 1
        """
        from src.agents.query_router.agent import QueryRouterAgent

        today = (reference_date or date.today()).isoformat()
        question_with_date = (
            f"Today's date: {today}\n\nUser question: {question}"
        )
        try:
            draft = QueryRouterAgent().plan(question_with_date)
        except Exception:  # noqa: BLE001
            logger.warning(
                "QueryRouterAgent failed, using default plan",
                exc_info=True,
            )
            return self._default_plan()
        if draft is None:
            return self._default_plan()
        return self._validate_plan(draft)

    @staticmethod
    def _validate_plan(draft: Any) -> RetrievalPlan:
        """Apply the table/column whitelist to a pydantic ``RetrievalPlan``.

        Returns a legacy dataclass :class:`RetrievalPlan` whose
        ``duckdb_queries`` only reference tables and columns in
        :data:`_ALLOWED_TABLES`, and whose ``chromadb_collections``
        match :data:`COLLECTION_NAMES`. Anything the agent returned
        outside the allowlist is silently dropped (SQL safety is a
        non-negotiable boundary).

        sensitivity_tier: 1
        """
        queries: list[DuckDBQuerySpec] = []
        for q in draft.duckdb_queries:
            spec = LLMRouter._validate_query_spec({
                "table": q.table,
                "columns": list(q.columns),
                "where": q.where,
                "order_by": q.order_by,
                "limit": q.limit,
            })
            if spec is not None:
                queries.append(spec)

        collections = [
            c for c in draft.chromadb_collections
            if c in COLLECTION_NAMES
        ]
        return RetrievalPlan(
            duckdb_queries=queries,
            chromadb_collections=collections,
            use_graph=bool(draft.use_graph),
            reasoning=str(draft.reasoning or ""),
        )

    @staticmethod
    def _validate_query_spec(
        raw: dict[str, Any],
    ) -> DuckDBQuerySpec | None:
        """Validate a single DuckDB query spec against the whitelist.

        Returns None if the table or columns are not allowed.

        sensitivity_tier: 1
        """
        table = raw.get("table", "")
        if table not in _ALLOWED_TABLES:
            logger.debug("Rejected unknown table: %s", table)
            return None

        allowed_cols = _ALLOWED_TABLES[table]
        raw_cols = raw.get("columns", [])
        if not isinstance(raw_cols, list) or not raw_cols:
            raw_cols = list(allowed_cols - {"metadata"})

        columns = [c for c in raw_cols if c in allowed_cols]
        if not columns:
            logger.debug(
                "No valid columns for table %s", table,
            )
            return None

        # Validate WHERE clause
        where = raw.get("where")
        if where is not None:
            where = str(where).strip()
            if not where:
                where = None
            elif _DANGEROUS_PATTERNS.search(where):
                logger.warning(
                    "Rejected dangerous WHERE clause: %s",
                    where,
                )
                where = None

        # Validate ORDER BY
        order_by = raw.get("order_by")
        if order_by is not None:
            order_by = str(order_by).strip()
            if not order_by:
                order_by = None
            elif _DANGEROUS_PATTERNS.search(order_by):
                order_by = None

        limit = min(int(raw.get("limit", 10)), 50)

        return DuckDBQuerySpec(
            table=table,
            columns=columns,
            where=where,
            order_by=order_by,
            limit=limit,
        )

    @staticmethod
    def _default_plan() -> RetrievalPlan:
        """Return a safe default plan when LLM routing fails.

        Searches all ChromaDB collections with no DuckDB or graph.

        sensitivity_tier: 1
        """
        return RetrievalPlan(
            duckdb_queries=[],
            chromadb_collections=list(COLLECTION_NAMES),
            use_graph=False,
            reasoning="default: LLM routing unavailable",
        )


# ------------------------------------------------------------------
# Entity extraction helpers (kept from original)
# ------------------------------------------------------------------

_CAPITALIZED_WORD_RE = re.compile(
    r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b",
)

_SKIP_WORDS: frozenset[str] = frozenset(
    {
        "the", "a", "an", "i", "my", "your", "he", "she", "it",
        "we", "they", "this", "that", "is", "are", "was", "were",
        "do", "does", "did", "have", "has", "had", "will", "would",
        "can", "could", "should", "may", "might", "must", "shall",
        "being", "been", "be", "not", "no", "yes", "but", "and",
        "or", "if", "then", "so", "what", "who", "how", "when",
        "where", "why", "which", "there", "here", "all", "each",
        "every", "some", "any", "many", "much", "more", "most",
        "other", "another", "new", "old", "great", "good", "bad",
        "first", "last", "next", "only", "just", "also", "very",
        "tell", "about", "source", "tags", "location", "attendees",
        "quick", "reminder", "following", "please", "hi", "hey",
        "monday", "tuesday", "wednesday", "thursday", "friday",
        "saturday", "sunday", "january", "february", "march",
        "april", "june", "july", "august", "september", "october",
        "november", "december",
    }
)


# ------------------------------------------------------------------
# Pure functions
# ------------------------------------------------------------------


def extract_entities(
    texts: list[str],
    name_to_node_id: dict[str, str],
) -> list[tuple[str, str]]:
    """Extract entity names and resolve to Kuzu node IDs.

    Args:
        texts: Text strings to scan for entities.
        name_to_node_id: Lowercased name variants -> node IDs.

    Returns:
        List of (entity_name, node_id) tuples.

    sensitivity_tier: 2
    """
    found: dict[str, str] = {}
    claimed: set[str] = set()
    combined = " ".join(texts).lower()

    # Strategy 1: match known names as whole words, longest variant
    # first. Both properties matter for picking the right person:
    # a substring test lets "ana" match inside "Mariana", and without
    # longest-first the bare first name can claim a mention that the
    # full name identifies unambiguously.
    for name_variant in find_names(list(name_to_node_id), combined):
        node_id = name_to_node_id[name_variant]
        if node_id in claimed:
            continue
        found[name_variant] = node_id
        claimed.add(node_id)

    # Strategy 2: capitalized words from original texts
    for text in texts:
        for match in _CAPITALIZED_WORD_RE.finditer(text):
            word = match.group()
            lower_word = word.lower()
            if lower_word in _SKIP_WORDS:
                continue
            if lower_word in name_to_node_id:
                nid = name_to_node_id[lower_word]
                if nid not in claimed:
                    found[lower_word] = nid
                    claimed.add(nid)

    return list(found.items())


def normalize_vector_distance(distance: float) -> float:
    """Convert ChromaDB L2 distance to 0-1 relevance score.

    sensitivity_tier: N/A
    """
    return max(0.0, min(1.0, 1.0 - distance / 2.0))


def merge_and_deduplicate(
    items: list[ContextItem],
    max_items: int,
) -> list[ContextItem]:
    """Deduplicate by source_id, sort by relevance, cap.

    sensitivity_tier: N/A
    """
    best: dict[str, ContextItem] = {}
    for item in items:
        existing = best.get(item.source_id)
        if existing is None or item.relevance > existing.relevance:
            best[item.source_id] = item
    sorted_items = sorted(
        best.values(),
        key=lambda x: x.relevance,
        reverse=True,
    )
    return sorted_items[:max_items]


def build_safe_query(spec: DuckDBQuerySpec) -> str | None:
    """Build a safe SQL query from a validated DuckDBQuerySpec.

    Returns None if the table or columns are not whitelisted.

    sensitivity_tier: 1
    """
    if spec.table not in _ALLOWED_TABLES:
        return None

    allowed = _ALLOWED_TABLES[spec.table]
    safe_columns = [c for c in spec.columns if c in allowed]
    if not safe_columns:
        return None

    cols = ", ".join(safe_columns)
    sql = f"SELECT {cols} FROM {spec.table}"  # noqa: S608

    if spec.where:
        if _DANGEROUS_PATTERNS.search(spec.where):
            return None
        sql += f" WHERE {spec.where}"

    if spec.order_by:
        if _DANGEROUS_PATTERNS.search(spec.order_by):
            pass  # Skip order_by, still run the query
        else:
            sql += f" ORDER BY {spec.order_by}"

    sql += f" LIMIT {min(spec.limit, 50)}"
    return sql


# ------------------------------------------------------------------
# QueryEngine
# ------------------------------------------------------------------


class QueryEngine:
    """Hybrid GraphRAG query engine with LLM-driven routing.

    Uses an LLM router to decide which databases to query, then
    combines vector, graph, and SQL retrieval into a unified
    QueryContext.

    sensitivity_tier: 3
    """

    def __init__(
        self,
        duckdb: DatabaseEngine,
        kuzu: GraphEngine | None = None,
        chromadb: VectorEngine | None = None,
        llm_provider: LLMProvider | None = None,
    ) -> None:
        """Initialize the query engine.

        Args:
            duckdb: DuckDB engine for structured queries.
            kuzu: Kuzu engine for graph traversal (optional).
            chromadb: ChromaDB engine for vector search (optional).
            llm_provider: LLM provider for routing. If None, creates
                one from user settings.
        """
        self._duck = duckdb
        self._kuzu = kuzu
        self._chroma = chromadb

        self._rule_router = RuleBasedRouter()
        if llm_provider is not None:
            self._llm_router = LLMRouter(llm_provider)
        else:
            from src.models.llm_provider import (
                create_provider_from_settings,
            )
            self._llm_router = LLMRouter(create_provider_from_settings())

        self._name_to_node_id = self._build_name_index()
        self._known_names = frozenset(self._name_to_node_id.keys())

    @timed()
    def query(
        self,
        question: str,
        max_context_items: int = 10,
        max_sensitivity_tier: int = 2,
        reference_date: date | None = None,
    ) -> QueryContext:
        """Execute a hybrid retrieval query using LLM routing.

        Pipeline:
          1. LLM router generates a RetrievalPlan
          2. Execute DuckDB queries from the plan
          3. Vector search on planned ChromaDB collections
          4. Entity extraction from results
          5. Optional graph traversal
          6. Assemble and return context

        Args:
            question: Natural language question.
            max_context_items: Maximum context items to return.
            max_sensitivity_tier: Maximum sensitivity tier (1-3).
            reference_date: Date for "today" queries.

        Returns:
            QueryContext with results from queried databases.

        sensitivity_tier: 3
        """
        effective_date = reference_date or date.today()
        timing: dict[str, float] = {}
        total_start = _now_ms()

        # Step 1: Try rule-based routing first, fall back to LLM
        t0 = _now_ms()
        rule_plan, confidence = self._rule_router.plan(
            question, effective_date,
        )
        if rule_plan is not None and confidence >= 0.7:
            plan = rule_plan
            logger.debug(
                "Rule-based routing (confidence=%.2f): %s",
                confidence, plan.reasoning,
            )
        else:
            plan = self._llm_router.plan(question, effective_date)
            logger.debug(
                "LLM routing (rule confidence=%.2f): %s",
                confidence, plan.reasoning,
            )
        timing["routing"] = _now_ms() - t0

        # Steps 2+3: DuckDB + vector search in parallel (independent
        # data stores, no shared mutable state).
        t0 = _now_ms()
        want_vector = (
            self._chroma is not None and plan.chromadb_collections
        )

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=2) as pool:
            duck_future = pool.submit(
                self._execute_plan_queries,
                plan, max_sensitivity_tier,
            )
            if want_vector:
                vec_future = pool.submit(
                    self._vector_search,
                    question, max_sensitivity_tier,
                    plan.chromadb_collections,
                )
            else:
                vec_future = None

            structured_data = duck_future.result()
            vector_results: list[dict[str, Any]] = (
                vec_future.result() if vec_future is not None else []
            )

        timing["structured_queries"] = _now_ms() - t0
        timing["vector_search"] = 0.0

        # Steps 4+5: Entity extraction + graph — only when graph is
        # requested. Skipping entity extraction when use_graph=false
        # saves 2 DB queries that were previously always executed.
        t0 = _now_ms()
        entities: list[tuple[str, str]] = []
        graph_context: list[dict[str, Any]] = []
        graph_degraded = False
        if plan.use_graph and self._kuzu is not None:
            vector_texts = [r["document"] for r in vector_results]
            entities = extract_entities(
                [question] + vector_texts,
                self._name_to_node_id,
            )
            if entities:
                graph_context, graph_degraded = self._graph_traverse(
                    entities, max_sensitivity_tier,
                )
        timing["entity_extraction"] = _now_ms() - t0
        timing["graph_traversal"] = 0.0

        # Step 6: Assemble
        t0 = _now_ms()
        context = self._assemble_context(
            question=question,
            vector_results=vector_results,
            graph_context=graph_context,
            structured_data=structured_data,
            entities=entities,
            collections_searched=plan.chromadb_collections,
            max_context_items=max_context_items,
            max_sensitivity_tier=max_sensitivity_tier,
            effective_date=effective_date,
            timing=timing,
            routing_reasoning=plan.reasoning,
            graph_degraded=graph_degraded,
        )
        timing["assembly"] = _now_ms() - t0
        timing["total"] = _now_ms() - total_start
        context.metadata["timing_ms"] = timing
        return context

    # ----------------------------------------------------------
    # Plan execution
    # ----------------------------------------------------------

    def _execute_plan_queries(
        self,
        plan: RetrievalPlan,
        max_tier: int,
    ) -> list[dict[str, Any]]:
        """Execute DuckDB queries from the retrieval plan.

        Adds sensitivity_tier filter and source_table tag to results.
        For calendar event tables we additionally inject
        ``event_origin = 'personal'`` so retrieval never returns
        team-awareness or subscribed-calendar entries — those events
        belong to the dashboard's awareness surfaces, not to Brain's
        "what do I have" answers.

        sensitivity_tier: 3
        """
        results: list[dict[str, Any]] = []
        for spec in plan.duckdb_queries:
            sql = build_safe_query(spec)
            if sql is None:
                continue

            existing_where = spec.where or ""
            extra_clauses: list[str] = []
            if "sensitivity_tier" not in existing_where:
                extra_clauses.append(f"sensitivity_tier <= {max_tier}")
            if (
                spec.table in _CALENDAR_TABLES
                and "event_origin" not in existing_where
            ):
                extra_clauses.append(
                    "COALESCE(event_origin, 'personal') = 'personal'"
                )

            if extra_clauses:
                combined = " AND ".join(extra_clauses)
                if spec.where:
                    sql = sql.replace(
                        f"WHERE {spec.where}",
                        f"WHERE ({spec.where}) AND {combined}",
                    )
                else:
                    # Insert WHERE before ORDER BY or LIMIT
                    parts = sql.split(" ORDER BY ")
                    if len(parts) == 2:
                        sql = (
                            f"{parts[0]} "
                            f"WHERE {combined} "
                            f"ORDER BY {parts[1]}"
                        )
                    else:
                        parts = sql.split(" LIMIT ")
                        sql = (
                            f"{parts[0]} "
                            f"WHERE {combined} "
                            f"LIMIT {parts[1]}"
                        )

            try:
                rows = self._duck.query(sql)
                for r in rows:
                    r["source_table"] = spec.table
                results.extend(rows)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Plan query failed for %s: %s",
                    spec.table,
                    sql,
                )

        return results

    # ----------------------------------------------------------
    # Context assembly
    # ----------------------------------------------------------

    def _assemble_context(
        self,
        *,
        question: str,
        vector_results: list[dict[str, Any]],
        graph_context: list[dict[str, Any]],
        structured_data: list[dict[str, Any]],
        entities: list[tuple[str, str]],
        collections_searched: list[str],
        max_context_items: int,
        max_sensitivity_tier: int,
        effective_date: date,
        timing: dict[str, float],
        routing_reasoning: str = "",
        graph_degraded: bool = False,
    ) -> QueryContext:
        """Assemble final QueryContext from retrieval results.

        sensitivity_tier: 3
        """
        # Load topic contacts for relevance boosting
        topic_contacts = load_topic_contacts(self._duck)

        all_items = (
            self._vector_to_items(vector_results)
            + self._graph_to_items(graph_context)
            + self._structured_to_items(
                structured_data, topic_contacts,
            )
        )
        merged = merge_and_deduplicate(
            all_items,
            max_context_items,
        )
        merged_ids = {
            (it.source_id, it.source_type) for it in merged
        }

        final_vector = _filter_unique(
            vector_results,
            lambda r: _extract_base_id(r.get("id", "")),
            "vector",
            merged_ids,
        )
        final_graph = _filter_unique(
            graph_context,
            _graph_item_id,
            "graph",
            merged_ids,
        )
        final_structured = _filter_unique(
            structured_data,
            lambda r: r.get("id", ""),
            "structured",
            merged_ids,
        )

        # Enforce hard cap on total items
        # Structured data gets priority (precise SQL results > fuzzy vector)
        remaining = max_context_items
        final_structured = final_structured[:remaining]
        remaining -= len(final_structured)
        final_graph = final_graph[:remaining]
        remaining -= len(final_graph)
        final_vector = final_vector[:remaining]

        tiers: set[int] = set()
        for item in merged:
            tiers.add(item.sensitivity_tier)

        sources_used: list[str] = []
        if final_vector:
            sources_used.append("chromadb")
        if final_graph:
            sources_used.append("kuzu")
        if final_structured:
            sources_used.append("duckdb")

        metadata: dict[str, Any] = {
            "timing_ms": timing,
            "sources_used": sources_used,
            "sensitivity_tiers_encountered": tiers,
            "collections_searched": collections_searched,
            "entities_extracted": [n for n, _ in entities],
            "routing_reasoning": routing_reasoning,
            "max_sensitivity_tier": max_sensitivity_tier,
            "reference_date": effective_date.isoformat(),
            # True when a Kuzu traversal failed (e.g. lock contention
            # during a pipeline reindex) — the answer was produced
            # WITHOUT some or all graph context.  Consumers can surface
            # a "graph context unavailable" hint instead of presenting
            # a silently degraded answer as complete.
            "graph_degraded": graph_degraded,
        }

        return QueryContext(
            question=question,
            vector_results=final_vector,
            graph_context=final_graph,
            structured_data=final_structured,
            metadata=metadata,
        )

    # ----------------------------------------------------------
    # Vector search
    # ----------------------------------------------------------

    @timed()
    def _vector_search(
        self,
        question: str,
        max_tier: int,
        collections: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Search the chunk corpus via the Phase 4 hybrid pipeline.

        Vector + BM25 → RRF → record-dedup → top-k. Falls back to
        pure cosine if the BM25 FTS5 mirror is unavailable (e.g. on
        a fresh install before the first ``Indexer.full_reindex``
        populates it).

        Returns the same shape the rest of QueryEngine expects:
        ``[{id, document, metadata, distance, collection}, ...]``,
        sorted with the most-relevant first.

        sensitivity_tier: 3
        """
        if self._chroma is None:
            return []

        try:
            from src.core.retrieval.pipeline import HybridSearch

            pipeline = HybridSearch(
                chroma=self._chroma, sqlite_db=self._duck,
            )
            hits = pipeline.search(
                question,
                top_k=10,
                max_tier=max_tier,
                collections=collections,
            )
            if hits:
                return [
                    {
                        "id": h.id,
                        "document": h.document,
                        "metadata": h.metadata,
                        # Fused score is "higher is better"; the rest
                        # of QueryEngine expects "lower distance is
                        # better", so we map score → 1-score for the
                        # sort key.
                        "distance": 1.0 - h.score,
                        "collection": h.metadata.get("collection", ""),
                        "origin": h.origin,
                    }
                    for h in hits
                ]
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "hybrid pipeline unavailable, falling back to cosine: %s",
                exc,
            )

        # Fallback: legacy pure-cosine search (kept verbatim).
        where_filter: dict[str, Any] = {
            "sensitivity_tier": {"$lte": max_tier},
        }
        all_results: list[dict[str, Any]] = []
        target_collections = collections or list(COLLECTION_NAMES)
        for name in target_collections:
            try:
                results = self._chroma.search(
                    name,
                    question,
                    n_results=3,
                    where=where_filter,
                )
                for r in results:
                    r["collection"] = name
                all_results.extend(results)
            except Exception:  # noqa: BLE001
                logger.warning("Vector search failed for '%s'", name)
        all_results.sort(
            key=lambda r: r.get("distance", float("inf")),
        )
        return all_results

    # ----------------------------------------------------------
    # Name index
    # ----------------------------------------------------------

    def _build_name_index(self) -> dict[str, str]:
        """Build name variants -> Kuzu Person node ID mapping.

        Variants are accumulated as ``variant -> {owner ids}`` and only
        the single-owner ones survive (see
        :func:`~src.core.name_matching.build_unique_index`). Two
        contacts sharing a first name used to race for that token via
        ``setdefault``, permanently awarding e.g. ``"ana"`` to whichever
        was indexed first and mis-resolving every later mention of the
        other Ana.

        sensitivity_tier: 2
        """
        variants: dict[str, set[str]] = defaultdict(set)
        # Full names are recorded separately: a full name identifies a
        # person even when their first name is shared, so it must not be
        # dropped just because a bare token collides.
        full_names: dict[str, set[str]] = defaultdict(set)

        def _record(name: str, node_id: str) -> None:
            lower_name = name.lower()
            full_names[lower_name].add(node_id)
            for part in lower_name.split():
                if len(part) > 2:
                    variants[part].add(node_id)

        if self._kuzu is not None:
            try:
                persons = self._kuzu.query(
                    "MATCH (p:Person) "
                    "RETURN p.id AS id, p.name AS name",
                )
                for person in persons:
                    name = person.get("name", "")
                    if name:
                        _record(name, person["id"])
            except Exception:  # noqa: BLE001
                logger.warning("Failed to load Kuzu Person nodes")

        try:
            contacts = self._duck.query(
                "SELECT name, email FROM raw_contacts",
            )
            for contact in contacts:
                name = contact.get("name", "")
                if not name:
                    continue
                # Resolve the contact onto an existing Person node by
                # full name, else by an unambiguous name part.
                owners = full_names.get(name.lower(), set())
                if not owners:
                    for part in name.lower().split():
                        part_owners = variants.get(part, set())
                        if len(part_owners) == 1:
                            owners = part_owners
                            break
                if len(owners) == 1:
                    _record(name, next(iter(owners)))
        except Exception:  # noqa: BLE001
            logger.warning("Failed to load DuckDB contacts")

        index = build_unique_index(full_names)
        index.update(build_unique_index(variants))
        return index

    # ----------------------------------------------------------
    # Graph traversal
    # ----------------------------------------------------------

    @timed()
    def _graph_traverse(
        self,
        entities: list[tuple[str, str]],
        max_tier: int,
    ) -> tuple[list[dict[str, Any]], bool]:
        """1-hop and 2-hop Kuzu traversals for entities.

        Returns ``(results, degraded)`` — *degraded* is True when any
        traversal failed (typically Kuzu lock contention during a
        pipeline reindex), so callers can surface that graph context is
        missing instead of silently answering without it.

        sensitivity_tier: 3
        """
        if self._kuzu is None:
            return [], False

        results: list[dict[str, Any]] = []
        degraded = False

        for entity_name, node_id in entities:
            if not node_id:
                continue
            params = {"nid": node_id, "max_tier": max_tier}

            # 1-hop outgoing
            try:
                rows = self._kuzu.query(
                    "MATCH (a:Person {id: $nid})-[r]->(b) "
                    "WHERE r.sensitivity_tier <= $max_tier "
                    "RETURN a.id AS from_id, "
                    "a.name AS from_name, "
                    "r.weight AS weight, "
                    "r.sensitivity_tier AS rel_tier, "
                    "b.id AS to_id, "
                    "COALESCE(b.name, b.title, b.id) "
                    "AS to_name, "
                    "b.sensitivity_tier AS to_tier",
                    params,
                )
                for row in rows:
                    row["hop"] = 1
                    row["source_entity"] = entity_name
                results.extend(rows)
            except Exception:  # noqa: BLE001
                degraded = True
                logger.warning(
                    "1-hop out failed for %s", node_id,
                )

            # 1-hop incoming
            try:
                rows = self._kuzu.query(
                    "MATCH (b)-[r]->(a:Person {id: $nid}) "
                    "WHERE r.sensitivity_tier <= $max_tier "
                    "RETURN b.id AS from_id, "
                    "COALESCE(b.name, b.title, b.id) "
                    "AS from_name, "
                    "r.weight AS weight, "
                    "r.sensitivity_tier AS rel_tier, "
                    "a.id AS to_id, "
                    "a.name AS to_name",
                    params,
                )
                for row in rows:
                    row["hop"] = 1
                    row["direction"] = "incoming"
                    row["source_entity"] = entity_name
                results.extend(rows)
            except Exception:  # noqa: BLE001
                degraded = True
                logger.warning(
                    "1-hop in failed for %s", node_id,
                )

            # 2-hop: Person -> mid -> end
            try:
                rows = self._kuzu.query(
                    "MATCH (a:Person {id: $nid})"
                    "-[r1]->(mid)"
                    "-[r2]->(end) "
                    "WHERE r1.sensitivity_tier <= $max_tier "
                    "AND r2.sensitivity_tier <= $max_tier "
                    "RETURN a.id AS from_id, "
                    "a.name AS from_name, "
                    "mid.id AS mid_id, "
                    "COALESCE(mid.name, mid.title, mid.id) "
                    "AS mid_name, "
                    "end.id AS end_id, "
                    "COALESCE(end.name, end.title, end.id) "
                    "AS end_name, "
                    "r1.weight AS weight1, "
                    "r2.weight AS weight2",
                    params,
                )
                for row in rows:
                    row["hop"] = 2
                    row["source_entity"] = entity_name
                results.extend(rows)
            except Exception:  # noqa: BLE001
                degraded = True
                logger.warning(
                    "2-hop failed for %s", node_id,
                )

        return results, degraded

    # ----------------------------------------------------------
    # Conversion helpers
    # ----------------------------------------------------------

    def _vector_to_items(
        self,
        results: list[dict[str, Any]],
    ) -> list[ContextItem]:
        """Convert vector results to ContextItems.

        sensitivity_tier: 3
        """
        items: list[ContextItem] = []
        for r in results:
            base_id = _extract_base_id(r.get("id", ""))
            tier = r.get("metadata", {}).get(
                "sensitivity_tier",
                1,
            )
            items.append(
                ContextItem(
                    source_id=base_id,
                    source_type="vector",
                    source_db="chromadb",
                    content=r.get("document", ""),
                    relevance=normalize_vector_distance(
                        r.get("distance", 2.0),
                    ),
                    sensitivity_tier=tier,
                    metadata=r.get("metadata", {}),
                ),
            )
        return items

    def _graph_to_items(
        self,
        results: list[dict[str, Any]],
    ) -> list[ContextItem]:
        """Convert graph results to ContextItems.

        sensitivity_tier: 3
        """
        items: list[ContextItem] = []
        for r in results:
            hop = r.get("hop", 1)
            source_id = _graph_item_id(r)

            if hop == 1:
                content = (
                    f"{r.get('from_name', '?')} "
                    f"-> {r.get('to_name', '?')}"
                )
                weight = r.get("weight", 0.5)
            else:
                content = (
                    f"{r.get('from_name', '?')} "
                    f"-> {r.get('mid_name', '?')} "
                    f"-> {r.get('end_name', '?')}"
                )
                w1 = r.get("weight1", 0.5)
                w2 = r.get("weight2", 0.5)
                weight = (w1 + w2) / 2

            tier = r.get(
                "rel_tier",
                r.get("to_tier", 2),
            )
            items.append(
                ContextItem(
                    source_id=source_id,
                    source_type="graph",
                    source_db="kuzu",
                    content=content,
                    relevance=weight * 0.8,
                    sensitivity_tier=tier,
                    metadata=r,
                ),
            )
        return items

    def _structured_to_items(
        self,
        results: list[dict[str, Any]],
        topic_contacts: dict[str, dict] | None = None,
    ) -> list[ContextItem]:
        """Convert DuckDB results to ContextItems.

        Applies topic-based relevance boosting when a result's
        sender/contact matches a high-importance topic contact.

        sensitivity_tier: 3
        """
        items: list[ContextItem] = []
        tc = topic_contacts or {}
        for r in results:
            source_id = r.get("id", "")
            table = r.get("source_table", "unknown")
            content = _format_structured(r, table)
            relevance = _structured_relevance(table)

            # Topic-boost: increase relevance for results
            # from important topic contacts
            relevance = _topic_boost_relevance(
                relevance, r, tc,
            )

            items.append(
                ContextItem(
                    source_id=source_id,
                    source_type="structured",
                    source_db="duckdb",
                    content=content,
                    relevance=relevance,
                    sensitivity_tier=r.get(
                        "sensitivity_tier",
                        2,
                    ),
                    metadata={
                        "source_table": table,
                        **r,
                    },
                ),
            )
        return items


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _filter_unique(
    items: list[dict[str, Any]],
    id_fn: Any,
    source_type: str,
    merged_ids: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Return items whose id is in merged_ids, deduped."""
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for item in items:
        item_id = id_fn(item)
        if (item_id, source_type) in merged_ids and item_id not in seen:
            seen.add(item_id)
            result.append(item)
    return result


_STRUCTURED_RELEVANCE: dict[str, float] = {
    "raw_calendar_events": 0.85,
    "raw_health_metrics": 0.80,
    "raw_contacts": 0.80,
    "raw_messages": 0.70,
    "raw_notes": 0.65,
    "raw_emails": 0.65,
    "raw_reminders": 0.70,
    "raw_files": 0.50,
    "raw_workouts": 0.75,
    "raw_listening_history": 0.60,
    "raw_voice_memos": 0.55,
}


def _topic_boost_relevance(
    base_relevance: float,
    row: dict[str, Any],
    topic_contacts: dict[str, dict],
) -> float:
    """Boost relevance when a row involves a topic contact.

    +0.10 for contacts with importance >= 7 (critical).
    +0.05 for contacts with importance >= 5 (active).
    Capped at 1.0.

    sensitivity_tier: 1
    """
    if not topic_contacts:
        return base_relevance

    # Check common sender/contact fields
    fields = (
        "sender_name", "sender", "from_address",
        "attendees", "contact_name", "name",
    )
    for fld in fields:
        val = row.get(fld)
        if not val:
            continue
        val_lower = str(val).lower()
        for tc_name, tc_data in topic_contacts.items():
            if tc_name in val_lower or val_lower in tc_name:
                imp = tc_data.get("importance", 0)
                if imp >= 7:
                    return min(1.0, base_relevance + 0.10)
                if imp >= 5:
                    return min(1.0, base_relevance + 0.05)
    return base_relevance


def _structured_relevance(table: str) -> float:
    """Return relevance score for a structured result by table.

    LLM-routed structured data scores higher since the LLM decided
    it was relevant to the query.

    sensitivity_tier: N/A
    """
    return _STRUCTURED_RELEVANCE.get(table, 0.5)


def _now_ms() -> float:
    """Current time in milliseconds."""
    return datetime.now().timestamp() * 1000


def _extract_base_id(doc_id: str) -> str:
    """Strip chunk suffix from a document ID."""
    if "-chunk-" in doc_id:
        return doc_id.split("-chunk-")[0]
    return doc_id


def _graph_item_id(row: dict[str, Any]) -> str:
    """Generate a stable ID for a graph result row."""
    hop = row.get("hop", 1)
    if hop == 1:
        return (
            f"graph-{row.get('from_id', '')}"
            f"-{row.get('to_id', '')}"
        )
    return (
        f"graph-{row.get('from_id', '')}"
        f"-{row.get('mid_id', '')}"
        f"-{row.get('end_id', '')}"
    )


def _format_structured(
    row: dict[str, Any],
    table: str,
) -> str:
    """Format a structured result as readable text.

    sensitivity_tier: inherits from caller
    """
    if table == "raw_calendar_events":
        return (
            f"Calendar: {row.get('title', '')} "
            f"at {row.get('location', '?')} "
            f"({row.get('start_time', '')})"
        )
    if table == "raw_contacts":
        return (
            f"Contact: {row.get('name', '')} "
            f"({row.get('relationship', '')}) - "
            f"{row.get('notes', '')}"
        )
    if table == "raw_messages":
        return (
            f"Message from {row.get('sender', '?')}: "
            f"{str(row.get('content', ''))[:200]}"
        )
    if table == "raw_health_metrics":
        return (
            f"Health: {row.get('metric_type', '')} = "
            f"{row.get('value', '')} {row.get('unit', '')}"
        )
    if table == "raw_notes":
        return (
            f"Note: {row.get('title', '')} - "
            f"{str(row.get('content', ''))[:200]}"
        )
    if table == "raw_emails":
        return (
            f"Email: {row.get('subject', '')} "
            f"from {row.get('from_address', '?')}"
        )
    if table == "raw_reminders":
        status = "done" if row.get("completed") else "pending"
        return (
            f"Reminder: {row.get('title', '')} "
            f"({status}, due: {row.get('due_date', '?')})"
        )
    if table == "raw_listening_history":
        return (
            f"Music: {row.get('track_name', '')} "
            f"by {row.get('artist', '?')}"
        )
    if table == "raw_workouts":
        return (
            f"Workout: {row.get('workout_type', '')} "
            f"({row.get('duration_min', '?')} min, "
            f"{row.get('calories', '?')} cal)"
        )
    if table == "raw_files":
        return f"File: {row.get('filename', row.get('filepath', ''))}"
    if table == "raw_voice_memos":
        dur = row.get('duration_seconds', '?')
        return f"Voice memo: {row.get('title', '')} ({dur}s)"
    return str(row)
