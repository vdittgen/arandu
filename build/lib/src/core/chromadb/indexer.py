"""DuckDB-to-ChromaDB indexing pipeline.

Reads structured data from DuckDB raw tables, composes embeddable text,
chunks long content, and upserts into ChromaDB collections with full
metadata.

Phase 3 changes (vs. earlier indexer):

* ``compose_*_text`` functions prepend structured headers so high-
  signal fields (sender, recipient, name, relationship, date) land
  in the embedding rather than only in metadata. Embedded text now
  drives semantic ranking; metadata stays for query-time filtering.
* ``chunk_text`` uses ``tiktoken`` for token-accurate splitting and
  applies a configurable token-overlap between chunks so long content
  doesn't lose context at boundaries.
* Per-chunk metadata gains ``layer`` (raw vs mart) and
  ``recency_bucket`` (today / this_week / this_month / older). Both
  are filterable from the read path in Phase 4.
* New mart indexers (`_index_mart_today`, `_index_mart_work`, …)
  embed the human-readable summary rows alongside raw records, in
  the same domain collections but flagged with ``layer="mart"``.

sensitivity_tier: 3 (processes all raw user data including health/financial)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from src.core.chromadb.engine import VectorEngine
from src.core.profiler import timed
from src.core.sqlite.engine import DatabaseEngine

logger = logging.getLogger(__name__)

MAX_TOKENS_PER_CHUNK = 500
APPROX_CHARS_PER_TOKEN = 4
DEFAULT_CHUNK_OVERLAP = 50  # tokens

# Lazy tiktoken encoding — loaded once on first use.
_ENCODING: Any | None = None


def _get_encoding() -> Any:
    """Return the cached ``tiktoken`` encoder (``cl100k_base``).

    cl100k_base is OpenAI's GPT-3.5/4/embedding-3 tokenizer and
    serves as a good approximation across embedding providers — it
    won't be byte-perfect for bge-m3 but stays well within the model
    context limit and keeps chunking deterministic across swaps.

    sensitivity_tier: N/A
    """
    global _ENCODING
    if _ENCODING is None:
        import tiktoken

        _ENCODING = tiktoken.get_encoding("cl100k_base")
    return _ENCODING


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------


@dataclass(frozen=True)
class IndexDocument:
    """A single document prepared for ChromaDB insertion.

    sensitivity_tier: inherits from source record
    """

    id: str
    text: str
    collection: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------
# Pure text composition functions
# ------------------------------------------------------------------


def _structured(fields: list[tuple[str, Any]]) -> str:
    """Render ``[(label, value), ...]`` as ``"Label: value | Label: value"``.

    Skips fields whose value is falsy (empty string, None, 0 for
    optional counters). Keeps the leading label visible in the
    embedded text so semantic ranking can weight on the field
    identity, not only the value.

    sensitivity_tier: N/A
    """
    parts = [f"{label}: {value}" for label, value in fields if value]
    return " | ".join(parts)


def compose_message_text(row: dict[str, Any]) -> str:
    """Compose embeddable text from a raw_messages row.

    Output shape::

        From: <sender> | To: <recipient> | Channel: <source> |
        When: <timestamp>

        <content>

    The structured header puts proper nouns ("From: Mom") in front
    of the body so semantic ranking can lean on participant identity
    even when the body is short or vague. Content is appended after
    a blank line so the embedder treats it as a separate semantic
    block.

    sensitivity_tier: 3
    """
    header = _structured(
        [
            ("From", row.get("sender")),
            ("To", row.get("recipient")),
            ("Channel", row.get("source")),
            ("When", row.get("timestamp")),
        ],
    )
    content = str(row.get("content") or "").strip()
    return f"{header}\n\n{content}" if header and content else (
        content or header
    )


def compose_calendar_text(row: dict[str, Any]) -> str:
    """Compose embeddable text from a raw_calendar_events row.

    Output shape::

        Event: <title> | When: <start> - <end> |
        Location: <location> | Attendees: <attendees>

        <description>

    The event title leads (most queries are title-based), followed
    by participants and time. Description is appended as its own
    block when present so long-form notes don't drown out the title
    in token-weighted retrieval.

    sensitivity_tier: 2
    """
    start = row.get("start_time", "")
    end = row.get("end_time", "")
    when = f"{start} - {end}" if start and end else (start or end)
    header = _structured(
        [
            ("Event", row.get("title")),
            ("When", when),
            ("Location", row.get("location")),
            ("Attendees", row.get("attendees")),
        ],
    )
    description = str(row.get("description") or "").strip()
    return (
        f"{header}\n\n{description}" if header and description else
        (description or header)
    )


def compose_note_text(row: dict[str, Any]) -> str:
    """Compose embeddable text from a raw_notes row.

    Output shape::

        Note: <title> | Tags: <tags> | Source: <source>

        <content>

    sensitivity_tier: 2
    """
    header = _structured(
        [
            ("Note", row.get("title")),
            ("Tags", row.get("tags")),
            ("Source", row.get("source")),
        ],
    )
    content = str(row.get("content") or "").strip()
    return (
        f"{header}\n\n{content}" if header and content else
        (content or header)
    )


def compose_health_metric_text(row: dict[str, Any]) -> str:
    """Compose embeddable text from a raw_health_metrics row.

    Output shape::

        Metric: <metric_type> | Value: <value> <unit> |
        Recorded: <recorded_at> | Source: <source>

    sensitivity_tier: 3
    """
    value = row.get("value", "")
    unit = row.get("unit", "")
    value_with_unit = f"{value} {unit}".strip() if value or unit else ""
    return _structured(
        [
            ("Metric", row.get("metric_type")),
            ("Value", value_with_unit),
            ("Recorded", row.get("recorded_at")),
            ("Source", row.get("source")),
        ],
    )


def compose_contact_text(row: dict[str, Any]) -> str:
    """Compose embeddable text from a raw_contacts row.

    Output shape::

        Contact: <name> | Relationship: <relationship> |
        Email: <email> | Last contact: <last_contact>

        <notes>

    The name field is prefixed with ``Contact:`` so queries like
    "Who is X?" or "Tell me about X" match on a labelled identity
    rather than a bare name string that competes with every other
    contact. This is the single biggest text-side improvement for
    the contact-heavy 0% recall observed pre-Phase-3.

    sensitivity_tier: 2
    """
    header = _structured(
        [
            ("Contact", row.get("name")),
            ("Relationship", row.get("relationship")),
            ("Email", row.get("email")),
            ("Last contact", row.get("last_contact")),
        ],
    )
    notes = str(row.get("notes") or "").strip()
    return (
        f"{header}\n\n{notes}" if header and notes else (notes or header)
    )


def compose_mart_text(row: dict[str, Any]) -> str:
    """Compose embeddable text from a generic ``mart_*`` row.

    Marts share a common shape across the user-facing summary tables
    (``mart_today`` / ``mart_work`` / ``mart_personal`` / …): ``title``
    + ``detail`` + a timestamp-ish column. We compose them uniformly
    so the indexer doesn't need a bespoke composer per mart.

    sensitivity_tier: 2
    """
    when = (
        row.get("occurred_at")
        or row.get("event_date")
        or row.get("_loaded_at")
    )
    header = _structured(
        [
            ("Summary", row.get("title")),
            ("Type", row.get("item_type") or row.get("category")),
            ("When", when),
            ("Contact", row.get("contact_name")),
            ("Topic", row.get("topic")),
        ],
    )
    detail = str(row.get("detail") or "").strip()
    return (
        f"{header}\n\n{detail}" if header and detail else (detail or header)
    )


# ------------------------------------------------------------------
# Chunking
# ------------------------------------------------------------------


def chunk_text(
    text: str,
    max_tokens: int = MAX_TOKENS_PER_CHUNK,
    overlap_tokens: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split text into token-bounded chunks with optional overlap.

    Uses ``tiktoken.cl100k_base`` for exact token counting (vs the
    pre-Phase-3 char/4 heuristic, which under- or over-counted for
    multibyte and CJK text by 30-50%). Falls back to the
    character-based splitter if the encoder fails to import — keeps
    the indexer functional on stripped-down environments.

    Sliding-window overlap (default 50 tokens) preserves context
    across chunk boundaries: a contact note that splits mid-sentence
    still has the start of the next chunk see the tail of the previous.
    Set ``overlap_tokens=0`` to disable.

    Empty input returns ``[text]`` unchanged (preserves the empty
    string so chunk-index metadata stays stable for callers).

    sensitivity_tier: N/A
    """
    if not text:
        return [text]
    try:
        enc = _get_encoding()
        return _token_chunk(text, enc, max_tokens, overlap_tokens)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "tiktoken unavailable, falling back to char chunker: %s", exc,
        )
        return _char_chunk(text, max_tokens)


def _token_chunk(
    text: str,
    enc: Any,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """Sliding-window token chunking via ``tiktoken``.

    sensitivity_tier: N/A
    """
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return [text]
    if overlap_tokens >= max_tokens:
        # Pathological config — overlap would consume the whole window.
        overlap_tokens = max_tokens // 4
    step = max(1, max_tokens - overlap_tokens)
    chunks: list[str] = []
    for start in range(0, len(tokens), step):
        window = tokens[start : start + max_tokens]
        if not window:
            break
        chunks.append(enc.decode(window))
        if start + max_tokens >= len(tokens):
            break
    return chunks


def _char_chunk(text: str, max_tokens: int) -> list[str]:
    """Character-based fallback when tiktoken isn't available.

    Kept verbatim from the pre-Phase-3 implementation so a broken
    tokenizer env doesn't regress the legacy chunking behaviour.

    sensitivity_tier: N/A
    """
    max_chars = max_tokens * APPROX_CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return [text]
    chunks = _split_and_merge(text.split("\n\n"), max_chars)
    if all(len(c) <= max_chars for c in chunks):
        return chunks
    result: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            result.append(chunk)
        else:
            result.extend(_split_and_merge(chunk.split(". "), max_chars))
    final: list[str] = []
    for chunk in result:
        if len(chunk) <= max_chars:
            final.append(chunk)
        else:
            final.extend(_split_and_merge(chunk.split(" "), max_chars))
    return final


def _split_and_merge(
    parts: list[str],
    max_chars: int,
) -> list[str]:
    """Merge parts into chunks that fit within max_chars.

    sensitivity_tier: N/A
    """
    chunks: list[str] = []
    current = ""
    for part in parts:
        candidate = f"{current}\n\n{part}" if current else part
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = part
    if current:
        chunks.append(current)
    return chunks if chunks else [""]


# ------------------------------------------------------------------
# Metadata helpers
# ------------------------------------------------------------------


def recency_bucket(
    ts: str | datetime | None,
    *,
    now: datetime | None = None,
) -> str:
    """Bucket a timestamp into ``today / this_week / this_month / older``.

    Used as a filterable metadata field so Phase 4 can boost recent
    records at query time (a common implicit user preference —
    "what did we discuss?" usually means "recently"). Returns
    ``"unknown"`` for unparseable timestamps so downstream code
    doesn't need to handle ``None``.

    sensitivity_tier: 1
    """
    if not ts:
        return "unknown"
    if isinstance(ts, str):
        try:
            parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            return "unknown"
    else:
        parsed = ts
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    reference = now or datetime.now(tz=UTC)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=UTC)
    delta = reference - parsed
    if delta < timedelta(days=1) and delta > timedelta(days=-1):
        return "today"
    if delta < timedelta(days=7):
        return "this_week"
    if delta < timedelta(days=30):
        return "this_month"
    return "older"


# ------------------------------------------------------------------
# Domain classification
# ------------------------------------------------------------------

_HEALTH_KEYWORDS = frozenset(
    {
        "therapy",
        "therapist",
        "doctor",
        "dentist",
        "physical",
        "medical",
        "health",
        "anxiety",
        "mental-health",
        "sleep",
        "blood",
        "vitamin",
    }
)

_WORK_KEYWORDS = frozenset(
    {
        "standup",
        "stand-up",
        "planning",
        "1-on-1",
        "review",
        "sprint",
        "retro",
        "sync",
        "meeting",
        "scrum",
        "roadmap",
    }
)

_SOCIAL_KEYWORDS = frozenset(
    {
        "dinner",
        "lunch",
        "concert",
        "party",
        "birthday",
        "brunch",
        "hangout",
        "game night",
    }
)


def classify_message_domain(
    row: dict[str, Any],
    contacts: list[dict[str, Any]] | None = None,
) -> str:
    """Classify which collection a message belongs to.

    Mirrors the logic in int_personal_enriched.sql.

    sensitivity_tier: 1
    """
    source = str(row.get("source", "")).lower()
    sender = str(row.get("sender", "")).lower()
    recipient = str(row.get("recipient", "")).lower()

    if source == "slack":
        return "work"
    if "@company.com" in sender or "@company.com" in recipient:
        return "work"

    # Check contact relationship if available
    if contacts:
        rel = _find_relationship(sender, contacts)
        if rel in ("doctor", "therapist"):
            return "health"
        if rel in ("family", "friend"):
            return "personal"

    if source == "imessage":
        return "personal"

    return "personal"


def classify_calendar_domain(row: dict[str, Any]) -> str:
    """Classify which collection a calendar event belongs to.

    Mirrors int_events_enriched.sql event_category logic.

    sensitivity_tier: 1
    """
    title = str(row.get("title", "")).lower()
    desc = str(row.get("description", "")).lower()
    combined = f"{title} {desc}"

    if any(kw in combined for kw in _HEALTH_KEYWORDS):
        return "health"
    if any(kw in combined for kw in _WORK_KEYWORDS):
        return "work"
    if any(kw in combined for kw in _SOCIAL_KEYWORDS):
        return "social"
    if "flight" in combined:
        return "personal"

    return "personal"


def classify_note_domain(row: dict[str, Any]) -> str:
    """Classify which collection a note belongs to.

    Uses tags and title keywords.

    sensitivity_tier: 1
    """
    tags = str(row.get("tags", "")).lower()
    title = str(row.get("title", "")).lower()
    combined = f"{tags} {title}"

    work_tags = {"work", "meetings", "planning", "architecture"}
    health_tags = {
        "health",
        "mental-health",
        "sleep",
        "doctor",
    }
    social_tags = {"friends", "social"}
    ideas_tags = {
        "ideas",
        "coding",
        "learning",
        "content",
        "youtube",
    }

    if any(t in combined for t in health_tags):
        return "health"
    if any(t in combined for t in work_tags):
        return "work"
    if any(t in combined for t in social_tags):
        return "social"
    if any(t in combined for t in ideas_tags):
        return "ideas"

    return "personal"


def classify_contact_domain(row: dict[str, Any]) -> str:
    """Classify which collection a contact belongs to.

    sensitivity_tier: 1
    """
    rel = str(row.get("relationship", "")).lower()
    if rel == "colleague":
        return "work"
    return "personal"


def _find_relationship(
    identifier: str,
    contacts: list[dict[str, Any]],
) -> str:
    """Find a contact's relationship by email or name.

    sensitivity_tier: 2
    """
    identifier = identifier.lower()
    for c in contacts:
        email = str(c.get("email", "")).lower()
        name = str(c.get("name", "")).lower()
        if identifier == email or identifier == name:
            return str(c.get("relationship", "")).lower()
    return ""


# ------------------------------------------------------------------
# Indexer class
# ------------------------------------------------------------------


class Indexer:
    """Indexes DuckDB data into ChromaDB vector collections.

    sensitivity_tier: 3 (reads all raw data including high-sensitivity)
    """

    def __init__(
        self,
        duckdb: DatabaseEngine,
        chromadb: VectorEngine,
    ) -> None:
        """Initialize the indexer.

        Args:
            duckdb: DuckDB engine for reading raw tables.
            chromadb: ChromaDB engine for writing embeddings.
        """
        self._duck = duckdb
        self._chroma = chromadb

    @timed()
    def full_reindex(self) -> dict[str, int]:
        """Clear all collections and rebuild from scratch.

        Returns:
            Dict mapping collection name to document count indexed.

        sensitivity_tier: 3
        """
        logger.info("Starting full reindex…")
        self._clear_all_collections()
        docs = self._collect_all_documents()
        counts = self._upsert_documents(docs)
        logger.info("Full reindex complete: %s", counts)
        return counts

    @timed()
    def incremental_index(
        self,
        since: datetime,
    ) -> dict[str, int]:
        """Index only records created after the given timestamp.

        Args:
            since: Only records with created_at >= this are indexed.

        Returns:
            Dict mapping collection name to document count indexed.

        sensitivity_tier: 3
        """
        logger.info("Incremental index since %s…", since)
        docs = self._collect_all_documents(since=since)
        counts = self._upsert_documents(docs)
        logger.info("Incremental index complete: %s", counts)
        return counts

    # -- Private helpers -------------------------------------------

    def _clear_all_collections(self) -> None:
        """Delete all documents from every collection.

        sensitivity_tier: N/A
        """
        from src.core.chromadb.engine import COLLECTION_NAMES

        for name in COLLECTION_NAMES:
            col = self._chroma.get_or_create_collection(name)
            if col.count() > 0:
                all_ids = col.get()["ids"]
                col.delete(ids=all_ids)
        logger.info("Cleared all collections")

    def _collect_all_documents(
        self,
        since: datetime | None = None,
    ) -> list[IndexDocument]:
        """Read all tables and build IndexDocuments.

        sensitivity_tier: 3
        """
        contacts = self._load_contacts()
        docs: list[IndexDocument] = []
        docs.extend(self._index_messages(contacts, since))
        docs.extend(self._index_calendar_events(since))
        docs.extend(self._index_notes(since))
        docs.extend(self._index_health_metrics(since))
        docs.extend(self._index_contacts(since))
        docs.extend(self._index_marts(since))
        return docs

    def _load_contacts(self) -> list[dict[str, Any]]:
        """Load contacts for relationship lookup.

        sensitivity_tier: 2
        """
        return self._duck.query("SELECT name, email, relationship FROM raw_contacts")

    def _index_messages(
        self,
        contacts: list[dict[str, Any]],
        since: datetime | None = None,
    ) -> list[IndexDocument]:
        """Index raw_messages into per-domain collections.

        sensitivity_tier: 3
        """
        sql = (
            "SELECT id, source, sender, recipient, content, "
            "timestamp, sensitivity_tier, created_at "
            "FROM raw_messages"
        )
        if since:
            sql += f" WHERE created_at >= '{since.isoformat()}'"
        rows = self._duck.query(sql)

        docs: list[IndexDocument] = []
        for row in rows:
            text = compose_message_text(row)
            domain = classify_message_domain(row, contacts)
            docs.extend(
                _to_index_docs(
                    record_id=row["id"],
                    text=text,
                    collection=domain,
                    source_table="raw_messages",
                    timestamp=str(row.get("timestamp", "")),
                    sensitivity_tier=row.get(
                        "sensitivity_tier",
                        2,
                    ),
                    source=str(row.get("source", "")),
                    domain=domain,
                ),
            )
        return docs

    def _index_calendar_events(
        self,
        since: datetime | None = None,
    ) -> list[IndexDocument]:
        """Index raw_calendar_events.

        sensitivity_tier: 2
        """
        sql = (
            "SELECT id, title, description, start_time, "
            "end_time, location, attendees, "
            "sensitivity_tier, created_at "
            "FROM raw_calendar_events"
        )
        if since:
            sql += f" WHERE created_at >= '{since.isoformat()}'"
        rows = self._duck.query(sql)

        docs: list[IndexDocument] = []
        for row in rows:
            text = compose_calendar_text(row)
            domain = classify_calendar_domain(row)
            docs.extend(
                _to_index_docs(
                    record_id=row["id"],
                    text=text,
                    collection=domain,
                    source_table="raw_calendar_events",
                    timestamp=str(row.get("start_time", "")),
                    sensitivity_tier=row.get(
                        "sensitivity_tier",
                        2,
                    ),
                    source="calendar",
                    domain=domain,
                ),
            )
        return docs

    def _index_notes(
        self,
        since: datetime | None = None,
    ) -> list[IndexDocument]:
        """Index raw_notes.

        sensitivity_tier: 2
        """
        sql = (
            "SELECT id, title, content, source, tags, "
            "created_at, sensitivity_tier "
            "FROM raw_notes"
        )
        if since:
            sql += f" WHERE created_at >= '{since.isoformat()}'"
        rows = self._duck.query(sql)

        docs: list[IndexDocument] = []
        for row in rows:
            text = compose_note_text(row)
            domain = classify_note_domain(row)
            docs.extend(
                _to_index_docs(
                    record_id=row["id"],
                    text=text,
                    collection=domain,
                    source_table="raw_notes",
                    timestamp=str(row.get("created_at", "")),
                    sensitivity_tier=row.get(
                        "sensitivity_tier",
                        1,
                    ),
                    source=str(row.get("source", "")),
                    domain=domain,
                ),
            )
        return docs

    def _index_health_metrics(
        self,
        since: datetime | None = None,
    ) -> list[IndexDocument]:
        """Index raw_health_metrics (always to 'health').

        sensitivity_tier: 3
        """
        sql = (
            "SELECT id, metric_type, value, unit, "
            "recorded_at, source, sensitivity_tier, created_at "
            "FROM raw_health_metrics"
        )
        if since:
            sql += f" WHERE created_at >= '{since.isoformat()}'"
        rows = self._duck.query(sql)

        docs: list[IndexDocument] = []
        for row in rows:
            text = compose_health_metric_text(row)
            docs.extend(
                _to_index_docs(
                    record_id=row["id"],
                    text=text,
                    collection="health",
                    source_table="raw_health_metrics",
                    timestamp=str(row.get("recorded_at", "")),
                    sensitivity_tier=3,
                    source=str(row.get("source", "")),
                    domain="health",
                ),
            )
        return docs

    def _index_contacts(
        self,
        since: datetime | None = None,
    ) -> list[IndexDocument]:
        """Index raw_contacts.

        sensitivity_tier: 2
        """
        sql = (
            "SELECT id, name, email, relationship, notes, "
            "last_contact, sensitivity_tier, created_at "
            "FROM raw_contacts"
        )
        if since:
            sql += f" WHERE created_at >= '{since.isoformat()}'"
        rows = self._duck.query(sql)

        docs: list[IndexDocument] = []
        for row in rows:
            text = compose_contact_text(row)
            domain = classify_contact_domain(row)
            docs.extend(
                _to_index_docs(
                    record_id=row["id"],
                    text=text,
                    collection=domain,
                    source_table="raw_contacts",
                    timestamp=str(row.get("last_contact", "")),
                    sensitivity_tier=row.get(
                        "sensitivity_tier",
                        2,
                    ),
                    source="contacts",
                    domain=domain,
                ),
            )
        return docs

    def _index_marts(
        self,
        since: datetime | None = None,
    ) -> list[IndexDocument]:
        """Index ``mart_*`` summary tables alongside the raw records.

        Mart rows land in the same domain collections as raw records
        but carry ``layer="mart"`` metadata so the reranker (Phase 4)
        can boost or filter them independently. Mart→domain map is
        explicit (no classifier) because each mart targets one
        bounded surface — ``mart_today`` is "personal" by intent,
        ``mart_work`` is "work", etc.

        Skips marts that don't exist yet (e.g. mart_communications
        is created lazily by the pipeline) — DuckDB raises a
        catalog error which we swallow per-table.

        sensitivity_tier: 3
        """
        # Empty raw rows ⇒ empty marts. The shapes are heterogeneous,
        # so we drive each mart with its own column list. ``occurred_at``
        # is the canonical recency timestamp where present.
        mart_specs: list[tuple[str, str, str]] = [
            ("mart_today", "personal", "occurred_at"),
            ("mart_work", "work", "occurred_at"),
            ("mart_personal", "personal", "occurred_at"),
            ("mart_health", "health", "occurred_at"),
            ("mart_communications", "personal", "occurred_at"),
        ]

        docs: list[IndexDocument] = []
        for table, domain, ts_col in mart_specs:
            try:
                sql = f"SELECT * FROM {table}"
                if since:
                    sql += (
                        f" WHERE {ts_col} >= '{since.isoformat()}'"
                    )
                rows = self._duck.query(sql)
            except Exception as exc:  # noqa: BLE001
                logger.debug("mart %s unavailable: %s", table, exc)
                continue
            for row in rows:
                text = compose_mart_text(row)
                if not text.strip():
                    continue
                # Namespace mart IDs so they don't collide with raw
                # records — mart_today often copies the source ID
                # verbatim, and both would land in the same collection.
                raw_id = (
                    row.get("id")
                    or row.get(ts_col)
                    or row.get("_loaded_at")
                )
                record_id = f"{table}:{raw_id}"
                docs.extend(
                    _to_index_docs(
                        record_id=record_id,
                        text=text,
                        collection=domain,
                        source_table=table,
                        timestamp=str(
                            row.get(ts_col) or row.get("_loaded_at") or "",
                        ),
                        sensitivity_tier=int(
                            row.get("sensitivity_tier") or 2,
                        ),
                        source=table,
                        domain=domain,
                        layer="mart",
                    ),
                )
        return docs

    def _upsert_documents(
        self,
        docs: list[IndexDocument],
    ) -> dict[str, int]:
        """Upsert documents into ChromaDB and the BM25 FTS5 mirror.

        Dual-writes keep the two ranking signals over the same
        corpus — Phase 4's hybrid pipeline assumes any chunk in
        Chroma is also queryable via BM25. The FTS5 write is
        best-effort: if it fails the vector side still succeeds so
        retrieval degrades gracefully to pure cosine.

        sensitivity_tier: 3
        """
        from src.core.retrieval import bm25

        # Make sure the FTS table exists before the first upsert.
        try:
            bm25.init_table(self._duck)
        except Exception as exc:  # noqa: BLE001
            logger.warning("bm25 init_table failed: %s", exc)

        grouped: dict[str, list[IndexDocument]] = defaultdict(
            list,
        )
        for doc in docs:
            grouped[doc.collection].append(doc)

        counts: dict[str, int] = {}
        for collection_name, col_docs in grouped.items():
            self._chroma.add_documents(
                collection_name,
                documents=[d.text for d in col_docs],
                metadatas=[d.metadata for d in col_docs],
                ids=[d.id for d in col_docs],
            )
            counts[collection_name] = len(col_docs)

        # BM25 dual-write across all collections in one batch.
        try:
            bm25.upsert_documents(
                self._duck,
                [
                    {
                        "id": d.id,
                        "record_id": str(d.metadata.get("record_id") or d.id),
                        "text": d.text,
                        "collection": d.collection,
                        "layer": str(d.metadata.get("layer") or "raw"),
                        "sensitivity_tier": int(
                            d.metadata.get("sensitivity_tier") or 2,
                        ),
                    }
                    for d in docs
                ],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("bm25 upsert failed (continuing): %s", exc)

        return counts


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _to_index_docs(
    *,
    record_id: str,
    text: str,
    collection: str,
    source_table: str,
    timestamp: str,
    sensitivity_tier: int,
    source: str,
    domain: str,
    layer: str = "raw",
) -> list[IndexDocument]:
    """Chunk text and create IndexDocument(s) with metadata.

    ``layer`` distinguishes raw_* records ("raw") from mart_* summary
    rows ("mart"). Phase 4 uses this as a filterable hint so the
    reranker can pull from either layer at query time.
    ``recency_bucket`` is computed once per record (not per chunk)
    and replicated across chunks so filters are stable.

    sensitivity_tier: inherits from caller
    """
    chunks = chunk_text(text)
    bucket = recency_bucket(timestamp)
    docs: list[IndexDocument] = []

    for i, chunk in enumerate(chunks):
        doc_id = f"{record_id}-chunk-{i}" if len(chunks) > 1 else record_id
        docs.append(
            IndexDocument(
                id=doc_id,
                text=chunk,
                collection=collection,
                metadata={
                    "source_table": source_table,
                    "record_id": record_id,
                    "timestamp": timestamp,
                    "sensitivity_tier": sensitivity_tier,
                    "domain": domain,
                    "chunk_index": i,
                    "source": source,
                    "layer": layer,
                    "recency_bucket": bucket,
                },
            ),
        )
    return docs
