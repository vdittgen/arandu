"""Brain context helpers — formatting, truncation, summary.

Pure functions that project a :class:`QueryContext` into prompt-ready
text plus a structured sources list. Shared by the legacy ``BrainAgent``
(during transition) and the new :class:`BrainAgentV2` orchestrator.

sensitivity_tier: 3 (touches retrieved context which may include tier-3 data)
"""

from __future__ import annotations

from typing import Any

from src.core.query_engine import QueryContext
from src.models.prompt_cache import PROMPTS_DIR, FrozenPromptTemplate

MAX_CONTEXT_CHARS = 16_000  # ~4000 tokens (legacy)
MAX_CONTEXT_TOKENS = 2048  # safe cap for 8b model

_BRAIN_TEMPLATE = FrozenPromptTemplate(
    PROMPTS_DIR / "brain_ask_v1.txt",
)
SYSTEM_PROMPT = _BRAIN_TEMPLATE.prefix

_TIER_LABELS: dict[int, str] = {
    1: "[PUBLIC]",
    2: "[PERSONAL]",
    3: "[SENSITIVE]",
}


def _tier_label(tier: int) -> str:
    """Map a numeric sensitivity tier to a display label.

    sensitivity_tier: N/A
    """
    return _TIER_LABELS.get(tier, "[UNKNOWN]")


def _graph_source_id(row: dict[str, Any]) -> str:
    """Generate a stable source ID for a graph result."""
    return f"graph-{row.get('from_id', '')}-{row.get('to_id', '')}"


def _format_structured_item(
    row: dict[str, Any],
    table: str,
) -> str:
    """Format a structured result row as readable text.

    sensitivity_tier: inherits from caller
    """
    if table == "raw_calendar_events":
        return (
            f"{row.get('title', '')} "
            f"at {row.get('location', '?')} "
            f"({row.get('start_time', '')})"
        )
    if table == "raw_contacts":
        return (
            f"{row.get('name', '')} "
            f"({row.get('relationship', '')}) — "
            f"{row.get('notes', '')}"
        )
    if table == "raw_messages":
        content = str(row.get("content", ""))[:200]
        return f"From {row.get('sender', '?')}: {content}"
    if table == "raw_health_metrics":
        return (
            f"{row.get('metric_type', '')} = "
            f"{row.get('value', '')} {row.get('unit', '')}"
        )
    if table == "raw_notes":
        return f"{row.get('title', '')} — {str(row.get('content', ''))[:200]}"
    return str(row)


def format_context(
    ctx: QueryContext,
) -> tuple[str, list[dict[str, Any]]]:
    """Format a QueryContext into prompt text and a sources list.

    Returns:
        Tuple of (formatted_text, sources_list).

    sensitivity_tier: 3
    """
    sections: list[str] = []
    sources: list[dict[str, Any]] = []

    if ctx.structured_data:
        lines: list[str] = ["--- Database Records ---"]
        for r in ctx.structured_data:
            tier = r.get("sensitivity_tier", 2)
            label = _tier_label(tier)
            table = r.get("source_table", "unknown")
            text = _format_structured_item(r, table)
            lines.append(f"{label} [{table}] {text}")
            entry: dict[str, Any] = {
                "id": r.get("id", ""),
                "type": "structured",
                "table": table,
                "sensitivity_tier": tier,
            }
            # Expose channel + the inbound text so downstream callers
            # (channel inference, language-hint extraction) can decide
            # which connector to route a reply through and what
            # language to write the body in. ``raw_messages.source``
            # is the column-of-truth (whatsapp / imessage / gmail /
            # slack / ...); ``raw_emails`` is always the email
            # channel by definition.
            if table == "raw_messages":
                msg_source = r.get("source")
                if msg_source:
                    entry["source"] = str(msg_source)
                if r.get("is_from_me") is not None:
                    entry["is_from_me"] = bool(r.get("is_from_me"))
                content = r.get("content")
                if isinstance(content, str) and content:
                    entry["content"] = content
                sender_name = r.get("sender_name")
                if sender_name:
                    entry["sender_name"] = str(sender_name)
                timestamp = r.get("timestamp")
                if timestamp:
                    entry["timestamp"] = str(timestamp)
            elif table == "raw_emails":
                entry["source"] = "email"
                body = r.get("body") or r.get("snippet")
                if isinstance(body, str) and body:
                    entry["content"] = body
                if r.get("is_from_me") is not None:
                    entry["is_from_me"] = bool(r.get("is_from_me"))
                subject = r.get("subject")
                if subject:
                    entry["subject"] = str(subject)
            sources.append(entry)
        sections.append("\n".join(lines))

    if ctx.graph_context:
        lines = ["--- Relationships ---"]
        for r in ctx.graph_context:
            tier = r.get(
                "rel_tier",
                r.get("to_tier", 2),
            )
            label = _tier_label(tier)
            hop = r.get("hop", 1)
            if hop == 1:
                line = (
                    f"{label} {r.get('from_name', '?')}"
                    f" -> {r.get('to_name', '?')}"
                )
            else:
                line = (
                    f"{label} {r.get('from_name', '?')} "
                    f"-> {r.get('mid_name', '?')} "
                    f"-> {r.get('end_name', '?')}"
                )
            lines.append(line)
            sources.append(
                {
                    "id": _graph_source_id(r),
                    "type": "graph",
                    "sensitivity_tier": tier,
                }
            )
        sections.append("\n".join(lines))

    if ctx.vector_results:
        lines = ["--- Messages & Documents ---"]
        for r in ctx.vector_results:
            meta = r.get("metadata") or {}
            tier = meta.get("sensitivity_tier", 1)
            label = _tier_label(tier)
            collection = r.get("collection", "?")
            doc = r.get("document", "")
            ts = meta.get("timestamp", "")
            ts_str = f" ({ts})" if ts else ""
            lines.append(
                f"{label} [{collection}]{ts_str} {doc}",
            )
            # ``add_documents`` requires every chunk's metadata to
            # carry ``source`` (raw_messages.source — the channel
            # column-of-truth), ``timestamp``, ``sensitivity_tier``,
            # ``domain``. Propagate the same fields the structured
            # branch does so channel + language inference works
            # equally well whether the message arrived via a DuckDB
            # join or a semantic search. ``id`` is the chunk id, which
            # for message-derived chunks resolves to the originating
            # raw_messages.id — callers that need canonical truth can
            # round-trip through SQLite via that id.
            entry: dict[str, Any] = {
                "id": r.get("id", ""),
                "type": "vector",
                "collection": collection,
                "sensitivity_tier": tier,
            }
            if meta.get("source"):
                entry["source"] = str(meta["source"])
            if isinstance(doc, str) and doc:
                entry["content"] = doc
            if meta.get("is_from_me") is not None:
                entry["is_from_me"] = bool(meta["is_from_me"])
            if meta.get("sender_name"):
                entry["sender_name"] = str(meta["sender_name"])
            if ts:
                entry["timestamp"] = str(ts)
            sources.append(entry)
        sections.append("\n".join(lines))

    return "\n\n".join(sections), sources


def estimate_tokens(text: str) -> int:
    """Estimate token count using a word-based heuristic.

    Approximation: token_count ≈ word_count × 1.3.
    Conservative for English text with the llama tokenizer.

    sensitivity_tier: N/A
    """
    return int(len(text.split()) * 1.3)


def truncate_context(
    text: str,
    max_tokens: int = MAX_CONTEXT_TOKENS,
) -> str:
    """Truncate context text by dropping trailing lines.

    Items are already ordered by relevance (most relevant first),
    so dropping from the end preserves the best context.

    sensitivity_tier: N/A
    """
    if estimate_tokens(text) <= max_tokens:
        return text
    lines = text.split("\n")
    while (
        estimate_tokens("\n".join(lines)) > max_tokens
        and len(lines) > 1
    ):
        lines.pop()
    result = "\n".join(lines)
    if estimate_tokens(result) > max_tokens:
        max_chars = max_tokens * 4
        result = result[:max_chars]
    return result


def build_context_summary(ctx: QueryContext) -> str:
    """Build a brief text summary of retrieved context.

    sensitivity_tier: 1
    """
    parts: list[str] = []
    nv = len(ctx.vector_results)
    ng = len(ctx.graph_context)
    ns = len(ctx.structured_data)
    if nv:
        parts.append(f"{nv} document match{'es' if nv != 1 else ''}")
    if ng:
        parts.append(
            f"{ng} graph relationship{'s' if ng != 1 else ''}",
        )
    if ns:
        parts.append(
            f"{ns} database record{'s' if ns != 1 else ''}",
        )
    sources = ctx.metadata.get("sources_used", [])
    src_str = ", ".join(sources) if sources else "none"
    items_str = ", ".join(parts) if parts else "no results"
    return f"Found {items_str} from {src_str}."
