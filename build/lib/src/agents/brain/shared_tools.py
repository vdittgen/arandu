"""Shared pydantic-ai tool definitions for Brain *and* Chat.

Brain v2 and Chat both need the same surface to reach the user's
personal context (``recall_context``), the open web (``web_search``),
MCP actions (``propose_action``), and notification preferences
(``update_notification_preferences``). Keeping the tool bodies in one
module guarantees Chat ↔ Brain parity — when the user asks Chat to
"send X a message", the Confirm/Cancel UI receives the same payload
shape as if they had asked Brain directly.

The orchestrator captures sources, context_summary, and any pending
action proposal via a small :class:`ToolBox` adapter so the bodies do
not need to know whether they are running inside Brain or Chat.

sensitivity_tier: 3 (recall_context returns Tier 3 personal context)
"""

from __future__ import annotations

import functools
import inspect
import logging
import re
import time
import unicodedata
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from pydantic_ai import RunContext
else:
    try:
        from pydantic_ai import RunContext
    except ImportError:  # pragma: no cover
        RunContext = Any  # type: ignore[assignment,misc]

from src.agents.brain.actions import (
    ActionProposal,
    RecipientDisambiguationProposal,
    build_action_proposal,
    match_action_intent,
)
from src.agents.brain.context import (
    build_context_summary,
    format_context,
    truncate_context,
)
from src.agents.brain.notifications import (
    NotificationAction,
    apply_notification_action,
)
from src.core.query_engine import QueryEngine

logger = logging.getLogger(__name__)


# Portuguese + English stopwords that don't carry retrieval signal.
# Used by ``_normalize_recall_key`` so paraphrased queries collapse
# to the same per-turn cache key. Conservative list — keep verbs and
# nouns; drop articles, prepositions, pronouns, interrogatives, and
# common connector words.
_RECALL_STOPWORDS: frozenset[str] = frozenset({
    # Portuguese
    "os", "as", "um", "uma", "uns", "umas",
    "de", "do", "da", "dos", "das", "no", "na", "nos", "nas",
    "em", "por", "para", "com", "sem", "que", "qual", "quais",
    "quem", "onde", "quando", "como", "ou", "ser", "tem", "ter",
    "tenho", "voce", "voc", "alguma", "algum", "qualquer",
    "todos", "todas", "todo", "toda", "sobre", "contexto",
    "buscar", "procurar", "nota", "notas", "lembrete", "prompt",
    "salvo", "conversa", "conversas", "arquivos", "arquivo",
    "documentos", "documento",
    # English
    "the", "and", "any", "all", "some", "what", "who", "where",
    "when", "how", "why", "search", "find", "about", "from",
    "for", "with", "without", "have", "has", "had", "are", "was",
    "were", "does", "did", "note", "notes", "prompt", "saved",
    "conversation", "conversations", "file", "files", "document",
    "documents",
})


def _normalize_recall_key(query: str) -> str:
    """Aggressively normalize a query for cache deduplication.

    Lowercase, strip diacritics, tokenize, drop stopwords + short
    tokens, sort. Trades precision for recall so paraphrases like
    "Aumento salário Mariana?" and "Mariana aumento de salário"
    collapse to one key. Falls back to the stripped ASCII form for
    short or all-stopword queries.

    sensitivity_tier: 1
    """
    nfkd = unicodedata.normalize("NFKD", query.lower())
    ascii_only = "".join(
        c for c in nfkd if not unicodedata.combining(c)
    )
    tokens = re.findall(r"[a-z0-9]+", ascii_only)
    meaningful = sorted({
        t for t in tokens
        if len(t) >= 3 and t not in _RECALL_STOPWORDS
    })
    if meaningful:
        return " ".join(meaningful)
    return ascii_only.strip()


class _ToolHostDeps(Protocol):
    """Minimum shape the shared tools need from the host orchestrator."""

    @property
    def max_sensitivity_tier(self) -> int: ...

    @property
    def reference_date(self) -> Any: ...

    @property
    def web_search_enabled(self) -> bool: ...


@dataclass
class ToolBox:
    """Mutable per-run state the shared tools read and write.

    A single instance lives on the host orchestrator (Brain or Chat)
    for the lifetime of one ``ask`` / ``ask_stream`` call. The
    orchestrator resets the fields between runs and reads them after
    the agent loop finishes.

    ``reply_context`` is populated when the run originated from a
    "Draft reply" click on a specific inbound message (Today's Loops /
    Inbox / Open Loops). It pins the source channel + original
    ``raw_messages.id`` so ``propose_action`` can hard-lock the channel
    instead of inferring it from fuzzy semantic context.

    sensitivity_tier: 3
    """

    sources: list[dict[str, Any]]
    context_summary: str
    pending_proposal: ActionProposal | RecipientDisambiguationProposal | None
    pending_watcher: dict[str, str] | None = None
    reply_context: dict[str, Any] | None = None
    task_context: dict[str, Any] | None = None
    tool_call_log: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def empty(cls) -> ToolBox:
        """sensitivity_tier: 1"""
        return cls(
            sources=[],
            context_summary="",
            pending_proposal=None,
            pending_watcher=None,
            reply_context=None,
            tool_call_log=[],
        )


_EventSink = Callable[[dict[str, Any]], None]


def seed_toolbox_from_reply_context(
    toolbox: ToolBox,
    reply_context: dict[str, Any] | None,
    query_engine: Any,
) -> None:
    """Pin a ``reply_context`` to the toolbox and seed sources.

    Called by orchestrators (Brain, Chat) at the start of an ``ask`` /
    ``ask_stream`` run when the user clicked "Draft reply" on a known
    inbound message. Stores the reply context (consumed by
    ``propose_action`` to hard-lock the action channel) and inserts the
    original ``raw_messages`` row at the front of ``sources`` so the
    LLM has the authoritative sender/JID/email/content even if its
    semantic recall misses or misranks the row.

    Failures here are best-effort — a missing message simply means the
    Brain falls back to its normal recall path.

    sensitivity_tier: 2
    """
    if not reply_context:
        return
    toolbox.reply_context = reply_context
    message_id = str(reply_context.get("message_id") or "").strip()
    if not message_id:
        return
    # QueryEngine stores the analytical engine in ``_duck`` (private
    # by convention; other tools — e.g. ChatAgent's shortcircuit at
    # chat/v1.py — access it the same way).
    duck = getattr(query_engine, "_duck", None)
    if duck is None:
        return
    try:
        rows = duck.query(
            "SELECT id, source, sender, sender_name, recipient, "
            "content, is_from_me, timestamp "
            "FROM raw_messages WHERE id = ? LIMIT 1",
            [message_id],
        )
    except Exception:  # noqa: BLE001
        logger.debug(
            "reply_context message seed query failed", exc_info=True,
        )
        return
    if not rows:
        return
    row = rows[0]
    entry: dict[str, Any] = {
        "id": str(row.get("id") or message_id),
        "type": "structured",
        "table": "raw_messages",
        "source": str(
            row.get("source") or reply_context.get("source") or "",
        ),
        "sensitivity_tier": 2,
    }
    sender = row.get("sender")
    if sender:
        entry["sender"] = str(sender)
    sender_name = row.get("sender_name")
    if sender_name:
        entry["sender_name"] = str(sender_name)
    recipient = row.get("recipient")
    if recipient:
        entry["recipient"] = str(recipient)
    content = row.get("content")
    if isinstance(content, str) and content:
        entry["content"] = content
    is_from_me = row.get("is_from_me")
    if is_from_me is not None:
        entry["is_from_me"] = bool(is_from_me)
    timestamp = row.get("timestamp")
    if timestamp is not None:
        entry["timestamp"] = (
            timestamp.isoformat() if hasattr(timestamp, "isoformat")
            else str(timestamp)
        )
    toolbox.sources.insert(0, entry)


def _truncate(text: str, limit: int = 200) -> str:
    """Trim free-form text for inclusion in a stream summary.

    sensitivity_tier: 1
    """
    s = text if isinstance(text, str) else str(text)
    if len(s) <= limit:
        return s
    return s[: limit - 1] + "…"


def _wrap_tool_for_events(
    tool_fn: Callable[..., Any],
    *,
    tool_name: str,
    sink_getter: Callable[[], _EventSink | None],
    args_summary_fn: Callable[..., str],
    result_summary_fn: Callable[[Any], str] | None = None,
    toolbox: ToolBox | None = None,
    tool_log_getter: Callable[[], list[Any] | None] | None = None,
) -> Callable[..., Any]:
    """Wrap a pydantic-ai tool body so it emits start/done events.

    ``sink_getter`` is called once per invocation so the wrapper picks
    up the host orchestrator's current sink even if it was swapped
    between calls (e.g. ``ask`` vs. ``ask_stream``). If the sink is
    ``None``, the wrapper is a transparent no-op for events.

    ``tool_log_getter`` is the reflective-runner hook: when bound, each
    completed call appends a ``ToolCallEntry`` to the list so the
    Reflector can see what tools have already been used. The wrapper
    still runs the tool when neither sink nor log is bound.

    sensitivity_tier: 1
    """
    async def _wrapped(*args: Any, **kwargs: Any) -> Any:
        sink = sink_getter()
        tool_log = tool_log_getter() if tool_log_getter is not None else None
        if sink is None and tool_log is None:
            return await tool_fn(*args, **kwargs)
        call_id = uuid.uuid4().hex[:8]
        try:
            args_summary = _truncate(args_summary_fn(*args, **kwargs))
        except Exception:  # noqa: BLE001
            args_summary = ""
        if sink is not None:
            try:
                sink({
                    "type": "tool_call_start",
                    "call_id": call_id,
                    "name": tool_name,
                    "args_summary": args_summary,
                })
            except Exception:  # noqa: BLE001
                logger.debug("tool_call_start sink failed", exc_info=True)
        start = time.monotonic()
        try:
            result = await tool_fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            duration_ms = (time.monotonic() - start) * 1000.0
            if sink is not None:
                try:
                    sink({
                        "type": "tool_call_done",
                        "call_id": call_id,
                        "name": tool_name,
                        "duration_ms": duration_ms,
                        "status": "error",
                        "result_summary": "",
                        "error": _truncate(str(exc)),
                    })
                except Exception:  # noqa: BLE001
                    logger.debug(
                        "tool_call_done sink failed", exc_info=True,
                    )
            if tool_log is not None:
                from src.agents.core.reflection import ToolCallEntry
                tool_log.append(ToolCallEntry(
                    name=tool_name,
                    args_summary=args_summary,
                    result_summary="",
                    duration_ms=duration_ms,
                    status="error",
                ))
            raise
        duration_ms = (time.monotonic() - start) * 1000.0
        if result_summary_fn is not None:
            try:
                result_summary = _truncate(result_summary_fn(result))
            except Exception:  # noqa: BLE001
                result_summary = ""
        else:
            result_summary = _truncate(str(result))
        if sink is not None:
            try:
                sink({
                    "type": "tool_call_done",
                    "call_id": call_id,
                    "name": tool_name,
                    "duration_ms": duration_ms,
                    "status": "ok",
                    "result_summary": result_summary,
                })
            except Exception:  # noqa: BLE001
                logger.debug("tool_call_done sink failed", exc_info=True)
        if toolbox is not None:
            toolbox.tool_call_log.append({
                "name": tool_name,
                "args_summary": args_summary,
                "result_summary": result_summary,
            })
        if tool_log is not None:
            from src.agents.core.reflection import ToolCallEntry
            tool_log.append(ToolCallEntry(
                name=tool_name,
                args_summary=args_summary,
                result_summary=result_summary,
                duration_ms=duration_ms,
                status="ok",
            ))
        return result

    # pydantic-ai builds the tool's JSON schema by inspecting the
    # function signature; without preserving it the wrapper looks like
    # ``(*args, **kwargs)`` and schema generation fails with
    # "First parameter of tools that take context must be annotated
    # with RunContext[...]". ``functools.wraps`` copies ``__doc__`` and
    # ``__name__``; we additionally pin ``__signature__`` so
    # ``inspect.signature(_wrapped)`` returns the original.
    functools.wraps(tool_fn)(_wrapped)
    try:
        _wrapped.__signature__ = inspect.signature(tool_fn)  # type: ignore[attr-defined]
    except (TypeError, ValueError):  # pragma: no cover
        pass
    return _wrapped


def register_shared_tools(
    pa_agent: Any,
    *,
    query_engine: QueryEngine,
    deps_provider: Any,
    toolbox: ToolBox,
    tool_registry: Any | None = None,
    mcp_client_factory: Any | None = None,
    provider: Any | None = None,
    event_sink_getter: Callable[[], _EventSink | None] | None = None,
    tool_log_getter: Callable[[], list[Any] | None] | None = None,
    exclude_tools: frozenset[str] = frozenset(),
) -> None:
    """Attach host-agnostic tools to ``pa_agent``.

    ``deps_provider`` is a zero-arg callable returning the current
    :class:`BrainDepsV2` / :class:`ChatDeps`; we call it inside each
    tool so a host that swaps deps between turns still serves fresh
    state. ``toolbox`` carries the side-channel outputs (sources,
    context_summary, pending_proposal) that the host harvests after
    the agent loop returns.

    ``exclude_tools`` is an optional set of tool names to skip
    registration for. Chat uses this to omit tools that are handled
    by deterministic short-circuits or by Brain internally.

    sensitivity_tier: 3
    """
    engine = query_engine
    # Per-turn cache: the LLM often calls recall_context multiple times
    # with rephrased versions of the same query. ``_normalize_recall_key``
    # collapses paraphrases (lowercase, strip accents, drop stopwords,
    # sort tokens) so the cache catches them without re-running the
    # full RAG pipeline (routing LLM + DuckDB + vector + graph).
    _recall_cache: dict[str, str] = {}

    async def recall_context(
        _ctx: RunContext[None], query: str,
    ) -> str:
        """Retrieve personal context for ``query`` from local stores."""
        cache_key = _normalize_recall_key(query)
        if cache_key in _recall_cache:
            return _recall_cache[cache_key]

        deps = deps_provider() if callable(deps_provider) else deps_provider
        max_tier = getattr(deps, "max_sensitivity_tier", 2)
        ref_date = getattr(deps, "reference_date", None)
        qctx = engine.query(
            query,
            max_sensitivity_tier=max_tier,
            reference_date=ref_date,
        )
        text, sources = format_context(qctx)
        text = truncate_context(text)
        toolbox.sources.extend(sources)
        if toolbox.context_summary:
            toolbox.context_summary += "\n\n"
        toolbox.context_summary += build_context_summary(qctx)
        result = text or "No personal context found."
        _recall_cache[cache_key] = result
        return result

    async def web_search(
        _ctx: RunContext[None], query: str, max_results: int = 5,
    ) -> str:
        """Web search fallback when personal context is insufficient."""
        deps = deps_provider() if callable(deps_provider) else deps_provider
        if not getattr(deps, "web_search_enabled", True):
            return "Web search disabled by user preference."
        try:
            from src.core.web_search import search as _ws
        except ImportError:
            return "Web search unavailable."
        try:
            results = _ws(query, max_results=max_results)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Web search failed: %s", exc)
            return f"Web search error: {exc}"
        if not results:
            return "No web results."
        toolbox.sources.extend({"type": "web", **r} for r in results)
        lines = ["Web results:"]
        for r in results:
            title = r.get("title", "(untitled)")
            url = r.get("href") or r.get("url", "")
            snippet = r.get("body") or r.get("snippet", "")
            lines.append(f"- [{title}]({url}) — {snippet}")
        return "\n".join(lines)

    async def propose_action(
        _ctx: RunContext[None], question: str,
    ) -> str:
        """Propose an MCP action for user confirmation.

        Call this when the user asks you to *do* something with one
        of their connectors (create a note, send an email, schedule
        an event, etc.) rather than just answering a question.
        """
        if tool_registry is None:
            return "Action proposals unavailable — no tool registry."

        # Defensive guard: the LLM occasionally rewrites a pure
        # interrogative ("what do I have today?") into something with
        # a write verb ("create event for today's schedule") and
        # passes the rewrite here, which then matches a destructive
        # tool. Match the matcher's own ``_is_pure_question`` rule
        # against the *user's original message* so we refuse to
        # propose anything when they only asked a question.
        try:
            import re as _re

            from src.agents.tool_registry import (
                _ACTION_VERBS,
                _is_pure_question,
            )

            deps = (
                deps_provider() if callable(deps_provider) else deps_provider
            )
            user_text = str(getattr(deps, "question", "") or "").lower()
            if user_text:
                user_words = set(_re.findall(r"[a-zà-ÿ]+", user_text))
                user_verbs = user_words & _ACTION_VERBS
                if _is_pure_question(user_text, user_verbs):
                    return (
                        "That looks like a question, not a request to "
                        "perform an action. Use recall_context or "
                        "ask_brain to answer it."
                    )
        except Exception:  # noqa: BLE001
            # Guard is best-effort — fall through to the matcher.
            logger.debug("propose_action question guard failed", exc_info=True)

        # Infer the inbound channel from the user's text + the
        # grounded sources we accumulated during this turn. The
        # matcher will use this to prefer same-channel tools (e.g.
        # WhatsApp reply when the original message was on WhatsApp,
        # not the default ``reply_email``).
        #
        # If the run originated from a "Draft reply" click that pinned
        # the source channel, skip inference and build an *explicit*
        # ChannelHint — ``filter_tools_by_channel`` then drops any
        # non-matching tool entirely instead of merely re-ranking.
        from src.agents.brain.channel_inference import (
            SOURCE_TO_CHANNEL,
            ChannelHint,
            infer_action_channel,
        )
        reply_ctx = toolbox.reply_context or {}
        src_value = str(reply_ctx.get("source") or "").strip().lower()
        if src_value in SOURCE_TO_CHANNEL:
            channel_hint = ChannelHint(
                channel=SOURCE_TO_CHANNEL[src_value],
                confidence="explicit",
            )
        else:
            channel_hint = infer_action_channel(
                question,
                context_text=toolbox.context_summary or "",
                sources=list(toolbox.sources),
            )

        action = match_action_intent(
            question, tool_registry, channel_hint=channel_hint,
        )
        if action is None:
            return "No matching action available for that request."
        try:
            proposal = build_action_proposal(
                action,
                question,
                toolbox.context_summary or "",
                tool_registry=tool_registry,
                mcp_client_factory=mcp_client_factory,
                duckdb=engine._duck,  # noqa: SLF001
                provider=provider,
                sources=list(toolbox.sources),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Action proposal build failed: %s", exc)
            return f"Action proposal failed: {exc}"
        toolbox.pending_proposal = proposal
        if isinstance(proposal, RecipientDisambiguationProposal):
            return (
                f"Need to confirm the recipient for "
                f"{proposal.display_name}. Awaiting user choice."
            )
        return (
            f"Proposed {proposal.display_name}. "
            "Awaiting user confirmation."
        )

    async def create_watcher(
        _ctx: RunContext[None], name: str, prompt: str,
    ) -> str:
        """Open the Watcher creation wizard on the user's screen.

        Call this when the user wants to set up an automated agent,
        watcher, or monitor that runs on a schedule — for example
        "create an agent to check my email for bills", "set up a
        watcher for tech news", "I want something that tracks my
        stocks every morning", etc.

        Parameters:
          name: short descriptive name for the watcher (2–6 words)
          prompt: the user's full request describing what to watch for
        """
        toolbox.pending_watcher = {"name": name, "prompt": prompt}
        return (
            f'Opening the Watcher wizard for "{name}". '
            "The user can configure sources, schedule, and "
            "notification channels in the wizard."
        )

    async def update_notification_preferences(
        _ctx: RunContext[None],
        action: NotificationAction,
        category: str | None = None,
    ) -> str:
        """Update or show the user's notification preferences."""
        try:
            from src.core.sqlite.engine import DatabaseEngine
            from src.notifications.preference_service import (
                PreferenceService,
            )

            prefs = PreferenceService(db_engine=DatabaseEngine())
            return apply_notification_action(prefs, action, category)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "update_notification_preferences failed: %s", exc,
            )
            return (
                "Sorry, I couldn't update your notification "
                "preferences right now. Please try again."
            )

    # -- Skill tools (progressive disclosure L2 + L3) -------------------

    async def load_skill(
        _ctx: RunContext[None],
        skill_id: str,
    ) -> str:
        """Load full instructions for an installed skill.

        Call this when a user's request matches one of the skills
        listed in the system prompt. Returns the full SKILL.md
        instructions to follow.
        """
        from src.agent_runtime.skill_loader import SkillLoader

        loader = SkillLoader()
        doc = loader.load(skill_id)
        if doc is None:
            return f"Skill '{skill_id}' not found."
        return doc.instructions

    async def load_skill_resource(
        _ctx: RunContext[None],
        skill_id: str,
        path: str,
    ) -> str:
        """Read a resource file bundled with a skill.

        Use when a skill's instructions reference an additional
        file (template, checklist, etc.) via a relative path.
        """
        from src.agent_runtime.skill_loader import SkillLoader

        loader = SkillLoader()
        content = loader.load_resource(skill_id, path)
        if content is None:
            return f"Resource '{path}' not found in skill '{skill_id}'."
        return content

    sink_getter: Callable[[], _EventSink | None] = (
        event_sink_getter if event_sink_getter is not None else (lambda: None)
    )

    _tools: list[tuple[Any, str, Any, Any]] = [
        (
            recall_context, "recall_context",
            lambda _ctx, query: f'query="{query}"',
            lambda r: f"{len(toolbox.sources)} sources",
        ),
        (
            web_search, "web_search",
            (
                lambda _ctx, query, max_results=5:
                    f'query="{query}", max_results={max_results}'
            ),
            lambda r: _truncate(str(r), 120),
        ),
        (
            propose_action, "propose_action",
            lambda _ctx, question: f'intent="{question}"',
            lambda r: _truncate(str(r), 120),
        ),
        (
            create_watcher,
            "create_watcher",
            lambda _ctx, name, prompt: f'name="{name}", prompt="{prompt}"',
            lambda r: _truncate(str(r), 120),
        ),
        (
            update_notification_preferences,
            "update_notification_preferences",
            (
                lambda _ctx, action, category=None:
                    f"action={action}, category={category!r}"
            ),
            lambda r: _truncate(str(r), 120),
        ),
        (
            load_skill, "load_skill",
            lambda _ctx, skill_id: f'skill_id="{skill_id}"',
            lambda r: _truncate(str(r), 120),
        ),
        (
            load_skill_resource, "load_skill_resource",
            lambda _ctx, skill_id, path: f'skill="{skill_id}", path="{path}"',
            lambda r: _truncate(str(r), 120),
        ),
    ]
    for fn, name, args_fn, result_fn in _tools:
        if name in exclude_tools:
            continue
        pa_agent.tool(_wrap_tool_for_events(
            fn,
            tool_name=name,
            sink_getter=sink_getter,
            args_summary_fn=args_fn,
            result_summary_fn=result_fn,
            toolbox=toolbox,
            tool_log_getter=tool_log_getter,
        ))


__all__ = ["ToolBox", "register_shared_tools"]
