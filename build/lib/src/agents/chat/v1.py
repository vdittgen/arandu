"""Chat Agent v1 — conversational layer above Brain.

The chat agent owns the conversation surface that the user sees on the
Chat page. It runs its own LLM (selectable independently of Brain) and
exposes:

- ``ask_brain(question)`` — the user's grounded reasoning layer. The
  chat LLM is told to call this for any factual / personal / memory
  question and treat the response as authoritative.
- ``recall_context(query)`` — direct retrieval against local stores.
- ``propose_action(question)`` — MCP action proposals.
- Delegation tools for every registered user-authored agent — pulled
  dynamically at instance-build time.

Brain stays an orchestrator in its own right with its own sub-agents
(sensitivity, labeler, triage, fact_extractor, insight, etc.).
Chat delegates to Brain via ``ask_brain`` and lets Brain handle
internal sub-agent orchestration — Chat never exposes Brain's
sub-agents directly.

sensitivity_tier: 3 (sees the full user question + tool outputs)
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from collections.abc import Generator
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any, Literal, TypedDict

if TYPE_CHECKING:
    from pydantic_ai import RunContext

    from src.agents.tool_registry import ToolRegistry
    from src.core.query_engine import QueryEngine
    from src.models.llm_provider import LLMProvider
else:
    try:
        from pydantic_ai import RunContext
    except ImportError:  # pragma: no cover
        RunContext = Any  # type: ignore[assignment,misc]

from src.agents.brain.shared_tools import (
    ToolBox,
    register_shared_tools,
    seed_toolbox_from_reply_context,
)
from src.agents.core.agent_base import SBOrchestrator
from src.agents.core.cancel_registry import cancel_token, release
from src.agents.core.model_factory import local_endpoint, remote_endpoint
from src.agents.core.output_types import ChatResponse
from src.agents.core.scheduler import Tier
from src.agents.core.task_budget import TaskBudget
from src.agents.firewall.egress_firewall import default_egress_firewall
from src.agents.firewall.injection_firewall import (
    InjectionRejected,
    default_injection_firewall,
)

logger = logging.getLogger(__name__)


# Chat only exposes Brain (via ask_brain) at the static level.
# Brain's own sub-agents (sensitivity, labeler, triage, etc.) are
# Brain's internal concern — Chat delegates to Brain and lets Brain
# handle further orchestration. User-authored agents are added
# dynamically since they are user-facing workflows.
_STATIC_SUBAGENT_PREFIX: tuple[str, ...] = (
    "brain",
)


DEFAULT_SYSTEM_PROMPT = """You are Arandu's chat assistant — a \
conversational interface to the user's personal data and capabilities. \
You both answer questions AND take action on the user's behalf.

You have access to these tools:

1. **ask_brain(question)** — the user's grounded reasoning layer. \
Call this for ANY question that touches the user's personal data, \
memories, schedule, contacts, files, or history. Brain handles its \
own retrieval and sub-agent orchestration internally. Treat its \
response as authoritative — quote it, do not paraphrase facts.

2. **web_search(query, max_results)** — the open web. Use when the \
question is clearly not personal and ask_brain returns nothing useful.

3. **propose_action(question)** — propose an MCP action when the \
user asks you to DO something: send a message, create a note / event \
/ reminder, schedule, draft, reply, move, flag, delete, update. The \
system asks the user to confirm before executing.

   **Hard rule**: propose_action is ONLY for messages where the user \
explicitly used a write/mutation verb (send, create, schedule, draft, \
delete, update, move, flag, cancel, reply, etc.). Read-only questions \
like "what do I have today?", "show me my schedule", "list my tasks" \
are NEVER actions — use ask_brain. If unsure, prefer ask_brain.

   **Recipient resolution is automatic.** When the request names a \
contact ("send a whatsapp to Amor", "email Sarah", "reply to João"), \
call propose_action *directly* with the user's verbatim request. Do \
NOT call ask_brain first to look up who the contact is — the action \
pipeline runs its own contact lookup against your saved contacts and \
will surface a disambiguation card if multiple matches exist. A \
pre-flight ask_brain call here only adds latency and a redundant \
LLM round-trip.

4. **User-authored agents** (delegate_user_<id>) — custom workflows \
the user has built. Invoke them when the user references them by name \
or when their description matches the request.

When NOT to call any tool:
- Greetings, small talk, formatting help, clarifying questions, \
simple acknowledgements.
- Questions about you (the assistant) or the Arandu product.
- Questions answerable from the User Context section below: the \
current date/time, the user's name, timezone, or language.
- Simple general knowledge, math, definitions, or translations \
that do not depend on the user's personal data.

Style: conversational, concise, plain Markdown. Don't list tools to \
the user. Don't speculate when grounded data is unavailable — say so \
and offer to dig further.

Never defer work inside a turn — if a tool returns empty, retry \
ONCE with a different query in the SAME turn, then state plainly \
what came back empty. Make at most 2 tool attempts on the same \
question; do not keep rephrasing past that.
"""


@dataclass
class ChatDeps:
    """Per-call inputs for :class:`ChatAgent`.

    sensitivity_tier: 3
    """

    question: str
    max_sensitivity_tier: int = 2
    reference_date: date | None = None


class StreamChunk(TypedDict, total=False):
    """One JSON-line chunk emitted by :meth:`ChatAgent.ask_stream`.

    Shape matches Brain v2's :class:`StreamChunk` so the existing
    frontend stream handler (``useStreamingChat``) consumes either
    without changes.

    sensitivity_tier: 3
    """

    type: Literal[
        "context", "token", "done", "error", "action_proposal",
        "tool_call_start", "tool_call_done",
        "run_started",
        "self_review_start", "self_review_done",
        "extended_research_announced",
        "user_stopped_research",
    ]
    context_summary: str
    sources: list[dict[str, Any]]
    token: str
    model: str
    latency_ms: float
    error: str
    proposal: dict[str, Any]
    call_id: str
    name: str
    args_summary: str
    duration_ms: float
    status: str
    result_summary: str
    # Reflective-runner extras.
    run_id: str
    task_class: str
    expected_total_ms: float
    elapsed_ms: float
    reason: str
    suggested_class: str


@dataclass
class _Grounding:
    """Sources + context summary captured from tool calls during one run.

    sensitivity_tier: 3
    """

    sources: list[dict[str, Any]] = field(default_factory=list)
    context_summary: str = ""


_TOOLLESS_SYSTEM_PROMPT = (
    "You are Arandu, a concise, helpful on-device assistant. Answer the "
    "user directly. You are running in a lightweight mode without access to "
    "the user's personal data or tools, so don't claim to look anything up."
)


def _looks_like_tool_failure(text: str) -> bool:
    """True if ``text`` is the signature of a tool-capability failure.

    Covers a model that 'does not support tools' (Ollama 400) and small
    models that emit malformed tool calls (pydantic-ai ToolRetryError /
    ModelHTTPError). The failure can surface as the agent's error OR be
    swallowed into the answer string (e.g. via the ask_brain grounding
    step), so we check both.
    """
    t = (text or "").lower()
    return (
        "does not support tools" in t
        or "toolretryerror" in t
        or "modelhttperror" in t
        or ("status_code: 400" in t and "tool" in t)
    )


def _toolless_completion(model: str, question: str) -> str | None:
    """Plain, tool-less Ollama completion.

    Graceful fallback for models that can't drive the agent's tool loop —
    e.g. a model that 'does not support tools', or one too small to emit
    well-formed tool calls. Returns the reply text, or None on failure.

    sensitivity_tier: 1 (the prompt is the user's chat message)
    """
    try:
        import ollama

        from src.models.llm_provider import load_llm_settings

        host = load_llm_settings().get("llm_host", "http://localhost:11434")
        resp = ollama.Client(host=host, timeout=120.0).chat(
            model=model,
            messages=[
                {"role": "system", "content": _TOOLLESS_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
        )
        msg = getattr(resp, "message", None)
        content = getattr(msg, "content", None)
        if content is None and isinstance(resp, dict):
            content = (resp.get("message") or {}).get("content")
        content = (content or "").strip()
        return content or None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Tool-less chat fallback failed: %s", exc)
        return None


class ChatAgent(SBOrchestrator[ChatDeps, ChatResponse]):
    """Conversational orchestrator. Delegates grounding to Brain.

    Runs an independently-configured chat model and uses tools to
    reach Brain and the registered sub-agent / user-agent surface.

    sensitivity_tier: 3
    """

    agent_id = "chat"
    output_type = ChatResponse
    tier = Tier.INTERACTIVE
    system_prompt = DEFAULT_SYSTEM_PROMPT

    def __init__(
        self,
        *,
        deps: ChatDeps | None = None,
        query_engine: QueryEngine | None = None,
        tool_registry: ToolRegistry | None = None,
        mcp_client_factory: Any | None = None,
        provider: LLMProvider | None = None,
    ) -> None:
        super().__init__(deps=deps)
        self._grounding = _Grounding()
        self._query_engine = query_engine
        self._tool_registry = tool_registry
        self._mcp_client_factory = mcp_client_factory
        self._provider = provider
        # Side-channel populated by the shared tools; mirrors Brain v2.
        self._toolbox: ToolBox = ToolBox.empty()

    def _resolve_provider(self) -> Any:
        """Return ``self._provider`` or build one from settings."""
        if self._provider is None:
            from src.models.llm_provider import (
                create_provider_from_settings,
            )
            self._provider = create_provider_from_settings()
        return self._provider

    @property  # type: ignore[override]
    def subagents(self) -> tuple[str, ...]:
        """Static prefix + every registered user agent at instance time.

        Pulled lazily so a freshly-registered user agent appears as a
        tool the next time a :class:`ChatAgent` is instantiated. The
        pydantic-ai agent is cached on ``_pa_agent`` for the lifetime
        of one instance, so live additions in the same session require
        a chat restart to surface.

        sensitivity_tier: 1
        """
        from src.agents.core.registry import all_agents

        dynamic: list[str] = []
        for defn in all_agents():
            aid = defn.agent_id
            if aid in _STATIC_SUBAGENT_PREFIX or aid == self.agent_id:
                continue
            if aid.startswith("user.") or "user" in defn.tags:
                dynamic.append(aid)
        return (*_STATIC_SUBAGENT_PREFIX, *dynamic)

    def build_prompt(self, deps: ChatDeps) -> str:
        """User-message body for one run.

        When a task_context is active, appends the task details so the
        LLM can help the user complete the task.

        sensitivity_tier: 3
        """
        prompt = deps.question
        tc = getattr(self._toolbox, "task_context", None)
        if tc and self._toolbox.sources:
            src = self._toolbox.sources[0]
            if src.get("table") == "_tasks":
                lines = [
                    "\n\n[Working on a task — help the user complete it]",
                ]
                if src.get("title"):
                    lines.append(f"Task: {src['title']}")
                if src.get("goal_title"):
                    lines.append(f"Goal: {src['goal_title']}")
                if src.get("notes"):
                    lines.append(f"Notes: {str(src['notes'])[:500]}")
                if src.get("due_at"):
                    lines.append(f"Due: {src['due_at']}")
                prompt += "\n".join(lines)
        return prompt

    # ------------------------------------------------------------------
    # Tool registration
    # ------------------------------------------------------------------

    def register_tools(self, pa_agent: Any) -> None:
        """Attach Chat's tool surface to the underlying pydantic-ai agent.

        Chat's tool reach is *Brain's shared tool surface* (recall_context,
        web_search, propose_action, update_notification_preferences) plus
        :func:`ask_brain` (Brain as a callable second-opinion layer) plus
        delegations to every registered sub-agent and user agent.

        :func:`ask_brain` calls :meth:`BrainAgentV2.ask` directly so the
        chat agent can harvest Brain's separately-emitted ``sources`` +
        ``context_summary`` into :attr:`_grounding`. Every other
        sub-agent uses the default JSON-dump delegator from the base
        class.

        sensitivity_tier: 3
        """
        from src.agents.brain.shared_tools import _wrap_tool_for_events
        from src.agents.core.registry import get_agent

        host = self

        if self._query_engine is not None:
            register_shared_tools(
                pa_agent,
                query_engine=self._query_engine,
                deps_provider=lambda: self._deps or ChatDeps(
                    question="",
                ),
                toolbox=self._toolbox,
                tool_registry=self._tool_registry,
                mcp_client_factory=self._mcp_client_factory,
                provider=self._resolve_provider(),
                event_sink_getter=lambda: host._event_sink(),
                tool_log_getter=lambda: host._tool_log_sink(),
                exclude_tools=frozenset({
                    "recall_context",
                    "update_notification_preferences",
                }),
            )

        async def ask_brain(_ctx: RunContext[None], question: str) -> str:
            """Ask Brain — the user's grounded reasoning layer.

            Use for any factual / personal / memory question. Brain
            returns an authoritative answer drawn from the user's
            local stores. Quote its conclusions; cite its sources.
            """
            definition = get_agent("brain")
            if definition is None or definition.factory is None:
                return "Brain agent unavailable in this session."
            brain = definition.factory()
            tier = self._deps.max_sensitivity_tier if self._deps else 2
            reference_date = self._deps.reference_date if self._deps else None
            try:
                response = await asyncio.to_thread(
                    brain.ask,
                    question,
                    max_sensitivity_tier=tier,
                    reference_date=reference_date,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("ask_brain failed: %s", exc, exc_info=True)
                return f"Brain call failed: {exc}"
            if response.sources:
                self._grounding.sources.extend(response.sources)
            if response.context_summary:
                if self._grounding.context_summary:
                    self._grounding.context_summary += "\n\n"
                self._grounding.context_summary += response.context_summary
            return response.answer

        pa_agent.tool(_wrap_tool_for_events(
            ask_brain,
            tool_name="ask_brain",
            sink_getter=lambda: host._event_sink(),
            args_summary_fn=lambda _ctx, question: f'question="{question}"',
            result_summary_fn=lambda r: (
                f"answer ({len(str(r))} chars)" if r else "no answer"
            ),
        ))
        # Generic delegations for every other sub-agent, plus every
        # dynamically-discovered user agent. Mirrors the body of
        # :meth:`SBOrchestrator.register_tools` but skips "brain"
        # since we wired our own ``ask_brain`` above.
        for sub_id in self.subagents:
            if sub_id == "brain":
                continue
            definition = get_agent(sub_id)
            description = (
                definition.description if definition else
                f"Invoke sub-agent {sub_id}"
            )

            def _make_tool(child_id: str, desc: str) -> Any:
                async def _delegate(
                    _ctx: RunContext[None], prompt: str,
                ) -> str:
                    """Delegate to child agent."""
                    child = get_agent(child_id)
                    if child is None or child.factory is None:
                        return f"sub-agent {child_id} unavailable"
                    instance = child.factory()
                    record = instance.run(prompt)
                    if record.output is None:
                        return (
                            f"sub-agent {child_id} returned no output"
                        )
                    return record.output.model_dump_json()

                _delegate.__doc__ = desc
                return _delegate

            raw_fn = _make_tool(sub_id, description)
            tool_fn = _wrap_tool_for_events(
                raw_fn,
                tool_name=f"delegate_{sub_id}",
                sink_getter=lambda: host._event_sink(),
                args_summary_fn=lambda _ctx, prompt: f'prompt="{prompt}"',
                result_summary_fn=lambda r: f"{len(str(r))} chars",
            )
            tool_fn.__name__ = (
                f"delegate_{sub_id.replace('.', '_').replace('-', '_')}"
            )
            tool_fn.__doc__ = description
            try:
                pa_agent.tool(tool_fn)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Could not attach delegation tool for %s",
                    sub_id, exc_info=True,
                )

    # ------------------------------------------------------------------
    # Task context seeding ("Work on this")
    # ------------------------------------------------------------------

    def _seed_task_context(
        self,
        task_context: dict[str, Any],
    ) -> None:
        """Seed the toolbox with task/habit details for "Work on this".

        sensitivity_tier: 2
        """
        if self._query_engine is None:
            return
        duck = getattr(self._query_engine, "_duck", None)
        if duck is None:
            return

        task_id = str(task_context.get("task_id") or "").strip()
        if not task_id:
            return

        try:
            rows = duck.query(
                "SELECT t.id, t.title, t.notes, t.status, t.due_at, "
                "  t.importance, t.goal_id, "
                "  g.title AS goal_title, g.category "
                "FROM _tasks t "
                "LEFT JOIN _goals g ON t.goal_id = g.id "
                "WHERE t.id = ? LIMIT 1",
                [task_id],
            )
        except Exception:  # noqa: BLE001
            logger.debug("task_context seed failed", exc_info=True)
            return

        if not rows:
            return

        row = dict(rows[0])
        self._toolbox.task_context = task_context
        self._toolbox.sources.insert(0, {
            "id": task_id,
            "type": "structured",
            "table": "_tasks",
            "sensitivity_tier": 2,
            **row,
        })

    # ------------------------------------------------------------------
    # Draft-reply fast path
    # ------------------------------------------------------------------

    def _try_draft_reply_shortcircuit(
        self,
        question: str,
        reply_context: dict[str, Any],
        start: float,
    ) -> ChatResponse | None:
        """Build an action proposal without an LLM call.

        When the user clicked "Draft reply" on a known inbound message,
        the seeded source already carries the sender JID (the
        destination for the reply). We call the action-proposal
        pipeline directly, passing the sender JID as a pre-resolved
        handle so ``resolve_recipient`` is skipped entirely.

        Returns ``None`` when the short-circuit can't fire (missing
        tool registry, no matching action, etc.) — the caller falls
        through to the normal LLM path.

        sensitivity_tier: 3
        """
        if self._tool_registry is None:
            return None

        src = self._toolbox.sources[0]
        sender_jid = src.get("sender")
        source_channel = src.get("source") or reply_context.get("source")
        if not sender_jid or not source_channel:
            return None

        try:
            from src.agents.brain.actions import (
                build_action_proposal,
                match_action_intent,
            )
            from src.agents.brain.channel_inference import (
                SOURCE_TO_CHANNEL,
                ChannelHint,
            )
            from src.core.profiler import timed_block

            channel = SOURCE_TO_CHANNEL.get(
                source_channel.lower(), source_channel.lower(),
            )
            channel_hint = ChannelHint(
                channel=channel, confidence="explicit",
            )

            with timed_block("draft_reply.match_action_intent"):
                action = match_action_intent(
                    question, self._tool_registry,
                    channel_hint=channel_hint,
                )
            if action is None:
                return None

            # Let the disambiguation card render so the user can confirm
            # or change the recipient. Pre-resolving to the raw sender
            # JID skipped the picker entirely; worse, WhatsApp's @lid
            # format isn't a real phone number, so the contact-preview
            # fuzzy-matched into the wrong contact. The picker always
            # offers the alternative contacts the resolver found by
            # name, and the brain-path UX already trained users to
            # confirm/change there.
            with timed_block("draft_reply.build_action_proposal_total"):
                proposal = build_action_proposal(
                    action,
                    question,
                    self._toolbox.context_summary or "",
                    tool_registry=self._tool_registry,
                    mcp_client_factory=self._mcp_client_factory,
                    duckdb=self._query_engine._duck,  # noqa: SLF001
                    provider=self._resolve_provider(),
                    sources=list(self._toolbox.sources),
                    skip_recipient_resolution=False,
                )

            self._toolbox.pending_proposal = proposal
            latency_ms = (time.monotonic() - start) * 1000.0

            return ChatResponse(
                answer=(
                    f"Proposed reply to "
                    f"{src.get('sender_name', 'contact')}."
                ),
                sources=list(self._toolbox.sources),
                context_summary=self._toolbox.context_summary or "",
                model="shortcircuit.draft_reply",
                latency_ms=latency_ms,
            )
        except Exception:  # noqa: BLE001
            logger.debug(
                "draft-reply short-circuit failed, falling through",
                exc_info=True,
            )
            return None

    # ------------------------------------------------------------------
    # Notification preferences fast path
    # ------------------------------------------------------------------

    _NOTIF_PATTERNS: tuple[tuple[str, str, str | None], ...] = (
        ("mute all", "mute_all", None),
        ("mute notifications", "mute_all", None),
        ("silenciar notificações", "mute_all", None),
        ("silenciar notificacoes", "mute_all", None),
        ("unmute", "unmute", None),
        ("reativar notificações", "unmute", None),
        ("reativar notificacoes", "unmute", None),
        ("show notification", "show", None),
        ("notification preferences", "show", None),
        ("preferências de notificação", "show", None),
        ("preferencias de notificacao", "show", None),
    )

    def _try_notification_shortcircuit(
        self,
        question: str,
        start: float,
    ) -> ChatResponse | None:
        """Handle notification preference requests without an LLM call.

        sensitivity_tier: 1
        """
        q = question.lower().strip()
        for pattern, action, category in self._NOTIF_PATTERNS:
            if pattern in q:
                try:
                    from src.agents.brain.notifications import (
                        apply_notification_action,
                    )
                    from src.core.sqlite.engine import DatabaseEngine
                    from src.notifications.preference_service import (
                        PreferenceService,
                    )

                    prefs = PreferenceService(
                        db_engine=DatabaseEngine(),
                    )
                    answer = apply_notification_action(
                        prefs, action, category,
                    )
                    return ChatResponse(
                        answer=answer,
                        sources=[],
                        context_summary="",
                        model="shortcircuit.notification_prefs",
                        latency_ms=(time.monotonic() - start) * 1000.0,
                    )
                except Exception:  # noqa: BLE001
                    logger.debug(
                        "notification short-circuit failed",
                        exc_info=True,
                    )
                    return None
        return None

    # ------------------------------------------------------------------
    # Schedule query fast path
    # ------------------------------------------------------------------

    _SCHEDULE_PATTERNS: tuple[str, ...] = (
        "what do i have today",
        "what's on my schedule",
        "what is on my schedule",
        "my schedule today",
        "my calendar today",
        "meetings today",
        "what do i have on my calendar",
        "what are my meetings",
        "show my schedule",
        "show my calendar",
        "o que eu tenho hoje",
        "o que tenho hoje",
        "minha agenda hoje",
        "minha agenda de hoje",
        "quais são meus compromissos",
        "quais sao meus compromissos",
        "qual a minha agenda",
        "meus compromissos de hoje",
    )

    def _try_schedule_shortcircuit(
        self,
        question: str,
        start: float,
        reference_date: date | None,
    ) -> ChatResponse | None:
        """Answer schedule queries directly from raw_calendar_events.

        Only fires for today's schedule. Falls through for future dates,
        date ranges, or more complex queries that need Brain reasoning.

        sensitivity_tier: 2
        """
        q = question.lower().strip().rstrip("?. ")
        if not any(p in q for p in self._SCHEDULE_PATTERNS):
            return None

        if self._query_engine is None:
            return None

        try:
            from datetime import datetime

            from src.core.calendar_filters import personal_events_for_date

            duck = getattr(self._query_engine, "_duck", None)
            if duck is None:
                return None

            target = reference_date or date.today()
            events = personal_events_for_date(
                duck, target.isoformat(),
                columns=(
                    "title, start_time, end_time, location, "
                    "attendees"
                ),
            )

            if not events:
                answer = (
                    f"You have no events on your calendar for "
                    f"{target.strftime('%A, %B %-d')}."
                )
            else:
                lines = [
                    f"Here's your schedule for "
                    f"{target.strftime('%A, %B %-d')}:\n",
                ]
                for ev in events:
                    st = ev.get("start_time", "")
                    et = ev.get("end_time", "")
                    try:
                        st_fmt = datetime.fromisoformat(
                            str(st),
                        ).strftime("%-I:%M %p")
                        et_fmt = datetime.fromisoformat(
                            str(et),
                        ).strftime("%-I:%M %p")
                        time_str = f"{st_fmt} – {et_fmt}"
                    except (ValueError, TypeError):
                        time_str = str(st)
                    title = ev.get("title", "Untitled")
                    loc = ev.get("location")
                    line = f"- **{time_str}** — {title}"
                    if loc:
                        line += f" ({loc})"
                    lines.append(line)
                answer = "\n".join(lines)

            sources = [
                {
                    "type": "structured",
                    "table": "raw_calendar_events",
                    "sensitivity_tier": 2,
                    **ev,
                }
                for ev in events
            ]
            return ChatResponse(
                answer=answer,
                sources=sources,
                context_summary="",
                model="shortcircuit.schedule",
                latency_ms=(time.monotonic() - start) * 1000.0,
            )
        except Exception:  # noqa: BLE001
            logger.debug(
                "schedule short-circuit failed, falling through",
                exc_info=True,
            )
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(
        self,
        question: str,
        *,
        max_sensitivity_tier: int = 2,
        reference_date: date | None = None,
        reply_context: dict[str, Any] | None = None,
        task_context: dict[str, Any] | None = None,
        budget: TaskBudget | None = None,
    ) -> ChatResponse:
        """Run one conversational turn end-to-end.

        ``reply_context``: seeds an inbound message for "Draft reply".
        ``task_context``: seeds a task/habit for "Work on this".

        ``budget``: wall-clock budget for the reflective runner.
        Defaults to ``TaskBudget.interactive_fast()``; daily-brief-style
        callers can pass ``background_deep`` to let the model spend more
        time on synthesis.

        sensitivity_tier: 3
        """
        start = time.monotonic()
        deps = ChatDeps(
            question=question,
            max_sensitivity_tier=max_sensitivity_tier,
            reference_date=reference_date,
        )
        self._deps = deps
        self._grounding = _Grounding()
        self._toolbox = ToolBox.empty()
        # Every Chat run uses the reflective runner — see BrainAgentV2.ask
        # for the rationale.
        effective_budget = budget or TaskBudget.interactive_fast()
        if getattr(self, "_stream_run_id", None) is None:
            import uuid as _uuid
            self._stream_run_id = _uuid.uuid4().hex[:12]
        if getattr(self, "_stream_tool_log", None) is None:
            self._stream_tool_log = []
        if reply_context:
            from src.core.profiler import timed_block
            with timed_block("draft_reply.seed_toolbox"):
                seed_toolbox_from_reply_context(
                    self._toolbox, reply_context, self._query_engine,
                )
        else:
            seed_toolbox_from_reply_context(
                self._toolbox, reply_context, self._query_engine,
            )

        if task_context:
            self._seed_task_context(task_context)

        if reply_context and self._toolbox.sources:
            from src.core.profiler import timed_block as _tb
            with _tb("draft_reply.shortcircuit_total"):
                fast = self._try_draft_reply_shortcircuit(
                    question, reply_context, start,
                )
            if fast is not None:
                return fast

        # Deterministic fast paths for simple, unambiguous intents.
        for shortcircuit in (
            lambda: self._try_notification_shortcircuit(question, start),
            lambda: self._try_schedule_shortcircuit(
                question, start, reference_date,
            ),
        ):
            fast = shortcircuit()
            if fast is not None:
                return fast

        injection_fw = default_injection_firewall()
        try:
            injection_fw.assert_allowed(
                question, calling_agent_id=self.agent_id,
            )
        except InjectionRejected:
            return ChatResponse(
                answer=(
                    "I can't answer that — the request looks like a "
                    "prompt-injection attempt."
                ),
                sources=[],
                context_summary="",
                model="firewall.injection",
                latency_ms=(time.monotonic() - start) * 1000.0,
            )

        try:
            from src.agents.core.agent_block_store import (
                default_agent_block_store,
            )
            block = default_agent_block_store().get_block(self.agent_id)
        except Exception:  # noqa: BLE001
            block = None
        if block is not None:
            return ChatResponse(
                answer=(
                    f"I can't process that here — this agent is "
                    f"blocked ({block})."
                ),
                sources=[],
                context_summary="",
                model="firewall.egress",
                latency_ms=(time.monotonic() - start) * 1000.0,
            )

        egress = default_egress_firewall().classify(
            question,
            calling_agent_id=self.agent_id,
            agent_max_tier=max_sensitivity_tier,
        )

        record = self.run(
            deps, route=egress.route, budget=effective_budget,
        )
        endpoint = (
            remote_endpoint() if egress.route == "remote"
            else local_endpoint()
        )
        actual_model = endpoint.model_name

        # Combine grounding harvested from ask_brain with sources +
        # context emitted by the shared tools (recall_context /
        # web_search). Same merge discipline as Brain v2.
        all_sources = [
            *self._grounding.sources,
            *self._toolbox.sources,
        ]
        combined_summary = self._grounding.context_summary
        if self._toolbox.context_summary:
            if combined_summary:
                combined_summary += "\n\n"
            combined_summary += self._toolbox.context_summary

        out_answer = (
            record.output.answer if record.output is not None else ""
        )
        err = record.error or ""
        run_failed = record.output is None or record.error is not None
        # A tool-capability failure can land in record.error OR be swallowed
        # into the answer (the ask_brain grounding step turns Brain's error
        # into a string). Detect either.
        tool_failure = (
            "tool" in err.lower()
            or _looks_like_tool_failure(err)
            or _looks_like_tool_failure(out_answer)
        )

        if run_failed or tool_failure:
            if err:
                logger.warning("Chat agent run failed: %s", err)
            # Graceful degrade: small/weak models can't drive the tool loop
            # (no tool support, or malformed tool calls). Retry once as a
            # plain, tool-less completion so the user still gets a reply —
            # ungrounded, but better than an error.
            if tool_failure:
                degraded = _toolless_completion(actual_model, question)
                if degraded:
                    logger.info(
                        "Chat degraded to tool-less completion (%s)",
                        actual_model,
                    )
                    return ChatResponse(
                        answer=degraded,
                        sources=all_sources,
                        context_summary=combined_summary,
                        model=actual_model,
                        latency_ms=record.duration_ms,
                    )
            return ChatResponse(
                answer="I couldn't generate a response for that.",
                sources=all_sources,
                context_summary=combined_summary,
                model=actual_model,
                latency_ms=record.duration_ms,
            )

        return ChatResponse(
            answer=out_answer,
            sources=all_sources,
            context_summary=combined_summary,
            model=actual_model,
            latency_ms=record.duration_ms,
        )

    def ask_stream(
        self,
        question: str,
        *,
        max_sensitivity_tier: int = 2,
        reference_date: date | None = None,
        reply_context: dict[str, Any] | None = None,
        task_context: dict[str, Any] | None = None,
        budget: TaskBudget | None = None,
    ) -> Generator[StreamChunk, None, None]:
        """Stream the response as JSON-line chunks.

        sensitivity_tier: 3
        """
        import uuid as _uuid

        q: queue.Queue[dict[str, Any] | None] = queue.Queue()
        box: dict[str, Any] = {}

        effective_budget = budget or TaskBudget.interactive_fast()
        run_id = _uuid.uuid4().hex[:12]
        self._stream_run_id = run_id
        self._stream_tool_log = []
        cancel_token(run_id)

        def _run() -> None:
            try:
                box["response"] = self.ask(
                    question,
                    max_sensitivity_tier=max_sensitivity_tier,
                    reference_date=reference_date,
                    reply_context=reply_context,
                    task_context=task_context,
                    budget=effective_budget,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Chat agent stream failed")
                box["error"] = exc
            finally:
                q.put(None)

        self._stream_event_sink = q.put
        yield {
            "type": "run_started",
            "run_id": run_id,
            "task_class": effective_budget.task_class.value,
            "expected_total_ms": effective_budget.expected_total_s * 1000.0,
        }
        worker = threading.Thread(target=_run, daemon=True)
        worker.start()
        try:
            while True:
                ev = q.get()
                if ev is None:
                    break
                yield ev  # type: ignore[misc]
        finally:
            worker.join()
            self._stream_event_sink = None
            release(run_id)
            self._stream_run_id = None
            self._stream_tool_log = None

        if "error" in box:
            # Graceful degrade if the model can't drive the tool loop.
            err_text = str(box["error"])
            if _looks_like_tool_failure(err_text):
                model = local_endpoint().model_name
                degraded = _toolless_completion(model, question)
                if degraded:
                    yield {"type": "token", "token": degraded}
                    yield {"type": "done", "model": model, "latency_ms": 0.0}
                    return
            yield {"type": "error", "error": err_text}
            return

        response = box["response"]

        # Graceful degrade: a tool-incapability failure (no tool support, or
        # a small model emitting malformed tool calls) surfaces as an
        # error-looking answer rather than an exception (the ask_brain
        # grounding step turns Brain's error into a string). Replace it with
        # a plain, tool-less completion so the user gets a real reply.
        if _looks_like_tool_failure(response.answer):
            degraded = _toolless_completion(response.model, question)
            if degraded:
                response = ChatResponse(
                    answer=degraded,
                    sources=response.sources,
                    context_summary=response.context_summary,
                    model=response.model,
                    latency_ms=response.latency_ms,
                )

        yield {
            "type": "context",
            "context_summary": response.context_summary,
            "sources": response.sources,
        }

        if self._toolbox.pending_watcher is not None:
            yield {
                "type": "watcher_proposal",
                **self._toolbox.pending_watcher,
                "latency_ms": response.latency_ms,
            }
            return

        # If an action was proposed during this turn, flush it as the
        # terminal chunk (the frontend's useStreamingChat treats it as
        # such). Mirror Brain v2's behaviour so the Confirm/Cancel UI
        # receives an identical payload shape from either orchestrator.
        if self._toolbox.pending_proposal is not None:
            from dataclasses import asdict

            from src.agents.brain.actions import (
                RecipientDisambiguationProposal,
            )
            pending = self._toolbox.pending_proposal
            chunk_type = (
                "recipient_disambiguation"
                if isinstance(pending, RecipientDisambiguationProposal)
                else "action_proposal"
            )
            yield {
                "type": chunk_type,
                "proposal": asdict(pending),
                "latency_ms": response.latency_ms,
            }
            return

        from src.agents.brain.parts import split_answer_into_parts
        parts = split_answer_into_parts(
            response.answer, sensitivity_tier=max_sensitivity_tier,
        )
        if len(parts) == 1 and parts[0].mime == "text/markdown":
            yield {"type": "token", "token": response.answer}
        else:
            for part in parts:
                yield {
                    "type": "part_start",
                    "part_id": part.id,
                    "mime": part.mime,
                    "title": part.title,
                    "display": part.display,
                    "sensitivity_tier": part.sensitivity_tier,
                }
                yield {
                    "type": "part_done",
                    "part_id": part.id,
                    "data": part.data,
                }

        yield {
            "type": "done",
            "model": response.model,
            "latency_ms": response.latency_ms,
        }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def _installed_skill_ids() -> tuple[str, ...]:
    """Return IDs of all installed skills for the registry entry."""
    try:
        from src.agent_runtime.skill_loader import SkillLoader
        return tuple(s.id for s in SkillLoader().discover())
    except Exception:
        return ()


def register_chat_agent() -> None:
    """Register the chat agent with the global registry.

    Locked agent: only ``model_route`` + ``model_override`` are
    configurable from the Agents page. System prompt + tool list are
    owned by code.

    sensitivity_tier: 1
    """
    from src.agents.core.config_store import AgentConfig
    from src.agents.core.registry import (
        AgentDefinition,
        get_agent,
        register_agent,
    )
    from src.agents.core.scheduler import Tier as _Tier

    if get_agent("chat") is not None:
        return

    chat_tools = (
        "ask_brain",
        "web_search",
        "propose_action",
    )
    default = AgentConfig(
        agent_id="chat",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        model_route="inherit",
        model_override=None,
        enabled_tools=chat_tools,
        enabled_skills=(),
        editable=False,
    )
    register_agent(AgentDefinition(
        agent_id="chat",
        name="Chat",
        description=(
            "Conversational orchestrator. Delegates grounding to "
            "Brain (ask_brain), uses web_search for non-personal "
            "queries, and propose_action for MCP mutations. "
            "Notification preferences and schedule queries are "
            "handled by deterministic short-circuits. User-authored "
            "agents are exposed as delegation tools."
        ),
        category="orchestrator",
        parent_agent=None,
        tier=_Tier.INTERACTIVE,
        max_sensitivity_tier=3,
        editable=False,
        default_config=default,
        available_tools=(*chat_tools, "load_skill", "load_skill_resource"),
        available_skills=_installed_skill_ids(),
        output_schema="ChatResponse",
        pattern="orchestrator",
        factory=lambda: ChatAgent(),
        tags=("locked",),
        subagents=_STATIC_SUBAGENT_PREFIX,
    ))


__all__ = [
    "ChatAgent",
    "ChatDeps",
    "ChatResponse",
    "StreamChunk",
    "register_chat_agent",
]
