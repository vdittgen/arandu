"""Brain Agent v2 — Pydantic AI orchestrator.

The new Brain is an :class:`SBOrchestrator` whose tools are real
``pydantic_ai.Agent`` tools. Phase 2 ships two foundational tools:

- ``recall_context`` — queries the hybrid GraphRAG retrieval engine and
  returns a Markdown context block.
- ``web_search`` — falls back to the web when personal context is
  sparse. Honours the user's web-search-enabled preference.

Phase 3 will add delegation tools wrapping the migrated sub-agents
(``triage``, ``fact_extract``, ``classify_sensitivity``, etc.) via the
``subagents`` field on :class:`SBOrchestrator`.

Routing: before each ``ask`` / ``ask_stream`` call the egress firewall
classifies the question. The chosen route (``"remote"`` or ``"local"``)
selects which model the underlying ``pydantic_ai.Agent`` uses — Brain
v2 is the first call site that actually switches providers based on
egress decisions.

sensitivity_tier: 3 (sees the full user question + retrieved context)
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from collections.abc import Generator
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal, TypedDict

from src.agents.brain.context import (
    SYSTEM_PROMPT as LEGACY_SYSTEM_PROMPT,
)
from src.agents.brain.shared_tools import (
    ToolBox,
    register_shared_tools,
    seed_toolbox_from_reply_context,
)
from src.agents.core.agent_base import SBOrchestrator
from src.agents.core.cancel_registry import cancel_token, release
from src.agents.core.model_factory import local_endpoint, remote_endpoint
from src.agents.core.output_types import BrainResponse
from src.agents.core.scheduler import Tier
from src.agents.core.task_budget import TaskBudget
from src.agents.firewall.egress_firewall import default_egress_firewall
from src.agents.firewall.injection_firewall import (
    InjectionRejected,
    default_injection_firewall,
)
from src.core.query_engine import QueryEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Deps + streaming chunk shape
# ---------------------------------------------------------------------------


@dataclass
class BrainDepsV2:
    """Per-call inputs for :class:`BrainAgentV2`.

    Carrying ``QueryEngine`` as a dep (rather than constructor field)
    matches pydantic-ai's deps pattern — different runs may want
    different engines, tiers, or reference dates.

    sensitivity_tier: 3
    """

    question: str
    max_sensitivity_tier: int = 2
    reference_date: date | None = None
    web_search_enabled: bool = True


class StreamChunk(TypedDict, total=False):
    """One JSON-line chunk emitted by :meth:`BrainAgentV2.ask_stream`.

    Superset of the legacy ``BrainAgent.ask_stream`` JSON-line contract.
    The frontend (``src/interface/hooks/useStreamingChat.ts``) already
    accepts the ``action_proposal`` variant.

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


# ---------------------------------------------------------------------------
# The orchestrator
# ---------------------------------------------------------------------------


class BrainAgentV2(SBOrchestrator[BrainDepsV2, BrainResponse]):
    """Pydantic AI Brain orchestrator.

    Phase 2 surface:

    - :meth:`ask` — single-question synchronous answer.
    - :meth:`ask_stream` — JSON-line streaming generator matching the
      legacy contract (context → token → done).

    Sub-agent delegation is wired but the registry has no migrated
    sub-agents yet; Phase 3 populates ``subagents``.

    sensitivity_tier: 3
    """

    agent_id = "brain"
    output_type = BrainResponse
    tier = Tier.INTERACTIVE
    system_prompt = LEGACY_SYSTEM_PROMPT
    # Sub-agents the Brain may delegate to. The SBOrchestrator base
    # attaches each as a pydantic-ai tool; the tool body looks the
    # sub-agent up in the registry at *call* time, so unregistered
    # ids return a graceful "unavailable" string instead of crashing.
    # New entries land here as each sub-agent migrates to SBAgent.
    subagents: tuple[str, ...] = (
        "sensitivity",
        "labeler",
        "triage",
        "fact_extractor",
        "message_evaluator",
        "pending_reply",
        "contact_context",
        "actionable_events",
        "task_curator",
    )

    def __init__(
        self,
        query_engine: QueryEngine,
        *,
        deps: BrainDepsV2 | None = None,
        tool_registry: Any | None = None,
        mcp_client_factory: Any | None = None,
        provider: Any | None = None,
    ) -> None:
        super().__init__(deps=deps)
        self._query_engine = query_engine
        self._tool_registry = tool_registry
        self._mcp_client_factory = mcp_client_factory
        # Used by the propose_action tool for LLM-driven param
        # extraction and WHERE-clause generation. Falls back to the
        # default settings-driven provider on first use if not supplied.
        self._provider = provider
        # Mutable side-channel populated by the shared tools during one
        # run. ``ask`` resets it before each call; ``ask_stream`` reads
        # the captured ``pending_proposal`` to emit the terminal
        # ``action_proposal`` chunk.
        self._toolbox: ToolBox = ToolBox.empty()

    def _resolve_provider(self) -> Any:
        """Return ``self._provider`` or build one from settings."""
        if self._provider is None:
            from src.models.llm_provider import (
                create_provider_from_settings,
            )
            self._provider = create_provider_from_settings()
        return self._provider

    # ------------------------------------------------------------------
    # Tool registration
    # ------------------------------------------------------------------

    def register_tools(self, pa_agent: Any) -> None:
        """Attach Brain's tools to the underlying pydantic-ai agent.

        The four host-agnostic tools (recall_context, web_search,
        propose_action, update_notification_preferences) live in
        :mod:`src.agents.brain.shared_tools` so the Chat orchestrator
        can register the exact same surface. Sub-agent delegations
        registered by the parent ``SBOrchestrator`` helper are added
        afterwards via ``super().register_tools()``.

        sensitivity_tier: 3
        """
        register_shared_tools(
            pa_agent,
            query_engine=self._query_engine,
            deps_provider=lambda: self._deps or BrainDepsV2(
                question="",
            ),
            toolbox=self._toolbox,
            tool_registry=self._tool_registry,
            mcp_client_factory=self._mcp_client_factory,
            provider=self._resolve_provider(),
            event_sink_getter=lambda: self._event_sink(),
            tool_log_getter=lambda: self._tool_log_sink(),
        )
        # Add sub-agent delegations on top.
        super().register_tools(pa_agent)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_prompt(self, deps: BrainDepsV2) -> str:
        """User-message body for one run.

        sensitivity_tier: 3
        """
        return deps.question

    def ask(
        self,
        question: str,
        *,
        max_sensitivity_tier: int = 2,
        reference_date: date | None = None,
        reply_context: dict[str, Any] | None = None,
        budget: TaskBudget | None = None,
    ) -> BrainResponse:
        """Answer one question.

        Runs the firewalls + scheduler via ``SBAgent.run``. Returns a
        :class:`BrainResponse` whose ``sources`` and ``context_summary``
        come from the tool calls the LLM made during this run.

        ``reply_context``: when set, identifies the inbound message this
        ask originated from (a "Draft reply" click). The original
        message is seeded into the toolbox so contact resolution is
        deterministic, and ``propose_action`` hard-locks the action
        channel to its ``source``.

        ``budget``: wall-clock budget for the reflective runner. ``None``
        means ``TaskBudget.interactive_fast()`` — the default for chat
        questions. Callers like ``cmd_get_daily_brief`` pass
        ``TaskBudget.background_deep()`` when the user expects a slower
        synthesis.

        sensitivity_tier: 3
        """
        start = time.monotonic()
        deps = BrainDepsV2(
            question=question,
            max_sensitivity_tier=max_sensitivity_tier,
            reference_date=reference_date,
        )
        self._deps = deps
        self._toolbox = ToolBox.empty()
        # Every Brain run gets the reflective runner — the whole point
        # of removing ``tool_calls_limit`` is that background callers
        # (proactive insight, reply handler) also stop self-flagellating
        # past the old 4-tool wall. ``ask_stream`` pre-binds run_id +
        # tool log + event sink; plain ``ask`` callers get fresh ones.
        effective_budget = budget or TaskBudget.interactive_fast()
        if getattr(self, "_stream_run_id", None) is None:
            self._stream_run_id = uuid.uuid4().hex[:12]
        if getattr(self, "_stream_tool_log", None) is None:
            self._stream_tool_log = []
        seed_toolbox_from_reply_context(
            self._toolbox, reply_context, self._query_engine,
        )

        # Injection check first — same chokepoint discipline as
        # AgentContext.ask_llm. The verdict is recorded in the audit
        # chain by the firewall itself; the response carries a generic
        # message rather than the verdict reason to avoid echoing
        # injection payload back to a caller that may log it.
        injection_fw = default_injection_firewall()
        try:
            injection_fw.assert_allowed(
                question, calling_agent_id=self.agent_id,
            )
        except InjectionRejected:
            return BrainResponse(
                answer=(
                    "I can't answer that — the request looks like a "
                    "prompt-injection attempt."
                ),
                sources=[],
                context_summary="",
                model="firewall.injection",
                latency_ms=(time.monotonic() - start) * 1000.0,
            )

        # Eval-failure gate (only ever populated under local-only mode).
        try:
            from src.agents.core.agent_block_store import (
                default_agent_block_store,
            )
            block = default_agent_block_store().get_block(self.agent_id)
        except Exception:  # noqa: BLE001
            block = None
        if block is not None:
            return BrainResponse(
                answer=(
                    f"I can't process that here — this agent is "
                    f"blocked ({block})."
                ),
                sources=[],
                context_summary="",
                model="firewall.egress",
                latency_ms=(time.monotonic() - start) * 1000.0,
            )

        # Egress route — drives the model selection in SBAgent.run.
        egress = default_egress_firewall().classify(
            question,
            calling_agent_id=self.agent_id,
            agent_max_tier=max_sensitivity_tier,
        )

        record = self.run(
            deps, route=egress.route, budget=effective_budget,
        )
        # The LLM has no reliable way to name itself in structured
        # output, so report the actual configured endpoint instead.
        endpoint = (
            remote_endpoint() if egress.route == "remote"
            else local_endpoint()
        )
        actual_model = endpoint.model_name
        if record.output is None or record.error is not None:
            return BrainResponse(
                answer=(
                    record.error
                    or "I couldn't generate an answer for that."
                ),
                sources=list(self._toolbox.sources),
                context_summary=self._toolbox.context_summary,
                model=actual_model,
                latency_ms=record.duration_ms,
            )

        # Merge tool-captured sources into the LLM's structured output.
        # The LLM may also have populated ``sources`` from inside the
        # response — keep both, deduplicating by source id.
        merged_sources = _merge_sources(
            record.output.sources, self._toolbox.sources,
        )
        response = BrainResponse(
            answer=record.output.answer,
            sources=merged_sources,
            context_summary=(
                record.output.context_summary
                or self._toolbox.context_summary
            ),
            model=actual_model,
            latency_ms=record.duration_ms,
        )
        _try_auto_skill(
            question, response.answer, self._toolbox.tool_call_log,
        )
        return response

    def ask_stream(
        self,
        question: str,
        *,
        max_sensitivity_tier: int = 2,
        reference_date: date | None = None,
        reply_context: dict[str, Any] | None = None,
        budget: TaskBudget | None = None,
    ) -> Generator[StreamChunk, None, None]:
        """Stream a response as JSON-line chunks.

        Runs ``self.ask`` in a worker thread while pumping
        ``tool_call_start`` / ``tool_call_done`` events through a
        thread-safe queue, so the chat UI can show steps live as tools
        execute. The terminal ``context`` / ``token`` / ``done`` chunks
        follow the worker's completion.

        Emits ``run_started`` as the first chunk so the frontend can
        stash the ``run_id`` for the Stop button. Reflective-runner
        events (``self_review_*``, ``extended_research_announced``,
        ``user_stopped_research``) flow through the same queue.

        sensitivity_tier: 3
        """
        q: queue.Queue[dict[str, Any] | None] = queue.Queue()
        box: dict[str, Any] = {}

        effective_budget = budget or TaskBudget.interactive_fast()
        run_id = uuid.uuid4().hex[:12]
        self._stream_run_id = run_id
        self._stream_tool_log = []
        # Register the cancel token up front so a ``stop_research`` IPC
        # arriving before the worker starts is still honoured.
        cancel_token(run_id)

        def _run() -> None:
            try:
                box["response"] = self.ask(
                    question,
                    max_sensitivity_tier=max_sensitivity_tier,
                    reference_date=reference_date,
                    reply_context=reply_context,
                    budget=effective_budget,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Brain v2 stream failed")
                box["error"] = exc
            finally:
                q.put(None)

        self._stream_event_sink = q.put
        # Emit ``run_started`` synchronously so the very first chunk
        # carries the run_id — the frontend won't ship a Stop button
        # until it has one.
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
            yield {"type": "error", "error": str(box["error"])}
            return

        response = box["response"]

        yield {
            "type": "context",
            "context_summary": response.context_summary,
            "sources": response.sources,
        }
        # Frontend's useStreamingChat handles action_proposal as a
        # terminal chunk: cleanup() runs and isStreaming flips false.
        # Mirror legacy BrainAgent.ask_stream — emit the proposal in
        # place of token/done and return.
        if self._toolbox.pending_watcher is not None:
            yield {
                "type": "watcher_proposal",
                **self._toolbox.pending_watcher,
                "latency_ms": response.latency_ms,
            }
            return

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

        # Split the answer into typed parts so the frontend can render
        # diagrams / chart specs / HTML with the right component instead
        # of as plain text. A single markdown answer becomes one part —
        # which costs nothing — but answers with embedded ```mermaid /
        # ```vega-lite blocks become multiple ordered parts.
        from src.agents.brain.parts import split_answer_into_parts
        parts = split_answer_into_parts(
            response.answer, sensitivity_tier=max_sensitivity_tier,
        )
        if len(parts) == 1 and parts[0].mime == "text/markdown":
            # Back-compat: a plain markdown answer goes through the
            # legacy `token` chunk so older clients keep working.
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
# Helpers
# ---------------------------------------------------------------------------


def _installed_skill_ids() -> tuple[str, ...]:
    """Return IDs of all installed skills for the registry entry."""
    try:
        from src.agent_runtime.skill_loader import SkillLoader
        return tuple(s.id for s in SkillLoader().discover())
    except Exception:
        return ()


def _try_auto_skill(
    question: str,
    answer: str,
    tool_call_log: list[dict[str, Any]],
) -> None:
    """Fire-and-forget skill creation from a successful multi-step run.

    sensitivity_tier: 2
    """
    import threading

    from src.agents.skill_creator.agent import (
        maybe_create_skill,
        save_auto_learned_skill,
    )

    def _run() -> None:
        try:
            result = maybe_create_skill(
                user_query=question,
                agent_output=answer,
                tool_calls=tool_call_log,
                agent_id="brain",
            )
            if result is not None:
                save_auto_learned_skill(result)
        except Exception:
            logger.debug("auto-skill creation failed", exc_info=True)

    if len(tool_call_log) >= 3:
        threading.Thread(target=_run, daemon=True).start()


def _merge_sources(
    primary: list[dict[str, Any]],
    secondary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Concatenate two source lists, removing duplicates by id.

    sensitivity_tier: 1
    """
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for src in (*primary, *secondary):
        key = str(src.get("id") or src.get("url") or src.get("title") or src)
        if key in seen:
            continue
        seen.add(key)
        merged.append(src)
    return merged


def register_brain_v2(*, query_engine: QueryEngine | None = None) -> None:
    """Register Brain v2 with the agent registry.

    Brain is **non-editable** in the Agents page — its prompt and tool
    list are part of the orchestration contract and changing them at
    runtime would break the firewalls + sub-agent delegations. Users
    can still inspect the default config; the editor refuses writes.

    sensitivity_tier: 1
    """
    from src.agents.core.config_store import AgentConfig
    from src.agents.core.registry import (
        AgentDefinition,
        get_agent,
        register_agent,
    )
    from src.agents.core.scheduler import Tier as _Tier

    if get_agent("brain") is not None:
        return

    default = AgentConfig(
        agent_id="brain",
        system_prompt=LEGACY_SYSTEM_PROMPT,
        model_route="inherit",
        model_override=None,
        enabled_tools=(
            "recall_context",
            "web_search",
            "propose_action",
            "update_notification_preferences",
        ),
        enabled_skills=(),
        editable=False,
    )
    register_agent(AgentDefinition(
        agent_id="brain",
        name="Brain",
        description=(
            "Top-level orchestrator. Decides which sub-agents to "
            "invoke and assembles a grounded, source-attributed answer."
        ),
        category="orchestrator",
        parent_agent=None,
        tier=_Tier.INTERACTIVE,
        max_sensitivity_tier=3,
        editable=False,
        default_config=default,
        available_tools=(
            "recall_context",
            "web_search",
            "propose_action",
            "update_notification_preferences",
            "load_skill",
            "load_skill_resource",
        ),
        available_skills=_installed_skill_ids(),
        output_schema="BrainResponse",
        pattern="orchestrator",
        factory=(
            (lambda: BrainAgentV2(query_engine=query_engine))
            if query_engine is not None
            else None
        ),
        tags=("locked",),
        # Mirror the runtime delegation list onto the registry entry so
        # the Agents page can render the architecture without
        # instantiating Brain.
        subagents=BrainAgentV2.subagents,
    ))


def bootstrap_agents(*, query_engine: QueryEngine | None = None) -> None:
    """Register Brain v2 + every Phase 3a sub-agent in the right order.

    Called once at process start from ``cli.py``. Idempotent.

    sensitivity_tier: 1
    """
    from src.agents.action_proposal_judge import (
        register_action_proposal_judge,
    )
    from src.agents.actionable_events import (
        register_actionable_events_agent,
    )
    from src.agents.contact_context import (
        register_contact_context_agent,
    )
    from src.agents.core.reflection import register_reflector_agent
    from src.agents.daily_scheduler import register_daily_scheduler_agent
    from src.agents.dataset_creator import (
        register_dataset_creator_agent,
    )
    from src.agents.dataset_validator import (
        register_dataset_validator_agent,
    )
    from src.agents.event_categorizer import (
        register_event_categorizer_agent,
    )
    from src.agents.fact_extractor import register_fact_extractor_agent
    from src.agents.firewall.registration import (
        register_egress_firewall_agent,
        register_injection_firewall_agent,
    )
    from src.agents.goal_extractor import register_goal_extractor_agent
    from src.agents.habit_suggester import register_habit_suggester_agent
    from src.agents.labeler import register_labeler_agent
    from src.agents.message_eval import (
        register_message_evaluator_agent,
    )
    from src.agents.model_generator import (
        register_model_generator_agent,
    )
    from src.agents.model_picker import (
        register_model_picker_agent,
    )
    from src.agents.pending_reply import register_pending_reply_agent
    from src.agents.prompt_engineer import (
        register_prompt_engineer_agent,
    )
    from src.agents.query_router import register_query_router_agent
    from src.agents.relationship_tracker import (
        register_relationship_tracker_agent,
    )
    from src.agents.schema_discovery import (
        register_schema_discovery_agent,
    )
    from src.agents.sensitivity import register_sensitivity_agent
    from src.agents.task_completion import register_task_completion_agent
    from src.agents.task_curator import register_task_curator_agent
    from src.agents.task_proposer import register_task_proposer_agent
    from src.agents.topic_extractor import (
        register_topic_extractor_agent,
    )
    from src.agents.triage import register_triage_agent

    # Self-review checkpoint used by the reflective runner inside
    # Brain/Chat — registered first so the orchestrator instances can
    # look it up via the registry.
    register_reflector_agent()
    # Sub-agents first so Brain's delegation tools see them at
    # construction time. Each function is idempotent.
    register_action_proposal_judge()
    register_sensitivity_agent()
    register_labeler_agent()
    register_triage_agent()
    register_fact_extractor_agent()
    register_message_evaluator_agent()
    register_pending_reply_agent()
    register_contact_context_agent()
    register_actionable_events_agent()
    # Tasks/goals/habits/schedule sub-agents — registered before the
    # task_curator orchestrator so its delegation tools resolve at
    # construction time. The curator is added to Brain.subagents above.
    register_goal_extractor_agent()
    register_task_proposer_agent()
    register_task_completion_agent()
    register_daily_scheduler_agent()
    register_habit_suggester_agent()
    register_task_curator_agent()
    # Indirect sub-agents — invoked by QueryEngine / pipeline rather
    # than delegated by Brain, but still editable via the Agents page.
    register_query_router_agent()
    register_topic_extractor_agent()
    register_event_categorizer_agent()
    register_schema_discovery_agent()
    register_model_generator_agent()
    register_relationship_tracker_agent()
    # Locked firewall agents — appear in the registry so the Agents
    # page can render their cards + the manual "Run eval" button.
    register_injection_firewall_agent()
    register_egress_firewall_agent()
    # Dataset validator gates user-uploaded eval datasets. Registered
    # before user agents so the upload handler can resolve it. Dataset
    # creator proposes starter datasets — same lifecycle.
    register_dataset_validator_agent()
    register_dataset_creator_agent()
    # Model picker recommends best-overall + cost-effective models for
    # any user agent based on its spec — surfaced in the agent wizard.
    register_model_picker_agent()
    # Prompt engineer rewrites a user agent's system prompt + description
    # for clarity, expected output, language, format, scope, and safety.
    register_prompt_engineer_agent()
    from src.agents.skill_creator.agent import register_skill_creator_agent
    register_skill_creator_agent()
    register_brain_v2(query_engine=query_engine)
    # Chat agent — sits above Brain in the call graph. Registered AFTER
    # Brain so its delegation tools can resolve "brain" at construct
    # time. User agents are pulled dynamically by ChatAgent.subagents
    # so order vs. ``register_user_agents`` does not matter.
    from src.agents.chat import register_chat_agent
    register_chat_agent()
    # User-authored agents — pulled from SQLite and mounted with
    # ``editable=True`` so the Agents page can update them.
    try:
        from src.agents.user_agents.registration import (
            register_user_agents,
        )
    except ImportError:
        pass
    else:
        register_user_agents(query_engine=query_engine)


__all__ = [
    "BrainAgentV2",
    "BrainDepsV2",
    "StreamChunk",
    "bootstrap_agents",
    "register_brain_v2",
]
