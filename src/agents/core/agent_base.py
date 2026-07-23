"""Pydantic AI agent base classes.

Three first-class patterns:

- :class:`SBAgent` — single-workflow agent. Stateless transformation,
  one LLM call, structured output. Used by ~11 of our 14 LLM components.
- :class:`SBOrchestrator` — registers sub-agents as tools. The parent's
  LLM picks which sub-agent to invoke at runtime. Used by Brain,
  Proactive Intelligence, and Reply Handler.
- :class:`SBDeepAgent` — autonomous planning + sandboxed file/code ops +
  sub-task delegation. Built on the same ``pydantic_ai.Agent`` but with
  a control loop and ``Plan`` scratchpad.

A fourth pattern, *programmatic hand-off*, is a coding convention rather
than a class: application code calls ``agent_a.run(...)`` then
``agent_b.run(...)``. No base needed.

All three classes route LLM calls through the scheduler, the egress
firewall (for tier-aware routing), and the injection firewall (for
prompt scanning). They append every decision to the audit chain.

The pydantic-ai import is deferred so unrelated test modules can still
import this file even when pydantic-ai-slim is missing.

sensitivity_tier: varies
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from pydantic_ai import RunContext
else:
    try:
        from pydantic_ai import RunContext
    except ImportError:  # pragma: no cover
        RunContext = Any  # type: ignore[assignment,misc]

from src.agents.core.audit import default_chain, hash_payload
from src.agents.core.config_store import current_model_override
from src.agents.core.model_factory import default_factory
from src.agents.core.output_types import AgentOutput, Plan, PlanStep
from src.agents.core.sandbox import (
    SandboxResult,
    WorkspaceError,
    resolve_in_workspace,
    run_python,
    run_sql,
    workspace_for,
)
from src.agents.core.scheduler import Route, Tier, default_scheduler

logger = logging.getLogger(__name__)


def _truncate_summary(text: str, limit: int = 200) -> str:
    """Trim free-form text for inclusion in a tool-call stream event.

    sensitivity_tier: 1
    """
    s = text if isinstance(text, str) else str(text)
    if len(s) <= limit:
        return s
    return s[: limit - 1] + "…"

# Bounded thread pool used to offload ``pa_agent.run_sync`` calls when
# the caller is already inside a running event loop. pydantic-ai's
# ``run_sync`` is ``loop.run_until_complete(...)`` on the *currently
# set* loop, so calling it twice in the same thread raises
# ``RuntimeError: This event loop is already running``. Hopping to a
# worker thread gives pydantic-ai a fresh loop without monkey-patching
# asyncio. Max-workers caps fan-out so a runaway recursion can't
# exhaust the process.
_PA_EXECUTOR = ThreadPoolExecutor(
    max_workers=8, thread_name_prefix="sbagent-pa",
)


def _run_pa_sync(pa_agent: Any, prompt: str) -> Any:
    """Run ``pa_agent.run_sync(prompt)`` safely even from inside a loop.

    If no event loop is running in the current thread, calls
    ``run_sync`` directly. Otherwise submits to a worker thread (which
    has no bound loop) and blocks on the result. Mirrors the pattern
    Chat already uses for its ``ask_brain`` tool (``asyncio.to_thread``)
    but works from synchronous call sites too.

    No hard ``usage_limits`` are passed — pydantic-ai's default
    ``request_limit=50`` is the only ceiling, and that's a safety net
    against runaway recursion rather than a budget. Single-call agents
    use this helper directly; orchestrators that need wall-clock
    reflection use :func:`_run_pa_with_reflection_sync` instead.

    sensitivity_tier: 1
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return pa_agent.run_sync(prompt)
    return _PA_EXECUTOR.submit(
        lambda: pa_agent.run_sync(prompt),
    ).result()


# ---------------------------------------------------------------------------
# Reflective runner: wall-clock budget + self-review + user cancel
# ---------------------------------------------------------------------------

# Stop messages injected into the pydantic-ai loop. The "by_user" variant
# is what the model sees when the user clicks Stop in the chat banner;
# the "by_review" variant fires when self-review concludes more research
# is not worth the time. Both produce the same behavior — the model
# finalizes its answer using whatever context it already has and does
# not call further tools.
_STOP_REQUEST_BY_USER = (
    "STOP_REQUEST: the user has asked you to stop researching. "
    "Synthesize the final answer with the context you already have. "
    "Do NOT call any more tools."
)
_STOP_REQUEST_BY_REVIEW = (
    "STOP_REQUEST: internal self-review determined that further "
    "research is not justified for this question. Synthesize the "
    "final answer with the context you already have. Do NOT call any "
    "more tools."
)


def _emit_event(
    sink: Callable[[dict[str, Any]], None] | None,
    event: dict[str, Any],
) -> None:
    """Forward an event to ``sink`` if one is bound, swallowing errors.

    Reflective-runner events are advisory — a broken sink must never
    abort a run.

    sensitivity_tier: 1
    """
    if sink is None:
        return
    try:
        sink(event)
    except Exception:  # noqa: BLE001
        logger.debug("reflective sink failed", exc_info=True)


async def _run_pa_with_reflection_async(
    pa_agent: Any,
    prompt: str,
    *,
    budget: Any,
    run_id: str,
    reflector: Any,
    event_sink: Callable[[dict[str, Any]], None] | None,
    tool_log: list[Any],
) -> Any:
    """Drive a pydantic-ai run with reflection checkpoints + cancel.

    Replaces ``pa_agent.run_sync(prompt)`` for orchestrators (Brain,
    Chat) where the user benefits from "stop researching when enough is
    enough" behavior. Walks ``agent.iter(prompt)`` node-by-node so we
    can:

    1. Mutate the next ``ModelRequestNode`` to append a STOP_REQUEST
       ``UserPromptPart`` when the user cancels or self-review says
       to wrap up.
    2. Run a Tier A self-review every ``budget.first_reflect_at_s`` /
       ``budget.reflect_interval_s`` of wall-clock time and emit
       ``self_review_*`` events for the UI.
    3. Promote the active task class (FAST → DEEP) when review says
       the question is genuinely complex, and emit
       ``extended_research_announced`` so the UI shows the "Researching
       deeper" banner.

    The final ``agent_run.result`` is returned — same shape as
    ``pa_agent.run_sync`` would return — so callers can read
    ``result.output`` / ``result.usage()`` as before.

    sensitivity_tier: 2 (carries the user's prompt)
    """
    from pydantic_ai.messages import ModelRequest, UserPromptPart
    from pydantic_graph import End

    from src.agents.core.cancel_registry import should_stop
    from src.agents.core.task_budget import TaskBudget

    active_budget = budget
    next_reflect_at = active_budget.first_reflect_at_s
    pending_inject: str | None = None
    cancel_emitted = False
    start = time.monotonic()

    async with pa_agent.iter(prompt) as agent_run:
        next_node = agent_run.next_node
        while not isinstance(next_node, End):
            elapsed = time.monotonic() - start

            # If we've queued a STOP_REQUEST, inject it at the first
            # ModelRequestNode we see and skip further checks until the
            # model produces its final answer.
            if pending_inject is not None:
                request = getattr(next_node, "request", None)
                if isinstance(request, ModelRequest):
                    request.parts.append(
                        UserPromptPart(content=pending_inject),
                    )
                    pending_inject = None
            elif not cancel_emitted and should_stop(run_id):
                _emit_event(event_sink, {
                    "type": "user_stopped_research",
                    "elapsed_ms": elapsed * 1000.0,
                })
                pending_inject = _STOP_REQUEST_BY_USER
                cancel_emitted = True
                # If the very next node is already a ModelRequestNode,
                # inject immediately so we don't waste another LLM turn.
                request = getattr(next_node, "request", None)
                if isinstance(request, ModelRequest):
                    request.parts.append(
                        UserPromptPart(content=pending_inject),
                    )
                    pending_inject = None
            elif elapsed >= next_reflect_at:
                _emit_event(event_sink, {
                    "type": "self_review_start",
                    "elapsed_ms": elapsed * 1000.0,
                })
                verdict = reflector.review(
                    prompt, tuple(tool_log), elapsed,
                    active_budget.task_class,
                )
                _emit_event(event_sink, {
                    "type": "self_review_done",
                    "continue": bool(verdict.continue_research),
                    "reason": verdict.reason,
                    "suggested_class": verdict.suggested_class,
                    "elapsed_ms": elapsed * 1000.0,
                })
                if not verdict.continue_research:
                    pending_inject = _STOP_REQUEST_BY_REVIEW
                    request = getattr(next_node, "request", None)
                    if isinstance(request, ModelRequest):
                        request.parts.append(
                            UserPromptPart(content=pending_inject),
                        )
                        pending_inject = None
                else:
                    # Only honour promotions FAST -> DEEP. Demotions
                    # back to FAST and cross-class promotions to
                    # BACKGROUND_DEEP (caller-only) are ignored.
                    promoted = _maybe_promote_class(
                        active_budget.task_class,
                        verdict.suggested_class,
                    )
                    if promoted is not None:
                        active_budget = TaskBudget.for_class(promoted)
                        _emit_event(event_sink, {
                            "type": "extended_research_announced",
                            "reason": verdict.reason,
                            "task_class": promoted.value,
                            "expected_total_ms": (
                                active_budget.expected_total_s * 1000.0
                            ),
                        })
                    next_reflect_at = (
                        elapsed + active_budget.reflect_interval_s
                    )

            next_node = await agent_run.next(next_node)
        return agent_run.result


def _maybe_promote_class(current: Any, suggested: str) -> Any | None:
    """Return the promoted class, or ``None`` if no promotion is allowed.

    Only ``INTERACTIVE_FAST → INTERACTIVE_DEEP`` is honoured here. The
    ``BACKGROUND_DEEP`` class is reserved for explicit caller opt-in
    (daily brief, weekly digest) — the model can't elect into it
    mid-run, and a ``BACKGROUND_DEEP`` run can't be downgraded by
    review either.

    sensitivity_tier: 1
    """
    from src.agents.core.task_budget import TaskClass
    if current is TaskClass.INTERACTIVE_FAST and (
        suggested == TaskClass.INTERACTIVE_DEEP.value
    ):
        return TaskClass.INTERACTIVE_DEEP
    return None


def _run_pa_with_reflection_sync(
    pa_agent: Any,
    prompt: str,
    *,
    budget: Any,
    run_id: str,
    reflector: Any,
    event_sink: Callable[[dict[str, Any]], None] | None,
    tool_log: list[Any],
) -> Any:
    """Synchronous wrapper around the async reflective runner.

    Mirrors :func:`_run_pa_sync` — submits to ``_PA_EXECUTOR`` when a
    loop is already bound to the current thread, otherwise runs the
    coroutine directly.

    sensitivity_tier: 1
    """
    async def _run() -> Any:
        return await _run_pa_with_reflection_async(
            pa_agent, prompt,
            budget=budget,
            run_id=run_id,
            reflector=reflector,
            event_sink=event_sink,
            tool_log=tool_log,
        )

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_run())
    return _PA_EXECUTOR.submit(lambda: asyncio.run(_run())).result()


def _model_settings_for(model: Any) -> dict[str, Any] | None:
    """Per-model pydantic-ai settings to disable hidden reasoning chains.

    Qwen3 ships with "thinking" mode on by default and emits a hidden
    chain-of-thought before the JSON output. For classifier / judge
    workloads that costs 3-5× latency for no measurable quality gain
    (the rules these agents apply don't benefit from CoT). Pass
    ``chat_template_kwargs={"enable_thinking": False}`` via ``extra_body``
    so OpenAI-compatible hosts (vLLM, etc.) emit straight JSON.

    Returns ``None`` when no special settings apply — the caller skips
    the ``model_settings`` kwarg so reasoning models (R-class etc.)
    keep their defaults.

    sensitivity_tier: 1
    """
    model_name = getattr(model, "model_name", None) or ""
    if isinstance(model_name, str) and model_name.lower().startswith("qwen"):
        return {
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": False},
            },
        }
    return None


# (provider, model) pairs we've already warned about for missing
def _record_pa_spend(
    *,
    result: Any,
    route: str,
    agent_id: str,
    model_override: str | None = None,
) -> None:
    """No-op in Arandu — local Ollama inference is free.

    Per-call spend recording is a reserved extension point; it can be
    restored by overriding this function or wrapping the provider.

    sensitivity_tier: 1
    """
    return


def _record_run_log(
    *,
    agent_id: str,
    prompt: str | None,
    output: Any,
    duration_ms: float | None,
    route: str | None,
    status: str,
    error: str | None = None,
) -> None:
    """Best-effort append to the per-agent run log.

    Failures here must never abort an otherwise-successful agent run,
    so any exception is logged at debug and swallowed.

    sensitivity_tier: varies
    """
    if not agent_id:
        return
    try:
        from src.agents.core.run_log import default_run_log
        default_run_log().record(
            agent_id=agent_id,
            input=prompt,
            output=output,
            duration_ms=duration_ms,
            route=route,
            status=status,
            error=error,
        )
    except Exception:  # noqa: BLE001
        logger.debug("agent run-log append failed", exc_info=True)


def _agent_max_tier(agent_id: str) -> int:
    """Read the registered ``max_sensitivity_tier`` for ``agent_id``.

    Falls back to ``3`` (the safest tier) when the agent is not in the
    registry — that pins the prompt to local Ollama under the balanced
    policy until the agent is properly registered.

    sensitivity_tier: 1
    """
    try:
        from src.agents.core.registry import get_agent
    except ImportError:  # pragma: no cover
        return 3
    definition = get_agent(agent_id) if agent_id else None
    if definition is None:
        return 3
    return int(getattr(definition, "max_sensitivity_tier", 3))


def _firewall_route_for_agent(
    *,
    agent_id: str,
    prompt: str,
    agent_max_tier: int,
) -> str:
    """Run the injection + egress firewalls and return the chosen route.

    Used by :meth:`SBAgent.run` when the caller did not pre-resolve a
    route. The injection check raises a :class:`RuntimeError` on a
    rejected prompt — :meth:`SBAgent.run` catches and records it as an
    agent error so the calling code sees a clean :class:`AgentRunRecord`
    rather than a stack trace.

    Returns ``"blocked"`` when the egress firewall refuses the call;
    the caller short-circuits without acquiring a scheduler permit.

    sensitivity_tier: 1
    """
    try:
        from src.agents.firewall.egress_firewall import (
            default_egress_firewall,
        )
        from src.agents.firewall.injection_firewall import (
            InjectionRejected,
            default_injection_firewall,
        )
    except ImportError:  # pragma: no cover — defensive
        return "remote"

    try:
        default_injection_firewall().assert_allowed(
            prompt, calling_agent_id=agent_id,
        )
    except InjectionRejected as exc:
        msg = (
            f"prompt rejected by injection firewall: "
            f"{exc.verdict.reason}"
        )
        raise RuntimeError(msg) from exc

    egress = default_egress_firewall().classify(
        prompt,
        calling_agent_id=agent_id,
        agent_max_tier=agent_max_tier,
    )
    return egress.route


OutT = TypeVar("OutT", bound=AgentOutput)
DepsT = TypeVar("DepsT")


# ---------------------------------------------------------------------------
# Run record
# ---------------------------------------------------------------------------


@dataclass
class AgentRunRecord:
    """Result envelope shared by all three agent patterns.

    sensitivity_tier: varies
    """

    agent_id: str
    output: AgentOutput | None
    duration_ms: float
    llm_calls: int
    error: str | None = None
    audit_hashes: list[str] = field(default_factory=list)
    # Deep-agent specific fields. Empty for SBAgent / SBOrchestrator.
    plan: Plan | None = None
    steps: list[DeepAgentStep] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Single-workflow agent
# ---------------------------------------------------------------------------


class SBAgent(Generic[DepsT, OutT]):
    """Single-LLM-call agent with typed structured output.

    Subclasses set:

    - ``agent_id`` — stable id matching the registry
    - ``output_type`` — a ``BaseModel`` subclass from ``output_types``
    - ``tier`` — scheduler tier (default ``BACKGROUND``)
    - ``system_prompt`` — default system prompt (overridden by
      ``AgentConfigStore`` at resolve time)

    Subclasses MUST override :meth:`build_prompt` to convert their typed
    deps into a user-message string. Tool registration happens in
    :meth:`register_tools` (default: no tools).

    The actual ``pydantic_ai.Agent`` instance is built lazily on first
    :meth:`run` call so tests can construct a subclass without
    pydantic-ai installed.

    sensitivity_tier: varies
    """

    agent_id: str = ""
    output_type: type[OutT] | None = None
    tier: Tier = Tier.BACKGROUND
    system_prompt: str = ""

    def __init__(self, *, deps: DepsT | None = None) -> None:
        self._deps = deps
        self._pa_agent: Any | None = None

    # ----- subclass hooks ---------------------------------------------

    def build_prompt(self, deps: DepsT) -> str:
        """Return the user-message body for this run.

        Subclasses override to project their typed deps into prose
        + structured-data hints. Default returns ``str(deps)``.

        sensitivity_tier: varies
        """
        return str(deps)

    def register_tools(self, pa_agent: Any) -> None:
        """Hook to attach pydantic-ai tools. Default: no tools.

        Subclasses (typically ``SBOrchestrator``) register here.

        sensitivity_tier: 1
        """
        return None

    # ----- public API -------------------------------------------------

    def run(
        self,
        deps: DepsT,
        *,
        route: Route | None = None,
        budget: Any | None = None,
    ) -> AgentRunRecord:
        """Execute one LLM call and return the structured output.

        When ``route`` is ``None`` the egress firewall classifies the
        prompt and picks the destination — this is the path pipeline
        agents like :class:`TopicExtractorAgent` should take so they
        cannot leak Tier 3 content even when invoked from code that
        forgot to pre-classify. Callers (notably ``BrainV2``) that have
        already run :meth:`EgressFirewall.classify` pass the result
        through explicitly.

        ``budget`` enables the reflective runner: a wall-clock budget
        + Tier A self-review checkpoint + user-cancel channel. Only
        meaningful for tool-using orchestrators (Brain, Chat) — single-
        call agents have no loop to reflect on. When ``None`` (the
        default), the agent is executed via plain ``run_sync`` with no
        budget machinery, matching the legacy behaviour.

        sensitivity_tier: varies
        """
        start = time.monotonic()
        scheduler = default_scheduler()
        chain = default_chain()
        prompt = self.build_prompt(deps)
        audit_hashes: list[str] = []

        if route is None:
            route = _firewall_route_for_agent(
                agent_id=self.agent_id,
                prompt=prompt,
                agent_max_tier=_agent_max_tier(self.agent_id),
            )
        if route == "blocked":
            duration_ms = (time.monotonic() - start) * 1000.0
            audit_hashes.append(
                chain.append(
                    event_type="agent_run",
                    agent_id=self.agent_id,
                    decision="blocked",
                    payload_hash=hash_payload(prompt),
                    extra={"reason": "egress firewall blocked"},
                ),
            )
            return AgentRunRecord(
                agent_id=self.agent_id,
                output=None,
                duration_ms=duration_ms,
                llm_calls=0,
                error="egress firewall blocked the call",
                audit_hashes=audit_hashes,
            )

        with scheduler.acquire(self.tier, route=route, agent_id=self.agent_id):
            try:
                pa_agent = self._get_pa_agent(route=route)
                if budget is not None:
                    run_id = getattr(self, "_stream_run_id", None) or (
                        uuid.uuid4().hex[:12]
                    )
                    tool_log = getattr(self, "_stream_tool_log", None)
                    if tool_log is None:
                        tool_log = []
                        self._stream_tool_log = tool_log
                    reflector = getattr(self, "_reflector", None)
                    if reflector is None:
                        from src.agents.core.reflection import Reflector
                        reflector = Reflector()
                        self._reflector = reflector
                    sink = getattr(self, "_stream_event_sink", None)
                    result = _run_pa_with_reflection_sync(
                        pa_agent, prompt,
                        budget=budget,
                        run_id=run_id,
                        reflector=reflector,
                        event_sink=sink,
                        tool_log=tool_log,
                    )
                else:
                    result = _run_pa_sync(pa_agent, prompt)
                output = result.output if hasattr(result, "output") else result
                _record_pa_spend(
                    result=result,
                    route=route,
                    agent_id=self.agent_id,
                    model_override=current_model_override(self.agent_id),
                )
            except Exception as exc:  # noqa: BLE001
                duration_ms = (time.monotonic() - start) * 1000.0
                audit_hashes.append(
                    chain.append(
                        event_type="agent_run",
                        agent_id=self.agent_id,
                        decision="error",
                        payload_hash=hash_payload(prompt),
                        extra={"error": type(exc).__name__},
                    ),
                )
                logger.exception(
                    "SBAgent %s failed", self.agent_id,
                )
                _record_run_log(
                    agent_id=self.agent_id,
                    prompt=prompt,
                    output=None,
                    duration_ms=duration_ms,
                    route=route,
                    status="error",
                    error=str(exc),
                )
                return AgentRunRecord(
                    agent_id=self.agent_id,
                    output=None,
                    duration_ms=duration_ms,
                    llm_calls=1,
                    error=str(exc),
                    audit_hashes=audit_hashes,
                )

        duration_ms = (time.monotonic() - start) * 1000.0
        audit_hashes.append(
            chain.append(
                event_type="agent_run",
                agent_id=self.agent_id,
                decision="ok",
                payload_hash=hash_payload(prompt),
            ),
        )
        _record_run_log(
            agent_id=self.agent_id,
            prompt=prompt,
            output=output,
            duration_ms=duration_ms,
            route=route,
            status="ok",
        )
        return AgentRunRecord(
            agent_id=self.agent_id,
            output=output,
            duration_ms=duration_ms,
            llm_calls=1,
            audit_hashes=audit_hashes,
        )

    # ----- internals --------------------------------------------------

    def _resolve_system_prompt(self) -> str:
        """Build the effective system prompt, injecting user context for
        interactive orchestrators (chat, brain).

        sensitivity_tier: 2
        """
        if self.agent_id not in ("chat", "brain"):
            return self.system_prompt
        try:
            from src.core.user_context import build_user_context
            prompt = f"{self.system_prompt}\n\n{build_user_context()}"
        except Exception:  # noqa: BLE001
            return self.system_prompt
        try:
            from src.core.sqlite.engine import DatabaseEngine
            from src.core.user_context import (
                build_active_topics_context,
                build_learned_facts_context,
            )
            db = DatabaseEngine()
            facts = build_learned_facts_context(db)
            if facts:
                prompt += f"\n\n{facts}"
            topics = build_active_topics_context(db)
            if topics:
                prompt += f"\n\n{topics}"
        except Exception:  # noqa: BLE001
            pass
        try:
            from src.agent_runtime.skill_loader import build_skill_menu
            skill_menu = build_skill_menu(max_tier=2)
            if skill_menu:
                prompt += f"\n\n{skill_menu}"
        except Exception:  # noqa: BLE001
            pass
        return prompt

    def _get_pa_agent(self, *, route: Route) -> Any:
        if self._pa_agent is not None:
            return self._pa_agent
        try:
            from pydantic_ai import Agent  # type: ignore
        except ImportError as exc:  # pragma: no cover
            msg = (
                "pydantic-ai-slim[openai] not installed; "
                f"cannot run agent {self.agent_id!r}"
            )
            raise RuntimeError(msg) from exc
        if self.output_type is None:
            msg = f"Agent {self.agent_id!r} must set output_type"
            raise TypeError(msg)
        override = current_model_override(self.agent_id) if self.agent_id else None
        model = default_factory().get(route, model_override=override)
        effective_prompt = self._resolve_system_prompt()
        agent_kwargs: dict[str, Any] = {
            "model": model,
            "output_type": self.output_type,
            "system_prompt": effective_prompt,
        }
        settings = _model_settings_for(model)
        if settings:
            agent_kwargs["model_settings"] = settings
        agent = Agent(**agent_kwargs)
        self.register_tools(agent)
        self._pa_agent = agent
        return agent


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class SBOrchestrator(SBAgent[DepsT, OutT]):
    """Agent whose tools are *delegations* to other registered agents.

    Subclasses set ``subagents`` to a tuple of ``agent_id`` values. At
    construction time the orchestrator registers a pydantic-ai tool per
    sub-agent. The LLM decides which sub-agents to invoke and in what
    order to assemble its structured output.

    Brain, Proactive Intelligence, and Reply Handler are the three
    intended orchestrators in this codebase.

    sensitivity_tier: varies
    """

    subagents: tuple[str, ...] = ()

    def _event_sink(self) -> Callable[[dict[str, Any]], None] | None:
        """Stream-event sink for tool wrappers. ``None`` disables events.

        Streaming hosts (``BrainAgentV2.ask_stream`` /
        ``ChatAgent.ask_stream``) override this for the duration of a
        run by setting ``self._stream_event_sink``.

        sensitivity_tier: 1
        """
        return getattr(self, "_stream_event_sink", None)

    def _tool_log_sink(self) -> list[Any] | None:
        """Optional list that captures :class:`ToolCallEntry` per tool.

        Bound by reflective-runner hosts (Brain/Chat ``ask`` /
        ``ask_stream``) before invoking :meth:`run` so the Reflector
        sees the live tool log at each checkpoint. ``None`` outside
        reflective runs; tool wrappers skip appending in that case.

        sensitivity_tier: 1
        """
        return getattr(self, "_stream_tool_log", None)

    def register_tools(self, pa_agent: Any) -> None:
        """Attach one tool per sub-agent id.

        Each tool's docstring uses the sub-agent's registry description
        so the parent LLM can choose intelligently. The tool body looks
        up the sub-agent at call time (lazy) so child registrations
        don't have to precede the orchestrator's construction.

        sensitivity_tier: 1
        """
        from src.agents.core.registry import get_agent

        host = self

        for sub_id in self.subagents:
            definition = get_agent(sub_id)
            description = (
                definition.description if definition else
                f"Invoke sub-agent {sub_id}"
            )

            def _make_tool(child_id: str, desc: str) -> Any:
                async def _delegate(_ctx: RunContext[None], prompt: str) -> str:
                    """Delegate to child agent."""
                    sink = host._event_sink()
                    call_id = uuid.uuid4().hex[:8]
                    if sink is not None:
                        try:
                            sink({
                                "type": "tool_call_start",
                                "call_id": call_id,
                                "name": f"delegate_{child_id}",
                                "args_summary": _truncate_summary(prompt),
                            })
                        except Exception:  # noqa: BLE001
                            logger.debug(
                                "delegate sink failed", exc_info=True,
                            )
                    start = time.monotonic()
                    child = get_agent(child_id)
                    if child is None or child.factory is None:
                        result_text = (
                            f"sub-agent {child_id} unavailable"
                        )
                    else:
                        instance = child.factory()
                        # Re-use the child's typed deps shape: a single
                        # string prompt is the minimum contract.
                        record = instance.run(prompt)
                        if record.output is None:
                            result_text = (
                                f"sub-agent {child_id} returned no output"
                            )
                        else:
                            result_text = record.output.model_dump_json()
                    duration_ms = (time.monotonic() - start) * 1000.0
                    result_summary = _truncate_summary(result_text, 120)
                    if sink is not None:
                        try:
                            sink({
                                "type": "tool_call_done",
                                "call_id": call_id,
                                "name": f"delegate_{child_id}",
                                "duration_ms": duration_ms,
                                "status": "ok",
                                "result_summary": result_summary,
                            })
                        except Exception:  # noqa: BLE001
                            logger.debug(
                                "delegate sink failed", exc_info=True,
                            )
                    tool_log = host._tool_log_sink()
                    if tool_log is not None:
                        from src.agents.core.reflection import ToolCallEntry
                        tool_log.append(ToolCallEntry(
                            name=f"delegate_{child_id}",
                            args_summary=_truncate_summary(prompt),
                            result_summary=result_summary,
                            duration_ms=duration_ms,
                            status="ok",
                        ))
                    return result_text

                _delegate.__doc__ = desc
                return _delegate

            tool_fn = _make_tool(sub_id, description)
            tool_fn.__name__ = f"delegate_{sub_id.replace('-', '_')}"
            try:
                pa_agent.tool(tool_fn)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Could not attach delegation tool for %s",
                    sub_id, exc_info=True,
                )


# ---------------------------------------------------------------------------
# Deep agent
# ---------------------------------------------------------------------------


@dataclass
class DeepAgentStep:
    """One step recorded during a deep-agent run.

    sensitivity_tier: 1
    """

    step_id: str
    tool: str
    args: dict[str, Any]
    ok: bool
    duration_ms: float
    summary: str = ""


class SBDeepAgent(SBAgent[DepsT, OutT]):
    """Autonomous agent with planning, workspace, and sandboxed code.

    The base class provides:

    - A per-run workspace under ``~/.arandu/data/deep_agents/{agent_id}/
      {run_id}/workspace/`` accessed via the workspace tools.
    - Plan / update_plan tools backed by an internal ``Plan`` model.
    - ``execute_code`` / ``execute_sql`` tools backed by
      :mod:`src.agents.core.sandbox`. Off by default; subclasses opt in
      via ``allow_code_execution``.
    - ``delegate`` tool that runs another registered agent.

    Subclasses set ``max_steps`` to cap the number of tool calls inside a
    single run (default 50). Once exceeded, the run halts and returns
    whatever output the LLM produced last.

    sensitivity_tier: varies
    """

    max_steps: int = 50
    allowed_subagents: tuple[str, ...] = ()
    allow_code_execution: bool = False
    allow_sql: bool = True

    def run(  # type: ignore[override]
        self, deps: DepsT, *, route: Route = "remote",
    ) -> AgentRunRecord:
        """Execute a deep-agent run with plan + tool loop.

        sensitivity_tier: varies
        """
        start = time.monotonic()
        scheduler = default_scheduler()
        chain = default_chain()
        run_id = uuid.uuid4().hex[:12]
        workspace = workspace_for(self.agent_id, run_id)
        plan = Plan(goal=self.build_prompt(deps), steps=[], revision=0)
        steps: list[DeepAgentStep] = []
        audit_hashes: list[str] = []

        with scheduler.acquire(
            self.tier, route=route, agent_id=self.agent_id,
        ):
            try:
                pa_agent = self._get_deep_pa_agent(
                    route=route, workspace=workspace,
                    plan_ref={"plan": plan, "steps": steps},
                )
                result = _run_pa_sync(pa_agent, self.build_prompt(deps))
                output = result.output if hasattr(result, "output") else result
                _record_pa_spend(
                    result=result,
                    route=route,
                    agent_id=self.agent_id,
                    model_override=current_model_override(self.agent_id),
                )
                error: str | None = None
            except Exception as exc:  # noqa: BLE001
                logger.exception("SBDeepAgent %s failed", self.agent_id)
                output = None
                error = str(exc)

        audit_hashes.append(
            chain.append(
                event_type="deep_agent_run",
                agent_id=self.agent_id,
                decision="error" if error else "ok",
                payload_hash=hash_payload(self.build_prompt(deps)),
                extra={"run_id": run_id, "steps": len(steps)},
            ),
        )
        duration_ms = (time.monotonic() - start) * 1000.0
        _record_run_log(
            agent_id=self.agent_id,
            prompt=plan.goal,
            output=output,
            duration_ms=duration_ms,
            route=route,
            status="error" if error else "ok",
            error=error,
        )
        return AgentRunRecord(
            agent_id=self.agent_id,
            output=output,
            duration_ms=duration_ms,
            llm_calls=1 + len(steps),
            error=error,
            audit_hashes=audit_hashes,
            plan=plan,
            steps=steps,
        )

    def _get_deep_pa_agent(
        self,
        *,
        route: Route,
        workspace: Any,
        plan_ref: dict[str, Any],
    ) -> Any:
        try:
            from pydantic_ai import Agent  # type: ignore
        except ImportError as exc:  # pragma: no cover
            msg = "pydantic-ai-slim[openai] not installed"
            raise RuntimeError(msg) from exc
        if self.output_type is None:
            msg = f"Deep agent {self.agent_id!r} must set output_type"
            raise TypeError(msg)
        override = current_model_override(self.agent_id) if self.agent_id else None
        model = default_factory().get(route, model_override=override)
        agent_kwargs: dict[str, Any] = {
            "model": model,
            "output_type": self.output_type,
            "system_prompt": self.system_prompt,
        }
        settings = _model_settings_for(model)
        if settings:
            agent_kwargs["model_settings"] = settings
        agent = Agent(**agent_kwargs)
        _attach_deep_tools(
            agent,
            workspace=workspace,
            plan_ref=plan_ref,
            allowed_subagents=self.allowed_subagents,
            allow_code=self.allow_code_execution,
            allow_sql=self.allow_sql,
            sql_db_path=_default_sql_db_path(),
        )
        return agent


# ---------------------------------------------------------------------------
# Deep-agent tool wiring
# ---------------------------------------------------------------------------


def _default_sql_db_path() -> str:
    """Path to the read-only SQLite database deep agents may query.

    sensitivity_tier: 1
    """
    from pathlib import Path

    return str(Path.home() / ".arandu" / "data" / "arandu.sqlite3")


def _attach_deep_tools(
    pa_agent: Any,
    *,
    workspace: Any,
    plan_ref: dict[str, Any],
    allowed_subagents: Sequence[str],
    allow_code: bool,
    allow_sql: bool,
    sql_db_path: str,
) -> None:
    """Register the standard deep-agent tools on ``pa_agent``.

    Tools registered:

    - ``write_plan(goal, steps)`` — replace the plan scratchpad.
    - ``update_plan_step(step_id, status, notes)`` — mutate one step.
    - ``read_workspace(path)`` / ``write_workspace(path, content)`` /
      ``list_workspace()`` — file ops scoped to the run workspace.
    - ``delegate(subagent_id, prompt)`` — invoke another registered
      agent if its id is in ``allowed_subagents``.
    - ``execute_python(source)`` — only if ``allow_code`` is True.
    - ``execute_sql(query)`` — only if ``allow_sql`` is True.

    sensitivity_tier: 1
    """

    async def write_plan(
        _ctx: RunContext[None], goal: str, steps: list[dict[str, str]],
    ) -> str:
        """Replace the plan with a new goal and step list."""
        plan = Plan(
            goal=goal,
            steps=[
                PlanStep(
                    id=str(s.get("id", uuid.uuid4().hex[:8])),
                    description=str(s.get("description", "")),
                    status="pending",
                )
                for s in steps
            ],
            revision=plan_ref["plan"].revision + 1,
        )
        plan_ref["plan"] = plan
        return f"plan revised: {len(plan.steps)} steps"

    async def update_plan_step(
        _ctx: RunContext[None], step_id: str, status: str, notes: str = "",
    ) -> str:
        """Update one plan step's status and notes."""
        plan: Plan = plan_ref["plan"]
        valid = {"pending", "in_progress", "completed", "blocked"}
        if status not in valid:
            return f"invalid status: {status}"
        new_steps = []
        for step in plan.steps:
            if step.id == step_id:
                new_steps.append(PlanStep(
                    id=step.id, description=step.description,
                    status=status, notes=notes or step.notes,
                ))
            else:
                new_steps.append(step)
        plan_ref["plan"] = Plan(
            goal=plan.goal, steps=new_steps, revision=plan.revision + 1,
        )
        return f"step {step_id} -> {status}"

    async def write_workspace(_ctx: RunContext[None], path: str, content: str) -> str:
        """Write content to a file inside the workspace."""
        try:
            target = resolve_in_workspace(workspace, path)
        except WorkspaceError as exc:
            return f"rejected: {exc}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        plan_ref["steps"].append(DeepAgentStep(
            step_id=uuid.uuid4().hex[:8],
            tool="write_workspace",
            args={"path": path, "bytes": len(content)},
            ok=True, duration_ms=0.0,
            summary=f"wrote {len(content)} bytes to {path}",
        ))
        return f"wrote {len(content)} bytes"

    async def read_workspace(_ctx: RunContext[None], path: str) -> str:
        """Read a file from the workspace."""
        try:
            target = resolve_in_workspace(workspace, path)
        except WorkspaceError as exc:
            return f"rejected: {exc}"
        if not target.exists():
            return f"not found: {path}"
        return target.read_text(encoding="utf-8")

    async def list_workspace(_ctx: RunContext[None]) -> list[str]:
        """List files in the workspace (relative paths)."""
        return [str(p.relative_to(workspace)) for p in workspace.rglob("*")
                if p.is_file()]

    async def delegate(
        _ctx: RunContext[None], subagent_id: str, prompt: str,
    ) -> str:
        """Run another registered agent and return its JSON output."""
        if subagent_id not in allowed_subagents:
            return f"delegation not allowed for {subagent_id}"
        from src.agents.core.registry import get_agent
        definition = get_agent(subagent_id)
        if definition is None or definition.factory is None:
            return f"unknown sub-agent {subagent_id}"
        record = definition.factory().run(prompt)
        plan_ref["steps"].append(DeepAgentStep(
            step_id=uuid.uuid4().hex[:8],
            tool="delegate",
            args={"subagent_id": subagent_id},
            ok=record.error is None,
            duration_ms=record.duration_ms,
            summary=record.error or "delegated",
        ))
        if record.output is None:
            return record.error or "no output"
        return record.output.model_dump_json()

    pa_agent.tool(write_plan)
    pa_agent.tool(update_plan_step)
    pa_agent.tool(write_workspace)
    pa_agent.tool(read_workspace)
    pa_agent.tool(list_workspace)
    pa_agent.tool(delegate)

    if allow_code:
        async def execute_python(_ctx: RunContext[None], source: str) -> dict[str, Any]:
            """Run a short Python snippet in the sandbox."""
            res: SandboxResult = run_python(source)
            plan_ref["steps"].append(DeepAgentStep(
                step_id=uuid.uuid4().hex[:8],
                tool="execute_python",
                args={"len": len(source)},
                ok=res.ok,
                duration_ms=res.duration_ms,
                summary=res.exit_reason,
            ))
            return {
                "ok": res.ok, "stdout": res.stdout,
                "stderr": res.stderr, "exit": res.exit_reason,
            }

        pa_agent.tool(execute_python)

    if allow_sql:
        async def execute_sql(_ctx: RunContext[None], query: str) -> dict[str, Any]:
            """Run a read-only SELECT against the user's SQLite DB."""
            res = run_sql(query, db_path=sql_db_path)
            plan_ref["steps"].append(DeepAgentStep(
                step_id=uuid.uuid4().hex[:8],
                tool="execute_sql",
                args={"len": len(query)},
                ok=res.ok,
                duration_ms=res.duration_ms,
                summary=res.exit_reason,
            ))
            return {
                "ok": res.ok,
                "rows": res.rows or [],
                "error": res.stderr if not res.ok else "",
            }

        pa_agent.tool(execute_sql)


__all__ = [
    "AgentRunRecord",
    "DeepAgentStep",
    "SBAgent",
    "SBDeepAgent",
    "SBOrchestrator",
]
