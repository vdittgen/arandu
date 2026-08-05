"""Task adapters that bridge YAML cases to the agent surface.

Each task takes the raw ``inputs`` value from a dataset case and
returns the same structured output the eval evaluators expect. Tasks
are sync — pydantic-evals' ``Dataset.evaluate_sync`` runs them on a
worker thread per case so the scheduler can still apply concurrency
caps in production-like fashion.

When an agent's model can't be constructed (no remote endpoint
configured, pydantic-ai missing), the wrapper raises
:class:`ModelUnavailableError` so the eval runner translates it into
a ``skipped`` row rather than a confusing all-cases-failed verdict.

sensitivity_tier: N/A
"""

from __future__ import annotations

import json
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

from src.agents.actionable_events import ActionableEventsAgent
from src.agents.contact_context import ContactContextAgent
from src.agents.dataset_validator import DatasetValidatorAgent
from src.agents.event_categorizer import EventCategorizerAgent
from src.agents.fact_extractor import FactExtractorAgent
from src.agents.firewall.egress_firewall import default_egress_firewall
from src.agents.firewall.injection_firewall import (
    InjectionFirewall,
    default_injection_firewall,
)
from src.agents.goal_extractor import GoalExtractorAgent
from src.agents.habit_suggester import HabitSuggesterAgent
from src.agents.labeler import LabelerAgent
from src.agents.message_eval import MessageEvaluatorAgent
from src.agents.model_generator import ModelGeneratorAgent
from src.agents.pending_reply import PendingReplyAgent
from src.agents.query_router import QueryRouterAgent
from src.agents.relationship_tracker import RelationshipTrackerAgent
from src.agents.schema_discovery import SchemaDiscoveryAgent
from src.agents.sensitivity import SensitivityAgent
from src.agents.task_proposer import TaskProposerAgent
from src.agents.topic_extractor import TopicExtractorAgent
from src.agents.triage import TriageAgent, TriageMessage

# Per-case sink for orchestrator delegations, set by
# :func:`task_curator_task` around each case. A ContextVar (not a plain
# list) because ``Dataset.evaluate_sync`` runs cases concurrently and
# each needs its own record of which sub-agent the router chose.
_CURATOR_DELEGATIONS: ContextVar[list[str] | None] = ContextVar(
    "_CURATOR_DELEGATIONS", default=None,
)


class ModelUnavailableError(RuntimeError):
    """Raised when an eval task can't run because no LLM is reachable.

    The eval runner translates this into a ``skipped`` row so the UI
    distinguishes "needs a model" from "evaluator failed".
    """


def _to_triage_message(item: dict[str, Any]) -> TriageMessage:
    return TriageMessage(
        message_id=str(item.get("message_id") or item.get("id") or ""),
        content=str(item.get("content") or ""),
        sender_name=str(item.get("sender_name") or ""),
        source=str(item.get("source") or ""),
    )


# ---------------------------------------------------------------------------
# Brain + locked agents
# ---------------------------------------------------------------------------


def brain_task() -> Callable[[Any], Any]:
    """Run a question through Brain v2 with a hermetic query fixture.

    Each YAML case supplies its retrieval context inline under
    ``inputs.fixture`` (see :class:`evals.fixtures.FakeQueryEngine`).
    The agent receives the user question, the orchestrator calls the
    fixture instead of a real ``QueryEngine``, and the resulting
    :class:`BrainResponse` is what the LLM judge grades.

    ``ModelUnavailableError`` is raised only when the underlying
    pydantic-ai model can't be constructed (no remote endpoint
    configured), so the run records as ``skipped`` rather than
    failing every case.

    sensitivity_tier: N/A
    """
    from src.agents.brain.v2 import BrainAgentV2

    def task(inputs: Any) -> Any:
        from evals.fixtures import FakeQueryEngine

        if isinstance(inputs, str):
            question = inputs
            fixture: Any = None
        elif isinstance(inputs, dict):
            question = str(inputs.get("question") or "")
            fixture = inputs.get("fixture")
        else:
            raise ModelUnavailableError(
                f"brain task: unsupported inputs shape {type(inputs).__name__}",
            )
        if not question:
            raise ModelUnavailableError("brain task: empty question")
        agent = BrainAgentV2(query_engine=FakeQueryEngine(fixture))
        try:
            response = agent.ask(question)
        except RuntimeError as exc:
            if "pydantic-ai" in str(exc).lower():
                raise ModelUnavailableError(str(exc)) from exc
            raise
        if response is None:
            raise ModelUnavailableError("brain returned no response")
        return response

    return task


def injection_firewall_task(
    firewall: InjectionFirewall | None = None,
) -> Callable[[str], Any]:
    """Run a prompt through the injection firewall (heuristic only).

    sensitivity_tier: N/A
    """
    fw = firewall or default_injection_firewall()

    def task(prompt: str) -> Any:
        return fw.scan(prompt, calling_agent_id="evals.firewall")

    return task


def injection_scan_task() -> Callable[[Any], Any]:
    """Run a prompt (and optional context) through the LLM-judge scanner.

    Exercises the *semantic* pass of the injection firewall directly
    via :func:`run_injection_scan`, bypassing the regex heuristic
    layer. The dataset cases bypass that layer by construction, so
    every grading signal here belongs to the LLM judge.

    Inputs accept either a plain string (``prompt`` only) or a dict
    with ``prompt`` and ``context`` keys.

    ``ModelUnavailableError`` is raised when the local model stack is
    unreachable or the agent returns no verdict, so the eval runner
    records a clean ``skipped`` row rather than failing every case.

    sensitivity_tier: 1
    """
    from src.agents.firewall.injection_scan_agent import run_injection_scan

    def task(inputs: Any) -> Any:
        if isinstance(inputs, str):
            prompt = inputs
            context = ""
        elif isinstance(inputs, dict):
            prompt = str(inputs.get("prompt", ""))
            context = str(inputs.get("context", ""))
        else:
            raise ModelUnavailableError(
                f"injection_scan: unsupported inputs shape "
                f"{type(inputs).__name__}",
            )
        if not prompt:
            raise ModelUnavailableError("injection_scan: empty prompt")
        try:
            verdict = run_injection_scan(prompt=prompt, context=context)
        except RuntimeError as exc:
            if "pydantic-ai" in str(exc).lower():
                raise ModelUnavailableError(str(exc)) from exc
            raise
        if verdict is None:
            raise ModelUnavailableError("injection_scan returned no verdict")
        return verdict

    return task


def egress_firewall_task() -> Callable[[dict[str, Any]], Any]:
    """Run the egress firewall's classify() with YAML-supplied inputs.

    Input shape::

        {prompt: str, agent_max_tier: int, explicit_tier: int|null,
         calling_agent_id: str|null}

    sensitivity_tier: N/A
    """
    fw = default_egress_firewall()

    def task(inputs: dict[str, Any]) -> Any:
        return fw.classify(
            inputs.get("prompt", ""),
            calling_agent_id=str(inputs.get("calling_agent_id") or "evals"),
            agent_max_tier=int(inputs.get("agent_max_tier", 1)),
            explicit_tier=inputs.get("explicit_tier"),
        )

    return task


# ---------------------------------------------------------------------------
# Single-input string tasks (classifiers + authors)
# ---------------------------------------------------------------------------


def dataset_validator_task() -> Callable[[str], Any]:
    """Run the dataset validator on a YAML string.

    The dataset cases supply the YAML payload as ``inputs`` and the
    evaluator inspects the returned :class:`DatasetValidationReport`.
    The validator's structural pass is deterministic; we use the
    public ``validate`` helper so the eval is stable even when no LLM
    is reachable.

    sensitivity_tier: N/A
    """
    agent = DatasetValidatorAgent()

    def task(text: str) -> Any:
        return agent.validate(text)

    return task


def sensitivity_task() -> Callable[[str], Any]:
    """Classify a single string into a sensitivity tier.

    sensitivity_tier: N/A
    """
    agent = SensitivityAgent()

    def task(text: str) -> Any:
        return _safe_run(agent, text)

    return task


def labeler_task() -> Callable[[str], Any]:
    """Label a single text emotionally.

    sensitivity_tier: N/A
    """
    agent = LabelerAgent()

    def task(text: str) -> Any:
        return _safe_label(agent, text)

    return task


def fact_extractor_task() -> Callable[[str], Any]:
    """Extract a batch of fact drafts from a conversation block.

    sensitivity_tier: N/A
    """
    agent = FactExtractorAgent()

    def task(conversation: str) -> Any:
        return _safe_extract(agent, conversation)

    return task


def relationship_tracker_task() -> Callable[[str], Any]:
    """Write one relationship-nudge note from a contact context block.

    sensitivity_tier: N/A
    """
    agent = RelationshipTrackerAgent()

    def task(context: str) -> Any:
        return _safe_author(agent, context)

    return task


def query_router_task() -> Callable[[str], Any]:
    """Produce a retrieval plan for a user question.

    sensitivity_tier: N/A
    """
    agent = QueryRouterAgent()

    def task(question: str) -> Any:
        return _safe_plan(agent, question)

    return task


# ---------------------------------------------------------------------------
# Batch tasks (dict inputs)
# ---------------------------------------------------------------------------


def triage_task() -> Callable[[dict[str, Any]], Any]:
    """Triage a batch of messages.

    sensitivity_tier: N/A
    """
    agent = TriageAgent()

    def task(inputs: dict[str, Any]) -> Any:
        messages = [_to_triage_message(m) for m in inputs.get("messages", [])]
        out = agent.triage(messages)
        _raise_if_missing(out, "triage")
        return out

    return task


def message_eval_task() -> Callable[[dict[str, Any]], Any]:
    """Pick topic-relevant notifications from a batch of messages.

    sensitivity_tier: N/A
    """
    agent = MessageEvaluatorAgent()

    def task(inputs: dict[str, Any]) -> Any:
        out = agent.evaluate(
            messages=inputs.get("messages", []),
            topics=inputs.get("topics", {}),
            today_events=inputs.get("today_events", []),
            existing_pending_ids=inputs.get("existing_pending_ids", []),
        )
        _raise_if_missing(out, "message_evaluator")
        return out

    return task


def pending_reply_task() -> Callable[[dict[str, Any]], Any]:
    """Detect which messages need a user reply.

    sensitivity_tier: N/A
    """
    agent = PendingReplyAgent()

    def task(inputs: dict[str, Any]) -> Any:
        out = agent.detect(
            messages=inputs.get("messages", []),
            topics=inputs.get("topics", {}),
        )
        _raise_if_missing(out, "pending_reply")
        return out

    return task


def contact_context_task() -> Callable[[dict[str, Any]], Any]:
    """Summarise per-contact situations.

    sensitivity_tier: N/A
    """
    agent = ContactContextAgent()

    def task(inputs: dict[str, Any]) -> Any:
        out = agent.summarize(
            contacts=inputs.get("contacts", []),
            topics=inputs.get("topics", {}),
        )
        _raise_if_missing(out, "contact_context")
        return out

    return task


def actionable_events_task() -> Callable[[dict[str, Any]], Any]:
    """Pick calendar events that need user action.

    sensitivity_tier: N/A
    """
    agent = ActionableEventsAgent()

    def task(inputs: dict[str, Any]) -> Any:
        out = agent.detect(events=inputs.get("events", []))
        _raise_if_missing(out, "actionable_events")
        return out

    return task


def topic_extractor_task() -> Callable[[dict[str, Any]], Any]:
    """Extract topics from a contact's chat history.

    sensitivity_tier: N/A
    """
    agent = TopicExtractorAgent()

    def task(inputs: dict[str, Any]) -> Any:
        out = agent.extract(
            contact_name=str(inputs.get("contact_name", "")),
            messages_block=str(inputs.get("messages_block", "")),
        )
        _raise_if_missing(out, "topic_extractor")
        return out

    return task


def event_categorizer_task() -> Callable[[dict[str, Any]], Any]:
    """Categorise one calendar event into the closed mart vocabulary.

    Inputs accept the same shape as :class:`EventCategorizerDeps` —
    ``title`` is the only required field, the rest are optional.

    sensitivity_tier: N/A
    """
    agent = EventCategorizerAgent()

    def task(inputs: dict[str, Any]) -> Any:
        out = agent.categorize(
            title=str(inputs.get("title", "")),
            description=str(inputs.get("description", "")),
            location=str(inputs.get("location", "")),
            start_time=str(inputs.get("start_time", "")),
            attendees=str(inputs.get("attendees", "")),
            attendee_names=str(inputs.get("attendee_names", "")),
        )
        _raise_if_missing(out, "event_categorizer")
        return out

    return task


def schema_discovery_task() -> Callable[[dict[str, Any]], Any]:
    """Map sample records to SQLite columns.

    sensitivity_tier: N/A
    """
    agent = SchemaDiscoveryAgent()

    def task(inputs: dict[str, Any]) -> Any:
        out = agent.discover(
            tool_name=str(inputs.get("tool_name", "")),
            sample_records=inputs.get("sample_records", []),
            known_tables=inputs.get("known_tables", []),
        )
        _raise_if_missing(out, "schema_discovery")
        return out

    return task


def goal_extractor_task() -> Callable[[dict[str, Any]], Any]:
    """Mine goals from a mixed evidence bundle.

    Inputs accept the same shape as :class:`GoalExtractorDeps`:
    ``messages``, ``notes``, ``facts``, ``chat_excerpts``,
    ``known_topics``. All keys are optional; an empty bundle is
    rejected so the eval row records as ``skipped``.

    sensitivity_tier: N/A
    """
    agent = GoalExtractorAgent()

    def task(inputs: dict[str, Any]) -> Any:
        out = agent.extract(
            messages=inputs.get("messages", []),
            notes=inputs.get("notes", []),
            facts=inputs.get("facts", []),
            chat_excerpts=inputs.get("chat_excerpts", []),
            known_topics=inputs.get("known_topics", []),
        )
        _raise_if_missing(out, "goal_extractor")
        return out

    return task


def habit_suggester_task() -> Callable[[dict[str, Any]], Any]:
    """Suggest atomic habits for a list of active goals.

    Inputs accept the same shape as :class:`HabitSuggesterDeps`:
    ``goals`` (required), ``linked_topics`` (optional).

    sensitivity_tier: N/A
    """
    agent = HabitSuggesterAgent()

    def task(inputs: dict[str, Any]) -> Any:
        out = agent.suggest(
            goals=inputs.get("goals", []),
            linked_topics=inputs.get("linked_topics", []),
        )
        _raise_if_missing(out, "habit_suggester")
        return out

    return task


def task_proposer_task() -> Callable[[dict[str, Any]], Any]:
    """Propose tasks from a batch of recent messages.

    Inputs accept the same shape as :class:`TaskProposerDeps`:
    ``messages`` (required), ``topics`` and ``goals`` (optional).

    sensitivity_tier: N/A
    """
    agent = TaskProposerAgent()

    def task(inputs: dict[str, Any]) -> Any:
        out = agent.propose(
            messages=inputs.get("messages", []),
            topics=inputs.get("topics", []),
            goals=inputs.get("goals", []),
        )
        _raise_if_missing(out, "task_proposer")
        return out

    return task


def task_curator_task() -> Callable[[dict[str, Any]], Any]:
    """Route a task/goal/schedule request to the right sub-agent.

    This suite grades *routing*, not the sub-agents' own output, so the
    five children are swapped for stubs. Without that, one routing row
    fans out into a real goal-extractor / scheduler LLM call — slow,
    costly, and it would grade the child rather than the router.

    Each stub records its ``agent_id`` when invoked; the recorded list
    is returned alongside the curator's answer as
    ``{"answer": ..., "delegated": [...]}`` for
    :class:`~evals.evaluators.DelegatedTo`.

    Concurrency: ``Dataset.evaluate_sync`` runs cases in parallel, so
    the stubs are installed **once** here rather than swapped in and out
    around each case, and each case's delegations are collected through
    a :class:`~contextvars.ContextVar`. Swapping per case raced — one
    case restored the real definitions while another was mid-run, so
    delegations went unrecorded or landed in a neighbouring case's list
    and every row scored as "delegated to nobody".

    sensitivity_tier: N/A
    """
    import dataclasses

    from src.agents.core.registry import (
        get_agent,
        register_agent,
        unregister_agent,
    )
    from src.agents.daily_scheduler import register_daily_scheduler_agent
    from src.agents.goal_extractor import register_goal_extractor_agent
    from src.agents.habit_suggester import register_habit_suggester_agent
    from src.agents.task_completion import register_task_completion_agent
    from src.agents.task_curator import (
        TaskCuratorAgent,
        register_task_curator_agent,
    )
    from src.agents.task_proposer import register_task_proposer_agent

    # The registry is empty in a bare eval process — nothing imports
    # ``register_brain_v2`` here. Register just this orchestrator's
    # family (each call is idempotent) so the delegation tools resolve
    # to real definitions we can clone into stubs. Without this every
    # tool returns "sub-agent unavailable" and the router flails
    # against dead tools instead of being graded on its choice.
    register_goal_extractor_agent()
    register_task_proposer_agent()
    register_task_completion_agent()
    register_daily_scheduler_agent()
    register_habit_suggester_agent()
    register_task_curator_agent()

    agent = TaskCuratorAgent()

    def _stub_definition(original: Any, child_id: str) -> Any:
        """Clone ``original`` with a factory that only records the call.

        The delegation wrapper only calls ``record.output.model_dump_json()``
        on whatever the child returns, so a duck-typed stand-in is enough
        — and avoids having to satisfy each child's real output schema.
        """

        class _StubOutput:
            @staticmethod
            def model_dump_json() -> str:
                return json.dumps({"stub": child_id, "result": "ok"})

        class _StubRecord:
            error = None
            output = _StubOutput()

        class _Stub:
            agent_id = child_id

            def run(self, _prompt: Any) -> Any:
                sink = _CURATOR_DELEGATIONS.get()
                if sink is not None:
                    sink.append(child_id)
                return _StubRecord()

        return dataclasses.replace(original, factory=_Stub)

    for child_id in agent.subagents:
        existing = get_agent(child_id)
        if existing is not None:
            unregister_agent(child_id)
            register_agent(_stub_definition(existing, child_id))

    def task(inputs: dict[str, Any]) -> Any:
        delegated: list[str] = []
        token = _CURATOR_DELEGATIONS.set(delegated)
        try:
            record = agent.run(str(inputs.get("question", "")))
        finally:
            _CURATOR_DELEGATIONS.reset(token)

        _raise_if_missing(record.output, "task_curator")
        return {
            "answer": getattr(record.output, "answer", ""),
            "delegated": delegated,
        }

    return task


def model_generator_task() -> Callable[[dict[str, Any]], Any]:
    """Generate a SQLMesh model from a discovered schema.

    sensitivity_tier: N/A
    """
    agent = ModelGeneratorAgent()

    def task(inputs: dict[str, Any]) -> Any:
        out = agent.generate(
            schema=inputs.get("schema", {}),
            layer=inputs.get("layer", "staging"),
            connector_id=str(inputs.get("connector_id", "")),
        )
        _raise_if_missing(out, "model_generator")
        return out

    return task


# ---------------------------------------------------------------------------
# Helpers — translate construction failures into a clean "skipped"
# ---------------------------------------------------------------------------


def _safe_run(agent: Any, text: str) -> Any:
    try:
        record = agent.run(text)
    except RuntimeError as exc:
        if "pydantic-ai" in str(exc).lower():
            raise ModelUnavailableError(str(exc)) from exc
        raise
    if record.output is None:
        raise ModelUnavailableError(
            record.error or f"{agent.agent_id} produced no output",
        )
    return record.output


def _safe_label(agent: LabelerAgent, text: str) -> Any:
    out = agent.label(text)
    if out is None:
        raise ModelUnavailableError("labeler produced no output")
    return out


def _safe_author(agent: Any, prompt: str) -> Any:
    try:
        out = agent.author(prompt)
    except RuntimeError as exc:
        if "pydantic-ai" in str(exc).lower():
            raise ModelUnavailableError(str(exc)) from exc
        raise
    if out is None:
        raise ModelUnavailableError(f"{agent.agent_id} produced no output")
    return out


def _safe_extract(agent: FactExtractorAgent, conversation: str) -> Any:
    out = agent.extract(conversation)
    if out is None:
        raise ModelUnavailableError("fact_extractor produced no output")
    return out


def _safe_plan(agent: QueryRouterAgent, question: str) -> Any:
    out = agent.plan(question)
    if out is None:
        raise ModelUnavailableError("query_router produced no output")
    return out


def _raise_if_missing(out: Any, name: str) -> None:
    if out is None:
        raise ModelUnavailableError(f"{name} produced no output")


# ---------------------------------------------------------------------------
# Generic user-agent adapter
# ---------------------------------------------------------------------------


def user_agent_task(agent_id: str) -> Callable[[Any], Any]:
    """Run one user-authored agent on each case input.

    Resolves the registered :class:`AgentDefinition` from the global
    registry, constructs the agent via its factory, and dispatches the
    case input straight to :meth:`SBAgent.run`. The case input is
    passed through as the agent's deps — user agents accept a single
    string (see :class:`_UserAgent`), so the YAML's ``inputs`` field
    should be a string. Dicts and other non-string shapes are rejected
    early with a clear :class:`ModelUnavailableError` so the eval row
    records a clean ``skipped`` instead of a confusing crash.

    Output expansion: user agents emit :class:`BrainResponse`, whose
    ``answer`` field is a free-text string. Users routinely instruct
    the LLM to put structured JSON inside that string. To let dataset
    evaluators reference the parsed fields directly (e.g.
    ``FieldEquals`` on ``purchase_id``), :func:`_expand_brain_response`
    surfaces the JSON keys at the top level of the returned object
    when ``answer`` parses cleanly.

    sensitivity_tier: 1
    """
    from src.agents.core.registry import get_agent

    definition = get_agent(agent_id)
    if definition is None:
        raise ModelUnavailableError(
            f"user agent not registered: {agent_id}",
        )
    if definition.factory is None:
        raise ModelUnavailableError(
            f"user agent has no factory: {agent_id}",
        )
    factory = definition.factory

    def task(inputs: Any) -> Any:
        if not isinstance(inputs, str):
            raise ModelUnavailableError(
                f"user agent {agent_id}: inputs must be a string, "
                f"got {type(inputs).__name__}",
            )
        agent = factory()
        try:
            record = agent.run(inputs)
        except RuntimeError as exc:
            if "pydantic-ai" in str(exc).lower():
                raise ModelUnavailableError(str(exc)) from exc
            raise
        if record is None or record.output is None:
            raise ModelUnavailableError(
                record.error if record else f"{agent_id} produced no output",
            )
        return _expand_brain_response(record.output)

    return task


def _expand_brain_response(output: Any) -> Any:
    """Surface JSON-embedded fields alongside ``BrainResponse`` attrs.

    User agents are typed as ``BrainResponse`` returners — their
    structured output lives inside ``answer`` as a JSON string. The
    eval evaluators look up fields via ``getattr`` + dict-get
    (:func:`evals.evaluators._resolve_attr`), so we return a merged
    dict that exposes:

    - the top-level ``BrainResponse`` fields (``answer``, ``model``,
      ``sources``, etc.) so prose evaluators on ``answer`` still work
    - any keys parsed from the JSON inside ``answer`` so structured
      evaluators (``FieldEquals`` on ``purchase_id``) work out of the
      box

    Tolerant of markdown code fences (```json ... ```), leading/
    trailing prose around the JSON object, and outputs that aren't
    JSON at all — in those cases the raw output is returned unchanged.

    sensitivity_tier: 1 (operational shape only)
    """
    answer = getattr(output, "answer", None)
    if not isinstance(answer, str) or not answer.strip():
        return output

    parsed = _parse_json_loose(answer)
    if not isinstance(parsed, dict):
        return output

    merged: dict[str, Any] = {
        "answer": answer,
        "model": getattr(output, "model", None),
        "latency_ms": getattr(output, "latency_ms", None),
        "sources": getattr(output, "sources", None),
        "parts": getattr(output, "parts", None),
        "context_summary": getattr(output, "context_summary", None),
    }
    # Parsed JSON keys overlay so evaluators see the user's fields.
    # We intentionally let parsed keys override BrainResponse keys
    # when they collide — the user's JSON is the test surface.
    merged.update(parsed)
    return merged


def _parse_json_loose(text: str) -> Any:
    """Best-effort JSON parse tolerant of markdown fences and prose.

    sensitivity_tier: N/A
    """
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        # Drop opening fence (``` or ```json) and closing fence.
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    try:
        return json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        pass
    # Fallback: scan for the largest balanced JSON object substring.
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(candidate[start:end + 1])
        except (json.JSONDecodeError, ValueError):
            return None
    return None


# ---------------------------------------------------------------------------
# action_params — extract MCP tool parameters from a free-form request
# ---------------------------------------------------------------------------


def action_params_task() -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Run :func:`extract_action_params` on a free-form user request.

    Inputs:
        ``question``       — the user's natural-language ask
        ``tool_name``      — display name for the tool (used in the prompt)
        ``input_schema``   — JSON Schema dict for the tool's parameters
        ``context``        — optional personal-context paragraph

    Output is a dict of the extracted parameter values so evaluators
    can ``FieldEquals`` / ``FieldContains`` against specific keys.

    sensitivity_tier: N/A
    """
    from dataclasses import dataclass as _dc

    from src.agents.brain.actions import extract_action_params

    @_dc(frozen=True)
    class _StubAction:
        display_name: str
        tool_name: str

    def run(inputs: dict[str, Any]) -> dict[str, Any]:
        question = str(inputs.get("question", ""))
        tool_name = str(inputs.get("tool_name", "create_event"))
        input_schema = inputs.get("input_schema") or {}
        context = str(inputs.get("context", "") or "")
        action = _StubAction(
            display_name=tool_name,
            tool_name=tool_name,
        )
        try:
            extracted, missing = extract_action_params(
                question, action, input_schema, context, None,  # type: ignore[arg-type]
            )
        except Exception as exc:  # noqa: BLE001
            raise ModelUnavailableError(str(exc)) from exc
        return {"extracted": extracted, "missing": list(missing)}

    return run


# ---------------------------------------------------------------------------
# action_proposal_judge — independent verifier over a proposed action
# ---------------------------------------------------------------------------


def action_proposal_judge_task() -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Run the judge sub-agent against a fully-formed proposal.

    Inputs:
        ``user_message``      — the user's free-form request
        ``tool_name``         — chosen tool id (e.g. ``create_event``)
        ``tool_schema``       — JSON schema for the tool's parameters
        ``proposed_arguments``— payload the primary extractor would
                                produce (often deliberately bad in
                                the dataset cases so the judge has
                                something to catch)
        ``today_iso``         — optional pin for date math

    Output is the :class:`ActionProposalVerdict` rendered to a dict
    (``ok``, ``reasons``, ``patches``, ``cannot_recover``) so
    ``FieldEquals`` / ``FieldContains`` evaluators can grade specific
    components of the verdict.

    sensitivity_tier: 2
    """
    from src.agents.action_proposal_judge import (
        JudgeDeps,
        register_action_proposal_judge,
    )

    def run(inputs: dict[str, Any]) -> dict[str, Any]:
        register_action_proposal_judge()
        from src.agents.core.registry import get_agent
        defn = get_agent("action_proposal_judge")
        if defn is None or defn.factory is None:
            raise ModelUnavailableError(
                "action_proposal_judge agent not registered",
            )
        agent = defn.factory()
        deps = JudgeDeps(
            user_message=str(inputs.get("user_message", "")),
            tool_name=str(inputs.get("tool_name", "")),
            proposed_arguments=inputs.get("proposed_arguments") or {},
            tool_schema=inputs.get("tool_schema") or {},
            today_iso=str(inputs.get("today_iso") or ""),
        )
        try:
            record = agent.run(deps)
        except RuntimeError as exc:
            if "pydantic-ai" in str(exc).lower():
                raise ModelUnavailableError(str(exc)) from exc
            raise
        if record.output is None:
            raise ModelUnavailableError(
                record.error or "judge returned no output",
            )
        verdict = record.output
        return {
            "ok": verdict.ok,
            "reasons": list(verdict.reasons),
            "patches": dict(verdict.patches),
            "cannot_recover": verdict.cannot_recover,
        }

    return run


# ---------------------------------------------------------------------------
# Registry — dataset filename → task factory
# ---------------------------------------------------------------------------


TASK_REGISTRY: dict[str, Callable[[], Callable[[Any], Any]]] = {
    # Locked
    "brain_qa.yaml": brain_task,
    "firewall_prompts.yaml": injection_firewall_task,
    "injection_scan.yaml": injection_scan_task,
    "egress_routing.yaml": egress_firewall_task,
    # Direct sub-agents
    "sensitivity.yaml": sensitivity_task,
    "labeler.yaml": labeler_task,
    "triage.yaml": triage_task,
    "fact_extractor.yaml": fact_extractor_task,
    "message_eval.yaml": message_eval_task,
    "pending_reply.yaml": pending_reply_task,
    "contact_context.yaml": contact_context_task,
    "actionable_events.yaml": actionable_events_task,
    "action_params.yaml": action_params_task,
    "action_proposal_judge.yaml": action_proposal_judge_task,
    # Indirect sub-agents
    "query_router.yaml": query_router_task,
    "topic_extractor.yaml": topic_extractor_task,
    "event_categorizer.yaml": event_categorizer_task,
    "schema_discovery.yaml": schema_discovery_task,
    "model_generator.yaml": model_generator_task,
    "relationship_tracker.yaml": relationship_tracker_task,
    "dataset_validator.yaml": dataset_validator_task,
    # Goals + habits planner
    "goal_extractor.yaml": goal_extractor_task,
    "habit_suggester.yaml": habit_suggester_task,
    "task_proposer.yaml": task_proposer_task,
    "task_curator.yaml": task_curator_task,
}


__all__ = [
    "ModelUnavailableError",
    "TASK_REGISTRY",
    "action_params_task",
    "action_proposal_judge_task",
    "actionable_events_task",
    "brain_task",
    "contact_context_task",
    "dataset_validator_task",
    "egress_firewall_task",
    "event_categorizer_task",
    "fact_extractor_task",
    "goal_extractor_task",
    "habit_suggester_task",
    "injection_firewall_task",
    "injection_scan_task",
    "labeler_task",
    "message_eval_task",
    "model_generator_task",
    "pending_reply_task",
    "query_router_task",
    "relationship_tracker_task",
    "schema_discovery_task",
    "sensitivity_task",
    "task_curator_task",
    "task_proposer_task",
    "topic_extractor_task",
    "triage_task",
    "user_agent_task",
]
