"""Pydantic ports of the agent output dataclasses.

Each LLM-using component currently returns a frozen ``@dataclass``. To use
``pydantic-ai``'s typed output we re-declare them as ``BaseModel``s here.

The frozen dataclasses in ``src/agents/*.py`` stay in place during the
migration so the old call sites keep working. As each sub-agent is moved
to ``SBAgent`` (Phase 3), its dataclass is deleted and call sites switch
to the model here.

sensitivity_tier: varies (these schemas carry user-derived content)
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------


class AgentOutput(BaseModel):
    """Base for every structured agent output.

    Frozen-by-default and strict — extra fields are rejected. Sub-agents
    should subclass this so the firewall and audit layers can rely on a
    common surface (e.g. for redacting fields above a tier cap).

    sensitivity_tier: 1
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


# ---------------------------------------------------------------------------
# Firewall outputs
# ---------------------------------------------------------------------------


class InjectionVerdict(AgentOutput):
    """Decision returned by ``InjectionFirewall``.

    sensitivity_tier: 1
    """

    allowed: bool
    category: Literal[
        "safe",
        "injection",
        "data_bleed",
        "role_override",
        "jailbreak",
    ] = "safe"
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""


class ActionProposalVerdict(AgentOutput):
    """Independent judge's verdict on a proposed MCP action.

    The judge inspects the user's original message and the arguments
    the primary extractor produced, and decides whether the proposal
    faithfully reflects what the user asked for. Run with a different
    LLM family from the primary extractor so the two are independent —
    one model's hallucination is unlikely to be exactly mirrored by
    another family's.

    Output shape is intentionally constructive:

    - ``ok`` — proposal matches the user's intent as-is.
    - ``reasons`` — short bullet list of concerns (max 3, ≤ 120 chars
      each). Surfaced to the UI when present.
    - ``patches`` — corrected values the judge would write into
      ``arguments``. Caller applies these BEFORE rendering the
      confirmation card. Only fields the judge is confident about.
    - ``cannot_recover`` — set to True when the judge cannot construct
      a faithful proposal from the user's message at all (e.g. the
      user's request is too ambiguous to act on). The caller should
      refuse the action and ask the user to clarify.

    sensitivity_tier: 2
    """

    ok: bool = True
    reasons: list[str] = Field(default_factory=list, max_length=3)
    patches: dict[str, Any] = Field(default_factory=dict)
    cannot_recover: bool = False


class EgressDecision(AgentOutput):
    """Decision returned by ``EgressFirewall``.

    Routes a prompt to either the dedicated remote endpoint or the local
    fallback, based on the maximum sensitivity tier the prompt may carry.

    sensitivity_tier: 1
    """

    route: Literal["remote", "local", "blocked"]
    max_tier: int = Field(ge=1, le=3)
    reason: str = ""


# ---------------------------------------------------------------------------
# Domain output types (ports of existing dataclasses)
# ---------------------------------------------------------------------------


class SensitivityVerdict(AgentOutput):
    """Tier classification for a piece of free-text content.

    Port of the dict returned by ``src/models/sensitivity_classifier.py``.

    sensitivity_tier: varies (classification target may be any tier)
    """

    tier: Literal[1, 2, 3]
    reason: str


class TriageDecision(AgentOutput):
    """Per-message keep/drop verdict.

    Port of ``src/agents/message_triage.TriageDecision``.

    sensitivity_tier: 2
    """

    message_id: str
    keep: bool
    reason: str
    is_promo: bool = False
    is_automated: bool = False
    is_ack_only: bool = False


class TriageBatch(AgentOutput):
    """Container for multiple triage verdicts.

    ``pydantic_ai.Agent`` requires a single ``BaseModel`` as ``output_type``,
    so list-shaped outputs need a wrapper. The agent's prompt asks the
    LLM to fill ``decisions`` with one entry per input message.

    sensitivity_tier: 2
    """

    decisions: list[TriageDecision] = Field(default_factory=list)


class EmotionalLabel(AgentOutput):
    """Structured emotional dimensions for a single text.

    Port of the dict returned by ``src/models/labeler.py``.

    sensitivity_tier: 3 (emotions)
    """

    primary_emotion: Literal[
        "joy", "sadness", "anger", "fear",
        "surprise", "disgust", "trust", "anticipation",
    ]
    intensity: float = Field(ge=0.0, le=1.0)
    feelings: list[str] = Field(default_factory=list)
    desires: list[str] = Field(default_factory=list)
    actors: list[str] = Field(default_factory=list)
    environment: str = ""
    domain: Literal["personal", "work", "health", "social", "spiritual"] = (
        "personal"
    )


class LearnedFact(AgentOutput):
    """A single fact extracted from user content.

    Port of ``src/agents/fact_learner.LearnedFact``.

    sensitivity_tier: 2-3 (depends on category)
    """

    id: str
    category: Literal[
        "preference",
        "relationship",
        "biographical",
        "habit",
        "opinion",
        "health",
        "work",
        "location",
    ]
    subject: str
    predicate: str
    content: str
    confidence: float = Field(ge=0.0, le=1.0)
    source_type: str
    source_id: str | None = None
    extracted_at: str
    confirmed_at: str | None = None


class LearnedFactDraft(AgentOutput):
    """LLM-generated portion of a learned fact.

    The agent only produces the semantic fields. ``id``, ``confidence``,
    ``source_type``, ``source_id``, and ``extracted_at`` come from the
    caller (which knows the conversation context).

    sensitivity_tier: 2-3 (depends on category)
    """

    category: Literal[
        "preference",
        "relationship",
        "biographical",
        "habit",
        "opinion",
        "health",
        "work",
        "location",
    ]
    subject: str
    predicate: str
    content: str
    sensitivity_tier: int = Field(ge=1, le=3)


class LearnedFactBatch(AgentOutput):
    """Container for a batch of fact drafts.

    sensitivity_tier: 2-3
    """

    facts: list[LearnedFactDraft] = Field(default_factory=list)


class Insight(AgentOutput):
    """A proactive insight surfaced from question-pattern analysis.

    Port of ``src/agents/insight_generator.Insight``.

    sensitivity_tier: 2
    """

    id: str
    domain: str
    title: str
    content: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    trigger: str
    pattern: str | None = None
    generated_at: str
    sensitivity_tier: int = Field(ge=1, le=3)
    suggested_followup: str | None = None


class DigestSummary(AgentOutput):
    """Weekly digest output authored by ``WeeklyDigestAgent``.

    Three named sections + an optional top-line highlight. Empty
    strings render as omitted in the UI; the LLM is encouraged to
    leave a section blank if the week had no relevant activity.

    sensitivity_tier: 2
    """

    highlight: str = ""
    communication: str = ""
    schedule: str = ""
    notes: str = ""


class RelationshipNudge(AgentOutput):
    """A warm reach-out suggestion authored by
    ``RelationshipTrackerAgent``.

    sensitivity_tier: 2
    """

    contact_name: str
    nudge: str
    suggested_topic: str | None = None


class InsightDraft(AgentOutput):
    """LLM-generated portion of a proactive insight.

    The agent only produces the user-facing prose. ``id``, ``trigger``,
    ``pattern``, ``generated_at``, ``sensitivity_tier``, ``sources``,
    and ``domain`` are filled in by the caller from the surrounding
    context.

    sensitivity_tier: 2
    """

    title: str
    content: str
    suggested_followup: str | None = None


class MessageNotification(AgentOutput):
    """Importance-ranked notification candidate.

    Port of ``src/agents/message_evaluator.MessageNotification``.

    sensitivity_tier: 2
    """

    id: str
    message_ids: list[str]
    notification_type: Literal["topic_action", "topic_enrichment"]
    importance: int = Field(ge=1, le=10)
    domain: str
    summary: str
    contacts: list[str] = Field(default_factory=list)
    related_context: str = ""
    created_at: str


class MessageNotificationDraft(AgentOutput):
    """LLM-generated portion of a topic-driven notification.

    The agent picks which messages warrant notifying and why. The
    orchestrator fills in ``id``, ``created_at``, ``contacts`` (from
    the message records), and groups multiple drafts into a single
    :class:`MessageNotification` when appropriate.

    sensitivity_tier: 2
    """

    message_id: str
    notification_type: Literal["topic_action", "topic_enrichment"]
    importance: int = Field(ge=1, le=10)
    domain: str
    summary: str
    related_to: str = ""


class MessageNotificationBatch(AgentOutput):
    """Container for multiple notification drafts.

    sensitivity_tier: 2
    """

    notifications: list[MessageNotificationDraft] = Field(
        default_factory=list,
    )


class PendingReply(AgentOutput):
    """A message awaiting a user response.

    Port of ``src/agents/proactive_intelligence.PendingReply``.

    sensitivity_tier: 2
    """

    id: str
    message_id: str
    source: str
    contact_name: str
    domain: str
    preview: str
    importance: int = Field(ge=1, le=10)
    reason: str
    message_at: str
    detected_at: str
    sensitivity_tier: int = Field(ge=1, le=3)


class PendingReplyDraft(AgentOutput):
    """LLM-generated portion of a pending-reply candidate.

    The agent classifies whether a message needs the user's reply,
    its importance, the conversational domain, and a short reason.
    The orchestrator looks up the source / contact / timestamp from
    the underlying message record and stamps the rest.

    sensitivity_tier: 2
    """

    message_id: str
    needs_reply: bool
    importance: int = Field(ge=1, le=10)
    domain: Literal[
        "personal", "work", "family", "social", "health",
    ]
    reason: str


class PendingReplyBatch(AgentOutput):
    """Container for multiple pending-reply drafts.

    sensitivity_tier: 2
    """

    replies: list[PendingReplyDraft] = Field(default_factory=list)


class ContactContext(AgentOutput):
    """Per-contact summary for the proactive panel.

    Port of ``src/agents/proactive_intelligence.ContactContext``.

    sensitivity_tier: 2
    """

    contact_id: str
    contact_name: str
    phone: str | None = None
    email: str | None = None
    total_messages: int = 0
    messages_7d: int = 0
    last_message_at: str | None = None
    active_context: str | None = None
    context_domains: list[str] = Field(default_factory=list)
    context_priority: int = Field(ge=0, le=10)
    birthday: str | None = None
    has_upcoming_birthday: bool = False
    updated_at: str


class ContactContextDraft(AgentOutput):
    """LLM-generated portion of a contact context summary.

    The agent describes the current situation for one contact: a short
    free-text summary, relevant life domains, and a 0-3 priority
    score. The orchestrator stamps id / name / counts / timestamps.

    sensitivity_tier: 2
    """

    contact_id: str
    active_context: str
    context_domains: list[str] = Field(default_factory=list)
    context_priority: int = Field(ge=0, le=3)


class ContactContextBatch(AgentOutput):
    """Container for multiple contact-context drafts.

    sensitivity_tier: 2
    """

    contexts: list[ContactContextDraft] = Field(default_factory=list)


class ActionableEventDraft(AgentOutput):
    """LLM-generated portion of an actionable calendar event.

    The agent decides whether an upcoming event needs the user's
    action (prepare, bring something, RSVP, send birthday wishes) and
    rates its importance 1-10. The orchestrator joins event metadata.

    sensitivity_tier: 2
    """

    event_id: str
    action_needed: str
    importance: int = Field(ge=1, le=10)


class ActionableEventBatch(AgentOutput):
    """Container for multiple actionable-event drafts.

    sensitivity_tier: 2
    """

    events: list[ActionableEventDraft] = Field(default_factory=list)


class DuckDBQuerySpec(AgentOutput):
    """One structured DuckDB query the router wants to run.

    Port of ``src/core/query_engine.DuckDBQuerySpec``. The router
    chooses columns and an optional WHERE / ORDER BY clause; the
    QueryEngine renders this into safe parameterised SQL before
    execution.

    sensitivity_tier: 1
    """

    table: str
    columns: list[str] = Field(default_factory=list)
    where: str | None = None
    order_by: str | None = None
    limit: int = 10


class FieldMappingDraft(AgentOutput):
    """One source-field → target-column mapping suggestion.

    Port of the LLM-relevant slice of
    ``src/extensions/ingestion/schema_discovery.FieldMapping``.

    sensitivity_tier: 1
    """

    source_name: str
    target_column: str
    target_type: str
    sensitivity_tier: int = Field(ge=1, le=3)
    transform: str | None = None


class SchemaDiscoveryDraft(AgentOutput):
    """LLM-generated portion of a schema-discovery result.

    The legacy ``SchemaDiscoveryAgent`` orchestrator owns the
    rule-based pass, value-scanning, and confidence aggregation. This
    draft is what the LLM step contributes: target table, domain,
    field mappings, and proposed dedup key.

    sensitivity_tier: 1
    """

    target_table: str
    is_new_table: bool
    domain: str
    fields: list[FieldMappingDraft] = Field(default_factory=list)
    dedup_key: list[str] = Field(default_factory=list)


class GeneratedSQLModel(AgentOutput):
    """A single SQLMesh model file generated by the model generator.

    sensitivity_tier: 1
    """

    name: str
    layer: Literal["staging", "intermediate", "marts"]
    sql: str
    sensitivity_summary: str = ""


class DatasetValidationReport(AgentOutput):
    """Verdict for a user-uploaded eval dataset.

    The :mod:`src.agents.dataset_validator` agent inspects a YAML
    payload and returns this report. ``valid`` is ``True`` only when
    structural checks pass; ``firewall_verdict`` is filled in by
    :class:`InjectionFirewall` and is one of ``allow``, ``warn``,
    ``block``.

    sensitivity_tier: 1
    """

    valid: bool
    errors: list[str] = Field(default_factory=list)
    proposals: list[str] = Field(default_factory=list)
    firewall_verdict: Literal["allow", "warn", "block"] = "allow"


class DatasetSuggestion(AgentOutput):
    """A proposed eval dataset for a newly-created user agent.

    Produced by :mod:`src.agents.dataset_creator`. The agent reads the
    target user agent's name + description + system_prompt + max
    sensitivity tier, infers the purpose, and either proposes a starter
    dataset YAML or refuses with concrete improvement hints when the
    purpose is unclear.

    ``system_prompt_additions`` carries one-line edits the user should
    append to their system prompt so the LLM actually produces the
    tokens the dataset expects. The creator infers these from its
    own generated cases — e.g. if every case expects ``status``
    drawn from a closed Portuguese vocabulary but the prompt doesn't
    pin it, the dataset will fail evals with English defaults even
    against a flagship model. Each entry is independently appendable.

    sensitivity_tier: 1
    """

    can_create: bool
    reason_if_not: str | None = None
    purpose_summary: str = ""
    output_shape: Literal[
        "structured", "prose", "classification", "mixed", "unknown",
    ] = "unknown"
    eval_strategy: Literal[
        "deterministic", "llm_judge", "hybrid",
    ] = "llm_judge"
    dataset_yaml: str = ""
    case_count: int = 0
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    notes: list[str] = Field(default_factory=list)
    improvement_hints: list[str] = Field(default_factory=list)
    system_prompt_additions: list[str] = Field(default_factory=list)


class RetrievalPlan(AgentOutput):
    """The Brain's pre-retrieval plan.

    Port of ``src/core/query_engine.RetrievalPlan``.

    sensitivity_tier: 1
    """

    duckdb_queries: list[DuckDBQuerySpec] = Field(default_factory=list)
    chromadb_collections: list[str] = Field(default_factory=list)
    use_graph: bool = False
    reasoning: str = ""


# Canonical 3-bucket category used by Topic, GoalDraft,
# TaskProposalDraft, ScheduleSlot, and Project. Single source of truth
# so the planner can reason about goals + topics + tasks in one frame.
GoalCategory = Literal["personal", "life", "work"]


class Topic(AgentOutput):
    """One ongoing topic extracted from a conversation.

    Port of the dict shape returned by
    ``src/pipeline/intermediate/int_contact_topics.py``.

    sensitivity_tier: 2
    """

    topic: str
    description: str
    importance: int = Field(ge=1, le=10)
    status: Literal["active", "resolved", "stale"]
    # Optional during the rollout — old LLM responses won't carry it.
    # Filled by the extended TopicExtractorAgent prompt; the pipeline
    # back-fills existing rows on the first run.
    category: GoalCategory | None = None


class TopicBatch(AgentOutput):
    """Container for multiple topics extracted from one contact's chat.

    sensitivity_tier: 2
    """

    topics: list[Topic] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Goals — goals aggregation table, populated by the Brain (and the user).
# ---------------------------------------------------------------------------


class GoalDraft(AgentOutput):
    """LLM-generated portion of a goal mined from the user's sources.

    The orchestrator stamps ``id``, ``created_at``, ``status``, and
    resolves ``linked_topic_hint`` against ``_topics`` to set the
    canonical ``_topics.linked_goal_id`` back-reference.

    sensitivity_tier: 2
    """

    title: str
    description: str
    category: GoalCategory
    horizon: Literal["short", "medium", "long"]
    target_date: str | None = None
    importance: int = Field(ge=1, le=10)
    why: str
    source_kind: Literal[
        "message", "note", "fact", "chat", "topic",
    ]
    source_ref: str
    # The proposer/extractor names a topic it thinks this goal subsumes;
    # the curator resolves it to a ``_topics.id`` post-hoc.
    linked_topic_hint: str | None = None


class GoalBatch(AgentOutput):
    """Container for goals mined in one pass.

    sensitivity_tier: 2
    """

    goals: list[GoalDraft] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Tasks — proposer + completion-detector outputs.
# ---------------------------------------------------------------------------


class TaskProposalDraft(AgentOutput):
    """LLM-generated portion of a proposed task.

    The proposer reads recent messages, active topics, and active goals
    and emits *tasks* — explicit work the user must do — distinct from
    pending replies (a separate sub-agent). When the task plausibly
    serves a goal, ``parent_goal_hint`` names it; the orchestrator
    resolves the hint to ``_tasks.goal_id``.

    sensitivity_tier: 2
    """

    title: str
    notes: str = ""
    category: GoalCategory
    importance: int = Field(ge=1, le=10)
    due_at: str | None = None
    source_message_ids: list[str] = Field(default_factory=list)
    parent_topic_hint: str | None = None
    parent_goal_hint: str | None = None
    reason: str


class TaskProposalBatch(AgentOutput):
    """Container for proposed tasks from one curator pass.

    sensitivity_tier: 2
    """

    tasks: list[TaskProposalDraft] = Field(default_factory=list)


class TaskCompletionDraft(AgentOutput):
    """LLM-generated verdict that an open task has been completed.

    ``confidence`` < 0.7 sends the verdict to the review queue rather
    than auto-closing the task.

    sensitivity_tier: 2
    """

    task_id: str
    evidence_message_id: str
    evidence_summary: str
    confidence: float = Field(ge=0.0, le=1.0)


class TaskCompletionBatch(AgentOutput):
    """Container for completion verdicts in one pass.

    sensitivity_tier: 2
    """

    completions: list[TaskCompletionDraft] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Daily scheduler + habit suggester outputs.
# ---------------------------------------------------------------------------


class ScheduleSlot(AgentOutput):
    """One time-bound slot on the user's day.

    ``kind`` distinguishes a fixed calendar event (immovable) from a
    task or habit slot the scheduler placed.

    sensitivity_tier: 2
    """

    kind: Literal["event", "task", "habit"]
    ref_id: str
    title: str
    start: str
    end: str
    why: str = ""
    category: GoalCategory | None = None
    goal_id: str | None = None


class DailySchedule(AgentOutput):
    """A scheduler output for a single day.

    ``category_balance`` reports minutes-per-category across the placed
    slots so the UI can show the day's mix at a glance.

    sensitivity_tier: 2
    """

    schedule_date: str
    slots: list[ScheduleSlot] = Field(default_factory=list)
    unscheduled_overflow: list[str] = Field(default_factory=list)
    rationale: str = ""
    category_balance: dict[str, int] = Field(default_factory=dict)


class HabitDraft(AgentOutput):
    """LLM-generated portion of a habit suggestion.

    Atomic-habits coupling: every habit must name the goal it serves.
    The prompt refuses to emit goal-less habits.

    sensitivity_tier: 1
    """

    title: str
    cadence: Literal["daily", "weekly", "specific_days"]
    days_of_week: list[Literal[
        "mon", "tue", "wed", "thu", "fri", "sat", "sun",
    ]] = Field(default_factory=list)
    preferred_window: Literal["morning", "midday", "evening", "any"]
    goal_id: str
    why: str
    reason: str


class HabitBatch(AgentOutput):
    """Container for habit suggestions in one pass.

    sensitivity_tier: 1
    """

    habits: list[HabitDraft] = Field(default_factory=list)


class EventCategoryDecision(AgentOutput):
    """Categorisation verdict for a single calendar event.

    Produced by :class:`EventCategorizerAgent`. The closed vocabulary
    matches the ``event_category`` audit on ``int_events_enriched``:
    ``meeting`` covers work meetings/standups/syncs; ``social`` covers
    personal gatherings (dinner, party, concert); ``health`` covers
    medical/therapy appointments; ``travel`` covers flights/trips;
    ``other`` is the fall-through when nothing else fits.

    sensitivity_tier: 2
    """

    category: Literal["meeting", "social", "health", "travel", "other"]
    reason: str = ""


class ActionProposal(AgentOutput):
    """A user-facing action proposed by the Brain.

    Port of ``src/agents/brain_agent.ActionProposal``.

    sensitivity_tier: 2
    """

    proposal_id: str
    connector_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""


class MessagePart(AgentOutput):
    """A typed piece of an assistant message.

    Renderable as a single artifact on the frontend (markdown bubble,
    code block, chart, table, image, sandboxed HTML, etc.). The chat
    ``brain-stream`` channel emits parts incrementally; the UI mounts a
    component from the artifact registry keyed by ``mime``.

    ``data`` is a ``str`` for textual MIMEs (``text/markdown``,
    ``text/x-python``, ``text/html``, ``text/vnd.mermaid``) and a
    nested object for JSON specs (``application/vnd.vega-lite+json``,
    ``application/vnd.arandu.table+json``).

    sensitivity_tier: varies (carries user-derived content)
    """

    id: str
    mime: str
    title: str = ""
    data: Any
    display: Literal["inline", "panel"] = "inline"
    sensitivity_tier: int = Field(default=1, ge=1, le=3)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelOption(AgentOutput):
    """One model recommendation produced by :class:`ModelPickerAgent`.

    ``model_id`` is a live id returned by the configured endpoint's
    ``/models`` listing — the CLI handler validates this before
    surfacing the response to the UI. ``route`` decides which endpoint
    the id belongs to; the UI applies both fields when the user clicks
    "Use".

    sensitivity_tier: 1
    """

    model_id: str
    route: Literal["remote", "local"]
    rationale: str = ""


class ModelRecommendation(AgentOutput):
    """A model-picker suggestion for a (saved or unsaved) user agent.

    Produced by :mod:`src.agents.model_picker`. The agent reads the
    target agent's name + description + system prompt + skills + tools
    + max sensitivity tier, infers the purpose, and either proposes a
    best-overall + cost-effective pair from the live model catalog or
    refuses with concrete improvement hints when the purpose is
    unclear.

    sensitivity_tier: 1
    """

    can_recommend: bool
    reason_if_not: str | None = None
    purpose_summary: str = ""
    best_overall: ModelOption | None = None
    cost_effective: ModelOption | None = None
    notes: list[str] = Field(default_factory=list)
    improvement_hints: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class Improvement(AgentOutput):
    """One categorised edit the prompt engineer recommends.

    ``original_snippet`` is verbatim from the user's input (empty
    string when the issue is a missing section rather than a wrong
    one). ``suggested_replacement`` is what should appear in the
    rewritten prompt for that snippet (or the full new section when
    the snippet is empty). ``rationale`` is one sentence explaining
    why the change helps. ``target`` picks which field the edit
    applies to.

    sensitivity_tier: 1
    """

    category: Literal[
        "clarity",
        "expected_output",
        "language",
        "format",
        "scope",
        "safety",
    ]
    original_snippet: str = ""
    suggested_replacement: str
    rationale: str
    target: Literal["system_prompt", "description"] = "system_prompt"


class PromptSuggestion(AgentOutput):
    """A prompt-engineer rewrite of a user agent's prompt + description.

    Produced by :mod:`src.agents.prompt_engineer`. Mirrors the refusal
    pattern of :class:`DatasetSuggestion`. When ``can_improve=False``,
    ``reason_if_not`` is required and ``improvements`` carries 2-4
    imperative entries the user can act on manually; the rewrite
    fields are left empty.

    The UI offers two independent apply paths:

    * ``improved_system_prompt`` / ``improved_description`` — full
      rewrites that replace the originals when the user clicks
      "Apply full rewrite".
    * ``system_prompt_additions`` — short imperative lines (≤140
      chars, 0-4 entries) the user can append verbatim without
      taking the full rewrite. Same surgical layer as
      :class:`DatasetSuggestion.system_prompt_additions`.

    sensitivity_tier: 1
    """

    can_improve: bool
    reason_if_not: str | None = None
    improved_system_prompt: str = ""
    improved_description: str = ""
    system_prompt_additions: list[str] = Field(default_factory=list)
    improvements: list[Improvement] = Field(default_factory=list)
    change_summary: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    notes: list[str] = Field(default_factory=list)


class BrainResponse(AgentOutput):
    """Final response from the Brain orchestrator.

    Port of ``src/agents/brain_agent.BrainResponse``.

    sensitivity_tier: varies
    """

    answer: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    context_summary: str = ""
    model: str
    latency_ms: float = 0.0
    parts: list[MessagePart] = Field(default_factory=list)


class ChatResponse(AgentOutput):
    """Final response from the Chat orchestrator.

    Mirrors :class:`BrainResponse` but emitted by the dedicated chat
    agent. Sources / context_summary surface when the chat LLM
    delegated to Brain (or another grounded tool) during the turn —
    pure chit-chat turns leave them empty.

    sensitivity_tier: varies
    """

    answer: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    context_summary: str = ""
    model: str
    latency_ms: float = 0.0
    parts: list[MessagePart] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Deep agent surface
# ---------------------------------------------------------------------------


class PlanStep(AgentOutput):
    """A single step in a ``Plan``.

    sensitivity_tier: 1
    """

    id: str
    description: str
    status: Literal["pending", "in_progress", "completed", "blocked"] = (
        "pending"
    )
    notes: str = ""


class Plan(AgentOutput):
    """A deep agent's plan, surfaced to the UI before execution.

    sensitivity_tier: 1
    """

    goal: str
    steps: list[PlanStep] = Field(default_factory=list)
    revision: int = 0


class ReflectionVerdict(AgentOutput):
    """Self-review decision returned by the Reflector.

    Emitted at each reflection checkpoint inside an orchestrator's
    pydantic-ai loop. The reflective runner consumes this verdict to
    decide whether to inject a ``STOP_REQUEST`` or to continue (and
    possibly promote the task class).

    - ``continue_research`` — when ``True``, the runner lets the model
      keep using tools. When ``False``, it injects a stop message at
      the next ``ModelRequestNode``.
    - ``reason`` — short, user-facing explanation (≤ 120 chars). When
      research is being extended, this is what the chat banner /
      background notification displays.
    - ``suggested_class`` — class the run *should* be promoted to.
      A FAST run can promote to DEEP; a DEEP run can stay DEEP. The
      runner ignores demotions and ignores ``BACKGROUND_DEEP`` here
      since that class is only ever set by the caller.

    sensitivity_tier: 1
    """

    continue_research: bool
    reason: str = Field(default="", max_length=240)
    suggested_class: Literal[
        "interactive_fast", "interactive_deep",
    ] = "interactive_fast"


__all__ = [
    "ActionProposal",
    "ActionableEventBatch",
    "ActionableEventDraft",
    "AgentOutput",
    "BrainResponse",
    "ContactContext",
    "ContactContextBatch",
    "ContactContextDraft",
    "DailySchedule",
    "DatasetSuggestion",
    "DatasetValidationReport",
    "DigestSummary",
    "DuckDBQuerySpec",
    "EgressDecision",
    "ActionProposalVerdict",
    "EmotionalLabel",
    "EventCategoryDecision",
    "FieldMappingDraft",
    "GeneratedSQLModel",
    "GoalBatch",
    "GoalCategory",
    "GoalDraft",
    "HabitBatch",
    "HabitDraft",
    "Improvement",
    "InjectionVerdict",
    "Insight",
    "InsightDraft",
    "LearnedFact",
    "LearnedFactBatch",
    "LearnedFactDraft",
    "MessagePart",
    "MessageNotification",
    "MessageNotificationBatch",
    "MessageNotificationDraft",
    "ModelOption",
    "ModelRecommendation",
    "PendingReply",
    "PendingReplyBatch",
    "PendingReplyDraft",
    "Plan",
    "PlanStep",
    "PromptSuggestion",
    "ReflectionVerdict",
    "RelationshipNudge",
    "RetrievalPlan",
    "ScheduleSlot",
    "SchemaDiscoveryDraft",
    "SensitivityVerdict",
    "TaskCompletionBatch",
    "TaskCompletionDraft",
    "TaskProposalBatch",
    "TaskProposalDraft",
    "Topic",
    "TopicBatch",
    "TriageBatch",
    "TriageDecision",
]
