"""Brain action helpers — MCP action proposal primitives.

Pure module-level templates, connector metadata, and stateless helper
functions used to:

1. Classify whether a user message is an action intent.
2. Query DuckDB for candidate records the user might be referring to.
3. Render a candidate list for confirmation.
4. Generate parameter values for the chosen action via LLM.

Shared by the legacy ``BrainAgent`` (during transition) and the new
:class:`BrainAgentV2` ``propose_action`` tool (Phase B).

sensitivity_tier: 2 (touches user messages + structured personal data)
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any

from src.agents.firewall.egress_firewall import Lane
from src.models.llm_gateway import GatewayBlocked, chat_via_firewalls
from src.models.llm_provider import LLMProvider
from src.models.prompt_cache import PROMPTS_DIR, FrozenPromptTemplate

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# ActionProposal — the wire shape returned to the UI
# ----------------------------------------------------------------------


# Tool names whose side-effects are read-only against third-party
# services — listing, searching, and retrieving without write effects.
# Proposals targeting these tools carry ``risk='low'`` and auto-execute
# without prompting the user for confirmation. Writes (send_, create_,
# update_, delete_, reply_, move_, flag_, …) stay ``risk='high'``.
_LOW_RISK_PREFIXES: tuple[str, ...] = (
    "search_", "get_", "list_", "find_",
    "read_", "recall_", "fetch_",
)
_LOW_RISK_EXACT: frozenset[str] = frozenset({"web_search"})


def is_low_risk_tool(tool_name: str) -> bool:
    """True when ``tool_name`` is safe to auto-execute.

    sensitivity_tier: 1
    """
    if tool_name in _LOW_RISK_EXACT:
        return True
    return tool_name.startswith(_LOW_RISK_PREFIXES)


@dataclass(frozen=True)
class ActionProposal:
    """A proposed MCP action awaiting user confirmation.

    Carries everything the ``confirm-action`` CLI handler needs to
    execute the action statelessly. The same dataclass is rendered as
    JSON in the ``action_proposal`` stream chunk consumed by the
    frontend.

    sensitivity_tier: 2
    """

    proposal_id: str
    connector_id: str
    connector_name: str
    tool_name: str
    display_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    missing_params: list[str] = field(default_factory=list)
    command: str = ""
    args: tuple[str, ...] = ()
    risk: str = "high"
    """``"low"`` for read-only tools the frontend may auto-confirm.

    Defaults to ``"high"`` so any new tool is opt-in safe.
    """
    recipient_preview: dict[str, Any] | None = None
    """Resolved recipient identity for messaging / email tools, e.g.
    ``{"name": "Elmara", "phone": "+5511...", "channel": "whatsapp",
       "resolved": True}``. Surfaced on the confirmation card so the
    user can verify the contact before sending. ``resolved: False``
    means we couldn't match the name to a saved contact — the card
    must warn the user instead of silently shipping a possibly-wrong
    destination. ``None`` for non-messaging tools (calendar, etc.)
    where there's no recipient to verify.
    """


@dataclass(frozen=True)
class RecipientDisambiguationProposal:
    """A pending recipient choice that blocks a messaging action.

    Returned by :func:`build_action_proposal` whenever a messaging
    tool's recipient field needs human confirmation — which is every
    time, by design: a wrong recipient on a private message is a
    worse failure than one extra tap. The frontend renders a
    candidate-picker; the user's selection feeds back via
    ``resume_action_with_recipient`` which re-enters the proposal
    builder with the chosen handle injected.

    sensitivity_tier: 3 (candidates carry contact details)
    """

    proposal_id: str
    connector_id: str
    connector_name: str
    tool_name: str
    display_name: str
    channel: str
    original_name: str
    candidates: list[dict[str, Any]] = field(default_factory=list)
    draft_arguments: dict[str, Any] = field(default_factory=dict)
    command: str = ""
    args: tuple[str, ...] = ()
    question: str = ""
    context_text: str = ""


# ----------------------------------------------------------------------
# Frozen prompt templates
# ----------------------------------------------------------------------
# Each frozen prefix is loaded from src/models/prompts/. The suffix
# (dynamic portion) stays inline so callers .format() the full prompt
# without changing call sites.

_PARAM_EXTRACT_TEMPLATE = FrozenPromptTemplate(
    PROMPTS_DIR / "param_extractor_v1.txt",
)
_PARAM_EXTRACT_SUFFIX = """\
{tool_name}
Parameters (JSON Schema): {schema}

User request: {question}

{context_section}

IMPORTANT RULES (read in order — earlier rules win):

1. USER LITERAL VALUES ARE LAW. If the user's request explicitly \
gives a value — a quoted title ("Play Tennis with Tiago"), a named \
person ("with Tiago"), an explicit date/time ("tomorrow 7am", \
"next Monday at 3pm"), a specific list or location — you MUST copy \
that value verbatim into the matching parameter. Do NOT substitute it \
with anything from the database records, even if the records look \
related. The database is for resolving *relative* references, never \
for overriding explicit ones.

2. RESOLVE RELATIVE REFERENCES from the personal context AND database \
records ONLY when the user has not given an explicit value. For \
example:
  "the last note" → most recently created/updated note in the records.
  "tomorrow's meeting" → the event title for tomorrow.
  "the email from João" → the email subject from that sender.
When the user says "last", "latest", "most recent", "última", "último", \
use the most recent matching item's actual title/subject/name.

3. DATE / TIME RESOLUTION. Compute ISO 8601 timestamps relative to \
today ({today}). "tomorrow 7am" → tomorrow's date + 07:00:00 in the \
user's local time. Always set start_time AND end_time for events — if \
the user only gave a start, default end_time to start_time + 1 hour. \
Never leave a required date/time parameter null when the user gave a \
literal time reference.

4. PARAMETER NAMES ARE BINDING. Use the schema's exact parameter names \
(``title``, ``start_time``, ``end_time``, ``location``, etc.). Do not \
invent new ones; do not rename them.

5. REPLY LANGUAGE. If the context contains a "Most recent inbound \
message" block, ANY ``body``/``content``/``text`` parameter you write \
MUST be in the SAME natural language as that message. Detect the \
language from the inbound text itself, not from the user's command \
("reply to her" is in English even when the inbound message is in \
Portuguese — write the body in Portuguese). Never default to English \
when the inbound message is in another language.

6. REPLY VOICE. When you write the body of a message, email, or any \
outbound text on the user's behalf, write it AS THE USER. Do NOT \
introduce yourself or any assistant persona ("Arandu", "your AI", \
"the assistant says...") — that text is going to the user's friend \
verbatim and would read as a third party hijacking the conversation. \
Use the first person ("eu", "I"). Match the tone and register of the \
inbound message (casual/formal, emoji-or-not). Never sign as anyone \
other than the user.

7. NO PLACEHOLDERS IN OUTPUT. The context may contain redaction \
placeholders like ``__PERSON_2726__``, ``__EMAIL_3__``, ``__PHONE_1__`` \
— those are a privacy chokepoint for the LLM call, not real \
substrings the user would ever type or read. Never emit a literal \
placeholder token in a parameter value. If you would otherwise write \
the placeholder (because you don't know the underlying name), leave \
the field general ("oi tudo bem?" rather than "oi __PERSON_2726__!"). \
A downstream rehydration pass restores names from the registry, but \
that only works if you treat placeholders as opaque entities, not \
as part of the message body.

8. GRAMMATICAL NUMBER. When the action targets ONE named person, \
addressing pronouns/verbs MUST be singular — Portuguese ``você`` \
(not ``vocês``), Spanish ``tú``/``usted`` (not ``ustedes``/``vosotros``), \
English ``you`` referring to a single person. The agent often \
defaults to plural ("para vocês", "you all") when the inbound \
message used plural pronouns — that's wrong. Look at the recipient \
profile / the ``to`` field: if it names a single person, singular \
addressing throughout.

9. RELATIONAL TONE. If the context contains a "Recipient profile" \
block, match the register implied by the ``relationship`` label and \
mimic the style of the listed recent outbound samples (those are how \
the user actually writes to this person). Spouses / partners / close \
family / close friends get warmth and casual intimacy; colleagues / \
clients get reserved professional tone. Never default to corporate \
neutral when the relationship is intimate — "Oi, tudo bem?" beats \
"Estimada Senhora, gostaria de informar..." when writing to your \
wife.

10. OUTPUT FORMAT. Emit ONLY a valid JSON object with the parameter \
values. Use ``null`` ONLY when no information is available anywhere \
(neither request nor context nor records). Never output the literal \
string ``"None"`` or ``"null"`` — use JSON null.\
"""

_ACTION_WHERE_TEMPLATE = FrozenPromptTemplate(
    PROMPTS_DIR / "action_where_v1.txt",
)
_ACTION_WHERE_SUFFIX = """\
{table}
Columns: {columns}
Today's date: {today}

User request: {question}

RULES:
- Output ONLY the WHERE clause condition (no SELECT, no WHERE keyword).
- Use SQLite syntax (date(), datetime(), strftime(), CAST, etc.).
- For "yesterday" use: created_at >= date('now', '-1 day') \
AND created_at < date('now')
- For "last N hours" use: created_at >= datetime('now', '-N hours')
- For "today" use: created_at >= date('now')
- For "this week" use: created_at >= date('now', '-7 days')
- Use the appropriate timestamp column (created_at, updated_at, start_time, \
date, due_date) based on the table.
- Keep it simple. One condition or AND-combined conditions only.
- If the user mentions a specific title/name, use LIKE with %% wildcards \
(SQLite LIKE is case-insensitive for ASCII by default).
- NEVER use subqueries or joins.\
"""

_INTENT_CLASSIFY_TEMPLATE = FrozenPromptTemplate(
    PROMPTS_DIR / "intent_classify_v1.txt",
)
_INTENT_CLASSIFY_SUFFIX = """\
{tools}

{context_section}User message: {message}

Classify as ONE of:

- "query" — The user is asking about THEIR OWN life, schedule, people, \
or data. Use this whenever the message contains words like: \
my, mine, meu, minha, I, eu, me, our, nosso. \
ALSO use this when the user asks about appointments, meetings, events, \
contacts, messages, notes, emails, reminders, health — even without \
explicit personal pronouns. \
Examples: "when is my next dentist appointment", "quando é minha próxima \
consulta", "what meetings do I have tomorrow", "quais são meus eventos", \
"show my recent messages", "who sent me an email", "próximo compromisso"

- "web_search" — The user wants REAL-TIME or CURRENT information \
that requires searching the internet. This includes: sports scores/schedules, \
news, weather, stock prices, upcoming events, tournament brackets, \
match results, current standings, release dates, or ANY question about \
something happening NOW or in the near future. \
Also use this when the user says a previous answer was wrong/outdated \
("está errado", "wrong", "not right", "outdated", "desatualizado"). \
Examples: "próximo jogo do Sinner", "when does X play next", \
"latest news about Y", "what's the score of Z"

- "action" — The user wants to perform a specific action \
using one of the available tools.

- "general_knowledge" — ONLY for timeless facts, definitions, math, \
translations, or explanations that have NOTHING to do with the user's \
personal life or real-time events. \
Examples: "what is the capital of France?", "explain quantum computing", \
"translate 'hello' to Spanish", "what is 15% of 200?". \
This is ONLY for pure factual questions with NO personal or real-time component.

IMPORTANT RULES:
- If the user mentions "my", "meu", "minha", "I", "eu" → ALWAYS "query"
- If the user asks about appointments, calendar, contacts, messages, \
notes, emails, reminders, health → ALWAYS "query"
- "general_knowledge" is ONLY for pure factual questions with NO \
personal or real-time component
- When in doubt between "query" and "general_knowledge", choose "query"

If "action", also specify which tool_name best matches.
If "web_search", also provide a concise search query:
- Keep the query in the SAME LANGUAGE as the user's message
- Extract the actual subject/entity/topic from the message \
and conversation context
- Remove command words (search, busque, find, procure)
- Resolve pronouns (he/she/dele/dela) using the conversation context. \
If a "Message being replied to" is present, it is the PRIMARY context \
for pronoun resolution — use names and entities from it.
- For recent conversation context, prioritize the MOST RECENT messages; \
older messages are less relevant.
- Example: "Busque os próximos jogos dele" with context about \
João Fonseca → search_query: "João Fonseca próximos jogos tênis"

Respond with ONLY valid JSON:
{{"intent": "query"}}
{{"intent": "web_search", "search_query": "actual search terms"}}
{{"intent": "action", "tool": "tool_name_here"}}
{{"intent": "general_knowledge"}}\
"""


# ----------------------------------------------------------------------
# Per-connector table metadata for action candidate queries
# ----------------------------------------------------------------------

_CONNECTOR_TABLE_INFO: dict[str, dict[str, str]] = {
    "apple-notes": {
        "table": "raw_notes",
        "columns": "title, created_at, updated_at",
        "order_by": "COALESCE(updated_at, created_at) DESC",
    },
    "apple-calendar": {
        "table": "raw_calendar_events",
        "columns": "title, start_time, end_time, location, attendees",
        "order_by": "start_time DESC",
    },
    "apple-contacts": {
        "table": "raw_contacts",
        "columns": "name, phone, email",
        "order_by": "name",
    },
    "apple-mail": {
        "table": "raw_emails",
        "columns": "subject, from_address, date, is_read",
        "order_by": "date DESC",
    },
    "apple-reminders": {
        "table": "raw_reminders",
        "columns": "title, due_date, completed",
        "order_by": "due_date DESC NULLS LAST",
    },
}


def _build_candidate_queries(
    table: str,
    columns: str,
    order_by: str,
    where_clause: str,
) -> list[str]:
    """Build SQL queries for action candidates — targeted first, fallback second.

    sensitivity_tier: N/A
    """
    queries: list[str] = []
    if where_clause:
        queries.append(
            f"SELECT {columns} FROM {table} "
            f"WHERE {where_clause} "
            f"ORDER BY {order_by} LIMIT 20"
        )
    queries.append(
        f"SELECT {columns} FROM {table} "
        f"ORDER BY {order_by} LIMIT 15"
    )
    return queries


def _parse_json_from_llm(text: str) -> dict[str, Any] | None:
    """Best-effort parse of JSON from an LLM response.

    Handles cases where the LLM wraps JSON in markdown code fences.

    sensitivity_tier: N/A
    """
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, TypeError):
        pass

    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(1).strip())
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, TypeError):
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            result = json.loads(text[start:end + 1])
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, TypeError):
            pass

    return None


# ----------------------------------------------------------------------
# Pure helpers — called by BrainAgent methods (during transition) and
# by Brain v2's propose_action tool (Phase B).
# ----------------------------------------------------------------------


def match_action_intent(
    text: str,
    tool_registry: Any,
    *,
    channel_hint: Any = None,
) -> Any | None:
    """Check if text matches an action intent via ``tool_registry``.

    Returns the matched action (a registry-defined object) or ``None``.

    ``channel_hint`` is an optional :class:`ChannelHint` (or any
    object exposing ``.channel`` and ``.confidence``). When present,
    the candidate list is re-ranked / filtered so tools whose
    connector serves the hinted channel are preferred. This prevents
    a ``reply`` request from a WhatsApp message ending up in
    ``reply_email`` just because email's tool happened to score first
    on the verb match. See :mod:`channel_inference` for the rules.

    sensitivity_tier: 1
    """
    if tool_registry is None:
        return None
    matches = tool_registry.match_intent(text)
    if not matches:
        return None
    if channel_hint is not None and getattr(channel_hint, "channel", ""):
        from src.agents.brain.channel_inference import filter_tools_by_channel
        matches = filter_tools_by_channel(matches, channel_hint)
        if not matches:
            return None
    return matches[0]


def generate_action_where_clause(
    question: str,
    table: str,
    columns: str,
    provider: LLMProvider,  # noqa: ARG001 — kept for backwards-compat signature
) -> str:
    """Generate a SQL WHERE clause for ``question`` via the firewall gateway.

    Returns a WHERE clause string (without the ``WHERE`` keyword), or an
    empty string if the LLM fails or returns something suspicious. The
    ``provider`` argument is ignored — provider selection now happens
    inside :func:`chat_via_firewalls` based on the egress firewall's
    decision so the WHERE-clause prompt can't route around the
    privacy policy.

    sensitivity_tier: 1
    """
    prompt = _ACTION_WHERE_TEMPLATE.prefix + (
        _ACTION_WHERE_SUFFIX.format(
            table=table,
            columns=columns,
            question=question,
            today=date.today().isoformat(),
        )
    )
    try:
        resp = chat_via_firewalls(
            [
                {
                    "role": "system",
                    "content": "You generate SQL WHERE clauses.",
                },
                {"role": "user", "content": prompt},
            ],
            agent_id="brain.actions.where",
            lane=Lane.INTERACTIVE,
            agent_max_tier=2,
        )
        clause = resp.content.strip()
        clause = clause.strip("`").strip()
        if clause.upper().startswith("WHERE "):
            clause = clause[6:]
        if any(
            kw in clause.upper()
            for kw in ("SELECT", "DROP", "INSERT", "UPDATE", "DELETE")
        ):
            return ""
        return clause
    except GatewayBlocked as exc:
        logger.info("WHERE clause generation blocked by firewall: %s", exc)
        return ""
    except Exception:  # noqa: BLE001
        logger.debug(
            "LLM WHERE clause generation failed", exc_info=True,
        )
        return ""


def query_action_candidates(
    connector_id: str,
    question: str,
    duckdb: Any,
    provider: LLMProvider,
) -> list[dict[str, Any]]:
    """Query DuckDB for records matching the user's action request.

    ``duckdb`` is the engine instance (typically ``QueryEngine._duck``).
    ``provider`` is used to generate a targeted WHERE clause; falls back
    to a recent-records query on failure.

    Returns the matching rows annotated with a ``_table`` metadata key.

    sensitivity_tier: 2
    """
    table_info = _CONNECTOR_TABLE_INFO.get(connector_id)
    if not table_info:
        return []

    table = table_info["table"]
    columns = table_info["columns"]
    order_by = table_info["order_by"]

    where_clause = generate_action_where_clause(
        question, table, columns, provider,
    )

    for sql in _build_candidate_queries(
        table, columns, order_by, where_clause,
    ):
        try:
            rows = duckdb.query(sql)
            if rows:
                return [{**row, "_table": table} for row in rows]
        except Exception:  # noqa: BLE001
            logger.debug(
                "Action candidate query failed: %s", sql,
                exc_info=True,
            )

    return []


def format_candidates_message(
    candidates: list[dict[str, Any]],
    connector_id: str,  # noqa: ARG001 (reserved for future per-connector formatting)
) -> str:
    """Format action candidate records for display to the user.

    sensitivity_tier: 2
    """
    lines: list[str] = [f"I found {len(candidates)} matching items:\n"]
    for i, row in enumerate(candidates, 1):
        parts = [
            f"{k}={v}" for k, v in row.items()
            if v is not None and k != "_table"
        ]
        lines.append(f"  {i}. {', '.join(parts)}")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Action proposal construction
# ----------------------------------------------------------------------


def resolve_connector_command(
    tool_registry: Any,
    connector_id: str,
) -> tuple[str, tuple[str, ...]]:
    """Look up the MCP server command/args for a connector.

    sensitivity_tier: 1
    """
    if tool_registry is None:
        return ("", ())
    catalog = tool_registry._catalog  # noqa: SLF001
    template = catalog.get(connector_id)
    if template is not None:
        return (template.command, template.args)
    return ("", ())


def fetch_tool_schema(
    mcp_client_factory: Any,
    connector_id: str,
    tool_name: str,
    command: str,
    args: tuple[str, ...],
) -> dict[str, Any] | None:
    """Fetch real ``input_schema`` from the MCP server via ``list_tools``.

    Returns the input_schema dict or ``None`` on failure.

    sensitivity_tier: 1
    """
    if not command or mcp_client_factory is None:
        return None
    try:
        with mcp_client_factory(command, args, 10.0) as client:
            tools = client.list_tools()
            for tool in tools:
                if tool.name == tool_name:
                    return tool.input_schema
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to fetch schema for %s/%s: %s",
            connector_id, tool_name, exc,
        )
    return None


def get_action_data_context(
    connector_id: str,
    question: str,
    duckdb: Any,
    provider: LLMProvider,
) -> str:
    """Query DuckDB for records relevant to the action.

    Provides actual data (titles, dates, names) so the LLM can extract
    correct parameters instead of guessing.

    sensitivity_tier: 2
    """
    rows = query_action_candidates(connector_id, question, duckdb, provider)
    if not rows:
        return ""
    lines: list[str] = [f"Matching {rows[0].get('_table', '')} records:"]
    for row in rows:
        parts = [
            f"{k}={v}" for k, v in row.items()
            if v is not None and k != "_table"
        ]
        lines.append(f"  - {', '.join(parts)}")
    return "\n".join(lines)


def extract_action_params(
    question: str,
    action: Any,
    input_schema: dict[str, Any],
    context_text: str,
    provider: LLMProvider,  # noqa: ARG001 — kept for signature compatibility
) -> tuple[dict[str, Any], list[str]]:
    """Extract tool parameters from the user message via the gateway.

    The ``provider`` argument is ignored — the firewall gateway picks
    the provider based on the egress decision. Action parameters are
    Tier 3 by default because they can contain healthcare / financial
    references; the gateway routes accordingly.

    Returns ``(extracted_params, missing_required_params)``.

    sensitivity_tier: 3
    """
    schema_str = (
        json.dumps(input_schema, indent=2)
        if input_schema else "{}"
    )
    required = list(input_schema.get("required", []))

    context_section = (
        f"Personal context:\n{context_text}" if context_text else ""
    )

    prompt = _PARAM_EXTRACT_TEMPLATE.prefix + (
        _PARAM_EXTRACT_SUFFIX.format(
            tool_name=action.display_name,
            schema=schema_str,
            question=question,
            context_section=context_section,
            today=date.today().isoformat(),
        )
    )

    try:
        resp = chat_via_firewalls(
            [
                {"role": "system", "content": "You extract JSON params."},
                {"role": "user", "content": prompt},
            ],
            agent_id="brain.actions.params",
            lane=Lane.INTERACTIVE,
            agent_max_tier=3,
        )
        raw_text = resp.content.strip()
        extracted = _parse_json_from_llm(raw_text)
        if extracted is not None:
            missing = [
                p for p in required if extracted.get(p) is None
            ]
            return (extracted, missing)
    except GatewayBlocked as exc:
        logger.info(
            "Action parameter extraction blocked by firewall for %s: %s",
            action.tool_name, exc,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "LLM parameter extraction failed for %s: %s",
            action.tool_name, exc,
        )

    return ({}, required)


def _is_create_tool(tool_name: str) -> bool:
    """Return True when ``tool_name`` indicates a creation flow.

    Creation tools ("create_event", "create_note", "create_reminder",
    "create_draft", "schedule_*"…) take new user-supplied values, so
    feeding them existing DB records as "context" only invites the LLM
    to substitute the user's literals with adjacent unrelated records.
    Read / update / delete tools genuinely need that context.

    sensitivity_tier: 1
    """
    if not tool_name:
        return False
    lowered = tool_name.lower()
    return (
        lowered.startswith("create_")
        or lowered.startswith("add_")
        or lowered.startswith("new_")
        or lowered.startswith("schedule_")
        or lowered.startswith("compose_")
        or lowered.startswith("draft_")
    )


# Schema field names that signal the LLM is writing free-form prose.
# Their presence is what makes the judge worth its 5-15s latency —
# without one, the proposal is structural (ids, enums, booleans) and
# the user-literal override already covers the main hallucination class.
_BODY_FIELD_NAMES: frozenset[str] = frozenset({
    "body", "text", "content", "message", "subject",
    "title", "description", "note", "caption",
})


def _judge_needed(tool_name: str, input_schema: dict[str, Any]) -> bool:
    """Decide whether the judge LLM is worth running for this tool.

    Skip when the tool has no body-like field and is a low-stakes
    structural op (delete_, flag_, move_, trash_, play_, stop_,
    search_…). For those, the LLM picks an identifier and the
    deterministic user-value override + schema validation are
    sufficient — the judge's tax (~5-15s of Qwen latency) buys
    nothing. Keep the judge for any tool where the LLM writes prose
    (send_message, send_email, create_event, etc.).

    sensitivity_tier: 1
    """
    schema_props = (input_schema or {}).get("properties") or {}
    if any(k.lower() in _BODY_FIELD_NAMES for k in schema_props):
        return True
    lowered = (tool_name or "").lower()
    structural_prefixes = (
        "delete_", "remove_", "flag_", "unflag_", "move_", "trash_",
        "play_", "stop_", "pause_", "search_", "get_", "list_",
    )
    if any(lowered.startswith(p) for p in structural_prefixes):
        return False
    # Default: keep the safety net. Unknown tools / odd schemas get the
    # judge — the cost of one wrong skip outweighs the latency.
    return True


def _apply_user_value_overrides(
    extracted: dict[str, Any],
    input_schema: dict[str, Any],
    user_values: Any,
) -> tuple[dict[str, Any], bool]:
    """Force-overwrite ``extracted`` with deterministic user values.

    Returns the patched dict plus a bool indicating whether anything
    changed (for audit logging). Only writes a field when:

    1. The user's value is non-empty, AND
    2. The field is part of the tool's input schema (so we never
       invent fields the tool doesn't accept).

    sensitivity_tier: 2
    """
    schema_props = (input_schema or {}).get("properties") or {}
    patched = dict(extracted)
    changed = False
    field_map = (
        ("title", user_values.title),
        # Tools may use either ``start_time`` (Apple bridge) or
        # ``start_date`` / ``starts_at`` (other connectors). Write
        # whichever the schema declares.
        ("start_time", user_values.start_time),
        ("start_date", user_values.start_time),
        ("starts_at", user_values.start_time),
        ("end_time", user_values.end_time),
        ("end_date", user_values.end_time),
        ("ends_at", user_values.end_time),
    )
    for field_name, value in field_map:
        if value is None:
            continue
        if field_name not in schema_props:
            continue
        if patched.get(field_name) != value:
            patched[field_name] = value
            changed = True
    return patched, changed


_RECIPIENT_FIELD_KEYS: tuple[str, ...] = (
    "to", "recipient", "phone", "email", "address",
)


def _channel_from_connector(connector_id: str) -> str | None:
    """Map a connector id to the messaging channel it carries.

    Returns ``None`` for non-messaging connectors (calendar,
    reminders, etc.) so the recipient pipeline cleanly opts out.

    sensitivity_tier: 1
    """
    cid = (connector_id or "").lower()
    if "whatsapp" in cid:
        return "whatsapp"
    if "mail" in cid or "gmail" in cid or "outlook" in cid:
        return "email"
    if "messages" in cid or "imessage" in cid or "sms" in cid:
        return "imessage"
    return None


def _extract_recipient_field(extracted: dict[str, Any]) -> str | None:
    """Pull the recipient value from common field names.

    sensitivity_tier: 3
    """
    for key in _RECIPIENT_FIELD_KEYS:
        value = extracted.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _looks_like_handle(value: str) -> bool:
    """True when ``value`` is already a routable handle (phone / email / JID).

    The resolver only fires on bare names. If the user already
    typed a phone number or the extractor produced a JID we skip the
    disambiguation step — there's nothing left to disambiguate.

    sensitivity_tier: 1
    """
    stripped = value.strip()
    if not stripped:
        return False
    if "@" in stripped:
        return True
    digit_count = sum(1 for ch in stripped if ch.isdigit())
    return digit_count >= 7 and digit_count >= len(stripped) - 5


def _build_recipient_preview(
    *,
    extracted: dict[str, Any],
    connector_id: str,
    db: Any,
) -> dict[str, Any] | None:
    """Resolve the recipient of a messaging/email proposal for the card.

    Looks at the standard ``to`` / ``recipient`` / ``email`` /
    ``phone`` keys on ``extracted``, decides the channel from the
    connector, and looks the candidate up in ``raw_contacts``.
    Returns a preview dict the frontend renders on the confirmation
    card so the user can verify the destination before pressing
    Confirm.

    ``resolved=False`` means the name didn't match any saved contact
    — that's the "to: WhatsApp" bug scenario, where the LLM put a
    channel name (or another garbage string) into the recipient
    field. The card warns the user instead of silently shipping it.

    Returns ``None`` for connectors / tools that don't have a
    recipient field (calendar create, etc.) so the card stays clean.

    sensitivity_tier: 3 (contact details)
    """
    channel = _channel_from_connector(connector_id)
    if channel is None:
        return None

    raw_candidate = _extract_recipient_field(extracted)
    if raw_candidate is None:
        return None

    preview: dict[str, Any] = {
        "channel": channel,
        "input": raw_candidate,
        "name": raw_candidate,
        "phone": None,
        "email": None,
        "resolved": False,
    }

    # Reject obvious garbage (the "to: WhatsApp" failure). A channel
    # name in the recipient field is never a contact.
    lowered = raw_candidate.lower()
    if lowered in {"whatsapp", "email", "mail", "imessage", "sms"}:
        preview["warning"] = (
            f"Recipient looks like a channel name ('{raw_candidate}'), "
            "not a contact — double-check before sending."
        )
        return preview

    if db is None:
        return preview

    try:
        rows = db.query(
            "SELECT name, phone, email FROM raw_contacts "
            "WHERE name = ? "
            "   OR name LIKE ? "
            "   OR phone = ? "
            "   OR email = ? "
            "ORDER BY LENGTH(name) ASC "
            "LIMIT 1",
            [
                raw_candidate,
                f"%{raw_candidate}%",
                raw_candidate,
                raw_candidate,
            ],
        )
    except Exception:  # noqa: BLE001
        logger.debug(
            "recipient preview lookup failed", exc_info=True,
        )
        return preview

    for row in rows or []:
        contact_name = (
            row.get("name") if isinstance(row, dict) else row[0]
        )
        contact_phone = (
            row.get("phone") if isinstance(row, dict) else row[1]
        )
        contact_email = (
            row.get("email") if isinstance(row, dict) else row[2]
        )
        if contact_name:
            preview["name"] = str(contact_name).strip()
        if contact_phone:
            preview["phone"] = str(contact_phone).strip()
        if contact_email:
            preview["email"] = str(contact_email).strip()
        # For each channel, decide whether the lookup gives us the
        # destination identifier we actually need to confirm.
        if channel == "whatsapp" and preview["phone"]:
            preview["resolved"] = True
        elif channel == "email" and preview["email"]:
            preview["resolved"] = True
        elif channel == "imessage" and (
            preview["phone"] or preview["email"]
        ):
            preview["resolved"] = True
        break

    if not preview["resolved"]:
        preview["warning"] = (
            f"No saved contact found for '{raw_candidate}' — "
            "double-check the destination before sending."
        )
    return preview


def _resolve_recipient_name(
    sources: list[dict[str, Any]] | None,
) -> str | None:
    """Pull the recipient's display name from the most recent inbound
    message in ``sources``.

    The reply target is whoever wrote the most recent inbound message
    in the grounded context. ``format_context`` exposes
    ``sender_name`` on raw_messages-derived sources; we prefer that
    over ``sender`` (which is often a phone JID / email address) so
    the downstream contacts lookup hits the canonical name.

    sensitivity_tier: 2
    """
    if not sources:
        return None
    for src in sources:
        is_from_me = src.get("is_from_me")
        if isinstance(is_from_me, bool) and is_from_me:
            continue
        if isinstance(is_from_me, (int, float)) and is_from_me:
            continue
        if isinstance(is_from_me, str) and is_from_me.strip().lower() in {
            "true", "1", "yes",
        }:
            continue
        for key in ("sender_name", "from_name", "name"):
            value = src.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _apply_judge_patches(
    extracted: dict[str, Any],
    schema: dict[str, Any],
    patches: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Merge the judge's ``patches`` into the post-override payload.

    Runs after :func:`_apply_user_value_overrides` so the judge sees
    the user's literal values already in place and only patches what
    the LLM left wrong. Only writes fields present in
    ``schema.properties`` — the judge cannot invent new tool
    parameters. JSON-null patches clear the field; the literal strings
    ``"None"`` / ``"null"`` are coerced to null first (defense in
    depth — the judge prompt forbids them but the primary may still
    inherit them).

    sensitivity_tier: 2
    """
    schema_props = (schema or {}).get("properties") or {}
    patched = dict(extracted)
    applied: list[str] = []
    for key, value in (patches or {}).items():
        if key not in schema_props:
            continue
        if isinstance(value, str) and value.strip().lower() in {
            "none", "null", "",
        }:
            value = None
        if patched.get(key) != value:
            patched[key] = value
            applied.append(key)
    return patched, applied


def build_action_proposal(
    action: Any,
    question: str,
    context_text: str,
    *,
    tool_registry: Any,
    mcp_client_factory: Any,
    duckdb: Any,
    provider: LLMProvider,
    sources: list[dict[str, Any]] | None = None,
    skip_recipient_resolution: bool = False,
    preresolved_arguments: dict[str, Any] | None = None,
) -> ActionProposal | RecipientDisambiguationProposal:
    """Build a complete :class:`ActionProposal` for ``action``.

    Pipeline (in order):

    1. Resolve the tool's command + JSON schema (via MCP fetch if not
       already on the registry entry).
    2. For non-creation tools, enrich the LLM prompt with matching DB
       records. Creation tools skip this step — feeding existing
       records to the LLM during a *create* call invites it to
       substitute the user's literal values with adjacent unrelated
       records ("Play Tennis with Tiago" → "Coffee chat with Sarah").
    3. Run :func:`extract_user_given_values` to pull literal titles
       and explicit date/time references out of the user's prompt
       deterministically.
    4. Run the LLM extractor for everything else.
    5. **Hard-override** the LLM output with the deterministic values
       wherever they exist. The user's literal text wins.
    6. Run the **independent judge** (``action_proposal_judge``) on a
       different LLM family over the post-override payload. The judge
       catches subtle hallucinations the regex extractor doesn't —
       unjustified locations, off-by-one dates, "None"-string leakage.
       Its ``patches`` are applied; ``cannot_recover=true`` surfaces a
       ``⚠ Judge:`` warning in the proposal's description.
       If the judge call fails or its model is unreachable, the
       proposal ships as-is — the judge is a safety net, not a
       blocker.

    sensitivity_tier: 3
    """
    from src.agents.brain.user_value_extractor import extract_user_given_values
    from src.core.profiler import timed_block

    with timed_block("draft_reply.resolve_connector_command"):
        command, args = resolve_connector_command(
            tool_registry, action.connector_id,
        )

    input_schema = getattr(action, "input_schema", None)
    if not input_schema and mcp_client_factory is not None:
        with timed_block("draft_reply.fetch_tool_schema"):
            fetched = fetch_tool_schema(
                mcp_client_factory,
                action.connector_id,
                action.tool_name,
                command,
                args,
            )
        if fetched:
            input_schema = fetched

    # 2. Pull existing DB records only for tools that genuinely
    # reference existing data (update_/delete_/reply_/search_…).
    if not _is_create_tool(getattr(action, "tool_name", "") or ""):
        with timed_block("draft_reply.get_action_data_context"):
            data_context = get_action_data_context(
                action.connector_id, question, duckdb, provider,
            )
        if data_context:
            context_text = (
                f"{context_text}\n\n"
                f"Relevant database records:\n{data_context}"
            )

    # 2b. Reply-language anchor — when the action is replying to or
    # composing a follow-up to an inbound message, the extractor and
    # the judge both need to see the inbound text so they can match
    # the reply language to it. We don't detect the language here
    # (the LLMs do that); we just expose the snippet.
    from src.agents.brain.channel_inference import (
        infer_inbound_language_hint,
    )
    with timed_block("draft_reply.infer_inbound_language_hint"):
        inbound_snippet = infer_inbound_language_hint(sources, context_text)
    if inbound_snippet:
        context_text = (
            f"{context_text}\n\n"
            f"Most recent inbound message (write the reply in the "
            f"SAME language as this text):\n{inbound_snippet}"
        )

    # 2c. Recipient profile — relationship + recent outbound samples.
    # If the inbound message has a known sender we can name, look up
    # the user's curated relationship label and a handful of recent
    # outbound messages to them. Lets the extractor + judge calibrate
    # tone (intimate for spouse, formal for colleague) and get
    # grammatical number right (singular for a named single person).
    try:
        from src.agents.brain.recipient_profile import (
            format_profile_for_prompt,
            lookup_recipient_profile,
        )
        with timed_block("draft_reply.lookup_recipient_profile"):
            recipient_name = _resolve_recipient_name(sources)
            if recipient_name and duckdb is not None:
                profile = lookup_recipient_profile(recipient_name, duckdb)
                profile_block = format_profile_for_prompt(profile)
                if profile_block:
                    context_text = f"{context_text}\n\n{profile_block}"
    except Exception:  # noqa: BLE001
        logger.debug(
            "recipient_profile lookup failed for %s — proposal still ships",
            getattr(action, "tool_name", "<unknown>"),
            exc_info=True,
        )

    # 3. Deterministic pre-extraction of literal user values.
    with timed_block("draft_reply.extract_user_given_values"):
        user_values = extract_user_given_values(question)

    # 4. LLM extraction for everything we can't parse deterministically.
    with timed_block("draft_reply.extract_action_params_llm"):
        extracted, missing = extract_action_params(
            question, action, input_schema or {}, context_text, provider,
        )

    # 5. Hard override — user literals win over LLM output. This is
    # the structural fix that prevents the "Coffee chat with Sarah"
    # class of hallucination.
    extracted, overridden = _apply_user_value_overrides(
        extracted, input_schema or {}, user_values,
    )
    if overridden:
        logger.info(
            "build_action_proposal: applied user-value overrides for %s",
            action.tool_name,
        )

    # 6. Independent judge — runs on a different LLM family from the
    # primary over the *post-override* payload. Catches the subtle
    # hallucinations regex can't (unjustified locations, off-by-one
    # dates, "None"-string leakage). Safety net, not a blocker: a
    # judge that crashes the create flow when its model is offline
    # would be worse than no judge at all.
    #
    # Skip the judge entirely for low-stakes structural tools — the
    # ones whose schema has no LLM-creative body field (delete_, flag_,
    # move_, trash_, play_, search_…). For those, the LLM picks an
    # identifier and that's it; user-literal override already handles
    # the title/id case, and the judge's 5-15s tax buys nothing.
    judge_warning: str = ""
    if _judge_needed(action.tool_name, input_schema or {}):
        try:
            from src.agents.action_proposal_judge import judge_action_proposal
            with timed_block("draft_reply.judge_action_proposal_llm"):
                verdict = judge_action_proposal(
                    user_message=question,
                    tool_name=action.tool_name,
                    tool_schema=input_schema or {},
                    proposed_arguments=extracted,
                )
        except Exception:  # noqa: BLE001
            logger.debug(
                "action_proposal_judge import/call failed", exc_info=True,
            )
            verdict = None
    else:
        logger.debug(
            "judge skipped for low-stakes tool %s (no body-like field)",
            action.tool_name,
        )
        verdict = None

    if verdict is not None and not verdict.ok:
        if verdict.patches:
            extracted, applied = _apply_judge_patches(
                extracted, input_schema or {}, verdict.patches,
            )
            if applied:
                logger.info(
                    "judge patched %d field(s) of %s proposal: %s",
                    len(applied), action.tool_name, ", ".join(applied),
                )
        if verdict.cannot_recover:
            # Judge believes the user's message is too ambiguous to
            # act on. We still surface the proposal (so the user sees
            # the Cancel button instead of the agent silently dropping
            # the request) but flag the warning loudly.
            judge_warning = (
                "⚠ Judge: " + (verdict.reasons[0] if verdict.reasons
                else "the request is too ambiguous to act on safely.")
            )

    # 7. Defense-in-depth rehydration. ``chat_via_firewalls``
    # rehydrates LLM output through a per-call ``RedactionMap`` that
    # only sees placeholders newly minted during that single call.
    # If the primary extractor inherits a placeholder from the
    # context (because a name was registered earlier in the session
    # but didn't appear in this call's input), the per-call map
    # won't know to reverse it and the user would see
    # ``__PERSON_2726__`` on the confirmation card. Run every stringy
    # arg through the process-wide registry as a safety net — this
    # is the same instance ``redact_with_registry`` writes to, so
    # any name we've ever placeholdered is restorable here.
    try:
        from src.models.redaction_registry import (
            default_redaction_registry,
        )
        with timed_block("draft_reply.rehydration"):
            registry = default_redaction_registry()
            rehydrated_args: dict[str, Any] = {}
            for key, value in extracted.items():
                if isinstance(value, str) and value:
                    rehydrated_args[key] = registry.rehydrate(value)
                else:
                    rehydrated_args[key] = value
            extracted = rehydrated_args
    except Exception:  # noqa: BLE001
        # Surface at WARNING (not DEBUG) so production logs catch it.
        # A silent swallow here is what lets a bare placeholder ride
        # through to _build_recipient_preview and on to the user.
        logger.warning(
            "registry rehydration failed for %s — recipient resolver "
            "will run on the un-rehydrated value as a fallback",
            action.tool_name, exc_info=True,
        )

    # Recompute missing required params after override + judge passes.
    required = list((input_schema or {}).get("required", []))
    missing = [p for p in required if extracted.get(p) is None]

    display = action.display_name
    param_summary = ", ".join(
        f"{k}={v!r}" for k, v in extracted.items()
        if v is not None
    )
    base_description = (
        f"{display}: {param_summary}" if param_summary
        else f"{display} (parameters to be confirmed)"
    )
    description = (
        f"{base_description}\n\n{judge_warning}"
        if judge_warning else base_description
    )

    # 7b. Recipient disambiguation — for messaging tools, look up the
    # bare name in the contacts marts / Apple MCP and hand the user
    # a candidate-picker before we let the action card show. Bypassed
    # only on the resumption path where the user has already chosen
    # a candidate (``skip_recipient_resolution``).
    if not skip_recipient_resolution:
        channel = _channel_from_connector(action.connector_id)
        raw_recipient = _extract_recipient_field(extracted)
        if channel and raw_recipient and not _looks_like_handle(raw_recipient):
            from src.agents.brain.recipient_resolver import (
                resolve_recipient,
            )
            with timed_block("draft_reply.resolve_recipient"):
                resolution = resolve_recipient(
                    raw_recipient,
                    channel,
                    duckdb,
                    tool_registry=tool_registry,
                    mcp_client_factory=mcp_client_factory,
                )
            return RecipientDisambiguationProposal(
                proposal_id=str(uuid.uuid4()),
                connector_id=action.connector_id,
                connector_name=action.connector_name,
                tool_name=action.tool_name,
                display_name=action.display_name,
                channel=channel,
                original_name=resolution.original_name,
                candidates=[
                    {
                        "name": c.name,
                        "handle": c.handle,
                        "relationship": c.relationship,
                        "active_topic": c.active_topic,
                        "topic_importance": c.topic_importance,
                        "notification_priority": c.notification_priority,
                        "source": c.source,
                    }
                    for c in resolution.candidates
                ],
                draft_arguments=dict(extracted),
                command=command,
                args=args,
                question=question,
                context_text=context_text,
            )
    elif preresolved_arguments is not None:
        extracted = {**extracted, **preresolved_arguments}

    # 8. Recipient preview — resolves the recipient to a saved contact
    # for messaging/email tools so the confirmation card can show
    # "To: Elmara · +55 11 99999-1234" instead of a bare name (or
    # the channel name when the LLM fumbled the extraction).
    with timed_block("draft_reply.build_recipient_preview"):
        recipient_preview = _build_recipient_preview(
            extracted=extracted,
            connector_id=action.connector_id,
            db=duckdb,
        )

    return ActionProposal(
        proposal_id=str(uuid.uuid4()),
        connector_id=action.connector_id,
        connector_name=action.connector_name,
        tool_name=action.tool_name,
        display_name=display,
        arguments=extracted,
        description=description,
        missing_params=missing,
        command=command,
        args=args,
        risk="low" if is_low_risk_tool(action.tool_name) else "high",
        recipient_preview=recipient_preview,
    )


def proposal_to_chunk(
    proposal: ActionProposal,
    *,
    latency_ms: float,
) -> dict[str, Any]:
    """Render an :class:`ActionProposal` as an ``action_proposal`` chunk.

    Matches the legacy ``BrainAgent.ask_stream`` wire format the frontend
    already consumes (see ``useStreamingChat.ts``).

    sensitivity_tier: 2
    """
    return {
        "type": "action_proposal",
        "proposal": asdict(proposal),
        "latency_ms": latency_ms,
    }


def disambiguation_proposal_to_chunk(
    proposal: RecipientDisambiguationProposal,
    *,
    latency_ms: float,
) -> dict[str, Any]:
    """Render a recipient-disambiguation proposal as a stream chunk.

    sensitivity_tier: 3
    """
    return {
        "type": "recipient_disambiguation",
        "proposal": asdict(proposal),
        "latency_ms": latency_ms,
    }


def resume_action_from_disambiguation(
    *,
    disambiguation: dict[str, Any],
    candidate: dict[str, Any],
    duckdb: Any,
) -> ActionProposal:
    """Convert a chosen disambiguation candidate into an ActionProposal.

    Deterministic — no LLM call. The disambiguation proposal already
    carries every piece of state the action needs (command, args,
    tool/connector identifiers, draft_arguments); the resumption
    just merges the candidate's handle into the recipient field and
    rebuilds the recipient preview for the confirmation card.

    sensitivity_tier: 3
    """
    channel = str(disambiguation.get("channel") or "")
    handle_field = _recipient_field_for_channel(channel)
    handle = str(candidate.get("handle") or "").strip()
    if not handle:
        msg = "Selected candidate has no handle for this channel"
        raise ValueError(msg)
    draft = dict(disambiguation.get("draft_arguments") or {})
    arguments = {
        **draft,
        handle_field: handle,
    }
    if candidate.get("name"):
        arguments.setdefault("recipient_display_name", candidate["name"])
    recipient_preview = _build_recipient_preview(
        extracted={**arguments, "to": candidate.get("name") or handle},
        connector_id=str(disambiguation.get("connector_id") or ""),
        db=duckdb,
    )
    display_name = str(disambiguation.get("display_name") or "")
    param_summary = ", ".join(
        f"{k}={v!r}" for k, v in arguments.items() if v is not None
    )
    description = (
        f"{display_name}: {param_summary}" if param_summary else display_name
    )
    return ActionProposal(
        proposal_id=str(uuid.uuid4()),
        connector_id=str(disambiguation.get("connector_id") or ""),
        connector_name=str(disambiguation.get("connector_name") or ""),
        tool_name=str(disambiguation.get("tool_name") or ""),
        display_name=display_name,
        arguments=arguments,
        description=description,
        missing_params=[],
        command=str(disambiguation.get("command") or ""),
        args=tuple(disambiguation.get("args") or ()),
        risk="low" if is_low_risk_tool(
            str(disambiguation.get("tool_name") or ""),
        ) else "high",
        recipient_preview=recipient_preview,
    )


def _recipient_field_for_channel(channel: str) -> str:
    """Default arg key the connector expects for this channel.

    sensitivity_tier: 1
    """
    if channel == "email":
        return "to"
    return "to"
