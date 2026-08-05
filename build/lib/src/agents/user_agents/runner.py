"""Runtime helpers that fire user-authored agents.

Two execution paths share this module:

- **Scheduled batch** — ``run_user_agent_batch`` walks every unprocessed
  item from the agent's bound data tools (the ``data``-typed entries
  in ``enabled_mcp_tools``) and invokes the agent once per item. Uses
  ``_user_agent_processed_items`` as the per-agent cursor so the next
  tick (or next ``Run now`` click) only sees what arrived in the
  meantime. After the per-item loop, if ``delivery_tools`` is
  non-empty and at least one item was processed successfully, an
  LLM-summarized digest is dispatched to each delivery tool via the
  post-batch hook.
- **Scheduled generic** — ``run_user_agent_generic`` mirrors today's
  behavior for agents with no data tools: one invocation per tick
  with a generic Portuguese trigger string.

``get_user_agent_status`` is the read side that powers the Agents page
schedule strip: next-run countdown, last-run summary, pending count.

sensitivity_tier: 2
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.core.db_helpers import ensure_tables, table_exists
from src.extensions.cron import cron_matches

if TYPE_CHECKING:
    from src.core.data_layer import DataLayer

logger = logging.getLogger(__name__)

# Cap batch size per (agent, source) to bound LLM cost on a single
# tick. Anything not picked up here is processed on the next tick —
# the cursor in ``_user_agent_processed_items`` keeps us monotonic.
_BATCH_LIMIT_PER_SOURCE = 50

# Truncate stored agent output to keep ``output_text`` rows compact —
# the full text already lives in ``deep_agent_runs`` when present.
_OUTPUT_TEXT_LIMIT = 4096

# Match the file used by ``cmd_run_scheduled_agents`` so "last run"
# stays consistent between scheduled-tick and run-now paths.
_SCHEDULE_STATE_PATH = (
    Path.home() / ".arandu" / "data" / "agent_schedule_state.json"
)

_NEXT_RUN_LOOKAHEAD_MINUTES = 60 * 24 * 7  # 1 week


@dataclass
class BatchRunSummary:
    """Outcome of one batch / generic run.

    ``processed`` is how many items the agent saw this invocation. For
    the generic path it is always 0 or 1 — the agent fires once with no
    item context. ``delivery_calls`` records each post-batch delivery
    attempt (one entry per tool in ``row.delivery_tools``) so the UI
    can surface partial failure.

    sensitivity_tier: 2
    """

    agent_id: str
    mode: str  # "batch" | "generic"
    checked: int = 0
    processed: int = 0
    errors: int = 0
    skipped: int = 0
    run_ids: list[str] = field(default_factory=list)
    error_messages: list[str] = field(default_factory=list)
    delivery_calls: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "mode": self.mode,
            "checked": self.checked,
            "processed": self.processed,
            "errors": self.errors,
            "skipped": self.skipped,
            "run_ids": list(self.run_ids),
            "error_messages": list(self.error_messages),
            "delivery_calls": list(self.delivery_calls),
        }


# ---------------------------------------------------------------------------
# Schema for the per-agent cursor table.
# ---------------------------------------------------------------------------

_PROCESSED_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS _user_agent_processed_items (
    agent_id      VARCHAR NOT NULL,
    source_table  VARCHAR NOT NULL,
    item_id       VARCHAR NOT NULL,
    connector_id  VARCHAR NOT NULL,
    processed_at  TEXT DEFAULT CURRENT_TIMESTAMP,
    run_id        VARCHAR,
    status        VARCHAR,
    output_text   TEXT,
    PRIMARY KEY (agent_id, source_table, item_id)
)
"""

_PROCESSED_TABLE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_uapi_agent
    ON _user_agent_processed_items (agent_id, processed_at DESC)
"""


def _ensure_processed_table(db_engine: Any) -> None:
    ensure_tables(db_engine, [_PROCESSED_TABLE_DDL, _PROCESSED_TABLE_INDEX])


def seed_existing_items(db_engine: Any, agent_id: str, tool_ids: list[str]) -> int:
    """Mark all existing source items as processed so the agent starts fresh.

    Returns the number of items seeded.

    sensitivity_tier: 1
    """
    _ensure_processed_table(db_engine)
    total = 0
    for tool_id in tool_ids:
        source_table = _data_tool_target_table(tool_id)
        if source_table is None or not table_exists(db_engine, source_table):
            continue
        connector_id = tool_id.split(":")[0] if ":" in tool_id else ""
        id_col = "id"
        rows = db_engine.query(
            f"SELECT CAST({id_col} AS VARCHAR) AS item_id FROM {source_table}",  # noqa: S608
        )
        for r in rows:
            db_engine.execute(
                """
                INSERT OR IGNORE INTO _user_agent_processed_items (
                    agent_id, source_table, item_id, connector_id,
                    status
                ) VALUES (?, ?, ?, ?, 'seeded')
                """,
                [agent_id, source_table, r["item_id"], connector_id],
            )
            total += 1
    return total


# ---------------------------------------------------------------------------
# Catalog-driven data-tool resolution
# ---------------------------------------------------------------------------


def _catalog() -> Any:
    """Return a memoised :class:`ConnectorCatalog` instance.

    Loading the catalog parses the bundled JSON file; the runner can
    invoke this several times per tick, so memoise on the module.

    sensitivity_tier: 1
    """
    cached = getattr(_catalog, "_instance", None)
    if cached is None:
        from src.extensions.connectors.catalog import ConnectorCatalog
        cached = ConnectorCatalog()
        _catalog._instance = cached  # type: ignore[attr-defined]
    return cached


def _data_tool_target_table(tool_id: str) -> str | None:
    """Return the ``target_table`` of a catalog data tool, or ``None``.

    Accepts a ``"connector_id:tool_name"`` id and looks the tool up in
    the catalog. Returns ``None`` when the tool is unknown or its
    type is not ``"data"`` — callers treat that as "skip this binding,
    it isn't a viable source on this build".

    sensitivity_tier: 1
    """
    if ":" not in tool_id:
        return None
    connector_id, tool_name = tool_id.split(":", 1)
    template = _catalog().get(connector_id)
    if template is None:
        return None
    for tool in template.tools:
        if tool.tool_name == tool_name and tool.tool_type == "data":
            return tool.target_table
    return None


def data_tool_ids_for_row(row: Any) -> list[str]:
    """Return the subset of ``row.enabled_mcp_tools`` that are data tools.

    Used by both the batch runner and the status endpoint so the same
    rule decides "is this agent a source-driven batch agent or a
    generic one?". Public (no underscore) because ``cmd_agents_run_now``
    also routes on it.

    sensitivity_tier: 1
    """
    out: list[str] = []
    for tool_id in row.enabled_mcp_tools:
        if _data_tool_target_table(tool_id) is not None:
            out.append(tool_id)
    return out


# ---------------------------------------------------------------------------
# Item formatting
# ---------------------------------------------------------------------------


def _format_email(row: dict[str, Any]) -> str:
    subject = (row.get("subject") or "").strip()
    sender = (row.get("from_address") or row.get("sender") or "").strip()
    body = (
        row.get("body_preview")
        or row.get("body")
        or row.get("content")
        or ""
    ).strip()
    parts = [f"De: {sender}" if sender else None,
             f"Assunto: {subject}" if subject else None,
             "",
             body]
    return "\n".join(p for p in parts if p is not None)


def _format_message(row: dict[str, Any]) -> str:
    sender = (
        row.get("sender_name")
        or row.get("sender")
        or "desconhecido"
    )
    content = (row.get("content") or "").strip()
    return f"De: {sender}\n\n{content}"


def _format_item(source_table: str, row: dict[str, Any]) -> str:
    if source_table == "raw_emails":
        return _format_email(row)
    return _format_message(row)


# ---------------------------------------------------------------------------
# Unprocessed fetch
# ---------------------------------------------------------------------------


def _fetch_unprocessed_emails(
    db_engine: Any, agent_id: str, limit: int,
) -> list[dict[str, Any]]:
    if not table_exists(db_engine, "raw_emails"):
        return []
    rows = db_engine.query(
        """
        SELECT e.id,
               e.from_address,
               e.subject,
               e.body_preview,
               e.date
        FROM raw_emails e
        LEFT JOIN _user_agent_processed_items p
            ON p.agent_id = ?
           AND p.source_table = 'raw_emails'
           AND p.item_id = CAST(e.id AS VARCHAR)
        WHERE p.item_id IS NULL
        ORDER BY e.date ASC
        LIMIT ?
        """,
        [agent_id, limit],
    )
    for r in rows:
        r["_table"] = "raw_emails"
    return rows


def _fetch_unprocessed_messages(
    db_engine: Any, agent_id: str, limit: int,
) -> list[dict[str, Any]]:
    if not table_exists(db_engine, "raw_messages"):
        return []
    from src.core.db_helpers import get_table_columns

    cols = get_table_columns(db_engine, "raw_messages")
    sender_col = "sender_name" if "sender_name" in cols else "sender"
    rows = db_engine.query(
        f"""
        SELECT m.id,
               m.sender,
               m.{sender_col} AS sender_name,
               m.content,
               m.timestamp
        FROM raw_messages m
        LEFT JOIN _user_agent_processed_items p
            ON p.agent_id = ?
           AND p.source_table = 'raw_messages'
           AND p.item_id = CAST(m.id AS VARCHAR)
        WHERE p.item_id IS NULL
          AND m.is_from_me = 0
        ORDER BY m.timestamp ASC
        LIMIT ?
        """,
        [agent_id, limit],
    )
    for r in rows:
        r["_table"] = "raw_messages"
    return rows


def _count_unprocessed_for_source(
    db_engine: Any, agent_id: str, source_table: str,
) -> int:
    if not table_exists(db_engine, source_table):
        return 0
    if source_table == "raw_emails":
        rows = db_engine.query(
            """
            SELECT COUNT(*) AS n
            FROM raw_emails e
            LEFT JOIN _user_agent_processed_items p
                ON p.agent_id = ?
               AND p.source_table = 'raw_emails'
               AND p.item_id = CAST(e.id AS VARCHAR)
            WHERE p.item_id IS NULL
            """,
            [agent_id],
        )
    else:
        rows = db_engine.query(
            """
            SELECT COUNT(*) AS n
            FROM raw_messages m
            LEFT JOIN _user_agent_processed_items p
                ON p.agent_id = ?
               AND p.source_table = 'raw_messages'
               AND p.item_id = CAST(m.id AS VARCHAR)
            WHERE p.item_id IS NULL
              AND m.is_from_me = 0
            """,
            [agent_id],
        )
    if not rows:
        return 0
    return int(rows[0].get("n") or 0)


# ---------------------------------------------------------------------------
# Cursor write
# ---------------------------------------------------------------------------


def _record_processed(
    db_engine: Any,
    *,
    agent_id: str,
    source_table: str,
    item_id: str,
    connector_id: str,
    run_id: str | None,
    status: str,
    output_text: str | None,
) -> None:
    truncated = (
        (output_text[:_OUTPUT_TEXT_LIMIT] + "…")
        if output_text and len(output_text) > _OUTPUT_TEXT_LIMIT
        else output_text
    )
    db_engine.execute(
        """
        INSERT OR REPLACE INTO _user_agent_processed_items (
            agent_id, source_table, item_id, connector_id,
            run_id, status, output_text
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            agent_id, source_table, item_id, connector_id,
            run_id, status, truncated,
        ],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_user_agent_batch(
    layer: DataLayer, agent_id: str,
) -> BatchRunSummary:
    """Process every unprocessed item from this agent's data tools.

    One LLM invocation per item. Errors mark the item processed (with
    ``status='error'``) so a poison message never hot-loops the cron.
    After the loop, if ``delivery_tools`` is non-empty and
    ``processed >= 1``, dispatch the post-batch delivery hook.

    sensitivity_tier: 3
    """
    from src.agents.core.registry import get_agent
    from src.agents.user_agents.store import UserAgentStore

    store = UserAgentStore()
    try:
        row = store.get(agent_id)
    finally:
        store.close()

    if row is None:
        return BatchRunSummary(
            agent_id=agent_id, mode="batch",
            error_messages=[f"{agent_id}: not found"],
        )

    data_tool_ids = data_tool_ids_for_row(row)
    if not data_tool_ids:
        # Caller should have routed to ``run_user_agent_generic`` —
        # but be defensive: fall back rather than no-op.
        return run_user_agent_generic(layer, agent_id)

    definition = get_agent(agent_id)
    if definition is None or definition.factory is None:
        return BatchRunSummary(
            agent_id=agent_id, mode="batch",
            error_messages=[f"{agent_id}: not registered"],
        )

    db_engine = layer.duckdb
    _ensure_processed_table(db_engine)

    summary = BatchRunSummary(agent_id=agent_id, mode="batch")
    agent = definition.factory()
    # Collect successful per-item outputs so the post-batch hook can
    # summarise them. Errored items intentionally do not contribute.
    successful_outputs: list[tuple[str, str]] = []

    for tool_id in data_tool_ids:
        source_table = _data_tool_target_table(tool_id)
        if source_table is None:
            summary.skipped += 1
            summary.error_messages.append(
                f"unknown data tool: {tool_id}",
            )
            continue
        connector_id = tool_id.split(":", 1)[0]

        if source_table == "raw_emails":
            items = _fetch_unprocessed_emails(
                db_engine, agent_id, _BATCH_LIMIT_PER_SOURCE,
            )
        else:
            items = _fetch_unprocessed_messages(
                db_engine, agent_id, _BATCH_LIMIT_PER_SOURCE,
            )

        summary.checked += len(items)
        for item in items:
            item_id = str(item.get("id", ""))
            if not item_id:
                summary.skipped += 1
                continue
            trigger = _format_item(source_table, item)
            try:
                record = agent.run(trigger)
                if record.error is None:
                    status = "success"
                    summary.processed += 1
                    output_text = (
                        str(record.output) if record.output is not None
                        else None
                    )
                    if output_text:
                        successful_outputs.append((item_id, output_text))
                else:
                    status = "error"
                    summary.errors += 1
                    summary.error_messages.append(
                        f"{item_id}: {record.error}",
                    )
                    output_text = record.error
                run_id_value: str | None = None
            except Exception as exc:  # noqa: BLE001
                status = "error"
                summary.errors += 1
                summary.error_messages.append(f"{item_id}: {exc}")
                output_text = str(exc)
                run_id_value = None

            try:
                _record_processed(
                    db_engine,
                    agent_id=agent_id,
                    source_table=source_table,
                    item_id=item_id,
                    connector_id=connector_id,
                    run_id=run_id_value,
                    status=status,
                    output_text=output_text,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to record processed item %s: %s",
                    item_id, exc,
                )

    if summary.processed >= 1 and row.delivery_tools:
        _post_batch_deliver(agent, row, successful_outputs, summary)

    return summary


# ---------------------------------------------------------------------------
# Post-batch delivery hook
# ---------------------------------------------------------------------------


_DELIVERY_FALLBACK_FIELD = "text"


def _post_batch_deliver(
    agent: Any,
    row: Any,
    outputs: list[tuple[str, str]],
    summary: BatchRunSummary,
) -> None:
    """Summarize the batch outputs and dispatch each delivery tool.

    Fires only when at least one item was processed successfully.
    Failures (in either the LLM summary call or any individual
    delivery dispatch) are recorded on the summary but never raise —
    per-item processing is already committed.

    sensitivity_tier: 3
    """
    try:
        summary_text = _summarize_outputs_for_delivery(agent, row, outputs)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Delivery summary failed for %s: %s", row.agent_id, exc,
        )
        summary.errors += 1
        summary.error_messages.append(
            f"delivery summary failed: {exc}",
        )
        for tool_id in row.delivery_tools:
            summary.delivery_calls.append({
                "tool_id": tool_id,
                "status": "error",
                "error": f"summary failed: {exc}",
            })
        return

    for tool_id in row.delivery_tools:
        try:
            target_args = dict(row.delivery_targets.get(tool_id, {}))
            args = _coerce_delivery_args(tool_id, summary_text, target_args)
            result = _invoke_delivery_tool(tool_id, args)
            summary.delivery_calls.append({
                "tool_id": tool_id,
                "status": "success",
                "error": None,
                "result_preview": _truncate_for_log(result),
            })
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Delivery dispatch failed for %s via %s: %s",
                row.agent_id, tool_id, exc,
            )
            summary.errors += 1
            summary.error_messages.append(
                f"delivery via {tool_id}: {exc}",
            )
            summary.delivery_calls.append({
                "tool_id": tool_id,
                "status": "error",
                "error": str(exc),
            })


def _summarize_outputs_for_delivery(
    agent: Any,
    row: Any,
    outputs: list[tuple[str, str]],
) -> str:
    """Run one LLM call against ``agent`` to produce a delivery digest.

    Reuses the user agent so the digest inherits its voice and tool
    access (e.g. ``recall_context`` for relevant brain memory). The
    digest is consumed by the runner — delivery tools are invoked by
    :func:`_post_batch_deliver`, not by the LLM, so even if the model
    decides to call other tools during summarisation we still
    guarantee exactly one delivery per tool per tick.

    sensitivity_tier: 3
    """
    body_lines = [
        f"Você acabou de processar {len(outputs)} item(ns) novos. "
        "Gere UMA mensagem curta de entrega para o destinatário "
        "(formato livre, em prosa, sem listar IDs).",
        "",
        f"Papel deste agente: {row.description or row.name}",
        "",
        "Saída por item:",
    ]
    for item_id, output in outputs:
        body_lines.append(f"[{item_id}] {output}")

    record = agent.run("\n".join(body_lines))
    if record.error is not None:
        raise RuntimeError(record.error)
    if record.output is None:
        raise RuntimeError("agent produced no output for delivery digest")
    text = getattr(record.output, "answer", None)
    if not isinstance(text, str) or not text.strip():
        text = str(record.output)
    return text.strip()


def _coerce_delivery_args(
    tool_id: str,
    summary_text: str,
    target_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Map ``summary_text`` into the delivery tool's input schema.

    ``target_args`` is the agent's per-tool static override
    (``row.delivery_targets[tool_id]``) and takes precedence — those
    keys are kept as-is. The heuristic then fills the FIRST string
    field from ``input_schema.required`` (then other string
    properties) that is not already set, using ``summary_text``.

    Falls back to ``{"text": summary_text}`` (merged with
    ``target_args``) when the schema is empty or has no string
    fields — the bridge can error if that doesn't match, and the
    failure is recorded in the delivery summary without aborting
    other deliveries.

    sensitivity_tier: 1
    """
    out: dict[str, Any] = dict(target_args or {})
    if ":" not in tool_id:
        out.setdefault(_DELIVERY_FALLBACK_FIELD, summary_text)
        return out
    connector_id, tool_name = tool_id.split(":", 1)
    template = _catalog().get(connector_id)
    if template is None:
        out.setdefault(_DELIVERY_FALLBACK_FIELD, summary_text)
        return out
    schema: dict[str, Any] | None = None
    for tool in template.tools:
        if tool.tool_name == tool_name and tool.tool_type == "action":
            schema = tool.input_schema or {}
            break
    if not schema:
        out.setdefault(_DELIVERY_FALLBACK_FIELD, summary_text)
        return out
    properties: dict[str, Any] = schema.get("properties") or {}
    required: list[str] = list(schema.get("required") or [])
    walk_order: list[str] = required + [
        k for k in properties.keys() if k not in required
    ]
    for prop_name in walk_order:
        if prop_name in out:
            continue
        prop_schema = properties.get(prop_name) or {}
        if prop_schema.get("type") == "string":
            out[prop_name] = summary_text
            return out
    out.setdefault(_DELIVERY_FALLBACK_FIELD, summary_text)
    return out


def _invoke_delivery_tool(
    tool_id: str, arguments: dict[str, Any],
) -> str:
    """Dispatch ``tool_id`` through the appropriate transport.

    ``whatsapp:send_message`` bypasses MCPClient and queues a request
    directly on the running listener — only one Baileys connection
    per phone is allowed, so spawning a second subprocess via stdio
    MCP would conflict with the persistent listener that the rest of
    the app relies on. Other tools go through MCPClient as before.

    Raises on failure (caught by ``_post_batch_deliver``).

    sensitivity_tier: 2
    """
    if ":" not in tool_id:
        raise ValueError(
            f"invalid delivery tool id: {tool_id!r} "
            "(expected connector:tool)",
        )
    if tool_id == "whatsapp:send_message":
        return _invoke_whatsapp_send(arguments)
    connector_id, tool_name = tool_id.split(":", 1)
    from src.extensions.mcp.client import MCPClient

    client = MCPClient(connector_id=connector_id)
    result = client.call_tool(tool_name, arguments)
    return str(result)


def _invoke_whatsapp_send(arguments: dict[str, Any]) -> str:
    """Queue a WhatsApp text send via the persistent listener IPC.

    ``arguments["to"]`` accepts the literal sentinel ``"self"``
    (resolves to ``<self_lid>@lid`` for the self-chat thread,
    falling back to ``<self_jid>@s.whatsapp.net``) or any JID/phone
    the listener can route to.

    sensitivity_tier: 2
    """
    from src.extensions.bridges.whatsapp.listener import (
        send_text_via_running_listener,
    )
    from src.extensions.bridges.whatsapp.paths import (
        resolve_self_jid,
        resolve_self_lid,
    )

    to_raw = str(arguments.get("to", "")).strip()
    text = str(arguments.get("text", "")).strip()
    if not text:
        raise RuntimeError("whatsapp:send_message requires non-empty 'text'")
    if not to_raw:
        raise RuntimeError("whatsapp:send_message requires 'to'")

    if to_raw.lower() == "self":
        self_lid = resolve_self_lid()
        if self_lid:
            target = f"{self_lid}@lid"
        else:
            self_jid = resolve_self_jid()
            if not self_jid:
                raise RuntimeError(
                    "whatsapp:send_message to='self' but listener has "
                    "not resolved the self JID yet",
                )
            target = f"{self_jid}@s.whatsapp.net"
    else:
        target = to_raw

    response = send_text_via_running_listener(
        to=target, message=text, timeout_seconds=15.0,
    )
    if response is None:
        raise RuntimeError("whatsapp listener is not running")
    status = str(response.get("status") or "").lower()
    if status != "sent":
        err = response.get("error") or response
        raise RuntimeError(f"whatsapp send failed: {err}")
    return f"sent (message_id={response.get('message_id')})"


def _truncate_for_log(value: str, limit: int = 200) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "…"


def run_user_agent_generic(
    layer: DataLayer, agent_id: str,  # noqa: ARG001
) -> BatchRunSummary:
    """Fire a sourceless user agent once with the generic trigger.

    Mirrors the legacy ``_tick_scheduled_user_agents`` body so the
    sourceless code path is unchanged from a user perspective — only
    its location moves.

    sensitivity_tier: 1
    """
    from src.agents.core.registry import get_agent
    from src.agents.user_agents.store import UserAgentStore

    store = UserAgentStore()
    try:
        row = store.get(agent_id)
    finally:
        store.close()

    if row is None:
        return BatchRunSummary(
            agent_id=agent_id, mode="generic",
            error_messages=[f"{agent_id}: not found"],
        )

    definition = get_agent(agent_id)
    if definition is None or definition.factory is None:
        return BatchRunSummary(
            agent_id=agent_id, mode="generic",
            error_messages=[f"{agent_id}: not registered"],
        )

    summary = BatchRunSummary(agent_id=agent_id, mode="generic")
    agent = definition.factory()
    trigger = (
        f"Execute sua tarefa agendada agora. "
        f"Contexto: {row.description}"
    )
    try:
        record = agent.run(trigger)
        summary.checked = 1
        if record.error is None:
            summary.processed = 1
        else:
            summary.errors = 1
            summary.error_messages.append(record.error)
    except Exception as exc:  # noqa: BLE001
        summary.errors = 1
        summary.error_messages.append(str(exc))

    return summary


# ---------------------------------------------------------------------------
# Status (powers the Agents page schedule strip)
# ---------------------------------------------------------------------------


def _load_schedule_state() -> dict[str, str]:
    if not _SCHEDULE_STATE_PATH.exists():
        return {}
    try:
        return json.loads(_SCHEDULE_STATE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _next_cron_fire(
    cron_expr: str, *, after: datetime,
) -> datetime | None:
    """Walk forward minute-by-minute until the cron matches.

    Caps at one week so misconfigured crons don't spin forever.

    sensitivity_tier: N/A
    """
    cursor = (after + timedelta(minutes=1)).replace(
        second=0, microsecond=0,
    )
    end = after + timedelta(minutes=_NEXT_RUN_LOOKAHEAD_MINUTES)
    while cursor <= end:
        if cron_matches(cron_expr, cursor):
            return cursor
        cursor += timedelta(minutes=1)
    return None


def _last_run_status(
    db_engine: Any, agent_id: str,
) -> tuple[str | None, str | None]:
    """Look up the most recent ``deep_agent_runs`` row for this agent.

    Returns ``(status, error)`` — both may be ``None`` if the agent has
    never run through the deep-agent pipeline (which is the case for
    sourceless SBAgent invocations: ``deep_agent_runs`` only records
    SBDeepAgent runs).

    sensitivity_tier: 1
    """
    if not table_exists(db_engine, "deep_agent_runs"):
        return None, None
    try:
        rows = db_engine.query(
            """
            SELECT status, error
            FROM deep_agent_runs
            WHERE agent_id = ?
            ORDER BY started_at DESC
            LIMIT 1
            """,
            [agent_id],
        )
    except Exception:  # noqa: BLE001
        return None, None
    if not rows:
        return None, None
    return rows[0].get("status"), rows[0].get("error")


def get_user_agent_status(
    layer: DataLayer, agent_id: str,
) -> dict[str, Any]:
    """Return scheduling/runtime status for the Agents page strip.

    sensitivity_tier: 1
    """
    from src.agents.user_agents.store import UserAgentStore

    store = UserAgentStore()
    try:
        row = store.get(agent_id)
    finally:
        store.close()

    if row is None:
        return {"agent_id": agent_id, "error": "not found"}

    state = _load_schedule_state()
    last_run_at = state.get(agent_id)

    now = datetime.now(tz=timezone.utc)
    next_run_at: str | None = None
    if row.schedule_enabled and row.schedule_cron:
        next_dt = _next_cron_fire(row.schedule_cron, after=now)
        if next_dt is not None:
            next_run_at = next_dt.isoformat()

    db_engine = layer.duckdb
    _ensure_processed_table(db_engine)

    data_tool_ids = data_tool_ids_for_row(row)
    pending_count = 0
    for tool_id in data_tool_ids:
        source_table = _data_tool_target_table(tool_id)
        if source_table is None:
            continue
        pending_count += _count_unprocessed_for_source(
            db_engine, agent_id, source_table,
        )

    last_status, last_error = _last_run_status(db_engine, agent_id)

    return {
        "agent_id": agent_id,
        "schedule_cron": row.schedule_cron,
        "schedule_enabled": row.schedule_enabled,
        "enabled_data_tools": list(data_tool_ids),
        "delivery_tools": list(row.delivery_tools),
        "last_run_at": last_run_at,
        "last_status": last_status,
        "last_error": last_error,
        "next_run_at": next_run_at,
        "pending_count": pending_count,
    }


__all__ = [
    "BatchRunSummary",
    "data_tool_ids_for_row",
    "get_user_agent_status",
    "run_user_agent_batch",
    "run_user_agent_generic",
]
