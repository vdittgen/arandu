"""WhatsApp self-chat reply handler.

Detects user replies in the WhatsApp self-chat (where Brain notifications
are sent) and routes them to ``BrainAgentV2`` for processing, handles
STOP opt-out commands, and supports action execution with text-based
confirmation.

Detection: Brain notifications start with the brain emoji prefix (🧠).
Any other message in the self-chat is treated as a user reply.

Action flow:
1. User sends action request → Brain's ``propose_action`` tool fires
   inside ``ask_stream()``
2. Handler sends confirmation message back → stores pending action
   (30-min TTL)
3. User replies "yes"/"sim" → handler executes via ActionExecutor
4. Result sent back + connector re-synced

sensitivity_tier: 3 (processes user messages through BrainAgentV2)
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from src.agents.brain.actions import (
    format_candidates_message,
    query_action_candidates,
    resolve_connector_command,
)

if TYPE_CHECKING:
    from src.agents.action_executor import ActionExecutor
    from src.agents.brain import BrainAgentV2
    from src.core.sqlite.engine import DatabaseEngine

logger = logging.getLogger(__name__)

BRAIN_PREFIX = "\U0001f9e0 "  # 🧠
_LEGACY_PREFIX = "Arandu - "  # recognised in old messages


def _is_brain_message(content: str) -> bool:
    """Return True if *content* was sent by Arandu (current or legacy prefix)."""
    return content.startswith(BRAIN_PREFIX) or content.startswith(_LEGACY_PREFIX)


# Pending action confirmation TTL.
_ACTION_TTL = timedelta(minutes=30)

# Maps STOP keywords to notification preference categories.
_STOP_COMMANDS: dict[str, str] = {
    "STOP REPLIES": "pending_replies",
    "STOP PEOPLE": "important_people",
    "STOP BIRTHDAYS": "birthday_reminders",
    "STOP EVENTS": "event_actions",
    "STOP CALENDAR": "calendar_conflicts",
    "STOP HEALTH": "health_alerts",
    "STOP ACTIONS": "action_results",
    "STOP ALERTS": "topic_action",
    "STOP ENRICHMENT": "topic_enrichment",
    "STOP DIGEST": "conversation_digest",
    "STOP PIPELINE": "pipeline_summary",
    "STOP ALL": "_global",
}

# Confirmation intent keywords (case-insensitive, stripped).
_CONFIRM_WORDS: frozenset[str] = frozenset({
    "yes", "y", "sim", "s", "ok", "confirm", "go", "do it", "proceed",
})
_REJECT_WORDS: frozenset[str] = frozenset({
    "no", "n", "não", "nao", "cancel", "cancelar", "nevermind",
})


# Keywords that indicate batch/temporal action requests requiring
# multi-step confirmation (query first, then act).
_BATCH_KEYWORDS: frozenset[str] = frozenset({
    "all", "every", "todos", "todas", "each", "everything",
    "the new", "the recent", "the latest",
    "as novas", "as últimas", "os novos", "os últimos",
})

_TEMPORAL_KEYWORDS: frozenset[str] = frozenset({
    "yesterday", "today", "last hour", "last 2 hour", "last 3 hour",
    "last 4 hour", "last 5 hour", "last 6 hour", "last 12 hour",
    "last 24 hour", "this morning", "this afternoon", "this evening",
    "this week", "last week", "ontem", "hoje", "esta manhã",
    "esta semana", "semana passada", "última hora", "últimas horas",
    "created in", "created today", "created yesterday",
    "from today", "from yesterday", "from this week",
})


def _is_batch_or_temporal(text: str) -> bool:
    """Detect if an action request implies multiple items or a time range.

    sensitivity_tier: 1
    """
    lower = text.lower()
    for kw in _BATCH_KEYWORDS:
        if kw in lower:
            return True
    for kw in _TEMPORAL_KEYWORDS:
        if kw in lower:
            return True
    return False


def _parse_item_selection(
    text: str, total: int,
) -> list[int] | None:
    """Parse item indices from user reply (e.g. "1, 3, 5").

    Returns a list of 0-based indices, or ``None`` if the text
    is not an item selection (e.g. "yes", "no", free text).

    sensitivity_tier: 1
    """
    import re

    # Match comma/space-separated numbers like "1, 3, 5" or "1 3 5"
    nums = re.findall(r"\d+", text.strip())
    if not nums or len(nums) < 1:
        return None

    # Only treat as selection if the text is mostly numbers
    stripped = re.sub(r"[\d,\s]+", "", text.strip())
    if len(stripped) > 5:
        return None

    indices: list[int] = []
    for n in nums:
        idx = int(n) - 1  # Convert 1-based to 0-based
        if 0 <= idx < total:
            indices.append(idx)

    return indices if indices else None


def _parse_stop_command(text: str) -> str | None:
    """Parse a STOP opt-out command from message text.

    Returns the mapped preference category, or ``None`` if not a STOP
    command.

    sensitivity_tier: 1
    """
    stripped = text.strip().upper()
    return _STOP_COMMANDS.get(stripped)


def _parse_recipient_pick(
    text: str, candidate_count: int,
) -> int | str | None:
    """Parse a disambiguation reply into an index, ``"none"``, or ``None``.

    Returns:
        - 0-based integer when the user typed a single number in range
        - ``"none"`` when the user typed cancel/none/nenhum
        - ``None`` when the text is neither (treat as a fresh query)

    sensitivity_tier: 1
    """
    import re

    stripped = text.strip().lower()
    if stripped in {"none", "nenhum", "nenhuma", "cancel", "cancelar", "no"}:
        return "none"
    if candidate_count <= 0:
        return None
    nums = re.findall(r"\d+", stripped)
    if len(nums) != 1:
        return None
    residue = re.sub(r"[\d\s.,)]", "", stripped)
    if residue:
        return None
    idx = int(nums[0]) - 1
    if 0 <= idx < candidate_count:
        return idx
    return None


def _format_candidate_label(candidate: dict[str, Any]) -> str:
    """Render one disambiguation row for the WhatsApp picker message.

    sensitivity_tier: 3
    """
    name = str(candidate.get("name") or "Unknown").strip()
    parts: list[str] = [name]
    relationship = str(candidate.get("relationship") or "").strip()
    if relationship:
        parts.append(relationship)
    topic = str(candidate.get("active_topic") or "").strip()
    if topic:
        parts.append(f"topic: {topic}")
    return " — ".join(parts)


def _parse_confirmation_intent(text: str) -> str | None:
    """Parse a confirmation or rejection intent from message text.

    Returns ``"confirm"``, ``"reject"``, or ``None`` (not a confirmation
    reply — treat as a new query).

    sensitivity_tier: 1
    """
    normalised = text.strip().lower()
    if normalised in _CONFIRM_WORDS:
        return "confirm"
    if normalised in _REJECT_WORDS:
        return "reject"
    return None


def _get_conversation_context(
    db_engine: DatabaseEngine,
    self_jid: str,
    send_phone: str | None = None,
    self_lid: str | None = None,
    limit: int = 10,
) -> str:
    """Build recent self-chat conversation context for BrainAgent.

    *self_jid* is the bare WhatsApp JID number (e.g. ``"554892011083"``).
    *send_phone* is the optional full phone number (e.g. ``"5548992011083"``).
    *self_lid* is the Linked Device ID (e.g. ``"161048623628515"``).
    Queries the last *limit* messages from the self-chat to provide
    conversation continuity when answering user replies.

    sensitivity_tier: 2
    """
    phone_jid = f"{self_jid}@s.whatsapp.net"
    jids = [phone_jid]
    if send_phone and send_phone != self_jid:
        jids.append(f"{send_phone}@s.whatsapp.net")
    if self_lid:
        jids.append(f"{self_lid}@lid")

    or_clauses = " OR ".join(
        "(recipient = ? OR chat_name = ?)" for _ in jids
    )
    jid_params: list[str] = []
    for j in jids:
        jid_params.extend([j, j])

    rows = db_engine.query(
        f"""
        SELECT content, timestamp
        FROM raw_messages
        WHERE ({or_clauses})
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        [*jid_params, limit],
    )

    if not rows:
        return ""

    lines: list[str] = ["Recent self-chat conversation:"]
    for row in reversed(rows):
        content = str(row.get("content", "")).strip()
        if not content:
            continue
        if _is_brain_message(content):
            # Truncate bot responses to prevent the LLM from
            # reinforcing its own previous hallucinations (e.g.
            # wrong dates repeated across multiple responses).
            short = content[:80]
            if "I can do this for you" in content:
                lines.append(
                    "[Arandu (past action proposal, "
                    f"NOT executed)]: {short}"
                )
            else:
                lines.append(f"[Arandu]: {short}")
        else:
            lines.append(f"[User]: {content}")

    return "\n".join(lines)


def _lookup_quoted_text(msg_id: str) -> str:
    """Look up quoted message text from Baileys store.json.

    *msg_id* has the format ``"JID:KEY_ID"`` (e.g.
    ``"161048623628515@lid:3A63985CF8782C3C1EAC"``).
    Searches all JID keys in the store for a message with matching
    key id and returns the quoted text if the message is a thread
    reply.

    Non-fatal: returns empty string on any failure.

    sensitivity_tier: 2
    """
    try:
        import json as _json

        from src.extensions.bridges.whatsapp.paths import (
            resolve_whatsapp_store_path,
        )

        store_path = resolve_whatsapp_store_path()
        if not store_path.exists():
            return ""

        # Extract the key ID portion after ":"
        parts = msg_id.rsplit(":", 1)
        if len(parts) != 2:
            return ""
        key_id = parts[1]

        data = _json.loads(
            store_path.read_text(encoding="utf-8"),
        )
        store_msgs = data.get("messages", {})

        # Search across all JID keys for the message
        for jid_key, messages in store_msgs.items():
            if not isinstance(messages, list):
                continue
            for entry in messages:
                if not isinstance(entry, dict):
                    continue
                if entry.get("key", {}).get("id") != key_id:
                    continue
                # Found matching message — check for quote
                msg_body = entry.get("message", {}) or {}
                ext = msg_body.get("extendedTextMessage", {})
                ctx_info = (
                    ext.get("contextInfo", {})
                    or msg_body.get("contextInfo", {})
                    or {}
                )
                quoted_msg = ctx_info.get("quotedMessage", {})
                if not quoted_msg:
                    return ""
                return str(
                    quoted_msg.get("conversation", "")
                    or quoted_msg.get(
                        "extendedTextMessage", {},
                    ).get("text", "")
                ).strip()
    except Exception:  # noqa: BLE001
        pass
    return ""


class ReplyTracker:
    """DuckDB persistence for tracking processed self-chat replies.

    Follows the ``PreferenceService`` pattern: table DDL + simple CRUD.

    sensitivity_tier: 2
    """

    def __init__(self, db_engine: DatabaseEngine) -> None:
        self._db = db_engine
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Create the ``_reply_tracker`` table if it doesn't exist.

        sensitivity_tier: 1
        """
        try:
            self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS _reply_tracker (
                    message_id    VARCHAR PRIMARY KEY,
                    message_text  TEXT NOT NULL,
                    processed_at  TEXT NOT NULL,
                    reply_type    VARCHAR NOT NULL,
                    response_text TEXT,
                    response_sent INTEGER DEFAULT 0,
                    error         TEXT
                )
                """
            )
        except Exception:  # noqa: BLE001
            logger.debug(
                "Skipped _reply_tracker table creation (read-only mode)",
                exc_info=True,
            )

    def is_processed(self, message_id: str) -> bool:
        """Check whether a message has already been processed.

        sensitivity_tier: 1
        """
        rows = self._db.query(
            "SELECT 1 FROM _reply_tracker WHERE message_id = ?",
            [message_id],
        )
        return bool(rows)

    def was_processed_as_audio(self, message_id: str) -> bool:
        """Check if a message was processed with ``[audio]`` placeholder.

        Used to allow re-processing after transcription updates the content
        from ``[audio]`` to ``[voice note] ...``.

        sensitivity_tier: 1
        """
        rows = self._db.query(
            "SELECT 1 FROM _reply_tracker "
            "WHERE message_id = ? AND message_text = '[audio]'",
            [message_id],
        )
        return bool(rows)

    def mark_processed(
        self,
        message_id: str,
        text: str,
        reply_type: str,
        response_text: str | None = None,
        response_sent: bool = False,
        error: str | None = None,
    ) -> None:
        """Record a processed reply.

        sensitivity_tier: 2
        """
        now_ts = datetime.now(tz=timezone.utc).isoformat()
        self._db.execute(
            "INSERT INTO _reply_tracker "
            "(message_id, message_text, processed_at, reply_type, "
            "response_text, response_sent, error) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [message_id, text, now_ts, reply_type, response_text, response_sent, error],
        )

    def get_last_check_time(self) -> datetime | None:
        """Return the most recent ``processed_at`` timestamp.

        Used to narrow the SQL query for new self-chat messages.

        sensitivity_tier: 1
        """
        rows = self._db.query(
            "SELECT MAX(processed_at) AS last_ts FROM _reply_tracker",
        )
        if not rows or rows[0]["last_ts"] is None:
            return None
        val = rows[0]["last_ts"]
        if isinstance(val, datetime):
            return val
        try:
            return datetime.fromisoformat(str(val))
        except (ValueError, TypeError):
            return None


class PendingActionStore:
    """DuckDB persistence for action proposals awaiting user confirmation.

    Only one pending action at a time. Proposals auto-expire after
    ``_ACTION_TTL`` (30 minutes).

    sensitivity_tier: 2
    """

    def __init__(self, db_engine: DatabaseEngine) -> None:
        self._db = db_engine
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Create the ``_pending_actions`` table if it doesn't exist.

        sensitivity_tier: 1
        """
        try:
            self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS _pending_actions (
                    proposal_id   VARCHAR PRIMARY KEY,
                    proposal_json TEXT NOT NULL,
                    description   TEXT NOT NULL,
                    created_at    TEXT NOT NULL,
                    status        VARCHAR DEFAULT 'pending'
                )
                """
            )
        except Exception:  # noqa: BLE001
            logger.debug(
                "Skipped _pending_actions table creation (read-only mode)",
                exc_info=True,
            )

    def store(
        self,
        proposal_id: str,
        proposal_json: str,
        description: str,
    ) -> None:
        """Insert a new pending action.

        sensitivity_tier: 2
        """
        now_ts = datetime.now(tz=timezone.utc).isoformat()
        self._db.execute(
            "INSERT INTO _pending_actions "
            "(proposal_id, proposal_json, description, created_at, status) "
            "VALUES (?, ?, ?, ?, 'pending')",
            [proposal_id, proposal_json, description, now_ts],
        )

    def get_pending(self) -> dict[str, Any] | None:
        """Return the most recent non-expired pending action, or ``None``.

        sensitivity_tier: 1
        """
        cutoff = (datetime.now(tz=timezone.utc) - _ACTION_TTL).isoformat()
        rows = self._db.query(
            """
            SELECT proposal_id, proposal_json, description, created_at
            FROM _pending_actions
            WHERE status = 'pending' AND created_at > ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            [cutoff],
        )
        return rows[0] if rows else None

    def get_recently_expired(self) -> dict[str, Any] | None:
        """Return the most recent expired-by-TTL pending action.

        Returns actions with ``status='pending'`` whose ``created_at``
        is older than ``_ACTION_TTL``.  Used to detect late confirmation
        replies ("yes" sent after the action window closed).

        sensitivity_tier: 1
        """
        cutoff = (
            datetime.now(tz=timezone.utc) - _ACTION_TTL
        ).isoformat()
        rows = self._db.query(
            """
            SELECT proposal_id, proposal_json, description,
                   created_at
            FROM _pending_actions
            WHERE status = 'pending' AND created_at <= ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            [cutoff],
        )
        return rows[0] if rows else None

    def resolve(self, proposal_id: str, status: str) -> None:
        """Mark a pending action as confirmed, rejected, or expired.

        sensitivity_tier: 1
        """
        self._db.execute(
            "UPDATE _pending_actions SET status = ? WHERE proposal_id = ?",
            [status, proposal_id],
        )


class ReplyHandler:
    """Detect and process user replies in the WhatsApp self-chat.

    The handler:
    1. Queries ``raw_messages`` for new self-chat messages
    2. Filters out Brain notifications (🧠 prefix)
    3. Handles STOP opt-out commands via ``PreferenceService``
    4. Routes other messages to ``BrainAgent.ask()`` and sends back the answer
    5. Detects action intents via ``ask_stream()`` and manages text-based
       confirmation → execution flow (when ``action_executor`` is provided)

    sensitivity_tier: 3
    """

    def __init__(
        self,
        db_engine: DatabaseEngine,
        brain_agent: BrainAgentV2,
        phone: str,
        self_jid: str | None = None,
        self_lid: str | None = None,
        send_fn: Callable[[str, str], bool] | None = None,
        action_executor: ActionExecutor | None = None,
        sync_fn: Callable[[str], None] | None = None,
    ) -> None:
        self._db = db_engine
        self._brain = brain_agent
        # _send_phone: full international number for sending (e.g. "5548992011083")
        self._send_phone = phone.lstrip("+")
        # _jid: bare WhatsApp JID number for DB queries (e.g. "554892011083").
        # May differ from send_phone due to country-specific normalization
        # (e.g. Brazil drops a leading "9" in subscriber numbers).
        self._jid = self_jid or self._send_phone
        # _lid: bare Linked Device ID for self-chat delivery (e.g. "161048623628515").
        # In multi-device Baileys, the phone's self-chat thread uses @lid JIDs,
        # NOT @s.whatsapp.net.  Sending to @s.whatsapp.net creates a separate
        # chat thread on the phone instead of landing in the self-chat.
        self._lid = self_lid
        self._tracker = ReplyTracker(db_engine)
        self._send_fn = send_fn or self._send_via_outbox
        # Action execution support (optional — backward compatible).
        self._action_executor = action_executor
        self._sync_fn = sync_fn
        self._pending_store: PendingActionStore | None = (
            PendingActionStore(db_engine) if action_executor else None
        )

    def process_new_replies(self) -> int:
        """Find and process new self-chat messages. Returns count processed.

        sensitivity_tier: 3
        """
        # Invalidate query cache so we see messages inserted by
        # _poll_self_chat_messages or the ingestion adapter since
        # the last cycle.
        self._db.invalidate_cache()
        messages = self._fetch_new_self_chat_messages()
        if not messages:
            return 0

        processed = 0
        for msg in messages:
            msg_id = str(msg.get("id", ""))
            content = str(msg.get("content", "")).strip()

            if not msg_id or not content:
                continue

            # Skip Brain's own notifications
            if _is_brain_message(content):
                continue

            # Skip already processed — but allow re-processing if the
            # message was previously handled as [audio] and has since been
            # transcribed to [voice note].
            if self._tracker.is_processed(msg_id):
                is_now_transcribed = content.startswith("[voice note] ")
                if is_now_transcribed and self._tracker.was_processed_as_audio(msg_id):
                    pass  # allow re-processing
                else:
                    continue

            # Skip media placeholders that can't be meaningfully answered.
            # [audio] is deferred — transcription may update it to
            # [voice note] on the next cycle so we don't mark it processed.
            if content == "[audio]":
                continue
            # Non-transcribable media — mark processed so we don't retry.
            if content in ("[image]", "[sticker]", "[document]"):
                self._tracker.mark_processed(
                    msg_id, content, "skipped_media",
                )
                processed += 1
                continue

            # Strip [voice note] prefix so BrainAgent sees clean text.
            if content.startswith("[voice note] "):
                content = content[len("[voice note] "):]

            # Extract quoted text from metadata or Baileys store.
            quoted_text = ""
            raw_meta = msg.get("metadata") or ""
            if raw_meta:
                try:
                    import json as _json
                    meta = _json.loads(raw_meta)
                    quoted_text = str(
                        meta.get("quoted_text", ""),
                    ).strip()
                except (ValueError, TypeError):
                    pass
            if not quoted_text:
                quoted_text = _lookup_quoted_text(msg_id)

            try:
                self._handle_single_reply(
                    msg_id, content, quoted_text=quoted_text,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to process reply %s: %s", msg_id, exc,
                )
                self._tracker.mark_processed(
                    msg_id, content, "error", error=str(exc),
                )
            processed += 1

        return processed

    def _fetch_new_self_chat_messages(self) -> list[dict[str, Any]]:
        """Query raw_messages for recent self-chat messages.

        Checks both the Baileys-normalized JID and the send-phone JID,
        since older entries may use either format.

        sensitivity_tier: 2
        """
        phone_jid = f"{self._jid}@s.whatsapp.net"
        # Build list of JIDs to match (both Baileys and MCP-thread)
        jids = [phone_jid]
        if self._send_phone != self._jid:
            jids.append(f"{self._send_phone}@s.whatsapp.net")
        # Multi-device: self-chat messages arrive under @lid JID
        if self._lid:
            jids.append(f"{self._lid}@lid")

        jid_params: list[str] = []
        for j in jids:
            jid_params.extend([j, j])  # recipient + chat_name

        or_clauses = " OR ".join(
            "(recipient = ? OR chat_name = ?)" for _ in jids
        )

        last_check = self._tracker.get_last_check_time()
        if last_check is not None:
            rows = self._db.query(
                f"""
                SELECT id, content, timestamp, metadata
                FROM raw_messages
                WHERE ({or_clauses})
                  AND timestamp > ?
                ORDER BY timestamp ASC
                """,
                [*jid_params, last_check.isoformat()],
            )
        else:
            # First run: look at the last 24 hours to catch recent
            # messages without processing the entire self-chat history.
            rows = self._db.query(
                f"""
                SELECT id, content, timestamp, metadata
                FROM raw_messages
                WHERE ({or_clauses})
                  AND timestamp > datetime('now', '-24 hours')
                ORDER BY timestamp ASC
                """,
                jid_params,
            )
        return rows

    def _handle_single_reply(
        self,
        message_id: str,
        text: str,
        *,
        quoted_text: str = "",
    ) -> None:
        """Process one user reply message.

        *quoted_text* is the text of the message being replied to
        (from WhatsApp thread quotes).  When present, it replaces
        the recent-messages conversation context so the LLM can
        resolve pronouns accurately.

        Routing priority:
        1. STOP opt-out commands
        2. Pending action confirmation/rejection (if action support enabled)
        3. Regular brain query (may detect new action intents)

        sensitivity_tier: 3
        """
        # 1. STOP commands always take priority.
        stop_category = _parse_stop_command(text)
        if stop_category:
            self._handle_stop(message_id, text, stop_category)
            return

        # 2. Check if this is a confirmation reply to a pending action.
        if self._pending_store is not None:
            pending = self._pending_store.get_pending()
            if pending:
                proposal = json.loads(pending["proposal_json"])
                # 2a. Disambiguation pick — numeric ("1") or "none".
                if proposal.get("_kind") == "disambiguation":
                    n_cands = len(proposal.get("candidates") or [])
                    pick = _parse_recipient_pick(text, n_cands)
                    if pick == "none":
                        self._cancel_pending_action(
                            message_id, text, pending,
                        )
                        return
                    if isinstance(pick, int):
                        self._resume_disambiguation(
                            message_id, text, pending, pick,
                        )
                        return
                    # Unrecognised reply → expire and fall through.
                    self._pending_store.resolve(
                        pending["proposal_id"], "expired",
                    )
                else:
                    intent = _parse_confirmation_intent(text)
                    if intent == "confirm":
                        self._execute_pending_action(
                            message_id, text, pending,
                        )
                        return
                    if intent == "reject":
                        self._cancel_pending_action(
                            message_id, text, pending,
                        )
                        return
                    # Check if user selected specific items
                    # (e.g. "1, 3, 5") for a batch proposal.
                    if proposal.get("batch"):
                        n = len(proposal.get("candidates", []))
                        sel = _parse_item_selection(text, n)
                        if sel is not None:
                            self._execute_pending_action(
                                message_id, text, pending,
                            )
                            return
                    # Not a confirmation — expire the pending action
                    # and fall through to brain query.
                    self._pending_store.resolve(
                        pending["proposal_id"], "expired",
                    )
            else:
                # No active pending action — check if the user is
                # trying to confirm one that already expired.
                expired = self._pending_store.get_recently_expired()
                if expired and _parse_confirmation_intent(text):
                    self._handle_expired_action(
                        message_id, text, expired,
                    )
                    return

        # 3. Regular brain query (may detect new actions via ask_stream).
        self._handle_brain_query(
            message_id, text, quoted_text=quoted_text,
        )

    def _handle_stop(
        self, message_id: str, text: str, category: str,
    ) -> None:
        """Process a STOP opt-out command.

        sensitivity_tier: 1
        """
        from src.notifications.preference_service import PreferenceService

        prefs = PreferenceService(self._db)

        if category == "_global":
            prefs.mute_all()
            confirmation = (
                f"{BRAIN_PREFIX}All notifications muted for 24 hours.\n\n"
                "Send a message anytime to resume."
            )
        else:
            prefs.update_preference(category, enabled=False)
            confirmation = (
                f"{BRAIN_PREFIX}Notifications for '{category}' disabled.\n\n"
                "Send a message anytime to ask me anything."
            )

        sent = self._send_response(confirmation)
        self._tracker.mark_processed(
            message_id, text, "stop_command",
            response_text=confirmation,
            response_sent=sent,
        )

    def _handle_brain_query(
        self,
        message_id: str,
        text: str,
        *,
        quoted_text: str = "",
    ) -> None:
        """Send question to BrainAgent and reply with the answer.

        When *quoted_text* is provided (user replied to a specific
        message in the WhatsApp thread), it is used as the sole
        conversation context instead of the recent-messages history.
        This avoids context pollution from unrelated older messages.

        When ``action_executor`` is configured, uses a multi-step flow:
        1. Check for action intent
        2. If batch/temporal action → query candidates, present to user
        3. If single action → propose directly
        4. Otherwise → regular brain query

        sensitivity_tier: 3
        """
        if quoted_text:
            # Thread reply — use only the quoted message as context
            # so pronoun resolution is accurate (e.g. "dele" → the
            # person mentioned in the quoted message).
            context = (
                f"Message being replied to:\n{quoted_text}"
            )
        else:
            context = _get_conversation_context(
                self._db, self._jid,
                send_phone=self._send_phone,
                self_lid=self._lid,
                limit=10,
            )

        question = text
        if context:
            question = f"{context}\n\nNew question: {text}"

        # When action execution is available, let the LLM decide
        # whether the message is a question or an action request.
        # Rule-based match_action_intent is too aggressive for
        # conversational Portuguese (e.g. "busque" triggers Search
        # Contacts even when the user means "look up on the web").
        if self._action_executor is not None:
            answer, proposal = self._ask_with_action_detection(
                question, raw_text=text,
            )
            if proposal is not None:
                self._propose_action(message_id, text, proposal)
                return
        else:
            try:
                response = self._brain.ask(
                    question, max_sensitivity_tier=2,
                )
                answer = response.answer
            except Exception as exc:  # noqa: BLE001
                logger.warning("BrainAgent.ask() failed for reply: %s", exc)
                answer = "Sorry, I couldn't process your question right now."

        formatted = (
            f"{BRAIN_PREFIX}{answer}\n\n"
            "---\n"
            "Ask me anything. Reply STOP ALL to opt out."
        )

        sent = self._send_response(formatted)
        self._tracker.mark_processed(
            message_id, text, "brain_query",
            response_text=answer,
            response_sent=sent,
        )

    def _handle_multi_step_action(
        self,
        message_id: str,
        text: str,
        question: str,
        matched_action: Any,
    ) -> bool:
        """Execute a multi-step action: query candidates → present → act.

        For batch operations ("delete all notes from yesterday"), first
        queries DuckDB for matching records, then presents them to the
        user before proposing actions.

        Returns ``True`` if the message was handled (response sent).

        sensitivity_tier: 3
        """
        connector_id = matched_action.connector_id

        # Query for matching candidates (uses Brain v2's deps).
        candidates = query_action_candidates(
            connector_id,
            text,
            self._brain._query_engine._duck,  # noqa: SLF001
            self._brain._resolve_provider(),  # noqa: SLF001
        )

        if not candidates:
            # No matches — inform the user
            formatted = (
                f"{BRAIN_PREFIX}I looked for matching items but "
                "found nothing. Could you rephrase your request?\n\n"
                "---\n"
                "Ask me anything. Reply STOP ALL to opt out."
            )
            sent = self._send_response(formatted)
            self._tracker.mark_processed(
                message_id, text, "action_no_candidates",
                response_text=formatted,
                response_sent=sent,
            )
            return True

        if len(candidates) == 1:
            # Single match — go straight to action proposal via ask_stream
            # (the data context will include this record)
            return False

        # Multiple matches — present them and ask for confirmation.
        # Build the candidates list message.
        listing = format_candidates_message(candidates, connector_id)
        display = matched_action.display_name

        formatted = (
            f"{BRAIN_PREFIX}Before I {display.lower()}, let me show "
            f"you what I found:\n\n"
            f"{listing}\n\n"
            f"Should I {display.lower()} all of these? "
            'Reply "yes" to confirm, "no" to cancel, '
            "or specify which ones (e.g. \"1, 3, 5\")."
        )

        # Store the batch pending action BEFORE sending the message
        # so we never send an orphaned confirmation without a stored
        # proposal (the user's "Yes" would fall through to BrainAgent).
        if self._pending_store is not None:
            import uuid
            proposal_id = str(uuid.uuid4())

            # Resolve connector command/args from the v2 tool registry.
            command, cmd_args = resolve_connector_command(
                self._brain._tool_registry,  # noqa: SLF001
                connector_id,
            )

            # Serialize candidates — DuckDB returns datetime objects
            # which are not JSON-serializable.
            serializable_candidates = []
            for c in candidates:
                row: dict[str, Any] = {}
                for k, v in c.items():
                    if k == "_table":
                        continue
                    if isinstance(v, datetime):
                        row[k] = v.isoformat()
                    else:
                        row[k] = v
                serializable_candidates.append(row)

            batch_proposal = {
                "proposal_id": proposal_id,
                "connector_id": connector_id,
                "connector_name": matched_action.connector_name,
                "tool_name": matched_action.tool_name,
                "display_name": display,
                "command": command,
                "args": list(cmd_args),
                "candidates": serializable_candidates,
                "batch": True,
            }
            self._pending_store.store(
                proposal_id=proposal_id,
                proposal_json=json.dumps(batch_proposal),
                description=(
                    f"{display} {len(candidates)} items"
                ),
            )

        sent = self._send_response(formatted)

        self._tracker.mark_processed(
            message_id, text, "action_candidates_presented",
            response_text=formatted,
            response_sent=sent,
        )
        return True

    # ------------------------------------------------------------------
    # Action support
    # ------------------------------------------------------------------

    def _ask_with_action_detection(
        self,
        question: str,
        raw_text: str | None = None,  # noqa: ARG002 (kept for API compat)
    ) -> tuple[str, dict[str, Any] | None]:
        """Consume ``ask_stream()`` and return (answer, proposal_or_none).

        Brain v2 decides action vs. answer via the LLM picking the
        ``propose_action`` tool; the legacy ``match_text`` hint that
        scoped intent matching to the raw user message is no longer
        needed (kept on the signature for caller backward compat).

        The returned proposal dict carries a ``_kind`` marker —
        ``"action"`` for a normal action proposal or
        ``"disambiguation"`` when the user must pick a recipient
        before the brain can finalise the action.

        sensitivity_tier: 3
        """
        answer_chunks: list[str] = []
        proposal: dict[str, Any] | None = None

        try:
            for event in self._brain.ask_stream(
                question, max_sensitivity_tier=2,
            ):
                etype = event.get("type")
                if etype == "action_proposal":
                    proposal = event.get("proposal")
                    if proposal is not None:
                        proposal = {**proposal, "_kind": "action"}
                elif etype == "recipient_disambiguation":
                    proposal = event.get("proposal")
                    if proposal is not None:
                        proposal = {**proposal, "_kind": "disambiguation"}
                elif etype == "token":
                    answer_chunks.append(event.get("token", ""))
                elif etype in ("done", "error"):
                    if etype == "error" and not answer_chunks:
                        answer_chunks.append(
                            "Sorry, I couldn't process your question right now.",
                        )
                    break
        except Exception as exc:  # noqa: BLE001
            logger.warning("ask_stream() failed: %s", exc)
            if not answer_chunks:
                answer_chunks.append(
                    "Sorry, I couldn't process your question right now.",
                )

        return "".join(answer_chunks), proposal

    def _propose_action(
        self,
        message_id: str,
        text: str,
        proposal: dict[str, Any],
    ) -> None:
        """Send an action confirmation message and store the pending proposal.

        If the proposal has missing required parameters, sends an info
        message instead and does not create a pending action.

        sensitivity_tier: 2
        """
        if proposal.get("_kind") == "disambiguation":
            self._propose_recipient_disambiguation(message_id, text, proposal)
            return

        missing = proposal.get("missing_params") or []
        display = proposal.get("display_name", "Action")
        description = proposal.get("description", "")

        if missing:
            formatted = (
                f"{BRAIN_PREFIX}I'd like to help, but I need more "
                f"information:\n\n"
                f"  Missing: {', '.join(missing)}\n\n"
                "Please provide the missing details and try again."
            )
            sent = self._send_response(formatted)
            self._tracker.mark_processed(
                message_id, text, "action_missing_params",
                response_text=formatted,
                response_sent=sent,
            )
            return

        confirmation = (
            f"{BRAIN_PREFIX}I can do this for you:\n\n"
            f"  {display}: {description}\n\n"
            'Reply "yes" to confirm or "no" to cancel.'
        )
        sent = self._send_response(confirmation)

        # Store the pending action for the next poll cycle.
        if self._pending_store is not None:
            proposal_id = proposal.get("proposal_id", "")
            self._pending_store.store(
                proposal_id=proposal_id,
                proposal_json=json.dumps(proposal),
                description=description,
            )

        self._tracker.mark_processed(
            message_id, text, "action_proposal",
            response_text=confirmation,
            response_sent=sent,
        )

    def _propose_recipient_disambiguation(
        self,
        message_id: str,
        text: str,
        proposal: dict[str, Any],
    ) -> None:
        """Present the user with a numbered candidate list to pick from.

        Stores the disambiguation proposal in the pending-action table
        so the next reply (a number, or "none") can be routed to
        :meth:`_resume_disambiguation`.

        sensitivity_tier: 3
        """
        candidates = proposal.get("candidates") or []
        original_name = proposal.get("original_name") or "this contact"
        display = proposal.get("display_name", "Send Message")

        if not candidates:
            msg = (
                f"{BRAIN_PREFIX}I couldn't find a saved contact for "
                f"'{original_name}'. Reply with a phone number to send "
                f"to that destination, or rephrase the request."
            )
            sent = self._send_response(msg)
            self._tracker.mark_processed(
                message_id, text, "action_disambiguation_empty",
                response_text=msg, response_sent=sent,
            )
            return

        lines: list[str] = [
            f"{BRAIN_PREFIX}I found multiple matches for "
            f'"{original_name}" — which one should I use for {display}?',
            "",
        ]
        for idx, cand in enumerate(candidates, start=1):
            label = _format_candidate_label(cand)
            lines.append(f"{idx}. {label}")
        lines.append("")
        lines.append('Reply with a number, or "none" to cancel.')
        message = "\n".join(lines)
        sent = self._send_response(message)

        if self._pending_store is not None:
            proposal_id = proposal.get("proposal_id", "")
            self._pending_store.store(
                proposal_id=proposal_id,
                proposal_json=json.dumps(proposal),
                description=f"Pick recipient for {display}",
            )

        self._tracker.mark_processed(
            message_id, text, "recipient_disambiguation",
            response_text=message, response_sent=sent,
        )

    def _resume_disambiguation(
        self,
        message_id: str,
        text: str,
        pending: dict[str, Any],
        choice: int,
    ) -> None:
        """Promote a disambiguation pick into a normal action proposal.

        Calls into :func:`resume_action_from_disambiguation`, replaces
        the pending row with the resulting ``ActionProposal``, and
        sends the standard "Reply yes / no" confirmation.

        sensitivity_tier: 3
        """
        from dataclasses import asdict

        from src.agents.brain.actions import (
            resume_action_from_disambiguation,
        )

        disambiguation = json.loads(pending["proposal_json"])
        candidates = disambiguation.get("candidates") or []
        if choice < 0 or choice >= len(candidates):
            self._send_response(
                f"{BRAIN_PREFIX}That choice is out of range. "
                f"Reply with a number 1–{len(candidates)} or 'none'.",
            )
            return
        candidate = candidates[choice]

        try:
            new_proposal = resume_action_from_disambiguation(
                disambiguation=disambiguation,
                candidate=candidate,
                duckdb=self._db,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "resume_action_from_disambiguation failed: %s", exc,
            )
            self._send_response(
                f"{BRAIN_PREFIX}Sorry, I couldn't prepare that action. "
                "Try again with a fresh request.",
            )
            self._pending_store.resolve(  # type: ignore[union-attr]
                disambiguation.get("proposal_id", ""), "expired",
            )
            return

        # Swap the pending row from disambiguation → action.
        self._pending_store.resolve(  # type: ignore[union-attr]
            disambiguation.get("proposal_id", ""), "expired",
        )
        proposal_dict = {**asdict(new_proposal), "_kind": "action"}
        self._propose_action(message_id, text, proposal_dict)

    def _execute_pending_action(
        self,
        message_id: str,
        text: str,
        pending: dict[str, Any],
    ) -> None:
        """Execute a confirmed pending action (single or batch).

        For batch proposals, executes the action for each candidate
        item sequentially and reports aggregate results.

        sensitivity_tier: 3
        """
        proposal = json.loads(pending["proposal_json"])
        proposal_id = proposal.get("proposal_id", "")

        assert self._action_executor is not None  # noqa: S101

        if proposal.get("batch"):
            self._execute_batch_action(
                message_id, text, proposal, proposal_id,
            )
            return

        self._execute_single_action(
            message_id, text, proposal, proposal_id,
        )

    def _execute_single_action(
        self,
        message_id: str,
        text: str,
        proposal: dict[str, Any],
        proposal_id: str,
    ) -> None:
        """Execute a single confirmed action.

        sensitivity_tier: 3
        """
        assert self._action_executor is not None  # noqa: S101

        try:
            result = self._action_executor.execute(
                connector_id=proposal["connector_id"],
                command=proposal["command"],
                args=tuple(proposal.get("args", ())),
                tool_name=proposal["tool_name"],
                arguments=proposal.get("arguments", {}),
                proposal_id=proposal_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Action execution failed: %s", exc)
            error_msg = self._build_failure_message(
                str(exc), proposal,
            )
            self._send_response(error_msg)
            if self._pending_store is not None:
                self._pending_store.resolve(
                    proposal_id, "confirmed",
                )
            self._tracker.mark_processed(
                message_id, text, "action_confirmed",
                error=str(exc),
            )
            return

        if result.status == "success":
            response_msg = (
                f"{BRAIN_PREFIX}Done! {result.output}\n\n"
                "---\n"
                "Ask me anything. Reply STOP ALL to opt out."
            )
            # Re-sync so new data appears immediately.
            if self._sync_fn is not None:
                try:
                    self._sync_fn(proposal["connector_id"])
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "Post-action re-sync failed",
                        exc_info=True,
                    )
        else:
            error = result.error or "Unknown error"
            response_msg = self._build_failure_message(
                error, proposal,
            )

        sent = self._send_response(response_msg)
        if self._pending_store is not None:
            self._pending_store.resolve(
                proposal_id, "confirmed",
            )
        self._tracker.mark_processed(
            message_id, text, "action_confirmed",
            response_text=(
                result.output if result.status == "success"
                else response_msg
            ),
            response_sent=sent,
        )

    def _execute_batch_action(
        self,
        message_id: str,
        text: str,
        proposal: dict[str, Any],
        proposal_id: str,
    ) -> None:
        """Execute a batch action across multiple candidate items.

        Uses LLM to map each candidate's data to tool parameters,
        then executes sequentially. Reports aggregate results.

        sensitivity_tier: 3
        """
        assert self._action_executor is not None  # noqa: S101

        candidates = proposal.get("candidates", [])
        display = proposal.get("display_name", "Action")
        connector_id = proposal["connector_id"]
        tool_name = proposal["tool_name"]
        command = proposal.get("command", "")
        args = tuple(proposal.get("args", ()))

        # Parse user's selection if they specified indices (e.g. "1, 3, 5")
        selected = _parse_item_selection(text, len(candidates))
        if selected is not None:
            candidates = [candidates[i] for i in selected]

        succeeded = 0
        failed = 0
        errors: list[str] = []

        # Send a progress message
        self._send_response(
            f"{BRAIN_PREFIX}Working on it... "
            f"{display} for {len(candidates)} items."
        )

        for candidate in candidates:
            # Use LLM to map candidate data to tool params
            params = self._candidate_to_params(
                candidate, tool_name, connector_id,
            )
            if not params:
                failed += 1
                errors.append(
                    f"Could not map params for: {candidate}"
                )
                continue

            try:
                result = self._action_executor.execute(
                    connector_id=connector_id,
                    command=command,
                    args=args,
                    tool_name=tool_name,
                    arguments=params,
                    proposal_id=f"{proposal_id}-{succeeded + failed}",
                )
                if result.status == "success":
                    succeeded += 1
                else:
                    failed += 1
                    errors.append(
                        f"{result.error or 'Unknown error'}"
                    )
            except Exception as exc:  # noqa: BLE001
                failed += 1
                errors.append(str(exc))

        # Re-sync after batch
        if succeeded > 0 and self._sync_fn is not None:
            try:
                self._sync_fn(connector_id)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Post-batch re-sync failed", exc_info=True,
                )

        # Build result message
        if failed == 0:
            response_msg = (
                f"{BRAIN_PREFIX}Done! Successfully completed "
                f"{display.lower()} for {succeeded} items.\n\n"
                "---\n"
                "Ask me anything. Reply STOP ALL to opt out."
            )
        else:
            error_summary = "; ".join(errors[:3])
            response_msg = (
                f"{BRAIN_PREFIX}{display}: "
                f"{succeeded} succeeded, {failed} failed.\n"
                f"Errors: {error_summary}\n\n"
                "---\n"
                "Ask me anything. Reply STOP ALL to opt out."
            )

        sent = self._send_response(response_msg)
        if self._pending_store is not None:
            self._pending_store.resolve(proposal_id, "confirmed")
        self._tracker.mark_processed(
            message_id, text, "action_batch_confirmed",
            response_text=response_msg,
            response_sent=sent,
        )

    def _candidate_to_params(
        self,
        candidate: dict[str, Any],
        tool_name: str,
        connector_id: str,
    ) -> dict[str, Any] | None:
        """Map a candidate record's data to MCP tool parameters.

        Uses the candidate's primary identifier (title, name, subject)
        as the tool parameter. Keeps it simple — no LLM call.

        sensitivity_tier: 2
        """
        # Map connector→primary key field used by the MCP tool
        key_mappings: dict[str, tuple[str, str]] = {
            "apple-notes": ("title", "title"),
            "apple-calendar": ("title", "title"),
            "apple-contacts": ("name", "name"),
            "apple-mail": ("subject", "subject"),
            "apple-reminders": ("title", "title"),
        }
        mapping = key_mappings.get(connector_id)
        if not mapping:
            return None

        candidate_field, param_name = mapping
        value = candidate.get(candidate_field)
        if not value:
            return None

        return {param_name: value}

    def _build_failure_message(
        self,
        error: str,
        proposal: dict[str, Any],
    ) -> str:
        """Build an error message with alternative suggestions.

        When an action fails, queries for similar items the user
        might have meant and includes them in the error message.

        sensitivity_tier: 2
        """
        connector_id = proposal.get("connector_id", "")
        args = proposal.get("arguments", {})

        # Try to find similar items to suggest
        suggestion = ""
        try:
            # Build a search query from the failed params
            search_terms = " ".join(
                str(v) for v in args.values() if v
            )
            if search_terms and connector_id:
                candidates = query_action_candidates(
                    connector_id,
                    search_terms,
                    self._brain._query_engine._duck,  # noqa: SLF001
                    self._brain._resolve_provider(),  # noqa: SLF001
                )
                if candidates:
                    items = []
                    for c in candidates[:5]:
                        parts = [
                            f"{k}={v}"
                            for k, v in c.items()
                            if v is not None and k != "_table"
                        ]
                        items.append(f"  - {', '.join(parts)}")
                    suggestion = (
                        "\n\nHere are some items I found:\n"
                        + "\n".join(items)
                        + "\n\nTry again with the exact name."
                    )
        except Exception:  # noqa: BLE001
            logger.debug(
                "Failed to find alternatives", exc_info=True,
            )

        return (
            f"{BRAIN_PREFIX}Action failed: {error}"
            f"{suggestion}\n\n"
            "---\n"
            "Ask me anything. Reply STOP ALL to opt out."
        )

    def _handle_expired_action(
        self,
        message_id: str,
        text: str,
        expired: dict[str, Any],
    ) -> None:
        """Inform the user that a pending action has expired.

        Resolves the expired action and sends a message suggesting
        the user re-request the action.

        sensitivity_tier: 1
        """
        proposal_id = expired.get("proposal_id", "")
        description = expired.get("description", "the action")
        expired_msg = (
            f"{BRAIN_PREFIX}The action expired before you "
            "confirmed:\n\n"
            f"  {description}\n\n"
            "Please repeat your request if you still "
            "want to proceed.\n\n"
            "---\n"
            "Ask me anything. Reply STOP ALL to opt out."
        )
        sent = self._send_response(expired_msg)
        if self._pending_store is not None:
            self._pending_store.resolve(
                proposal_id, "expired",
            )
        self._tracker.mark_processed(
            message_id, text, "action_expired",
            response_text="Action expired.",
            response_sent=sent,
        )

    def _cancel_pending_action(
        self,
        message_id: str,
        text: str,
        pending: dict[str, Any],
    ) -> None:
        """Cancel a pending action.

        sensitivity_tier: 1
        """
        proposal_id = pending.get("proposal_id", "")
        cancel_msg = (
            f"{BRAIN_PREFIX}Action cancelled.\n\n"
            "---\n"
            "Ask me anything. Reply STOP ALL to opt out."
        )
        sent = self._send_response(cancel_msg)
        if self._pending_store is not None:
            self._pending_store.resolve(proposal_id, "rejected")
        self._tracker.mark_processed(
            message_id, text, "action_rejected",
            response_text="Action cancelled.",
            response_sent=sent,
        )

    def _send_response(self, message: str) -> bool:
        """Send a response message using the configured send function.

        Returns ``True`` if delivery was successful.

        sensitivity_tier: 2
        """
        try:
            # In multi-device Baileys, the phone's self-chat thread uses @lid
            # JIDs (Linked Device IDs), NOT @s.whatsapp.net.  Sending to a
            # phone-number @s.whatsapp.net JID creates a SEPARATE chat thread
            # (e.g. "+55 48 99201-1083") instead of landing in the "Arandu
            # (You)" self-chat.  Use @lid when available.
            if self._lid:
                to_jid = f"{self._lid}@lid"
            else:
                to_jid = f"{self._jid}@s.whatsapp.net"
            return self._send_fn(to_jid, message)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Reply send failed: %s", exc)
            return False

    def _send_via_outbox(self, to: str, message: str) -> bool:
        """Fallback: send via listener outbox IPC.

        sensitivity_tier: 2
        """
        from src.extensions.bridges.whatsapp.listener import (
            send_text_via_running_listener,
        )

        response = send_text_via_running_listener(
            to=to, message=message, timeout_seconds=20.0,
        )
        if response is None:
            logger.warning("Reply send failed: listener not running")
            return False

        status = str(response.get("status", "")).strip().lower()
        if status == "sent":
            return True

        logger.warning(
            "Reply send status=%s error=%s",
            status, response.get("error"),
        )
        return False
