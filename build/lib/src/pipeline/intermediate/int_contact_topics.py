"""Python-based pipeline model: extracts ongoing topics per contact via LLM.

Groups recent messages (last 90 days) by contact, sends conversation
samples to the LLM, and extracts specific contextual topics like
"hiring a psychologist for Repensar" or "father's cancer treatment".

Each topic has an importance score (1-10) and status (active/resolved/stale).
Topics are used by BrainAgent for proactive guidance and notification priority.

Caches results per contact — only re-extracts when new messages arrive.
Falls back gracefully when LLM is unavailable — returns cached results.

sensitivity_tier: 3 (processes raw message content)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import typing as t

logger = logging.getLogger(__name__)

# Default lookback/messages window — uses small numbers for the local
# Ollama path.  When a remote-capable provider (OpenAI-compat or
# Anthropic) is in use we expand the window in :func:`_runtime_limits`
# because larger models can hold a richer conversation context.
_LOOKBACK_DAYS = 30
_MIN_MESSAGES = 5
_MAX_MESSAGES_PER_CONTACT = 30
# Remote-capable models earn a wider window. Numbers chosen to fit
# comfortably inside a 32K context window typical of recent LLMs.
_REMOTE_LOOKBACK_DAYS = 90
_REMOTE_MAX_MESSAGES_PER_CONTACT = 100


def _runtime_limits() -> tuple[int, int, bool]:
    """Return ``(lookback_days, max_msgs, remote_capable)``.

    Reads the configured provider name from settings without
    instantiating an :class:`LLMProvider` — pipeline LLM work now goes
    through :class:`TopicExtractorAgent`, not a raw provider.

    sensitivity_tier: 1
    """
    if _is_remote_capable():
        return (
            _REMOTE_LOOKBACK_DAYS,
            _REMOTE_MAX_MESSAGES_PER_CONTACT,
            True,
        )
    return (_LOOKBACK_DAYS, _MAX_MESSAGES_PER_CONTACT, False)


def _is_remote_capable() -> bool:
    """True when the configured provider can handle bigger windows.

    sensitivity_tier: 1
    """
    try:
        from src.models.llm_provider import load_llm_settings
    except Exception:  # noqa: BLE001
        return False
    name = load_llm_settings().get("llm_provider", "")
    return name in ("openai_compat", "anthropic")

# The LLM prompts and multi-contact batching live in
# :mod:`src.agents.topic_extractor.agent` (pydantic-ai).
# This pipeline module is now a coordinator: SQL pre-filtering,
# per-contact dispatch, and DB cache management.

_CACHE_TABLE = "_contact_topics_cache"

if t.TYPE_CHECKING:
    from src.core.sqlite.engine import DatabaseEngine


def _normalize_phone(raw: str) -> str:
    """Strip a phone string to its last 10 digits for fuzzy matching.

    sensitivity_tier: 1
    """
    digits = re.sub(r"\D", "", raw)
    return digits[-10:] if len(digits) >= 10 else digits


def _looks_like_jid(name: str) -> bool:
    """Return True if *name* looks like a raw JID or phone number."""
    return bool(
        name.endswith("@s.whatsapp.net")
        or name.endswith("@lid")
        or name.endswith("@g.us")
        or re.fullmatch(r"\+?\d[\d\s\-()]+", name)
    )


def _build_contact_lookup(db: DatabaseEngine) -> dict[str, str]:
    """Build a phone/JID/email → display name lookup.

    Sources (priority order):
    1. ``raw_contacts`` — Apple Contacts with phone numbers.
    2. ``raw_messages`` sender_name — already resolved during ingestion
       via the adapter's 8-step chain (push_name, remoteJidAlt, etc.).
    3. ``raw_messages`` chat_name — for 1:1 chats where chat_name
       is a real name (not a JID).
    4. ``raw_contacts`` — Apple Contacts with email addresses.

    sensitivity_tier: 2
    """
    lookup: dict[str, str] = {}

    # Source 1: Apple Contacts (normalized phone → name)
    try:
        rows = db.query(
            "SELECT name, phone FROM raw_contacts "
            "WHERE phone IS NOT NULL AND name IS NOT NULL "
            "AND phone != '' AND name != ''",
        )
        for row in rows:
            phone = _normalize_phone(str(row["phone"]))
            if phone:
                lookup[phone] = str(row["name"])
    except Exception:  # noqa: BLE001
        logger.debug("Could not query raw_contacts for lookup", exc_info=True)

    # Source 2: sender_name from raw_messages (resolved during ingestion)
    try:
        rows = db.query(
            "SELECT DISTINCT sender, sender_name FROM raw_messages "
            "WHERE sender IS NOT NULL AND sender != '' "
            "AND sender != 'me' "
            "AND sender_name IS NOT NULL AND sender_name != '' "
            "AND sender_name != 'Unknown'",
        )
        for row in rows:
            sender = str(row["sender"]).strip()
            name = str(row["sender_name"]).strip()
            if sender and name and not _looks_like_jid(name):
                if sender not in lookup:
                    lookup[sender] = name
                if sender.endswith("@s.whatsapp.net"):
                    phone = _normalize_phone(
                        sender.removesuffix("@s.whatsapp.net"),
                    )
                    if phone and phone not in lookup:
                        lookup[phone] = name
                elif sender.endswith("@lid"):
                    if sender not in lookup:
                        lookup[sender] = name
    except Exception:  # noqa: BLE001
        logger.debug("Could not query sender_name lookup", exc_info=True)

    # Source 3: chat_name from 1:1 chats (JID → name)
    try:
        rows = db.query(
            "SELECT DISTINCT sender, chat_name FROM raw_messages "
            "WHERE is_group = 0 AND chat_name IS NOT NULL "
            "AND chat_name != '' AND sender IS NOT NULL "
            "AND sender != ''",
        )
        for row in rows:
            sender = str(row["sender"]).strip()
            chat_name = str(row["chat_name"]).strip()
            if sender and chat_name and not _looks_like_jid(chat_name):
                if sender not in lookup:
                    lookup[sender] = chat_name
                if sender.endswith("@s.whatsapp.net"):
                    phone = _normalize_phone(
                        sender.removesuffix("@s.whatsapp.net"),
                    )
                    if phone and phone not in lookup:
                        lookup[phone] = chat_name
    except Exception:  # noqa: BLE001
        logger.debug("Could not query chat_name lookup", exc_info=True)

    # Source 4: Apple Contacts (email → name)
    try:
        rows = db.query(
            "SELECT name, email FROM raw_contacts "
            "WHERE email IS NOT NULL AND name IS NOT NULL "
            "AND email != '' AND name != ''",
        )
        for row in rows:
            email = str(row["email"]).strip().lower()
            if email and email not in lookup:
                lookup[email] = str(row["name"])
    except Exception:  # noqa: BLE001
        logger.debug("Could not query raw_contacts email lookup", exc_info=True)

    return lookup


def _resolve_contact_name(
    raw_name: str,
    lookup: dict[str, str],
) -> str:
    """Resolve a raw contact_name to a human-readable name.

    Checks: direct match, @s.whatsapp.net phone extraction,
    @lid JID match, phone-formatted string (``+digits``).

    sensitivity_tier: 2
    """
    # Direct lookup (covers email addresses, JIDs, and any key)
    if raw_name in lookup:
        return lookup[raw_name]

    # Phone extraction from @s.whatsapp.net JID
    if raw_name.endswith("@s.whatsapp.net"):
        phone = _normalize_phone(
            raw_name.removesuffix("@s.whatsapp.net"),
        )
        if phone in lookup:
            return lookup[phone]

    # Phone-formatted name like "+554899334725"
    if raw_name.startswith("+") and raw_name[1:].isdigit():
        phone = _normalize_phone(raw_name)
        if phone in lookup:
            return lookup[phone]

    # @lid JID — try direct lookup (populated from chat_name)
    if raw_name.endswith("@lid") and raw_name in lookup:
        return lookup[raw_name]

    return raw_name


def _build_messages_block(
    msgs: list[dict[str, t.Any]],
) -> str:
    """Build the messages section for a single contact's prompt."""
    lines: list[str] = []
    for m in msgs:
        direction = "\u2192" if m["is_from_me"] else "\u2190"
        dt = m["timestamp"][:10] if m["timestamp"] else "?"
        content = (m["content"] or "")[:200]
        lines.append(f"[{dt}] {direction} {content}")
    return "\n".join(lines)


_SENT_FOLDER_PATTERNS = ("sent", "enviado", "enviad", "gesendet", "envoyé")


def _is_sent_folder(folder: str | None) -> bool:
    """Return True when *folder* looks like a sent-mail folder.

    sensitivity_tier: 1
    """
    if not folder:
        return False
    lower = folder.lower()
    return any(p in lower for p in _SENT_FOLDER_PATTERNS)


def _fetch_email_rows(
    db: DatabaseEngine,
    lookback: int,
    max_per_contact: int,
    contact_lookup: dict[str, str],
) -> dict[str, list[dict[str, t.Any]]]:
    """Fetch emails and normalize to the same shape as message rows.

    Returns a dict of ``{resolved_contact_name: [row, ...]}`` where each
    row has keys ``contact_name``, ``sender``, ``content``, ``timestamp``,
    ``is_from_me`` — matching the shape ``_build_messages_block`` expects.

    sensitivity_tier: 3
    """
    try:
        tables = db.query(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name = 'raw_emails'",
        )
        if not tables:
            return {}
    except Exception:  # noqa: BLE001
        return {}

    try:
        rows = db.query(
            f"""
            SELECT from_address, to_addresses, subject,
                   body_preview, date, folder
            FROM raw_emails
            WHERE date IS NOT NULL
              AND DATE(date) >= DATE('now', '-{lookback} days')
            ORDER BY date DESC
            """,
        )
    except Exception:  # noqa: BLE001
        logger.debug("Could not query raw_emails", exc_info=True)
        return {}

    by_contact: dict[str, list[dict[str, t.Any]]] = {}
    for row in rows:
        from_addr = str(row.get("from_address") or "").strip().lower()
        folder = row.get("folder")
        is_from_me = _is_sent_folder(folder)

        if is_from_me:
            to_raw = row.get("to_addresses")
            recipients = _parse_recipients(to_raw)
            if not recipients:
                continue
            raw_contact = recipients[0].strip().lower()
        else:
            if not from_addr:
                continue
            raw_contact = from_addr

        name = _resolve_contact_name(raw_contact, contact_lookup)

        subject = str(row.get("subject") or "").strip()
        body = str(row.get("body_preview") or "").strip()
        content = f"{subject}\n{body}".strip() if subject else body
        if not content:
            continue

        if name not in by_contact:
            by_contact[name] = []
        if len(by_contact[name]) >= max_per_contact:
            continue

        by_contact[name].append({
            "contact_name": name,
            "sender": from_addr if not is_from_me else "me",
            "content": content[:400],
            "timestamp": str(row.get("date") or ""),
            "is_from_me": 1 if is_from_me else 0,
        })

    return by_contact


def _parse_recipients(raw: t.Any) -> list[str]:
    """Parse ``to_addresses`` (JSON array or string) into a list.

    sensitivity_tier: 2
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(v) for v in raw if v]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(v) for v in parsed if v]
        except (json.JSONDecodeError, TypeError):
            pass
        if raw.strip():
            return [raw.strip()]
    return []


def _extract_topics_single(
    contact_name: str,
    msgs: list[dict[str, t.Any]],
) -> list[dict[str, t.Any]] | None:
    """Extract topics for a single contact via :class:`TopicExtractorAgent`.

    Returns:
        List of topic dicts on success (may be empty for casual chat).
        None on agent failure (used for failure counting).

    sensitivity_tier: 3
    """
    from src.agents.topic_extractor.agent import TopicExtractorAgent

    block = _build_messages_block(msgs)
    try:
        batch = TopicExtractorAgent().extract(
            contact_name=contact_name,
            messages_block=block,
        )
    except Exception as exc:  # noqa: BLE001
        # Let PreemptedError propagate — not an LLM failure
        from src.models.llm_provider import PreemptedError
        if isinstance(exc, PreemptedError):
            raise
        logger.warning(
            "TopicExtractorAgent failed for %s: %s",
            contact_name, exc,
        )
        return None
    if batch is None:
        return None
    return [
        {
            "topic": t_.topic,
            "description": t_.description,
            "importance": t_.importance,
            "status": t_.status,
            "category": t_.category,
        }
        for t_ in batch.topics
    ]


def _validate_topic(
    topic: dict[str, t.Any],
) -> dict[str, t.Any] | None:
    """Validate and normalize a single topic dict."""
    name = str(topic.get("topic", "")).strip()
    if not name or len(name) < 3:
        return None

    importance = topic.get("importance", 5)
    try:
        importance = max(1, min(10, int(importance)))
    except (TypeError, ValueError):
        importance = 5

    status = str(topic.get("status", "active")).lower()
    if status not in ("active", "resolved", "stale"):
        status = "active"

    raw_category = topic.get("category")
    if raw_category is None:
        category = None
    else:
        cat = str(raw_category).lower().strip()
        category = cat if cat in ("personal", "life", "work") else None

    return {
        "topic": name[:200],
        "description": str(topic.get("description", ""))[:500],
        "importance": importance,
        "status": status,
        "category": category,
    }


# v5: topics now carry a personal/life/work category in their payload
_CACHE_VERSION = "v5"


def _msg_fingerprint(
    msgs: list[dict[str, t.Any]],
    max_msgs: int = _REMOTE_MAX_MESSAGES_PER_CONTACT,
) -> str:
    """Hash message IDs + timestamps to detect new messages.

    Includes ``_CACHE_VERSION`` so prompt changes invalidate old caches.
    Sampling is bounded by *max_msgs* to keep fingerprints stable when
    the per-cycle limits shrink (e.g. user switches from remote back to
    local Ollama).
    """
    raw = _CACHE_VERSION + "|" + "|".join(
        f"{m.get('timestamp', '')}:{len(m.get('content', ''))}"
        for m in msgs[:max_msgs]
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _ensure_cache_table(db: DatabaseEngine) -> None:
    """Create the cache table if it doesn't exist.

    sensitivity_tier: 1
    """
    try:
        db.execute(f"""
            CREATE TABLE IF NOT EXISTS {_CACHE_TABLE} (
                contact_name    TEXT PRIMARY KEY,
                fingerprint     TEXT NOT NULL,
                topics_json     TEXT NOT NULL,
                extracted_at    TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
    except Exception:  # noqa: BLE001
        logger.debug("Cache table creation skipped", exc_info=True)


def _get_cached_topics(
    db: DatabaseEngine,
    contact_name: str,
    fingerprint: str,
) -> list[dict[str, t.Any]] | None:
    """Return cached topics if fingerprint matches, else None.

    sensitivity_tier: 1
    """
    try:
        rows = db.query(
            f"SELECT topics_json FROM {_CACHE_TABLE} "
            "WHERE contact_name = ? AND fingerprint = ?",
            [contact_name, fingerprint],
        )
        if rows:
            return json.loads(rows[0]["topics_json"])
    except Exception:  # noqa: BLE001
        pass
    return None


def _store_cached_topics(
    db: DatabaseEngine,
    contact_name: str,
    fingerprint: str,
    topics: list[dict[str, t.Any]],
) -> None:
    """Store extracted topics in cache.

    sensitivity_tier: 1
    """
    try:
        db.execute(
            f"INSERT OR REPLACE INTO {_CACHE_TABLE} "
            "(contact_name, fingerprint, topics_json, extracted_at) "
            "VALUES (?, ?, ?, datetime('now'))",
            [contact_name, fingerprint, json.dumps(topics)],
        )
    except Exception:  # noqa: BLE001
        logger.debug(
            "Could not cache topics for %s", contact_name,
        )


def execute(db: DatabaseEngine) -> list[dict[str, t.Any]]:
    """Extract per-contact conversation topics via LLM.

    Uses a cache table to skip contacts whose messages haven't
    changed since last extraction. Only contacts with new messages
    trigger LLM calls.

    sensitivity_tier: 3
    """
    _ensure_cache_table(db)

    lookback, max_msgs, remote_capable = _runtime_limits()

    # Build contact name lookup from raw_contacts + chat names
    contact_lookup = _build_contact_lookup(db)

    # Find chats where the user actively participates (sent messages).
    # Includes both 1:1 contacts and groups the user writes in.
    # Skips groups the user only reads silently.
    active_chats = db.query(
        f"""
        SELECT DISTINCT chat_name
        FROM raw_messages
        WHERE is_from_me = 1
          AND chat_name IS NOT NULL AND chat_name != ''
          AND DATE(timestamp) >= DATE('now', '-{lookback} days')
        """
    )
    active_chat_set = {str(r["chat_name"]) for r in active_chats}

    logger.info(
        "Found %d active chats in last %d days (remote=%s)",
        len(active_chat_set), lookback, remote_capable,
    )

    # Group by contact, resolve names, and merge duplicates
    by_contact: dict[str, list[dict[str, t.Any]]] = {}

    # Source 1: WhatsApp / raw_messages
    if active_chat_set:
        placeholders = ",".join("?" for _ in active_chat_set)
        contact_rows = db.query(
            f"""
            SELECT
                COALESCE(
                    NULLIF(NULLIF(NULLIF(m.sender_name, ''), 'Unknown'), 'me'),
                    m.sender
                ) AS contact_name,
                m.sender,
                m.content,
                m.timestamp,
                m.is_from_me
            FROM raw_messages m
            WHERE m.sender IS NOT NULL AND m.sender != ''
              AND DATE(m.timestamp) >= DATE('now', '-{lookback} days')
              AND m.content IS NOT NULL AND m.content != ''
              AND m.chat_name IN ({placeholders})
            ORDER BY m.timestamp DESC
            """,
            list(active_chat_set),
        )

        for row in contact_rows:
            raw_name = row["contact_name"]
            if raw_name == "me":
                continue
            name = _resolve_contact_name(raw_name, contact_lookup)
            if _looks_like_jid(name):
                sender_resolved = _resolve_contact_name(
                    row["sender"], contact_lookup,
                )
                if not _looks_like_jid(sender_resolved):
                    name = sender_resolved
            if name not in by_contact:
                by_contact[name] = []
            if len(by_contact[name]) < max_msgs:
                by_contact[name].append(row)

    # Source 2: emails from raw_emails
    email_groups = _fetch_email_rows(
        db, lookback, max_msgs, contact_lookup,
    )
    for name, msgs in email_groups.items():
        if name in by_contact:
            merged = sorted(
                by_contact[name] + msgs,
                key=lambda m: m.get("timestamp") or "",
                reverse=True,
            )[:max_msgs]
            by_contact[name] = merged
        else:
            by_contact[name] = msgs

    # Filter to contacts with enough messages
    active = {
        k: v for k, v in by_contact.items()
        if len(v) >= _MIN_MESSAGES
    }

    if not active:
        return []

    # Check cache: split into cached (unchanged) and needs_llm (new)
    cached_topics: dict[str, list[dict[str, t.Any]]] = {}
    needs_llm: dict[str, list[dict[str, t.Any]]] = {}

    for name, msgs in active.items():
        fp = _msg_fingerprint(msgs)
        cached = _get_cached_topics(db, name, fp)
        if cached is not None:
            cached_topics[name] = cached
        else:
            needs_llm[name] = msgs

    logger.info(
        "Topic extraction: %d contacts cached, %d need LLM "
        "(from %d active, %d total)",
        len(cached_topics),
        len(needs_llm),
        len(active),
        len(by_contact),
    )

    # Process contacts needing LLM (if any). TopicExtractorAgent runs
    # one contact at a time — the legacy multi-contact batch prompt is
    # gone (each contact is now an isolated SBAgent invocation routed
    # through the scheduler + firewalls + audit chain).
    llm_topics: dict[str, list[dict[str, t.Any]]] = {}
    if needs_llm:
        from src.models.llm_provider import PreemptedError
        consecutive_failures = 0
        for name, msgs in needs_llm.items():
            try:
                topics = _extract_topics_single(name, msgs)
            except PreemptedError:
                logger.info(
                    "Preempted during topic extraction for %s "
                    "— will retry on next run",
                    name,
                )
                break

            if topics is None:
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    logger.warning(
                        "TopicExtractorAgent unavailable — 3 "
                        "consecutive failures, stopping",
                    )
                    break
                continue

            consecutive_failures = 0
            llm_topics[name] = topics
            _store_cached_topics(
                db, name, _msg_fingerprint(msgs), topics,
            )

    # Merge cached + fresh LLM results
    all_topics = {**cached_topics, **llm_topics}

    # Build output rows
    rows: list[dict[str, t.Any]] = []
    for contact_name, topics in all_topics.items():
        if not isinstance(topics, list):
            continue
        for raw_topic in topics:
            if not isinstance(raw_topic, dict):
                continue
            topic = _validate_topic(raw_topic)
            if topic is None:
                continue

            msgs = active.get(contact_name, [])
            last_ts = msgs[0]["timestamp"] if msgs else None
            first_ts = msgs[-1]["timestamp"] if msgs else None

            rows.append(
                {
                    "contact_name": contact_name,
                    "topic": topic["topic"],
                    "description": topic["description"],
                    "importance": topic["importance"],
                    "status": topic["status"],
                    "category": topic.get("category"),
                    "first_seen": first_ts,
                    "last_seen": last_ts,
                    "sensitivity_tier": 3,
                }
            )

    logger.info(
        "Total %d topics across %d contacts (%d from cache)",
        len(rows),
        len(set(r["contact_name"] for r in rows)),
        len(cached_topics),
    )
    return rows
