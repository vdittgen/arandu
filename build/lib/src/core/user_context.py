"""User context assembly for BrainAgent system prompts.

Reads user profile fields from ``~/.arandu/settings.json`` and computes
dynamic context (current date/time, age).  The assembled text is injected
into the BrainAgent system prompt so the LLM knows who the user is and
when they are asking.

``infer_user_profile()`` auto-detects profile fields from existing data
(WhatsApp phone country code, contacts, emails) so the user doesn't have
to fill everything manually.

sensitivity_tier: 2 (user name, location, birthday, bio)
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any

from src.models.llm_provider import load_llm_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Country-code → (country, default_timezone, primary_language)
# Used by infer_user_profile() to map WhatsApp phone prefixes.
# ---------------------------------------------------------------------------
COUNTRY_CODE_MAP: dict[str, tuple[str, str, str]] = {
    "1": ("United States", "America/New_York", "English"),
    "7": ("Russia", "Europe/Moscow", "Russian"),
    "27": ("South Africa", "Africa/Johannesburg", "English"),
    "30": ("Greece", "Europe/Athens", "Greek"),
    "31": ("Netherlands", "Europe/Amsterdam", "Dutch"),
    "32": ("Belgium", "Europe/Brussels", "French"),
    "33": ("France", "Europe/Paris", "French"),
    "34": ("Spain", "Europe/Madrid", "Spanish"),
    "36": ("Hungary", "Europe/Budapest", "Hungarian"),
    "39": ("Italy", "Europe/Rome", "Italian"),
    "41": ("Switzerland", "Europe/Zurich", "German"),
    "44": ("United Kingdom", "Europe/London", "English"),
    "46": ("Sweden", "Europe/Stockholm", "Swedish"),
    "47": ("Norway", "Europe/Oslo", "Norwegian"),
    "48": ("Poland", "Europe/Warsaw", "Polish"),
    "49": ("Germany", "Europe/Berlin", "German"),
    "51": ("Peru", "America/Lima", "Spanish"),
    "52": ("Mexico", "America/Mexico_City", "Spanish"),
    "53": ("Cuba", "America/Havana", "Spanish"),
    "54": ("Argentina", "America/Argentina/Buenos_Aires", "Spanish"),
    "55": ("Brazil", "America/Sao_Paulo", "Portuguese"),
    "56": ("Chile", "America/Santiago", "Spanish"),
    "57": ("Colombia", "America/Bogota", "Spanish"),
    "58": ("Venezuela", "America/Caracas", "Spanish"),
    "60": ("Malaysia", "Asia/Kuala_Lumpur", "Malay"),
    "61": ("Australia", "Australia/Sydney", "English"),
    "62": ("Indonesia", "Asia/Jakarta", "Indonesian"),
    "63": ("Philippines", "Asia/Manila", "Filipino"),
    "64": ("New Zealand", "Pacific/Auckland", "English"),
    "65": ("Singapore", "Asia/Singapore", "English"),
    "66": ("Thailand", "Asia/Bangkok", "Thai"),
    "81": ("Japan", "Asia/Tokyo", "Japanese"),
    "82": ("South Korea", "Asia/Seoul", "Korean"),
    "84": ("Vietnam", "Asia/Ho_Chi_Minh", "Vietnamese"),
    "86": ("China", "Asia/Shanghai", "Chinese"),
    "90": ("Turkey", "Europe/Istanbul", "Turkish"),
    "91": ("India", "Asia/Kolkata", "Hindi"),
    "92": ("Pakistan", "Asia/Karachi", "Urdu"),
    "93": ("Afghanistan", "Asia/Kabul", "Pashto"),
    "98": ("Iran", "Asia/Tehran", "Persian"),
    "212": ("Morocco", "Africa/Casablanca", "Arabic"),
    "213": ("Algeria", "Africa/Algiers", "Arabic"),
    "234": ("Nigeria", "Africa/Lagos", "English"),
    "351": ("Portugal", "Europe/Lisbon", "Portuguese"),
    "352": ("Luxembourg", "Europe/Luxembourg", "French"),
    "353": ("Ireland", "Europe/Dublin", "English"),
    "354": ("Iceland", "Atlantic/Reykjavik", "Icelandic"),
    "358": ("Finland", "Europe/Helsinki", "Finnish"),
    "370": ("Lithuania", "Europe/Vilnius", "Lithuanian"),
    "380": ("Ukraine", "Europe/Kyiv", "Ukrainian"),
    "506": ("Costa Rica", "America/Costa_Rica", "Spanish"),
    "507": ("Panama", "America/Panama", "Spanish"),
    "591": ("Bolivia", "America/La_Paz", "Spanish"),
    "593": ("Ecuador", "America/Guayaquil", "Spanish"),
    "595": ("Paraguay", "America/Asuncion", "Spanish"),
    "598": ("Uruguay", "America/Montevideo", "Spanish"),
    "852": ("Hong Kong", "Asia/Hong_Kong", "Chinese"),
    "886": ("Taiwan", "Asia/Taipei", "Chinese"),
    "972": ("Israel", "Asia/Jerusalem", "Hebrew"),
    "971": ("UAE", "Asia/Dubai", "Arabic"),
    "966": ("Saudi Arabia", "Asia/Riyadh", "Arabic"),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_user_context(
    settings: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> str:
    """Assemble user context text for LLM system prompt injection.

    Args:
        settings: Pre-loaded settings dict.  If ``None``, reads from disk.
        now: Current datetime for testing.  If ``None``, uses real time.

    Returns:
        Formatted context string.  Always contains at least the current
        date/time line so the LLM knows when the user is asking.

    sensitivity_tier: 2
    """
    if settings is None:
        settings = load_llm_settings()

    parts: list[str] = []

    # --- Static profile fields ---
    name = settings.get("user_name")
    if name:
        parts.append(f"User's name: {name}")

    birthday = settings.get("user_birthday")
    if birthday:
        age = _compute_age(birthday, now)
        if age is not None:
            parts.append(f"User's age: {age}")

    location = settings.get("user_location")
    if location:
        parts.append(f"User's location: {location}")

    tz_name = settings.get("user_timezone")
    if tz_name:
        parts.append(f"User's timezone: {tz_name}")

    language = settings.get("user_language")
    if language:
        parts.append(f"User's preferred language: {language}")

    bio = settings.get("user_bio")
    if bio:
        parts.append(f"About the user: {bio}")

    # --- Dynamic context (computed at call time) ---
    current = now or datetime.now(tz=timezone.utc)

    # Resolve timezone: explicit setting → system timezone → UTC
    effective_tz = tz_name
    if not effective_tz:
        effective_tz = _get_system_timezone()

    if effective_tz:
        try:
            import zoneinfo

            tz = zoneinfo.ZoneInfo(effective_tz)
            current = current.astimezone(tz)
        except (ImportError, KeyError):
            logger.debug("Could not resolve timezone %s", effective_tz)

    date_en = (
        f"{current.strftime('%A, %B %d, %Y at %H:%M')} "
        f"({current.strftime('%Y-%m-%d')})"
    )
    localized = _localize_date(current, language)
    if localized:
        parts.append(
            f"Current date and time: {localized} — {date_en}"
        )
    else:
        parts.append(f"Current date and time: {date_en}")

    return "--- User Context ---\n" + "\n".join(parts)


def build_learned_facts_context(
    db_engine: Any,
    max_facts: int = 15,
    max_chars: int = 2000,
) -> str:
    """Assemble learned facts text for LLM system prompt injection.

    Reads active, non-dismissed facts from ``_learned_facts`` and
    groups them by category for readability.  Confirmed facts are
    prioritized.  Increments ``times_used`` for retrieved facts.

    Args:
        db_engine: A ``DatabaseEngine`` instance for reading facts.
        max_facts: Maximum facts to include.
        max_chars: Character budget for the output.

    Returns:
        Formatted facts string, or empty string if no facts.

    sensitivity_tier: 2
    """
    try:
        # Check if table exists
        tables = db_engine.query(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name = '_learned_facts'"
        )
        if not tables:
            return ""

        rows = db_engine.query(
            "SELECT id, category, content, confirmed_at "
            "FROM _learned_facts "
            "WHERE dismissed_at IS NULL "
            "AND superseded_by IS NULL "
            "AND confidence >= 0.5 "
            "ORDER BY "
            "  CASE WHEN confirmed_at IS NOT NULL "
            "    THEN 0 ELSE 1 END, "
            "  confidence DESC, "
            "  extracted_at DESC "
            f"LIMIT {max_facts}"
        )
        if not rows:
            return ""

        # Group by category
        by_category: dict[str, list[str]] = {}
        fact_ids: list[str] = []
        total_chars = 0

        for row in rows:
            content = str(row["content"])
            if total_chars + len(content) > max_chars:
                break
            cat = str(row["category"]).title()
            by_category.setdefault(cat, []).append(content)
            fact_ids.append(str(row["id"]))
            total_chars += len(content)

        if not by_category:
            return ""

        # Increment usage counters (non-fatal)
        if fact_ids:
            try:
                from src.agents.fact_extractor import FactLearner

                learner = FactLearner(db_engine=db_engine)
                learner.increment_usage(fact_ids)
            except Exception:  # noqa: BLE001
                pass

        # Format output
        lines: list[str] = ["--- Learned Facts (from conversations) ---"]
        for cat, facts in by_category.items():
            lines.append(f"{cat}: {'. '.join(facts)}")

        return "\n".join(lines)

    except Exception:  # noqa: BLE001
        logger.debug("Failed to build learned facts context", exc_info=True)
        return ""


def build_active_topics_context(
    db_engine: Any,
    max_contacts: int = 10,
    max_chars: int = 1500,
) -> str:
    """Assemble active conversation topics for LLM system prompt.

    Reads ``mart_contact_summary`` for contacts with important ongoing
    topics and formats them as a text block.  Gives BrainAgent awareness
    of what matters most in the user's conversations.

    Args:
        db_engine: A ``DatabaseEngine`` instance for querying.
        max_contacts: Maximum contacts to include.
        max_chars: Character budget for the output.

    Returns:
        Formatted topics string, or empty string if none available.

    sensitivity_tier: 2
    """
    if db_engine is None:
        return ""

    try:
        rows = db_engine.query(
            "SELECT contact_name, top_topic, max_topic_importance, "
            "active_topics_json, messages_7d "
            "FROM mart_contact_summary "
            "WHERE max_topic_importance >= 5 "
            "  AND top_topic IS NOT NULL "
            "ORDER BY notification_priority DESC "
            f"LIMIT {int(max_contacts)}"
        )
        if not rows:
            return ""

        lines: list[str] = ["--- Active Topics (from conversations) ---"]
        total_chars = 0

        for row in rows:
            name = row.get("contact_name", "?")
            importance = row.get("max_topic_importance", 5)
            msgs = row.get("messages_7d", 0)

            header = (
                f"{name} (importance {importance}/10, "
                f"{msgs} msgs this week):"
            )

            # Parse individual topics from JSON
            topic_lines: list[str] = []
            raw_json = row.get("active_topics_json")
            if raw_json:
                try:
                    import json

                    topics = json.loads(raw_json) if isinstance(
                        raw_json, str,
                    ) else raw_json
                    for t in topics[:3]:
                        if not isinstance(t, dict):
                            continue
                        t_name = t.get("topic", "")
                        t_status = t.get("status", "active")
                        t_imp = t.get("importance", 5)
                        if t_name:
                            topic_lines.append(
                                f"  - {t_name} "
                                f"({t_status}, importance {t_imp})"
                            )
                except (ValueError, TypeError, KeyError):
                    pass

            # Fallback: use top_topic if JSON parsing failed
            if not topic_lines:
                top = row.get("top_topic", "")
                if top:
                    topic_lines.append(
                        f"  - {top} (active, importance {importance})"
                    )

            if not topic_lines:
                continue

            block = header + "\n" + "\n".join(topic_lines)
            if total_chars + len(block) > max_chars:
                break
            lines.append(block)
            total_chars += len(block)

        if len(lines) <= 1:
            return ""

        return "\n".join(lines)

    except Exception:  # noqa: BLE001
        logger.debug(
            "Failed to build active topics context", exc_info=True,
        )
        return ""


def infer_user_profile(
    data_layer: Any,
) -> dict[str, str]:
    """Infer user characteristics from available data.

    Analyzes WhatsApp phone number (country code), contacts, and emails
    to auto-detect the user's name, location, timezone, and language.

    Only returns fields it can confidently infer.  Never infers birthday
    or bio (too personal — user must set manually).

    Args:
        data_layer: A ``DataLayer`` instance with DuckDB access.

    Returns:
        Dict of inferred profile field names → values.
        Empty dict if no data is available for inference.

    sensitivity_tier: 2
    """
    inferred: dict[str, str] = {}

    # 1. Try WhatsApp phone → country/timezone/language
    country_info = _infer_from_phone()
    if country_info:
        country, tz, lang = country_info
        inferred["user_location"] = country
        inferred["user_timezone"] = tz
        inferred["user_language"] = lang

    # 2. Fallback timezone: system timezone
    if "user_timezone" not in inferred:
        sys_tz = _get_system_timezone()
        if sys_tz:
            inferred["user_timezone"] = sys_tz

    # 3. Try to find the user's name from contacts or emails
    name = _infer_name(data_layer)
    if name:
        inferred["user_name"] = name

    return inferred


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


# Weekday/month names for languages where small LLMs hallucinate
# translations.  Only languages where we've seen real errors.
_WEEKDAYS: dict[str, list[str]] = {
    "português": [
        "segunda-feira", "terça-feira", "quarta-feira",
        "quinta-feira", "sexta-feira", "sábado", "domingo",
    ],
    "spanish": [
        "lunes", "martes", "miércoles", "jueves",
        "viernes", "sábado", "domingo",
    ],
}

_MONTHS: dict[str, list[str]] = {
    "português": [
        "janeiro", "fevereiro", "março", "abril", "maio",
        "junho", "julho", "agosto", "setembro", "outubro",
        "novembro", "dezembro",
    ],
    "spanish": [
        "enero", "febrero", "marzo", "abril", "mayo",
        "junio", "julio", "agosto", "septiembre", "octubre",
        "noviembre", "diciembre",
    ],
}


def _localize_date(
    dt: datetime,
    language: str | None,
) -> str | None:
    """Format date in the user's language if supported.

    Returns localized string or None if language is unsupported.

    sensitivity_tier: 1
    """
    if not language:
        return None
    lang_key = language.lower().strip()
    weekdays = _WEEKDAYS.get(lang_key)
    months = _MONTHS.get(lang_key)
    if not weekdays or not months:
        return None
    wday = weekdays[dt.weekday()]
    month = months[dt.month - 1]
    return (
        f"{wday}, {dt.day} de {month} de {dt.year} "
        f"às {dt.strftime('%H:%M')}"
    )


def _compute_age(birthday_iso: str, now: datetime | None = None) -> int | None:
    """Compute age in years from an ISO date string.

    sensitivity_tier: 2
    """
    try:
        bd = date.fromisoformat(birthday_iso)
        today = (now or datetime.now(tz=timezone.utc)).date()
        age = today.year - bd.year
        if (today.month, today.day) < (bd.month, bd.day):
            age -= 1
        return age if age >= 0 else None
    except (ValueError, TypeError):
        return None


def _infer_from_phone() -> tuple[str, str, str] | None:
    """Extract country info from WhatsApp phone number.

    Reads the user's bare JID (e.g. ``"554892011083"``) and matches
    the longest country code prefix against ``COUNTRY_CODE_MAP``.

    Returns ``(country, timezone, language)`` or ``None``.

    sensitivity_tier: 2
    """
    try:
        from src.extensions.bridges.whatsapp.paths import resolve_self_jid

        jid = resolve_self_jid()
        if not jid:
            return None

        # Try longest prefix first (3 digits, then 2, then 1)
        for length in (3, 2, 1):
            prefix = jid[:length]
            if prefix in COUNTRY_CODE_MAP:
                return COUNTRY_CODE_MAP[prefix]

        return None
    except Exception:  # noqa: BLE001
        logger.debug("Failed to infer from WhatsApp phone", exc_info=True)
        return None


def _get_system_timezone() -> str | None:
    """Get the system's IANA timezone name.

    sensitivity_tier: 1
    """
    try:
        import zoneinfo

        # Python 3.12+ has datetime.now().astimezone().tzname()
        # but we need the IANA name, not the abbreviation.
        # Try the tzpath approach first.
        local_tz = datetime.now(tz=timezone.utc).astimezone().tzinfo
        if local_tz is not None:
            tz_name = str(local_tz)
            # Check if it's a valid IANA name
            try:
                zoneinfo.ZoneInfo(tz_name)
                return tz_name
            except (KeyError, ValueError):
                pass

        # Fallback: read /etc/localtime symlink (macOS/Linux)
        import os

        localtime = "/etc/localtime"
        if os.path.islink(localtime):
            target = os.path.realpath(localtime)
            # e.g. /usr/share/zoneinfo/America/Sao_Paulo
            parts = target.split("/zoneinfo/")
            if len(parts) == 2:
                tz_name = parts[1]
                zoneinfo.ZoneInfo(tz_name)
                return tz_name
    except Exception:  # noqa: BLE001
        logger.debug("Failed to detect system timezone", exc_info=True)

    return None


def _infer_name(data_layer: Any) -> str | None:
    """Infer the user's name from contacts or emails.

    Strategy 1: Cross-reference WhatsApp self JID phone with contacts.
    Strategy 2: Find the most frequent ``from_address`` in sent emails.

    sensitivity_tier: 2
    """
    try:
        # Strategy 1: WhatsApp phone → contacts name lookup
        name = _name_from_contacts(data_layer)
        if name:
            return name

        # Strategy 2: Email from_address
        name = _name_from_emails(data_layer)
        if name:
            return name
    except Exception:  # noqa: BLE001
        logger.debug("Failed to infer user name", exc_info=True)

    return None


def _name_from_contacts(data_layer: Any) -> str | None:
    """Find user's name by matching WhatsApp phone to contacts.

    sensitivity_tier: 2
    """
    try:
        from src.extensions.bridges.whatsapp.paths import resolve_self_jid

        jid = resolve_self_jid()
        if not jid:
            return None

        # The JID is the full phone number (e.g. "554892011083")
        # Contacts may store phone in various formats.
        # Search for contacts whose phone contains the last 8-9 digits.
        phone_suffix = jid[-9:] if len(jid) > 9 else jid[-8:]

        rows = data_layer.duckdb.query(
            "SELECT name FROM raw_contacts "
            "WHERE phone IS NOT NULL "
            "AND REPLACE(REPLACE(REPLACE(phone, ' ', ''), '-', ''), '+', '') "
            "LIKE '%' || ? "
            "AND name IS NOT NULL AND name != '' "
            "LIMIT 1",
            [phone_suffix],
        )
        if rows:
            return str(rows[0][0]).strip() or None
    except Exception:  # noqa: BLE001
        logger.debug("Failed to match phone to contacts", exc_info=True)

    return None


def _name_from_emails(data_layer: Any) -> str | None:
    """Extract user's name from most frequent sent email address.

    Looks for a name part in the from_address field
    (e.g. ``"Vinicius Dittgen <vini@email.com>"``).

    sensitivity_tier: 2
    """
    try:
        rows = data_layer.duckdb.query(
            "SELECT from_address, COUNT(*) as cnt "
            "FROM raw_emails "
            "WHERE from_address IS NOT NULL AND from_address != '' "
            "GROUP BY from_address "
            "ORDER BY cnt DESC "
            "LIMIT 1",
        )
        if not rows:
            return None

        addr = str(rows[0][0]).strip()
        # Try to extract name from "Name <email>" format
        if "<" in addr:
            name_part = addr.split("<")[0].strip().strip('"').strip("'")
            if name_part and len(name_part) > 1:
                return name_part

        return None
    except Exception:  # noqa: BLE001
        logger.debug("Failed to extract name from emails", exc_info=True)

    return None
