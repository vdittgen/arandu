"""Skill registry — stateless utilities available to agents.

Skills are pure functions that may use the LLM but cannot access
databases directly. They are registered globally and discovered
by agents via ``call_skill()``.

sensitivity_tier: 1 (skill metadata only)
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.agents.firewall.egress_firewall import Lane
from src.models.llm_gateway import GatewayBlocked, chat_via_firewalls

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Skill:
    """A registered stateless skill.

    sensitivity_tier: 1
    """

    id: str
    name: str
    description: str
    execute_fn: Callable[..., Any]
    parameters: dict[str, str] = field(default_factory=dict)
    category: str = "general"
    uses_llm: bool = False


class SkillRegistry:
    """Global registry of stateless skills.

    sensitivity_tier: 1
    """

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        """Add a skill to the registry.

        sensitivity_tier: 1
        """
        self._skills[skill.id] = skill

    def get(self, skill_id: str) -> Skill | None:
        """Look up a skill by ID.

        sensitivity_tier: 1
        """
        return self._skills.get(skill_id)

    def list_skills(self) -> list[Skill]:
        """Return all registered skills.

        sensitivity_tier: 1
        """
        return list(self._skills.values())

    def search(self, query: str) -> list[Skill]:
        """Search skills by name or description (substring match).

        sensitivity_tier: 1
        """
        q = query.lower()
        return [
            s
            for s in self._skills.values()
            if q in s.name.lower() or q in s.description.lower()
        ]

    def execute(self, skill_id: str, **kwargs: Any) -> Any:
        """Execute a skill by ID.

        Raises:
            KeyError: If skill_id is not registered.

        sensitivity_tier: 1
        """
        skill = self._skills.get(skill_id)
        if skill is None:
            msg = f"Skill '{skill_id}' not found"
            raise KeyError(msg)
        return skill.execute_fn(**kwargs)

    def register_builtin_skills(self) -> None:
        """Register the default built-in skills.

        sensitivity_tier: 1
        """
        self.register(Skill(
            id="summarize-text",
            name="Summarize Text",
            description=(
                "Summarize long text into a concise "
                "paragraph using the local LLM."
            ),
            execute_fn=_skill_summarize_text,
            parameters={
                "text": "The text to summarize",
                "max_length": "Max output words (default 100)",
            },
            category="text",
            uses_llm=True,
        ))
        self.register(Skill(
            id="classify-sentiment",
            name="Classify Sentiment",
            description=(
                "Classify text as positive, negative, "
                "or neutral using the local LLM."
            ),
            execute_fn=_skill_classify_sentiment,
            parameters={"text": "The text to classify"},
            category="text",
            uses_llm=True,
        ))
        self.register(Skill(
            id="extract-dates",
            name="Extract Dates",
            description=(
                "Extract date references from natural "
                "language text using regex patterns."
            ),
            execute_fn=_skill_extract_dates,
            parameters={
                "text": "The text to extract dates from",
            },
            category="text",
            uses_llm=False,
        ))
        self.register(Skill(
            id="format-markdown",
            name="Format Markdown",
            description="Clean and format text into well-structured markdown.",
            execute_fn=_skill_format_markdown,
            parameters={"text": "The text to format"},
            category="text",
            uses_llm=False,
        ))
        self.register(Skill(
            id="classify-emotion",
            name="Classify Emotion",
            description=(
                "Classify text into a nuanced emotion "
                "category (happy, sad, angry, anxious, etc.) "
                "using the local LLM."
            ),
            execute_fn=_skill_classify_emotion,
            parameters={"text": "The text to classify"},
            category="text",
            uses_llm=True,
        ))
        self.register(Skill(
            id="extract-entities",
            name="Extract Entities",
            description=(
                "Extract named entities (people, places, "
                "dates, emails, organizations) from text "
                "using regex and local LLM."
            ),
            execute_fn=_skill_extract_entities,
            parameters={"text": "The text to extract from"},
            category="analysis",
            uses_llm=True,
        ))
        self.register(Skill(
            id="classify-sensitivity",
            name="Classify Sensitivity",
            description=(
                "Classify text into a data sensitivity tier "
                "(1=low, 2=medium, 3=high) using the same "
                "rules as the firewall."
            ),
            execute_fn=_skill_classify_sensitivity,
            parameters={"text": "The text to classify"},
            category="analysis",
            uses_llm=False,
        ))


# ---------------------------------------------------------------------------
# Built-in skill implementations
# ---------------------------------------------------------------------------


def _skill_summarize_text(
    text: str,
    max_length: int = 100,
    ollama_host: str = "http://localhost:11434",  # noqa: ARG001
) -> str:
    """Summarize text via the firewall gateway.

    sensitivity_tier: varies (depends on input text)
    """
    try:
        resp = chat_via_firewalls(
            [{
                "role": "user",
                "content": (
                    f"Summarize the following text in at most "
                    f"{max_length} words. "
                    f"Be concise and factual.\n\n{text}"
                ),
            }],
            agent_id="skill.summarize-text",
            lane=Lane.CLASSIFIER,
            agent_max_tier=2,
        )
        return resp.content
    except GatewayBlocked:
        logger.info("Summarize skill blocked by firewall", exc_info=True)
    except Exception:
        logger.warning(
            "LLM summarize failed, returning truncated text",
            exc_info=True,
        )
    words = text.split()
    if len(words) > max_length:
        return " ".join(words[:max_length]) + "..."
    return " ".join(words[:max_length])


def _skill_classify_sentiment(
    text: str,
    ollama_host: str = "http://localhost:11434",  # noqa: ARG001
) -> str:
    """Classify sentiment via the firewall gateway.

    sensitivity_tier: varies (depends on input text)
    """
    try:
        resp = chat_via_firewalls(
            [{
                "role": "user",
                "content": (
                    "Classify the sentiment of the following "
                    "text as exactly one of: positive, negative, "
                    "neutral. Reply with only the label.\n\n"
                    f"{text}"
                ),
            }],
            agent_id="skill.classify-sentiment",
            lane=Lane.CLASSIFIER,
            agent_max_tier=2,
        )
        label = resp.content.strip().lower()
        if label in ("positive", "negative", "neutral"):
            return label
        return "neutral"
    except GatewayBlocked:
        logger.info("Sentiment skill blocked by firewall", exc_info=True)
        return "neutral"
    except Exception:
        logger.warning(
            "LLM sentiment failed, defaulting to neutral",
            exc_info=True,
        )
        return "neutral"


_DATE_PATTERNS = [
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    re.compile(
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        r"[a-z]*\s+\d{1,2},?\s*\d{4}\b",
        re.IGNORECASE,
    ),
]


def _skill_extract_dates(text: str) -> list[str]:
    """Extract date-like strings from text using regex.

    sensitivity_tier: 1 (no user data stored)
    """
    dates: list[str] = []
    for pattern in _DATE_PATTERNS:
        dates.extend(pattern.findall(text))
    return dates


def _skill_format_markdown(text: str) -> str:
    """Clean and format text as markdown.

    sensitivity_tier: 1 (text transformation only)
    """
    lines = text.strip().split("\n")
    formatted: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            formatted.append("")
        elif stripped.startswith("- ") or stripped.startswith("* "):
            formatted.append(stripped)
        elif stripped.endswith(":"):
            formatted.append(f"\n**{stripped}**")
        else:
            formatted.append(stripped)
    return "\n".join(formatted)


_VALID_EMOTIONS = frozenset({
    "happy", "sad", "angry", "anxious", "fearful",
    "surprised", "disgusted", "hopeful", "neutral",
})


def _skill_classify_emotion(
    text: str,
    ollama_host: str = "http://localhost:11434",  # noqa: ARG001
) -> str:
    """Classify emotion via the firewall gateway.

    Emotion analysis touches Tier 3 — the gateway pins it locally
    unless the user has opted into Tier 3 remote egress.

    sensitivity_tier: varies (depends on input text)
    """
    try:
        resp = chat_via_firewalls(
            [{
                "role": "user",
                "content": (
                    "Classify the primary emotion of the "
                    "following text as exactly one of: happy, "
                    "sad, angry, anxious, fearful, surprised, "
                    "disgusted, hopeful, neutral. "
                    "Reply with only the label.\n\n"
                    f"{text}"
                ),
            }],
            agent_id="skill.classify-emotion",
            lane=Lane.CLASSIFIER,
            agent_max_tier=3,
        )
        label = resp.content.strip().lower()
        return label if label in _VALID_EMOTIONS else "neutral"
    except GatewayBlocked:
        logger.info("Emotion skill blocked by firewall", exc_info=True)
        return "neutral"
    except Exception:
        logger.warning(
            "LLM emotion classification failed",
            exc_info=True,
        )
        return "neutral"


_ENTITY_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "email": [re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")],
    "phone": [re.compile(r"\+?\d[\d\-\s]{8,}\d")],
    "date": _DATE_PATTERNS,
    "money": [re.compile(r"\$[\d,]+\.?\d*")],
}


def _skill_extract_entities(
    text: str,
    ollama_host: str = "http://localhost:11434",  # noqa: ARG001
) -> dict[str, list[str]]:
    """Extract named entities using regex + optional LLM via the gateway.

    Phase 1: regex for emails, phones, dates, money.
    Phase 2: LLM for people, places, organizations.
    Falls back to regex-only when the gateway blocks the call (Tier 3,
    Ollama offline, etc.).

    sensitivity_tier: varies (depends on input text)
    """
    import json as _json

    entities: dict[str, list[str]] = {}

    # Phase 1: regex extraction
    for entity_type, patterns in _ENTITY_PATTERNS.items():
        matches: list[str] = []
        for pattern in patterns:
            matches.extend(pattern.findall(text))
        if matches:
            entities[entity_type] = list(set(matches))

    # Phase 2: LLM extraction for names, places, orgs
    try:
        resp = chat_via_firewalls(
            [{
                "role": "user",
                "content": (
                    "Extract named entities from the text "
                    "below. Return a JSON object with keys: "
                    "'people', 'places', 'organizations'. "
                    "Each value is a list of strings. "
                    "If none found, use empty lists.\n\n"
                    f"{text}"
                ),
            }],
            agent_id="skill.extract-entities",
            lane=Lane.CLASSIFIER,
            agent_max_tier=2,
        )
        try:
            llm_entities = _json.loads(resp.content)
        except (TypeError, ValueError):
            llm_entities = {}
        if isinstance(llm_entities, dict):
            for key in ("people", "places", "organizations"):
                vals = llm_entities.get(key, [])
                if vals and isinstance(vals, list):
                    entities[key] = [str(v) for v in vals]
    except GatewayBlocked:
        logger.info(
            "Entity extraction blocked by firewall, regex only",
            exc_info=True,
        )
    except Exception:
        logger.warning(
            "LLM entity extraction failed, regex only",
            exc_info=True,
        )

    return entities


def _skill_classify_sensitivity(text: str) -> dict[str, Any]:
    """Classify text into a sensitivity tier using rules.

    Wraps the existing SensitivityClassifier from
    src/models/sensitivity_classifier.py.

    sensitivity_tier: 1 (classification metadata only)
    """
    from src.models.sensitivity_classifier import (
        SensitivityClassifier,
    )

    classifier = SensitivityClassifier()
    tier = classifier.classify(text)
    tier_labels = {1: "low", 2: "medium", 3: "high"}
    return {"tier": tier, "label": tier_labels[tier]}


# ---------------------------------------------------------------------------
# User-authored skills
# ---------------------------------------------------------------------------


_PLACEHOLDER_RE = re.compile(r"\{\{\s*(\w+)\s*\}\}")


def _render_template(template: str, kwargs: dict[str, Any]) -> str:
    """Substitute ``{{var}}`` placeholders in ``template``.

    Missing placeholders are left as the empty string so the LLM can
    still produce a useful response; we never fail the call over a
    missing optional argument.

    sensitivity_tier: 1
    """
    def repl(match: re.Match[str]) -> str:
        name = match.group(1)
        return str(kwargs.get(name, ""))

    return _PLACEHOLDER_RE.sub(repl, template)


def _make_user_skill_executor(
    template: str,
    *,
    uses_llm: bool,
    skill_id: str = "user-skill",
) -> Callable[..., Any]:
    """Build the ``execute_fn`` closure for a user-defined skill.

    sensitivity_tier: 1
    """
    def execute(**kwargs: Any) -> Any:
        prompt = _render_template(template, kwargs)
        if not uses_llm:
            return prompt
        try:
            resp = chat_via_firewalls(
                [{"role": "user", "content": prompt}],
                agent_id=f"skill.{skill_id}",
                lane=Lane.CLASSIFIER,
                agent_max_tier=2,
            )
            return resp.content
        except GatewayBlocked as exc:
            logger.info(
                "User skill %s blocked by firewall: %s", skill_id, exc,
            )
            return prompt
        except Exception as exc:  # noqa: BLE001
            logger.warning("User skill LLM call failed: %s", exc)
            return prompt

    return execute


def register_user_skills_from_db(
    registry: SkillRegistry,
    store: Any,
) -> int:
    """Mount every row of the user-skills SQLite table.

    Returns the count of skills registered. Idempotent — re-registering
    overrides the previous entry.

    ``store`` is duck-typed: it just needs a ``list_all()`` method
    returning objects with the same attribute surface as
    :class:`UserSkillRow` (so tests can pass a stub).

    sensitivity_tier: 1
    """
    rows = store.list_all()
    for row in rows:
        registry.register(Skill(
            id=row.skill_id,
            name=row.name,
            description=row.description,
            execute_fn=_make_user_skill_executor(
                row.prompt_template,
                uses_llm=bool(row.uses_llm),
                skill_id=row.skill_id,
            ),
            parameters=dict(row.parameters),
            category=row.category,
            uses_llm=bool(row.uses_llm),
        ))
    return len(rows)
