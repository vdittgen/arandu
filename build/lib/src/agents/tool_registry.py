"""Tool registry — discovers MCP action tools from enabled connectors.

Bridges the extension system and the Brain Agent by providing a catalog
of available action tools and rule-based intent matching.

sensitivity_tier: 1 (reads connector metadata, no user data)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from src.extensions.connectors.catalog import ConnectorCatalog
from src.extensions.connectors.registry import ExtensionRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Action verb keywords for intent matching
# ---------------------------------------------------------------------------

_ACTION_VERBS: frozenset[str] = frozenset({
    # English
    "send", "create", "schedule", "delete", "add", "write",
    "play", "set", "update", "remove", "post", "draft",
    "compose", "make", "put", "cancel", "start", "stop",
    "search", "find", "reply", "respond", "flag", "move",
    "edit", "modify", "forward", "trash",
    # Portuguese (imperative + infinitive)
    "crie", "criar", "envie", "enviar", "mande", "mandar",
    "agende", "agendar", "marque", "marcar",
    "apague", "apagar", "remova", "remover", "deletar",
    "adicione", "adicionar", "escreva", "escrever",
    "toque", "tocar", "defina", "definir", "configure", "configurar",
    "atualize", "atualizar", "cancele", "cancelar",
    "faça", "fazer",
    "busque", "buscar", "procure", "procurar",
    "encontre", "encontrar",
    "responda", "responder",
    "mova", "mover",
    "sinalize", "sinalizar",
    "edite", "editar", "altere", "alterar",
    "modifique", "modificar",
    "encaminhe", "encaminhar",
    "exclua", "excluir",
})

# Map action verbs to tool name prefixes for scoring
_VERB_TO_PREFIX: dict[str, tuple[str, ...]] = {
    "send": ("send_",),
    "create": ("create_",),
    "schedule": ("create_event", "create_reminder"),
    "delete": ("delete_", "remove_"),
    "add": ("create_", "add_"),
    "write": ("write_", "create_note", "create_draft"),
    "play": ("play_", "get_current_playback"),
    "set": ("set_", "update_"),
    "update": ("update_", "set_"),
    "remove": ("remove_", "delete_"),
    "post": ("post_", "send_"),
    "draft": ("create_draft",),
    "compose": ("create_draft", "send_email"),
    "make": ("create_",),
    "cancel": ("delete_", "remove_"),
    "start": ("play_", "create_"),
    "stop": ("set_",),
    "search": ("search_",),
    "find": ("search_",),
    # Channels without a dedicated reply_* tool (WhatsApp, iMessage, etc.)
    # use send_message; the channel hint downstream picks the right one.
    "reply": ("reply_", "send_message"),
    "respond": ("reply_", "send_message"),
    "flag": ("flag_",),
    "move": ("move_",),
    "edit": ("update_",),
    "modify": ("update_",),
    "forward": ("send_email",),
    "trash": ("delete_email", "delete_"),
    # Portuguese → same tool prefixes as English equivalents
    "crie": ("create_",),
    "criar": ("create_",),
    "envie": ("send_",),
    "enviar": ("send_",),
    "mande": ("send_",),
    "mandar": ("send_",),
    "agende": ("create_event", "create_reminder"),
    "agendar": ("create_event", "create_reminder"),
    "marque": ("create_event", "create_reminder"),
    "marcar": ("create_event", "create_reminder"),
    "apague": ("delete_", "remove_"),
    "apagar": ("delete_", "remove_"),
    "remova": ("remove_", "delete_"),
    "remover": ("remove_", "delete_"),
    "deletar": ("delete_", "remove_"),
    "adicione": ("create_", "add_"),
    "adicionar": ("create_", "add_"),
    "escreva": ("write_", "create_note", "create_draft"),
    "escrever": ("write_", "create_note", "create_draft"),
    "toque": ("play_", "get_current_playback"),
    "tocar": ("play_", "get_current_playback"),
    "defina": ("set_", "update_"),
    "definir": ("set_", "update_"),
    "configure": ("set_", "update_"),
    "configurar": ("set_", "update_"),
    "atualize": ("update_", "set_"),
    "atualizar": ("update_", "set_"),
    "cancele": ("delete_", "remove_"),
    "cancelar": ("delete_", "remove_"),
    "faça": ("create_",),
    "fazer": ("create_",),
    "busque": ("search_",),
    "buscar": ("search_",),
    "procure": ("search_",),
    "procurar": ("search_",),
    "encontre": ("search_",),
    "encontrar": ("search_",),
    "responda": ("reply_", "send_message"),
    "responder": ("reply_", "send_message"),
    "mova": ("move_",),
    "mover": ("move_",),
    "sinalize": ("flag_",),
    "sinalizar": ("flag_",),
    "edite": ("update_",),
    "editar": ("update_",),
    "altere": ("update_",),
    "alterar": ("update_",),
    "modifique": ("update_",),
    "modificar": ("update_",),
    "encaminhe": ("send_email",),
    "encaminhar": ("send_email",),
    "exclua": ("delete_", "remove_"),
    "excluir": ("delete_", "remove_"),
}

# Nouns that help narrow tool matching
_NOUN_TO_TOOLS: dict[str, tuple[str, ...]] = {
    "message": ("send_message",),
    "email": (
        "send_email", "reply_email", "delete_email",
        "move_email", "flag_email", "search_emails", "create_draft",
    ),
    "event": ("create_event", "delete_event"),
    "calendar": ("create_event", "delete_event"),
    "meeting": ("create_event", "delete_event"),
    "reminder": ("create_reminder", "delete_reminder"),
    "note": ("create_note", "search_notes", "update_note", "delete_note"),
    "file": ("write_file",),
    "song": ("play_track",),
    "music": ("play_track", "get_current_playback"),
    "track": ("play_track",),
    "contact": ("search_contacts",),
    "inbox": ("search_emails",),
    "mailbox": ("move_email",),
    "trash": ("delete_email",),
    # Portuguese nouns
    "mensagem": ("send_message",),
    "evento": ("create_event", "delete_event"),
    "reunião": ("create_event", "delete_event"),
    "calendário": ("create_event", "delete_event"),
    "agenda": ("create_event", "delete_event"),
    "lembrete": ("create_reminder", "delete_reminder"),
    "nota": ("create_note", "search_notes", "update_note", "delete_note"),
    "arquivo": ("write_file",),
    "música": ("play_track",),
    "canção": ("play_track",),
    "contato": ("search_contacts",),
    "tarefa": ("create_reminder", "delete_reminder"),
    "lixeira": ("delete_email",),
    "caixa": ("search_emails", "move_email"),
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


_QUESTION_STARTERS: tuple[str, ...] = (
    # English
    "what", "when", "where", "how", "who", "which",
    "is", "are", "do", "does", "did", "was", "were",
    "will", "would", "should", "has", "have", "had",
    "tell me about", "show me", "list",
    # Portuguese
    "qual", "quais", "quando", "onde", "como", "quem",
    "o que", "me diga", "me mostre", "liste",
)

_REQUEST_PATTERNS: tuple[str, ...] = (
    # English
    "can you", "could you", "please", "i want to", "i need to",
    "i'd like to", "i would like to", "go ahead and",
    "help me",
    # Portuguese
    "você pode", "poderia", "por favor", "eu quero",
    "eu preciso", "eu gostaria", "me ajude",
)

def _is_pure_question(text: str, verbs: set[str]) -> bool:
    """Detect if text is a question using an action verb as a noun.

    Returns True when the text is purely interrogative (e.g. "What's
    on my schedule?") and the action verb appears to be used as a noun
    rather than an imperative.

    Returns False for request-style questions like "Can you send a
    message?" which should still be treated as actions.

    sensitivity_tier: 1
    """
    stripped = text.strip().rstrip("?!.,")

    # Check for explicit request patterns (always treat as action)
    for pattern in _REQUEST_PATTERNS:
        if pattern in stripped:
            return False

    # Check if text starts with a question word
    first_word = stripped.split()[0] if stripped.split() else ""
    starts_with_question = any(
        stripped.startswith(q) for q in _QUESTION_STARTERS
    )

    if not starts_with_question and "?" not in text:
        return False

    # It looks like a question. Now check if any matched verb appears
    # as the first word (imperative) — if so, it's not a pure question.
    if first_word in verbs:
        return False

    return True


@dataclass(frozen=True)
class ActionTool:
    """An available action tool from an enabled connector.

    sensitivity_tier: 1
    """

    connector_id: str
    connector_name: str
    tool_name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    display_name: str = ""
    # ``"action"`` for LLM-callable tools, ``"data"`` for poller tools
    # that populate a target table. Defaults to ``"action"`` so
    # existing call-sites that consume :class:`ActionTool` keep working.
    tool_type: str = "action"
    # Set on ``data`` tools; the name of the SQLite table the poller
    # writes to (e.g. ``"raw_emails"``). ``None`` for action tools.
    target_table: str | None = None

    def __post_init__(self) -> None:
        """Generate display_name if not provided.

        sensitivity_tier: 1
        """
        if not self.display_name:
            name = self.tool_name.replace("_", " ").title()
            object.__setattr__(self, "display_name", name)


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """Discovers action tools from enabled connectors.

    Reads from the ConnectorCatalog and ExtensionRegistry — no MCP
    connections needed. Provides intent matching for the Brain Agent.

    sensitivity_tier: 1
    """

    def __init__(
        self,
        catalog: ConnectorCatalog,
        registry: ExtensionRegistry,
    ) -> None:
        """Initialise the tool registry.

        Args:
            catalog: Bundled connector catalog.
            registry: Extension registry with enabled state.

        sensitivity_tier: 1
        """
        self._catalog = catalog
        self._registry = registry

    def get_available_actions(self) -> list[ActionTool]:
        """Return all action tools from enabled connectors.

        Iterates enabled connectors, looks up each in the catalog,
        and filters for tools with ``tool_type == "action"``.

        sensitivity_tier: 1
        """
        actions: list[ActionTool] = []

        for ext in self._registry.get_enabled():
            template = self._catalog.get(ext.connector_id)
            if template is None:
                continue

            for tool in template.tools:
                if tool.tool_type != "action":
                    continue

                actions.append(ActionTool(
                    connector_id=template.id,
                    connector_name=template.name,
                    tool_name=tool.tool_name,
                    description=template.description,
                    input_schema=tool.input_schema,
                    tool_type="action",
                ))

        logger.debug(
            "Found %d action tools from %d enabled connectors",
            len(actions),
            len(self._registry.get_enabled()),
        )
        return actions

    def get_available_tools(self) -> list[ActionTool]:
        """Return BOTH data and action tools from enabled connectors.

        Used by the user-agent picker so the UI can render a single
        per-connector card with sources (data tools) and callables
        (action tools) side by side. Data tools carry their
        ``target_table`` so the runner can map a selected source to
        the SQLite table without consulting a second source of truth.

        sensitivity_tier: 1
        """
        tools: list[ActionTool] = []

        for ext in self._registry.get_enabled():
            template = self._catalog.get(ext.connector_id)
            if template is None:
                continue

            for tool in template.tools:
                if tool.tool_type not in ("action", "data"):
                    continue
                target_table = (
                    getattr(tool, "target_table", None)
                    if tool.tool_type == "data"
                    else None
                )
                tools.append(ActionTool(
                    connector_id=template.id,
                    connector_name=template.name,
                    tool_name=tool.tool_name,
                    description=template.description,
                    input_schema=tool.input_schema,
                    tool_type=tool.tool_type,
                    target_table=target_table,
                ))

        logger.debug(
            "Found %d tools (data+action) from %d enabled connectors",
            len(tools),
            len(self._registry.get_enabled()),
        )
        return tools

    def get_action(
        self,
        connector_id: str,
        tool_name: str,
    ) -> ActionTool | None:
        """Look up a specific action tool by connector and tool name.

        sensitivity_tier: 1
        """
        template = self._catalog.get(connector_id)
        if template is None:
            return None

        for tool in template.tools:
            if tool.tool_name == tool_name and tool.tool_type == "action":
                return ActionTool(
                    connector_id=template.id,
                    connector_name=template.name,
                    tool_name=tool.tool_name,
                    description=template.description,
                )
        return None

    def match_intent(self, user_text: str) -> list[ActionTool]:
        """Match user text against available action tools.

        Uses rule-based keyword matching:
        1. Extract action verbs from user text
        2. Extract object nouns from user text
        3. Score each available tool against verb+noun matches
        4. Return ranked matches (best first)

        Returns an empty list for non-action queries.

        sensitivity_tier: 1
        """
        if not user_text or not user_text.strip():
            return []

        available = self.get_available_actions()
        if not available:
            return []

        text_lower = user_text.lower()
        words = set(re.findall(r"[a-zà-ÿ]+", text_lower))

        # Find action verbs in the text
        matched_verbs = words & _ACTION_VERBS
        if not matched_verbs:
            return []

        # Filter out false positives from question context.
        # "What's on my schedule?" uses "schedule" as a noun.
        # But "Can you send a message?" is a valid action request.
        if _is_pure_question(text_lower, matched_verbs):
            return []

        # Score each tool
        scored: list[tuple[float, ActionTool]] = []
        for tool in available:
            score = self._score_tool(tool, matched_verbs, words)
            if score > 0:
                scored.append((score, tool))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        return [tool for _, tool in scored]

    @staticmethod
    def _score_tool(
        tool: ActionTool,
        verbs: set[str],
        words: set[str],
    ) -> float:
        """Score how well a tool matches the detected verbs and nouns.

        sensitivity_tier: 1
        """
        score = 0.0
        tool_lower = tool.tool_name.lower()

        # Verb → prefix matching
        for verb in verbs:
            prefixes = _VERB_TO_PREFIX.get(verb, ())
            for prefix in prefixes:
                if tool_lower.startswith(prefix) or tool_lower == prefix:
                    score += 2.0
                    break

        # Noun → tool name matching
        for noun, tool_names in _NOUN_TO_TOOLS.items():
            if noun in words:
                if tool_lower in tool_names:
                    score += 3.0

        # Partial name match (tool name words appear in user text)
        tool_words = set(tool_lower.split("_"))
        overlap = tool_words & words
        score += len(overlap) * 0.5

        return score
