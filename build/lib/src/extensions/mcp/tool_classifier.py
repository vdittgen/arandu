"""MCP tool type classifier — DATA vs ACTION.

Classifies MCP tools as producing queryable data rows or performing
side-effect-only actions, using name patterns and description heuristics.

sensitivity_tier: 1 (analyzes tool metadata, no user data)
"""

from __future__ import annotations

from src.extensions.mcp.client import McpToolInfo

# ---------------------------------------------------------------------------
# Classification rules
# ---------------------------------------------------------------------------

_DATA_PREFIXES: tuple[str, ...] = (
    "list_", "get_", "search_", "query_", "fetch_", "read_",
)

_ACTION_PREFIXES: tuple[str, ...] = (
    "create_", "update_", "delete_", "send_", "set_", "post_",
    "remove_", "add_", "write_", "put_",
)

_DATA_DESCRIPTION_KEYWORDS: frozenset[str] = frozenset({
    "returns", "retrieves", "lists", "gets", "fetches",
    "searches", "queries", "reads", "provides", "shows",
})

_ACTION_DESCRIPTION_KEYWORDS: frozenset[str] = frozenset({
    "creates", "sends", "updates", "deletes", "removes",
    "writes", "modifies", "posts", "sets", "triggers",
})

_FILTER_PARAM_NAMES: frozenset[str] = frozenset({
    "limit", "offset", "count", "page", "query", "filter",
    "search", "sort", "order", "cursor", "after", "before",
    "start", "end", "from", "to", "since", "until",
})


def classify_tool(tool: McpToolInfo) -> str:
    """Classify an MCP tool as 'data' or 'action'.

    Rules are applied in priority order. Default is 'data' (conservative:
    better to discover schema than miss it).

    Args:
        tool: MCP tool info with name, description, and input schema.

    Returns:
        'data' or 'action'.

    sensitivity_tier: 1
    """
    name_lower = tool.name.lower()

    # Rule 1: Name prefix matching
    if any(name_lower.startswith(p) for p in _DATA_PREFIXES):
        return "data"
    if any(name_lower.startswith(p) for p in _ACTION_PREFIXES):
        return "action"

    # Rule 2: Description keyword matching
    desc_lower = tool.description.lower()
    desc_words = set(desc_lower.split())

    if desc_words & _DATA_DESCRIPTION_KEYWORDS:
        return "data"
    if desc_words & _ACTION_DESCRIPTION_KEYWORDS:
        return "action"

    # Rule 3: Input schema analysis — tools with no required params
    # or only filter-like params are likely data tools
    schema = tool.input_schema
    required = set(schema.get("required", []))
    if not required:
        return "data"

    properties = schema.get("properties", {})
    non_filter_required = required - _FILTER_PARAM_NAMES
    all_props_are_filters = all(
        p.lower() in _FILTER_PARAM_NAMES for p in properties
    )
    if not non_filter_required or all_props_are_filters:
        return "data"

    # Default: data (conservative)
    return "data"
