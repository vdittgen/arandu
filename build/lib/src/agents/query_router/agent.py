"""Pydantic AI query router.

Given a user question, produce a :class:`RetrievalPlan` selecting
which DuckDB tables (with structured columns + optional WHERE
clause), which ChromaDB collections, and whether to traverse the
Kuzu graph. The QueryEngine renders the plan into safe parameterised
SQL before execution.

Runs on the INTERACTIVE tier because every chat turn pays for it.

sensitivity_tier: 1
"""

from __future__ import annotations

import logging

from src.agents.core.agent_base import SBAgent
from src.agents.core.output_types import RetrievalPlan
from src.agents.core.scheduler import Tier
from src.models.prompt_cache import PROMPTS_DIR, FrozenPromptTemplate

logger = logging.getLogger(__name__)

_TEMPLATE = FrozenPromptTemplate(PROMPTS_DIR / "query_router_v1.txt")
DEFAULT_SYSTEM_PROMPT = _TEMPLATE.prefix


class QueryRouterAgent(SBAgent[str, RetrievalPlan]):
    """Decide which data sources to query for a user question.

    Deps is the user question string. Output is a :class:`RetrievalPlan`
    consumed by ``QueryEngine`` to dispatch parallel reads.

    sensitivity_tier: 1
    """

    agent_id = "query_router"
    output_type = RetrievalPlan
    tier = Tier.INTERACTIVE
    system_prompt = DEFAULT_SYSTEM_PROMPT

    def plan(self, question: str) -> RetrievalPlan | None:
        """Convenience wrapper returning the plan or None on failure.

        sensitivity_tier: 1
        """
        if not question or not question.strip():
            return None
        record = self.run(question)
        if record.output is None or record.error is not None:
            return None
        return record.output


def register_query_router_agent() -> None:
    """Register the query router in the global agent registry.

    sensitivity_tier: 1
    """
    from src.agents.core.config_store import AgentConfig
    from src.agents.core.registry import (
        AgentDefinition,
        get_agent,
        register_agent,
    )

    if get_agent("query_router") is not None:
        return

    default = AgentConfig(
        agent_id="query_router",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        model_route="inherit",
        model_override=None,
        enabled_tools=(),
        enabled_skills=(),
        editable=False,
    )
    register_agent(AgentDefinition(
        agent_id="query_router",
        name="Query Router",
        description=(
            "Decides which DuckDB tables, ChromaDB collections, and "
            "graph traversals to run for a user question. Called "
            "indirectly by Brain's recall_context tool."
        ),
        category="router",
        # Not a direct child of Brain — the QueryEngine invokes it
        # while servicing recall_context. We still parent it to brain
        # for UI grouping; the orchestrator does not delegate to it.
        parent_agent="brain",
        tier=Tier.INTERACTIVE,
        max_sensitivity_tier=2,
        editable=False,
        default_config=default,
        available_tools=(),
        available_skills=(),
        output_schema="RetrievalPlan",
        pattern="single",
        factory=QueryRouterAgent,
        tags=("router", "indirect"),
    ))


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "QueryRouterAgent",
    "register_query_router_agent",
]
