"""Query router as an SBAgent.

Migration target: ``LLMRouter`` inside ``src/core/query_engine.py``.
The QueryEngine keeps its rule-based fast path; the LLM fallback is
now an editable ``SBAgent`` that emits a typed :class:`RetrievalPlan`.

sensitivity_tier: 1
"""

from src.agents.query_router.agent import (
    DEFAULT_SYSTEM_PROMPT,
    QueryRouterAgent,
    register_query_router_agent,
)

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "QueryRouterAgent",
    "register_query_router_agent",
]
