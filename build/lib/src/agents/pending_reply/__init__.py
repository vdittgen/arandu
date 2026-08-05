"""Pending-reply detector as an SBAgent.

Migration target: the pending-reply classification step inside
``src/agents/proactive_intelligence.py``. The legacy orchestrator
keeps its DB scans (messages last 7d, contact join, dedup) and
notification persistence; this module provides the LLM step that
decides which messages actually need a user reply.

sensitivity_tier: 2
"""

from src.agents.pending_reply.agent import (
    DEFAULT_SYSTEM_PROMPT,
    PendingReplyAgent,
    PendingReplyDeps,
    register_pending_reply_agent,
)

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "PendingReplyAgent",
    "PendingReplyDeps",
    "register_pending_reply_agent",
]
