"""Chat agent package — conversational layer over Brain.

The chat agent runs its own model and exposes Brain + Brain's
sub-agents + user agents as tools. See :mod:`src.agents.chat.v1`.

sensitivity_tier: 3
"""

from src.agents.chat.v1 import (
    ChatAgent,
    ChatDeps,
    register_chat_agent,
)

__all__ = ["ChatAgent", "ChatDeps", "register_chat_agent"]
