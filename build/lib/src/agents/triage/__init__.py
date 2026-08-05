"""Message triage — SBAgent + DB cache.

:class:`TriageAgent` is the pure LLM keep/drop primitive;
:class:`MessageTriager` (re-exported from :mod:`.persistence`) wraps it
with the ``_triage_log`` cache so repeated runs over the same message
ids don't re-pay the LLM. Both names live here so callers can keep
doing ``from src.agents.triage import MessageTriager`` after the
Phase E relocation from ``src/agents/message_triage.py``.

sensitivity_tier: 3 (sees raw message content)
"""

from src.agents.triage.agent import (
    DEFAULT_SYSTEM_PROMPT,
    TriageAgent,
    TriageMessage,
    register_triage_agent,
)
from src.agents.triage.persistence import (
    MessageTriager,
    TriageDecision,
)

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "MessageTriager",
    "TriageAgent",
    "TriageDecision",
    "TriageMessage",
    "register_triage_agent",
]
