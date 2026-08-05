"""Message evaluator — SBAgent + DB persistence.

:class:`MessageEvaluatorAgent` is the LLM primitive (given an
assembled context, returns a typed :class:`MessageNotificationBatch`
of drafts). :class:`MessageEvaluator` (re-exported from
:mod:`.persistence`) owns DB scans, topic loading, today's events
lookup, and notification persistence (``_evaluated_messages``,
``_message_notifications``, ``_topics``).

sensitivity_tier: 3
"""

from src.agents.message_eval.agent import (
    DEFAULT_SYSTEM_PROMPT,
    MessageEvalDeps,
    MessageEvaluatorAgent,
    register_message_evaluator_agent,
)
from src.agents.message_eval.persistence import (
    MESSAGE_CONNECTORS,
    MessageEvaluator,
    MessageNotification,
    format_realtime_notification,
)

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "MESSAGE_CONNECTORS",
    "MessageEvalDeps",
    "MessageEvaluator",
    "MessageEvaluatorAgent",
    "MessageNotification",
    "format_realtime_notification",
    "register_message_evaluator_agent",
]
