"""Topic extractor as an SBAgent.

Migration target: the per-contact LLM step inside
``src/pipeline/intermediate/int_contact_topics.py``. The SQLMesh
pipeline keeps its message-batching, caching, and persistence logic.
This module supplies the LLM primitive — given a contact's recent
messages, return up to 5 ongoing :class:`Topic`s.

sensitivity_tier: 2
"""

from src.agents.topic_extractor.agent import (
    DEFAULT_SYSTEM_PROMPT,
    TopicExtractorAgent,
    TopicExtractorDeps,
    register_topic_extractor_agent,
)

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "TopicExtractorAgent",
    "TopicExtractorDeps",
    "register_topic_extractor_agent",
]
