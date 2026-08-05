"""Schema discovery as an SBAgent.

Migration target: ``src/extensions/ingestion/schema_discovery.py``.
Legacy keeps the rule-based first pass + confidence scoring; this
agent powers the LLM second pass that maps unknown fields to columns.

sensitivity_tier: 1
"""

from src.agents.schema_discovery.agent import (
    DEFAULT_SYSTEM_PROMPT,
    SchemaDiscoveryAgent,
    SchemaDiscoveryDeps,
    register_schema_discovery_agent,
)

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "SchemaDiscoveryAgent",
    "SchemaDiscoveryDeps",
    "register_schema_discovery_agent",
]
