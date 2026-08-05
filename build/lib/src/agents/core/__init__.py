"""Agent core — Pydantic AI base classes, scheduler, firewall hooks.

This package is the foundation of the agentic refactor. It introduces:

- ``SBAgent`` — single-workflow agent (most sub-agents)
- ``SBOrchestrator`` — delegates to sub-agents via tools (Brain, Proactive,
  Reply Handler)
- ``SBDeepAgent`` — autonomous planning + sandboxed file/code ops
- ``LLMScheduler`` — tier-based admission control and tier-aware routing
- ``ModelFactory`` — builds pydantic-ai models from user settings
- ``InjectionFirewall`` / ``EgressFirewall`` — non-editable prompt guards
- ``AgentRegistry`` — runtime catalog of registered agents
- ``AgentConfigStore`` — SQLite-backed editable overrides
- ``AuditChain`` — append-only SHA-256-chained decision log

No public symbols are auto-imported at package level — call sites should
import from the submodule they need to keep startup lean.

sensitivity_tier: N/A (infrastructure)
"""
