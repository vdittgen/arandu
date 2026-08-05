"""Dataset validator agent.

Reviews a user-uploaded eval dataset YAML, checks its structure
against the shared :mod:`evals.evaluators` catalog, and proposes
improvements. The CLI wraps this call with an :class:`InjectionFirewall`
scan before persisting the dataset to disk.

sensitivity_tier: 1
"""

from src.agents.dataset_validator.agent import (
    DEFAULT_SYSTEM_PROMPT,
    DatasetValidatorAgent,
    canonicalize_dataset_yaml,
    register_dataset_validator_agent,
    structural_check,
)

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "DatasetValidatorAgent",
    "canonicalize_dataset_yaml",
    "register_dataset_validator_agent",
    "structural_check",
]
