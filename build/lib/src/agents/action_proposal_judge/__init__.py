"""Independent judge that vets action proposals before the user sees them.

Run with a different LLM family from the primary extractor so a
hallucination in the primary is unlikely to be exactly mirrored by
the judge — a cheap structural defense against the
"user says A, agent proposes B" failure mode that plain prompts and
deterministic regex can't fully prevent.

sensitivity_tier: 2
"""

from src.agents.action_proposal_judge.agent import (
    ActionProposalJudge,
    JudgeDeps,
    judge_action_proposal,
    register_action_proposal_judge,
)

__all__ = [
    "ActionProposalJudge",
    "JudgeDeps",
    "judge_action_proposal",
    "register_action_proposal_judge",
]
