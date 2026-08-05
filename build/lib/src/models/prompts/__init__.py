"""Frozen prompt templates loaded by :class:`FrozenPromptTemplate`.

One ``.txt`` file per cached system prompt. Edits to any file in
this directory MUST be paired with an update to the matching hash
in :mod:`tests.unit.models.prompts.test_golden_prompts` — the
pre-commit script ``scripts/check_prompt_hashes.sh`` enforces this
locally.

sensitivity_tier: 1
"""
