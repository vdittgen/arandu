"""Frozen prompt templates for provider prompt caches.

A :class:`FrozenPromptTemplate` loads a byte-stable prefix from a
checked-in ``.txt`` file. No runtime substitution is allowed in the
prefix — *that is the whole point*. A single edit to a system prompt
that adds a trailing space silently invalidates every provider cache
entry and turns a $14/mo bill into $170/mo.

Variable content (user message, parameters, context windows) is
appended *after* the frozen prefix via :meth:`render`. The prefix
SHA-256 is exposed for two purposes:

1. The pre-commit golden test
   (:mod:`tests.unit.models.prompts.test_golden_prompts`) compares
   ``prefix_hash`` against a checked-in literal — any edit fails
   loudly until the literal is updated alongside the prompt.
2. Per-call observability — :class:`CacheHitMonitor` records the
   ratio of cached input tokens, which is the operational signal
   that a cache invalidation happened.

The class is small and deliberately stateless past construction.

sensitivity_tier: 1
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"


class FrozenPromptTemplate:
    """sensitivity_tier: 1"""

    def __init__(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"prompt template not found: {path}")
        self._path = path
        # The on-disk bytes ARE the template. No normalization, no
        # trailing-newline trimming. If the file has a trailing newline
        # so does the prefix; that's the user's intent.
        self._prefix = path.read_text(encoding="utf-8")
        self._prefix_hash_hex = hashlib.sha256(
            self._prefix.encode("utf-8"),
        ).hexdigest()

    @property
    def path(self) -> Path:
        """sensitivity_tier: 1"""
        return self._path

    @property
    def prefix(self) -> str:
        """The frozen system-prompt text exactly as it sits on disk.

        sensitivity_tier: 1
        """
        return self._prefix

    @property
    def prefix_hash(self) -> str:
        """sha256 hex digest of the prefix, prefixed with ``sha256:``.

        Used as a ``prompt_cache_key`` segment and as the constant
        compared in the golden test.

        sensitivity_tier: 1
        """
        return f"sha256:{self._prefix_hash_hex}"

    def render(
        self,
        variable_user_msg: str = "",
        *,
        system_role: str = "system",
        user_role: str = "user",
    ) -> list[dict[str, str]]:
        """Return chat-completion messages: frozen prefix + variable user.

        The variable string is appended verbatim; callers are
        responsible for formatting any per-call interpolation (schema
        JSON, retrieved context, user input, today's date) into that
        single string. Anything passed here is OUTSIDE the cached
        portion.

        Pass ``variable_user_msg=""`` for purely static prompts where
        the provider receives only the system message.

        sensitivity_tier: varies (depends on what the caller passes)
        """
        msgs: list[dict[str, str]] = [
            {"role": system_role, "content": self._prefix},
        ]
        if variable_user_msg != "":
            msgs.append({"role": user_role, "content": variable_user_msg})
        return msgs

    def render_inline(
        self, variable_suffix: str = "",
    ) -> list[dict[str, str]]:
        """Single user message: prefix concatenated with variable suffix.

        Use this when migrating a caller that originally built one
        combined string via ``template.format(...)``. The wire format
        stays identical (one ``user`` message), the static prefix is
        cacheable, and the LLM sees the same byte sequence as before.

        For new code prefer :meth:`render` (system + user split).

        sensitivity_tier: varies
        """
        return [
            {"role": "user", "content": self._prefix + variable_suffix},
        ]

    def render_combined(self, variables: Mapping[str, str] | None = None) -> str:
        """Concatenated form used by golden-file regression tests.

        Joins each message as ``f"{role}: {content}"`` separated by a
        fixed delimiter. ``variables`` (if any) is appended as a sorted
        ``key=value`` block on the user side so the golden file is
        stable across Python dict-order changes.

        Production code should call :meth:`render`, not this method.

        sensitivity_tier: 1
        """
        rendered = self.render(_format_variables(variables))
        return "\n---\n".join(
            f"{m['role']}: {m['content']}" for m in rendered
        )


def _format_variables(variables: Mapping[str, str] | None) -> str:
    if not variables:
        return ""
    return "\n".join(
        f"{key}={variables[key]}" for key in sorted(variables)
    )


__all__ = [
    "PROMPTS_DIR",
    "FrozenPromptTemplate",
]
