"""Persistent placeholder registry for redact-then-remote egress.

When the egress firewall sends a Tier 2+ prompt to the remote provider
under the ``remote-default`` policy, every entity that the redactor
catches must map to a *stable* placeholder so the response can be
rehydrated cleanly and so subsequent prompts referencing the same
entity reuse the same token. Per-request maps lose that across
sessions; this registry persists the mapping to SQLite.

The hot path is:

1. The registry exposes a compiled alternation regex covering every
   raw value it knows about (rebuilt on insert). One pass over the
   outbound text replaces every known entity with its placeholder.
2. The redactor's existing regex catches the residual (novel emails,
   phone numbers, person-shaped tokens). New matches get
   ``assign``'d here so the next prompt's compiled regex covers them.

The registry file holds raw user values paired with their placeholders
— that is Tier 3 data. The file is created with 0600 mode and lives in
the same ``~/.arandu/data/`` directory as the rest of the local DB.

sensitivity_tier: 3
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


DEFAULT_PATH = (
    Path.home() / ".arandu" / "data" / "redaction_registry.sqlite"
)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS aliases (
    kind          TEXT NOT NULL,
    raw_value     TEXT NOT NULL,
    placeholder   TEXT NOT NULL,
    first_seen_at TEXT NOT NULL,
    PRIMARY KEY (kind, raw_value)
)
"""

_INDEX_PLACEHOLDER = (
    "CREATE UNIQUE INDEX IF NOT EXISTS aliases_placeholder "
    "ON aliases(placeholder)"
)
_INDEX_KIND = (
    "CREATE INDEX IF NOT EXISTS aliases_kind ON aliases(kind)"
)


class RedactionRegistry:
    """SQLite-backed mapping of raw entity → stable placeholder.

    Thread-safe — one connection guarded by a re-entrant lock. The DB
    file is created with 0600 mode so other macOS users can't read it.

    sensitivity_tier: 3
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or DEFAULT_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        existed = self._path.exists()
        self._conn = sqlite3.connect(
            self._path, isolation_level=None, check_same_thread=False,
        )
        self._conn.execute(_SCHEMA)
        self._conn.execute(_INDEX_PLACEHOLDER)
        self._conn.execute(_INDEX_KIND)
        if not existed:
            try:
                os.chmod(self._path, 0o600)
            except OSError:
                logger.warning(
                    "Could not chmod 0600 on redaction registry at %s",
                    self._path,
                )
        # Compiled alternation across every known raw value. Rebuilt
        # lazily on first read and on every successful insert.
        self._compiled: re.Pattern[str] | None = None
        self._value_to_placeholder: dict[str, str] = {}
        self._placeholder_to_value: dict[str, str] = {}
        self._kind_counters: dict[str, int] = {}
        self._load_cache()

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _load_cache(self) -> None:
        """Populate the in-memory cache + kind counters from SQLite.

        Migrates any legacy ``<KIND_N>`` placeholders to the current
        ``__KIND_N__`` form in-place. The format change is required
        because some LLMs treat angle brackets as XML markup and strip
        them in structured output, so the rehydration regex would
        never match them on return.

        sensitivity_tier: 3
        """
        with self._lock:
            self._value_to_placeholder.clear()
            self._placeholder_to_value.clear()
            self._kind_counters.clear()
            migrations: list[tuple[str, str]] = []
            cur = self._conn.execute(
                "SELECT kind, raw_value, placeholder FROM aliases",
            )
            for kind, raw, placeholder in cur.fetchall():
                if placeholder.startswith("<") and placeholder.endswith(">"):
                    new_placeholder = f"__{placeholder[1:-1]}__"
                    migrations.append((new_placeholder, placeholder))
                    placeholder = new_placeholder
                self._value_to_placeholder[raw] = placeholder
                self._placeholder_to_value[placeholder] = raw
                n = _index_from_placeholder(placeholder, kind)
                if n is not None and n > self._kind_counters.get(kind, 0):
                    self._kind_counters[kind] = n
            for new_p, old_p in migrations:
                self._conn.execute(
                    "UPDATE aliases SET placeholder = ? WHERE placeholder = ?",
                    (new_p, old_p),
                )
            if migrations:
                logger.info(
                    "Migrated %d legacy <KIND_N> placeholders to __KIND_N__",
                    len(migrations),
                )
            self._rebuild_pattern()

    def _rebuild_pattern(self) -> None:
        """Recompile the alternation over every known raw value.

        Sorted by descending length so the longest match wins (a name
        like ``"Bob Smith"`` is preferred over the bare ``"Bob"``).

        sensitivity_tier: 3
        """
        if not self._value_to_placeholder:
            self._compiled = None
            return
        # ``re.escape`` to avoid regex metacharacters embedded in user
        # values blowing up the compile. ``\b`` boundaries so we only
        # match whole tokens — substring collisions are out of scope
        # for the hot path.
        keys = sorted(
            self._value_to_placeholder, key=len, reverse=True,
        )
        alternation = "|".join(re.escape(k) for k in keys)
        self._compiled = re.compile(rf"(?<!\w)(?:{alternation})(?!\w)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assign(self, kind: str, raw: str) -> str:
        """Return the placeholder for ``raw``, allocating one if new.

        ``kind`` is the entity category (``"PERSON"``, ``"EMAIL"``,
        ``"PHONE"``, ``"MONEY"``, ``"DATE"``, ``"ID"``, ``"PLACE"``,
        ...). Placeholders look like ``__PERSON_3__``. The double-
        underscore wrapper survives LLM JSON output cleanly — earlier
        ``<KIND_N>`` form was stripped by models treating it as XML.

        sensitivity_tier: 3
        """
        if not raw:
            return raw
        with self._lock:
            existing = self._value_to_placeholder.get(raw)
            if existing is not None:
                return existing
            self._kind_counters[kind] = self._kind_counters.get(kind, 0) + 1
            placeholder = f"__{kind}_{self._kind_counters[kind]}__"
            try:
                self._conn.execute(
                    "INSERT INTO aliases "
                    "(kind, raw_value, placeholder, first_seen_at) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        kind,
                        raw,
                        placeholder,
                        datetime.now(tz=UTC).isoformat(),
                    ),
                )
            except sqlite3.IntegrityError:
                # Another caller inserted the same raw value concurrently;
                # reload the cache and return whatever placeholder won.
                self._load_cache()
                return self._value_to_placeholder.get(raw, placeholder)
            self._value_to_placeholder[raw] = placeholder
            self._placeholder_to_value[placeholder] = raw
            self._rebuild_pattern()
            return placeholder

    def lookup(self, raw: str) -> str | None:
        """Return the placeholder for ``raw`` if registered, else ``None``.

        sensitivity_tier: 3
        """
        with self._lock:
            return self._value_to_placeholder.get(raw)

    def known_pattern(self) -> re.Pattern[str] | None:
        """Compiled regex matching every registered raw value.

        ``None`` when the registry is empty.

        sensitivity_tier: 3
        """
        with self._lock:
            return self._compiled

    def reverse(self, placeholder: str) -> str | None:
        """Return the raw value for ``placeholder`` if registered.

        sensitivity_tier: 3
        """
        with self._lock:
            return self._placeholder_to_value.get(placeholder)

    def rehydrate(self, text: str) -> str:
        """Replace every known placeholder with its raw value.

        Matches three forms so LLM-mangled output still rehydrates:

        - Canonical ``__KIND_N__`` — the form we emit today.
        - Legacy ``<KIND_N>`` — pre-migration shape from older saves.
        - Bare ``KIND_N`` — what most LLMs emit after they decide
          the wrapping (``__`` or ``<>``) was markdown / XML markup
          and strip it from their JSON output. We require word
          boundaries so plain words like ``ID_1`` in legitimate
          content don't accidentally rehydrate.

        Placeholders that don't resolve pass through unchanged so a
        hallucinated token doesn't raise.

        sensitivity_tier: 3 (output may carry restored raw values)
        """
        with self._lock:
            if not self._placeholder_to_value:
                return text
            bare_to_raw: dict[str, str] = {}
            for placeholder, raw in self._placeholder_to_value.items():
                bare = _strip_placeholder_wrapping(placeholder)
                if bare:
                    bare_to_raw[bare] = raw

            alternatives: list[str] = []
            for placeholder in self._placeholder_to_value:
                alternatives.append(re.escape(placeholder))
            for placeholder in self._placeholder_to_value:
                if placeholder.startswith("__") and placeholder.endswith("__"):
                    legacy = f"<{placeholder[2:-2]}>"
                    alternatives.append(re.escape(legacy))
            for bare in bare_to_raw:
                alternatives.append(rf"\b{re.escape(bare)}\b")

            pattern = re.compile("|".join(alternatives))

            def _sub(match: re.Match[str]) -> str:
                token = match.group(0)
                # Canonical form wins.
                if token in self._placeholder_to_value:
                    return self._placeholder_to_value[token]
                # Legacy <KIND_N> → canonical __KIND_N__.
                if token.startswith("<") and token.endswith(">"):
                    canonical = f"__{token[1:-1]}__"
                    if canonical in self._placeholder_to_value:
                        return self._placeholder_to_value[canonical]
                # Bare KIND_N (stripped wrapping).
                if token in bare_to_raw:
                    return bare_to_raw[token]
                return token

            return pattern.sub(_sub, text)

    def bulk_register(
        self,
        items: Iterable[tuple[str, str]],
    ) -> int:
        """Register many ``(kind, raw_value)`` pairs at once.

        Existing rows are left alone. Returns the number of *new*
        registrations. Used by :meth:`bootstrap_from_graph` to
        pre-populate the registry without rebuilding the compiled
        pattern once per row.

        sensitivity_tier: 3
        """
        registered = 0
        with self._lock:
            for kind, raw in items:
                if not raw or raw in self._value_to_placeholder:
                    continue
                self._kind_counters[kind] = self._kind_counters.get(kind, 0) + 1
                placeholder = f"__{kind}_{self._kind_counters[kind]}__"
                try:
                    self._conn.execute(
                        "INSERT INTO aliases "
                        "(kind, raw_value, placeholder, first_seen_at) "
                        "VALUES (?, ?, ?, ?)",
                        (
                            kind,
                            raw,
                            placeholder,
                            datetime.now(tz=UTC).isoformat(),
                        ),
                    )
                except sqlite3.IntegrityError:
                    continue
                self._value_to_placeholder[raw] = placeholder
                self._placeholder_to_value[placeholder] = raw
                registered += 1
            if registered:
                self._rebuild_pattern()
        return registered

    def bootstrap_from_graph(self, kuzu_engine: object) -> int:
        """Pre-populate the registry from the Kuzu knowledge graph.

        Reads ``(:Person {name})`` and ``(:Place {name})`` (the
        repository's analogues of "Person / Organization / Location" —
        Place doubles as both locations and named venues). Each name
        gets registered with the matching ``kind``. Cheap to call
        repeatedly: every insert is ``INSERT OR IGNORE`` semantics
        via the duplicate-key handling in :meth:`bulk_register`.

        sensitivity_tier: 3 (the graph holds Tier 2/3 entity surface)
        """
        items: list[tuple[str, str]] = []
        for label, kind in (("Person", "PERSON"), ("Place", "PLACE")):
            try:
                rows = kuzu_engine.query(  # type: ignore[attr-defined]
                    f"MATCH (n:{label}) RETURN n.name AS name",
                )
            except Exception:  # noqa: BLE001
                logger.debug(
                    "bootstrap_from_graph: %s lookup failed",
                    label, exc_info=True,
                )
                continue
            for row in rows or []:
                name = (row.get("name") if isinstance(row, dict)
                        else None)
                if isinstance(name, str) and name.strip():
                    items.append((kind, name.strip()))
        return self.bulk_register(items)

    def close(self) -> None:
        """sensitivity_tier: 3"""
        with self._lock:
            self._conn.close()


def _strip_placeholder_wrapping(placeholder: str) -> str | None:
    """Return the ``KIND_N`` form of ``placeholder`` if recognised.

    Returns ``None`` for tokens that aren't shaped like a placeholder
    so we don't accidentally register them as rehydration targets.

    sensitivity_tier: 1
    """
    if placeholder.startswith("__") and placeholder.endswith("__"):
        return placeholder[2:-2]
    if placeholder.startswith("<") and placeholder.endswith(">"):
        return placeholder[1:-1]
    return None


def _index_from_placeholder(placeholder: str, kind: str) -> int | None:
    """Extract N from ``__KIND_N__``. Returns ``None`` on mismatch.

    sensitivity_tier: 1
    """
    prefix = f"__{kind}_"
    suffix = "__"
    if not placeholder.startswith(prefix) or not placeholder.endswith(suffix):
        return None
    middle = placeholder[len(prefix):-len(suffix)]
    try:
        return int(middle)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Process-wide default
# ---------------------------------------------------------------------------

_default_registry: RedactionRegistry | None = None
_default_lock = threading.Lock()


def default_redaction_registry() -> RedactionRegistry:
    """Return the process-wide :class:`RedactionRegistry`.

    sensitivity_tier: 3
    """
    global _default_registry
    if _default_registry is None:
        with _default_lock:
            if _default_registry is None:
                _default_registry = RedactionRegistry()
    return _default_registry


def reset_redaction_registry_for_tests(
    *, path: Path | None = None,
) -> RedactionRegistry:
    """Re-create the process-wide registry — test isolation only.

    sensitivity_tier: 3
    """
    global _default_registry
    with _default_lock:
        if _default_registry is not None:
            _default_registry.close()
        _default_registry = RedactionRegistry(path=path)
    return _default_registry


__all__ = [
    "DEFAULT_PATH",
    "RedactionRegistry",
    "default_redaction_registry",
    "reset_redaction_registry_for_tests",
]
