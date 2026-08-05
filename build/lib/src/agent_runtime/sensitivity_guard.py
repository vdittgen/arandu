"""Python-side sensitivity enforcement for sandboxed agents.

The classification + access-control logic used to live in Rust
(``src-tauri/src/firewall/classifier.rs``); that scaffolding was
retired in ``refactor/firewall-consent-cleanup`` and this module is
now the sole enforcement point for sandboxed agent subprocesses.

Enforces:
- Field-level sensitivity classification
- Table access control based on manifest permissions
- ``WHERE sensitivity_tier <= N`` injection on all queries
- Write isolation to ``ext_{agent_id}_*`` tables only
- Audit logging of all data access decisions

sensitivity_tier: N/A (enforcement infrastructure)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.agent_runtime.models import AgentManifest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Field → tier mapping (mirrors Rust classifier.rs)
# Unknown fields default to tier 3 (fail-safe).
# ---------------------------------------------------------------------------

FIELD_TIERS: dict[str, int] = {
    # Tier 1 — general / structural
    "id": 1,
    "source": 1,
    "created_at": 1,
    "updated_at": 1,
    "metric_type": 1,
    "unit": 1,
    "filepath": 1,
    "filename": 1,
    "filetype": 1,
    "size_bytes": 1,
    "modified_at": 1,
    "tags": 1,
    "title": 1,
    "category": 1,
    "sensitivity_tier": 1,
    "_loaded_at": 1,
    # Tier 2 — personal
    "name": 2,
    "email": 2,
    "phone": 2,
    "sender": 2,
    "recipient": 2,
    "relationship": 2,
    "location": 2,
    "attendees": 2,
    "start_time": 2,
    "end_time": 2,
    "last_contact": 2,
    "description": 2,
    "notes": 2,
    "recorded_at": 2,
    "timestamp": 2,
    # Tier 3 — sensitive
    "content": 3,
    "content_preview": 3,
    "value": 3,
    "metadata": 3,
}

DEFAULT_TIER = 3

# Tables that agents can read (subject to manifest permissions).
KNOWN_TABLES: frozenset[str] = frozenset({
    "raw_messages",
    "raw_calendar_events",
    "raw_notes",
    "raw_health_metrics",
    "raw_contacts",
    "raw_files",
})

# SQL keywords that indicate destructive operations.
_DESTRUCTIVE_KEYWORDS = re.compile(
    r"\b(DROP|ALTER|TRUNCATE|CREATE)\b",
    re.IGNORECASE,
)

# Regex to extract table names from FROM and JOIN clauses.
_TABLE_PATTERN = re.compile(
    r"\bFROM\s+(\w+)"
    r"|\bJOIN\s+(\w+)",
    re.IGNORECASE,
)

# Detect existing WHERE clause.
_WHERE_PATTERN = re.compile(r"\bWHERE\b", re.IGNORECASE)

# Detect write operations on non-ext tables.
_WRITE_OPS = re.compile(
    r"\b(INSERT\s+INTO|UPDATE|DELETE\s+FROM)\s+(\w+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AccessDecision:
    """Result of a sensitivity guard check.

    sensitivity_tier: 1
    """

    allowed: bool
    reason: str
    effective_tier: int


class SensitivityGuard:
    """Python-side sensitivity enforcement for agent subprocesses.

    sensitivity_tier: N/A
    """

    def __init__(
        self,
        agent_id: str,
        manifest: AgentManifest,
        audit_path: Path | None = None,
    ) -> None:
        self._agent_id = agent_id
        self._manifest = manifest
        self._permitted_tables: dict[str, int] = {
            tp.table: tp.max_tier for tp in manifest.tables
        }
        self._audit_path = audit_path or (
            Path.home()
            / ".arandu"
            / "data"
            / "agents"
            / agent_id
            / "audit.jsonl"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify_fields(self, fields: list[str]) -> int:
        """Determine max sensitivity tier across requested fields.

        Unknown fields default to tier 3 (matches Rust classifier).
        Empty field list returns tier 1.

        sensitivity_tier: N/A
        """
        if not fields:
            return 1
        return max(FIELD_TIERS.get(f, DEFAULT_TIER) for f in fields)

    def check_table_access(self, table: str) -> AccessDecision:
        """Check if the agent's manifest allows access to this table.

        sensitivity_tier: N/A
        """
        if table.startswith("ext_"):
            return AccessDecision(
                allowed=True,
                reason="Extension table access allowed",
                effective_tier=1,
            )
        if table not in self._permitted_tables:
            return AccessDecision(
                allowed=False,
                reason=f"Table '{table}' not in manifest permissions",
                effective_tier=DEFAULT_TIER,
            )
        return AccessDecision(
            allowed=True,
            reason=(
                f"Table '{table}' permitted "
                f"up to tier {self._permitted_tables[table]}"
            ),
            effective_tier=self._permitted_tables[table],
        )

    def check_query(self, sql: str) -> AccessDecision:
        """Validate a SQL query against manifest permissions.

        Extracts table names from FROM/JOIN clauses and validates each.
        Rejects destructive DDL and writes to non-ext tables.

        sensitivity_tier: N/A
        """
        if _DESTRUCTIVE_KEYWORDS.search(sql):
            return AccessDecision(
                allowed=False,
                reason="DDL operations (DROP/ALTER/TRUNCATE/CREATE) not allowed",
                effective_tier=DEFAULT_TIER,
            )

        # Check for writes to non-ext tables.
        for match in _WRITE_OPS.finditer(sql):
            table = match.group(2)
            if not table.startswith("ext_"):
                return AccessDecision(
                    allowed=False,
                    reason=f"Write to non-extension table '{table}' not allowed",
                    effective_tier=DEFAULT_TIER,
                )

        tables = self._extract_tables(sql)
        if not tables:
            return AccessDecision(
                allowed=True,
                reason="No tables detected in query",
                effective_tier=1,
            )

        max_tier = 1
        for table in tables:
            decision = self.check_table_access(table)
            if not decision.allowed:
                return decision
            max_tier = max(max_tier, decision.effective_tier)

        return AccessDecision(
            allowed=True,
            reason=f"All tables permitted, max tier {max_tier}",
            effective_tier=max_tier,
        )

    def inject_tier_filter(self, sql: str, max_tier: int) -> str:
        """Add ``WHERE sensitivity_tier <= N`` to SELECT queries.

        If a WHERE clause already exists, appends ``AND``.

        sensitivity_tier: N/A
        """
        if max_tier >= 3:
            return sql

        tier_clause = f"sensitivity_tier <= {max_tier}"

        if _WHERE_PATTERN.search(sql):
            # Find the last WHERE and append AND before any GROUP BY/ORDER BY/LIMIT.
            parts = _WHERE_PATTERN.split(sql, maxsplit=1)
            return f"{parts[0]}WHERE {tier_clause} AND {parts[1]}"

        # Insert WHERE before GROUP BY, ORDER BY, LIMIT, or at the end.
        insert_re = re.compile(
            r"\b(GROUP\s+BY|ORDER\s+BY|LIMIT|HAVING)\b",
            re.IGNORECASE,
        )
        match = insert_re.search(sql)
        if match:
            pos = match.start()
            return f"{sql[:pos]}WHERE {tier_clause} {sql[pos:]}"

        return f"{sql} WHERE {tier_clause}"

    def validate_write_table(self, table_name: str) -> bool:
        """Verify table name matches ``ext_{agent_id}_*`` pattern.

        Also checks that the table is declared in manifest.write_tables.

        sensitivity_tier: N/A
        """
        expected_prefix = f"ext_{self._agent_id.replace('-', '_')}_"
        if not table_name.startswith(expected_prefix):
            return False
        return table_name in self._manifest.write_tables

    def log_access(
        self,
        decision: AccessDecision,
        details: dict[str, Any],
    ) -> None:
        """Append access decision to the agent's audit log.

        Written to ``~/.arandu/data/agents/{agent_id}/audit.jsonl``.

        sensitivity_tier: 1
        """
        entry = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "agent_id": self._agent_id,
            "allowed": decision.allowed,
            "reason": decision.reason,
            "effective_tier": decision.effective_tier,
            **details,
        }
        try:
            self._audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self._audit_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except OSError:
            logger.warning(
                "Failed to write audit log for agent %s",
                self._agent_id,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_tables(sql: str) -> list[str]:
        """Extract table names from FROM and JOIN clauses.

        sensitivity_tier: N/A
        """
        tables: list[str] = []
        for match in _TABLE_PATTERN.finditer(sql):
            table = match.group(1) or match.group(2)
            if table and table.upper() not in {"SELECT", "WHERE", "AND", "OR"}:
                tables.append(table)
        return tables
