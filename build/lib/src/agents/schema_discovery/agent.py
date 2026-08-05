"""Pydantic AI schema discoverer.

Given an MCP tool's name plus a sample of its output records, return
a :class:`SchemaDiscoveryDraft` mapping each field to a SQLite column
with a conservative sensitivity tier.

sensitivity_tier: 1
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from src.agents.core.agent_base import SBAgent
from src.agents.core.output_types import SchemaDiscoveryDraft
from src.agents.core.scheduler import Tier

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """\
You are a data schema analyst for Arandu, a privacy-first \
personal AI. Analyse the supplied MCP tool output and map every \
field to a SQLite-friendly column. Return a SchemaDiscoveryDraft \
matching the schema.

Canonical raw_* tables (rename source fields to the matching \
canonical column when you reuse one of these):
- raw_messages: id, source, sender, recipient, content, timestamp, \
sender_name, chat_name, is_from_me, is_group
- raw_calendar_events: id, title, description, start_time, end_time, \
location, attendees, is_all_day
- raw_notes: id, title, content, source, created_at, updated_at, tags
- raw_health_metrics: id, metric_type, value, unit, recorded_at, source
- raw_contacts: id, name, email, phone, relationship, notes, \
last_contact, birthday, address
- raw_files: id, filepath, filename, filetype, size_bytes, \
content_preview, modified_at
- raw_emails: id, subject, from_address, to_addresses, date, \
body_preview, folder, is_read
- raw_reminders: id, title, due_date, notes, completed, list_name
- raw_workouts: id, workout_type, duration_min, calories, \
heart_rate_avg, date
- raw_listening_history: id, track_name, artist, album, played_at, \
duration_ms
- raw_voice_memos: id, title, duration_seconds, recorded_at, transcript

Rules:
1. Be CONSERVATIVE with sensitivity tiers — when in doubt, choose \
the higher tier.
   - tier 1: general / public (preferences, categories, titles)
   - tier 2: personal (names, routines, locations, schedules)
   - tier 3: sensitive (health, finances, emotions, contact details \
like email/phone)
2. Use snake_case for all column names.
3. Choose ``target_type`` from: TEXT, VARCHAR, INTEGER, BIGINT, \
DOUBLE, BOOLEAN, REAL, JSON.
4. Only reuse a canonical raw_* table when its name appears in \
``known_tables`` (the caller's list of tables that actually exist in \
the destination database). The canonical schema block above is for \
renaming source fields once a reuse is authorised — it does NOT \
license claiming a table the caller hasn't listed. When you do reuse \
one, set ``target_table`` to that name, ``is_new_table`` false, and \
rename every source field to the matching canonical column \
(e.g. ``type`` → ``workout_type``, ``avg_hr`` → ``heart_rate_avg``, \
``from`` → ``from_address``, ``moving_time`` → ``duration_min``). \
Otherwise propose a new ``ext_*`` table and set ``is_new_table`` true.
5. ``dedup_key`` is a list of column names whose combined values \
uniquely identify a record. Prefer 1-2 columns.
6. ``transform`` is optional — a short hint like "datetime_iso", \
"json_string", "lowercase", or "seconds_to_minutes" when the raw \
value needs normalising.\
"""

_MAX_SAMPLE_CHARS = 8000


@dataclass(frozen=True)
class SchemaDiscoveryDeps:
    """Typed input bundle for :class:`SchemaDiscoveryAgent`.

    sensitivity_tier: 1
    """

    tool_name: str
    sample_records: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    known_tables: tuple[str, ...] = field(default_factory=tuple)


class SchemaDiscoveryAgent(
    SBAgent[SchemaDiscoveryDeps | str, SchemaDiscoveryDraft],
):
    """Map an MCP tool's output records to SQLite columns.

    sensitivity_tier: 1
    """

    agent_id = "schema_discovery"
    output_type = SchemaDiscoveryDraft
    tier = Tier.BACKGROUND
    system_prompt = DEFAULT_SYSTEM_PROMPT

    def build_prompt(
        self, deps: SchemaDiscoveryDeps | str,
    ) -> str:
        """Render deps into a JSON-laden user message.

        sensitivity_tier: 1
        """
        if isinstance(deps, str):
            return deps
        sample_blob = json.dumps(
            list(deps.sample_records)[:5], default=str,
        )
        if len(sample_blob) > _MAX_SAMPLE_CHARS:
            sample_blob = sample_blob[:_MAX_SAMPLE_CHARS] + "...]"
        return (
            f"Tool: {deps.tool_name}\n\n"
            f"Known SQLite tables: {list(deps.known_tables)}\n\n"
            "Sample records (first 5):\n"
            f"{sample_blob}\n\n"
            "Return a SchemaDiscoveryDraft."
        )

    def discover(
        self,
        *,
        tool_name: str,
        sample_records: list[dict[str, Any]],
        known_tables: list[str] | None = None,
    ) -> SchemaDiscoveryDraft | None:
        """Convenience entrypoint mirroring the legacy call shape.

        sensitivity_tier: 1
        """
        if not sample_records:
            return None
        deps = SchemaDiscoveryDeps(
            tool_name=tool_name,
            sample_records=tuple(sample_records),
            known_tables=tuple(known_tables or ()),
        )
        record = self.run(deps)
        if record.output is None or record.error is not None:
            return None
        return record.output


def register_schema_discovery_agent() -> None:
    """Register the schema discovery agent in the registry.

    sensitivity_tier: 1
    """
    from src.agents.core.config_store import AgentConfig
    from src.agents.core.registry import (
        AgentDefinition,
        get_agent,
        register_agent,
    )

    if get_agent("schema_discovery") is not None:
        return

    default = AgentConfig(
        agent_id="schema_discovery",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        model_route="inherit",
        model_override=None,
        enabled_tools=(),
        enabled_skills=(),
        editable=False,
    )
    register_agent(AgentDefinition(
        agent_id="schema_discovery",
        name="Schema Discovery",
        description=(
            "Maps MCP tool output records to SQLite columns with "
            "conservative sensitivity tiers. Called indirectly by "
            "the ingestion lifecycle, not by Brain."
        ),
        category="ingestion",
        parent_agent="brain",
        tier=Tier.BACKGROUND,
        max_sensitivity_tier=2,
        editable=False,
        default_config=default,
        available_tools=(),
        available_skills=(),
        output_schema="SchemaDiscoveryDraft",
        pattern="single",
        factory=SchemaDiscoveryAgent,
        tags=("ingestion", "indirect"),
    ))


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "SchemaDiscoveryAgent",
    "SchemaDiscoveryDeps",
    "register_schema_discovery_agent",
]
