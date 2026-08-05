"""Pydantic AI SQLMesh model generator.

Given a discovered schema and a target pipeline layer, return a
:class:`GeneratedSQLModel` carrying a complete SQLMesh model file.

sensitivity_tier: 1
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from src.agents.core.agent_base import SBAgent
from src.agents.core.output_types import GeneratedSQLModel
from src.agents.core.scheduler import Tier

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """\
You are a SQLMesh model generator for Arandu, a privacy-first \
personal AI. Generate DuckDB-dialect SQL models. Return a \
GeneratedSQLModel with ``name``, ``layer``, ``sql``, and \
``sensitivity_summary``.

Layer naming (REQUIRED — ``name`` MUST start with the layer prefix):
- staging      → ``ext_stg_<connector>_<entity>``
- intermediate → ``ext_int_<connector>_<entity>``
- marts        → ``ext_mart_<connector>_<entity>``

The input ``schema.target_table`` is shaped ``ext_<connector>_<entity>``. \
Derive ``name`` by replacing its leading ``ext_`` with the layer prefix \
(e.g. staging ``ext_strava_runs`` → ``ext_stg_strava_runs``).

Layer lineage (the SQL ``FROM`` clause):
- staging      → SELECT FROM the raw source table (the connector's \
``target_table`` as-is).
- intermediate → SELECT FROM ``ext_stg_*`` only.
- marts        → SELECT FROM ``ext_int_*`` only.

Other rules:
1. Every column MUST have a ``sensitivity_tier`` comment in the header \
of the SQL.
2. Staging models: CAST every column, add ``_loaded_at``, include \
audits.
3. Intermediate models: use CTEs, JOINs, CASE-driven categorisation.
4. Mart models: dashboard-ready with ``item_type``, ``title``, \
``detail``, ``occurred_at`` columns.
5. NEVER lower sensitivity tiers — only maintain or raise.
6. Use DuckDB syntax (not PostgreSQL or MySQL).
7. Include a MODEL declaration with ``kind FULL`` and ``grain``.
8. ``sensitivity_summary`` is a single sentence describing the \
highest tier in the model.\
"""


@dataclass(frozen=True)
class ModelGeneratorDeps:
    """Typed input bundle for :class:`ModelGeneratorAgent`.

    sensitivity_tier: 1
    """

    schema: dict[str, Any] = field(default_factory=dict)
    layer: Literal["staging", "intermediate", "marts"] = "staging"
    connector_id: str = ""


class ModelGeneratorAgent(
    SBAgent[ModelGeneratorDeps | str, GeneratedSQLModel],
):
    """Generate one SQLMesh model from a discovered schema.

    sensitivity_tier: 1
    """

    agent_id = "model_generator"
    output_type = GeneratedSQLModel
    tier = Tier.BACKGROUND
    system_prompt = DEFAULT_SYSTEM_PROMPT

    def build_prompt(
        self, deps: ModelGeneratorDeps | str,
    ) -> str:
        """Render deps into a JSON-laden user message.

        sensitivity_tier: 1
        """
        if isinstance(deps, str):
            return deps
        return (
            f"Connector: {deps.connector_id}\n"
            f"Layer: {deps.layer}\n\n"
            "Schema (JSON):\n"
            f"{json.dumps(deps.schema, sort_keys=True)}\n\n"
            "Return a GeneratedSQLModel for this layer."
        )

    def generate(
        self,
        *,
        schema: dict[str, Any],
        layer: Literal["staging", "intermediate", "marts"] = "staging",
        connector_id: str = "",
    ) -> GeneratedSQLModel | None:
        """Convenience entrypoint mirroring the legacy call shape.

        sensitivity_tier: 1
        """
        if not schema:
            return None
        deps = ModelGeneratorDeps(
            schema=schema, layer=layer, connector_id=connector_id,
        )
        record = self.run(deps)
        if record.output is None or record.error is not None:
            return None
        return record.output


def register_model_generator_agent() -> None:
    """Register the model generator agent in the registry.

    sensitivity_tier: 1
    """
    from src.agents.core.config_store import AgentConfig
    from src.agents.core.registry import (
        AgentDefinition,
        get_agent,
        register_agent,
    )

    if get_agent("model_generator") is not None:
        return

    default = AgentConfig(
        agent_id="model_generator",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        model_route="inherit",
        model_override=None,
        enabled_tools=(),
        enabled_skills=(),
        editable=False,
    )
    register_agent(AgentDefinition(
        agent_id="model_generator",
        name="Model Generator",
        description=(
            "Generates SQLMesh staging / intermediate / mart models "
            "from a discovered schema. Called indirectly by the "
            "ingestion lifecycle."
        ),
        category="ingestion",
        parent_agent="brain",
        tier=Tier.BACKGROUND,
        max_sensitivity_tier=2,
        editable=False,
        default_config=default,
        available_tools=(),
        available_skills=(),
        output_schema="GeneratedSQLModel",
        pattern="single",
        factory=ModelGeneratorAgent,
        tags=("ingestion", "indirect"),
    ))


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "ModelGeneratorAgent",
    "ModelGeneratorDeps",
    "register_model_generator_agent",
]
