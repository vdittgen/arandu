"""Auto-generate SQLMesh pipeline models for new data sources.

When schema discovery returns ``is_new_table: true``, this module generates
staging (+ optionally intermediate and mart) SQLMesh SQL models so the data
flows through the standard transformation pipeline without manual authoring.

Supports rule-based SQL template generation with optional LLM enhancement
via Ollama for richer intermediate/mart models.

sensitivity_tier: 1 (generates pipeline metadata, no user data)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.extensions.ingestion.schema_discovery import (
    DiscoveredMapping,
    FieldMapping,
)

logger = logging.getLogger(__name__)


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences from an LLM-generated SQL string."""
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.index("\n") if "\n" in text else len(text)
        text = text[first_newline + 1:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_MODELS_PER_EXTENSION = 10

_KNOWN_DOMAINS: frozenset[str] = frozenset({
    "messages", "calendar", "health", "notes", "contacts", "files", "email",
})

_DOMAIN_TO_COLLECTION: dict[str, str] = {
    "messages": "work",
    "email": "work",
    "calendar": "work",
    "health": "health",
    "notes": "personal",
    "files": "personal",
    "contacts": "social",
    "music": "ideas",
    "browser": "ideas",
    "general": "personal",
}

_GRAPH_EXTENSIONS: dict[str, dict[str, Any]] = {
    "music": {
        "node_ddl": (
            'CREATE NODE TABLE IF NOT EXISTS Track (\n'
            '    id               STRING,\n'
            '    name             STRING,\n'
            '    artist           STRING,\n'
            '    album            STRING,\n'
            '    sensitivity_tier INT64,\n'
            '    PRIMARY KEY (id)\n'
            ')',
        ),
        "relationship_ddl": (
            'CREATE REL TABLE IF NOT EXISTS LISTENED_TO (\n'
            '    FROM Person TO Track,\n'
            '    weight           DOUBLE,\n'
            '    timestamp        TIMESTAMP,\n'
            '    sensitivity_tier INT64\n'
            ')',
        ),
        "node_names": ("Track",),
        "relationship_names": ("LISTENED_TO",),
    },
    "browser": {
        "node_ddl": (
            'CREATE NODE TABLE IF NOT EXISTS WebPage (\n'
            '    id               STRING,\n'
            '    url              STRING,\n'
            '    title            STRING,\n'
            '    domain           STRING,\n'
            '    sensitivity_tier INT64,\n'
            '    PRIMARY KEY (id)\n'
            ')',
        ),
        "relationship_ddl": (
            'CREATE REL TABLE IF NOT EXISTS VISITED (\n'
            '    FROM Person TO WebPage,\n'
            '    weight           DOUBLE,\n'
            '    timestamp        TIMESTAMP,\n'
            '    sensitivity_tier INT64\n'
            ')',
        ),
        "node_names": ("WebPage",),
        "relationship_names": ("VISITED",),
    },
}

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeneratedModel:
    """A single generated SQLMesh model file.

    sensitivity_tier: 1
    """

    model_name: str
    layer: str
    filename: str
    sql_content: str
    sensitivity_summary: dict[int, int] = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class GraphExtension:
    """Kuzu graph schema extension suggestion.

    sensitivity_tier: 1
    """

    node_ddl: tuple[str, ...]
    relationship_ddl: tuple[str, ...]
    node_names: tuple[str, ...]
    relationship_names: tuple[str, ...]


@dataclass(frozen=True)
class ChromaDBMapping:
    """ChromaDB collection assignment for the new data.

    sensitivity_tier: 1
    """

    collection_name: str
    domain: str
    indexing_fields: tuple[str, ...]


@dataclass(frozen=True)
class ModelPreview:
    """Complete preview of generated models for user review.

    sensitivity_tier: 1
    """

    connector_id: str
    strategy: str
    models: tuple[GeneratedModel, ...]
    graph_extension: GraphExtension | None
    chromadb_mapping: ChromaDBMapping | None
    staging_dir: str
    warnings: tuple[str, ...]
    total_models: int
    sensitivity_summary: dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelResult:
    """Result of approving generated models.

    sensitivity_tier: 1
    """

    status: str
    models_installed: int
    files_created: tuple[str, ...]
    pipeline_models_added: tuple[str, ...]
    graph_extensions_applied: bool
    error: str | None = None


# ---------------------------------------------------------------------------
# SQL templates
# ---------------------------------------------------------------------------

_STAGING_TEMPLATE = """\
/*
    Staged {domain} data from {connector_id} with type casting and sensitivity metadata.

    Column Sensitivity Tiers:
{sensitivity_comments}
*/
MODEL (
    name {model_name},
    kind FULL,
    grain {grain},
    audits (
        not_null(columns=[{not_null_cols}]),
        unique_values(columns=[{unique_col}]),
        accepted_values(column=sensitivity_tier, is_in=(1, 2, 3))
    )
);

SELECT
{select_expressions},
    CAST(sensitivity_tier AS INTEGER)       AS sensitivity_tier,
    CURRENT_TIMESTAMP                       AS _loaded_at
FROM {source_table}
"""

_INTERMEDIATE_TEMPLATE = """\
/*
    Enriched {domain} data from {connector_id} with domain categorization.

    Column Sensitivity Tiers:
{sensitivity_comments}
        domain_category:  tier 1
        _loaded_at:       tier 1
*/
MODEL (
    name {model_name},
    kind FULL,
    grain {grain},
    audits (
        not_null(columns=[{not_null_cols}]),
        unique_values(columns=[{unique_col}]),
        accepted_values(column=sensitivity_tier, is_in=(1, 2, 3))
    )
);

SELECT
{select_cols},
    s.sensitivity_tier,
    '{domain}'                              AS domain_category,
    CURRENT_TIMESTAMP                       AS _loaded_at
FROM {staging_model} s
"""

_MART_TEMPLATE = """\
/*
    {domain} mart for dashboard display from {connector_id}.

    Column Sensitivity Tiers:
        item_type:        tier 1
        id:               tier 1
        title:            tier 2
        detail:           tier 3
        occurred_at:      tier 2
        category:         tier 1
        sensitivity_tier: tier 1
        _loaded_at:       tier 1
*/
MODEL (
    name {model_name},
    kind FULL,
    grain (item_type, id),
    audits (
        not_null(columns=[item_type, id, title, occurred_at, sensitivity_tier]),
        accepted_values(column=sensitivity_tier, is_in=(1, 2, 3))
    )
);

SELECT
    '{item_type}'                           AS item_type,
    CAST(s.{id_col} AS VARCHAR)            AS id,
    {title_expr}                            AS title,
    {detail_expr}                           AS detail,
    {timestamp_expr}                        AS occurred_at,
    {category_expr}                         AS category,
    CAST(NULL AS INTEGER)                   AS duration_minutes,
    s.sensitivity_tier,
    CAST(NULL AS VARCHAR)                   AS coaching_phrase,
    CURRENT_TIMESTAMP                       AS _loaded_at
FROM {source_model} s
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# The SQLMesh-generation system prompt lives in
# :mod:`src.agents.model_generator.agent` (pydantic-ai).


def _make_connector_slug(connector_id: str) -> str:
    """Convert a connector_id to a short slug for model names.

    sensitivity_tier: 1
    """
    slug = connector_id.removeprefix("custom-")
    return re.sub(r"[^a-z0-9]+", "_", slug.lower()).strip("_")


def _find_id_field(fields: tuple[FieldMapping, ...]) -> str:
    """Find the best ID field name from field mappings.

    sensitivity_tier: 1
    """
    for f in fields:
        if f.target_column == "id":
            return "id"
    for f in fields:
        if f.target_column.endswith("_id"):
            return f.target_column
    return fields[0].target_column if fields else "id"


def _find_title_field(
    fields: tuple[FieldMapping, ...],
) -> str | None:
    """Find a field suitable for display as a title.

    sensitivity_tier: 1
    """
    title_keywords = {"title", "name", "subject", "label", "track_name"}
    for f in fields:
        if f.target_column in title_keywords:
            return f.target_column
    return None


def _find_detail_field(
    fields: tuple[FieldMapping, ...],
) -> str | None:
    """Find a field suitable for display as detail text.

    sensitivity_tier: 1
    """
    detail_keywords = {
        "content", "description", "body", "text",
        "body_preview", "detail", "summary",
    }
    for f in fields:
        if f.target_column in detail_keywords:
            return f.target_column
    return None


def _find_timestamp_field(
    fields: tuple[FieldMapping, ...],
) -> str | None:
    """Find the best timestamp field for occurred_at.

    sensitivity_tier: 1
    """
    ts_keywords = {
        "timestamp", "created_at", "date", "played_at",
        "start_time", "sent_at", "received_at", "last_visited",
    }
    for f in fields:
        if f.target_column in ts_keywords:
            return f.target_column
    for f in fields:
        col = f.target_column
        if "time" in col or "date" in col or col.endswith("_at"):
            return col
    return None


def _build_sensitivity_comments(
    fields: tuple[FieldMapping, ...],
) -> str:
    """Build the sensitivity tier comment block for SQL header.

    sensitivity_tier: 1
    """
    lines: list[str] = []
    for f in fields:
        lines.append(f"        {f.target_column + ':':<22}tier {f.sensitivity_tier}")
    lines.append(f"        {'sensitivity_tier:':<22}tier 1")
    lines.append(f"        {'_loaded_at:':<22}tier 1")
    return "\n".join(lines)


def _build_sensitivity_summary(
    fields: tuple[FieldMapping, ...],
) -> dict[int, int]:
    """Build a tier count dict from field mappings.

    sensitivity_tier: 1
    """
    counts: dict[int, int] = {}
    for f in fields:
        counts[f.sensitivity_tier] = counts.get(f.sensitivity_tier, 0) + 1
    return counts


def _find_text_fields(
    fields: tuple[FieldMapping, ...],
) -> tuple[str, ...]:
    """Find text fields suitable for ChromaDB indexing.

    sensitivity_tier: 1
    """
    text_types = {"TEXT", "VARCHAR"}
    text_keywords = {
        "content", "description", "body", "text", "title",
        "subject", "name", "body_preview", "summary",
    }
    result: list[str] = []
    for f in fields:
        if f.target_type in text_types and f.target_column in text_keywords:
            result.append(f.target_column)
    return tuple(result)


# ---------------------------------------------------------------------------
# ModelGenerator
# ---------------------------------------------------------------------------


class ModelGenerator:
    """Auto-generates SQLMesh pipeline models from DiscoveredMapping.

    Decides integration strategy, generates SQL via rule-based templates
    (with optional LLM enhancement), and suggests Kuzu graph and ChromaDB
    collection extensions.

    sensitivity_tier: 1
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        host: str = "http://localhost:11434",
        project_root: Path = PROJECT_ROOT,
    ) -> None:
        self._model = model
        self._host = host
        self._project_root = project_root

    def generate(
        self,
        mapping: DiscoveredMapping,
        connector_id: str,
        force_full_pipeline: bool = False,
    ) -> ModelPreview:
        """Generate pipeline models for a discovered data source.

        Args:
            mapping: Schema discovery result with field mappings.
            connector_id: The extension connector ID.
            force_full_pipeline: Force ``full_pipeline`` strategy even when
                the domain normally maps to existing core models.

        Returns:
            ModelPreview with all generated models and metadata.

        Raises:
            ValueError: If mapping has no fields.

        sensitivity_tier: 1
        """
        if not mapping.fields:
            msg = "Cannot generate models from a mapping with no fields"
            raise ValueError(msg)

        strategy = self._determine_strategy(
            mapping,
            force_full_pipeline=force_full_pipeline,
        )
        warnings: list[str] = []
        models: list[GeneratedModel] = []

        if strategy == "extend_existing":
            staging_dir = self._staging_dir(connector_id)
            return ModelPreview(
                connector_id=connector_id,
                strategy=strategy,
                models=(),
                graph_extension=None,
                chromadb_mapping=self._determine_chromadb_collection(mapping),
                staging_dir=str(staging_dir),
                warnings=("No models needed — data fits existing table",),
                total_models=0,
                sensitivity_summary=_build_sensitivity_summary(mapping.fields),
            )

        slug = _make_connector_slug(connector_id)

        # Generate staging model
        stg = self._generate_staging_model(mapping, connector_id, slug)
        models.append(stg)

        if strategy == "staging_only":
            warnings.append(
                f"Domain '{mapping.domain}' has existing intermediate/mart models. "
                "Consider manually integrating this staging model into them."
            )

        # Generate intermediate + mart for full_pipeline
        if strategy == "full_pipeline":
            intermediate = self._generate_intermediate_model(
                mapping, connector_id, slug, stg.model_name,
            )
            models.append(intermediate)

            mart = self._generate_mart_model(
                mapping, connector_id, slug,
                intermediate.model_name, stg.model_name,
            )
            models.append(mart)

        # Safety guard: max models
        if len(models) > MAX_MODELS_PER_EXTENSION:
            msg = (
                f"Generated {len(models)} models, "
                f"exceeding maximum of {MAX_MODELS_PER_EXTENSION}"
            )
            raise ValueError(msg)

        # Validate ext_ prefix
        for m in models:
            basename = m.model_name.split(".")[-1]
            if not basename.startswith("ext_"):
                msg = f"Model name '{m.model_name}' missing ext_ prefix"
                raise ValueError(msg)

        # Graph and ChromaDB
        graph_ext = self._generate_graph_extension(mapping)
        chromadb_map = self._determine_chromadb_collection(mapping)

        # Aggregate sensitivity summary
        agg_summary: dict[int, int] = {}
        for m in models:
            for tier, count in m.sensitivity_summary.items():
                agg_summary[tier] = agg_summary.get(tier, 0) + count

        staging_dir = self._staging_dir(connector_id)

        return ModelPreview(
            connector_id=connector_id,
            strategy=strategy,
            models=tuple(models),
            graph_extension=graph_ext,
            chromadb_mapping=chromadb_map,
            staging_dir=str(staging_dir),
            warnings=tuple(warnings),
            total_models=len(models),
            sensitivity_summary=agg_summary,
        )

    # ------------------------------------------------------------------
    # Strategy
    # ------------------------------------------------------------------

    def _determine_strategy(
        self,
        mapping: DiscoveredMapping,
        force_full_pipeline: bool = False,
    ) -> str:
        """Decide integration strategy for the discovered data.

        sensitivity_tier: 1
        """
        if force_full_pipeline:
            return "full_pipeline"
        if not mapping.is_new_table:
            return "extend_existing"
        if mapping.domain in _KNOWN_DOMAINS:
            return "staging_only"
        return "full_pipeline"

    # ------------------------------------------------------------------
    # Staging model
    # ------------------------------------------------------------------

    def _generate_staging_model(
        self,
        mapping: DiscoveredMapping,
        connector_id: str,
        slug: str,
    ) -> GeneratedModel:
        """Generate a staging SQL model.

        sensitivity_tier: 1
        """
        model_name = f"staging.ext_stg_{mapping.domain}_{slug}"
        filename = f"ext_stg_{mapping.domain}_{slug}.sql"

        # Try LLM first, fall back to rule-based
        sql = self._generate_sql_via_llm(mapping, "staging", model_name)
        if sql is None:
            sql = self._generate_staging_sql_rule_based(
                mapping, connector_id, model_name,
            )

        return GeneratedModel(
            model_name=model_name,
            layer="staging",
            filename=filename,
            sql_content=sql,
            sensitivity_summary=_build_sensitivity_summary(mapping.fields),
            depends_on=(mapping.target_table,),
        )

    def _generate_staging_sql_rule_based(
        self,
        mapping: DiscoveredMapping,
        connector_id: str,
        model_name: str,
    ) -> str:
        """Generate staging SQL via template interpolation.

        sensitivity_tier: 1
        """
        id_field = _find_id_field(mapping.fields)
        grain = id_field

        # Build SELECT expressions
        select_lines: list[str] = []
        not_null_cols: list[str] = []
        for f in mapping.fields:
            if f.target_column == "sensitivity_tier":
                continue
            pad = max(1, 24 - len(f.target_column) - len(f.target_type))
            select_lines.append(
                f"    CAST({f.target_column} AS {f.target_type})"
                f"{' ' * pad}AS {f.target_column}"
            )
            not_null_cols.append(f.target_column)

        # Always include sensitivity_tier in not_null
        not_null_cols.append("sensitivity_tier")

        return _STAGING_TEMPLATE.format(
            domain=mapping.domain,
            connector_id=connector_id,
            sensitivity_comments=_build_sensitivity_comments(mapping.fields),
            model_name=model_name,
            grain=grain,
            not_null_cols=", ".join(not_null_cols),
            unique_col=id_field,
            select_expressions=",\n".join(select_lines),
            source_table=mapping.target_table,
        )

    # ------------------------------------------------------------------
    # Intermediate model
    # ------------------------------------------------------------------

    def _generate_intermediate_model(
        self,
        mapping: DiscoveredMapping,
        connector_id: str,
        slug: str,
        stg_model_name: str,
    ) -> GeneratedModel:
        """Generate an intermediate SQL model.

        sensitivity_tier: 1
        """
        model_name = f"intermediate.ext_int_{mapping.domain}_{slug}"
        filename = f"ext_int_{mapping.domain}_{slug}.sql"

        sql = self._generate_sql_via_llm(mapping, "intermediate", model_name)
        if sql is None:
            sql = self._generate_intermediate_sql_rule_based(
                mapping, connector_id, model_name, stg_model_name,
            )

        return GeneratedModel(
            model_name=model_name,
            layer="intermediate",
            filename=filename,
            sql_content=sql,
            sensitivity_summary=_build_sensitivity_summary(mapping.fields),
            depends_on=(stg_model_name,),
        )

    def _generate_intermediate_sql_rule_based(
        self,
        mapping: DiscoveredMapping,
        connector_id: str,
        model_name: str,
        stg_model_name: str,
    ) -> str:
        """Generate intermediate SQL via template interpolation.

        sensitivity_tier: 1
        """
        id_field = _find_id_field(mapping.fields)

        select_cols: list[str] = []
        not_null_cols: list[str] = []
        for f in mapping.fields:
            if f.target_column == "sensitivity_tier":
                continue
            select_cols.append(f"    s.{f.target_column}")
            not_null_cols.append(f.target_column)

        not_null_cols.append("sensitivity_tier")

        return _INTERMEDIATE_TEMPLATE.format(
            domain=mapping.domain,
            connector_id=connector_id,
            sensitivity_comments=_build_sensitivity_comments(mapping.fields),
            model_name=model_name,
            grain=id_field,
            not_null_cols=", ".join(not_null_cols),
            unique_col=id_field,
            select_cols=",\n".join(select_cols),
            staging_model=stg_model_name,
        )

    # ------------------------------------------------------------------
    # Mart model
    # ------------------------------------------------------------------

    def _generate_mart_model(
        self,
        mapping: DiscoveredMapping,
        connector_id: str,
        slug: str,
        int_model_name: str,
        stg_model_name: str,
    ) -> GeneratedModel:
        """Generate a mart SQL model.

        sensitivity_tier: 1
        """
        model_name = f"mart.ext_mart_{mapping.domain}_{slug}"
        filename = f"ext_mart_{mapping.domain}_{slug}.sql"

        sql = self._generate_sql_via_llm(mapping, "mart", model_name)
        if sql is None:
            sql = self._generate_mart_sql_rule_based(
                mapping, connector_id, model_name, int_model_name,
            )

        return GeneratedModel(
            model_name=model_name,
            layer="mart",
            filename=filename,
            sql_content=sql,
            sensitivity_summary=_build_sensitivity_summary(mapping.fields),
            depends_on=(int_model_name,),
        )

    def _generate_mart_sql_rule_based(
        self,
        mapping: DiscoveredMapping,
        connector_id: str,
        model_name: str,
        source_model: str,
    ) -> str:
        """Generate mart SQL via template interpolation.

        sensitivity_tier: 1
        """
        id_field = _find_id_field(mapping.fields)
        title_field = _find_title_field(mapping.fields)
        detail_field = _find_detail_field(mapping.fields)
        ts_field = _find_timestamp_field(mapping.fields)

        title_expr = (
            f"CAST(s.{title_field} AS VARCHAR)"
            if title_field
            else f"CAST(s.{id_field} AS VARCHAR)"
        )
        detail_expr = (
            f"CAST(s.{detail_field} AS TEXT)"
            if detail_field
            else "CAST(NULL AS TEXT)"
        )
        timestamp_expr = (
            f"s.{ts_field}"
            if ts_field
            else "CURRENT_TIMESTAMP"
        )
        category_expr = f"'{mapping.domain}'"

        return _MART_TEMPLATE.format(
            domain=mapping.domain,
            connector_id=connector_id,
            model_name=model_name,
            item_type=mapping.domain,
            id_col=id_field,
            title_expr=title_expr,
            detail_expr=detail_expr,
            timestamp_expr=timestamp_expr,
            category_expr=category_expr,
            source_model=source_model,
        )

    # ------------------------------------------------------------------
    # Graph extension
    # ------------------------------------------------------------------

    def _generate_graph_extension(
        self,
        mapping: DiscoveredMapping,
    ) -> GraphExtension | None:
        """Generate Kuzu graph schema extensions for the domain.

        sensitivity_tier: 1
        """
        ext = _GRAPH_EXTENSIONS.get(mapping.domain)
        if ext is None:
            return None

        return GraphExtension(
            node_ddl=tuple(ext["node_ddl"]),
            relationship_ddl=tuple(ext["relationship_ddl"]),
            node_names=tuple(ext["node_names"]),
            relationship_names=tuple(ext["relationship_names"]),
        )

    # ------------------------------------------------------------------
    # ChromaDB mapping
    # ------------------------------------------------------------------

    def _determine_chromadb_collection(
        self,
        mapping: DiscoveredMapping,
    ) -> ChromaDBMapping:
        """Determine which ChromaDB collection to use for the data.

        sensitivity_tier: 1
        """
        collection = _DOMAIN_TO_COLLECTION.get(
            mapping.domain, "personal",
        )
        text_fields = _find_text_fields(mapping.fields)

        return ChromaDBMapping(
            collection_name=collection,
            domain=mapping.domain,
            indexing_fields=text_fields,
        )

    # ------------------------------------------------------------------
    # LLM SQL generation
    # ------------------------------------------------------------------

    def _generate_sql_via_llm(
        self,
        mapping: DiscoveredMapping,
        layer: str,
        model_name: str,  # noqa: ARG002 (kept for API stability)
    ) -> str | None:
        """Generate SQL via :class:`ModelGeneratorAgent` (pydantic-ai).

        Returns the SQL string or None on failure.

        sensitivity_tier: 1
        """
        from src.agents.model_generator.agent import (
            ModelGeneratorAgent as _Agent,
        )

        schema = {
            "target_table": mapping.target_table,
            "domain": mapping.domain,
            "dedup_key": list(mapping.dedup_key),
            "fields": [
                {
                    "source_name": f.source_name,
                    "target_column": f.target_column,
                    "target_type": f.target_type,
                    "sensitivity_tier": f.sensitivity_tier,
                    "transform": f.transform,
                }
                for f in mapping.fields
            ],
        }
        try:
            result = _Agent().generate(
                schema=schema,
                layer=layer,  # type: ignore[arg-type]
                connector_id=mapping.tool_name,
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "ModelGeneratorAgent failed",
                exc_info=True,
            )
            return None
        if result is None:
            return None

        sql = _strip_markdown_fences(result.sql)
        # Basic validation: must contain MODEL and SELECT
        if "MODEL" not in sql or "SELECT" not in sql:
            logger.warning("Agent SQL missing MODEL or SELECT — discarding")
            return None
        if "ext_" not in sql:
            logger.warning("Agent SQL missing ext_ prefix — discarding")
            return None
        return sql

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _staging_dir(self, connector_id: str) -> Path:
        """Get the staging directory path for a connector.

        sensitivity_tier: 1
        """
        base = Path.home() / ".arandu" / "extensions" / connector_id / "generated"
        return base
