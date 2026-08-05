"""Human-in-the-loop review flow for generated pipeline models.

Stages generated SQLMesh models in a temporary directory, allows user
review, and on approval copies them to the pipeline and registers them
in the extension model registry.

Two-phase API:
  1. stage(preview)         → writes to staging dir
  2. approve(connector_id)  → copies to pipeline, registers, cleans up
     reject(connector_id)   → removes staging dir

sensitivity_tier: 1 (manages pipeline metadata, no user data)
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.core.sqlite.engine import DatabaseEngine
from src.extensions.ingestion.model_generator import (
    ChromaDBMapping,
    GeneratedModel,
    GraphExtension,
    ModelPreview,
    ModelResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EXTENSIONS_BASE = Path.home() / ".arandu" / "extensions"
_EXTENSION_MODELS_PATH = Path.home() / ".arandu" / "data" / "extension_models.json"

PROJECT_ROOT = Path(__file__).resolve().parents[3]

_LAYER_DIRS: dict[str, str] = {
    "staging": "src/pipeline/staging",
    "intermediate": "src/pipeline/intermediate",
    "mart": "src/pipeline/marts",
}


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _serialize_preview(preview: ModelPreview) -> dict[str, Any]:
    """Serialize a ModelPreview to a JSON-safe dict.

    sensitivity_tier: 1
    """
    models_list = []
    for m in preview.models:
        models_list.append({
            "model_name": m.model_name,
            "layer": m.layer,
            "filename": m.filename,
            "sql_content": m.sql_content,
            "sensitivity_summary": {
                str(k): v for k, v in m.sensitivity_summary.items()
            },
            "depends_on": list(m.depends_on),
        })

    graph = None
    if preview.graph_extension is not None:
        graph = asdict(preview.graph_extension)

    chromadb = None
    if preview.chromadb_mapping is not None:
        chromadb = asdict(preview.chromadb_mapping)

    return {
        "connector_id": preview.connector_id,
        "strategy": preview.strategy,
        "models": models_list,
        "graph_extension": graph,
        "chromadb_mapping": chromadb,
        "staging_dir": preview.staging_dir,
        "warnings": list(preview.warnings),
        "total_models": preview.total_models,
        "sensitivity_summary": {
            str(k): v for k, v in preview.sensitivity_summary.items()
        },
    }


def _deserialize_preview(data: dict[str, Any]) -> ModelPreview:
    """Deserialize a dict back to a ModelPreview.

    sensitivity_tier: 1
    """
    models = tuple(
        GeneratedModel(
            model_name=m["model_name"],
            layer=m["layer"],
            filename=m["filename"],
            sql_content=m["sql_content"],
            sensitivity_summary={
                int(k): v
                for k, v in m.get("sensitivity_summary", {}).items()
            },
            depends_on=tuple(m.get("depends_on", ())),
        )
        for m in data.get("models", [])
    )

    graph = None
    if data.get("graph_extension") is not None:
        g = data["graph_extension"]
        graph = GraphExtension(
            node_ddl=tuple(g.get("node_ddl", ())),
            relationship_ddl=tuple(g.get("relationship_ddl", ())),
            node_names=tuple(g.get("node_names", ())),
            relationship_names=tuple(g.get("relationship_names", ())),
        )

    chromadb = None
    if data.get("chromadb_mapping") is not None:
        c = data["chromadb_mapping"]
        chromadb = ChromaDBMapping(
            collection_name=c["collection_name"],
            domain=c["domain"],
            indexing_fields=tuple(c.get("indexing_fields", ())),
        )

    return ModelPreview(
        connector_id=data["connector_id"],
        strategy=data["strategy"],
        models=models,
        graph_extension=graph,
        chromadb_mapping=chromadb,
        staging_dir=data["staging_dir"],
        warnings=tuple(data.get("warnings", ())),
        total_models=data.get("total_models", len(models)),
        sensitivity_summary={
            int(k): v
            for k, v in data.get("sensitivity_summary", {}).items()
        },
    )


# ---------------------------------------------------------------------------
# Extension model registry helpers
# ---------------------------------------------------------------------------


def _load_extension_model_registry() -> list[dict[str, str]]:
    """Load the extension model registry from disk.

    sensitivity_tier: 1
    """
    if not _EXTENSION_MODELS_PATH.exists():
        return []
    try:
        data = json.loads(_EXTENSION_MODELS_PATH.read_text(encoding="utf-8"))
        return data.get("models", [])
    except (json.JSONDecodeError, OSError):
        logger.warning("Failed to read extension model registry", exc_info=True)
        return []


def _save_extension_model_registry(
    models: list[dict[str, str]],
) -> None:
    """Save the extension model registry to disk.

    sensitivity_tier: 1
    """
    _EXTENSION_MODELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {"models": models}
    _EXTENSION_MODELS_PATH.write_text(
        json.dumps(data, indent=2) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# ReviewFlow
# ---------------------------------------------------------------------------


class ReviewFlow:
    """Human-in-the-loop review for generated pipeline models.

    Two-phase flow:
    1. stage() — write generated SQL to staging directory
    2. approve() — validate, copy to pipeline, register
       reject() — remove staging directory

    sensitivity_tier: 1
    """

    def __init__(
        self,
        db_engine: DatabaseEngine | None = None,
        project_root: Path = PROJECT_ROOT,
    ) -> None:
        self._db_engine = db_engine
        self._project_root = project_root

    def stage(self, preview: ModelPreview) -> str:
        """Write generated models to the staging directory.

        Creates the staging directory with all .sql files and a
        manifest.json for later retrieval.

        Args:
            preview: The ModelPreview from ModelGenerator.generate().

        Returns:
            Path to the staging directory.

        sensitivity_tier: 1
        """
        staging_dir = Path(preview.staging_dir)
        staging_dir.mkdir(parents=True, exist_ok=True)

        # Write each SQL file
        for model in preview.models:
            sql_path = staging_dir / model.filename
            sql_path.write_text(model.sql_content, encoding="utf-8")
            logger.info("Staged model: %s", sql_path)

        # Write manifest
        manifest_path = staging_dir / "manifest.json"
        manifest_data = _serialize_preview(preview)
        manifest_path.write_text(
            json.dumps(manifest_data, indent=2) + "\n",
            encoding="utf-8",
        )

        logger.info(
            "Staged %d models for %s at %s",
            preview.total_models,
            preview.connector_id,
            staging_dir,
        )
        return str(staging_dir)

    def get_staged(self, connector_id: str) -> ModelPreview | None:
        """Read back a staged preview from the manifest file.

        Args:
            connector_id: The connector ID to look up.

        Returns:
            ModelPreview if staged files exist, None otherwise.

        sensitivity_tier: 1
        """
        staging_dir = _EXTENSIONS_BASE / connector_id / "generated"
        manifest_path = staging_dir / "manifest.json"

        if not manifest_path.exists():
            return None

        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            return _deserialize_preview(data)
        except (json.JSONDecodeError, KeyError, OSError):
            logger.warning(
                "Failed to read staged manifest for %s",
                connector_id,
                exc_info=True,
            )
            return None

    def approve(self, connector_id: str) -> ModelResult:
        """Approve and install staged models.

        1. Re-read staged models from manifest
        2. Dry-run validate each SQL model
        3. Copy .sql files to src/pipeline/{layer}/
        4. Register model names in extension_models.json
        5. Clean up staging directory

        Args:
            connector_id: The connector ID to approve.

        Returns:
            ModelResult with installation status.

        sensitivity_tier: 1
        """
        preview = self.get_staged(connector_id)
        if preview is None:
            return ModelResult(
                status="error",
                models_installed=0,
                files_created=(),
                pipeline_models_added=(),
                graph_extensions_applied=False,
                error=f"No staged models found for {connector_id}",
            )

        # Validate all models first
        for model in preview.models:
            valid, err = self.dry_run_validate(model)
            if not valid:
                return ModelResult(
                    status="error",
                    models_installed=0,
                    files_created=(),
                    pipeline_models_added=(),
                    graph_extensions_applied=False,
                    error=f"Validation failed for {model.model_name}: {err}",
                )

        files_created: list[str] = []
        pipeline_models: list[str] = []

        try:
            # Copy SQL files to pipeline directories
            for model in preview.models:
                layer_dir = _LAYER_DIRS.get(model.layer)
                if layer_dir is None:
                    logger.warning("Unknown layer: %s", model.layer)
                    continue

                dest_dir = self._project_root / layer_dir
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_path = dest_dir / model.filename
                dest_path.write_text(model.sql_content, encoding="utf-8")
                files_created.append(str(dest_path))
                pipeline_models.append(model.model_name)
                logger.info("Installed model: %s → %s", model.model_name, dest_path)

            # Register in extension model registry
            registry = _load_extension_model_registry()
            existing_names = {m["name"] for m in registry}
            for model in preview.models:
                if model.model_name not in existing_names:
                    registry.append({
                        "name": model.model_name,
                        "table_path": model.model_name,
                        "connector_id": connector_id,
                    })
            _save_extension_model_registry(registry)

            # Apply graph extensions if present
            graph_applied = False
            if preview.graph_extension is not None:
                graph_applied = self._apply_graph_extensions(
                    preview.graph_extension,
                )

            # Clean up staging directory
            staging_dir = _EXTENSIONS_BASE / connector_id / "generated"
            if staging_dir.exists():
                shutil.rmtree(staging_dir)

            return ModelResult(
                status="installed",
                models_installed=len(pipeline_models),
                files_created=tuple(files_created),
                pipeline_models_added=tuple(pipeline_models),
                graph_extensions_applied=graph_applied,
            )

        except Exception as exc:
            logger.error(
                "Model installation failed: %s", exc, exc_info=True,
            )
            return ModelResult(
                status="error",
                models_installed=0,
                files_created=tuple(files_created),
                pipeline_models_added=tuple(pipeline_models),
                graph_extensions_applied=False,
                error=str(exc),
            )

    def reject(self, connector_id: str) -> None:
        """Remove staged models without installing.

        Args:
            connector_id: The connector ID to reject.

        sensitivity_tier: 1
        """
        staging_dir = _EXTENSIONS_BASE / connector_id / "generated"
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
            logger.info("Rejected staged models for %s", connector_id)

    def dry_run_validate(
        self,
        model: GeneratedModel,
    ) -> tuple[bool, str | None]:
        """Validate generated SQL without committing.

        Uses DuckDB EXPLAIN to check syntax when a database engine is
        available. Falls back to basic structural checks.

        Args:
            model: The generated model to validate.

        Returns:
            Tuple of (is_valid, error_message_or_None).

        sensitivity_tier: 1
        """
        sql = model.sql_content

        # Basic structural checks
        if "MODEL" not in sql:
            return (False, "Missing MODEL declaration")
        if "SELECT" not in sql:
            return (False, "Missing SELECT statement")
        if "kind FULL" not in sql:
            return (False, "Missing 'kind FULL' in MODEL declaration")

        # Check ext_ prefix in model name
        basename = model.model_name.split(".")[-1]
        if not basename.startswith("ext_"):
            return (False, f"Model name '{model.model_name}' missing ext_ prefix")

        # DuckDB EXPLAIN validation if engine is available
        if self._db_engine is not None:
            # Extract the SELECT statement (everything after the semicolon
            # following the MODEL block)
            select_sql = self._extract_select(sql)
            if select_sql:
                try:
                    self._db_engine.execute(f"EXPLAIN {select_sql}")
                except Exception as exc:
                    return (False, f"SQL validation failed: {exc}")

        return (True, None)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_select(self, sql: str) -> str | None:
        """Extract the SELECT statement from a SQLMesh model SQL.

        sensitivity_tier: 1
        """
        # Find the closing ); of the MODEL block, then take everything after
        model_end = sql.find(");")
        if model_end == -1:
            return None
        remainder = sql[model_end + 2:].strip()
        if remainder.upper().startswith("SELECT"):
            return remainder
        return None

    def _apply_graph_extensions(
        self,
        graph: GraphExtension,
    ) -> bool:
        """Apply Kuzu graph schema extensions.

        Returns True if extensions were applied successfully.

        sensitivity_tier: 1
        """
        try:
            from src.core.kuzu.engine import GraphEngine

            engine = GraphEngine()
            for ddl in graph.node_ddl:
                engine.execute(ddl)
            for ddl in graph.relationship_ddl:
                engine.execute(ddl)
            logger.info(
                "Applied graph extensions: nodes=%s, rels=%s",
                graph.node_names,
                graph.relationship_names,
            )
            return True
        except Exception:
            logger.warning(
                "Failed to apply graph extensions", exc_info=True,
            )
            return False
