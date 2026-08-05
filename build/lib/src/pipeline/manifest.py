"""Pipeline manifest loader and dependency resolver.

Loads the pipeline manifest JSON and resolves model execution order via
topological sort based on declared dependencies.

sensitivity_tier: N/A (infrastructure)
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "pipeline_manifest.json"
_EXTENSION_MODELS_PATH = (
    Path.home() / ".arandu" / "data" / "extension_models.json"
)


@dataclass(frozen=True)
class AuditSpec:
    """Audit checks for a single model.

    sensitivity_tier: N/A
    """

    not_null: list[str] = field(default_factory=list)
    unique: list[str] = field(default_factory=list)
    accepted_values: dict[str, list[Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelSpec:
    """Specification for a single pipeline model.

    sensitivity_tier: N/A
    """

    name: str
    layer: str
    depends_on: list[str] = field(default_factory=list)
    grain: list[str] = field(default_factory=list)
    audits: AuditSpec = field(default_factory=AuditSpec)

    # SQL models
    sql_file: str | None = None

    # Python models
    model_type: str = "sql"  # "sql" or "python"
    python_module: str | None = None
    python_function: str | None = None


@dataclass
class PipelineManifest:
    """Full pipeline manifest with all model specs.

    sensitivity_tier: N/A
    """

    version: int
    models: list[ModelSpec]
    _by_name: dict[str, ModelSpec] = field(
        default_factory=dict, repr=False,
    )

    def __post_init__(self) -> None:
        self._by_name = {m.name: m for m in self.models}

    def get_model(self, name: str) -> ModelSpec | None:
        """Return model spec by name, or None."""
        return self._by_name.get(name)

    @property
    def model_names(self) -> list[str]:
        """All model names in manifest order."""
        return [m.name for m in self.models]


def _parse_audit(raw: dict[str, Any]) -> AuditSpec:
    """Parse audit section from manifest JSON.

    sensitivity_tier: N/A
    """
    return AuditSpec(
        not_null=raw.get("not_null", []),
        unique=raw.get("unique", []),
        accepted_values=raw.get("accepted_values", {}),
    )


def _parse_model(raw: dict[str, Any]) -> ModelSpec:
    """Parse a single model entry from manifest JSON.

    sensitivity_tier: N/A
    """
    model_type = raw.get("type", "sql")
    return ModelSpec(
        name=raw["name"],
        layer=raw["layer"],
        depends_on=raw.get("depends_on", []),
        grain=raw.get("grain", []),
        audits=_parse_audit(raw.get("audits", {})),
        sql_file=raw.get("sql_file"),
        model_type=model_type,
        python_module=raw.get("python_module"),
        python_function=raw.get("python_function"),
    )


def load_manifest(
    path: Path = DEFAULT_MANIFEST_PATH,
) -> PipelineManifest:
    """Load the pipeline manifest from JSON.

    Also merges any registered extension models from
    ``~/.arandu/data/extension_models.json``.

    Args:
        path: Path to the manifest JSON file.

    Returns:
        A PipelineManifest with all core + extension models.

    Raises:
        FileNotFoundError: If the manifest file doesn't exist.

    sensitivity_tier: N/A
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    models = [_parse_model(m) for m in data.get("models", [])]

    # Merge extension models
    ext_models = _load_extension_models()
    existing_names = {m.name for m in models}
    for ext in ext_models:
        if ext.name not in existing_names:
            models.append(ext)

    return PipelineManifest(
        version=data.get("version", 1),
        models=models,
    )


def _load_extension_models() -> list[ModelSpec]:
    """Load registered extension models from disk.

    sensitivity_tier: 1
    """
    if not _EXTENSION_MODELS_PATH.exists():
        return []
    try:
        data = json.loads(
            _EXTENSION_MODELS_PATH.read_text(encoding="utf-8"),
        )
        return [_parse_model(m) for m in data.get("models", [])]
    except (json.JSONDecodeError, OSError):
        logger.warning(
            "Failed to read extension models registry", exc_info=True,
        )
        return []


def topological_sort(models: list[ModelSpec]) -> list[ModelSpec]:
    """Sort models in dependency order (Kahn's algorithm).

    Models whose dependencies are all raw tables (not in the model list)
    come first.  Raises ValueError on circular dependencies.

    Args:
        models: List of model specs to sort.

    Returns:
        Models sorted so that every model comes after its dependencies.

    Raises:
        ValueError: If circular dependencies are detected.

    sensitivity_tier: N/A
    """
    model_names = {m.name for m in models}
    by_name = {m.name: m for m in models}

    # Build adjacency list: only track deps that are other models
    in_degree: dict[str, int] = defaultdict(int)
    dependents: dict[str, list[str]] = defaultdict(list)

    for m in models:
        in_degree.setdefault(m.name, 0)
        for dep in m.depends_on:
            if dep in model_names:
                in_degree[m.name] += 1
                dependents[dep].append(m.name)

    # Start with models that have no model-level dependencies
    queue: deque[str] = deque(
        name for name, degree in in_degree.items() if degree == 0
    )
    result: list[ModelSpec] = []

    while queue:
        name = queue.popleft()
        result.append(by_name[name])
        for dependent in dependents.get(name, []):
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if len(result) != len(models):
        remaining = model_names - {m.name for m in result}
        msg = f"Circular dependency detected among: {remaining}"
        raise ValueError(msg)

    return result


def resolve_execution_order(
    manifest: PipelineManifest,
    select_models: list[str] | None = None,
) -> list[ModelSpec]:
    """Return models in execution order, optionally filtered.

    When *select_models* is provided, includes only those models and
    their transitive dependencies.

    Args:
        manifest: The full pipeline manifest.
        select_models: Optional list of model names to include.

    Returns:
        Topologically sorted list of ModelSpec.

    sensitivity_tier: N/A
    """
    if select_models is None:
        return topological_sort(manifest.models)

    # Collect transitive dependencies
    by_name = {m.name: m for m in manifest.models}
    needed: set[str] = set()
    stack = list(select_models)

    while stack:
        name = stack.pop()
        if name in needed:
            continue
        needed.add(name)
        model = by_name.get(name)
        if model:
            for dep in model.depends_on:
                if dep in by_name:
                    stack.append(dep)

    selected = [m for m in manifest.models if m.name in needed]
    return topological_sort(selected)
