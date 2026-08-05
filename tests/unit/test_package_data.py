"""Every runtime asset under ``src/`` must be declared as package data.

setuptools ships only ``*.py`` unless told otherwise. When this was
missing, ``pip install`` produced a silently incomplete package — no
prompt templates, no SQLMesh models, no WhatsApp node bridge — and
nothing surfaced it, because the desktop bundle puts the source tree on
``sys.path`` ahead of site-packages. Only code importing ``src.*`` from
an installed copy ever hit it.

This is a static check on ``pyproject.toml`` rather than a build, so it
runs in milliseconds and fails the moment someone adds an asset of a
new type to a packaged directory.

sensitivity_tier: N/A
"""

from __future__ import annotations

from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"

# Extensions that are source code or editor/VCS bookkeeping, not runtime
# assets the Python package needs.
_NOT_ASSETS = {".py", ".pyc", ".tsx", ".ts", ".css", ".md", ".gitkeep",
               ".gitignore", ".html"}


def _declared_extensions() -> set[str]:
    with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
        cfg = tomllib.load(fh)
    package_data = (
        cfg.get("tool", {}).get("setuptools", {}).get("package-data", {})
    )
    assert package_data, (
        "pyproject.toml declares no [tool.setuptools.package-data]; "
        "pip install would ship only *.py"
    )
    return {
        Path(pattern).suffix
        for patterns in package_data.values()
        for pattern in patterns
    }


def _packaged_dirs() -> list[Path]:
    """Directories setuptools will treat as packages (have __init__.py)."""
    return [p.parent for p in SRC.rglob("__init__.py")]


def test_every_runtime_asset_extension_is_declared() -> None:
    declared = _declared_extensions()
    found: dict[str, Path] = {}
    for pkg in _packaged_dirs():
        for asset in pkg.iterdir():
            if not asset.is_file():
                continue
            suffix = asset.suffix
            if suffix and suffix not in _NOT_ASSETS:
                found.setdefault(suffix, asset)

    undeclared = {
        ext: path.relative_to(REPO_ROOT)
        for ext, path in found.items()
        if ext not in declared
    }
    assert not undeclared, (
        "runtime assets in packaged dirs whose extension is not in "
        f"[tool.setuptools.package-data]: {undeclared}. pip install "
        "would drop them and the failure only shows up when running "
        "from an installed copy."
    )


def test_prompt_templates_are_declared() -> None:
    """The specific regression: 18 frozen prompts silently not shipping."""
    prompts = list((SRC / "models" / "prompts").glob("*.txt"))
    assert prompts, "no prompt templates found — did they move?"
    assert ".txt" in _declared_extensions()


def test_sqlmesh_models_are_declared() -> None:
    """Same class of bug, different asset: the pipeline's .sql models."""
    models = list((SRC / "pipeline").rglob("*.sql"))
    assert models, "no SQLMesh models found — did they move?"
    assert ".sql" in _declared_extensions()
