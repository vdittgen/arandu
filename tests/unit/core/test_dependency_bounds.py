"""Guards on dependency bounds that a fresh install depends on.

These aren't style checks. `pydantic-ai` 2.x renamed `OpenAIModel` to
`OpenAIChatModel` and turned `ModelProfile` from a dataclass into a
TypedDict; the model factory uses both. Unbounded, a fresh install
resolves 2.x and *every* pydantic-ai agent fails at construction — with
an error that used to read "pydantic-ai-slim[openai] is not installed",
sending you after a package that is present.

A cap is easy to drop during a routine dependency bump, and nothing else
in the suite would notice: CI installs from a resolved environment, so
the break only shows up for someone installing fresh.

sensitivity_tier: 1
"""

from __future__ import annotations

from pathlib import Path

import pytest
import tomllib

PYPROJECT = Path(__file__).resolve().parents[3] / "pyproject.toml"


@pytest.fixture(scope="module")
def requirements() -> dict[str, list[str]]:
    """All declared requirement strings, grouped by extra ("" = runtime)."""
    data = tomllib.loads(PYPROJECT.read_text())
    project = data["project"]
    groups: dict[str, list[str]] = {"": list(project.get("dependencies", []))}
    for extra, reqs in project.get("optional-dependencies", {}).items():
        groups[extra] = list(reqs)
    return groups


def _dist_name(requirement: str) -> str:
    """Distribution name from a requirement string, minus extras/specifier."""
    return requirement.split("[")[0].split(">")[0].split("=")[0].split(";")[0].strip()


def _find(reqs: list[str], name: str) -> str:
    matches = [r for r in reqs if _dist_name(r) == name]
    assert matches, f"{name} not declared"
    return matches[0]


def test_pydantic_ai_is_capped_below_2(requirements: dict[str, list[str]]) -> None:
    """The model factory targets the 1.x API surface.

    Lift this only together with migrating to `OpenAIChatModel` and the
    `ModelProfile` TypedDict.
    """
    spec = _find(requirements[""], "pydantic-ai-slim")
    assert "<2" in spec, (
        f"pydantic-ai-slim must stay capped below 2.x, got {spec!r}. "
        "2.x renames OpenAIModel and changes ModelProfile; "
        "src/agents/core/model_factory.py depends on both."
    )


def test_pydantic_evals_tracks_pydantic_ai(
    requirements: dict[str, list[str]],
) -> None:
    """pydantic-evals pins pydantic-ai-slim exactly, so it must match.

    Without the bound, a 1.x pair is only reached by resolver
    backtracking rather than by declaration.
    """
    spec = _find(requirements["dev"], "pydantic-evals")
    assert "<2" in spec, (
        f"pydantic-evals must stay capped below 2.x, got {spec!r}. "
        "It pins pydantic-ai-slim exactly (2.x requires 2.x), so an "
        "unbounded dev extra fights the runtime cap."
    )
