"""File-based skill loader following the SKILL.md open standard.

Skills are folders containing a ``SKILL.md`` file with YAML frontmatter
(metadata) and a Markdown body (instructions). The loader implements
three-level progressive disclosure:

- **L1 (discover)** — name + description only, always in memory.
- **L2 (load)** — full SKILL.md body, read from disk on demand.
- **L3 (resource)** — bundled files beyond SKILL.md, read on demand.

sensitivity_tier: 1 (skill metadata and instructions are infrastructure)
"""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SKILLS_DIR: Path = Path.home() / ".arandu" / "skills"
BUILTIN_SKILLS_DIR: Path = Path(__file__).parent / "builtin_skills"

_FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(.*?)\n---\s*\n",
    re.DOTALL,
)


@dataclass(frozen=True)
class SkillMeta:
    """L1 metadata — lightweight, always in memory.

    sensitivity_tier: 1
    """

    id: str
    name: str
    description: str
    tags: tuple[str, ...] = ()
    sensitivity_tier: int = 1
    source: str = "user"
    version: int = 1
    path: Path = field(default_factory=lambda: Path())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tags": list(self.tags),
            "sensitivity_tier": self.sensitivity_tier,
            "source": self.source,
            "version": self.version,
        }


@dataclass(frozen=True)
class SkillDocument:
    """L2 — full SKILL.md content, loaded on demand.

    sensitivity_tier: 1
    """

    meta: SkillMeta
    instructions: str

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.meta.to_dict(),
            "instructions": self.instructions,
        }


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split YAML frontmatter from markdown body.

    sensitivity_tier: 1
    """
    import yaml  # lazy — avoids import cost at module level

    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text

    raw_yaml = match.group(1)
    body = text[match.end():]
    try:
        meta = yaml.safe_load(raw_yaml)
    except Exception:
        logger.warning("Failed to parse SKILL.md frontmatter")
        return {}, text

    if not isinstance(meta, dict):
        return {}, text
    return meta, body


def _meta_from_frontmatter(
    skill_id: str,
    fm: dict[str, Any],
    skill_path: Path,
) -> SkillMeta:
    """Build a ``SkillMeta`` from parsed frontmatter.

    sensitivity_tier: 1
    """
    tags_raw = fm.get("tags", [])
    if isinstance(tags_raw, str):
        tags_raw = [t.strip() for t in tags_raw.split(",")]
    return SkillMeta(
        id=skill_id,
        name=fm.get("name", skill_id),
        description=fm.get("description", ""),
        tags=tuple(str(t) for t in tags_raw) if tags_raw else (),
        sensitivity_tier=int(fm.get("sensitivity_tier", 1)),
        source=fm.get("source", "user"),
        version=int(fm.get("version", 1)),
        path=skill_path,
    )


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(name: str) -> str:
    return _SLUG_RE.sub("-", name.strip().lower()).strip("-") or "skill"


class SkillLoader:
    """File-based skill registry with progressive disclosure.

    sensitivity_tier: 1
    """

    def __init__(
        self,
        skills_dir: Path | None = None,
        *,
        seed_builtins: bool = True,
    ) -> None:
        self._dir = skills_dir or SKILLS_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, SkillMeta] = {}
        if seed_builtins:
            self._seed_builtins()
        self._scan()

    def _seed_builtins(self) -> None:
        """Copy built-in skills to the user directory and prune removed ones.

        sensitivity_tier: 1
        """
        if not BUILTIN_SKILLS_DIR.is_dir():
            return
        builtin_ids = {
            d.name for d in BUILTIN_SKILLS_DIR.iterdir() if d.is_dir()
        }
        for src_dir in BUILTIN_SKILLS_DIR.iterdir():
            if not src_dir.is_dir():
                continue
            dest = self._dir / src_dir.name
            skill_md = dest / "SKILL.md"
            if skill_md.exists():
                continue
            shutil.copytree(src_dir, dest, dirs_exist_ok=True)
            logger.info("Seeded built-in skill: %s", src_dir.name)
        for dest_dir in list(self._dir.iterdir()):
            if not dest_dir.is_dir():
                continue
            skill_md = dest_dir / "SKILL.md"
            if not skill_md.exists():
                continue
            try:
                fm, _ = _parse_frontmatter(
                    skill_md.read_text(encoding="utf-8"),
                )
            except Exception:
                continue
            if fm.get("source") == "builtin" and dest_dir.name not in builtin_ids:
                shutil.rmtree(dest_dir, ignore_errors=True)
                logger.info("Pruned removed built-in skill: %s", dest_dir.name)

    def _scan(self) -> None:
        """Scan skill directories and cache L1 metadata.

        sensitivity_tier: 1
        """
        self._cache.clear()
        for skill_dir in sorted(self._dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue
            try:
                text = skill_md.read_text(encoding="utf-8")
                fm, _ = _parse_frontmatter(text)
                skill_id = skill_dir.name
                self._cache[skill_id] = _meta_from_frontmatter(
                    skill_id, fm, skill_dir,
                )
            except Exception:
                logger.warning(
                    "Failed to load skill: %s", skill_dir.name,
                    exc_info=True,
                )

    def discover(self) -> list[SkillMeta]:
        """L1: return all skill metadata.

        sensitivity_tier: 1
        """
        return list(self._cache.values())

    def get_meta(self, skill_id: str) -> SkillMeta | None:
        """L1: look up one skill's metadata.

        sensitivity_tier: 1
        """
        return self._cache.get(skill_id)

    def load(self, skill_id: str) -> SkillDocument | None:
        """L2: read and parse the full SKILL.md for one skill.

        sensitivity_tier: 1
        """
        meta = self._cache.get(skill_id)
        if meta is None:
            return None
        skill_md = meta.path / "SKILL.md"
        if not skill_md.exists():
            return None
        text = skill_md.read_text(encoding="utf-8")
        _, body = _parse_frontmatter(text)
        return SkillDocument(meta=meta, instructions=body.strip())

    def load_resource(self, skill_id: str, rel_path: str) -> str | None:
        """L3: read a bundled file within a skill directory.

        sensitivity_tier: 1
        """
        meta = self._cache.get(skill_id)
        if meta is None:
            return None
        target = (meta.path / rel_path).resolve()
        if not str(target).startswith(str(meta.path.resolve())):
            return None
        if not target.is_file():
            return None
        return target.read_text(encoding="utf-8")

    def create(self, name: str, content: str) -> SkillMeta:
        """Create a new skill folder with the given SKILL.md content.

        sensitivity_tier: 1
        """
        slug = _slugify(name)
        skill_dir = self._dir / slug
        n = 2
        while skill_dir.exists():
            skill_dir = self._dir / f"{slug}-{n}"
            n += 1
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
        self._scan()
        skill_id = skill_dir.name
        meta = self._cache.get(skill_id)
        if meta is None:
            msg = f"Failed to register skill after creation: {skill_id}"
            raise RuntimeError(msg)
        return meta

    def update(self, skill_id: str, content: str) -> SkillMeta:
        """Overwrite a skill's SKILL.md content.

        sensitivity_tier: 1
        """
        meta = self._cache.get(skill_id)
        if meta is None:
            msg = f"Unknown skill: {skill_id!r}"
            raise KeyError(msg)
        (meta.path / "SKILL.md").write_text(content, encoding="utf-8")
        self._scan()
        updated = self._cache.get(skill_id)
        if updated is None:
            msg = f"Skill disappeared after update: {skill_id}"
            raise RuntimeError(msg)
        return updated

    def delete(self, skill_id: str) -> bool:
        """Remove a skill directory entirely.

        sensitivity_tier: 1
        """
        meta = self._cache.get(skill_id)
        if meta is None:
            return False
        shutil.rmtree(meta.path, ignore_errors=True)
        self._cache.pop(skill_id, None)
        return True

    def refresh(self) -> None:
        """Re-scan the skills directory.

        sensitivity_tier: 1
        """
        self._scan()

def build_skill_menu(
    *,
    max_tier: int = 3,
    only_ids: tuple[str, ...] | None = None,
) -> str:
    """Build an L1 skill menu for the system prompt.

    Returns only skill names and descriptions (~50 tokens per skill)
    so the LLM can decide which skill to activate via ``load_skill``.
    Full instructions are never injected upfront.

    ``only_ids``: when set, only include skills with these IDs
    (used by user agents with ``enabled_skills``).

    sensitivity_tier: 1
    """
    try:
        loader = SkillLoader()
    except Exception:
        return ""
    skills = loader.discover()
    if only_ids is not None:
        id_set = set(only_ids)
        skills = [s for s in skills if s.id in id_set]
    else:
        skills = [s for s in skills if s.sensitivity_tier <= max_tier]
    if not skills:
        return ""
    lines = [
        "## Available Skills",
        "You have access to the following skills. When a user's request "
        "matches a skill, call `load_skill` with the skill ID to get "
        "the full instructions before proceeding. Do not guess the "
        "procedure — always load the skill first.",
        "",
    ]
    for s in skills:
        tags = f" [{', '.join(s.tags)}]" if s.tags else ""
        lines.append(f"- **{s.id}**: {s.description}{tags}")
    return "\n".join(lines)
