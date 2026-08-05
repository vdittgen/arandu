"""SQLite-backed store for user-authored skills.

Skills are prompt templates with ``{{var}}`` placeholders. The
runtime substitutes the agent-supplied arguments into the template
and dispatches the result to the configured LLM provider (when
``uses_llm`` is set) or returns the substituted string directly.

sensitivity_tier: 1
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB_PATH: Path = (
    Path.home() / ".arandu" / "data" / "arandu.sqlite3"
)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS user_skills (
    skill_id        TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    description     TEXT NOT NULL DEFAULT '',
    category        TEXT NOT NULL DEFAULT 'general',
    prompt_template TEXT NOT NULL,
    parameters_json TEXT NOT NULL DEFAULT '{}',
    uses_llm        INTEGER NOT NULL DEFAULT 1,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
)
"""


_SLUG_RE = re.compile(r"[^a-z0-9_-]+")


def make_skill_id(name: str) -> str:
    """Derive a stable ``user.<slug>`` skill id.

    sensitivity_tier: 1
    """
    slug = _SLUG_RE.sub("-", name.strip().lower()).strip("-") or "skill"
    return f"user.{slug}"


@dataclass(frozen=True)
class UserSkillRow:
    """One row of the ``user_skills`` table.

    sensitivity_tier: 1
    """

    skill_id: str
    name: str
    description: str
    category: str
    prompt_template: str
    parameters: dict[str, str]
    uses_llm: bool
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, object]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "prompt_template": self.prompt_template,
            "parameters": dict(self.parameters),
            "uses_llm": self.uses_llm,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class UserSkillUpsert:
    """Mutable input for create / update.

    sensitivity_tier: 1
    """

    name: str
    description: str
    category: str
    prompt_template: str
    parameters: dict[str, str]
    uses_llm: bool = True


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


class UserSkillStore:
    """Read / write helper for ``user_skills``.

    sensitivity_tier: 1
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or DEFAULT_DB_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, isolation_level=None)
        self._conn.execute(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def list_all(self) -> list[UserSkillRow]:
        cur = self._conn.execute(
            """
            SELECT skill_id, name, description, category,
                   prompt_template, parameters_json, uses_llm,
                   created_at, updated_at
            FROM user_skills ORDER BY created_at ASC
            """,
        )
        return [_row_to_skill(r) for r in cur.fetchall()]

    def get(self, skill_id: str) -> UserSkillRow | None:
        cur = self._conn.execute(
            """
            SELECT skill_id, name, description, category,
                   prompt_template, parameters_json, uses_llm,
                   created_at, updated_at
            FROM user_skills WHERE skill_id = ?
            """,
            (skill_id,),
        )
        row = cur.fetchone()
        return _row_to_skill(row) if row is not None else None

    def insert(self, upsert: UserSkillUpsert) -> UserSkillRow:
        skill_id = make_skill_id(upsert.name)
        base = skill_id
        n = 2
        while self.get(skill_id) is not None:
            skill_id = f"{base}-{n}"
            n += 1
        now = _now_iso()
        self._conn.execute(
            """
            INSERT INTO user_skills (
                skill_id, name, description, category,
                prompt_template, parameters_json, uses_llm,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                skill_id,
                upsert.name,
                upsert.description,
                upsert.category,
                upsert.prompt_template,
                json.dumps(dict(upsert.parameters)),
                int(bool(upsert.uses_llm)),
                now,
                now,
            ),
        )
        row = self.get(skill_id)
        assert row is not None  # noqa: S101
        return row

    def update(
        self,
        skill_id: str,
        upsert: UserSkillUpsert,
    ) -> UserSkillRow:
        if self.get(skill_id) is None:
            msg = f"unknown user skill: {skill_id!r}"
            raise KeyError(msg)
        self._conn.execute(
            """
            UPDATE user_skills SET
                name = ?, description = ?, category = ?,
                prompt_template = ?, parameters_json = ?,
                uses_llm = ?, updated_at = ?
            WHERE skill_id = ?
            """,
            (
                upsert.name,
                upsert.description,
                upsert.category,
                upsert.prompt_template,
                json.dumps(dict(upsert.parameters)),
                int(bool(upsert.uses_llm)),
                _now_iso(),
                skill_id,
            ),
        )
        row = self.get(skill_id)
        assert row is not None  # noqa: S101
        return row

    def delete(self, skill_id: str) -> bool:
        cur = self._conn.execute(
            "DELETE FROM user_skills WHERE skill_id = ?", (skill_id,),
        )
        return cur.rowcount > 0


def _row_to_skill(row: tuple) -> UserSkillRow:
    params = json.loads(row[5] or "{}")
    if not isinstance(params, dict):
        params = {}
    return UserSkillRow(
        skill_id=row[0],
        name=row[1],
        description=row[2] or "",
        category=row[3] or "general",
        prompt_template=row[4],
        parameters={str(k): str(v) for k, v in params.items()},
        uses_llm=bool(row[6]),
        created_at=row[7],
        updated_at=row[8],
    )


__all__ = [
    "DEFAULT_DB_PATH",
    "UserSkillRow",
    "UserSkillStore",
    "UserSkillUpsert",
    "make_skill_id",
]
