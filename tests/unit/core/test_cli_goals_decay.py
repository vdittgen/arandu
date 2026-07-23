"""CLI-level tests for goal decay ranking + per-project digest (#41).

Exercises ``cmd_goals_list`` (decay enrichment + decay-aware ordering)
and ``cmd_projects_digest`` (per-project rollup) against a real
DataLayer.

sensitivity_tier: N/A — test infrastructure
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from src.agents.tasks import TaskCurator
from src.agents.tasks.decay import ARCHIVE_AFTER_DAYS
from src.agents.tasks.persistence import update_goal_fields
from src.core import cli as cli_mod
from src.core.data_layer import DataLayer


@pytest.fixture()
def layer(tmp_path: Path) -> DataLayer:
    lyr = DataLayer(base_path=tmp_path / "data")
    lyr.warmup()
    yield lyr
    lyr.close()


def _stdout_json(capsys):
    return json.loads(capsys.readouterr().out)


# ================================================================
# cmd_goals_list — decay enrichment + ranking
# ================================================================


class TestGoalsListDecay:
    def test_enriches_with_decay_state(self, layer, capsys) -> None:
        curator = TaskCurator(db_engine=layer.duckdb)
        fresh = curator.create_goal(
            title="Ship v2", category="work", source="brain",
        )
        stale = curator.create_goal(
            title="Learn cello", category="work", source="brain",
        )
        old = (datetime.now(timezone.utc)
               - timedelta(days=ARCHIVE_AFTER_DAYS - 2)).isoformat()
        update_goal_fields(layer.duckdb, stale.id, last_confirmed_at=old)

        assert cli_mod.cmd_goals_list(layer, "active", None) == 0
        rows = {r["id"]: r for r in _stdout_json(capsys)}

        assert rows[fresh.id]["decay_state"] == "fresh"
        assert rows[stale.id]["decay_state"] == "fading"
        assert "days_since_confirmed" in rows[stale.id]
        # The fading goal is penalised below the fresh one when their
        # base urgency ties (both have no tasks/dates).
        assert (
            rows[stale.id]["urgency_score"] < rows[fresh.id]["urgency_score"]
        )

    def test_user_goal_stays_fresh(self, layer, capsys) -> None:
        curator = TaskCurator(db_engine=layer.duckdb)
        g = curator.create_goal(
            title="Marathon", category="personal", source="user",
        )
        old = (datetime.now(timezone.utc)
               - timedelta(days=ARCHIVE_AFTER_DAYS + 40)).isoformat()
        update_goal_fields(layer.duckdb, g.id, last_confirmed_at=old)

        cli_mod.cmd_goals_list(layer, "active", None)
        rows = {r["id"]: r for r in _stdout_json(capsys)}
        assert rows[g.id]["decay_state"] == "fresh"


# ================================================================
# cmd_projects_digest — per-project rollup
# ================================================================


class TestProjectsDigest:
    def test_rolls_up_task_progress_per_project(self, layer, capsys) -> None:
        curator = TaskCurator(db_engine=layer.duckdb)
        goal = curator.create_goal(title="Website", category="work")
        proj = curator.create_project(
            name="Launch site", category="work", goal_id=goal.id,
        )
        t1 = curator.create_task(title="Design", project_id=proj.id)
        curator.create_task(title="Build", project_id=proj.id)
        curator.toggle_task_done(t1.id)  # 1 of 2 done

        assert cli_mod.cmd_projects_digest(layer, None) == 0
        digest = {d["project_id"]: d for d in _stdout_json(capsys)}

        row = digest[proj.id]
        assert row["name"] == "Launch site"
        assert row["goal_title"] == "Website"
        assert row["task_total"] == 2
        assert row["task_done"] == 1
        assert row["task_open"] == 1
        assert row["progress_pct"] == 50

    def test_sorted_by_attention_needed(self, layer, capsys) -> None:
        curator = TaskCurator(db_engine=layer.duckdb)
        calm = curator.create_project(name="Calm", category="work")
        busy = curator.create_project(name="Busy", category="work")
        curator.create_task(title="a", project_id=busy.id)
        curator.create_task(title="b", project_id=busy.id)
        curator.create_task(title="c", project_id=calm.id)

        cli_mod.cmd_projects_digest(layer, None)
        digest = _stdout_json(capsys)
        order = [d["name"] for d in digest if d["name"] in ("Busy", "Calm")]
        # Busy (more open work) ranks before Calm.
        assert order.index("Busy") < order.index("Calm")
