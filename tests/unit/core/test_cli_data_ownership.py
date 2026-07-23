"""Tests for Export data / Delete-my-data (Settings › Privacy).

Issue #42: both actions were disabled "Soon" buttons. These cover the
CLI backend — export dumps every SQLite table to an inspectable zip,
delete wipes the data directory — against a real temp DataLayer.

sensitivity_tier: N/A — test infrastructure
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from src.core import cli as cli_mod
from src.core.data_layer import DataLayer


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    """A DataLayer base dir seeded with one row across two tables."""
    base = tmp_path / "data"
    with DataLayer(base_path=base) as layer:
        db = layer.duckdb
        db.execute("CREATE TABLE raw_notes (id VARCHAR, body VARCHAR)")
        db.execute(
            "INSERT INTO raw_notes VALUES (?, ?)", ["n1", "buy milk"],
        )
        db.execute("CREATE TABLE _goals (id VARCHAR, title VARCHAR)")
        db.execute(
            "INSERT INTO _goals VALUES (?, ?)", ["g1", "ship export"],
        )
    return base


# ================================================================
# export-data
# ================================================================


class TestExportData:
    def test_produces_zip_with_all_tables_and_manifest(
        self, data_dir, tmp_path, capsys,
    ) -> None:
        out = tmp_path / "out"
        with DataLayer(base_path=data_dir, read_only=True) as layer:
            code = cli_mod.cmd_export_data(layer, str(out))
        assert code == 0

        result = json.loads(capsys.readouterr().out)
        assert result["ok"] is True
        archive = Path(result["path"])
        assert archive.exists() and archive.suffix == ".zip"
        assert result["tables"] >= 2
        assert result["total_rows"] >= 2

        with zipfile.ZipFile(archive) as zf:
            names = set(zf.namelist())
            assert "manifest.json" in names
            assert "tables/raw_notes.json" in names
            assert "tables/_goals.json" in names

            manifest = json.loads(zf.read("manifest.json"))
            assert manifest["app"] == "arandu"
            assert manifest["format_version"] == 1
            assert manifest["tables"]["raw_notes"] == 1
            assert manifest["tables"]["_goals"] == 1

            notes = json.loads(zf.read("tables/raw_notes.json"))
            assert notes == [{"id": "n1", "body": "buy milk"}]

    def test_defaults_output_dir_when_unspecified(
        self, data_dir, monkeypatch, tmp_path, capsys,
    ) -> None:
        fake_dir = tmp_path / "downloads"
        fake_dir.mkdir()
        monkeypatch.setattr(cli_mod, "_default_export_dir", lambda: fake_dir)
        with DataLayer(base_path=data_dir, read_only=True) as layer:
            cli_mod.cmd_export_data(layer, None)
        result = json.loads(capsys.readouterr().out)
        assert Path(result["path"]).parent == fake_dir


# ================================================================
# delete-all-data
# ================================================================


class TestDeleteAllData:
    def test_wipes_the_data_directory(self, data_dir, capsys) -> None:
        # Precondition: the SQLite store exists on disk.
        assert (data_dir / "arandu.sqlite3").exists()

        code = cli_mod.cmd_delete_all_data(str(data_dir))
        assert code == 0

        result = json.loads(capsys.readouterr().out)
        assert result["ok"] is True
        assert result["existed"] is True
        # The store is gone…
        assert not (data_dir / "arandu.sqlite3").exists()
        # …and an empty dir is left so the next launch initializes clean.
        assert data_dir.is_dir()
        assert list(data_dir.iterdir()) == []

    def test_missing_dir_is_not_an_error(self, tmp_path, capsys) -> None:
        target = tmp_path / "never_existed"
        code = cli_mod.cmd_delete_all_data(str(target))
        assert code == 0
        result = json.loads(capsys.readouterr().out)
        assert result["ok"] is True
        assert result["existed"] is False
        assert target.is_dir()

    def test_leaves_sibling_settings_untouched(
        self, tmp_path, capsys,
    ) -> None:
        """Only the data dir is wiped; ~/.arandu/settings.json survives."""
        arandu = tmp_path / ".arandu"
        data = arandu / "data"
        data.mkdir(parents=True)
        (data / "arandu.sqlite3").write_text("db")
        settings = arandu / "settings.json"
        settings.write_text('{"user_name": "Vinicius"}')

        cli_mod.cmd_delete_all_data(str(data))

        assert not (data / "arandu.sqlite3").exists()
        assert settings.exists()
        assert settings.read_text() == '{"user_name": "Vinicius"}'

    @pytest.mark.parametrize("kind", ["home", "cwd", "root"])
    def test_refuses_unsafe_paths(self, kind, monkeypatch, capsys) -> None:
        """An irreversible delete must never target $HOME, CWD, or /.

        A bare ``--data-dir ""`` parses to ``Path(".")`` at the argparse
        layer, i.e. the "cwd" case (the app install dir), so it's
        guarded too.
        """
        home = Path.home()
        targets = {
            "home": str(home),
            "cwd": ".",
            "root": "/",
        }
        # Guard against the test itself doing damage: rmtree must never
        # be reached for these inputs.
        import shutil

        monkeypatch.setattr(
            shutil,
            "rmtree",
            lambda *a, **k: pytest.fail("rmtree must not run on unsafe path"),
        )

        code = cli_mod.cmd_delete_all_data(targets[kind])
        assert code == 0
        result = json.loads(capsys.readouterr().out)
        assert result["ok"] is False
        assert "unsafe path" in result["error"]
        # The guarded locations are all still present.
        assert home.exists()
