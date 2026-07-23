"""Tests for post-chat-turn fact learning wired into the CLI.

Issue #37: the fact learner had zero production callers, so
``_learned_facts`` stayed empty forever. These tests cover the two new
seams: ``_emit_stream`` now returns the assistant's plain-text reply,
and ``_maybe_learn_facts`` persists facts from a finished turn.

The LLM step is stubbed at ``FactExtractorAgent.extract`` (same pattern
as tests/unit/agents/test_fact_learner.py) so the real ``FactLearner``
persistence path runs against a temp DuckDB and we can assert the table
is actually populated — the issue's core acceptance criterion.

sensitivity_tier: N/A — test infrastructure
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from src.agents.core.output_types import LearnedFactBatch, LearnedFactDraft
from src.core import cli as cli_mod
from src.core.sqlite.engine import DatabaseEngine


@pytest.fixture()
def tmp_db(tmp_path: Path) -> DatabaseEngine:
    engine = DatabaseEngine(db_path=tmp_path / "test_cli_facts.duckdb")
    yield engine
    engine.close()


@pytest.fixture()
def layer(tmp_db: DatabaseEngine) -> SimpleNamespace:
    """Minimal DataLayer stand-in exposing only ``.duckdb``."""
    return SimpleNamespace(duckdb=tmp_db)


def _draft(content: str = "User's favorite food is sushi") -> LearnedFactDraft:
    return LearnedFactDraft(
        category="preference",
        subject="self",
        predicate="favorite_food",
        content=content,
        sensitivity_tier=1,
    )


@pytest.fixture()
def stub_extract(monkeypatch):
    """Stub ``FactExtractorAgent.extract`` with a controllable batch."""
    fake = MagicMock(return_value=LearnedFactBatch(facts=[_draft()]))

    def _bound(self, conversation):  # noqa: ARG001
        result = fake(conversation)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(
        "src.agents.fact_extractor.agent.FactExtractorAgent.extract",
        _bound,
    )
    return fake


@pytest.fixture(autouse=True)
def _facts_enabled(monkeypatch):
    """Default the fact-learning setting on (isolate from real settings.json)."""
    monkeypatch.setattr(
        "src.models.llm_provider.load_llm_settings",
        lambda: {"learn_facts_from_chat": True},
    )


# ================================================================
# _maybe_learn_facts — the single production writer of _learned_facts
# ================================================================


def _active_facts(db: DatabaseEngine) -> list[dict]:
    return db.query(
        "SELECT content FROM _learned_facts "
        "WHERE dismissed_at IS NULL AND superseded_by IS NULL"
    )


class TestMaybeLearnFacts:
    def test_populates_learned_facts_table(self, layer, stub_extract) -> None:
        """The acceptance criterion: a chat turn writes into _learned_facts."""
        cli_mod._maybe_learn_facts(
            layer,
            "My favorite food is sushi.",
            "Great, I'll remember you love sushi!",
        )
        rows = _active_facts(layer.duckdb)
        assert len(rows) == 1
        assert rows[0]["content"] == "User's favorite food is sushi"

    def test_forwards_both_messages_to_extractor(
        self, layer, stub_extract,
    ) -> None:
        cli_mod._maybe_learn_facts(layer, "I live in Berlin.", "Noted!")
        conversation = stub_extract.call_args.args[0]
        assert "I live in Berlin." in conversation
        assert "Noted!" in conversation

    def test_empty_assistant_text_is_noop(self, layer, stub_extract) -> None:
        # No extractor call, no table touched — returns before building
        # the FactLearner at all.
        cli_mod._maybe_learn_facts(layer, "Hello", None)
        cli_mod._maybe_learn_facts(layer, "Hello", "")
        stub_extract.assert_not_called()

    def test_empty_user_message_is_noop(self, layer, stub_extract) -> None:
        cli_mod._maybe_learn_facts(layer, "", "Some reply")
        stub_extract.assert_not_called()

    def test_disabled_setting_skips_extraction(
        self, layer, stub_extract, monkeypatch,
    ) -> None:
        monkeypatch.setattr(
            "src.models.llm_provider.load_llm_settings",
            lambda: {"learn_facts_from_chat": False},
        )
        cli_mod._maybe_learn_facts(layer, "I like tea.", "Nice.")
        stub_extract.assert_not_called()

    def test_extractor_failure_is_swallowed(
        self, layer, stub_extract,
    ) -> None:
        """A failing LLM extraction must never break the chat turn."""
        stub_extract.side_effect = RuntimeError("LLM down")
        # Must not raise.
        cli_mod._maybe_learn_facts(layer, "I like tea.", "Nice.")
        assert _active_facts(layer.duckdb) == []

    def test_enabled_by_default_when_setting_absent(
        self, layer, stub_extract, monkeypatch,
    ) -> None:
        """Existing installs (no key in settings.json) opt in automatically."""
        monkeypatch.setattr(
            "src.models.llm_provider.load_llm_settings",
            lambda: {},
        )
        cli_mod._maybe_learn_facts(layer, "I use vim.", "Cool.")
        assert len(_active_facts(layer.duckdb)) == 1


# ================================================================
# _emit_stream — now returns the assistant reply text
# ================================================================


class TestEmitStreamReturn:
    def test_returns_accumulated_text(self, capsys) -> None:
        chunks = [
            {"type": "context", "sources": []},
            {"type": "token", "token": "Hello"},
            {"type": "token", "token": " world"},
            {"type": "done", "model": "m", "latency_ms": 5},
        ]
        text = cli_mod._emit_stream(iter(chunks))
        assert text == "Hello world"

    def test_returns_none_on_error_chunk(self, capsys) -> None:
        chunks = [
            {"type": "token", "token": "partial"},
            {"type": "error", "error": "boom"},
        ]
        assert cli_mod._emit_stream(iter(chunks)) is None

    def test_returns_none_when_no_text(self, capsys) -> None:
        chunks = [{"type": "done", "model": "m"}]
        assert cli_mod._emit_stream(iter(chunks)) is None

    def test_accumulates_text_even_without_session(self, capsys) -> None:
        """Text is returned for unsaved asks (no layer/session) too."""
        chunks = [{"type": "token", "token": "hi"}]
        assert cli_mod._emit_stream(iter(chunks), layer=None) == "hi"
