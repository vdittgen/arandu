"""Placeholder redaction round-trip + registry-backed flow.

Covers both the legacy ``redact`` shim and the new
``redact_with_registry`` entry point. Each test gets an isolated
:class:`RedactionRegistry` so test order doesn't leak placeholder
indices.

sensitivity_tier: N/A
"""

from __future__ import annotations

from pathlib import Path

import pytest
from src.models.redaction_registry import RedactionRegistry
from src.models.redactor import (
    redact_with_registry,
    rehydrate,
)


@pytest.fixture()
def registry(tmp_path: Path) -> RedactionRegistry:
    return RedactionRegistry(path=tmp_path / "redaction.sqlite")


def test_redact_replaces_email_and_phone(registry: RedactionRegistry) -> None:
    text = "Reach Alice at alice@example.com or +1 415 555 1212."
    redacted, mapping = redact_with_registry(text, registry=registry)
    assert "alice@example.com" not in redacted
    assert "+1 415 555 1212" not in redacted
    assert "__EMAIL_1__" in redacted
    assert "__PHONE_1__" in redacted
    assert "Alice" not in redacted
    assert mapping.reverse["__EMAIL_1__"] == "alice@example.com"


def test_redact_stable_within_request(registry: RedactionRegistry) -> None:
    text = "Alice called Alice's office, Alice answered."
    redacted, mapping = redact_with_registry(text, registry=registry)
    assert redacted.count("__PERSON_1__") == 3
    assert mapping.reverse["__PERSON_1__"] == "Alice"


def test_redact_stable_across_requests(registry: RedactionRegistry) -> None:
    """The same raw value must reuse the same placeholder across calls."""
    first, _ = redact_with_registry(
        "Alice called.", registry=registry,
    )
    second, mapping = redact_with_registry(
        "Alice is back.", registry=registry,
    )
    assert "__PERSON_1__" in first
    assert "__PERSON_1__" in second
    assert mapping.reverse["__PERSON_1__"] == "Alice"


def test_redact_skips_common_words(registry: RedactionRegistry) -> None:
    text = "On Monday in January I will travel."
    redacted, _ = redact_with_registry(text, registry=registry)
    assert "Monday" in redacted
    assert "January" in redacted


def test_redact_preserves_money_and_ids(registry: RedactionRegistry) -> None:
    text = "Charge $1,234.56 to account 1234567890."
    redacted, mapping = redact_with_registry(text, registry=registry)
    assert "$1,234.56" not in redacted
    assert "1234567890" not in redacted
    assert "__MONEY_1__" in redacted
    assert "__ID_1__" in redacted
    assert mapping.reverse["__ID_1__"] == "1234567890"


def test_rehydrate_round_trip(registry: RedactionRegistry) -> None:
    text = "Email Bob Smith at bob@x.com about $500."
    redacted, mapping = redact_with_registry(text, registry=registry)
    assert rehydrate(redacted, mapping) == text


def test_rehydrate_passes_unknown_placeholders_through(
    registry: RedactionRegistry,
) -> None:
    text = "see __PERSON_99__ later"
    _, mapping = redact_with_registry("", registry=registry)
    assert rehydrate(text, mapping) == text


def test_registry_rehydrate_handles_bare_placeholder(
    registry: RedactionRegistry,
) -> None:
    """LLMs that strip the ``__`` (treating it as markdown bold)
    emit bare ``PERSON_N``. Registry.rehydrate must still restore
    the original — that's the user-visible "Found 0 matches for
    PERSON_430" bug we're fixing."""
    registry.assign("PERSON", "Amor")
    placeholder = registry.lookup("Amor")
    assert placeholder == "__PERSON_1__"
    # Simulate the LLM emitting the stripped form in its JSON.
    bare = "PERSON_1"
    assert registry.rehydrate(f'{{"to": "{bare}"}}') == '{"to": "Amor"}'


def test_registry_rehydrate_handles_legacy_bracket_placeholder(
    registry: RedactionRegistry,
) -> None:
    """Older session output may still contain the pre-migration
    ``<PERSON_N>`` form. Rehydrate must restore those too."""
    registry.assign("PERSON", "Amor")
    assert registry.rehydrate("hi <PERSON_1>!") == "hi Amor!"


def test_registry_rehydrate_does_not_match_bare_in_unrelated_text(
    registry: RedactionRegistry,
) -> None:
    """Bare ``KIND_N`` matching uses word boundaries so the unrelated
    substring ``PERSON_1`` inside an identifier (e.g. ``PERSON_12``)
    or alphanumeric word doesn't get rehydrated."""
    registry.assign("PERSON", "Amor")
    # PERSON_12 is a different (unregistered) bare token — must not match.
    assert registry.rehydrate("see PERSON_12 later") == "see PERSON_12 later"
    # But surrounded by punctuation it's still a bare match.
    assert registry.rehydrate("[PERSON_1]") == "[Amor]"


def test_registry_compiled_regex_short_circuits_novel_pass(
    registry: RedactionRegistry,
) -> None:
    """Once a value is in the registry the compiled pattern catches it.

    The ``record`` call below precedes a redact_with_registry call that
    doesn't actually match the NAME regex (lowercase), proving pass 1
    fires before pass 2.
    """
    registry.assign("PERSON", "alice")
    redacted, mapping = redact_with_registry(
        "Tell alice the meeting moved.", registry=registry,
    )
    assert "alice" not in redacted
    assert "__PERSON_1__" in redacted
    assert mapping.reverse["__PERSON_1__"] == "alice"


def test_bootstrap_from_graph_registers_distinct_names(
    registry: RedactionRegistry,
) -> None:
    class _StubKuzu:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def query(self, q: str) -> list[dict[str, str]]:
            self.calls.append(q)
            if "Person" in q:
                return [{"name": "Alice"}, {"name": "Bob"}]
            if "Place" in q:
                return [{"name": "Berlin"}]
            return []

    registered = registry.bootstrap_from_graph(_StubKuzu())
    assert registered == 3
    assert registry.lookup("Alice") is not None
    assert registry.lookup("Berlin") is not None
    # Bootstrap is idempotent — re-running registers no new rows.
    assert registry.bootstrap_from_graph(_StubKuzu()) == 0


# ---------------------------------------------------------------------------
# PERSON precision (#82)
# ---------------------------------------------------------------------------


def test_name_separator_does_not_cross_newlines(
    registry: RedactionRegistry,
) -> None:
    """A name must not absorb the next line.

    `\\s+` between name words merged unrelated entities into one alias:
    a profile block registered "Marina Silva\\nUser" as a single PERSON,
    which never recurs verbatim — so the registry accumulated junk and
    stopped matching "Marina Silva" on its own.
    """
    text = "name: Marina Silva\nlocation: Porto Alegre"
    redacted, mapping = redact_with_registry(text, registry=registry)

    assert "Marina Silva" not in redacted
    assert "Porto Alegre" not in redacted
    assert set(mapping.reverse.values()) == {"Marina Silva", "Porto Alegre"}
    assert rehydrate(redacted, mapping) == text


def test_sentence_openers_are_not_people(
    registry: RedactionRegistry,
) -> None:
    """Instruction text must survive redaction unchanged.

    Without this the app's own system prompts came back as
    "__PERSON_1__ in at most three sentences", and "Answer" / "Never"
    entered the persistent registry — after which they were rewritten in
    every later prompt on every path.
    """
    text = (
        "Answer in at most three sentences. Never invent facts. "
        "Remember to check the calendar first."
    )
    redacted, mapping = redact_with_registry(text, registry=registry)

    assert redacted == text
    assert mapping.reverse == {}


def test_opener_is_trimmed_not_swallowed(
    registry: RedactionRegistry,
) -> None:
    """"Call Marina Silva" must redact the name, not the whole phrase.

    The pattern happily matches the verb as the first word of a name.
    Rejecting such a match would leave a real name exposed; accepting it
    would register "Call Marina Silva" as the alias, after which the
    bare name never matches again. So the opener is trimmed off.
    """
    redacted, mapping = redact_with_registry(
        "Call Marina Silva about the review", registry=registry,
    )

    assert redacted.startswith("Call ")
    assert "Marina Silva" not in redacted
    assert list(mapping.reverse.values()) == ["Marina Silva"]

    # ...and the alias is the bare name, so it matches on its own later.
    again, mapping2 = redact_with_registry(
        "Marina Silva replied", registry=registry,
    )
    assert "Marina Silva" not in again
    assert list(mapping.reverse) == list(mapping2.reverse)


def test_name_opening_a_sentence_is_still_redacted(
    registry: RedactionRegistry,
) -> None:
    """Position alone must never suppress redaction.

    Skipping every sentence-initial capitalised word would silently stop
    redacting a first name that happens to open a sentence. Over-
    redaction is a quality problem; this direction is a leak.
    """
    redacted, mapping = redact_with_registry(
        "Marina called me yesterday", registry=registry,
    )

    assert "Marina" not in redacted
    assert "Marina" in mapping.reverse.values()


def test_surname_after_an_abbreviation_is_still_redacted(
    registry: RedactionRegistry,
) -> None:
    """"Dr. Alvarez" — the abbreviation dot reads as a sentence end.

    Gating the skip on position alone leaked the surname; gating it on a
    known-opener list keeps it redacted.
    """
    redacted, mapping = redact_with_registry(
        "The therapist is Dr. Alvarez", registry=registry,
    )

    assert "Alvarez" not in redacted
    assert "Alvarez" in " ".join(mapping.reverse.values())


def test_labelled_value_is_redacted_whole(
    registry: RedactionRegistry,
) -> None:
    """A colon introduces a value, not a new sentence.

    Treating it as a sentence start trimmed "Porto" off the front of
    "Porto Alegre" and registered half a place name. Labelled values are
    exactly the shape `build_user_context()` emits.
    """
    redacted, mapping = redact_with_registry(
        "User's location: Porto Alegre", registry=registry,
    )

    assert "Porto Alegre" in mapping.reverse.values()
    assert "Porto" not in redacted
