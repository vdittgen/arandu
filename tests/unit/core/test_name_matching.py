"""Tests for whole-word contact/name matching.

Regression cover for the substring-matching bug that resolved a
mention of one person to a different contact.

sensitivity_tier: 1 (no user data)
"""

from __future__ import annotations

from src.core.name_matching import (
    build_unique_index,
    contains_name,
    find_names,
)


class TestContainsName:
    def test_matches_whole_word(self) -> None:
        assert contains_name("ana", "I called Ana today")

    def test_does_not_match_inside_longer_name(self) -> None:
        # The core bug: "ana" must not match inside "Mariana".
        assert not contains_name("ana", "I called Mariana today")
        assert not contains_name("ana", "Susana sent the deck")

    def test_does_not_match_inside_common_word(self) -> None:
        assert not contains_name("ana", "run the analysis")
        assert not contains_name("ana", "bought a banana")

    def test_is_case_insensitive(self) -> None:
        assert contains_name("Ana", "spoke to ana")
        assert contains_name("ana", "spoke to ANA")

    def test_matches_across_punctuation(self) -> None:
        assert contains_name("ana", "thanks, Ana!")
        assert contains_name("ana", "(Ana)")

    def test_handles_accented_names(self) -> None:
        assert contains_name("josé", "ping José about it")
        # A boundary must exist even with non-ASCII neighbours.
        assert not contains_name("jos", "ping José about it")

    def test_handles_names_with_punctuation(self) -> None:
        # re.escape + lookaround, so an apostrophe/hyphen is literal.
        assert contains_name("O'Brien", "ask O'Brien")
        assert contains_name("Ana-Paula", "ask Ana-Paula")

    def test_empty_inputs_never_match(self) -> None:
        assert not contains_name("", "anything")
        assert not contains_name("ana", "")


class TestFindNames:
    def test_returns_longest_match_first(self) -> None:
        found = find_names(["ana", "ana paula"], "ana paula called")
        assert found[0] == "ana paula"

    def test_filters_non_matches(self) -> None:
        found = find_names(["ana", "bob"], "ana called")
        assert found == ["ana"]

    def test_deduplicates_input(self) -> None:
        assert find_names(["ana", "ana"], "ana called") == ["ana"]

    def test_ordering_is_deterministic_for_equal_lengths(self) -> None:
        # Same length -> alphabetical, so callers get a stable result
        # instead of dict/set iteration order.
        assert find_names(["bob", "ana"], "ana and bob") == ["ana", "bob"]


class TestBuildUniqueIndex:
    def test_keeps_unambiguous_variants(self) -> None:
        idx = build_unique_index({"ana": {"p1"}, "bob": {"p2"}})
        assert idx == {"ana": "p1", "bob": "p2"}

    def test_drops_variants_claimed_by_two_people(self) -> None:
        # Two contacts named Ana: the bare token identifies neither, so
        # it must resolve to nobody rather than to whoever came first.
        idx = build_unique_index({"ana": {"p1", "p2"}, "silva": {"p1"}})
        assert "ana" not in idx
        assert idx["silva"] == "p1"

    def test_empty_input(self) -> None:
        assert build_unique_index({}) == {}
