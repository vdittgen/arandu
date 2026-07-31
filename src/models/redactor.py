"""Deterministic placeholder redaction for redact-then-remote egress.

Replaces high-signal entities (people names, money amounts, phone
numbers, emails, dates, account-like digit runs) with stable
placeholders before a prompt leaves the device, and rehydrates the
response on return.

Two layers, both deterministic:

1. **Known-entity pass.** A compiled regex built from the persistent
   :class:`RedactionRegistry` replaces every previously-seen raw
   value with its stable placeholder. The pipeline pre-populates the
   registry with names/places from Kuzu so most entities are caught
   here, even on the first prompt of a session.
2. **Novel-entity pass.** The :data:`_ENTITY_PATTERNS` regex catches
   emails / phone / money / digit-runs / dates / capitalised names
   the registry has not yet seen. Each new match is written back to
   the registry so the next prompt's pass-1 picks it up.

The hot path does **not** call an LLM. A local-model NER would
reintroduce the latency that drove the privacy-strategy rework in
the first place.

sensitivity_tier: 1 (output) / 3 (registry holds raw values)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.models.redaction_registry import (
    RedactionRegistry,
    default_redaction_registry,
)

_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
# Phone numbers must carry a + prefix or include a separator so a long
# undecorated digit run is treated as an opaque ID instead.
_PHONE_RE = re.compile(
    r"(?:\+\d[\d\-\s]{7,}\d|\b\d{1,3}[\-\s]\d{2,}[\-\s]\d{2,}(?:[\-\s]\d{2,})?\b)",
)
_MONEY_RE = re.compile(r"[\$€£¥]\s?[\d,]+(?:\.\d+)?")
_LONG_DIGIT_RE = re.compile(r"\b\d{9,}\b")
_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
# Separator between the words of a multi-word name: spaces, non-breaking
# spaces and hyphens, but deliberately NOT `\s`, which matches newlines.
# With `\s+` the pattern ran across line breaks and merged unrelated
# entities into one alias — a profile block redacted
# "Marina Silva\nUser" as a single PERSON, which then never recurred
# verbatim, so the registry both accumulated junk and failed to match
# "Marina Silva" on its own the next time.
_NAME_SEP = r"[  \-]+"
_NAME_RE = re.compile(
    rf"\b[A-Z][a-zà-ÿ]{{1,}}(?:{_NAME_SEP}[A-Z][a-zà-ÿ]{{1,}}){{0,3}}\b",
)

# A capitalised word opening a sentence is usually just a sentence.
# Matching those turned prose into placeholder soup and mangled the app's
# own system prompts ("Answer in three sentences" → "__PERSON_1__ in
# three sentences"), while writing those words into the persistent
# registry — after which they were rewritten in every later prompt, on
# every path.
#
# Sentence-initial means: start of the text, or after . ! ? or a line
# break, allowing for a bullet marker in between.
#
# Colons are deliberately NOT boundaries. "User's location: Porto
# Alegre" is a label/value pair, and treating the colon as a sentence
# start trimmed "Porto" off the front of the value — redacting only
# "Alegre" and registering half a place name. Labelled values are
# exactly the shape `build_user_context()` emits, so this matters.
_SENTENCE_START_RE = re.compile(r"(?:^|[.!?\n\r])[\s\-*•]*$")

# Position alone is NOT enough to skip a match, so the skip is gated on
# this list too. Trusting position would silently stop redacting "Marina
# called me", and — because an abbreviation dot reads as a sentence end —
# "Dr. Alvarez" as well. Over-redaction is a quality problem;
# under-redaction is a leak. So anything not recognised here stays
# redacted, and this list holds only words that are sentence openers
# rather than plausible names.
_SENTENCE_OPENERS: frozenset[str] = frozenset({
    # Imperatives common in system prompts and user requests.
    "Add", "Answer", "Ask", "Avoid", "Be", "Call", "Cancel", "Check",
    "Confirm", "Create", "Delete", "Describe", "Do", "Don", "Draft",
    "Email", "Ensure", "Explain", "Find", "Focus", "Follow", "Give",
    "Help", "Identify", "Ignore", "Include", "Keep", "Let", "List",
    "Make", "Move", "Never", "Note", "Only", "Open", "Prefer", "Read",
    "Remember", "Remind", "Remove", "Reply", "Return", "Review",
    "Schedule", "Send", "Set", "Show", "Start", "Stop", "Summarise",
    "Summarize", "Tell", "Update", "Use", "Write",
    # Adverbial / connective openers.
    "Above", "After", "Again", "Also", "Although", "Always", "And",
    "Because", "Before", "Both", "But", "Consider", "Each", "Either",
    "Every", "Finally", "First", "However", "If", "Instead", "Last",
    "Later", "Meanwhile", "Next", "Now", "Once", "Otherwise", "Please",
    "Since", "So", "Some", "Still", "Then", "There", "These", "This",
    "Those", "Thus", "Today", "Tomorrow", "When", "Where", "Which",
    "While", "Why", "Yesterday", "Yet",
    # Portuguese equivalents — the product ships pt-BR copy.
    "Adicione", "Agende", "Ajude", "Confirme", "Crie", "Envie",
    "Escreva", "Lembre", "Mostre", "Não", "Nunca", "Por", "Sempre",
    "Verifique",
})

# EMAIL → MONEY → DATE → PHONE → ID → PERSON. Money and date take
# precedence over generic digit runs; phone runs before ID so a
# delimited phone number is correctly tagged; PERSON is last so other
# entity matchers consume their text first.
_ENTITY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("EMAIL", _EMAIL_RE),
    ("MONEY", _MONEY_RE),
    ("DATE", _ISO_DATE_RE),
    ("PHONE", _PHONE_RE),
    ("ID", _LONG_DIGIT_RE),
    ("PERSON", _NAME_RE),
)

# Words that look like Capitalised names but almost never are; skip
# them to keep the placeholder set small.
_NAME_DENYLIST: frozenset[str] = frozenset({
    "I", "We", "You", "My", "Our", "The", "A", "An",
    "On", "In", "At", "Of", "From", "To", "By", "For",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
    "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
})


@dataclass
class RedactionMap:
    """Lookup of placeholder → original token used in one egress call.

    The persistent :class:`RedactionRegistry` is the source of truth.
    This per-call map is a thin view that the gateway uses when it
    rehydrates the response — it only carries the placeholders that
    were actually emitted in this outbound payload, so a hallucinated
    ``__PERSON_99__`` doesn't accidentally get expanded.

    sensitivity_tier: 3
    """

    forward: dict[str, str] = field(default_factory=dict)
    reverse: dict[str, str] = field(default_factory=dict)

    def record(self, original: str, placeholder: str) -> None:
        """Note that ``original`` was redacted to ``placeholder``.

        sensitivity_tier: 3
        """
        self.forward[original] = placeholder
        self.reverse[placeholder] = original


def _trim_sentence_initial(
    match: re.Match[str],
    original: str,
) -> tuple[str, str]:
    """Split a leading sentence-opening word off a PERSON match.

    A capitalised word is weak evidence of a name; at the start of a
    sentence it is usually just a sentence. But *skipping* such a match
    wholesale is wrong too, because the pattern happily swallows the
    verb: "Call Marina Silva" matches as one three-word name, so
    rejecting it would leave a real name unredacted, and accepting it
    would register "Call Marina Silva" as the alias — after which
    "Marina Silva" alone never matches again.

    So trim rather than reject: return ``(kept_prefix, name)`` where
    ``name`` is what should be redacted. ``name`` is empty when nothing
    is left, i.e. the match was only the sentence opener.

    Trimming happens only when the leading word is a recognised opener
    (:data:`_SENTENCE_OPENERS`). Position alone would turn "Marina
    called me" and "Dr. Alvarez" into leaks — see the note there.

    Registry-known values never reach here (pass 1 replaced them), so a
    single-word name still redacts once it has been seen in a fuller
    form or seeded from the graph.

    sensitivity_tier: 1
    """
    head, sep, tail = original.partition(" ")
    if head not in _SENTENCE_OPENERS:
        return "", original
    if not _SENTENCE_START_RE.search(match.string[: match.start()]):
        return "", original
    if not sep:
        # The match was only the opener — prose, not a person.
        return original, ""
    return head + sep, tail


def _redact_with(
    text: str,
    *,
    registry: RedactionRegistry,
    out_map: RedactionMap,
) -> str:
    """Two-pass deterministic redaction over ``text``.

    Pass 1 — registry-backed: replace every previously-seen raw value
    with its stable placeholder using the registry's compiled regex.

    Pass 2 — novel-entity: run the existing :data:`_ENTITY_PATTERNS`
    over the residual; each match is registered (so future prompts
    benefit from pass 1) and replaced with the resulting placeholder.

    sensitivity_tier: 3 (intermediate state) / 1 (return value)
    """
    redacted = text

    known_pattern = registry.known_pattern()
    if known_pattern is not None:
        def _replace_known(match: re.Match[str]) -> str:
            raw = match.group(0)
            placeholder = registry.lookup(raw)
            if placeholder is None:
                return raw
            out_map.record(raw, placeholder)
            return placeholder

        redacted = known_pattern.sub(_replace_known, redacted)

    for kind, pattern in _ENTITY_PATTERNS:
        def _replace(
            match: re.Match[str],
            _kind: str = kind,
        ) -> str:
            original = match.group(0)
            prefix = ""
            if _kind == "PERSON":
                tokens = original.split()
                if tokens and tokens[0] in _NAME_DENYLIST:
                    return original
                if original in _NAME_DENYLIST:
                    return original
                prefix, original = _trim_sentence_initial(match, original)
                if not original:
                    return prefix
            placeholder = registry.assign(_kind, original)
            out_map.record(original, placeholder)
            return prefix + placeholder

        redacted = pattern.sub(_replace, redacted)

    return redacted


def redact_with_registry(
    text: str,
    *,
    registry: RedactionRegistry | None = None,
) -> tuple[str, RedactionMap]:
    """Replace high-signal entities with stable placeholders.

    Returns the redacted text and the per-call :class:`RedactionMap`
    needed to rehydrate the model's response.

    ``registry`` defaults to the process-wide
    :func:`default_redaction_registry`. Tests pass a temp-file-backed
    instance for isolation.

    sensitivity_tier: 1 (text) / 3 (map)
    """
    reg = registry or default_redaction_registry()
    mapping = RedactionMap()
    redacted = _redact_with(text, registry=reg, out_map=mapping)
    return redacted, mapping


def redact(text: str) -> tuple[str, RedactionMap]:
    """Backwards-compatible wrapper around :func:`redact_with_registry`.

    sensitivity_tier: 1 (text) / 3 (map)
    """
    return redact_with_registry(text)


def rehydrate(text: str, mapping: RedactionMap) -> str:
    """Restore original tokens in ``text`` using the per-call ``mapping``.

    Placeholders without a matching entry pass through unchanged so a
    response that hallucinates a new ``__PERSON_99__`` token doesn't
    raise.

    sensitivity_tier: varies (output may contain Tier 3 values)
    """
    if not mapping.reverse:
        return text
    pattern = re.compile(
        "|".join(re.escape(p) for p in mapping.reverse),
    )
    return pattern.sub(lambda m: mapping.reverse[m.group(0)], text)


__all__ = [
    "RedactionMap",
    "redact",
    "redact_with_registry",
    "rehydrate",
]
