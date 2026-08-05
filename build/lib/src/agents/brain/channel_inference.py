"""Channel + language inference for action proposals.

Two structural defenses against the "reply on the wrong channel /
in the wrong language" failure mode:

- ``infer_action_channel`` — picks the channel the user is referring
  to (whatsapp / email / imessage / sms / phone) from explicit
  mentions in the message plus the source of the most recent inbound
  message captured in the grounded context. Lets the action matcher
  rank channel-appropriate tools first when the user says ``reply``.

- ``infer_inbound_language_hint`` — pulls the most recent inbound
  message's *body* from the grounded context so the param extractor
  and judge can match the language without needing a language-
  detection dependency. The LLM does the language matching; this
  helper just exposes the source text.

sensitivity_tier: 2 (grounded context can be Tier 3)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# Channel tokens we recognise. Keep in lock-step with the connector
# catalog: ``whatsapp``, ``apple-mail``, ``apple-messages``.
_CHANNEL_KEYWORDS: dict[str, tuple[str, ...]] = {
    "whatsapp": (
        "whatsapp", "whats app", "wpp", "wa", "zap", "zapzap",
    ),
    "email": (
        "email", "e-mail", "mail", "gmail", "outlook", "inbox",
        "e mail",
    ),
    "imessage": (
        "imessage", "i message", "messages app", "sms", "text message",
        "texting", "mensagem de texto",
    ),
}

# Map a connector_id to the abstract channel it serves. Tools are
# matched against connectors, so this is the bridge between the
# channel hint and the candidate tool list.
CONNECTOR_TO_CHANNEL: dict[str, str] = {
    "whatsapp": "whatsapp",
    "apple-mail": "email",
    "apple-messages": "imessage",
}

# Source values written into ``raw_messages.source`` by the bridges,
# plus the synthetic ``"email"`` we tag onto ``raw_emails`` rows in
# ``format_context``. This map is the *source-of-truth* boundary:
# whatever value lives in the DB column on the left → which abstract
# channel the action matcher should route a reply through on the
# right.
#
# Keep in lock-step with the producers:
#   - WhatsApp listener (src/extensions/bridges/whatsapp/listener.py)
#     writes ``source='whatsapp'`` on every insert.
#   - Apple bridge writes ``source='imessage'`` / ``'apple_mail'`` /
#     ``'apple_notes'`` depending on the surface.
#   - Gmail / Slack ingestion writes ``'gmail'`` / ``'slack'``.
SOURCE_TO_CHANNEL: dict[str, str] = {
    "whatsapp": "whatsapp",
    "apple_mail": "email",
    "apple-mail": "email",
    "email": "email",
    "gmail": "email",
    "outlook": "email",
    "imessage": "imessage",
    "apple_messages": "imessage",
    "sms": "imessage",
}


@dataclass(frozen=True)
class ChannelHint:
    """Where the user wants the action to land.

    ``confidence`` is ``"explicit"`` when the user said the channel
    name outright, ``"inferred"`` when we pulled it from the source of
    a recent inbound message, and ``""`` when we have no signal at
    all. Callers use confidence to decide whether to filter the
    candidate-tool list hard (explicit) or just rank-bias it
    (inferred).

    sensitivity_tier: 1
    """

    channel: str = ""
    confidence: str = ""


def infer_action_channel(
    question: str,
    context_text: str = "",
    sources: list[dict[str, Any]] | None = None,
) -> ChannelHint:
    """Pick the channel for an action request.

    Order of precedence:

    1. Explicit channel keyword in the user's message ("on whatsapp",
       "via email"). The user wins, full stop — confidence
       ``"explicit"``.
    2. Source of the most recent inbound message we know about, taken
       from ``sources`` (preferred — structured) or scraped from
       ``context_text`` (fallback — free-form recall output).
       Confidence ``"inferred"``.
    3. No signal — return an empty hint.

    sensitivity_tier: 2
    """
    explicit = _scan_keywords(question.lower())
    if explicit:
        return ChannelHint(channel=explicit, confidence="explicit")

    # 2. Sources first — structured ``source`` field is the cleanest
    # signal. Sources are typically appended in reverse chronological
    # order by the recall step, so the first match wins.
    for src in sources or []:
        candidate = str(src.get("source") or "").strip().lower()
        if candidate in SOURCE_TO_CHANNEL:
            return ChannelHint(
                channel=SOURCE_TO_CHANNEL[candidate],
                confidence="inferred",
            )

    # 2b. Free-form context scan — recall output sometimes embeds
    # the source as a textual prefix ("source=whatsapp" / "via
    # WhatsApp" / "Email from …"). Last-resort signal.
    if context_text:
        inferred = _scan_keywords(context_text.lower())
        if inferred:
            return ChannelHint(channel=inferred, confidence="inferred")

    return ChannelHint()


def _scan_keywords(text: str) -> str:
    """Return the first channel whose keyword appears in ``text``.

    sensitivity_tier: 1
    """
    for channel, keywords in _CHANNEL_KEYWORDS.items():
        for keyword in keywords:
            # Word-boundary match so "mail" matches "mail" / "email"
            # but not arbitrary substrings.
            if re.search(rf"\b{re.escape(keyword)}\b", text):
                return channel
    return ""


def filter_tools_by_channel(
    tools: list[Any],
    hint: ChannelHint,
) -> list[Any]:
    """Re-rank a ranked tool list to favour the hinted channel.

    Behaviour:

    - ``hint.channel`` empty → returns the input unchanged.
    - ``hint.confidence == "explicit"`` → drops every tool whose
      connector doesn't match the channel. Empty list if none match.
    - ``hint.confidence == "inferred"`` → preserves all tools but
      promotes matching-channel tools to the front. Falls back to
      the original ordering if no tool matches.

    sensitivity_tier: 1
    """
    if not hint.channel or not tools:
        return list(tools)
    matching: list[Any] = []
    other: list[Any] = []
    for tool in tools:
        connector_channel = CONNECTOR_TO_CHANNEL.get(
            getattr(tool, "connector_id", ""), "",
        )
        if connector_channel == hint.channel:
            matching.append(tool)
        else:
            other.append(tool)
    if hint.confidence == "explicit":
        return matching
    return matching + other


def infer_inbound_language_hint(
    sources: list[dict[str, Any]] | None,
    context_text: str = "",
) -> str:
    """Return the body of the most recent inbound message, if any.

    Used by the param-extractor / judge to match reply language to
    the inbound message — the LLM detects the language from the
    snippet rather than us guessing. Returns an empty string when no
    inbound message is available (the LLM falls back to the user's
    preferred language).

    sensitivity_tier: 2
    """
    if sources:
        for src in sources:
            # Only consider inbound messages — outbound is the user's
            # own writing and would just confirm their default tongue.
            # ``is_from_me`` may be a real bool (from format_context's
            # structured branch) or a stringy form ("True"/"1") when
            # the upstream serialiser flattened it; handle both.
            ifm = src.get("is_from_me")
            if isinstance(ifm, bool) and ifm:
                continue
            if isinstance(ifm, (int, float)) and ifm:
                continue
            if isinstance(ifm, str) and ifm.strip().lower() in {
                "true", "1", "yes",
            }:
                continue
            for key in ("content", "body", "snippet", "text"):
                value = src.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()[:400]
    # Fallback: try to pull a "From … : …" or similar line out of
    # ``context_text``. We deliberately don't try to be clever — a
    # short context line is good enough for the LLM to detect.
    if context_text:
        for line in context_text.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith(("from ", "de ", "sender")):
                return stripped[:400]
    return ""


__all__ = [
    "CONNECTOR_TO_CHANNEL",
    "ChannelHint",
    "SOURCE_TO_CHANNEL",
    "filter_tools_by_channel",
    "infer_action_channel",
    "infer_inbound_language_hint",
]
