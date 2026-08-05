"""Split a Brain answer string into typed message parts.

Modern LLMs return mixed-content responses: prose interleaved with
diagrams, charts, and embedded HTML. The legacy chat surface treated
them all as plain text. This helper turns a single markdown answer
into a small ordered list of :class:`MessagePart`s that the chat
renderer registry can mount with the right component:

- ``text/markdown`` for prose (the default — markdown's own renderer
  handles fenced code with ``text/x-{lang}`` syntax highlighting).
- ``text/vnd.mermaid`` for ```mermaid blocks (diagram).
- ``application/vnd.vega-lite+json`` for ```vega-lite / ```vegalite
  blocks (chart spec).
- ``text/html`` for ```html blocks tagged ``runnable`` (sandboxed
  iframe). Plain ``html`` fences stay inside the markdown stream and
  render as a code listing — we don't auto-execute every HTML block.

Parts are emitted in source order so the UI renders them in the same
order the model wrote them.

sensitivity_tier: varies (operates on assistant output)
"""

from __future__ import annotations

import re
from typing import Any

from src.agents.core.output_types import MessagePart

# Match a fenced code block. Captures: 1=info string, 2=body.
# Allows both ``` and ~~~ fences and an optional info string.
_FENCE_RE = re.compile(
    r"(?:^|\n)```([^\n`]*)\n(.*?)\n```(?=\n|$)",
    re.DOTALL,
)


def _slug(prefix: str, n: int) -> str:
    return f"{prefix}-{n}"


def _markdown_part(
    pid: str, body: str, sensitivity_tier: int
) -> MessagePart | None:
    """Build a markdown part if the body is non-blank."""
    if not body.strip():
        return None
    return MessagePart(
        id=pid,
        mime="text/markdown",
        data=body,
        sensitivity_tier=sensitivity_tier,
    )


def _classify_fence(
    info: str, body: str
) -> tuple[str, dict[str, Any]] | None:
    """Map a fence info-string to a non-markdown MIME, if any.

    Returns ``(mime, extra_metadata)`` or ``None`` when the fence
    should stay inside the surrounding markdown stream. ``body`` is
    only consulted for ``file`` fences (where it carries the URL or
    path the renderer should load).
    """
    tag = info.strip().lower()
    if not tag:
        return None
    head = tag.split()[0]
    if head == "mermaid":
        return ("text/vnd.mermaid", {})
    if head in {"vega-lite", "vegalite"}:
        return ("application/vnd.vega-lite+json", {})
    # HTML always renders through the sandboxed iframe. The HtmlSandbox
    # uses sandbox="allow-scripts" with no allow-same-origin and refuses
    # remote URLs for sensitive parts, so executing model-emitted HTML
    # can't reach the host page, cookies, or the network.
    if head == "html":
        return ("text/html", {})
    if head == "file":
        # Fence body is a single URL or local path; sniff the extension
        # to route to the right renderer (PDF, image, audio, video,
        # Office doc). Unknown extensions fall through so the fence
        # stays inline as plain markdown.
        url = _first_token(body)
        if not url:
            return None
        mime = _ext_to_mime(url)
        if mime is None:
            return None
        # Suppress the literal "file" info-string as the title; the
        # frontend falls back to a MIME-derived label, which reads
        # better in the artifact chip.
        return (mime, {"data": url, "title": ""})
    return None


# Map of lowercase file extension → MIME for the ``file`` fence. Office
# MIMEs follow the OpenXML spec; image/audio/video entries match what
# the existing MediaArtifact prefix routes already render.
_EXT_TO_MIME: dict[str, str] = {
    "pdf": "application/pdf",
    "docx": (
        "application/vnd.openxmlformats-officedocument"
        ".wordprocessingml.document"
    ),
    "xlsx": (
        "application/vnd.openxmlformats-officedocument"
        ".spreadsheetml.sheet"
    ),
    "pptx": (
        "application/vnd.openxmlformats-officedocument"
        ".presentationml.presentation"
    ),
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
    "svg": "image/svg+xml",
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "ogg": "audio/ogg",
    "mp4": "video/mp4",
    "webm": "video/webm",
    "mov": "video/quicktime",
}


def _first_token(body: str) -> str:
    """Return the first non-empty line of ``body``, trimmed."""
    for line in body.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _ext_to_mime(url: str) -> str | None:
    """Detect a renderable MIME from a URL or path's extension.

    Strips query string and URL fragment first so links like
    ``foo.pdf?token=abc`` and ``foo.pdf#page=2`` still classify.
    """
    clean = url.split("?", 1)[0].split("#", 1)[0]
    if "." not in clean:
        return None
    ext = clean.rsplit(".", 1)[-1].lower()
    return _EXT_TO_MIME.get(ext)


def split_answer_into_parts(
    answer: str,
    *,
    sensitivity_tier: int = 2,
) -> list[MessagePart]:
    """Split ``answer`` into typed message parts.

    The default is a single ``text/markdown`` part. Fenced blocks
    matching :func:`_classify_fence` get pulled out as standalone
    parts so they can render via the registry. Surrounding prose is
    preserved in markdown parts in source order.

    sensitivity_tier: parts inherit this value.
    """
    if not answer:
        return []

    parts: list[MessagePart] = []
    cursor = 0
    counter = 0

    for match in _FENCE_RE.finditer(answer):
        info = match.group(1) or ""
        body = match.group(2) or ""
        classified = _classify_fence(info, body)
        if classified is None:
            # Keep this fence inline; markdown renderer handles it.
            continue

        mime, meta = classified
        before = answer[cursor : match.start()].lstrip("\n")
        before_part = _markdown_part(
            _slug("p", counter), before, sensitivity_tier
        )
        if before_part is not None:
            parts.append(before_part)
            counter += 1

        # Preserve the original fence as a markdown code part so the
        # source shows inline with syntax highlighting; the typed part
        # below renders the artifact in the side panel alongside it.
        code_fence = f"```{info.strip()}\n{body}\n```"
        parts.append(
            MessagePart(
                id=_slug("p", counter),
                mime="text/markdown",
                data=code_fence,
                sensitivity_tier=sensitivity_tier,
            )
        )
        counter += 1

        # `meta["data"]` overrides the fence body for file fences: the
        # renderer wants the cleaned URL, not the raw fence contents.
        data = meta.get("data", body)
        if "title" in meta:
            title = meta["title"]
        else:
            title = info.strip() or _default_title(mime)
        parts.append(
            MessagePart(
                id=_slug("p", counter),
                mime=mime,
                title=title,
                data=data,
                display="panel",
                sensitivity_tier=sensitivity_tier,
            )
        )
        counter += 1
        cursor = match.end()

    tail = answer[cursor:].lstrip("\n")
    tail_part = _markdown_part(_slug("p", counter), tail, sensitivity_tier)
    if tail_part is not None:
        parts.append(tail_part)

    if not parts:
        # Whole answer was consumed by fences with no surrounding text;
        # if even that produced nothing, fall back to a single markdown
        # part so the renderer always has something to show.
        single = _markdown_part("p-0", answer, sensitivity_tier)
        if single is not None:
            parts.append(single)

    return parts


def _default_title(mime: str) -> str:
    if mime == "text/vnd.mermaid":
        return "Diagram"
    if mime == "application/vnd.vega-lite+json":
        return "Chart"
    if mime == "text/html":
        return "HTML preview"
    return mime
