"""Sandboxed code execution for deep agents.

Two execution surfaces are supported in Phase 1:

- ``run_python`` — runs a short Python snippet in a subprocess with a
  restricted import allow-list and CPU/wall-time/memory limits.
- ``run_sql`` — runs a read-only DuckDB query against a path the caller
  passes in. DDL and writes are rejected before the SQL ever reaches the
  database.

Both functions return a ``SandboxResult`` containing stdout, stderr, the
exit reason, and a duration. They never raise on user-code failure —
that's surfaced to the deep agent so it can adjust the plan.

What this sandbox is **not**:

- A full container. Subprocess + ``resource.setrlimit`` blocks most
  abuse, but a determined attacker with arbitrary Python can still
  exhaust system resources. We rely on the **prompt** firewall + the
  agent-side allow-list of tools to keep arbitrary-code generation
  rare; the subprocess limits are defense-in-depth.
- Cross-platform on Windows. Phase 1 ships macOS/Linux only because
  ``resource.setrlimit`` and the ``-I`` flag below are POSIX-specific.

sensitivity_tier: 1 (executes user-derived code; never stores results)
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SandboxResult:
    """Outcome of a single sandbox run.

    ``exit_reason`` distinguishes user-code errors (``"error"``) from
    enforcement actions (``"timeout"``, ``"memory"``, ``"rejected"``)
    so deep agents can react sensibly.

    sensitivity_tier: 1
    """

    ok: bool
    stdout: str
    stderr: str
    exit_reason: str  # "ok" | "error" | "timeout" | "memory" | "rejected"
    duration_ms: float
    rows: list[dict[str, Any]] | None = None


# ---------------------------------------------------------------------------
# Python sandbox
# ---------------------------------------------------------------------------

PYTHON_ALLOWED_IMPORTS: tuple[str, ...] = (
    "json", "re", "datetime", "itertools", "functools", "collections",
    "math", "statistics", "string", "textwrap", "uuid",
)

# Hard caps for ``run_python``.
PYTHON_DEFAULT_TIMEOUT_S = 30
PYTHON_MAX_OUTPUT_BYTES = 256 * 1024  # 256 KB
PYTHON_MEMORY_BYTES = 256 * 1024 * 1024  # 256 MB

_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+([\w\.]+)|import\s+([\w\.]+))",
    re.MULTILINE,
)

_FORBIDDEN_NAMES = {
    "os", "sys", "subprocess", "socket", "shutil", "ctypes",
    "asyncio", "multiprocessing", "threading", "pathlib",
    "tempfile", "fcntl", "signal", "select", "http", "urllib",
    "requests", "open", "exec", "eval", "compile", "__import__",
    "globals", "locals", "vars",
}


def _scan_python_source(source: str) -> str | None:
    """Return a reason string if ``source`` is rejected, else None.

    Two passes:

    1. Import statements: only the allow-list is permitted.
    2. Forbidden identifier scan: catches ``os.system``, ``__import__``,
       etc., even if the import would have been allowed.

    sensitivity_tier: 1
    """
    for match in _IMPORT_RE.finditer(source):
        module = (match.group(1) or match.group(2) or "").split(".")[0]
        if module not in PYTHON_ALLOWED_IMPORTS:
            return f"import not allowed: {module!r}"
    # Identifier scan — simple textual match is intentional. Any user
    # snippet that touches ``open(`` or ``os.`` is rejected regardless
    # of context. False positives are acceptable in a sandbox.
    for name in _FORBIDDEN_NAMES:
        if re.search(rf"\b{re.escape(name)}\b", source):
            return f"forbidden identifier: {name!r}"
    return None


def run_python(
    source: str,
    *,
    timeout_s: float = PYTHON_DEFAULT_TIMEOUT_S,
    workspace: Path | None = None,
) -> SandboxResult:
    """Execute a Python snippet in an isolated subprocess.

    The subprocess starts with ``-I`` (isolated mode: no PYTHONPATH,
    no user site-packages, no ``-X`` from env), an empty cwd in a
    temp directory, ``PYTHONDONTWRITEBYTECODE=1``, and rlimits for
    CPU + address space.

    sensitivity_tier: 1
    """
    rejection = _scan_python_source(source)
    if rejection is not None:
        return SandboxResult(
            ok=False, stdout="", stderr=rejection,
            exit_reason="rejected", duration_ms=0.0,
        )

    bootstrap = textwrap.dedent(
        """
        import resource as _r
        try:
            _r.setrlimit(_r.RLIMIT_AS, ({mem}, {mem}))
        except (ValueError, OSError):
            pass
        try:
            _r.setrlimit(_r.RLIMIT_CPU, (int({cpu}) + 1, int({cpu}) + 1))
        except (ValueError, OSError):
            pass
        """,
    ).format(mem=PYTHON_MEMORY_BYTES, cpu=int(timeout_s))

    full_source = bootstrap + "\n" + source

    with tempfile.TemporaryDirectory(prefix="sb_sandbox_") as tmpdir:
        cwd = workspace or Path(tmpdir)
        cwd.mkdir(parents=True, exist_ok=True)
        env = {
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "LC_ALL": "C.UTF-8",
            "LANG": "C.UTF-8",
        }
        argv = [sys.executable, "-I", "-c", full_source]
        start = time.monotonic()
        try:
            proc = subprocess.run(  # noqa: S603
                argv,
                cwd=str(cwd),
                env=env,
                input=b"",
                capture_output=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return SandboxResult(
                ok=False,
                stdout=_decode_capped(exc.stdout),
                stderr=_decode_capped(exc.stderr),
                exit_reason="timeout",
                duration_ms=(time.monotonic() - start) * 1000.0,
            )

    duration_ms = (time.monotonic() - start) * 1000.0
    stdout = _decode_capped(proc.stdout)
    stderr = _decode_capped(proc.stderr)
    if proc.returncode == 0:
        return SandboxResult(
            ok=True, stdout=stdout, stderr=stderr,
            exit_reason="ok", duration_ms=duration_ms,
        )
    # ``-9`` from RLIMIT_AS / cgroup OOM is reported as returncode -9.
    reason = "memory" if proc.returncode == -9 else "error"
    return SandboxResult(
        ok=False, stdout=stdout, stderr=stderr,
        exit_reason=reason, duration_ms=duration_ms,
    )


def _decode_capped(buf: bytes | None) -> str:
    if not buf:
        return ""
    if len(buf) > PYTHON_MAX_OUTPUT_BYTES:
        buf = buf[:PYTHON_MAX_OUTPUT_BYTES] + b"\n... [truncated]"
    return buf.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# SQL sandbox
# ---------------------------------------------------------------------------

# Only ``SELECT`` / ``WITH`` are allowed at the top level. Multiple
# statements are rejected so an attacker can't chain a SELECT with a
# DROP via a semicolon.
_SQL_FORBIDDEN = re.compile(
    r"""
    \b(
        INSERT | UPDATE | DELETE | DROP | ALTER | CREATE | TRUNCATE |
        REPLACE | ATTACH | DETACH | PRAGMA | COPY | EXPORT | IMPORT |
        LOAD | INSTALL | CALL | EXECUTE | GRANT | REVOKE | VACUUM
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)
_SQL_COMMENT = re.compile(r"(--.*?$)|(/\*.*?\*/)", re.MULTILINE | re.DOTALL)


def _scan_sql(query: str) -> str | None:
    """Return a rejection reason if ``query`` is unsafe, else None.

    sensitivity_tier: 1
    """
    cleaned = _SQL_COMMENT.sub(" ", query).strip()
    if ";" in cleaned.rstrip(";"):
        return "multi-statement SQL not allowed"
    if not re.match(r"^\s*(SELECT|WITH)\b", cleaned, re.IGNORECASE):
        return "only SELECT/WITH queries are allowed"
    forbidden = _SQL_FORBIDDEN.search(cleaned)
    if forbidden:
        return f"forbidden keyword: {forbidden.group(1).upper()}"
    return None


def run_sql(
    query: str,
    *,
    db_path: Path | str,
    timeout_s: float = 30.0,
    max_rows: int = 10000,
) -> SandboxResult:
    """Execute a read-only SELECT against the SQLite/DuckDB at ``db_path``.

    Phase 1 uses ``sqlite3`` directly because that's what the project's
    primary store is. A DuckDB path can be added later by branching on
    the file extension; both expose the same SELECT semantics.

    sensitivity_tier: 1
    """
    rejection = _scan_sql(query)
    if rejection is not None:
        return SandboxResult(
            ok=False, stdout="", stderr=rejection,
            exit_reason="rejected", duration_ms=0.0,
        )

    import sqlite3

    start = time.monotonic()
    path = str(db_path)
    try:
        # ``mode=ro`` rejects writes at the connection level.
        uri = f"file:{path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=timeout_s)
        try:
            cur = conn.execute(query)
            cols = [c[0] for c in cur.description or []]
            rows: list[dict[str, Any]] = []
            for raw in cur.fetchmany(max_rows + 1):
                rows.append(dict(zip(cols, raw, strict=False)))
                if len(rows) > max_rows:
                    break
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return SandboxResult(
            ok=False, stdout="", stderr=str(exc),
            exit_reason="error",
            duration_ms=(time.monotonic() - start) * 1000.0,
        )

    truncated = len(rows) > max_rows
    if truncated:
        rows = rows[:max_rows]
    duration_ms = (time.monotonic() - start) * 1000.0
    return SandboxResult(
        ok=True,
        stdout=json.dumps({"rows": rows, "truncated": truncated}),
        stderr="",
        exit_reason="ok",
        duration_ms=duration_ms,
        rows=rows,
    )


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------


WORKSPACE_ROOT = (
    Path.home() / ".arandu" / "data" / "deep_agents"
)


class WorkspaceError(Exception):
    """Raised on path-traversal or unreadable workspace operations.

    sensitivity_tier: 1
    """


def workspace_for(agent_id: str, run_id: str) -> Path:
    """Return (and create) the workspace directory for one deep-agent run.

    sensitivity_tier: 1
    """
    safe_agent = _safe_segment(agent_id)
    safe_run = _safe_segment(run_id)
    path = WORKSPACE_ROOT / safe_agent / safe_run / "workspace"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_segment(value: str) -> str:
    if not value or not re.match(r"^[A-Za-z0-9_\-\.]+$", value):
        msg = f"unsafe path segment: {value!r}"
        raise WorkspaceError(msg)
    return value


def resolve_in_workspace(workspace: Path, rel_path: str) -> Path:
    """Resolve ``rel_path`` inside ``workspace``, rejecting traversal.

    sensitivity_tier: 1
    """
    workspace = workspace.resolve()
    candidate = (workspace / rel_path).resolve()
    try:
        candidate.relative_to(workspace)
    except ValueError as exc:
        msg = f"path escapes workspace: {rel_path!r}"
        raise WorkspaceError(msg) from exc
    return candidate


__all__ = [
    "PYTHON_ALLOWED_IMPORTS",
    "SandboxResult",
    "WORKSPACE_ROOT",
    "WorkspaceError",
    "resolve_in_workspace",
    "run_python",
    "run_sql",
    "workspace_for",
]


# Mark a couple of symbols as intentionally available even though they
# aren't called inside this module.
_ = shlex
_ = os
