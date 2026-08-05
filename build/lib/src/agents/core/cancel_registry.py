"""File-backed cancel signals for in-flight agent runs.

Each streaming orchestrator run gets a ``run_id`` (a short hex token
generated at the top of ``ask_stream``). The frontend stashes this id
from the ``run_started`` event and can later call the
``stop_research`` IPC, which routes to ``cmd_stop_research`` in the
CLI (a *separate* subprocess), which calls :func:`request_cancel`
here.

Because the ``ask-stream`` subprocess and the ``stop-research``
subprocess are different Python processes, the registry can't live in
plain in-process state. Instead, the source of truth is an empty
flag file at ``~/.arandu/data/cancel/<run_id>.flag``: writing it
signals cancel, ``should_stop`` checks for its existence at each
reflection checkpoint, and ``release`` removes it when the run
completes.

A tiny in-process ``threading.Event`` cache is kept alongside so
tests (and same-process callers) can poll without touching the
filesystem; it's a fast-path optimization, not a separate channel.

sensitivity_tier: 1
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, threading.Event] = {}
_LOCK = threading.Lock()


def _cancel_dir() -> Path:
    """Return the directory that holds per-run cancel flags.

    Lives under ``~/.arandu/data/cancel/`` so it shares the data
    root with audit logs and embeddings. The directory is created on
    demand — callers don't need to ensure it exists.

    sensitivity_tier: 1
    """
    root = os.environ.get("ARANDU_DATA_DIR")
    base = Path(root) if root else Path.home() / ".arandu" / "data"
    cancel_dir = base / "cancel"
    cancel_dir.mkdir(parents=True, exist_ok=True)
    return cancel_dir


def _flag_path(run_id: str) -> Path:
    """Path of the cancel flag for ``run_id``.

    sensitivity_tier: 1
    """
    return _cancel_dir() / f"{run_id}.flag"


def cancel_token(run_id: str) -> threading.Event:
    """Reserve a fast-path event for ``run_id`` and return it.

    The reflective runner calls this at the top of a run. The returned
    event is set by same-process ``request_cancel`` calls to avoid an
    fs stat on every reflection checkpoint when the cancel originates
    locally (tests). Cross-process cancels still work via the file
    flag — :func:`should_stop` checks both sources.

    Always release the token via :func:`release` once the run is done.

    sensitivity_tier: 1
    """
    with _LOCK:
        ev = _REGISTRY.get(run_id)
        if ev is None:
            ev = threading.Event()
            _REGISTRY[run_id] = ev
        return ev


def request_cancel(run_id: str) -> bool:
    """Signal the run with id ``run_id`` to stop researching.

    Always writes the cross-process flag file. When the requester is in
    the same process as the streaming agent, also flips the in-process
    event so subsequent ``should_stop`` calls return ``True`` without
    an fs stat. Returns ``True`` if the flag was written successfully,
    ``False`` if the filesystem write failed (extremely rare).

    The signal is *not* a kill: the in-flight run keeps running, but at
    its next reflection checkpoint the reflective runner sees the flag
    and injects a ``STOP_REQUEST`` user message so the model finalizes
    its answer with the context it already has.

    sensitivity_tier: 1
    """
    try:
        _flag_path(run_id).touch()
    except OSError:
        logger.exception("cancel_registry could not write flag")
        return False
    with _LOCK:
        ev = _REGISTRY.get(run_id)
    if ev is not None:
        ev.set()
    return True


def should_stop(run_id: str) -> bool:
    """Return ``True`` if cancellation was requested for ``run_id``.

    Checks the in-process event first (fast-path) then falls back to
    the filesystem flag. Safe to call from any thread.

    sensitivity_tier: 1
    """
    with _LOCK:
        ev = _REGISTRY.get(run_id)
    if ev is not None and ev.is_set():
        return True
    try:
        return _flag_path(run_id).exists()
    except OSError:
        return False


def release(run_id: str) -> None:
    """Drop the registry entry and remove the on-disk flag.

    Call once a run completes (or errors out) so neither the in-memory
    map nor the cancel directory leak entries across long-lived
    processes.

    sensitivity_tier: 1
    """
    with _LOCK:
        _REGISTRY.pop(run_id, None)
    try:
        _flag_path(run_id).unlink(missing_ok=True)
    except OSError:
        logger.debug("cancel_registry could not unlink flag", exc_info=True)


def _reset_for_tests() -> None:
    """Drop every in-memory entry and remove all cancel flags. Tests only.

    sensitivity_tier: 1
    """
    with _LOCK:
        _REGISTRY.clear()
    try:
        cancel_dir = _cancel_dir()
    except OSError:
        return
    for flag in cancel_dir.glob("*.flag"):
        try:
            flag.unlink()
        except OSError:
            pass


__all__ = [
    "cancel_token",
    "release",
    "request_cancel",
    "should_stop",
]
