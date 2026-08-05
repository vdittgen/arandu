"""Cross-process Ollama access serialization via fcntl.flock.

A single lock file serializes all Ollama requests.  Higher-priority
callers (interactive, proactive) preempt lower-priority ones
(background) via the mechanism in ``ollama_preempt.py``.

Uses ``flock`` which is automatically released when the file
descriptor is closed or the process exits (even on SIGKILL) — no
stale lock cleanup needed.

sensitivity_tier: 1 (infrastructure)
"""

from __future__ import annotations

import fcntl
import logging
import os
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

_LOCK_DIR = Path.home() / ".arandu" / "data"
_LOCK_PATH = _LOCK_DIR / ".ollama_lock"

_DEFAULT_TIMEOUT = 120  # interactive / proactive
_BACKGROUND_TIMEOUT = 300  # background tasks can wait longer


def has_lock_contention() -> bool:
    """Return True if another process is waiting for the Ollama lock.

    Opens a separate fd and tries a non-blocking lock. If it fails,
    another process holds or is waiting for the lock. Used to decide
    whether to keep the model loaded for the next caller.

    sensitivity_tier: 1
    """
    try:
        fd = os.open(str(_LOCK_PATH), os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
            return False
        except OSError:
            return True
        finally:
            os.close(fd)
    except OSError:
        return False


@contextmanager
def ollama_lock(
    timeout: float = _DEFAULT_TIMEOUT,
    *,
    priority: str = "normal",
) -> Generator[None, None, None]:
    """Acquire Ollama access across all Arandu processes.

    Blocks until the lock is available or *timeout* seconds elapse.
    Raises ``TimeoutError`` if the lock cannot be acquired in time.

    *priority* selects the default timeout:

    - ``"interactive"`` / ``"normal"`` / ``"proactive"`` — 120s.
    - ``"background"`` — 300s (pipeline, insights, sync).

    Background callers also check for preemption signals while
    waiting — if a higher-priority caller wants the lock, the
    background caller gives up immediately instead of waiting
    the full timeout.

    sensitivity_tier: 1
    """
    from src.models.ollama_preempt import check_preempted

    _LOCK_DIR.mkdir(parents=True, exist_ok=True)

    if timeout == _DEFAULT_TIMEOUT and priority == "background":
        timeout = _BACKGROUND_TIMEOUT

    is_background = priority == "background"

    fd = os.open(str(_LOCK_PATH), os.O_CREAT | os.O_RDWR)
    acquired = False
    try:
        # Use wall-clock time — time.monotonic() pauses during macOS
        # laptop sleep, making the deadline stale after wake.
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError:
                # Background: yield immediately if preempted.
                if is_background and check_preempted():
                    from src.models.llm_provider import PreemptedError

                    raise PreemptedError  # noqa: TRY301
                time.sleep(0.5)

        if not acquired:
            raise TimeoutError(
                f"Could not acquire Ollama lock "
                f"within {timeout}s (priority={priority})"
            )
        yield
    finally:
        if acquired:
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
