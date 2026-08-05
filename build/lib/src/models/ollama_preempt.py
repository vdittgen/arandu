"""Cross-process Ollama preemption and quiet-window signaling.

Allows interactive chat requests to preempt in-flight background
Ollama work, and ensures background tasks only start when the user
has been idle for a configurable quiet window.

Mechanism: a single timestamp file written by interactive callers.
Background callers check the file before and during LLM calls.

sensitivity_tier: 1 (infrastructure, no user data)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_PREEMPT_PATH = Path.home() / ".arandu" / "data" / ".ollama_preempt"

# Seconds of user inactivity before background work may start.
QUIET_WINDOW_S = 30


def signal_preempt() -> None:
    """Write current timestamp to signal file.

    Called by the interactive tier before acquiring the Ollama lock.
    Tells any background/proactive task to yield.

    sensitivity_tier: 1
    """
    try:
        _PREEMPT_PATH.parent.mkdir(parents=True, exist_ok=True)
        _PREEMPT_PATH.write_text(str(time.time()))
    except OSError:
        pass  # best-effort


def clear_preempt() -> None:
    """Remove the signal file after interactive request completes.

    sensitivity_tier: 1
    """
    try:
        _PREEMPT_PATH.unlink(missing_ok=True)
    except OSError:
        pass


def check_preempted() -> bool:
    """Return True if an interactive request signaled within 2 seconds.

    Called by background/proactive callers before each LLM call.

    sensitivity_tier: 1
    """
    try:
        ts = float(_PREEMPT_PATH.read_text())
        return (time.time() - ts) < 2.0
    except (FileNotFoundError, ValueError, OSError):
        return False


def is_quiet_window() -> bool:
    """Return True if no interactive request in the last QUIET_WINDOW_S.

    Background callers should wait for a quiet window before starting
    Ollama work so they don't compete with interactive chat.

    sensitivity_tier: 1
    """
    try:
        ts = float(_PREEMPT_PATH.read_text())
        return (time.time() - ts) > QUIET_WINDOW_S
    except (FileNotFoundError, ValueError, OSError):
        return True  # no signal file = no recent activity


def wait_for_quiet_window(timeout: float = 300) -> bool:
    """Block until the user has been idle for QUIET_WINDOW_S.

    Polls every 2 seconds.  Returns True if quiet window reached,
    False if *timeout* expired while waiting.

    sensitivity_tier: 1
    """
    # Use wall-clock time — time.monotonic() pauses during macOS
    # laptop sleep, making the deadline stale after wake.
    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_quiet_window():
            return True
        time.sleep(2)
    return False
