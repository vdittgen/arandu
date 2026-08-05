"""Ollama lifecycle manager: health checks, model preloading, and status.

Provides a manager that other modules query for Ollama availability
before attempting LLM calls.  Centralizes server interaction that was
previously spread across brain_agent, labeler, and embedding modules.

sensitivity_tier: 1 (only model names and server status, no user data)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import ollama

from src.core.profiler import timed

logger = logging.getLogger(__name__)

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_CHAT_MODEL = "llama3.1:70b"
DEFAULT_EMBED_MODEL = "nomic-embed-text"

# Live download progress for the model pull is published here so the UI can
# poll it (see the Rust `get_model_pull_progress` command). Absent file = no
# pull in flight.
PULL_PROGRESS_PATH = (
    Path.home() / ".arandu" / "data" / "ollama_pull_progress.json"
)


def _chunk_field(chunk: Any, name: str) -> Any:
    """Read ``name`` from an ollama pull progress chunk (object or dict)."""
    if isinstance(chunk, dict):
        return chunk.get(name)
    return getattr(chunk, name, None)


def _write_pull_progress(
    model: str, status: str, completed: int, total: int
) -> None:
    """Atomically publish current pull progress for the UI to poll."""
    try:
        percent = round(completed / total * 100, 1) if total else 0.0
        payload = {
            "model": model,
            "status": status,
            "completed": int(completed),
            "total": int(total),
            "percent": percent,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        PULL_PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = PULL_PROGRESS_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(PULL_PROGRESS_PATH)
    except Exception:  # noqa: BLE001 — progress is best-effort
        pass


def _clear_pull_progress() -> None:
    """Remove the progress file — signals "no pull in flight" to the UI."""
    try:
        PULL_PROGRESS_PATH.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass


class ModelStatus(str, Enum):
    """Status of a model on the Ollama server.

    sensitivity_tier: 1
    """

    UNKNOWN = "unknown"
    OFFLINE = "offline"
    NOT_FOUND = "not_found"
    AVAILABLE = "available"


@dataclass(frozen=True)
class OllamaStatus:
    """Overall Ollama server and model status.

    sensitivity_tier: 1
    """

    server_reachable: bool
    chat_model: str
    chat_model_status: ModelStatus
    embed_model: str
    embed_model_status: ModelStatus
    server_version: str = ""


class OllamaManager:
    """Manages Ollama health, model preloading, and status queries.

    sensitivity_tier: 1
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        chat_model: str = DEFAULT_CHAT_MODEL,
        embed_model: str = DEFAULT_EMBED_MODEL,
    ) -> None:
        self._host = host
        self._chat_model = chat_model
        self._embed_model = embed_model
        # Default ollama.Client has timeout=None (infinite).
        # Use 10s timeout to prevent hanging status checks.
        self._client = ollama.Client(host=host, timeout=10.0)
        self._last_status: OllamaStatus | None = None

    @timed()
    def check_health(self) -> OllamaStatus:
        """Check Ollama server connectivity and model availability.

        sensitivity_tier: 1
        """
        try:
            model_list = self._client.list()
            available: set[str] = set()
            for m in model_list.models:
                available.add(m.model)
        except Exception:
            status = OllamaStatus(
                server_reachable=False,
                chat_model=self._chat_model,
                chat_model_status=ModelStatus.OFFLINE,
                embed_model=self._embed_model,
                embed_model_status=ModelStatus.OFFLINE,
            )
            self._last_status = status
            return status

        chat_status = self._resolve_model_status(
            self._chat_model,
            available,
        )
        embed_status = self._resolve_model_status(
            self._embed_model,
            available,
        )

        status = OllamaStatus(
            server_reachable=True,
            chat_model=self._chat_model,
            chat_model_status=chat_status,
            embed_model=self._embed_model,
            embed_model_status=embed_status,
        )
        self._last_status = status
        return status

    def ensure_running(self) -> bool:
        """Start Ollama server if not already running.

        Checks if the server is reachable; if not, looks for the
        ``ollama`` binary on PATH and spawns ``ollama serve`` in the
        background.  Waits up to 10 seconds for the server to start.

        Returns:
            True if the server is reachable (was already running or
            started successfully).

        sensitivity_tier: 1
        """
        if self._is_reachable():
            return True

        # When running under the Arandu desktop app, the Tauri
        # ``OllamaSupervisor`` owns the server lifecycle (it ties Ollama to the
        # app: started on launch, reaped on exit). Don't spawn a second,
        # un-owned ``ollama serve`` here — just report it as unreachable.
        if os.environ.get("ARANDU_OLLAMA_MANAGED") == "1":
            logger.info(
                "Ollama unreachable but Arandu manages its lifecycle "
                "(ARANDU_OLLAMA_MANAGED=1); not auto-starting.",
            )
            return False

        import shutil
        import subprocess

        ollama_path = shutil.which("ollama")
        if not ollama_path:
            logger.warning(
                "Ollama binary not found in PATH — cannot auto-start",
            )
            return False

        logger.info(
            "Starting Ollama server via '%s serve' "
            "(OLLAMA_NUM_PARALLEL=1)",
            ollama_path,
        )
        env = {**os.environ, "OLLAMA_NUM_PARALLEL": "1"}
        subprocess.Popen(
            [ollama_path, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )

        # Wait up to 10 seconds for the server to become reachable.
        for _ in range(20):
            time.sleep(0.5)
            if self._is_reachable():
                logger.info("Ollama server started successfully")
                return True

        logger.warning(
            "Ollama server did not start within 10 seconds",
        )
        return False

    def stop_server(self) -> bool:
        """Stop the Ollama server if it was auto-started by Arandu.

        Sends SIGTERM to the ``ollama serve`` process.  Only kills
        processes whose command line matches ``ollama serve``.

        Returns:
            True if a process was stopped or none was running.

        sensitivity_tier: 1
        """
        import subprocess

        try:
            result = subprocess.run(
                ["pkill", "-f", "ollama serve"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                logger.info("Ollama server stopped")
            else:
                logger.debug("No ollama serve process found to stop")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to stop Ollama: %s", exc)
            return False

    def _is_reachable(self) -> bool:
        """Check if Ollama server responds to a basic request.

        sensitivity_tier: 1
        """
        try:
            self._client.list()
            return True
        except Exception:  # noqa: BLE001
            return False

    def ensure_model_pulled(self, model: str | None = None) -> bool:
        """Ensure ``model`` (default: the configured chat model) is present
        locally, pulling it from the Ollama registry if missing.

        Matches on the exact tag (accepting the implicit ``:latest`` form) so
        a different tag of the same family (e.g. ``llama3.1:8b`` present when
        ``llama3.1:70b`` is wanted) does NOT count as available.

        The instance client uses a short 10s timeout for status checks; a
        model download can take many minutes, so the pull runs on a dedicated
        client with no timeout.

        Returns:
            True if the model is present (already pulled or freshly pulled).

        sensitivity_tier: 1
        """
        target = model or self._chat_model
        try:
            names = {m.model for m in self._client.list().models}
        except Exception:  # noqa: BLE001
            names = set()
        if target in names or f"{target}:latest" in names:
            return True

        logger.info("Model %s not present locally — pulling…", target)
        _write_pull_progress(target, "starting", 0, 0)
        try:
            puller = ollama.Client(host=self._host, timeout=None)
            last_pct = -1
            last_status = ""
            for chunk in puller.pull(target, stream=True):
                status = _chunk_field(chunk, "status") or ""
                completed = _chunk_field(chunk, "completed") or 0
                total = _chunk_field(chunk, "total") or 0
                pct = int(completed / total * 100) if total else 0
                # Throttle file writes — pull emits many chunks per layer.
                if status != last_status or pct != last_pct:
                    _write_pull_progress(target, status, completed, total)
                    last_status, last_pct = status, pct
            logger.info("Model %s pulled successfully", target)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to pull model %s: %s", target, exc)
            return False
        finally:
            _clear_pull_progress()

    @timed()
    def preload_model(self, model: str | None = None) -> bool:
        """Preload a model into Ollama's memory via a minimal prompt.

        Ollama keeps models loaded for ~5 min after last request.
        Calling this at startup eliminates cold-start delay.

        sensitivity_tier: 1
        """
        target = model or self._chat_model
        try:
            # Cold-loading a model can take far longer than the instance
            # client's 10s status-check timeout (minutes for a large model),
            # so the load runs on a dedicated client with no timeout.
            loader = ollama.Client(host=self._host, timeout=None)
            loader.chat(
                model=target,
                messages=[{"role": "user", "content": "hi"}],
                keep_alive="10m",
            )
            logger.info("Model %s preloaded successfully", target)
            return True
        except Exception as exc:
            logger.warning(
                "Failed to preload model %s: %s",
                target,
                exc,
            )
            return False

    @property
    def last_status(self) -> OllamaStatus | None:
        """Return the last cached status (call check_health first).

        sensitivity_tier: 1
        """
        return self._last_status

    def get_status_dict(self) -> dict[str, Any]:
        """Return status as a JSON-serializable dict.

        sensitivity_tier: 1
        """
        status = self._last_status or self.check_health()
        return {
            "server_reachable": status.server_reachable,
            "chat_model": status.chat_model,
            "chat_model_status": status.chat_model_status.value,
            "embed_model": status.embed_model,
            "embed_model_status": status.embed_model_status.value,
            "server_version": status.server_version,
        }

    def _resolve_model_status(
        self,
        model_name: str,
        available_models: set[str],
    ) -> ModelStatus:
        """Determine status for a specific model.

        sensitivity_tier: N/A
        """
        if model_name in available_models:
            return ModelStatus.AVAILABLE
        base = model_name.split(":")[0]
        for available in available_models:
            if available.startswith(base + ":"):
                return ModelStatus.AVAILABLE
        return ModelStatus.NOT_FOUND
