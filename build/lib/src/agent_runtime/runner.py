"""Agent runner — sandboxed execution of built-in and third-party agents.

Loads agent code, validates manifests, runs agents in-process (built-in)
or as isolated subprocesses (third-party), and collects results.

sensitivity_tier: 1 (manages agent lifecycle, no direct data access)
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import signal
import time
from pathlib import Path
from typing import Any

import yaml

from src.agent_runtime.base import BrainAgent
from src.agent_runtime.context import AgentContext
from src.agent_runtime.models import (
    AgentManifest,
    AgentResult,
    AgentStatus,
    TablePermission,
    TriggerMode,
)
from src.agent_runtime.sensitivity_guard import SensitivityGuard
from src.agent_runtime.skills import SkillRegistry

logger = logging.getLogger(__name__)

BUILTIN_AGENTS_DIR = Path(__file__).parent / "builtin"
USER_AGENTS_DIR = Path.home() / ".arandu" / "extensions"
AGENT_DATA_DIR = Path.home() / ".arandu" / "data" / "agents"


class AgentLoadError(Exception):
    """Raised when an agent manifest is invalid or cannot be loaded.

    sensitivity_tier: N/A
    """


class AgentRunner:
    """Loads, validates, and runs agents in sandboxed environments.

    Built-in agents execute in-process with full SensitivityGuard
    enforcement.  Third-party agents will use subprocess isolation
    (via :mod:`src.agent_runtime.worker`) in a future release.

    sensitivity_tier: 1
    """

    def __init__(
        self,
        db_engine: Any = None,
        builtin_dir: Path = BUILTIN_AGENTS_DIR,
        user_dir: Path = USER_AGENTS_DIR,
        skill_registry: SkillRegistry | None = None,
    ) -> None:
        self._db = db_engine
        self._builtin_dir = builtin_dir
        self._user_dir = user_dir
        self._manifests: dict[str, AgentManifest] = {}
        self._agent_dirs: dict[str, Path] = {}
        self._last_results: dict[str, AgentResult] = {}
        self._running: set[str] = set()
        self._skill_registry = skill_registry or SkillRegistry()
        if not self._skill_registry.list_skills():
            self._skill_registry.register_builtin_skills()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def discover_agents(self) -> list[AgentManifest]:
        """Scan builtin/ and user extensions for agent manifests.

        Returns list of valid manifests found.

        sensitivity_tier: 1
        """
        self._manifests.clear()
        self._agent_dirs.clear()
        manifests: list[AgentManifest] = []

        for search_dir in [self._builtin_dir, self._user_dir]:
            if not search_dir.exists():
                continue
            for agent_dir in sorted(search_dir.iterdir()):
                manifest_path = agent_dir / "manifest.yaml"
                if not manifest_path.exists():
                    continue
                try:
                    manifest = self.load_manifest(agent_dir)
                    self._manifests[manifest.id] = manifest
                    self._agent_dirs[manifest.id] = agent_dir
                    manifests.append(manifest)
                except (AgentLoadError, Exception) as exc:
                    logger.warning(
                        "Skipping invalid agent at %s: %s",
                        agent_dir,
                        exc,
                    )

        return manifests

    def load_manifest(self, agent_dir: Path) -> AgentManifest:
        """Load and validate manifest.yaml from an agent directory.

        Raises:
            AgentLoadError: If manifest is invalid.

        sensitivity_tier: 1
        """
        manifest_path = agent_dir / "manifest.yaml"
        if not manifest_path.exists():
            msg = f"No manifest.yaml found in {agent_dir}"
            raise AgentLoadError(msg)

        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            msg = f"manifest.yaml must be a mapping, got {type(raw).__name__}"
            raise AgentLoadError(msg)

        return self._validate_manifest(raw)

    def run_agent(
        self,
        agent_id: str,
        trigger: TriggerMode = TriggerMode.MANUAL,
        params: dict[str, Any] | None = None,
    ) -> AgentResult:
        """Execute an agent in-process with sandbox enforcement.

        sensitivity_tier: varies
        """
        if agent_id not in self._manifests:
            self.discover_agents()
        if agent_id not in self._manifests:
            return AgentResult(
                agent_id=agent_id,
                status="error",
                error=f"Agent '{agent_id}' not found",
            )

        manifest = self._manifests[agent_id]
        agent_dir = self._agent_dirs[agent_id]

        try:
            self._running.add(agent_id)
            start = time.monotonic()

            agent_cls = self._load_agent_class(agent_dir)
            guard = SensitivityGuard(agent_id=agent_id, manifest=manifest)
            skills = {
                s.id: s
                for s in self._skill_registry.list_skills()
                if s.id in manifest.skills or not manifest.skills
            }
            from src.models.llm_provider import (
                create_provider_from_settings,
            )

            provider = create_provider_from_settings(background=True)
            context = AgentContext(
                agent_id=agent_id,
                manifest=manifest,
                db_engine=self._db,
                guard=guard,
                skills=skills,
                llm_provider=provider,
            )

            agent = agent_cls()

            # Enforce timeout via signal alarm (Unix only).
            old_handler = None
            if hasattr(signal, "SIGALRM"):
                def _timeout_handler(signum: int, frame: Any) -> None:
                    timeout = manifest.timeout_seconds
                    msg = f"Agent '{agent_id}' exceeded timeout ({timeout}s)"
                    raise TimeoutError(msg)

                old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(manifest.timeout_seconds)

            try:
                result = agent.run(context)
            finally:
                if hasattr(signal, "SIGALRM"):
                    signal.alarm(0)
                    if old_handler is not None:
                        signal.signal(signal.SIGALRM, old_handler)

            duration_ms = (time.monotonic() - start) * 1000

            final_result = AgentResult(
                agent_id=agent_id,
                status=(
                    result.status
                    if isinstance(result, AgentResult) else "success"
                ),
                output=(
                    result.output
                    if isinstance(result, AgentResult) else str(result)
                ),
                tables_written=context.tables_written,
                rows_written=context.rows_written,
                llm_calls=context.llm_calls,
                duration_ms=round(duration_ms, 1),
                error=result.error if isinstance(result, AgentResult) else None,
            )

        except TimeoutError as exc:
            duration_ms = (time.monotonic() - start) * 1000
            final_result = AgentResult(
                agent_id=agent_id,
                status="timeout",
                duration_ms=round(duration_ms, 1),
                error=str(exc),
            )
        except Exception as exc:
            duration_ms = (time.monotonic() - start) * 1000
            logger.exception("Agent '%s' failed: %s", agent_id, exc)
            final_result = AgentResult(
                agent_id=agent_id,
                status="error",
                duration_ms=round(duration_ms, 1),
                error=str(exc),
            )
        finally:
            self._running.discard(agent_id)

        self._last_results[agent_id] = final_result
        return final_result

    def list_agents(self) -> list[AgentStatus]:
        """Return status of all discovered agents.

        sensitivity_tier: 1
        """
        if not self._manifests:
            self.discover_agents()

        statuses: list[AgentStatus] = []
        for agent_id, manifest in self._manifests.items():
            last = self._last_results.get(agent_id)
            status = "running" if agent_id in self._running else "idle"
            if last and last.status == "error":
                status = "error"

            statuses.append(AgentStatus(
                agent_id=manifest.id,
                name=manifest.name,
                description=manifest.description,
                category=manifest.category,
                status=status,
                builtin=manifest.builtin,
                triggers=tuple(t.value for t in manifest.triggers),
                max_sensitivity_tier=manifest.max_sensitivity_tier,
                last_run_at=None,
                last_result=last.status if last else None,
                error=last.error if last else None,
            ))

        return statuses

    def get_agent_result(self, agent_id: str) -> AgentResult | None:
        """Return the last result for a given agent.

        sensitivity_tier: varies
        """
        return self._last_results.get(agent_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_manifest(self, raw: dict[str, Any]) -> AgentManifest:
        """Validate raw YAML dict and convert to AgentManifest.

        sensitivity_tier: 1
        """
        required = ("id", "name", "version", "description", "author")
        for field in required:
            if field not in raw:
                msg = f"Missing required field: '{field}'"
                raise AgentLoadError(msg)

        max_tier = raw.get("max_sensitivity_tier", 1)
        if not (1 <= max_tier <= 3):
            msg = f"max_sensitivity_tier must be 1-3, got {max_tier}"
            raise AgentLoadError(msg)

        memory_mb = raw.get("memory_mb", 256)
        if memory_mb > 1024:
            msg = f"memory_mb must be <= 1024, got {memory_mb}"
            raise AgentLoadError(msg)

        timeout = raw.get("timeout_seconds", 60)
        if timeout > 300:
            msg = f"timeout_seconds must be <= 300, got {timeout}"
            raise AgentLoadError(msg)

        tables = tuple(
            TablePermission(
                table=t["table"],
                max_tier=t.get("max_tier", 1),
                columns=tuple(t.get("columns", ())),
            )
            for t in raw.get("tables", [])
        )

        for tp in tables:
            if not (1 <= tp.max_tier <= 3):
                msg = f"Table '{tp.table}' max_tier must be 1-3, got {tp.max_tier}"
                raise AgentLoadError(msg)

        triggers = tuple(
            TriggerMode(t) for t in raw.get("triggers", ["manual"])
        )

        return AgentManifest(
            id=raw["id"],
            name=raw["name"],
            version=raw["version"],
            description=raw["description"],
            author=raw["author"],
            tables=tables,
            max_sensitivity_tier=max_tier,
            can_use_llm=raw.get("can_use_llm", False),
            write_tables=tuple(raw.get("write_tables", [])),
            skills=tuple(raw.get("skills", [])),
            triggers=triggers,
            schedule=raw.get("schedule"),
            memory_mb=memory_mb,
            timeout_seconds=timeout,
            category=raw.get("category", "general"),
            builtin=raw.get("builtin", False),
        )

    def _load_agent_class(self, agent_dir: Path) -> type[BrainAgent]:
        """Dynamically import agent.py and find the BrainAgent subclass.

        sensitivity_tier: 1
        """
        agent_file = agent_dir / "agent.py"
        if not agent_file.exists():
            msg = f"No agent.py found in {agent_dir}"
            raise AgentLoadError(msg)

        spec = importlib.util.spec_from_file_location(
            f"agent_{agent_dir.name}",
            agent_file,
        )
        if spec is None or spec.loader is None:
            msg = f"Cannot load agent module from {agent_file}"
            raise AgentLoadError(msg)

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, BrainAgent)
                and obj is not BrainAgent
            ):
                return obj

        msg = f"No BrainAgent subclass found in {agent_file}"
        raise AgentLoadError(msg)
