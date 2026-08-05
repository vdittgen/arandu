"""Extension registry — persistent storage for enabled connector state.

Tracks which connectors are enabled, their configuration, and sync history.
Stored as a JSON file at ~/.arandu/data/extensions.json.

sensitivity_tier: 1 (connector config metadata, no user data)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_PATH = Path.home() / ".arandu" / "data" / "extensions.json"


@dataclass
class RegisteredExtension:
    """A connector that has been enabled by the user.

    sensitivity_tier: 1
    """

    connector_id: str
    enabled: bool = True
    status: str = "connected"  # "connected" | "needs_setup" | "error" | "disabled"
    env_values: dict[str, str] = field(default_factory=dict)
    enabled_at: str = ""
    last_sync_at: str | None = None
    last_success_at: str | None = None
    records_synced: int = 0
    error: str | None = None
    tools_count: int = 0
    command_line: str = ""
    missing_requirements: list[dict[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.enabled_at:
            self.enabled_at = datetime.now(timezone.utc).isoformat()


class ExtensionRegistry:
    """Persistent registry of enabled connectors.

    Reads/writes to a JSON file. Thread-safe for basic operations.

    sensitivity_tier: 1
    """

    def __init__(
        self,
        registry_path: Path = DEFAULT_REGISTRY_PATH,
    ) -> None:
        self._path = registry_path
        self._extensions: dict[str, RegisteredExtension] = {}
        self._load()

    def _load(self) -> None:
        """Load registry from disk.

        sensitivity_tier: 1
        """
        if not self._path.exists():
            self._extensions = {}
            return

        try:
            with self._path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            for entry in data.get("extensions", []):
                ext = RegisteredExtension(**entry)
                self._extensions[ext.connector_id] = ext
            logger.info(
                "Loaded %d extensions from registry",
                len(self._extensions),
            )
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("Failed to load registry: %s", exc)
            self._extensions = {}

    def _save(self) -> None:
        """Persist registry to disk.

        sensitivity_tier: 1
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "extensions": [
                asdict(ext) for ext in self._extensions.values()
            ],
        }
        with self._path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def register(
        self,
        connector_id: str,
        env_values: dict[str, str] | None = None,
        tools_count: int = 0,
        command_line: str = "",
    ) -> RegisteredExtension:
        """Register a connector as enabled.

        sensitivity_tier: 1
        """
        ext = RegisteredExtension(
            connector_id=connector_id,
            enabled=True,
            status="connected",
            env_values=env_values or {},
            tools_count=tools_count,
            command_line=command_line,
            missing_requirements=[],
        )
        self._extensions[connector_id] = ext
        self._save()
        return ext

    def register_needs_setup(
        self,
        connector_id: str,
        missing: list[dict[str, str]] | None = None,
    ) -> RegisteredExtension:
        """Register a connector as enabled but awaiting setup.

        Persists the needs_setup state so catalog refetches show it.

        sensitivity_tier: 1
        """
        ext = RegisteredExtension(
            connector_id=connector_id,
            enabled=True,
            status="needs_setup",
            missing_requirements=missing or [],
        )
        self._extensions[connector_id] = ext
        self._save()
        return ext

    def unregister(self, connector_id: str) -> None:
        """Mark a connector as disabled (preserves the entry).

        sensitivity_tier: 1
        """
        ext = self._extensions.get(connector_id)
        if ext:
            ext.enabled = False
            ext.status = "disabled"
            ext.missing_requirements = []
            self._save()

    def get(self, connector_id: str) -> RegisteredExtension | None:
        """Look up a registered extension.

        sensitivity_tier: 1
        """
        return self._extensions.get(connector_id)

    def get_enabled(self) -> list[RegisteredExtension]:
        """Return all enabled extensions.

        sensitivity_tier: 1
        """
        return [
            ext for ext in self._extensions.values() if ext.enabled
        ]

    def get_all(self) -> list[RegisteredExtension]:
        """Return all registered extensions (including disabled).

        sensitivity_tier: 1
        """
        return list(self._extensions.values())

    def update_sync_stats(
        self,
        connector_id: str,
        records_synced: int,
        error: str | None = None,
    ) -> None:
        """Update sync statistics for a connector.

        sensitivity_tier: 1
        """
        ext = self._extensions.get(connector_id)
        if ext is None:
            return
        ext.last_sync_at = datetime.now(timezone.utc).isoformat()
        ext.records_synced = records_synced
        if error:
            ext.status = "error"
            ext.error = error
            ext.missing_requirements = []
        else:
            ext.status = "connected"
            ext.error = None
            ext.missing_requirements = []
            # Track when rows last actually flowed so the UI can
            # distinguish "last attempt" from "last working sync".
            ext.last_success_at = ext.last_sync_at
        self._save()

    def update_status(
        self,
        connector_id: str,
        status: str,
        error: str | None = None,
    ) -> None:
        """Update the status of a connector.

        sensitivity_tier: 1
        """
        ext = self._extensions.get(connector_id)
        if ext is None:
            return
        ext.status = status
        ext.error = error
        if status != "needs_setup":
            ext.missing_requirements = []
        self._save()

    def update_tools_count(
        self,
        connector_id: str,
        tools_count: int,
    ) -> None:
        """Persist discovered MCP tool count for a connector.

        sensitivity_tier: 1
        """
        ext = self._extensions.get(connector_id)
        if ext is None:
            return
        ext.tools_count = max(0, int(tools_count))
        self._save()

    def is_enabled(self, connector_id: str) -> bool:
        """Check if a connector is currently enabled.

        sensitivity_tier: 1
        """
        ext = self._extensions.get(connector_id)
        return ext is not None and ext.enabled

    def remove(self, connector_id: str) -> None:
        """Completely remove a connector from the registry.

        sensitivity_tier: 1
        """
        self._extensions.pop(connector_id, None)
        self._save()
