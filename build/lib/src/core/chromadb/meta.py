"""Embedding-meta sentinel for ChromaDB.

A single JSON file at ``~/.arandu/data/chromadb/.embedding_meta.json``
records the model + dimension that built the current index. On
:class:`VectorEngine` init we compare it against the active embedding
function and emit a loud warning when they don't match — a mismatch
means the existing vectors were embedded by a different model and
queries will return garbage until the user runs ``migrate``.

We don't auto-rebuild: silently re-embedding tens of thousands of
documents on a settings change would either burn tokens (remote) or
block the UI for minutes (local). The user triggers the rebuild via
``python -m src.core.chromadb.migrate`` once they've considered the
cost.

sensitivity_tier: N/A (infrastructure metadata only)
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

META_FILENAME = ".embedding_meta.json"


@dataclass(frozen=True)
class EmbeddingMeta:
    """Recorded model identity at the time the index was last built.

    ``provider`` is the broad family (``"ollama"``, ``"openai"``,
    ``"voyage"``); ``model_name`` is the model id (e.g.
    ``"nomic-embed-text"`` or ``"text-embedding-3-large"``);
    ``dimension`` is the vector size and is the hard compatibility
    check — dimension mismatch is unrecoverable without a rebuild.

    sensitivity_tier: N/A
    """

    provider: str
    model_name: str
    dimension: int
    created_at: str


def meta_path(db_path: Path) -> Path:
    """Conventional sentinel path inside the ChromaDB directory.

    sensitivity_tier: N/A
    """
    return db_path / META_FILENAME


def read_meta(db_path: Path) -> EmbeddingMeta | None:
    """Load the sentinel, returning ``None`` when absent or malformed.

    sensitivity_tier: N/A
    """
    path = meta_path(db_path)
    if not path.exists():
        return None
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
        return EmbeddingMeta(
            provider=str(body["provider"]),
            model_name=str(body["model_name"]),
            dimension=int(body["dimension"]),
            created_at=str(body.get("created_at", "")),
        )
    except (json.JSONDecodeError, KeyError, OSError, ValueError) as exc:
        logger.warning("ignoring malformed embedding meta at %s: %s", path, exc)
        return None


def write_meta(db_path: Path, meta: EmbeddingMeta) -> None:
    """Atomically replace the sentinel file.

    Writes to a sibling tempfile then renames — avoids leaving a
    half-written meta if the process is killed mid-write.

    sensitivity_tier: N/A
    """
    db_path.mkdir(parents=True, exist_ok=True)
    target = meta_path(db_path)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".embedding_meta.",
        suffix=".tmp",
        dir=str(db_path),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(asdict(meta), f, indent=2, sort_keys=True)
        os.replace(tmp_path, target)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def current_meta(
    provider: str,
    model_name: str,
    dimension: int,
) -> EmbeddingMeta:
    """Build a fresh meta with the current UTC timestamp.

    sensitivity_tier: N/A
    """
    return EmbeddingMeta(
        provider=provider,
        model_name=model_name,
        dimension=dimension,
        created_at=datetime.now(tz=UTC).isoformat(),
    )


def check_compatibility(
    db_path: Path,
    active_provider: str,
    active_model: str,
    active_dimension: int,
) -> EmbeddingMeta | None:
    """Read the sentinel and compare against the active embedder.

    Returns the *stored* meta when there is a mismatch (caller logs /
    notifies); returns ``None`` when compatible or when no sentinel
    exists yet (fresh index). Side-effect: writes a new sentinel on
    first run so subsequent launches can detect drift.

    sensitivity_tier: N/A
    """
    stored = read_meta(db_path)
    if stored is None:
        # First run with sentinel support — record what's currently
        # in use so the next launch can compare.
        write_meta(
            db_path,
            current_meta(active_provider, active_model, active_dimension),
        )
        return None

    if (
        stored.provider == active_provider
        and stored.model_name == active_model
        and stored.dimension == active_dimension
    ):
        return None

    return stored
