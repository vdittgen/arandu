"""Standalone pipeline worker process.

Invoked as: ``python -m src.pipeline.worker run --trigger manual``

Communicates progress via stdout JSON lines. Handles SIGTERM for
graceful cancellation — finishes the current model, records partial
stats, and exits with code 2.

Exit codes:
    0 — success
    1 — error
    2 — cancelled (SIGTERM received)

sensitivity_tier: 1 (infrastructure metrics only)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pipeline lock file — prevents concurrent SQLite write contention.
# The WhatsApp listener (and other long-running writers) check this file
# before each write cycle and skip writes while the pipeline is active.
#
# Staleness protocol (mirrored in bridges/whatsapp/listener.py — keep in
# sync): a live worker heartbeats the lock mtime on every progress event,
# so the lock is stale when the mtime is older than
# ``PIPELINE_LOCK_STALE_SECONDS`` AND the recorded PID is dead — the PID
# check alone is not enough (macOS recycles PIDs quickly) and the mtime
# alone is not enough (a slow-but-alive worker must keep its lock).
# ``PIPELINE_LOCK_MAX_AGE_SECONDS`` is the hard cap past which the lock
# is stolen even from a live PID (matches the Tauri-side auto-reset).
# ---------------------------------------------------------------------------

PIPELINE_LOCK_PATH = Path.home() / ".arandu" / "data" / ".pipeline_running"
PIPELINE_LOCK_STALE_SECONDS = 900
PIPELINE_LOCK_MAX_AGE_SECONDS = 3600


def _is_pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is still running.

    sensitivity_tier: 1
    """
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _acquire_pipeline_lock() -> bool:
    """Create the pipeline lock file, rejecting if another worker is alive.

    Returns True if lock was acquired, False if another worker is running.

    sensitivity_tier: 1
    """
    try:
        PIPELINE_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        if PIPELINE_LOCK_PATH.exists():
            existing_pid_str = PIPELINE_LOCK_PATH.read_text(
                encoding="utf-8",
            ).strip()
            try:
                existing_pid = int(existing_pid_str)
            except (ValueError, TypeError):
                existing_pid = None
            is_other = (
                existing_pid
                and existing_pid != os.getpid()
                and _is_pid_alive(existing_pid)
            )
            try:
                age = time.time() - PIPELINE_LOCK_PATH.stat().st_mtime
            except OSError:
                age = 0.0
            # Stale: no heartbeat for 15 min and the PID is dead (or the
            # hard cap is exceeded — a dead worker's PID may have been
            # recycled by an unrelated live process).
            stale = age > PIPELINE_LOCK_STALE_SECONDS and (
                not is_other or age > PIPELINE_LOCK_MAX_AGE_SECONDS
            )
            if stale:
                PIPELINE_LOCK_PATH.unlink(missing_ok=True)
                logger.info(
                    "Removed stale pipeline lock (%.0fs old)", age,
                )
            elif is_other:
                return False
        PIPELINE_LOCK_PATH.write_text(str(os.getpid()), encoding="utf-8")
        return True
    except OSError as exc:
        logger.warning("Could not create pipeline lock file: %s", exc)
        return True  # Proceed on lock-file errors


def _heartbeat_pipeline_lock() -> None:
    """Refresh the lock file mtime so other processes see us as alive.

    Uses ``os.utime`` (not ``touch``) so a lock already released is NOT
    recreated by a late heartbeat.

    sensitivity_tier: 1
    """
    try:
        os.utime(PIPELINE_LOCK_PATH)
    except OSError:
        pass


def _release_pipeline_lock() -> None:
    """Remove the pipeline lock file.

    sensitivity_tier: 1
    """
    try:
        PIPELINE_LOCK_PATH.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Could not remove pipeline lock file: %s", exc)

# ---------------------------------------------------------------------------
# Cancellation flag
# ---------------------------------------------------------------------------

_cancel_event = threading.Event()


def _sigterm_handler(signum: int, frame: Any) -> None:  # noqa: ARG001
    """Set cancellation flag on SIGTERM.

    The runner checks this flag between models and exits gracefully.

    sensitivity_tier: 1
    """
    _cancel_event.set()


# ---------------------------------------------------------------------------
# JSON-line output
# ---------------------------------------------------------------------------


def _emit_json(event: dict[str, Any]) -> None:
    """Write a single JSON line to stdout.

    Doubles as the worker's liveness tick: every progress event
    refreshes the pipeline lock mtime so other processes (listener,
    future workers) don't mistake a slow run for a crashed one.

    sensitivity_tier: 1
    """
    _heartbeat_pipeline_lock()
    print(json.dumps(event, default=str), flush=True)


# ---------------------------------------------------------------------------
# ChromaDB re-indexing
# ---------------------------------------------------------------------------


def _reindex_chromadb(
    db: Any,
    since: Any,
) -> tuple[str, str | None]:
    """Re-index ChromaDB after a successful pipeline run.

    Falls back to a full reindex when all collections are empty
    (first run, or after a data wipe).  Otherwise performs an
    incremental reindex for records created since *since*.

    Non-fatal to the process (marts are valid even if indexing
    fails), but the outcome is returned so the run record can flag a
    failing index stage.

    Returns ``(status, error)`` where status is ``"success"`` or
    ``"error"`` and error is the verbatim cause (so a dimension
    mismatch message reaches the user) or ``None``.

    sensitivity_tier: 3
    """
    _emit_json({"type": "reindexing", "status": "starting"})
    try:
        from src.core.chromadb.engine import COLLECTION_NAMES, VectorEngine
        from src.core.chromadb.indexer import Indexer

        chroma = VectorEngine()

        # A model/provider switch makes every upsert fail with an
        # opaque dimension error; surface the actionable cause instead
        # of attempting a doomed reindex.
        mismatch = chroma.embedding_mismatch_message()
        if mismatch is not None:
            logger.warning("ChromaDB re-index skipped: %s", mismatch)
            _emit_json({"type": "reindex_error", "error": mismatch})
            return "error", mismatch

        indexer = Indexer(duckdb=db, chromadb=chroma)

        total_docs = sum(
            chroma.get_collection_count(name)
            for name in COLLECTION_NAMES
        )
        if total_docs == 0:
            counts = indexer.full_reindex()
        else:
            counts = indexer.incremental_index(since=since)

        _emit_json({
            "type": "reindex_complete",
            "counts": {k: v for k, v in counts.items()},
        })
        return "success", None
    except Exception as exc:  # noqa: BLE001
        logger.warning("ChromaDB re-index failed: %s", exc)
        _emit_json({
            "type": "reindex_error",
            "error": str(exc),
        })
        return "error", str(exc)


# ---------------------------------------------------------------------------
# Kuzu graph re-indexing
# ---------------------------------------------------------------------------


def _reindex_kuzu(db: Any) -> tuple[str, str | None]:
    """Re-index the Kuzu knowledge graph after a successful pipeline run.

    Full reindex when graph is empty, otherwise incremental.
    Non-fatal to the process, but the outcome is returned so the run
    record can flag a failing graph stage.

    Returns ``(status, error)``.

    sensitivity_tier: 2
    """
    _emit_json({"type": "graph_reindexing", "status": "starting"})
    try:
        from src.core.kuzu.engine import GraphEngine
        from src.core.kuzu.indexer import GraphIndexer
        from src.core.kuzu.schema import create_schema

        # Context-manage the engine: the read-write handle holds Kuzu's
        # exclusive cross-process lock, so release must be deterministic
        # — leaving it to GC keeps read-only query handles (ask/chat)
        # failing on retry for longer than necessary.
        with GraphEngine() as kuzu:
            create_schema(kuzu)

            rows = kuzu.query("MATCH (n) RETURN count(n) AS cnt")
            node_count = rows[0]["cnt"] if rows else 0

            indexer = GraphIndexer(duckdb=db, kuzu=kuzu)
            if node_count == 0:
                counts = indexer.full_reindex()
            else:
                counts = indexer.incremental_index()

        _emit_json({
            "type": "graph_reindex_complete",
            "counts": {k: v for k, v in counts.items()},
        })
        return "success", None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Kuzu re-index failed: %s", exc)
        _emit_json({
            "type": "graph_reindex_error",
            "error": str(exc),
        })
        return "error", str(exc)


# ---------------------------------------------------------------------------
# Notification evaluation
# ---------------------------------------------------------------------------


def _maybe_notify_pipeline(
    db: Any,
    result: Any,
) -> None:
    """Evaluate and optionally notify about pipeline results.

    Delivers over every configured channel — native macOS
    notifications by default, WhatsApp additionally when a phone
    number is configured. Non-fatal: notification errors are logged
    but never fail the pipeline run. Quick-bails if globally muted.

    sensitivity_tier: 2
    """
    try:
        from src.notifications.preference_service import PreferenceService

        prefs = PreferenceService(db_engine=db)

        if prefs.is_muted_globally():
            return

        # WhatsApp phone from settings.json — optional; native
        # delivery doesn't need it.
        phone = _read_whatsapp_phone()

        from src.models.llm_provider import create_provider_from_settings
        from src.notifications.models import DeliveryResult, NotificationRecord
        from src.notifications.notifier import deliver_notification, get_opt_out_text
        from src.notifications.orchestrator import BrainNotificationOrchestrator

        try:
            notif_llm = create_provider_from_settings(background=True)
        except Exception:  # noqa: BLE001
            notif_llm = None

        orchestrator = BrainNotificationOrchestrator(
            preference_service=prefs,
            db_engine=db,
            llm_provider=notif_llm,
        )

        run_result = {}
        if hasattr(result, "__dict__"):
            run_result = {
                k: v for k, v in result.__dict__.items()
                if not k.startswith("_")
            }
        decision = orchestrator.evaluate_pipeline_result(
            run_result=run_result,
            stats={},
        )

        delivery: DeliveryResult
        if decision.should_notify:
            delivery = deliver_notification(
                decision.message, decision.category, whatsapp_phone=phone,
            )
        else:
            delivery = DeliveryResult(
                status="skipped",
                timestamp=_utc_now(),
            )

        prefs.log_notification(
            NotificationRecord(
                id=prefs.new_record_id(),
                dedupe_key=decision.dedupe_key,
                category=decision.category,
                importance_score=decision.importance_score,
                decision="send" if decision.should_notify else "skip",
                delivery_status=delivery.status,
                message=decision.message,
                opt_out_text=get_opt_out_text(decision.category),
                source_type="pipeline",
                source_id=run_result.get("run_id", "unknown"),
                error=delivery.error,
                created_at=_utc_now(),
                message_id=delivery.message_id,
            ),
        )

    except Exception as exc:  # noqa: BLE001
        logger.warning("Pipeline notification evaluation failed: %s", exc)


def _read_whatsapp_phone() -> str | None:
    """Read the WhatsApp phone number from settings.json.

    sensitivity_tier: 1
    """
    try:
        from pathlib import Path

        settings_file = Path.home() / ".arandu" / "settings.json"
        if settings_file.exists():
            data = json.loads(settings_file.read_text(encoding="utf-8"))
            if data.get("notifications_enabled"):
                return data.get("whatsapp_notification_phone") or None
    except Exception:  # noqa: BLE001
        pass
    return None


def _utc_now() -> str:
    """Return the current UTC time as ISO 8601.

    sensitivity_tier: 1
    """
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Smart plan helper
# ---------------------------------------------------------------------------


def _build_smart_plan(
    db: Any,
    runner: Any,
) -> tuple[list[str] | None, str | None]:
    """Generate a smart refresh plan and emit it as a JSON event.

    Returns ``(select_models, plan_summary)`` on success, or
    ``(None, None)`` if the brain raises (falls back to full run).

    sensitivity_tier: 1
    """
    try:
        from src.core.query_tracker import QueryTracker
        from src.pipeline.pipeline_brain import PipelineBrain

        tracker = QueryTracker(db_engine=db)
        brain = PipelineBrain(
            query_tracker=tracker,
            pipeline_runner=runner,
        )
        brain.check_demand_for_new_marts()
        plan = brain.plan_refresh()
        _emit_json({"type": "plan", **plan.to_dict()})

        if not plan.models:
            _emit_json({
                "type": "done",
                "status": "nothing_to_do",
                "step_index": 0,
                "total_steps": 0,
                "elapsed_seconds": 0.0,
            })
            return [], plan.summary()

        return plan.get_ordered(), plan.summary()
    except Exception:  # noqa: BLE001
        logger.warning(
            "Smart plan failed, falling back to full run",
            exc_info=True,
        )
        return None, None


# ---------------------------------------------------------------------------
# Run command
# ---------------------------------------------------------------------------


def cmd_run(trigger: str, mode: str = "full") -> int:
    """Execute the pipeline and stream progress as JSON lines.

    When *mode* is ``"smart"``, the Pipeline Brain generates a
    prioritized plan and only selected models are executed.  Falls
    back to a full run if the brain raises.

    sensitivity_tier: 1
    """
    signal.signal(signal.SIGTERM, _sigterm_handler)

    from src.core.sqlite.engine import DatabaseEngine
    from src.pipeline.runner import PipelineRunner
    from src.pipeline.stats import ProcessingStats

    if not _acquire_pipeline_lock():
        _emit_json({
            "type": "error",
            "error": "Another pipeline worker is already running. "
                     "Wait for it to finish or kill the other process.",
        })
        return 1
    try:
        db = DatabaseEngine()
        stats = ProcessingStats()
        runner = PipelineRunner(duckdb=db, stats=stats)

        select_models = None
        plan_summary = None
        if mode == "smart":
            select_models, plan_summary = _build_smart_plan(
                db, runner,
            )
            # Empty list means nothing to do — skip the run.
            if select_models is not None and len(select_models) == 0:
                return 0

        result = runner.run(
            trigger=trigger,
            on_progress=_emit_json,
            cancel_check=_cancel_event.is_set,
            select_models=select_models,
        )
        if plan_summary is not None:
            result.plan_summary = plan_summary

        if result.status == "cancelled":
            return 2

        # Re-index ChromaDB and Kuzu after successful pipeline run so
        # the QueryEngine has fresh embeddings and graph for chat queries.
        if result.status == "success":
            vector_status, vector_error = _reindex_chromadb(
                db, result.started_at,
            )
            graph_status, graph_error = _reindex_kuzu(db)
            # The run record was written before re-index ran; patch it
            # so a later pipeline-status poll reflects a failing index
            # stage even though the marts themselves succeeded.
            stats.update_index_status(
                run_id=result.run_id,
                vector_index_status=vector_status,
                graph_index_status=graph_status,
                index_error=vector_error or graph_error,
            )
            _maybe_notify_pipeline(db, result)

        return 0 if result.status == "success" else 1
    except Exception as exc:  # noqa: BLE001
        logger.exception("Pipeline worker failed: %s", exc)
        _emit_json({"type": "error", "error": str(exc), "elapsed_seconds": 0})
        return 1
    finally:
        _release_pipeline_lock()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Parse arguments and dispatch to the run command.

    sensitivity_tier: 1
    """
    parser = argparse.ArgumentParser(
        prog="src.pipeline.worker",
        description="Standalone pipeline worker process",
    )
    sub = parser.add_subparsers(dest="command")
    run_parser = sub.add_parser("run", help="Execute the pipeline")
    run_parser.add_argument(
        "--trigger",
        type=str,
        default="manual",
        help="Trigger label (default: manual)",
    )
    run_parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["smart", "full"],
        help="Run mode (default: full)",
    )

    args = parser.parse_args()

    if args.command == "run":
        return cmd_run(args.trigger, args.mode)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
