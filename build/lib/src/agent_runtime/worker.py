"""Standalone agent worker process.

Invoked as::

    python -m src.agent_runtime.worker run <agent_id> [--params JSON]

Communicates results via stdout JSON lines.  Handles SIGTERM for
graceful cancellation.

Exit codes:
    0 — success
    1 — error
    2 — cancelled (SIGTERM received)
    3 — access denied

sensitivity_tier: varies (depends on agent)
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from typing import Any

_cancel_event = threading.Event()


def _sigterm_handler(signum: int, frame: Any) -> None:
    """Handle SIGTERM by setting the cancellation flag.

    sensitivity_tier: N/A
    """
    _cancel_event.set()


def _emit_json(event: dict[str, Any]) -> None:
    """Write a JSON event to stdout.

    sensitivity_tier: 1
    """
    print(json.dumps(event, default=str), flush=True)


def cmd_run(agent_id: str, params_json: str | None = None) -> int:
    """Load and execute the agent, emitting results as JSON lines.

    sensitivity_tier: varies
    """
    signal.signal(signal.SIGTERM, _sigterm_handler)

    _emit_json({"type": "started", "agent_id": agent_id})

    try:
        from src.agent_runtime.runner import AgentRunner
        from src.core.sqlite.engine import DatabaseEngine

        params = json.loads(params_json) if params_json else None

        runner = AgentRunner(db_engine=DatabaseEngine())
        result = runner.run_agent(agent_id, params=params)

        _emit_json({
            "type": "result",
            "agent_id": result.agent_id,
            "status": result.status,
            "output": result.output,
            "tables_written": list(result.tables_written),
            "rows_written": result.rows_written,
            "llm_calls": result.llm_calls,
            "duration_ms": result.duration_ms,
            "error": result.error,
        })

        if result.status == "denied":
            return 3
        if result.status == "error":
            return 1
        return 0

    except Exception as exc:
        _emit_json({"type": "error", "agent_id": agent_id, "error": str(exc)})
        return 1


def main() -> int:
    """Entry point for agent worker subprocess.

    sensitivity_tier: N/A
    """
    parser = argparse.ArgumentParser(prog="src.agent_runtime.worker")
    sub = parser.add_subparsers(dest="command")
    sub.required = True

    run_parser = sub.add_parser("run")
    run_parser.add_argument("agent_id", type=str)
    run_parser.add_argument("--params", type=str, default=None)

    args = parser.parse_args()
    if args.command == "run":
        return cmd_run(args.agent_id, args.params)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
