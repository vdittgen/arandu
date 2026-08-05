"""Simple terminal chat loop for Brain v2.

Initializes the data layer, builds a QueryEngine and ``BrainAgentV2``,
then reads questions from stdin in an interactive loop.

Usage:
    python -m src.agents.chat_cli

sensitivity_tier: 3 (user interacts with all data tiers)
"""

from __future__ import annotations

import logging
import sys

from src.agents.brain import BrainAgentV2
from src.agents.tool_registry import ToolRegistry
from src.core.data_layer import DataLayer
from src.core.query_engine import QueryEngine
from src.extensions.connectors.catalog import ConnectorCatalog
from src.extensions.connectors.registry import ExtensionRegistry


def main() -> int:
    """Run the interactive chat CLI.

    Returns:
        Exit code (0 for normal exit).

    sensitivity_tier: 3
    """
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    print("Arandu Chat — initializing databases...")
    layer = DataLayer()
    layer.initialize()

    ok, report = layer.health_check()
    if not ok:
        print(f"Health check failed: {report.errors}")
        layer.close()
        return 1

    qe = QueryEngine(
        duckdb=layer.duckdb,
        kuzu=layer.kuzu,
        chromadb=layer.chromadb,
    )
    tool_registry = ToolRegistry(
        catalog=ConnectorCatalog(),
        registry=ExtensionRegistry(),
    )
    agent = BrainAgentV2(
        query_engine=qe,
        tool_registry=tool_registry,
    )

    print("Ready. Type your question (or 'quit' to exit).\n")

    try:
        while True:
            try:
                question = input("You: ").strip()
            except EOFError:
                break

            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                break

            resp = agent.ask(question)

            print(f"\nArandu: {resp.answer}")
            if resp.sources:
                tiers = {s.get("sensitivity_tier") for s in resp.sources}
                print(
                    f"  [{len(resp.sources)} sources | "
                    f"tiers: {sorted(tiers)} | "
                    f"model: {resp.model} | "
                    f"{resp.latency_ms:.0f}ms]",
                )
            print()

    except KeyboardInterrupt:
        print("\n\nGoodbye.")
    finally:
        layer.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
