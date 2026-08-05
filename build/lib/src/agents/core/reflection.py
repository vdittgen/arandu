"""Self-review checkpoint for orchestrator runs.

The reflective runner (in :mod:`src.agents.core.agent_base`) calls
:meth:`Reflector.review` at each wall-clock checkpoint. The Reflector
makes a single Tier A LLM call that returns a
:class:`ReflectionVerdict`:

- ``continue_research=False`` — the runner injects a ``STOP_REQUEST``
  user message; the model finalizes its answer with the context it
  already has.
- ``continue_research=True`` — the runner keeps going. If
  ``suggested_class`` promotes to ``interactive_deep``, the runner
  swaps to that budget's cadence and emits
  ``extended_research_announced`` so the UI can show a "Researching
  deeper" banner with a Stop button.

In Arandu the reflector uses the same local model as everything
else (one Ollama model per install). The checkpoint adds ~1-2s of
wall-clock per review, which is acceptable at the budget cadence
(10s or 30s between checks).

sensitivity_tier: 1 (only ToolCallEntry summaries, not raw content)
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from src.agents.core.agent_base import SBAgent
from src.agents.core.output_types import ReflectionVerdict
from src.agents.core.scheduler import Tier
from src.agents.core.task_budget import TaskClass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolCallEntry:
    """One row in the tool log fed to the Reflector.

    Captured by the orchestrator's tool wrappers as each tool call
    finishes. Holds only summaries — no raw tool results — so the
    Reflector prompt stays small and free of Tier 2/3 payload.

    sensitivity_tier: 1 (summaries only, not raw content)
    """

    name: str
    args_summary: str
    result_summary: str
    duration_ms: float
    status: str  # "ok" | "error"


REFLECTOR_SYSTEM_PROMPT = (
    "You are the self-review checkpoint inside an AI assistant. Another "
    "agent is in the middle of answering a user's question. It has used "
    "some tools and is deciding whether to continue researching or wrap "
    "up.\n\n"
    "Your job: read the original question, the tool log, and the "
    "elapsed time. Decide:\n\n"
    "- continue_research: true if the question genuinely needs more "
    "research; false if the agent has enough context to write a useful "
    "answer now. Default to FALSE when the question is simple, lookup-"
    "style, or the tools already covered the key facts.\n"
    "- reason: one short sentence (under 120 chars). When continuing, "
    "explain WHY more research is justified — the user will see this "
    "in a 'Researching deeper' banner.\n"
    "- suggested_class: 'interactive_deep' if the question is complex "
    "and warrants a longer budget; otherwise 'interactive_fast'.\n\n"
    "Be conservative about extending. Most chat questions answer in <10s "
    "with 1-3 tools. Only extend when the tool log shows real progress "
    "is still being made and the question explicitly asks for synthesis "
    "across many sources, deep analysis, or planning."
)


class ReflectorAgent(SBAgent[str, ReflectionVerdict]):
    """Tier A self-review classifier.

    Registered with ``agent_id="reflector"`` so the tier map in
    :mod:`src.agents.core.model_tiers` routes it to Tier A. Single
    LLM call, no tools — uses ``SBAgent.run`` directly.

    sensitivity_tier: 1
    """

    agent_id = "reflector"
    output_type = ReflectionVerdict
    tier = Tier.SYSTEM
    system_prompt = REFLECTOR_SYSTEM_PROMPT


def _render_tool_log(tool_log: Sequence[ToolCallEntry]) -> str:
    """Render the tool log as compact lines for the Reflector prompt.

    Each line: ``[idx] name(args) → result (status, Nms)``. Truncated
    to 60 chars per field to keep the prompt small.

    sensitivity_tier: 1
    """
    if not tool_log:
        return "(no tools called yet)"
    lines: list[str] = []
    for idx, entry in enumerate(tool_log, start=1):
        args = entry.args_summary[:60]
        result = entry.result_summary[:60]
        lines.append(
            f"[{idx}] {entry.name}({args}) → {result} "
            f"({entry.status}, {entry.duration_ms:.0f}ms)",
        )
    return "\n".join(lines)


class Reflector:
    """Wrapper that builds the Reflector prompt and calls the agent.

    Kept separate from :class:`ReflectorAgent` so callers can pass a
    fake Reflector in tests without monkeypatching the registry.

    sensitivity_tier: 1
    """

    def __init__(self, agent: ReflectorAgent | None = None) -> None:
        self._agent = agent or ReflectorAgent()

    def review(
        self,
        original_question: str,
        tool_log: Sequence[ToolCallEntry],
        elapsed_s: float,
        current_class: TaskClass,
    ) -> ReflectionVerdict:
        """Return a verdict for the current run state.

        On any LLM failure, returns a "continue=False" verdict — failing
        closed (stop researching, finalize with current context) is the
        safer default than risking an infinite loop on a broken
        Reflector.

        sensitivity_tier: 1
        """
        prompt = self._build_prompt(
            original_question, tool_log, elapsed_s, current_class,
        )
        try:
            record = self._agent.run(prompt)
        except Exception:  # noqa: BLE001
            logger.exception("Reflector LLM call failed; failing closed")
            return ReflectionVerdict(
                continue_research=False,
                reason="reflection unavailable; wrapping up",
                suggested_class="interactive_fast",
            )
        if record.output is None or record.error is not None:
            logger.warning(
                "Reflector produced no output (error=%s); failing closed",
                record.error,
            )
            return ReflectionVerdict(
                continue_research=False,
                reason="reflection unavailable; wrapping up",
                suggested_class="interactive_fast",
            )
        return record.output

    @staticmethod
    def _build_prompt(
        original_question: str,
        tool_log: Sequence[ToolCallEntry],
        elapsed_s: float,
        current_class: TaskClass,
    ) -> str:
        """Render the Reflector user message.

        sensitivity_tier: 2 (carries the user's question)
        """
        return (
            f"Original question:\n{original_question}\n\n"
            f"Tool log so far:\n{_render_tool_log(tool_log)}\n\n"
            f"Elapsed: {elapsed_s:.1f}s\n"
            f"Current class: {current_class.value}\n\n"
            "Return a ReflectionVerdict."
        )


def register_reflector_agent() -> None:
    """Register the Reflector in the global agent registry.

    Skipped if already registered — safe to call multiple times during
    test setup or hot reloads.

    sensitivity_tier: 1
    """
    from src.agents.core.config_store import AgentConfig
    from src.agents.core.registry import (
        AgentDefinition,
        get_agent,
        register_agent,
    )

    if get_agent("reflector") is not None:
        return

    default = AgentConfig(
        agent_id="reflector",
        system_prompt=REFLECTOR_SYSTEM_PROMPT,
        model_route="inherit",
        model_override=None,
        enabled_tools=(),
        enabled_skills=(),
        editable=False,
    )
    register_agent(AgentDefinition(
        agent_id="reflector",
        name="Self-review checkpoint",
        description=(
            "Decides whether an orchestrator should continue researching "
            "or wrap up with the context it already has. Fires at each "
            "wall-clock reflection checkpoint inside Brain / Chat."
        ),
        category="classifier",
        parent_agent=None,
        tier=Tier.SYSTEM,
        max_sensitivity_tier=1,
        editable=False,
        default_config=default,
        available_tools=(),
        available_skills=(),
        output_schema="ReflectionVerdict",
        pattern="single",
        factory=ReflectorAgent,
        tags=("system", "classifier"),
    ))


__all__ = [
    "REFLECTOR_SYSTEM_PROMPT",
    "Reflector",
    "ReflectorAgent",
    "ToolCallEntry",
    "register_reflector_agent",
]
