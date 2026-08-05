"""Byte-stability tests for frozen prompt templates.

Two flavors of test per registered prompt:

1. **Hash stability** — the on-disk prefix SHA-256 must match the
   ``expected_hash`` literal below. Any edit to a ``.txt`` file in
   :mod:`src.models.prompts` (down to a trailing space) fails this
   assertion until the literal is updated alongside the prompt.
2. **Render regression** — :meth:`FrozenPromptTemplate.render_combined`
   on the registered fixture input must match the checked-in
   ``tests/fixtures/prompts/<lane>/<name>.golden.txt`` byte-for-byte.

Together these guarantee that:

- We can't accidentally invalidate the provider's prompt cache
  without consciously updating the hash literal (forcing review).
- We can't accidentally change the rendered shape of a prompt
  (changing what the LLM sees) without updating the golden fixture
  (also forcing review).

To register a new prompt:

1. Drop the ``.txt`` file into ``src/models/prompts/``.
2. Add an entry to :data:`REGISTRY` with the relative path,
   expected hash (run the test once with ``""`` to discover it),
   and the fixture path under ``tests/fixtures/prompts/``.
3. Generate the fixture by running render_combined with the
   variable input you want frozen.

sensitivity_tier: N/A
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from src.models.prompt_cache import PROMPTS_DIR, FrozenPromptTemplate

FIXTURES_DIR = (
    Path(__file__).parent.parent.parent.parent
    / "fixtures" / "prompts"
)


@dataclass(frozen=True)
class PromptCase:
    """One registered frozen prompt + its fixture inputs.

    ``variable_input`` is the exact string passed to ``render()``;
    keep it stable so the golden file is reproducible.

    When ``inline=True`` (interpolated prompts), the golden is
    rendered via :meth:`FrozenPromptTemplate.render_inline` — one
    user message with prefix concatenated with the rendered suffix.
    The suffix typically itself contains ``.format()`` calls; the
    caller is responsible for handing in the FULLY-RENDERED suffix
    so the golden captures exactly what hits the LLM.

    sensitivity_tier: 1
    """

    name: str
    lane: str
    template_filename: str
    expected_hash: str
    fixture_filename: str
    variable_input: str = ""
    inline: bool = False


# Populated as each migration commit lands. The whole tier of static
# prompts (8) ships first; interpolated (5) follow. The hash literals
# below are the source of truth — a mismatch means "the prompt was
# edited, here's the new hash; paste it after reviewing the diff".
REGISTRY: list[PromptCase] = [
    # ----- static prompts (Phase 2 first wave) -----
    PromptCase(
        name="sensitivity_classifier",
        lane="classifier",
        template_filename="sensitivity_classifier_v1.txt",
        expected_hash="sha256:30174959f291b82046f6a11685bdea07ec4f75a8afec67462162420e6c0a656f",
        fixture_filename="sensitivity_classifier.golden.txt",
        variable_input="",
    ),
    PromptCase(
        name="triage",
        lane="classifier",
        template_filename="triage_v1.txt",
        expected_hash="sha256:102bd56489fba52c4640292d8fdb110b6b073e47f6469e410353613825bbb4c8",
        fixture_filename="triage.golden.txt",
        variable_input=(
            "[1] id=msg_1 from=Alice source=whatsapp\n"
            "    can we move our 3pm to 4?"
        ),
    ),
    PromptCase(
        name="brain_ask",
        lane="interactive",
        template_filename="brain_ask_v1.txt",
        expected_hash="sha256:643b04298089c8acc036a8bd4797dce89d5f845f0405cbe878f559afc8c15441",
        fixture_filename="brain_ask.golden.txt",
        variable_input="How many meetings did I have last week?",
    ),
    PromptCase(
        name="query_router",
        lane="interactive",
        template_filename="query_router_v1.txt",
        expected_hash="sha256:ae07acc16423c14ab9ba0fe89488071a977bb57b752d9ca9abb72255957fa01c",
        fixture_filename="query_router.golden.txt",
        variable_input="How many meetings did I have last week?",
    ),
    PromptCase(
        name="fact_extractor",
        lane="background",
        template_filename="fact_extractor_v1.txt",
        expected_hash="sha256:389e8fb6ecd993c0c79f0af1b6f701489fbc3cf83dfc010f68971dd1373d0174",
        fixture_filename="fact_extractor.golden.txt",
        variable_input="Alice told Bob she is allergic to peanuts.",
    ),
    PromptCase(
        name="actionable_events",
        lane="escalation",
        template_filename="actionable_events_v1.txt",
        expected_hash="sha256:ae555003f57709b46f5399ebaf0631c99b21ce626218a8e5d7c79d32af98359a",
        fixture_filename="actionable_events.golden.txt",
        variable_input="event: dentist appointment tomorrow at 10am",
    ),
    PromptCase(
        name="message_eval",
        lane="escalation",
        template_filename="message_eval_v1.txt",
        expected_hash="sha256:74557fbdb405dd23bbe619229359c5840049f2424587dc45faf56591e7f068dc",
        fixture_filename="message_eval.golden.txt",
        variable_input=(
            "topic: dad health\n"
            'messages: ["hospital release", "new prescription"]'
        ),
    ),
    PromptCase(
        name="labeler_agent",
        lane="classifier",
        template_filename="labeler_agent_v1.txt",
        expected_hash="sha256:7735b23ab2d896d624c9669b04bc57a054bc37cdc2a9b442586d3b1f64cf4881",
        fixture_filename="labeler_agent.golden.txt",
        variable_input="Just got promoted at work, feeling great!",
    ),
    PromptCase(
        name="dataset_creator",
        lane="background",
        template_filename="dataset_creator_v1.txt",
        expected_hash="sha256:c5c3c4eb1244148470d401a5116a3ebb5516a723d149a5307c810cbea0fbe7a1",
        fixture_filename="dataset_creator.golden.txt",
        variable_input=(
            '{\n'
            '  "agent_id": "user.summarizer",\n'
            '  "available_tools": [],\n'
            '  "description": "Summarize a paragraph in one sentence.",\n'
            '  "existing_case_names": [],\n'
            '  "max_sensitivity_tier": 1,\n'
            '  "name": "Summarizer",\n'
            '  "output_schema": null,\n'
            '  "system_prompt": "You are a summarizer."\n'
            '}'
        ),
    ),
    PromptCase(
        name="model_picker",
        lane="background",
        template_filename="model_picker_v1.txt",
        expected_hash="sha256:8a9807d7834b54d730138ae05532936d59d1368e1999dc571c90dc3a9d115384",
        fixture_filename="model_picker.golden.txt",
        variable_input=(
            '{\n'
            '  "agent_id": "user.summarizer",\n'
            '  "available_local_models": ["llama3.1:8b"],\n'
            '  "available_remote_models": ["Qwen/Qwen2.5-7B-Instruct", '
            '"deepseek-ai/DeepSeek-V3.1"],\n'
            '  "description": "Summarize a paragraph in one sentence.",\n'
            '  "enabled_mcp_tools": [],\n'
            '  "enabled_skills": [],\n'
            '  "max_sensitivity_tier": 1,\n'
            '  "name": "Summarizer",\n'
            '  "output_schema": null,\n'
            '  "system_prompt": "You are a summarizer."\n'
            '}'
        ),
    ),
    PromptCase(
        name="prompt_engineer",
        lane="background",
        template_filename="prompt_engineer_v1.txt",
        expected_hash="sha256:1ca940952c55a150965e6efebb1836a2344b8de10b54a7acf3503a17fd111531",
        fixture_filename="prompt_engineer.golden.txt",
        variable_input=(
            '{\n'
            '  "agent_id": "user.summarizer",\n'
            '  "available_skills": [],\n'
            '  "available_tools": [],\n'
            '  "description": "Resume um parágrafo em uma frase.",\n'
            '  "enabled_mcp_tools": [],\n'
            '  "has_dataset": false,\n'
            '  "max_sensitivity_tier": 1,\n'
            '  "name": "Summarizer",\n'
            '  "output_schema": null,\n'
            '  "prior_eval_failures": [],\n'
            '  "system_prompt": "You are a helpful assistant."\n'
            '}'
        ),
    ),
    # ----- interpolated prompts (Phase 2 second wave) -----
    PromptCase(
        name="labeler_single",
        lane="classifier",
        template_filename="labeler_single_v1.txt",
        expected_hash="sha256:a5eeebbf36b003e645f91ceb8af0daf60a460beffe808cbcde823e3627ef8e7f",
        fixture_filename="labeler_single.golden.txt",
        variable_input="I am ready to go!\n",
        inline=True,
    ),
    PromptCase(
        name="labeler_batch",
        lane="classifier",
        template_filename="labeler_batch_v1.txt",
        expected_hash="sha256:bb340ed9e9d94ac55a24ed1396a3ad52bcb53d13ce6dd1a5162c36a9e72b0e75",
        fixture_filename="labeler_batch.golden.txt",
        variable_input="[1] msg one\n[2] msg two\n",
        inline=True,
    ),
    PromptCase(
        name="llm_classifier",
        lane="classifier",
        template_filename="llm_classifier_v1.txt",
        expected_hash="sha256:286f19293b9b1a802b9e657f753a87d37062a99d0096fa7cf95536160056205a",
        fixture_filename="llm_classifier.golden.txt",
        variable_input=(
            '{"type": "object", "properties": {"x": {"type": "string"}}}'
            "\n\nText:\nHello world\n\nRespond with ONLY a JSON object "
            "matching the schema (no markdown, no explanation).\n"
        ),
        inline=True,
    ),
    PromptCase(
        name="intent_classify",
        lane="interactive",
        template_filename="intent_classify_v1.txt",
        expected_hash="sha256:d97e6152026f2bcc26328562a6a55853cd685bfbf381a4a1a7fce8114663fd95",
        fixture_filename="intent_classify.golden.txt",
        variable_input="",  # filled below at fixture-capture time
        inline=True,
    ),
    PromptCase(
        name="param_extractor",
        lane="background",
        template_filename="param_extractor_v1.txt",
        expected_hash="sha256:56df8b93855cceb1da23212ad9cba3c28e83475e31b7a12c4c9f028839fa0d81",
        fixture_filename="param_extractor.golden.txt",
        variable_input="",
        inline=True,
    ),
    PromptCase(
        name="action_where",
        lane="background",
        template_filename="action_where_v1.txt",
        expected_hash="sha256:84f83a81f12bf4ab99ad1b9f5650e45129bd98022a1ca139716966da5d196e38",
        fixture_filename="action_where.golden.txt",
        variable_input="",
        inline=True,
    ),
]


@pytest.mark.parametrize("case", REGISTRY, ids=lambda c: c.name)
def test_prompt_prefix_byte_stable(case: PromptCase) -> None:
    template = FrozenPromptTemplate(PROMPTS_DIR / case.template_filename)
    assert template.prefix_hash == case.expected_hash, (
        f"\n{case.template_filename} was edited.\n"
        f"  expected: {case.expected_hash}\n"
        f"  actual:   {template.prefix_hash}\n"
        "If this edit was intentional, update REGISTRY in "
        f"{Path(__file__).name} with the new hash AFTER reviewing the diff."
    )


@pytest.mark.parametrize("case", REGISTRY, ids=lambda c: c.name)
def test_prompt_render_matches_golden(case: PromptCase) -> None:
    template = FrozenPromptTemplate(PROMPTS_DIR / case.template_filename)
    if case.inline:
        variable = _resolve_inline_input(case)
        messages = template.render_inline(variable)
        actual = "\n---\n".join(
            f"{m['role']}: {m['content']}" for m in messages
        )
    else:
        actual = template.render_combined(_variables_for(case))
    fixture_path = FIXTURES_DIR / case.lane / case.fixture_filename
    if not fixture_path.exists():
        pytest.fail(
            f"missing fixture {fixture_path} — generate it by capturing "
            f"render output for {case.name} and check it in.",
        )
    expected = fixture_path.read_text(encoding="utf-8")
    assert actual == expected, (
        f"{case.name} render drifted from {fixture_path}. "
        "Re-capture the fixture only after reviewing what changed."
    )


def _variables_for(case: PromptCase) -> dict[str, str]:
    """Translate the case's variable_input into the render_combined dict.

    ``render_combined`` expects a Mapping; the registry stores a
    single string for backward simplicity. Empty string → no vars.

    sensitivity_tier: 1
    """
    if not case.variable_input:
        return {}
    return {"input": case.variable_input}


def _resolve_inline_input(case: PromptCase) -> str:
    """Return the suffix string for an inline case.

    For short suffixes (labeler), the literal lives on
    ``case.variable_input``. For long suffixes (the 3 brain_agent
    prompts and llm_classifier) the literal lives in a sibling
    ``<lane>/<name>.input.txt`` fixture to keep the registry small.

    sensitivity_tier: 1
    """
    if case.variable_input:
        return case.variable_input
    input_path = (
        FIXTURES_DIR / case.lane
        / case.fixture_filename.replace(".golden.txt", ".input.txt")
    )
    if not input_path.exists():
        return ""
    return input_path.read_text(encoding="utf-8")
