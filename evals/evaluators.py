"""Custom evaluators for Arandu agent eval suites.

The pydantic-evals built-ins (``Equals``, ``IsInstance``, ``Contains``)
cover most cases; this module adds domain-specific ones that compare
typed agent outputs (TriageBatch, SensitivityVerdict, etc.) against
ground-truth shapes encoded in YAML.

All evaluators here are **structural** (no LLM round-trip) so the
fast eval pass can run in CI without an API key. The semantic
:class:`LLMJudgeOnReason` wrapper is provided for the suites that
benefit from it but degrades to a "skipped" verdict when no model is
available.

sensitivity_tier: N/A
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

from pydantic_evals.evaluators import (
    EvaluationReason,
    Evaluator,
    EvaluatorContext,
)

# ---------------------------------------------------------------------------
# Sensitivity classifier
# ---------------------------------------------------------------------------


@dataclass
class TierEquals(Evaluator):
    """Pass when the agent's tier matches the expected tier exactly.

    Inputs to the dataset are strings; outputs are ``SensitivityVerdict``
    or any object with a ``.tier`` attribute.

    sensitivity_tier: N/A
    """

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        output = ctx.output
        tier = getattr(output, "tier", None)
        if tier is None and isinstance(output, dict):
            tier = output.get("tier")
        if tier is None:
            return EvaluationReason(value=False, reason="no tier in output")
        expected = (ctx.expected_output or {}).get("tier")
        if expected is None:
            return EvaluationReason(value=False, reason="no expected tier")
        ok = int(tier) == int(expected)
        return EvaluationReason(
            value=ok,
            reason=(
                f"got tier={tier}; expected={expected}"
                if not ok
                else f"tier={tier}"
            ),
        )


# ---------------------------------------------------------------------------
# Triage
# ---------------------------------------------------------------------------


@dataclass
class TriageDecisionAccuracy(Evaluator):
    """Pass when every TriageDecision matches expected keep/drop flags.

    ``ctx.expected_output`` shape::

        {"decisions": [{"message_id": ..., "keep": bool,
                         "is_promo": bool, ...}, ...]}

    sensitivity_tier: N/A
    """

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        output = ctx.output
        decisions = getattr(output, "decisions", None)
        if decisions is None and isinstance(output, dict):
            decisions = output.get("decisions")
        if decisions is None:
            return EvaluationReason(
                value=False, reason="no decisions in output",
            )
        expected = (ctx.expected_output or {}).get("decisions", [])
        actual = {
            (d.message_id if hasattr(d, "message_id") else d["message_id"]): {
                "keep": d.keep if hasattr(d, "keep") else d["keep"],
                "is_promo": _attr(d, "is_promo"),
                "is_automated": _attr(d, "is_automated"),
                "is_ack_only": _attr(d, "is_ack_only"),
            }
            for d in decisions
        }
        mismatches: list[str] = []
        for exp in expected:
            mid = exp["message_id"]
            got = actual.get(mid)
            if got is None:
                mismatches.append(f"{mid}: missing")
                continue
            for key in ("keep", "is_promo", "is_automated", "is_ack_only"):
                if key in exp and exp[key] != got.get(key, False):
                    mismatches.append(
                        f"{mid}.{key}: got {got.get(key)!r}, "
                        f"expected {exp[key]!r}",
                    )
        if mismatches:
            return EvaluationReason(
                value=False, reason="; ".join(mismatches),
            )
        return EvaluationReason(
            value=True, reason=f"{len(expected)} decisions matched",
        )


def _attr(obj: Any, key: str) -> bool:
    if hasattr(obj, key):
        return getattr(obj, key)
    if isinstance(obj, dict):
        return bool(obj.get(key, False))
    return False


# ---------------------------------------------------------------------------
# Emotional labeler
# ---------------------------------------------------------------------------


@dataclass
class EmotionalLabelStructural(Evaluator):
    """Pass when ``primary_emotion`` + ``domain`` match and ``intensity``
    is within ``intensity_tolerance`` of expected.

    sensitivity_tier: N/A
    """

    intensity_tolerance: float = 0.25

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        output = ctx.output
        expected = ctx.expected_output or {}
        actual = {
            "primary_emotion": getattr(output, "primary_emotion", None),
            "intensity": getattr(output, "intensity", None),
            "domain": getattr(output, "domain", None),
        }
        problems: list[str] = []
        if (
            "primary_emotion" in expected
            and actual["primary_emotion"] != expected["primary_emotion"]
        ):
            problems.append(
                f"emotion: got {actual['primary_emotion']!r}, "
                f"expected {expected['primary_emotion']!r}",
            )
        if "domain" in expected and actual["domain"] != expected["domain"]:
            problems.append(
                f"domain: got {actual['domain']!r}, "
                f"expected {expected['domain']!r}",
            )
        if "intensity" in expected and actual["intensity"] is not None:
            diff = abs(actual["intensity"] - expected["intensity"])
            if diff > self.intensity_tolerance:
                problems.append(
                    f"intensity off by {diff:.2f} (>"
                    f"{self.intensity_tolerance})",
                )
        if problems:
            return EvaluationReason(
                value=False, reason="; ".join(problems),
            )
        return EvaluationReason(value=True, reason="label matches")


# ---------------------------------------------------------------------------
# Firewall
# ---------------------------------------------------------------------------


@dataclass
class FirewallAllowedMatches(Evaluator):
    """Pass when the firewall's ``allowed`` flag matches expected.

    Optionally checks ``category`` if present in expected output.

    sensitivity_tier: N/A
    """

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        output = ctx.output
        allowed = getattr(output, "allowed", None)
        if allowed is None and isinstance(output, dict):
            allowed = output.get("allowed")
        expected = ctx.expected_output or {}
        exp_allowed = expected.get("allowed")
        if exp_allowed is None:
            return EvaluationReason(
                value=False, reason="no expected allowed flag",
            )
        if bool(allowed) != bool(exp_allowed):
            return EvaluationReason(
                value=False,
                reason=f"allowed={allowed}, expected={exp_allowed}",
            )
        exp_category = expected.get("category")
        if exp_category is not None:
            cat = getattr(output, "category", None)
            if cat != exp_category:
                return EvaluationReason(
                    value=False,
                    reason=f"category={cat}, expected={exp_category}",
                )
        return EvaluationReason(value=True, reason=f"allowed={allowed}")


# ---------------------------------------------------------------------------
# Confidence range — generic shape evaluator
# ---------------------------------------------------------------------------


@dataclass
class ConfidenceInRange(Evaluator):
    """Pass when the agent's confidence is within ``[lo, hi]``.

    sensitivity_tier: N/A
    """

    lo: float = 0.0
    hi: float = 1.0

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        output = ctx.output
        confidence = getattr(output, "confidence", None)
        if confidence is None and isinstance(output, dict):
            confidence = output.get("confidence")
        if confidence is None:
            return EvaluationReason(
                value=False, reason="no confidence in output",
            )
        if not (self.lo <= confidence <= self.hi):
            return EvaluationReason(
                value=False,
                reason=(
                    f"confidence={confidence:.2f} outside "
                    f"[{self.lo}, {self.hi}]"
                ),
            )
        return EvaluationReason(value=True, reason=f"confidence={confidence:.2f}")


# ---------------------------------------------------------------------------
# Generic structural primitives — used by most datasets via YAML
# ---------------------------------------------------------------------------


def _resolve_attr(output: Any, dotted: str) -> Any:
    """Walk an attribute path (or dict key chain), returning None on miss.

    Accepts ``a.b.c`` to reach into nested attributes; dict-like
    intermediates are looked up via ``get`` so YAML-loaded payloads
    work unchanged. Integer parts (``replies.0.message_id``) index
    into lists, falling through to the dict/attr lookup if the
    segment isn't a sequence. ``*`` flattens across the current list:
    ``replies.*.reason`` returns a list of the ``reason`` from each
    reply, dropping items where the remainder of the path resolves to
    None.

    sensitivity_tier: N/A
    """
    cur: Any = output
    parts = dotted.split(".")
    for i, part in enumerate(parts):
        if cur is None:
            return None
        if part == "*":
            if not isinstance(cur, (list, tuple)):
                return None
            remainder = ".".join(parts[i + 1:])
            if not remainder:
                return list(cur)
            collected: list[Any] = []
            for item in cur:
                value = _resolve_attr(item, remainder)
                if value is not None:
                    collected.append(value)
            return collected
        if part.isdigit() and isinstance(cur, (list, tuple)):
            idx = int(part)
            if idx >= len(cur) or idx < -len(cur):
                return None
            cur = cur[idx]
            continue
        if hasattr(cur, part):
            cur = getattr(cur, part)
        elif isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def _judge_payload(target: Any) -> str:
    """Render a resolved field value for the LLM judge.

    Lists (from a ``*`` wildcard) are emitted as numbered items so the
    judge can grade each entry distinctly under a rubric that talks
    about "each" item. Scalars round-trip through ``str``.

    sensitivity_tier: N/A
    """
    if isinstance(target, (list, tuple)):
        return "\n".join(
            f"[{i + 1}] {item}" for i, item in enumerate(target)
        )
    return str(target)


def _is_missing_payload(target: Any) -> bool:
    """Treat ``None`` and empty wildcard collections as missing.

    sensitivity_tier: N/A
    """
    if target is None:
        return True
    if isinstance(target, (list, tuple)) and not target:
        return True
    return False


@dataclass
class FieldEquals(Evaluator):
    """Pass when ``output.<field>`` equals ``value``.

    ``field`` may be dotted to descend into sub-models / dicts.

    sensitivity_tier: N/A
    """

    field: str = ""
    value: Any = None

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        got = _resolve_attr(ctx.output, self.field)
        if got == self.value:
            return EvaluationReason(value=True, reason=f"{self.field}={got!r}")
        return EvaluationReason(
            value=False,
            reason=f"{self.field}: got {got!r}, expected {self.value!r}",
        )


@dataclass
class DelegatedTo(Evaluator):
    """Pass when an orchestrator delegated to exactly the right child.

    Reads the ``delegated`` list an orchestrator task adapter records
    (see :func:`evals.tasks.task_curator_task`) and compares it to
    ``expected``. Routing to the right child *plus* extra children is
    still a miss: each extra delegation is a wasted LLM call, and for
    the curator it means the user waits on e.g. a scheduler run they
    never asked for.

    ``expected: []`` asserts the orchestrator answered directly without
    delegating at all.

    sensitivity_tier: N/A
    """

    expected: list[str] = dc_field(default_factory=list)

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        output = ctx.output
        if not isinstance(output, dict) or "delegated" not in output:
            return EvaluationReason(
                value=False,
                reason="output has no 'delegated' list",
            )
        actual = list(output["delegated"])
        if actual == list(self.expected):
            return EvaluationReason(
                value=True,
                reason=f"delegated to {actual or 'nobody'}",
            )
        return EvaluationReason(
            value=False,
            reason=(
                f"delegated to {actual or 'nobody'}, "
                f"expected {list(self.expected) or 'nobody'}"
            ),
        )


@dataclass
class FieldNotEmpty(Evaluator):
    """Pass when ``output.<field>`` is truthy (non-empty / non-zero).

    sensitivity_tier: N/A
    """

    field: str = ""

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        got = _resolve_attr(ctx.output, self.field)
        if got:
            return EvaluationReason(value=True, reason=f"{self.field} populated")
        return EvaluationReason(
            value=False, reason=f"{self.field} is empty/missing",
        )


@dataclass
class FieldIn(Evaluator):
    """Pass when ``output.<field>`` is a member of ``choices``.

    sensitivity_tier: N/A
    """

    field: str = ""
    choices: tuple[Any, ...] = ()

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        got = _resolve_attr(ctx.output, self.field)
        if got in self.choices:
            return EvaluationReason(value=True, reason=f"{self.field}={got!r}")
        return EvaluationReason(
            value=False,
            reason=f"{self.field}={got!r} not in {list(self.choices)!r}",
        )


@dataclass
class FieldContains(Evaluator):
    """Pass when ``output.<field>`` (a string) contains ``substring``.

    Case-insensitive by default.

    sensitivity_tier: N/A
    """

    field: str = ""
    substring: str = ""
    case_sensitive: bool = False

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        got = _resolve_attr(ctx.output, self.field) or ""
        hay = str(got) if self.case_sensitive else str(got).lower()
        needle = self.substring if self.case_sensitive else self.substring.lower()
        if needle in hay:
            return EvaluationReason(
                value=True,
                reason=f"{self.field} contains {self.substring!r}",
            )
        return EvaluationReason(
            value=False,
            reason=f"{self.field} missing {self.substring!r}",
        )


@dataclass
class ListLengthInRange(Evaluator):
    """Pass when ``len(output.<field>)`` is within ``[min_n, max_n]``.

    sensitivity_tier: N/A
    """

    field: str = ""
    min_n: int = 0
    max_n: int = 1_000_000

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        got = _resolve_attr(ctx.output, self.field)
        if got is None:
            return EvaluationReason(
                value=False, reason=f"{self.field} missing",
            )
        try:
            n = len(got)
        except TypeError:
            return EvaluationReason(
                value=False, reason=f"{self.field} is not a sequence",
            )
        if self.min_n <= n <= self.max_n:
            return EvaluationReason(value=True, reason=f"{self.field} len={n}")
        return EvaluationReason(
            value=False,
            reason=(
                f"{self.field} len={n} outside [{self.min_n}, {self.max_n}]"
            ),
        )


@dataclass
class IntInRange(Evaluator):
    """Pass when ``output.<field>`` is an int within ``[lo, hi]``.

    sensitivity_tier: N/A
    """

    field: str = ""
    lo: int = 0
    hi: int = 10

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        got = _resolve_attr(ctx.output, self.field)
        if got is None:
            return EvaluationReason(
                value=False, reason=f"{self.field} missing",
            )
        try:
            val = int(got)
        except (TypeError, ValueError):
            return EvaluationReason(
                value=False,
                reason=f"{self.field}={got!r} not an int",
            )
        if self.lo <= val <= self.hi:
            return EvaluationReason(value=True, reason=f"{self.field}={val}")
        return EvaluationReason(
            value=False,
            reason=f"{self.field}={val} outside [{self.lo}, {self.hi}]",
        )


# ---------------------------------------------------------------------------
# Semantic / LLM-judge evaluators
# ---------------------------------------------------------------------------


@dataclass
class LLMJudgeOnReason(Evaluator):
    """Grade ``output.reason`` against a rubric via the remote LLM judge.

    The judge agent (``evals.judge.grade``) returns a 0-10 score and
    short reason. ``threshold`` defines the passing score (default 7,
    matching the rubric language: "solid response that meets the
    rubric"). When the judge is unreachable the case is recorded with
    ``value=True`` and a ``skipped: ...`` reason so offline runs
    surface only structural failures.

    sensitivity_tier: N/A
    """

    rubric: str = "Reason is specific and grounded in the input."
    threshold: int = 7
    field: str = "reason"  # path into output

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        from evals.judge import grade

        target = _resolve_attr(ctx.output, self.field)
        if _is_missing_payload(target):
            return EvaluationReason(
                value=False,
                reason=f"{self.field} missing from output",
            )
        verdict = grade(
            rubric=self.rubric,
            inputs=ctx.inputs,
            output_text=_judge_payload(target),
            threshold=self.threshold,
        )
        if verdict is None:
            return EvaluationReason(
                value=True,
                reason="skipped: judge unavailable",
            )
        suffix = f" — {verdict.reason}" if verdict.reason else ""
        return EvaluationReason(
            value=verdict.passed,
            reason=(
                f"score={verdict.score}/10 (>= {self.threshold}){suffix}"
            ),
        )


@dataclass
class LLMJudgeOnField(Evaluator):
    """Same as :class:`LLMJudgeOnReason` but for any field path.

    Use this for prose outputs like ``answer`` (BrainResponse),
    ``content`` (InsightDraft), or ``nudge`` (RelationshipNudge).
    ``field`` may be dotted (``a.b.c``) to descend into nested models.

    sensitivity_tier: N/A
    """

    field: str = "answer"
    rubric: str = ""
    threshold: int = 7

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        from evals.judge import grade

        target = _resolve_attr(ctx.output, self.field)
        if _is_missing_payload(target):
            return EvaluationReason(
                value=False,
                reason=f"{self.field} missing from output",
            )
        verdict = grade(
            rubric=self.rubric,
            inputs=ctx.inputs,
            output_text=_judge_payload(target),
            threshold=self.threshold,
        )
        if verdict is None:
            return EvaluationReason(
                value=True,
                reason="skipped: judge unavailable",
            )
        suffix = f" — {verdict.reason}" if verdict.reason else ""
        return EvaluationReason(
            value=verdict.passed,
            reason=(
                f"{self.field}: score={verdict.score}/10 "
                f"(>= {self.threshold}){suffix}"
            ),
        )


# ---------------------------------------------------------------------------
# Set-shaped batch evaluators
# ---------------------------------------------------------------------------


def _items_for_set_eval(
    output: Any, field: str,
) -> list[Any] | None:
    """Return the list at ``output.<field>`` or None.

    sensitivity_tier: N/A
    """
    seq = _resolve_attr(output, field)
    if seq is None:
        return None
    if not isinstance(seq, list):
        try:
            seq = list(seq)
        except TypeError:
            return None
    return seq


def _ids_from_items(items: list[Any], id_key: str) -> set[str]:
    """Extract a set of string ids from records or dicts.

    ``id_key`` may address either a single id (``message_id``) or a list
    of them (``TaskProposalDraft.source_message_ids`` — one task can be
    grounded in several messages). List values are flattened; without
    that a ``["m1"]`` stringified to ``"['m1']"`` and never matched the
    expected ``"m1"``.

    sensitivity_tier: N/A
    """
    found: set[str] = set()
    for item in items:
        val: Any = None
        if hasattr(item, id_key):
            val = getattr(item, id_key)
        elif isinstance(item, dict):
            val = item.get(id_key)
        if val is None:
            continue
        if isinstance(val, (list, tuple, set)):
            found.update(str(v) for v in val)
        else:
            found.add(str(val))
    return found


@dataclass
class ContainsIds(Evaluator):
    """Pass when every expected id is present in ``output.<field>``.

    ``output.<field>`` is a list of typed sub-models or dicts; each
    record exposes ``id_key`` (default ``"message_id"``). The expected
    id list comes from ``ctx.expected_output[expected_key]`` (default
    ``"ids"``). Extra ids are allowed unless ``exact`` is True.

    Use for the batch agents where the right answer is "this set of
    inputs survives the filter" (pending_reply, message_eval,
    actionable_events, contact_context, ...).

    sensitivity_tier: N/A
    """

    field: str = ""
    id_key: str = "message_id"
    expected_key: str = "ids"
    exact: bool = False

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        items = _items_for_set_eval(ctx.output, self.field)
        if items is None:
            return EvaluationReason(
                value=False, reason=f"{self.field} missing",
            )
        actual = _ids_from_items(items, self.id_key)
        expected_raw = (ctx.expected_output or {}).get(
            self.expected_key, [],
        )
        if not isinstance(expected_raw, list):
            return EvaluationReason(
                value=False,
                reason=f"expected_output.{self.expected_key} must be a list",
            )
        expected = {str(x) for x in expected_raw}
        missing = expected - actual
        if missing:
            return EvaluationReason(
                value=False,
                reason=f"missing ids: {sorted(missing)}",
            )
        if self.exact:
            extra = actual - expected
            if extra:
                return EvaluationReason(
                    value=False,
                    reason=f"unexpected ids: {sorted(extra)}",
                )
        return EvaluationReason(
            value=True,
            reason=f"matched {len(expected)} id(s)",
        )


@dataclass
class FactSetMatches(Evaluator):
    """Pass when every expected fact is present in ``output.facts``.

    Each expected entry is a dict with ``category``, ``subject``, and
    ``predicate`` (case-insensitive); the actual ``LearnedFactBatch``
    contains :class:`LearnedFactDraft` records. Comparison is on the
    triple — content prose is not compared (use ``LLMJudgeOnField``
    on the same dataset for that).

    ``predicate_alternatives`` lets a dataset accept a small set of
    synonym predicates when the LLM legitimately picks a different
    snake_case key for the same fact (e.g. ``medication`` vs
    ``takes_medication``). The mapping is keyed by the canonical
    predicate from the expected fact; (category, subject) still has
    to match exactly.

    sensitivity_tier: N/A
    """

    field: str = "facts"
    expected_key: str = "facts"
    extras_allowed: bool = True
    predicate_alternatives: dict[str, list[str]] = dc_field(default_factory=dict)

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        items = _items_for_set_eval(ctx.output, self.field)
        if items is None:
            return EvaluationReason(
                value=False, reason=f"{self.field} missing",
            )
        expected = (ctx.expected_output or {}).get(self.expected_key, [])
        if not isinstance(expected, list):
            return EvaluationReason(
                value=False,
                reason=f"expected_output.{self.expected_key} must be a list",
            )
        actual_triples = {
            _fact_triple(item) for item in items
        }
        alts_by_predicate = {
            pred.strip().lower(): [
                alt.strip().lower() for alt in alt_list
            ]
            for pred, alt_list in (self.predicate_alternatives or {}).items()
        }
        missing: list[str] = []
        accepted_alt_triples: set[tuple[str, str, str]] = set()
        for exp in expected:
            if not isinstance(exp, dict):
                missing.append(str(exp))
                continue
            triple = _fact_triple(exp)
            if triple in actual_triples:
                continue
            alts = alts_by_predicate.get(triple[2], [])
            hit = next(
                (
                    (triple[0], triple[1], alt)
                    for alt in alts
                    if (triple[0], triple[1], alt) in actual_triples
                ),
                None,
            )
            if hit is not None:
                accepted_alt_triples.add(hit)
                continue
            missing.append(f"{triple[0]}/{triple[1]}/{triple[2]}")
        if missing:
            return EvaluationReason(
                value=False,
                reason=f"missing facts: {missing}",
            )
        if not self.extras_allowed:
            expected_triples = {
                _fact_triple(e) for e in expected if isinstance(e, dict)
            }
            extras = actual_triples - expected_triples - accepted_alt_triples
            if extras:
                return EvaluationReason(
                    value=False,
                    reason=f"unexpected facts: {sorted(extras)}",
                )
        return EvaluationReason(
            value=True,
            reason=f"matched {len(expected)} fact(s)",
        )


def _fact_triple(record: Any) -> tuple[str, str, str]:
    """Return a case-folded (category, subject, predicate) tuple.

    sensitivity_tier: N/A
    """
    def _get(key: str) -> str:
        if hasattr(record, key):
            return str(getattr(record, key) or "").strip().lower()
        if isinstance(record, dict):
            return str(record.get(key, "") or "").strip().lower()
        return ""
    return (_get("category"), _get("subject"), _get("predicate"))


# Tiny helper for the ``math`` import — keeps the float comparisons honest.
def _close(a: float, b: float, tol: float = 1e-9) -> bool:  # pragma: no cover
    return math.isclose(a, b, abs_tol=tol)


__all__ = [
    "ConfidenceInRange",
    "ContainsIds",
    "DelegatedTo",
    "EmotionalLabelStructural",
    "FactSetMatches",
    "FieldContains",
    "FieldEquals",
    "FieldIn",
    "FieldNotEmpty",
    "FirewallAllowedMatches",
    "IntInRange",
    "LLMJudgeOnField",
    "LLMJudgeOnReason",
    "ListLengthInRange",
    "TierEquals",
    "TriageDecisionAccuracy",
]
