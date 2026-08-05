# Agent Evaluations

This directory holds **pydantic-evals** datasets and the runner used
to grade Pydantic AI agents against ground-truth cases. Unlike the
unit tests in `tests/`, evals exercise real `SBAgent` instances
through the same scheduler + firewall path as production and report
pass/fail per case rather than asserting individual behaviours.

## Cost

A full pass is **213 cases across 29 suites**, 57 of them judged. With
the injection-firewall companion on each agent call that is roughly
**480 LLM calls**, and when `llm_provider` is `arandu_cloud` they bill
the user's real balance. Measured over six batch days on a live
install: **~$0.16 per full run**.

(An earlier version of this file claimed ~$0.01. That was wrong by more
than an order of magnitude and predated the current pricing.)

## Skip-if-unchanged

`run_suite` fingerprints the three inputs that can change a result —
the **model** serving the agent, its **system prompt**, and the
**dataset** — and skips the suite when that hash has already passed.
Skipped rows are marked `(skipped — unchanged)` and contribute their
previous counts so totals stay readable.

- `--force` runs regardless.
- Only a clean pass (`status='ok'`) suppresses a re-run. A failure
  always re-runs — confirming whether it persists is the point.
- A revert (A → B → A) skips, because the fingerprint pins the inputs
  and A already passed on them.
- If any input can't be resolved the fingerprint is `None`, which means
  "cannot prove unchanged" and the suite runs.

The gate does **not** see agent source changes. After editing agent
code, use `--force`.

## Manual-only

Evals never run automatically. Three entry points trigger them:

- **Agents page → Run eval button** on a single agent card. Runs the
  agent's suite synchronously, persists the row to
  `agent_eval_runs`, and refreshes the card.
- **`make evals` / `python -m evals.run_evals`** from the CLI for a
  full batch (every suite or a chosen subset). CLI runs are recorded
  in `agent_eval_runs` with `trigger='cli'`, so eval spend is
  attributable — they used to leave no trace at all.

Auto-trigger on agent edit/save was removed in 0.5.0 so the judge
does not run on every settings tweak.

## Layout

```
evals/
  datasets/             # YAML datasets, one per suite
    sensitivity.yaml
    triage.yaml
    firewall_prompts.yaml
    ...
  evaluators.py         # Custom Evaluator classes
  fixtures.py           # In-memory query engine for brain_qa
  judge.py              # Remote-LLM judge wrapper
  tasks.py              # Adapters from dataset inputs -> agent calls
  run_evals.py          # CLI runner
```

## Running locally

```bash
# List available suites
python -m evals.run_evals --list

# Run one suite
python -m evals.run_evals --suite firewall_prompts

# Run multiple
python -m evals.run_evals --suite sensitivity,triage

# Run everything (default — invokes the LLM judge)
make evals

# Run everything without the LLM judge (free, structural-only)
make evals-offline

# JSON output (for CI consumption)
python -m evals.run_evals --json
```

Exit code is non-zero when any case fails — `make evals-fast` wires
this into CI so a regression in a deterministic evaluator blocks the
build.

## How prose outputs are graded

Datasets that produce free-form prose (Brain answers, insight
content, relationship nudges, weekly digests, ...) carry an
`LLMJudgeOnField` or `LLMJudgeOnReason` evaluator. Each judge call
hits the same local Ollama model the agents use via
`default_factory().get("local")` and returns a 0-10 score plus a
short reason. The case passes when `score >= threshold` (default 7).

Set `ARANDU_EVAL_JUDGE_DISABLED=1` (or use `make evals-offline`) to
skip judge calls — the case records as `skipped: judge unavailable`
and only structural assertions gate the suite.

## Suites that run without an LLM

These run end-to-end on heuristic-only paths. They are the always-on
guardrails:

| Suite              | Cases | Agent                  |
|--------------------|------:|------------------------|
| `firewall_prompts` |    14 | `InjectionFirewall`    |

## Suites that need a model

These call real agents through `default_factory().get(route)`. They
will fail with a runtime error when the remote endpoint isn't
configured. Configure `~/.arandu/settings.json`
(`llm_remote_base_url`, `llm_remote_api_key`) or set
`ARANDU_REMOTE_API_KEY` before running.

| Suite                  | Cases | Agent                       |
|------------------------|------:|-----------------------------|
| `sensitivity`          |    15 | `SensitivityAgent`          |
| `triage`               |     7 | `TriageAgent`               |
| `brain_qa`             |     6 | `BrainAgentV2` + fixture    |
| `insight`              |     5 | `InsightAgent`              |
| `fact_extractor`       |     7 | `FactExtractorAgent`        |
| `pending_reply`        |     6 | `PendingReplyAgent`         |
| `contact_context`      |     5 | `ContactContextAgent`       |
| `actionable_events`    |     6 | `ActionableEventsAgent`     |
| `topic_extractor`      |     5 | `TopicExtractorAgent`       |
| `event_categorizer`    |    17 | `EventCategorizerAgent`     |
| `weekly_digest`        |     5 | `WeeklyDigestAgent`         |
| `relationship_tracker` |     5 | `RelationshipTrackerAgent`  |
| `query_router`         |     7 | Retrieval planner           |
| `schema_discovery`     |     5 | `SchemaDiscoveryAgent`      |
| `message_eval`         |     6 | `MessageEvalAgent`          |
| `model_generator`      |     5 | `ModelGenerator`            |
| `labeler`              |     5 | `LabelerAgent`              |
| `egress_routing`       |     4 | `EgressFirewall`            |
| `dataset_validator`    |     6 | `DatasetValidatorAgent`     |

## Adding a dataset

1. Drop a YAML file under `evals/datasets/`. Each `case` needs
   `name`, `inputs`, and optionally `expected_output`.
2. Register a task function in `evals/tasks.py` that takes the case's
   `inputs` dict (or scalar) and returns the agent's structured
   output.
3. Add the filename → task mapping to `TASK_REGISTRY` at the bottom
   of `evals/tasks.py`.
4. Reference one or more evaluators (custom or built-in) under
   `evaluators:` in the YAML. For prose fields, add an
   `LLMJudgeOnField` with a tight rubric and a threshold so the
   judge has a deterministic pass/fail line.
5. Add an entry to the table above when the suite stabilises.

## Custom evaluators

Defined in `evals/evaluators.py`:

| Class                       | Purpose                                                |
|-----------------------------|--------------------------------------------------------|
| `TierEquals`                | Exact match on `output.tier`                           |
| `TriageDecisionAccuracy`    | Per-message keep/drop + flag match                     |
| `EmotionalLabelStructural`  | Emotion + domain match, intensity within tolerance     |
| `FirewallAllowedMatches`    | Boolean `allowed` match + optional `category`          |
| `ConfidenceInRange`         | `output.confidence` is within `[lo, hi]`               |
| `FieldEquals` / `FieldIn`   | Exact / set membership match on any field path         |
| `FieldContains`             | Substring match                                        |
| `FieldNotEmpty`             | Field is present + non-empty                           |
| `IntInRange`                | Integer within `[lo, hi]`                              |
| `ListLengthInRange`         | Output list length within `[min_n, max_n]`             |
| `ContainsIds`               | Expected ids present in output list (by `id_key`)      |
| `FactSetMatches`            | (cat, subject, predicate) triples present in `facts`   |
| `LLMJudgeOnReason`          | Remote LLM grades `output.reason` against a rubric     |
| `LLMJudgeOnField`           | Same, for any field path (default `answer`)            |

Built-in pydantic-evals evaluators (`Equals`, `IsInstance`,
`Contains`, etc.) can be referenced by name in YAML.

## Privacy

Dataset inputs may describe sensitive *topics* (health, finance,
mental health) so the firewall and tier classifier have material to
classify. They contain no real user data and are shipped open-source.
The judge sees only the case inputs + the agent's output, both
synthetic. When extending suites, prefer synthetic prose over
redacted user content — see `docs/PRIVACY.md`.
