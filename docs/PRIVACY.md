# Privacy & Data Egress

Arandu is a privacy-first AI operating system that runs as a
native desktop app. The promise is built on a few load-bearing
facts: the analytical store is embedded, the vector index is local,
the audit chain is on disk, and **every LLM call stays on the
user's machine**.

Arandu runs Ollama locally and never reaches a cloud provider.

If you are evaluating the project's privacy posture and this
document disagrees with code, **the code wins** — open an issue and
we'll reconcile.

## Threat model

We model three classes of risk:

1. **Local exfiltration** — software on the user's machine reading
   `~/.arandu/data/`. Out of scope for this document; addressed by
   the host OS sandbox and the project's `SensitivityGuard`.
2. **Network egress** — prompts and retrieved context leaving the
   machine on their way to an LLM. **In Arandu this never
   happens**: the provider factory in `src/models/llm_provider.py`
   is hard-locked to Ollama, and the egress firewall in
   `src/agents/firewall/egress_firewall.py` resolves every route to
   `local_ollama`.
3. **Provider-side compromise** — irrelevant to Arandu since
   no third-party provider is contacted.

## Single chokepoint

All outbound LLM calls — from `AgentContext.ask_llm` (built-in and
sandboxed agents), the Brain action helpers (`brain/actions.py`),
the skill registry (`agent_runtime/skills.py`), the notification
orchestrator, and the pydantic-ai `SBAgent.run()` base path — share
one entry point: `src/models/llm_gateway.py`
(`chat_via_firewalls(...)`). The gateway:

1. Checks the per-agent block table (set when an agent's eval suite
   failed against the user's configured local model).
2. Runs the prompt-injection firewall.
3. Asks the egress firewall for the route — always `local` in OSS.
4. Picks the provider for the resolved route (Ollama).
5. Acquires a scheduler permit at the right tier.

Pitfall #11 in `CLAUDE.md` forbids bypassing this — the gateway
exists so call sites don't have to choose between convenience and
correctness.

## Eval-gated local inference

`local_inference_for_sensitive` (Settings → Privacy) is an
eval-gate hook. Its user-facing value is "always on": every prompt
is local. Flipping the toggle still triggers
`cmd_set_local_inference_for_sensitive`
(`src/agents/cli_handlers.py`), which runs every agent's eval suite
from `AGENT_SUITE_MAP` against the user's configured local model.
This catches the case where a user swaps Ollama models and one or
more agents stop producing acceptable output.

**Runtime block.** A *later* eval failure for one agent writes a
row into `agent_blocked` (see
`src/agents/core/agent_block_store.py`). The gateway short-circuits
every call from that agent with `GatewayBlocked` until the user
re-runs the eval and sees it pass (the handler clears the row).

## Tier classification

The chosen tier for a single prompt is the *maximum* of:

- The calling agent's `max_sensitivity_tier` (declared in its
  manifest).
- An explicit tier passed by the caller.
- The coarse keyword scan (`keyword_tier_floor`) of the prompt +
  retrieved context.
- The LLM-driven sensitivity classifier (when the user has not
  disabled it for performance reasons).

In Arandu tier classification is informational — it doesn't
change where a prompt goes (every tier is local). It still drives
the audit log.

Failure mode of the LLM classifier is conservative: an unavailable
classifier returns `None` so the caller falls back to the keyword
floor + agent manifest tier — never relaxing the decision.

## Audit chain

Every firewall decision is appended to
`~/.arandu/data/audit.jsonl`. The file is SHA-256 chained;
tampering breaks the chain and is detected by `AuditChain.verify()`.

Audit event types relevant to egress:

- `egress_decision` — the route + tier for one call.
- `local_inference_toggle` — the user flipped the privacy toggle
  (carries `enabled` + a hash of the per-agent eval results).

There is **no per-call consent event** — no per-call data ever
leaves the device, so there is nothing to consent to.

## Redaction registry (local utility, not an egress path)

`redaction_registry.sqlite` maps high-signal entities (people,
places, emails, phone numbers, account/money amounts, dates) to
stable placeholders. It is not part of the main gateway's egress
path — `chat_via_firewalls` never calls it, since every route it
resolves is local. A small number of other call sites use it for
their own local purposes independent of egress: `brain/actions.py`
rehydrates redacted tool-argument values back through the registry
before resolving a recipient, and the injection-scan sub-agent (a
separate routing decision from the main gateway) redacts what it
sends to its own scan model when configured for that.

When the registry is in use it holds raw user values paired with
their placeholders — Tier 3 data. It is:

- Stored under `~/.arandu/data/` next to the rest of the local DB.
- Created with `0600` file mode.
- Annotated `sensitivity_tier: 3` at the module level so
  `SensitivityGuard` can refuse non-authorised access.
- Never sent over any network.

## Things this document does not cover

- Other connectors' privacy (calendar, mail, WhatsApp) — covered in
  their per-connector docs.

## Reporting

Privacy issues: open a confidential issue or contact the
maintainers directly. Do not include prompt content in bug
reports — hashes only.
