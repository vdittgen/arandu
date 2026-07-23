# File Map

Reference inventory of every meaningful file in the repo. Read CLAUDE.md for
the high-signal subset; come here when you need to look up a specific path.

After the `refactor/ddd-review-cleanup` reorganization (May 2026):
- Sandboxed agent runtime moved out of `src/extensions/` into `src/agent_runtime/`.
- `src/extensions/` split into `mcp/`, `connectors/`, `bridges/`, `ingestion/`, `builtin/`.
- `apple_bridge_mcp.py` → `bridges/apple/server.py` (intact; internal split is P1).
- WhatsApp Python wrapper → `bridges/whatsapp/`; Node client → `bridges/whatsapp/node/`.

---

## Rust backend (`src-tauri/src/`)

| File | Purpose |
|------|---------|
| `main.rs` | Tauri entry point, registers all commands |
| `lib.rs` | App setup, timers (Ollama preload, WhatsApp listener, startup sync, periodic sync, scheduled agents, insight gen, proactive intelligence) |
| `commands/mod.rs` | All IPC command handlers + `AppState` struct |
| `commands/bridge.rs` | `call_python_cli()`, `call_python_cli_stream()`, `spawn_pipeline_worker()` — subprocess bridge to Python (`python -m src.core.cli`) |
| `commands/types.rs` | Serializable DTOs shared between Rust and frontend (source of truth) |
| `firewall/types.rs` | `AuditEntry` row schema (mirrors `src/agents/core/audit.py`) |
| `firewall/audit.rs` | Append-only SHA-256 hash chain reader/writer for `~/.arandu/data/audit.jsonl` |

---

## Python backend (`src/`)

### Core

| File | Purpose |
|------|---------|
| `core/cli.py` | CLI bridge — subprocess entry point called by Rust (5,773 LOC; P1 split into `src/cli/`) |
| `core/data_layer.py` | Unified data access facade with caching, pagination, dedup |
| `core/db_helpers.py` | `utc_now_iso()`, `make_hash_id()`, `safe_str()`, `table_exists()`, `ensure_tables()` |
| `core/llm_helpers.py` | `parse_llm_json_array()`, `parse_llm_json_dict()`, `safe_chat_json()` |
| `core/llm_classifier.py` | Shared LLM classifier (replaces keyword filters) |
| `core/topic_loader.py` | `load_topic_contacts()`, `load_group_engagement()`, `load_today_events()`, `load_pending_reply_ids()` |
| `core/query_engine.py` | Hybrid GraphRAG: vector + graph + SQL |
| `core/web_search.py` | DuckDuckGo fallback for BrainAgent (ephemeral) |
| `core/user_context.py` | `build_user_context()`, `build_learned_facts_context()`, `infer_user_profile()` |
| `core/monitor.py` | Memory and performance monitoring |
| `core/query_tracker.py` | `QueryTracker` — logs queries, classifies by domain, maintains interest profile |
| `core/question_patterns.py` | `QuestionPatternDetector` — 10 intent patterns |
| `core/profiler.py` | Query profiling and analytics |
| `core/sqlite/engine.py` | SQLite WAL mode, LRU cache, busy timeout |
| `core/sqlite/schemas.py` | Table schemas for all raw tables |
| `core/sqlite/migrations.py` | Additive migrations for connector-introduced tables |
| `core/kuzu/engine.py` | Kuzu graph connection and Cypher queries |
| `core/kuzu/schema.py` | Node types (Person, Event, Place, Emotion, Idea, Topic, Self) and relationships |
| `core/kuzu/indexer.py` | `GraphIndexer` — populates Kuzu from SQLite raw tables |
| `core/chromadb/engine.py` | ChromaDB wrapper |
| `core/chromadb/embedding.py` | Ollama `nomic-embed-text` + fallback |
| `core/chromadb/indexer.py` | Document indexing pipeline |

### Agents (Brain + automations)

| File | Purpose |
|------|---------|
| `agents/brain_agent.py` | Main AI agent — streaming LLM, action proposals, fact extraction (P1: split into Query / Action / Facts collaborators) |
| `agents/fact_learner.py` | Progressive learning, contradiction resolution, `_learned_facts` table |
| `agents/insight_generator.py` | Proactive insights from question patterns, `_insights` table |
| `agents/proactive_intelligence.py` | 3-pillar evaluation: pending replies, contact contexts, actionable events (P1: per-pillar split) |
| `agents/message_evaluator.py` | Post-sync message evaluation with SQL prefilter + LLM |
| `agents/message_triage.py` | Cheap LLM triage prefilter for inbound messages |
| `agents/tool_registry.py` | `ToolRegistry` — discovers action tools, intent matching (P2: rename `ActionToolCatalog`) |
| `agents/action_executor.py` | `ActionExecutor` — executes confirmed MCP actions |
| `agents/chat_cli.py` | Optional REPL for the brain agent |

### Agent runtime (sandbox for 3rd-party agents)

| File | Purpose |
|------|---------|
| `agent_runtime/base.py` | `BrainAgent` ABC |
| `agent_runtime/runner.py` | `AgentRunner` — discover, load, execute agents |
| `agent_runtime/worker.py` | Subprocess entry point for third-party agents |
| `agent_runtime/context.py` | `AgentContext` — sandboxed API (no direct DB) |
| `agent_runtime/models.py` | `AgentManifest`, `TablePermission`, `TriggerMode`, `AgentResult` |
| `agent_runtime/sensitivity_guard.py` | Field classification + tier enforcement (mirrors Rust `classifier.rs`) |
| `agent_runtime/skills.py` | `SkillRegistry` + 7 built-in skills (P2: rename `SkillCatalog`) |

### Models / providers

| File | Purpose |
|------|---------|
| `models/llm_provider.py` | `LLMProvider` ABC, `OllamaProvider`, `AnthropicProvider`, `OpenAICompatibleProvider`, factory |
| `models/labeler.py` | Emotional labeling via `LLMProvider` |
| `models/ollama_manager.py` | Ollama lifecycle, preloading, auto-start |
| `models/ollama_preempt.py` | Cross-process preemption + quiet-window for Ollama |
| `models/ollama_lock.py` | Process-level lock for Ollama model swaps |
| `models/voice_transcriber.py` | Qwen3-ASR (MLX) or faster-whisper fallback |
| `models/sensitivity_classifier.py` | ML sensitivity tier classification |

### Pipeline

| File | Purpose |
|------|---------|
| `pipeline/pipeline_brain.py` | Interest-based smart refresh planner with 4 priority tiers (P2: extract `RefreshPolicy`) |
| `pipeline/runner.py` | Pipeline execution orchestrator with dependency resolution |
| `pipeline/executor.py` | Executes individual SQL models |
| `pipeline/auditor.py` | Validates audit rules after model execution |
| `pipeline/stats.py` | Pipeline statistics and `PipelineRun` dataclass |
| `pipeline/worker.py` | Standalone worker process with SIGTERM cancellation |
| `pipeline/manifest.py` | Manifest loader |
| `pipeline/intermediate/int_labeled_messages.py` | Python-side intermediate model |
| `pipeline/intermediate/int_contact_topics.py` | Python-side intermediate model |

### Extensions

| File | Purpose |
|------|---------|
| `extensions/models.py` | `ConnectorTemplate`, `ToolTemplate`, `FieldTemplate` dataclasses |
| `extensions/cron.py` | Cron expression matcher for scheduled agents |
| `extensions/mcp/client.py` | `McpClient` — JSON-RPC 2.0 over stdio for MCP servers |
| `extensions/mcp/installer.py` | `ExtensionInstaller` — discover→confirm install flow |
| `extensions/mcp/tool_classifier.py` | Classifies MCP tools as DATA vs ACTION |
| `extensions/connectors/catalog.py` | `ConnectorCatalog` — loads bundled JSON connectors |
| `extensions/connectors/catalog_data.json` | Bundled registry of 8 connectors |
| `extensions/connectors/registry.py` | `ExtensionRegistry` — persistent enabled/disabled state (P2: rename `ConnectorState`) |
| `extensions/connectors/connection_manager.py` | Enable/disable/reconnect flow |
| `extensions/connectors/sync_scheduler.py` | Timer-based periodic syncs with retry backoff |
| `extensions/connectors/requirements.py` | macOS permission, OAuth, env var checks |
| `extensions/bridges/apple/server.py` | Apple suite MCP-over-stdio: Calendar, Contacts, Notes, Reminders (2,750 LOC; P1 internal split) |
| `extensions/bridges/whatsapp/client.py` | Python wrapper for Baileys Node.js client (JSONL stdio) |
| `extensions/bridges/whatsapp/listener.py` | Listener lifecycle: `WhatsAppListenerService`, outbox IPC |
| `extensions/bridges/whatsapp/paths.py` | Auth dir, store path, self JID/LID resolution |
| `extensions/bridges/whatsapp/node/client.js` | Custom Baileys WhatsApp client (Node.js) |
| `extensions/ingestion/transforms.py` | 11 named field transform functions |
| `extensions/ingestion/adapter.py` | `IngestionAdapter` — sync, transform, dedup, upsert (transaction-safe) |
| `extensions/ingestion/sync_engine.py` | `SyncEngine` — orchestrates adapters per connector |
| `extensions/ingestion/schema_discovery.py` | Two-pass schema discovery (rules + LLM fallback) |
| `extensions/ingestion/model_generator.py` | Auto-generates pipeline models from mappings |
| `extensions/ingestion/review_flow.py` | Stage/approve/reject with dry-run validation |
| `extensions/builtin/weekly_digest/` | Weekly Digest agent |
| `extensions/builtin/relationship_tracker/` | Relationship Tracker agent |

### Notifications

| File | Purpose |
|------|---------|
| `notifications/models.py` | Notification dataclasses (tier 2) |
| `notifications/preference_service.py` | Preferences + log persistence |
| `notifications/orchestrator.py` | `BrainNotificationOrchestrator` — AI decision engine (P2: rename `NotificationOrchestrator`) |
| `notifications/notifier.py` | `WhatsAppNotifier` — delivery via listener IPC, opt-out handling |
| `notifications/reply_handler.py` | Self-chat reply routing, STOP commands, action confirmations + `PendingActionStore` (P1: split into router/tracker/parsing/store) |

---

## Pipeline layers (`src/pipeline/`)

| Layer | Models | Purpose |
|-------|--------|---------|
| Staging | `stg_messages`, `stg_notes`, `stg_contacts`, `stg_calendar_events`, `stg_health_metrics`, `stg_emails`, `stg_reminders` | Validate types, derive fields, audits |
| Intermediate | `int_personal_enriched`, `int_events_enriched`, `int_daily_summary`, `int_labeled_messages`, `int_communications_enriched`, `int_contact_topics` | Joins, enrichment, emotional labeling, topic extraction |
| Marts | `mart_today`, `mart_health`, `mart_personal`, `mart_work`, `mart_communications`, `mart_contact_summary` | User-facing analytics, per-contact aggregation |

---

## React frontend (`src/interface/`)

| File | Purpose |
|------|---------|
| `App.tsx` | Root — onboarding gate + BrowserRouter |
| `main.tsx` | React entry point |
| `pages/Dashboard.tsx` | Main dashboard (~1050 LOC, 9 inline sub-components) |
| `pages/Chat.tsx` | Chat interface (~700 LOC) |
| `pages/Explorer.tsx` | Data Sources — auto-discovers `raw_*` tables |
| `pages/DataMarts.tsx` | Data Models — pipeline tables grouped by layer |
| `pages/GraphExplorer.tsx` | Knowledge Graph |
| `pages/VectorExplorer.tsx` | Vector Store |
| `pages/Agents.tsx` | Agent management |
| `pages/ExtensionsPage.tsx` | Unified extensions management (~1400 LOC, 4 tabs) |
| `pages/SettingsPage.tsx` | Settings |
| `components/Layout.tsx` | Sidebar + content, exports `PipelineRefreshContext` |
| `components/PipelineRefreshModal.tsx` | Multi-step pipeline refresh modal |
| `components/Sidebar.tsx` | Navigation sidebar + `BackgroundTaskIndicator` |
| `components/PipelineStatusBar.tsx` | Pipeline freshness bar (5 states) |
| `components/FreshnessIndicator.tsx` | Colored dot indicator |
| `components/OnboardingWizard.tsx` | 5-step first-launch wizard |
| `components/GenericDataTable.tsx` | Shared table with tier-3 protection |
| `components/InstallExtensionModal.tsx` | 5-step MCP install wizard |
| `components/ConnectorConfigModal.tsx` | Read-only connector config viewer |
| `components/UpdateRequiredModal.tsx` | Fullscreen forced update modal |
| `hooks/useAsyncData.ts` | Generic IPC hook |
| `hooks/useStreamingChat.ts` | Streaming via `brain-stream` Tauri events |
| `hooks/useAudioRecording.ts` | Audio recording + transcription |
| `hooks/useAutoRefresh.ts` | Background auto-refresh |
| `hooks/useBackgroundTasks.ts` | Polls `get_active_tasks` (3s) |
| `hooks/usePipelineProgress.ts` | Pipeline modal state machine |
| `hooks/usePipelineStatus.ts` | Pipeline status polling (30s) |
| `hooks/useUpdateChecker.ts` | Forced update state machine |
| `utils/requestDedup.ts` | `dedupInvoke<T>()` — in-flight request dedup (NOT a cache) |
| `utils/timeFormat.ts` | `formatRelativeTime()`, `formatElapsedTime()` |

### Tailwind palette
`sb-sidebar: #0D1B2A`, `sb-main: #1B2838`, `sb-card: #243447`, `sb-text: #E0E7EE`, `sb-muted: #8899AA`, `sb-accent: #2E86AB`, `sb-success: #2D6A4F`, `sb-warning: #E76F51`, `sb-border: #2A3A4C`

### Frontend routes
| Route | Page | Route | Page |
|-------|------|-------|------|
| `/` | Dashboard | `/agents` | Agents |
| `/chat` | Chat | `/extensions` | ExtensionsPage |
| `/explorer` | Explorer | `/settings` | SettingsPage |
| `/marts` | DataMarts | `/data-sources` | → `/extensions` |
| `/graph` | GraphExplorer | `/vectors` | VectorExplorer |

### Data flow
```
React → dedupInvoke / useAsyncData → Tauri invoke → Rust command handler
  → call_python_cli(["subcommand"]) → python -m src.core.cli → JSON stdout → Rust → frontend
```

---

## IPC commands (Tauri ↔ Frontend)

Called via `invoke("command_name", { args })` or `dedupInvoke()`.

**Data queries** (Tier 1–2): `get_database_stats`, `get_today_summary`, `get_recent_messages`, `get_upcoming_events`, `get_notes`, `get_emails`, `list_tables`, `query_table`, `graph_summary`, `query_graph_nodes`, `query_graph_relationships`, `vector_summary`

**Brain Agent** (Tier 3): `ask_brain`, `ask_brain_stream` (emits `brain-stream` events), `get_chat_history`, `clear_chat_history`

**Pipeline** (Tier 1): `get_pipeline_status`, `trigger_pipeline_run`, `trigger_pipeline_run_stream`, `get_refresh_plan`, `get_pipeline_run_result`, `get_pipeline_run_history`, `cancel_pipeline_run`, `is_pipeline_running`

**Connectors** (Tier 1): `get_connector_catalog`, `toggle_connector`, `sync_connector_now`, `get_connector_details`, `install_extension_discover`, `install_extension_confirm`, `generate_models`, `approve_models`, `reject_models`

**Agents & Skills** (Tier 1–2): `list_agents`, `run_agent`, `get_agent_result`, `list_skills`

**Insights** (Tier 1–3): `get_insights`, `generate_insights`, `dismiss_insight`, `follow_up_insight`

**Proactive Intelligence** (Tier 1–3): `evaluate_proactive`, `get_pending_replies`, `get_contact_contexts`, `get_actionable_events`, `dismiss_pending_reply`, `dismiss_actionable_event`

**Actions** (Tier 1–3): `get_available_actions`, `confirm_action`, `cancel_action`

**Notifications** (Tier 1): `get_notification_preferences`, `update_notification_preference`, `mute_all_notifications`, `get_notification_log`

**Facts** (Tier 1–2): `get_learned_facts`, `get_facts_for_review`, `get_fact_stats`, `confirm_fact`, `dismiss_fact`, `edit_fact`

**System** (Tier 1–2): `get_settings`, `update_settings`, `get_ollama_status`, `preload_ollama_model`, `get_memory_usage`, `get_audit_log`, `health_check`, `get_interest_profile`, `get_domain_stats`, `infer_user_profile`, `transcribe_audio` (Tier 3), `get_active_tasks`

---

## Frontend patterns reference

- `useAsyncData(command, args?)`: returns `{ data, status, error, refetch, isLoading }`. No `lastFetchedAt` — track via `useRef` if needed.
- `usePipelineProgress()`: state machine — `idle → confirm → processing → success/error/cancelled`.
- `useStreamingChat()`: event types — `context`, `token`, `done`, `error`, `action_proposal`.
- `dedupInvoke()`: in-flight request dedup, NOT a cache.
- Inline sub-components are common (e.g. `GreetingSection` inside `Dashboard.tsx`).
- Icons from `lucide-react`; text sizes `text-xs` / `text-[11px]`.

---

## Lifecycle / timer reference (driven by Rust `lib.rs`)

| Timer | Cadence | Notes |
|---|---|---|
| Periodic sync | 15 min | Calls `sync-all-stale`. Python `SyncScheduler` timers die with subprocess. |
| Insight generation | 4 h | 30s startup delay |
| Proactive intelligence | 2 h | 90s startup delay; read commands pass `llm_provider=None` |
| Scheduled agents | 60 s | Cron matcher poll |
| Startup sync | 5 s after launch | Syncs all enabled connectors + pipeline if stale |
| WhatsApp listener restart | T+3 s | Stops stale listener, starts fresh if enabled |

CLI subprocesses use `kill_on_drop(true)`. Startup `pkill -f` clears orphans.

All `_maybe_*` hooks (`_maybe_notify_pipeline`, `_maybe_notify_action`, `_maybe_notify_insights`, `_maybe_evaluate_new_messages`, `_maybe_extract_facts`, `_maybe_transcribe_audio`, `_reindex_chromadb`) are wrapped in try/except and never fail their parent operations.
