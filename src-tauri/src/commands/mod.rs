pub mod bridge;
pub mod setup;
pub mod types;

use tokio::sync::Mutex;

use chrono::Utc;
use tauri::{AppHandle, Emitter, Manager, State};

use crate::firewall::audit::AuditLogger;
use crate::firewall::types::{AuditEntry, RedactionDetailResponse};

use self::bridge::{
    call_python_cli, call_python_cli_with_timeout, resolve_project_root,
    spawn_pipeline_worker,
};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::time::Instant;

use self::types::{
    AgentActivityResponse,
    AgentEvalDataset, AgentEvalProposalResponse, AgentEvalRunResponse, AgentEvalStatusResponse,
    AgentFullStatus,
    AgentRunResult, AgentRunning, AgentStream, AppSettings, AvailableModels,
    BackgroundTask, BrainResponse,
    ChatMessage, ChatSessionListResponse, ChatSessionSummary,
    Contact, DailyBrief, DatasetSuggestionResponse, DatasetValidationResponse,
    DomainSummary, Email, Event,
    HealthCheckResult, LoadSessionResponse, LocalInferenceToggleResponse,
    McpActionToolListResponse, MemoryUsage, Message, ModelPickerSpec, ModelPullProgress,
    ModelRecommendationResponse,
    Note, OllamaStatus, PipelineRunResult,
    PipelineRunStarted, PipelineRunSummary, PipelineStatus, PromptEngineerSpec,
    PromptSuggestionResponse, PydanticAgentListResponse,
    PydanticAgentPatch, PydanticAgentResponse, RebuildVectorIndexResult,
    Stats, SuggestedActions, Thread, TodaySummary, UnsavedAgentSpec,
    BatchRunSummary,
    UserAgentInput, UserAgentResponse, UserAgentStatus,
    ConnectorHistoryEntry, ExtensionLogOutput, UninstallResult,
};

// ---------------------------------------------------------------------------
// Managed state
// ---------------------------------------------------------------------------

/// Handle to an active pipeline worker subprocess.
///
/// The `Child` itself is moved into the cleanup task at spawn time so
/// `child.wait().await` doesn't hold the `pipeline_worker` mutex —
/// otherwise `cancel_pipeline_run` (which also locks that mutex)
/// would deadlock until the worker exits on its own.  Cancellation
/// is delivered via the cached pid, which is all Unix signal sending
/// needs.
pub struct PipelineWorkerHandle {
    pub run_id: String,
    /// Cached pid for signaling cancel / hard-kill on timeout.
    /// `None` only if the platform refused to report a pid (very rare).
    pub pid: Option<u32>,
}

/// Application state shared across all Tauri commands.
pub struct AppState {
    /// Pointer to the SQLite-persisted chat session the user is in.
    /// `None` = no session yet (next message creates one).
    pub active_chat_session: Mutex<Option<String>>,
    pub settings: Mutex<AppSettings>,
    pub project_root: String,
    pub pipeline_runs: Arc<Mutex<HashMap<String, PipelineRunSummary>>>,
    pub pipeline_worker: Arc<Mutex<Option<PipelineWorkerHandle>>>,
    /// `None` = idle, `Some(started_at)` = running since.
    /// Auto-resets after `PIPELINE_MAX_AGE` to prevent stuck-flag stalls.
    pub pipeline_running: Arc<Mutex<Option<Instant>>>,
    /// Serializes background write-mode CLI subprocesses so only one
    /// holds the SQLite write lock at a time.  Frontend IPC commands
    /// are NOT gated by this — they use SQLite's busy_timeout instead.
    pub cli_write_lock: Arc<tokio::sync::Mutex<()>>,
    /// Currently-running background tasks visible in the UI.
    pub active_tasks: Arc<Mutex<HashMap<String, BackgroundTask>>>,
}

impl AppState {
    pub fn new() -> Self {
        let settings = load_settings_from_disk().unwrap_or_default();
        Self {
            active_chat_session: Mutex::new(None),
            settings: Mutex::new(settings),
            project_root: resolve_project_root(),
            pipeline_runs: Arc::new(Mutex::new(HashMap::new())),
            pipeline_worker: Arc::new(Mutex::new(None)),
            pipeline_running: Arc::new(Mutex::new(None)),
            cli_write_lock: Arc::new(tokio::sync::Mutex::new(())),
            active_tasks: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

/// Register a background task so it appears in the UI.
pub async fn register_task(state: &AppState, id: &str, label: &str) {
    let task = BackgroundTask {
        id: id.to_string(),
        label: label.to_string(),
        started_at: Utc::now().to_rfc3339(),
    };
    state.active_tasks.lock().await.insert(id.to_string(), task);
}

/// Remove a background task when it finishes.
pub async fn unregister_task(state: &AppState, id: &str) {
    state.active_tasks.lock().await.remove(id);
}

/// Maximum time the pipeline_running flag can stay set before auto-reset.
/// Outer safety net — `PIPELINE_HARD_TIMEOUT` should fire first under
/// normal conditions and reap the worker; this only kicks in if the
/// cleanup task itself dies before clearing the flag.
const PIPELINE_MAX_AGE: std::time::Duration = std::time::Duration::from_secs(3600);

/// Wall-clock budget for a single pipeline worker subprocess.
///
/// Legitimate runs take 10–20s (per `pipeline_stats.jsonl`); 30 min is
/// generous headroom for slow upstream LLMs and well below the 1 h
/// `PIPELINE_MAX_AGE` so the inner timeout fires before the outer
/// safety net.
pub const PIPELINE_HARD_TIMEOUT: std::time::Duration =
    std::time::Duration::from_secs(30 * 60);

/// Grace period between SIGTERM and SIGKILL when reaping a runaway
/// worker.  The Python worker installs no SIGTERM handler so this is
/// mostly a courtesy — long enough for the OS to deliver the signal,
/// short enough that the user isn't waiting.
const PIPELINE_KILL_GRACE: std::time::Duration =
    std::time::Duration::from_secs(5);

/// Wall-clock budget for a single `ask` CLI invocation.
///
/// Normal answers take 5–30s; the worst case stacks a full SQLite
/// busy_timeout queue (30s, pipeline writing) on top of offline-Ollama
/// embedding retries (~30s).  Without a bound the UI freezes silently
/// for as long as the subprocess blocks.
const ASK_TIMEOUT_SECS: u64 = 120;

/// Send SIGTERM then SIGKILL to *pid* with a brief grace period.
///
/// No-op on the (rare) `None` pid case.  Errors from `kill(1)` are
/// swallowed because by the time we're calling this the process may
/// already be dead — that's success from our point of view.
#[cfg(unix)]
async fn terminate_worker_pid(pid: Option<u32>) {
    let Some(pid) = pid else { return };
    let _ = std::process::Command::new("kill")
        .args(["-TERM", &pid.to_string()])
        .status();
    tokio::time::sleep(PIPELINE_KILL_GRACE).await;
    let _ = std::process::Command::new("kill")
        .args(["-KILL", &pid.to_string()])
        .status();
}

#[cfg(not(unix))]
async fn terminate_worker_pid(pid: Option<u32>) {
    let Some(pid) = pid else { return };
    // On Windows we have no SIGTERM equivalent for an external pid;
    // taskkill /F is the direct hard-kill.
    let _ = std::process::Command::new("taskkill")
        .args(["/F", "/PID", &pid.to_string()])
        .status();
}

/// Reap any orphan pipeline worker still tracked in state, then clear
/// the slot.  Called from the `PIPELINE_MAX_AGE` auto-reset, from
/// `trigger_pipeline_run_stream` as a defensive guard before spawning
/// a new worker, and from the `ExitRequested` handler in `lib.rs` so a
/// graceful app quit never strands a worker holding the SQLite write
/// lock.
pub(crate) async fn reap_orphan_worker(state: &AppState) {
    let pid = {
        let mut guard = state.pipeline_worker.lock().await;
        guard.take().and_then(|h| h.pid)
    };
    if pid.is_some() {
        eprintln!(
            "[pipeline] reaping orphan worker pid={:?}",
            pid,
        );
        terminate_worker_pid(pid).await;
    }
}

/// Check if the pipeline is currently running.  If the flag has been set
/// for longer than `PIPELINE_MAX_AGE`, auto-reset it AND reap any
/// orphan worker subprocess — otherwise the flag clears but the
/// orphaned Python process stays alive holding the SQLite WAL open.
pub async fn is_pipeline_flag_set(state: &AppState) -> bool {
    let stale = {
        let mut guard = state.pipeline_running.lock().await;
        match *guard {
            Some(started_at) if started_at.elapsed() > PIPELINE_MAX_AGE => {
                eprintln!(
                    "[pipeline] auto-resetting stuck pipeline_running flag (set {:.0}s ago)",
                    started_at.elapsed().as_secs_f64()
                );
                *guard = None;
                true
            }
            Some(_) => return true,
            None => return false,
        }
    };
    if stale {
        reap_orphan_worker(state).await;
    }
    false
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Data commands (async — calls Python subprocess via bridge)
// ---------------------------------------------------------------------------

/// Get database statistics from all three engines.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn get_database_stats(state: State<'_, AppState>) -> Result<Stats, String> {
    let output = call_python_cli(&["stats"], &state.project_root).await?;
    serde_json::from_str(&output).map_err(|e| format!("Failed to parse stats JSON: {e}"))
}

/// Get today's summary: events, recent messages, note count.
///
/// # sensitivity_tier: 2
#[tauri::command]
pub async fn get_today_summary(state: State<'_, AppState>) -> Result<TodaySummary, String> {
    let output = call_python_cli(&["query-today"], &state.project_root).await?;
    serde_json::from_str(&output).map_err(|e| format!("Failed to parse today summary JSON: {e}"))
}

/// Get recent messages ordered by timestamp descending.
///
/// # sensitivity_tier: 2
#[tauri::command]
pub async fn get_recent_messages(
    limit: Option<usize>,
    offset: Option<usize>,
    state: State<'_, AppState>,
) -> Result<Vec<Message>, String> {
    let limit = limit.unwrap_or(50);
    let offset = offset.unwrap_or(0);
    let limit_str = limit.to_string();
    let offset_str = offset.to_string();
    let output = call_python_cli(
        &[
            "query-messages",
            "--limit",
            &limit_str,
            "--offset",
            &offset_str,
        ],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output).map_err(|e| format!("Failed to parse messages JSON: {e}"))
}

/// Get upcoming calendar events within the specified number of days.
///
/// # sensitivity_tier: 2
#[tauri::command]
pub async fn get_upcoming_events(
    days: Option<usize>,
    limit: Option<usize>,
    offset: Option<usize>,
    state: State<'_, AppState>,
) -> Result<Vec<Event>, String> {
    let days = days.unwrap_or(7);
    let limit = limit.unwrap_or(50);
    let offset = offset.unwrap_or(0);
    let days_str = days.to_string();
    let limit_str = limit.to_string();
    let offset_str = offset.to_string();
    let output = call_python_cli(
        &[
            "query-events",
            "--days",
            &days_str,
            "--limit",
            &limit_str,
            "--offset",
            &offset_str,
        ],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output).map_err(|e| format!("Failed to parse events JSON: {e}"))
}

/// Retrieve contacts from raw_contacts.
///
/// # sensitivity_tier: 2
#[tauri::command]
pub async fn get_contacts(
    limit: Option<usize>,
    offset: Option<usize>,
    state: State<'_, AppState>,
) -> Result<Vec<Contact>, String> {
    let limit = limit.unwrap_or(500);
    let offset = offset.unwrap_or(0);
    let limit_str = limit.to_string();
    let offset_str = offset.to_string();
    let output = call_python_cli(
        &[
            "query-contacts",
            "--limit",
            &limit_str,
            "--offset",
            &offset_str,
        ],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output).map_err(|e| format!("Failed to parse contacts JSON: {e}"))
}

/// Return notes from raw_notes, paginated.
///
/// # sensitivity_tier: 2
#[tauri::command]
pub async fn get_notes(
    limit: Option<usize>,
    offset: Option<usize>,
    state: State<'_, AppState>,
) -> Result<Vec<Note>, String> {
    let limit = limit.unwrap_or(100);
    let offset = offset.unwrap_or(0);
    let limit_str = limit.to_string();
    let offset_str = offset.to_string();
    let output = call_python_cli(
        &[
            "query-notes",
            "--limit",
            &limit_str,
            "--offset",
            &offset_str,
        ],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output).map_err(|e| format!("Failed to parse notes JSON: {e}"))
}

/// Return emails from raw_emails, paginated.
///
/// # sensitivity_tier: 2
#[tauri::command]
pub async fn get_emails(
    limit: Option<usize>,
    offset: Option<usize>,
    state: State<'_, AppState>,
) -> Result<Vec<Email>, String> {
    let limit = limit.unwrap_or(200);
    let offset = offset.unwrap_or(0);
    let limit_str = limit.to_string();
    let offset_str = offset.to_string();
    let output = call_python_cli(
        &[
            "query-emails",
            "--limit",
            &limit_str,
            "--offset",
            &offset_str,
        ],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output).map_err(|e| format!("Failed to parse emails JSON: {e}"))
}

// ---------------------------------------------------------------------------
// Generic table browser commands
// ---------------------------------------------------------------------------

/// List all tables matching a prefix with row counts and column info.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn list_tables(
    prefix: Option<String>,
    state: State<'_, AppState>,
) -> Result<Vec<types::TableInfo>, String> {
    let prefix = prefix.unwrap_or_default();
    let mut args = vec!["list-tables".to_string()];
    if !prefix.is_empty() {
        args.push("--prefix".to_string());
        args.push(prefix);
    }
    let args_refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    let output = call_python_cli(&args_refs, &state.project_root).await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse list-tables JSON: {e}"))
}

/// List all pipeline models registered in the manifest.
///
/// Returns every model the pipeline knows about — including those whose
/// SQLite tables haven't been materialized yet — so the Data Models page
/// can show "not built yet" rows alongside live tables.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn list_pipeline_models(
    state: State<'_, AppState>,
) -> Result<Vec<types::PipelineModel>, String> {
    let output =
        call_python_cli(&["list-pipeline-models"], &state.project_root).await?;
    serde_json::from_str(&output).map_err(|e| {
        format!("Failed to parse list-pipeline-models JSON: {e}")
    })
}

/// Query sample rows from a whitelisted DuckDB table.
///
/// # sensitivity_tier: 2
#[tauri::command]
pub async fn query_table(
    table: String,
    limit: Option<usize>,
    offset: Option<usize>,
    state: State<'_, AppState>,
) -> Result<types::TableSample, String> {
    let limit = limit.unwrap_or(25);
    let offset = offset.unwrap_or(0);
    let limit_str = limit.to_string();
    let offset_str = offset.to_string();
    let output = call_python_cli(
        &[
            "query-table",
            "--table",
            &table,
            "--limit",
            &limit_str,
            "--offset",
            &offset_str,
        ],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse query-table JSON: {e}"))
}

// ---------------------------------------------------------------------------
// Graph explorer commands
// ---------------------------------------------------------------------------

/// Get summary of all node and relationship types in the Kuzu graph.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn graph_summary(
    state: State<'_, AppState>,
) -> Result<types::GraphSummary, String> {
    let output = call_python_cli(&["graph-summary"], &state.project_root).await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse graph-summary JSON: {e}"))
}

/// Query sample nodes of a given type from the Kuzu graph.
///
/// # sensitivity_tier: 2
#[tauri::command]
pub async fn query_graph_nodes(
    node_type: String,
    limit: Option<usize>,
    state: State<'_, AppState>,
) -> Result<types::GraphNodeSample, String> {
    let limit = limit.unwrap_or(25);
    let limit_str = limit.to_string();
    let output = call_python_cli(
        &["query-graph-nodes", "--type", &node_type, "--limit", &limit_str],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse query-graph-nodes JSON: {e}"))
}

/// Query sample relationships of a given type from the Kuzu graph.
///
/// # sensitivity_tier: 2
#[tauri::command]
pub async fn query_graph_relationships(
    rel_type: String,
    limit: Option<usize>,
    state: State<'_, AppState>,
) -> Result<types::GraphRelSample, String> {
    let limit = limit.unwrap_or(25);
    let limit_str = limit.to_string();
    let output = call_python_cli(
        &["query-graph-rels", "--type", &rel_type, "--limit", &limit_str],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse query-graph-rels JSON: {e}"))
}

// ---------------------------------------------------------------------------
// Vector explorer commands
// ---------------------------------------------------------------------------

/// Get ChromaDB collection counts with sample documents.
///
/// # sensitivity_tier: 2
#[tauri::command]
pub async fn vector_summary(
    state: State<'_, AppState>,
) -> Result<types::VectorSummary, String> {
    let output = call_python_cli(&["vector-summary"], &state.project_root).await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse vector-summary JSON: {e}"))
}

// ---------------------------------------------------------------------------
// Chat commands
// ---------------------------------------------------------------------------

/// Ensure there's a current active chat session, creating one via the
/// Python CLI if necessary. Returns the session id.
///
/// # sensitivity_tier: 2
async fn ensure_active_session(state: &AppState) -> Result<String, String> {
    {
        let guard = state.active_chat_session.lock().await;
        if let Some(id) = guard.as_ref() {
            return Ok(id.clone());
        }
    }
    let output = call_python_cli(&["chat-session-create"], &state.project_root).await?;
    let parsed: serde_json::Value = serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse chat-session-create JSON: {e}"))?;
    let session_id = parsed
        .get("session_id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "chat-session-create missing session_id".to_string())?
        .to_string();
    *state.active_chat_session.lock().await = Some(session_id.clone());
    Ok(session_id)
}

/// Ask the Brain Agent a question grounded in personal context.
///
/// Persists both the question and the answer in the active chat session
/// (creating one if needed) so they survive an app restart.
///
/// # sensitivity_tier: 3
#[tauri::command]
pub async fn ask_brain(
    question: String,
    state: State<'_, AppState>,
) -> Result<BrainResponse, String> {
    let session_id = ensure_active_session(&state).await?;
    // Bounded: an unbounded call freezes the UI silently when the CLI
    // queues on the SQLite busy_timeout (pipeline writing) or retries
    // embeddings against an offline Ollama. 120s covers the worst case
    // (30s busy wait + 30s embedding retries + query time) with margin.
    let output = call_python_cli_with_timeout(
        &["ask", &question, "--session-id", &session_id],
        &state.project_root,
        ASK_TIMEOUT_SECS,
    )
    .await?;
    let response: BrainResponse = serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse brain response: {e}"))?;
    Ok(response)
}

/// Ask the Brain Agent a question without recording it in the user chat history.
///
/// Used by internal callers (dashboard widgets, background helpers) that need an
/// LLM response but must not pollute the user-visible Chat tab conversation.
///
/// # sensitivity_tier: 3
#[tauri::command]
pub async fn ask_brain_internal(
    question: String,
    state: State<'_, AppState>,
) -> Result<BrainResponse, String> {
    let output = call_python_cli_with_timeout(
        &["ask", &question],
        &state.project_root,
        ASK_TIMEOUT_SECS,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse brain response: {e}"))
}

/// Get the messages of the active chat session.
///
/// Returns an empty vec when no session is active yet.
///
/// # sensitivity_tier: 3
#[tauri::command]
pub async fn get_chat_history(state: State<'_, AppState>) -> Result<Vec<ChatMessage>, String> {
    let session_id = {
        let guard = state.active_chat_session.lock().await;
        guard.clone()
    };
    let Some(session_id) = session_id else {
        return Ok(Vec::new());
    };
    let output = call_python_cli(
        &["chat-session-load", &session_id],
        &state.project_root,
    )
    .await?;
    let parsed: LoadSessionResponse = serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse chat-session-load JSON: {e}"))?;
    Ok(parsed.messages)
}

/// Forget the active session pointer so the next message starts a fresh
/// session. Past sessions stay in the database — use `delete_chat_session`
/// to remove them.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn clear_chat_history(state: State<'_, AppState>) -> Result<(), String> {
    *state.active_chat_session.lock().await = None;
    Ok(())
}

/// List recent persisted chat sessions plus the active session pointer.
///
/// # sensitivity_tier: 3
#[tauri::command]
pub async fn list_chat_sessions(
    state: State<'_, AppState>,
) -> Result<ChatSessionListResponse, String> {
    let output = call_python_cli(&["chat-session-list"], &state.project_root).await?;
    let parsed: serde_json::Value = serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse chat-session-list JSON: {e}"))?;
    let sessions: Vec<ChatSessionSummary> = serde_json::from_value(
        parsed.get("sessions").cloned().unwrap_or(serde_json::Value::Array(Vec::new())),
    )
    .map_err(|e| format!("Failed to parse sessions array: {e}"))?;
    let active_session_id = state.active_chat_session.lock().await.clone();
    Ok(ChatSessionListResponse {
        sessions,
        active_session_id,
    })
}

/// Load a persisted chat session and mark it active.
///
/// # sensitivity_tier: 3
#[tauri::command]
pub async fn load_chat_session(
    session_id: String,
    state: State<'_, AppState>,
) -> Result<LoadSessionResponse, String> {
    let output = call_python_cli(
        &["chat-session-load", &session_id],
        &state.project_root,
    )
    .await?;
    let parsed: LoadSessionResponse = serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse chat-session-load JSON: {e}"))?;
    *state.active_chat_session.lock().await = Some(parsed.session_id.clone());
    Ok(parsed)
}

/// Create a fresh chat session, mark it active, and return its summary.
///
/// # sensitivity_tier: 2
#[tauri::command]
pub async fn new_chat_session(
    state: State<'_, AppState>,
) -> Result<ChatSessionSummary, String> {
    let output = call_python_cli(&["chat-session-create"], &state.project_root).await?;
    let parsed: serde_json::Value = serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse chat-session-create JSON: {e}"))?;
    let session_id = parsed
        .get("session_id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "chat-session-create missing session_id".to_string())?
        .to_string();
    *state.active_chat_session.lock().await = Some(session_id.clone());
    let now = Utc::now().to_rfc3339();
    Ok(ChatSessionSummary {
        id: session_id,
        title: "New chat".to_string(),
        created_at: now.clone(),
        updated_at: now,
        message_count: 0,
        preview: None,
    })
}

/// Delete a chat session and all its messages.
///
/// Clears the active-session pointer if the deleted session was active.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn delete_chat_session(
    session_id: String,
    state: State<'_, AppState>,
) -> Result<(), String> {
    call_python_cli(
        &["chat-session-delete", &session_id],
        &state.project_root,
    )
    .await?;
    let mut guard = state.active_chat_session.lock().await;
    if guard.as_deref() == Some(session_id.as_str()) {
        *guard = None;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Streaming & Ollama commands
// ---------------------------------------------------------------------------

/// Stream a Brain Agent response token-by-token via Tauri events.
///
/// Emits `brain-stream` events containing JSON chunks:
/// - `{"type":"context", ...}` — context metadata
/// - `{"type":"token", "token":"..."}` — each streamed token
/// - `{"type":"done", ...}` — completion signal
/// - `{"type":"error", ...}` — on failure
///
/// `reply_context` is set when the question originates from a "Draft
/// reply" click on a known inbound message — the Brain uses it to
/// hard-lock the proposed action channel and seed the original message
/// into context for accurate contact resolution.
///
/// # sensitivity_tier: 3
#[tauri::command]
pub async fn ask_brain_stream(
    question: String,
    reply_context: Option<types::ReplyContext>,
    task_context: Option<types::TaskContext>,
    app_handle: AppHandle,
    state: State<'_, AppState>,
) -> Result<(), String> {
    let session_id = ensure_active_session(&state).await?;
    let reply_ctx_json = match &reply_context {
        Some(ctx) => Some(
            serde_json::to_string(ctx)
                .map_err(|e| format!("Failed to serialize reply_context: {e}"))?,
        ),
        None => None,
    };
    let task_ctx_json = match &task_context {
        Some(ctx) => Some(
            serde_json::to_string(ctx)
                .map_err(|e| format!("Failed to serialize task_context: {e}"))?,
        ),
        None => None,
    };
    let mut args: Vec<&str> = vec![
        "ask-stream",
        &question,
        "--session-id",
        &session_id,
    ];
    if let Some(json) = reply_ctx_json.as_deref() {
        args.push("--reply-context");
        args.push(json);
    }
    if let Some(json) = task_ctx_json.as_deref() {
        args.push("--task-context");
        args.push(json);
    }
    bridge::call_python_cli_stream(
        &args,
        &state.project_root,
        &app_handle,
        "brain-stream",
    )
    .await
}

/// Signal an in-flight ask-stream run to stop researching.
///
/// Does NOT kill the Python subprocess — instead, it routes to
/// :func:`src.agents.core.cancel_registry.request_cancel` via a
/// short-lived CLI subprocess. The in-flight orchestrator sees the
/// signal at its next reflection checkpoint and injects a STOP_REQUEST
/// user message, so the model wraps up with whatever context it has.
///
/// Returns `Ok(())` even when the run id is unknown — the frontend
/// can treat that as "already finished" rather than an error.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn stop_research(
    run_id: String,
    state: State<'_, AppState>,
) -> Result<(), String> {
    let _ = call_python_cli(
        &["stop-research", &run_id],
        &state.project_root,
    )
    .await
    .map_err(|e| format!("stop_research failed: {e}"))?;
    Ok(())
}

/// Get Ollama server and model status.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn get_ollama_status(state: State<'_, AppState>) -> Result<OllamaStatus, String> {
    let output = match call_python_cli(&["ollama-status"], &state.project_root).await {
        Ok(out) => out,
        Err(e) => {
            eprintln!("[get_ollama_status] CLI error: {e}");
            return Ok(OllamaStatus::backend_error(format!(
                "Status check failed to run: {e}"
            )));
        }
    };
    if output.is_empty() {
        eprintln!("[get_ollama_status] WARNING: empty stdout from ollama-status CLI");
        return Ok(OllamaStatus::backend_error(
            "Status check returned no output.".to_string(),
        ));
    }
    match serde_json::from_str(&output) {
        Ok(status) => Ok(status),
        Err(e) => {
            eprintln!(
                "[get_ollama_status] parse error: {e}, raw ({} bytes): {:?}",
                output.len(),
                &output[..output.len().min(200)]
            );
            Ok(OllamaStatus::backend_error(format!(
                "Status check returned malformed output: {e}"
            )))
        }
    }
}

/// Preload the chat model into Ollama's memory.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn preload_ollama_model(state: State<'_, AppState>) -> Result<serde_json::Value, String> {
    let output = call_python_cli(&["ollama-preload"], &state.project_root).await?;
    serde_json::from_str(&output).map_err(|e| format!("Failed to parse preload result: {e}"))
}

/// Read the current model-pull download progress, if a pull is in flight.
///
/// Reads `~/.arandu/data/ollama_pull_progress.json` directly (written by
/// the Python pull loop) so the UI can poll cheaply once per second.
/// Returns `None` when no pull is running (file absent) or mid-write.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn get_model_pull_progress() -> Result<Option<ModelPullProgress>, String> {
    let Some(home) = dirs::home_dir() else {
        return Ok(None);
    };
    let path = home
        .join(".arandu")
        .join("data")
        .join("ollama_pull_progress.json");
    match tokio::fs::read_to_string(&path).await {
        // A partial/atomic-rename race parses as None — the next poll recovers.
        Ok(contents) => Ok(serde_json::from_str::<ModelPullProgress>(&contents).ok()),
        Err(_) => Ok(None),
    }
}

// ---------------------------------------------------------------------------
// Monitor commands
// ---------------------------------------------------------------------------

/// Get memory usage and database file sizes.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn get_memory_usage(state: State<'_, AppState>) -> Result<MemoryUsage, String> {
    let output = call_python_cli(&["monitor"], &state.project_root).await?;
    serde_json::from_str(&output).map_err(|e| format!("Failed to parse memory usage: {e}"))
}

// ---------------------------------------------------------------------------
// Pipeline commands
// ---------------------------------------------------------------------------

/// Get current pipeline status: last run, staleness, pending changes, estimate.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn get_pipeline_status(state: State<'_, AppState>) -> Result<PipelineStatus, String> {
    let output = call_python_cli(&["pipeline-status"], &state.project_root).await?;
    serde_json::from_str(&output).map_err(|e| format!("Failed to parse pipeline status: {e}"))
}

/// Start a pipeline run in the background; return immediately with run_id.
///
/// The run_id can be polled via `get_pipeline_run_result()`.
/// Rejects if a pipeline worker is already running.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn trigger_pipeline_run(
    state: State<'_, AppState>,
) -> Result<PipelineRunStarted, String> {
    // Concurrency guard: reject if any pipeline run is already running.
    {
        if is_pipeline_flag_set(&state).await {
            return Err("Pipeline is already running".to_string());
        }
        let mut running = state.pipeline_running.lock().await;
        *running = Some(Instant::now());
    }

    let run_id = uuid::Uuid::new_v4().to_string();
    let run_id_clone = run_id.clone();
    let project_root = state.project_root.clone();

    let pipeline_runs = Arc::clone(&state.pipeline_runs);
    let pipeline_running = Arc::clone(&state.pipeline_running);

    // Spawn background task — does not block the caller.
    tokio::spawn(async move {
        // Hard wall-clock budget so a hung LLM call inside the worker
        // can't keep the `pipeline_running` flag stuck.  Matches the
        // streamed-pipeline path's `PIPELINE_HARD_TIMEOUT`.
        let result = call_python_cli_with_timeout(
            &["pipeline-run", "--trigger", "api"],
            &project_root,
            PIPELINE_HARD_TIMEOUT.as_secs(),
        )
        .await;

        let summary: PipelineRunSummary = match &result {
            Ok(output) => match serde_json::from_str(output) {
                Ok(s) => s,
                Err(e) => PipelineRunSummary {
                    run_id: run_id_clone.clone(),
                    started_at: String::new(),
                    completed_at: String::new(),
                    duration_seconds: 0.0,
                    status: "failed".to_string(),
                    models_processed: vec![],
                    rows_processed: HashMap::new(),
                    rows_changed: HashMap::new(),
                    trigger: "api".to_string(),
                    error: Some(format!("Failed to parse pipeline output: {e}")),
                    plan_summary: None,
                    vector_index_status: None,
                    graph_index_status: None,
                    index_error: None,
                },
            },
            Err(e) => PipelineRunSummary {
                run_id: run_id_clone.clone(),
                started_at: String::new(),
                completed_at: String::new(),
                duration_seconds: 0.0,
                status: "failed".to_string(),
                models_processed: vec![],
                rows_processed: HashMap::new(),
                rows_changed: HashMap::new(),
                trigger: "api".to_string(),
                error: Some(format!("Pipeline subprocess failed: {e}")),
                plan_summary: None,
                vector_index_status: None,
                graph_index_status: None,
                index_error: None,
            },
        };

        {
            let mut runs = pipeline_runs.lock().await;
            runs.insert(run_id_clone, summary);
        }

        let mut running = pipeline_running.lock().await;
        *running = None;
    });

    Ok(PipelineRunStarted {
        run_id,
        status: "in_progress".to_string(),
    })
}

/// Poll for the result of a background pipeline run.
///
/// Returns `"in_progress"` while running, or the full result once complete.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn get_pipeline_run_result(
    run_id: String,
    state: State<'_, AppState>,
) -> Result<PipelineRunResult, String> {
    // Check if the pipeline worker is running with this run_id.
    {
        let worker = state.pipeline_worker.lock().await;
        if let Some(ref handle) = *worker {
            if handle.run_id == run_id {
                return Ok(PipelineRunResult {
                    run_id,
                    status: "in_progress".to_string(),
                    result: None,
                });
            }
        }
    }

    // Check completed runs.
    let runs = state.pipeline_runs.lock().await;
    if let Some(summary) = runs.get(&run_id) {
        return Ok(PipelineRunResult {
            run_id,
            status: "completed".to_string(),
            result: Some(summary.clone()),
        });
    }

    // Unknown run_id.
    Ok(PipelineRunResult {
        run_id,
        status: "not_found".to_string(),
        result: None,
    })
}

/// Start a pipeline run in an isolated worker process with streaming progress.
///
/// Spawns `nice -n 10 python3 -m src.pipeline.worker run --trigger <trigger>`
/// so the pipeline never blocks the UI thread. Emits `pipeline-progress` events.
/// Only one pipeline run is allowed at a time (rejects if already running).
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn trigger_pipeline_run_stream(
    trigger: String,
    mode: Option<String>,
    app_handle: AppHandle,
    state: State<'_, AppState>,
) -> Result<PipelineRunStarted, String> {
    // Concurrency guard: reject if any pipeline run is already running.
    {
        if is_pipeline_flag_set(&state).await {
            return Err("Pipeline is already running".to_string());
        }
        let mut running = state.pipeline_running.lock().await;
        *running = Some(Instant::now());
    }

    // Wait for any in-flight write-mode CLI command (startup-sync,
    // sync-all-stale) to finish before spawning the pipeline worker.
    // This prevents SQLite write-lock contention and Ollama request
    // queueing between the pipeline and a concurrent sync process.
    // Times out after 120s — if startup-sync is still running, reject
    // with a user-friendly error instead of creating contention.
    let write_lock = Arc::clone(&state.cli_write_lock);
    match tokio::time::timeout(
        std::time::Duration::from_secs(120),
        write_lock.lock(),
    ).await {
        Ok(_guard) => {
            // Lock acquired — other write commands are done.  Drop
            // immediately; the pipeline worker uses a file-based lock
            // (`.pipeline_running`) to coordinate with long-lived
            // writers like the WhatsApp listener.
            drop(_guard);
        }
        Err(_) => {
            eprintln!("[trigger_pipeline_run_stream] write lock wait timed out after 120s, rejecting");
            let mut running = state.pipeline_running.lock().await;
            *running = None;
            return Err(
                "Startup sync is still running. Please try again in a moment.".to_string()
            );
        }
    }

    // Defensive guard: if a previous cleanup task died before clearing
    // the handle (and the `PIPELINE_MAX_AGE` auto-reset hasn't fired
    // yet), kill any lingering worker so we don't stack two of them.
    reap_orphan_worker(&state).await;

    let run_id = uuid::Uuid::new_v4().to_string();
    let project_root = state.project_root.clone();

    let run_mode = mode.unwrap_or_else(|| "full".to_string());
    let mut child = match spawn_pipeline_worker(
        &trigger,
        &run_mode,
        &project_root,
        &app_handle,
        "pipeline-progress",
    )
    .await
    {
        Ok(child) => child,
        Err(e) => {
            let mut running = state.pipeline_running.lock().await;
            *running = None;
            return Err(e);
        }
    };

    // Cache the pid before moving the child into the cleanup task —
    // `cancel_pipeline_run` and the hard-timeout reaper signal by pid.
    let pid = child.id();

    // Store the cancel handle.  The `Child` itself is NOT stored here;
    // it lives in the cleanup task below so `child.wait().await`
    // doesn't hold the `pipeline_worker` mutex (which would deadlock
    // cancel).
    {
        let mut worker = state.pipeline_worker.lock().await;
        *worker = Some(PipelineWorkerHandle {
            run_id: run_id.clone(),
            pid,
        });
    }

    // Spawn a cleanup task that owns the Child, waits on it with a
    // wall-clock budget, hard-kills if it overruns, and clears the
    // tracking slots.
    let pipeline_worker = Arc::clone(&state.pipeline_worker);
    let pipeline_running = Arc::clone(&state.pipeline_running);
    tokio::spawn(async move {
        match tokio::time::timeout(
            PIPELINE_HARD_TIMEOUT,
            child.wait(),
        )
        .await
        {
            Ok(Ok(_status)) => {}
            Ok(Err(e)) => {
                eprintln!("[pipeline] child.wait() failed: {e}");
            }
            Err(_elapsed) => {
                eprintln!(
                    "[pipeline] worker exceeded {}s budget — killing pid={:?}",
                    PIPELINE_HARD_TIMEOUT.as_secs(),
                    pid,
                );
                terminate_worker_pid(pid).await;
                // After the kill, the child handle still needs to be
                // reaped so it doesn't linger as a zombie.
                let _ = child.wait().await;
            }
        }

        let mut guard = pipeline_worker.lock().await;
        *guard = None;
        let mut running = pipeline_running.lock().await;
        *running = None;
    });

    Ok(PipelineRunStarted {
        run_id,
        status: "in_progress".to_string(),
    })
}

/// Cancel the currently running pipeline worker.
///
/// Sends SIGTERM to the worker subprocess on Unix (graceful shutdown),
/// or kills it on Windows. The worker finishes the current model,
/// records partial stats, and exits with a "cancelled" status event.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn cancel_pipeline_run(state: State<'_, AppState>) -> Result<(), String> {
    // Grab the pid under the lock, then release it before signaling.
    // The cleanup task owns the Child handle and will reap on next
    // `wait()` return; we just need to deliver the signal.
    let pid = {
        let worker = state.pipeline_worker.lock().await;
        match worker.as_ref() {
            Some(handle) => handle.pid,
            None => {
                return Err(
                    "No pipeline run is currently in progress".to_string(),
                );
            }
        }
    };
    #[cfg(unix)]
    {
        if let Some(pid) = pid {
            let _ = std::process::Command::new("kill")
                .args(["-TERM", &pid.to_string()])
                .status();
        }
    }
    #[cfg(not(unix))]
    {
        if let Some(pid) = pid {
            let _ = std::process::Command::new("taskkill")
                .args(["/PID", &pid.to_string()])
                .status();
        }
    }
    Ok(())
}

/// Check if a pipeline worker is currently running.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn is_pipeline_running(state: State<'_, AppState>) -> Result<bool, String> {
    Ok(is_pipeline_flag_set(&state).await)
}

/// Get recent pipeline run history.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn get_pipeline_run_history(
    limit: usize,
    state: State<'_, AppState>,
) -> Result<Vec<PipelineRunSummary>, String> {
    let limit_str = limit.to_string();
    let output = call_python_cli(
        &["pipeline-run-history", "--limit", &limit_str],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output).map_err(|e| format!("Failed to parse pipeline history: {e}"))
}

// ---------------------------------------------------------------------------
// Firewall commands
// ---------------------------------------------------------------------------

/// Get recent entries from the audit log with pagination support.
///
/// # sensitivity_tier: 1 (audit metadata only)
#[tauri::command]
pub async fn get_audit_log(
    limit: Option<usize>,
    offset: Option<usize>,
) -> Result<Vec<AuditEntry>, String> {
    let limit = limit.unwrap_or(100);
    let offset = offset.unwrap_or(0);
    tokio::task::spawn_blocking(move || {
        let logger = AuditLogger::default_path().map_err(|e| format!("Audit logger error: {e}"))?;
        logger
            .get_recent_paginated(limit, offset)
            .map_err(|e| format!("Failed to read audit log: {e}"))
    })
    .await
    .map_err(|e| format!("Task join error: {e}"))?
}

/// Walk the SHA-256 chain and confirm every entry's `previous_hash`
/// matches the hash of the prior serialized line. Returns `true` for a
/// clean chain, `false` if any link is broken (tampering or schema drift).
///
/// # sensitivity_tier: 1 (audit metadata only)
#[tauri::command]
pub async fn verify_audit_chain() -> Result<bool, String> {
    tokio::task::spawn_blocking(move || {
        let logger = AuditLogger::default_path().map_err(|e| format!("Audit logger error: {e}"))?;
        logger
            .verify_chain()
            .map_err(|e| format!("Failed to verify audit chain: {e}"))
    })
    .await
    .map_err(|e| format!("Task join error: {e}"))?
}

/// Return the original/redacted payload pair persisted alongside a
/// redacting `egress_decision` / `egress_redaction` audit row.
///
/// `payload_hash` is taken straight from the audit row's `payload_hash`
/// field — we reject anything that isn't a hex SHA-256 to keep the
/// filesystem path safe. Returns `{detail: null}` when no blob exists
/// (e.g. a non-redacting row) so the frontend can render an empty
/// state without an error toast.
///
/// # sensitivity_tier: 3 (returns raw Tier 3 message content)
#[tauri::command]
pub async fn get_redaction_detail(
    payload_hash: String,
    state: State<'_, AppState>,
) -> Result<RedactionDetailResponse, String> {
    if payload_hash.len() != 64 || !payload_hash.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err("payload_hash must be a 64-char hex SHA-256".to_string());
    }
    let output = call_python_cli(
        &["get-redaction-detail", "--payload-hash", &payload_hash],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse redaction detail: {e}"))
}

// ---------------------------------------------------------------------------
// Settings commands
// ---------------------------------------------------------------------------

/// Get current application settings.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn get_settings(state: State<'_, AppState>) -> Result<AppSettings, String> {
    let settings = state.settings.lock().await;
    Ok(settings.clone())
}

/// Update application settings and persist to disk.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn update_settings(
    settings: AppSettings,
    app_handle: AppHandle,
    state: State<'_, AppState>,
) -> Result<(), String> {
    // Snapshot previous values for change detection.
    let (prev_prevent_sleep, prev_battery, prev_login, prev_menu_bar) = {
        let current = state.settings.lock().await;
        (
            current.prevent_sleep,
            current.prevent_sleep_on_battery,
            current.launch_at_login,
            current.menu_bar_mode,
        )
    };
    let new_prevent_sleep = settings.prevent_sleep;
    let new_battery = settings.prevent_sleep_on_battery;
    let new_login = settings.launch_at_login;
    let new_menu_bar = settings.menu_bar_mode;

    let settings_for_disk = settings.clone();
    tokio::task::spawn_blocking(move || save_settings_to_disk(&settings_for_disk))
        .await
        .map_err(|e| format!("Task join error: {e}"))??;
    let mut current = state.settings.lock().await;
    *current = settings;
    drop(current);

    // --- Keep-awake toggles ---
    if new_prevent_sleep != prev_prevent_sleep || new_battery != prev_battery {
        if let Some(caf_state) = app_handle.try_state::<std::sync::Arc<crate::keep_awake::CaffeinateHandle>>() {
            let caf: std::sync::Arc<crate::keep_awake::CaffeinateHandle> = caf_state.inner().clone();
            tauri::async_runtime::spawn(async move {
                crate::keep_awake::apply_caffeinate(&caf, new_prevent_sleep, new_battery).await;
            });
        }
    }

    if new_login != prev_login {
        crate::keep_awake::apply_launch_at_login(new_login);
    }

    if new_menu_bar != prev_menu_bar {
        crate::keep_awake::apply_menu_bar_mode(&app_handle, new_menu_bar);
    }

    Ok(())
}

/// Rebuild ChromaDB + BM25 under a new embedding model.
///
/// Wraps `python -m src.core.cli rebuild-vector-index ...` which in
/// turn calls `src.core.chromadb.migrate`. The full reindex can take
/// minutes for larger corpora — the UI should disable interaction
/// and show progress while this resolves.
///
/// `provider` is one of `ollama` or `openai`. `dry_run=true`
/// estimates cost without dropping or rebuilding.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn rebuild_vector_index(
    to_model: String,
    provider: Option<String>,
    api_key: Option<String>,
    base_url: Option<String>,
    dimensions: Option<i64>,
    dry_run: Option<bool>,
    state: State<'_, AppState>,
) -> Result<RebuildVectorIndexResult, String> {
    let provider = provider.unwrap_or_else(|| "ollama".to_string());
    let dry_run = dry_run.unwrap_or(false);

    let mut args: Vec<String> = vec![
        "rebuild-vector-index".to_string(),
        "--to-model".to_string(),
        to_model,
        "--provider".to_string(),
        provider,
    ];
    if let Some(k) = api_key {
        args.push("--api-key".to_string());
        args.push(k);
    }
    if let Some(u) = base_url {
        args.push("--base-url".to_string());
        args.push(u);
    }
    if let Some(d) = dimensions {
        args.push("--dimensions".to_string());
        args.push(d.to_string());
    }
    if dry_run {
        args.push("--dry-run".to_string());
    }

    let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
    let output = call_python_cli(&arg_refs, &state.project_root).await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse rebuild_vector_index JSON: {e}"))
}

// ---------------------------------------------------------------------------
// Settings persistence
// ---------------------------------------------------------------------------

fn settings_path() -> Result<std::path::PathBuf, String> {
    let home = dirs::home_dir().ok_or("Could not determine home directory")?;
    Ok(home.join(".arandu").join("settings.json"))
}

pub(crate) fn load_settings_from_disk() -> Result<AppSettings, String> {
    let path = settings_path()?;
    if !path.exists() {
        return Ok(AppSettings::default());
    }
    let contents =
        std::fs::read_to_string(&path).map_err(|e| format!("Failed to read settings: {e}"))?;

    // Detect pre-E1.4 settings files that lack the onboarding field entirely.
    // These users should skip the wizard.  Files that DO contain the field
    // (even if false) represent a wizard in progress — respect that value.
    let is_legacy = !contents.contains("onboarding_completed");

    let mut settings: AppSettings = match serde_json::from_str(&contents) {
        Ok(s) => s,
        Err(strict_err) => {
            // A user's settings file must NEVER be silently discarded: the
            // caller falls back to AppSettings::default() on Err, after
            // which onboarding overwrites the file — so a single field a
            // struct change made unreadable (e.g. a new app version, a
            // Pro-written field, a stray null) would reset the WHOLE config
            // and bounce the user back through onboarding, losing their
            // provider choice and prefs. Preserve the original verbatim,
            // then recover every field that still parses.
            backup_unparseable_settings(&path, &contents);
            recover_settings_leniently(&contents).ok_or_else(|| {
                format!("settings unreadable and unrecoverable: {strict_err}")
            })?
        }
    };

    if is_legacy && !settings.onboarding_completed {
        settings.onboarding_completed = true;
        let _ = save_settings_to_disk(&settings);
    }
    Ok(settings)
}

/// Copy an unparseable settings file aside before any lossy fallback, so
/// the user's data is never destroyed. Timestamped so repeated failures
/// don't clobber each other. Best-effort — failure here is non-fatal.
fn backup_unparseable_settings(path: &std::path::Path, contents: &str) {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let backup = path.with_file_name(format!("settings.corrupt-{ts}.json"));
    if let Err(e) = std::fs::write(&backup, contents) {
        eprintln!("settings backup failed ({}): {e}", backup.display());
    } else {
        eprintln!(
            "settings could not be parsed; original preserved at {}",
            backup.display(),
        );
    }
}

/// Recover an AppSettings from a partly-incompatible settings object by
/// keeping every field that still deserializes and letting the rest fall
/// back to its default. One malformed value can no longer reset the whole
/// config. Returns `None` only when the file isn't even a JSON object.
fn recover_settings_leniently(contents: &str) -> Option<AppSettings> {
    let value: serde_json::Value = serde_json::from_str(contents).ok()?;
    let obj = value.as_object()?;
    // Add fields back one at a time, dropping any that break the parse.
    // Relies on every AppSettings field having a serde default, so a
    // subset object always deserializes.
    let mut acc = serde_json::Map::new();
    for (key, val) in obj {
        acc.insert(key.clone(), val.clone());
        let candidate = serde_json::Value::Object(acc.clone());
        if serde_json::from_value::<AppSettings>(candidate).is_err() {
            acc.remove(key); // incompatible field → falls back to default
        }
    }
    serde_json::from_value(serde_json::Value::Object(acc)).ok()
}

fn save_settings_to_disk(settings: &AppSettings) -> Result<(), String> {
    let path = settings_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create settings directory: {e}"))?;
    }
    let json = serde_json::to_string_pretty(settings)
        .map_err(|e| format!("Failed to serialize settings: {e}"))?;
    std::fs::write(&path, json).map_err(|e| format!("Failed to write settings: {e}"))
}

// ---------------------------------------------------------------------------
// Connector commands
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
fn macos_permission_settings_url(permission: &str) -> Option<&'static str> {
    match permission {
        "macOS Calendar" => Some(
            "x-apple.systempreferences:com.apple.preference.security?Privacy_Automation",
        ),
        "macOS Contacts" => Some(
            "x-apple.systempreferences:com.apple.preference.security?Privacy_Automation",
        ),
        "macOS Notes" => Some(
            "x-apple.systempreferences:com.apple.preference.security?Privacy_Automation",
        ),
        "macOS Mail" => Some(
            "x-apple.systempreferences:com.apple.preference.security?Privacy_Automation",
        ),
        "Full Disk Access" => Some(
            "x-apple.systempreferences:com.apple.preference.security?Privacy_AllFiles",
        ),
        _ => None,
    }
}

/// Open macOS System Settings to the requested privacy permission pane.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub fn open_macos_permission_settings(permission: String) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        let url = macos_permission_settings_url(&permission).unwrap_or(
            "x-apple.systempreferences:com.apple.preference.security",
        );
        // Hardcode the absolute paths: when the app is launched via a
        // LaunchAgent (or any context with a sparse PATH) plain "open" /
        // "osascript" fail with ENOENT and the click looks dead.
        let output = std::process::Command::new("/usr/bin/open")
            .arg(url)
            .output()
            .map_err(|e| format!("Failed to spawn /usr/bin/open: {e}"))?;
        if !output.status.success() {
            return Err(format!(
                "/usr/bin/open exited with status {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
        // On macOS 14+ a URL-scheme open from another app frequently
        // launches System Settings behind the calling window without
        // taking focus, making the click look like a no-op. Force it
        // to the foreground.
        let _ = std::process::Command::new("/usr/bin/osascript")
            .args([
                "-e",
                r#"tell application "System Settings" to activate"#,
            ])
            .status();
        Ok(())
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = permission;
        Err("Opening macOS permission settings is only supported on macOS".to_string())
    }
}

/// Get the connector catalog with current status for each connector.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn get_connector_catalog(
    state: State<'_, AppState>,
) -> Result<Vec<types::ConnectorCatalogEntry>, String> {
    let output = call_python_cli(&["connector-catalog"], &state.project_root).await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse connector catalog JSON: {e}"))
}

/// Aggregated system health: overall verdict, data-flow stages, and a
/// flat list of actionable issues. Single source of truth for the
/// unified status surface (passthrough; the frontend owns the shape).
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn get_system_health(
    state: State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let output = call_python_cli(&["system-health"], &state.project_root).await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse system health JSON: {e}"))
}

/// Toggle a connector on or off.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn toggle_connector(
    connector_id: String,
    enabled: bool,
    user_inputs: Option<serde_json::Value>,
    state: State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let enabled_str = if enabled { "true" } else { "false" };
    let mut args = vec![
        "toggle-connector".to_string(),
        connector_id,
        "--enabled".to_string(),
        enabled_str.to_string(),
    ];

    if let Some(raw_inputs) = user_inputs {
        let normalized_inputs = match raw_inputs {
            serde_json::Value::String(s) => serde_json::from_str::<serde_json::Value>(&s)
                .map_err(|e| format!("Invalid user_inputs JSON string: {e}"))?,
            other => other,
        };
        if !normalized_inputs.is_object() {
            return Err("user_inputs must be a JSON object".to_string());
        }
        let inputs_str =
            serde_json::to_string(&normalized_inputs).map_err(|e| format!("Failed to serialize inputs: {e}"))?;
        args.push("--user-inputs".to_string());
        args.push(inputs_str);
    }

    let str_args: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    let output = call_python_cli(&str_args, &state.project_root).await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse toggle result JSON: {e}"))
}

/// Trigger an immediate sync for a connector.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn sync_connector_now(
    connector_id: String,
    state: State<'_, AppState>,
) -> Result<types::ConnectorSyncResult, String> {
    let output =
        call_python_cli(&["sync-connector", &connector_id], &state.project_root).await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse sync result JSON: {e}"))
}

/// Get full details for a single connector.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn get_connector_details(
    connector_id: String,
    state: State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let output =
        call_python_cli(&["connector-details", &connector_id], &state.project_root).await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse connector details JSON: {e}"))
}

/// Return WhatsApp listener runtime status (phase, qr pairing data, etc.).
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn get_whatsapp_listener_status(
    state: State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let output = call_python_cli(&["whatsapp-listener-status"], &state.project_root).await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse whatsapp listener status JSON: {e}"))
}

// ---------------------------------------------------------------------------
// Extension installer commands
// ---------------------------------------------------------------------------

/// Discover tools and schema from an MCP server command.
///
/// `env` carries Tier 3 secrets (API tokens) to expose to the MCP
/// server subprocess. Serialized to JSON and passed via `--env`.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn install_extension_discover(
    command: String,
    args: Vec<String>,
    name: Option<String>,
    env: Option<std::collections::HashMap<String, String>>,
    state: State<'_, AppState>,
) -> Result<types::InstallPreview, String> {
    let mut cli_args = vec!["discover-extension".to_string(), command];
    if let Some(n) = name {
        cli_args.push("--name".to_string());
        cli_args.push(n);
    }
    if let Some(env_map) = env.as_ref().filter(|m| !m.is_empty()) {
        let env_json = serde_json::to_string(env_map)
            .map_err(|e| format!("Failed to serialize env vars: {e}"))?;
        cli_args.push("--env".to_string());
        cli_args.push(env_json);
    }
    // Insert `--` so argparse treats MCP server args (e.g. `-y`) as
    // positional values, not CLI flags.
    cli_args.push("--".to_string());
    cli_args.extend(args);
    let str_args: Vec<&str> = cli_args.iter().map(|s| s.as_str()).collect();
    let output = call_python_cli(&str_args, &state.project_root).await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse install preview JSON: {e}"))
}

/// Confirm and finalize an extension install.
///
/// `env` is persisted to the connector registry so future syncs can
/// relaunch the MCP server with the same secrets.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn install_extension_confirm(
    preview_json: String,
    name: Option<String>,
    env: Option<std::collections::HashMap<String, String>>,
    state: State<'_, AppState>,
) -> Result<types::InstallConfirmResult, String> {
    let mut cli_args = vec![
        "confirm-extension".to_string(),
        "--preview-json".to_string(),
        preview_json,
    ];
    if let Some(n) = name {
        cli_args.push("--name".to_string());
        cli_args.push(n);
    }
    if let Some(env_map) = env.as_ref().filter(|m| !m.is_empty()) {
        let env_json = serde_json::to_string(env_map)
            .map_err(|e| format!("Failed to serialize env vars: {e}"))?;
        cli_args.push("--env".to_string());
        cli_args.push(env_json);
    }
    let str_args: Vec<&str> = cli_args.iter().map(|s| s.as_str()).collect();
    let output = call_python_cli(&str_args, &state.project_root).await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse install confirm JSON: {e}"))
}

// ---------------------------------------------------------------------------
// Model generator commands
// ---------------------------------------------------------------------------

/// Generate pipeline models for a new data source.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn generate_models(
    connector_id: String,
    mapping_json: String,
    state: State<'_, AppState>,
) -> Result<types::ModelPreview, String> {
    let output = call_python_cli(
        &["generate-models", "--connector-id", &connector_id, "--mapping-json", &mapping_json],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse model preview JSON: {e}"))
}

/// Approve staged models and install into the pipeline.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn approve_models(
    connector_id: String,
    state: State<'_, AppState>,
) -> Result<types::ModelApproveResult, String> {
    let output = call_python_cli(
        &["approve-models", "--connector-id", &connector_id],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse model approve JSON: {e}"))
}

/// Reject staged models without installing.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn reject_models(
    connector_id: String,
    state: State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let output = call_python_cli(
        &["reject-models", "--connector-id", &connector_id],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse model reject JSON: {e}"))
}

// ---------------------------------------------------------------------------
// Agent runner commands
// ---------------------------------------------------------------------------

/// List all discovered agents with status.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn list_agents(state: State<'_, AppState>) -> Result<Vec<AgentFullStatus>, String> {
    let output = call_python_cli(&["list-agents"], &state.project_root).await?;
    serde_json::from_str(&output).map_err(|e| format!("Failed to parse agents JSON: {e}"))
}

/// Run an agent by ID and return the result.
///
/// # sensitivity_tier: 2
#[tauri::command]
pub async fn run_agent(
    agent_id: String,
    params: Option<String>,
    state: State<'_, AppState>,
) -> Result<AgentRunResult, String> {
    let mut cli_args = vec![
        "run-agent".to_string(),
        "--agent-id".to_string(),
        agent_id,
    ];
    if let Some(p) = params {
        cli_args.push("--params".to_string());
        cli_args.push(p);
    }
    let str_args: Vec<&str> = cli_args.iter().map(|s| s.as_str()).collect();
    let output = call_python_cli(&str_args, &state.project_root).await?;
    serde_json::from_str(&output).map_err(|e| format!("Failed to parse agent run JSON: {e}"))
}

/// Get the last result from an agent run.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn get_agent_result(
    agent_id: String,
    state: State<'_, AppState>,
) -> Result<AgentRunResult, String> {
    let output = call_python_cli(
        &["get-agent-result", "--agent-id", &agent_id],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output).map_err(|e| format!("Failed to parse agent result JSON: {e}"))
}

/// List all registered skills (delegates to v2 SKILL.md loader).
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn list_skills(state: State<'_, AppState>) -> Result<Vec<serde_json::Value>, String> {
    let output = call_python_cli(&["skills-list-v2"], &state.project_root).await?;
    serde_json::from_str(&output).map_err(|e| format!("Failed to parse skills JSON: {e}"))
}

// ---------------------------------------------------------------------------
// Action tool commands
// ---------------------------------------------------------------------------

/// List available MCP action tools from enabled connectors.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn get_available_actions(
    state: State<'_, AppState>,
) -> Result<Vec<types::AvailableAction>, String> {
    let output = call_python_cli(&["list-actions"], &state.project_root).await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse available actions JSON: {e}"))
}

/// Execute a confirmed MCP action.
///
/// # sensitivity_tier: 3
#[tauri::command]
pub async fn confirm_action(
    proposal_json: String,
    state: State<'_, AppState>,
) -> Result<types::ActionResult, String> {
    let output = call_python_cli(
        &["confirm-action", "--proposal-json", &proposal_json],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse action result JSON: {e}"))
}

/// Cancel a proposed action.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn cancel_action(
    proposal_id: String,
    state: State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let output = call_python_cli(
        &["cancel-action", "--proposal-id", &proposal_id],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse cancel result JSON: {e}"))
}

/// Resume a disambiguation proposal with the user's chosen candidate.
///
/// Returns the new action_proposal payload (same wire shape the
/// streaming chunk uses) so the frontend renders the regular
/// confirmation card on the spot.
///
/// # sensitivity_tier: 3
#[tauri::command]
pub async fn resume_action_with_recipient(
    disambiguation_json: String,
    candidate_json: String,
    state: State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let output = call_python_cli(
        &[
            "resume-action-with-recipient",
            "--disambiguation-json",
            &disambiguation_json,
            "--candidate-json",
            &candidate_json,
        ],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output).map_err(|e| {
        format!("Failed to parse resume_action_with_recipient JSON: {e}")
    })
}

/// Search the user's contacts for a Send Message recipient.
///
/// Powers the inline search input on the recipient disambiguation
/// card. ``include_apple=false`` searches only the local DB (fast —
/// runs on every debounced keystroke). ``include_apple=true`` spawns
/// the apple-contacts MCP subprocess as well (slower — gated behind
/// the "Also search Apple Contacts" button).
///
/// # sensitivity_tier: 3
#[tauri::command]
pub async fn search_recipient_candidates(
    query: String,
    channel: String,
    include_apple: bool,
    limit: Option<u32>,
    state: State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let limit_str = limit.unwrap_or(5).to_string();
    let mut args: Vec<&str> = vec![
        "search-recipient-candidates",
        "--query",
        &query,
        "--channel",
        &channel,
        "--limit",
        &limit_str,
    ];
    if include_apple {
        args.push("--include-apple");
    }
    let output = call_python_cli(&args, &state.project_root).await?;
    serde_json::from_str(&output).map_err(|e| {
        format!("Failed to parse search_recipient_candidates JSON: {e}")
    })
}

// ---------------------------------------------------------------------------
// Extension management commands
// ---------------------------------------------------------------------------

/// Uninstall an extension: disable connector, remove from registry, clean up models.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn uninstall_extension(
    connector_id: String,
    preserve_data: Option<bool>,
    state: State<'_, AppState>,
) -> Result<UninstallResult, String> {
    let preserve = if preserve_data.unwrap_or(true) {
        "true"
    } else {
        "false"
    };
    let output = call_python_cli(
        &["uninstall-extension", &connector_id, "--preserve-data", preserve],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse uninstall result JSON: {e}"))
}

/// Get sync history for a connector.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn get_connector_history(
    connector_id: String,
    limit: Option<usize>,
    state: State<'_, AppState>,
) -> Result<Vec<ConnectorHistoryEntry>, String> {
    let limit = limit.unwrap_or(20);
    let limit_str = limit.to_string();
    let output = call_python_cli(
        &["connector-history", &connector_id, "--limit", &limit_str],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse connector history JSON: {e}"))
}

/// Get recent log lines for an extension.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn get_extension_logs(
    extension_id: String,
    lines: Option<usize>,
    state: State<'_, AppState>,
) -> Result<ExtensionLogOutput, String> {
    let lines = lines.unwrap_or(50);
    let lines_str = lines.to_string();
    let output = call_python_cli(
        &["extension-logs", &extension_id, "--lines", &lines_str],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse extension logs JSON: {e}"))
}

// ---------------------------------------------------------------------------
// Health check command
// ---------------------------------------------------------------------------

/// Check all system components and report their status.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn health_check(
    state: State<'_, AppState>,
) -> Result<HealthCheckResult, String> {
    let output = call_python_cli(
        &["health"],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse health JSON: {e}"))
}

/// Get the user's interest profile.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn get_interest_profile(
    state: State<'_, AppState>,
) -> Result<Vec<types::InterestArea>, String> {
    let output = call_python_cli(
        &["get-interests"],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output).map_err(|e| {
        format!("Failed to parse interest profile JSON: {e}")
    })
}

/// Get per-domain query statistics.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn get_domain_stats(
    state: State<'_, AppState>,
) -> Result<Vec<types::DomainStats>, String> {
    let output = call_python_cli(
        &["get-domain-stats"],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output).map_err(|e| {
        format!("Failed to parse domain stats JSON: {e}")
    })
}

/// Return a smart pipeline refresh plan.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn get_refresh_plan(
    state: State<'_, AppState>,
) -> Result<types::RefreshPlan, String> {
    let output = call_python_cli(
        &["plan-refresh"],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output).map_err(|e| {
        format!("Failed to parse refresh plan JSON: {e}")
    })
}

/// Get active (non-dismissed) insights.
///
/// # sensitivity_tier: 1 (reads stored data)
#[tauri::command]
pub async fn get_insights(
    state: State<'_, AppState>,
    limit: Option<i32>,
) -> Result<Vec<types::Insight>, String> {
    let limit_str = limit.unwrap_or(3).to_string();
    let output = call_python_cli(
        &["get-insights", "--limit", &limit_str],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output).map_err(|e| {
        format!("Failed to parse insights JSON: {e}")
    })
}

/// Generate daily insights from recent question patterns.
///
/// Requires Ollama to be running.
///
/// # sensitivity_tier: 3
#[tauri::command]
pub async fn generate_insights(
    state: State<'_, AppState>,
) -> Result<Vec<types::Insight>, String> {
    let output = call_python_cli(
        &["generate-insights"],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output).map_err(|e| {
        format!("Failed to parse generated insights JSON: {e}")
    })
}

/// Dismiss an insight by its ID.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn dismiss_insight(
    state: State<'_, AppState>,
    insight_id: String,
) -> Result<serde_json::Value, String> {
    let output = call_python_cli(
        &["dismiss-insight", "--insight-id", &insight_id],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output).map_err(|e| {
        format!("Failed to parse dismiss response: {e}")
    })
}

/// Follow up on an insight (boosts domain interest).
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn follow_up_insight(
    state: State<'_, AppState>,
    insight_id: String,
) -> Result<serde_json::Value, String> {
    let output = call_python_cli(
        &["follow-up-insight", "--insight-id", &insight_id],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output).map_err(|e| {
        format!("Failed to parse follow-up response: {e}")
    })
}

// ---------------------------------------------------------------------------
// Proactive intelligence commands
// ---------------------------------------------------------------------------

/// Run proactive intelligence evaluation (all 3 pillars).
///
/// Emits a ``arandu:proactive-refreshed`` Tauri event on success so
/// dashboard widgets that depend on the proactive tables (active
/// threads, inbox, agent stream) can refetch immediately — the 2h
/// background cycle would otherwise leave those panels stale.
///
/// # sensitivity_tier: 3
#[tauri::command]
pub async fn evaluate_proactive(
    app_handle: AppHandle,
    state: State<'_, AppState>,
) -> Result<types::ProactiveResult, String> {
    let output = call_python_cli(
        &["evaluate-proactive"],
        &state.project_root,
    )
    .await?;
    let result: types::ProactiveResult = serde_json::from_str(&output).map_err(|e| {
        format!("Failed to parse proactive result: {e}")
    })?;
    if let Err(e) = app_handle.emit("arandu:proactive-refreshed", ()) {
        eprintln!("[evaluate_proactive] WARNING: failed to emit proactive-refreshed event: {e}");
    }
    Ok(result)
}

/// Get pending replies (messages needing response).
///
/// # sensitivity_tier: 2
#[tauri::command]
pub async fn get_pending_replies(
    state: State<'_, AppState>,
) -> Result<Vec<types::PendingReply>, String> {
    let output = call_python_cli(
        &["get-pending-replies"],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output).map_err(|e| {
        format!("Failed to parse pending replies: {e}")
    })
}

/// Get proactive loop freshness (last successful evaluation + counts).
///
/// Powers the Dashboard's starvation indicator — without it a skipped
/// or failing proactive loop is indistinguishable from "nothing to
/// report".
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn get_proactive_status(
    state: State<'_, AppState>,
) -> Result<types::ProactiveStatus, String> {
    let output = call_python_cli(
        &["proactive-status"],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output).map_err(|e| {
        format!("Failed to parse proactive status: {e}")
    })
}

/// Get contact contexts (important people with ongoing situations).
///
/// # sensitivity_tier: 3
#[tauri::command]
pub async fn get_contact_contexts(
    state: State<'_, AppState>,
) -> Result<Vec<types::ContactContext>, String> {
    let output = call_python_cli(
        &["get-contact-contexts"],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output).map_err(|e| {
        format!("Failed to parse contact contexts: {e}")
    })
}

/// Get actionable events (calendar events and birthdays needing action).
///
/// # sensitivity_tier: 2
#[tauri::command]
pub async fn get_actionable_events(
    state: State<'_, AppState>,
) -> Result<Vec<types::ActionableEvent>, String> {
    let output = call_python_cli(
        &["get-actionable-events"],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output).map_err(|e| {
        format!("Failed to parse actionable events: {e}")
    })
}

/// Prep briefs for upcoming meetings with known attendees.
///
/// Deterministically composed by the Python CLI from cached contact
/// contexts + the enriched-events / contact-topics marts (no LLM), so
/// it's cheap to call on page load.
///
/// # sensitivity_tier: 3
#[tauri::command]
pub async fn get_meeting_prep_briefs(
    within_hours: Option<f64>,
    state: State<'_, AppState>,
) -> Result<Vec<types::MeetingPrepBrief>, String> {
    let hours = within_hours.unwrap_or(24.0).to_string();
    let output = call_python_cli(
        &["get-meeting-prep-briefs", "--within-hours", &hours],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output).map_err(|e| {
        format!("Failed to parse meeting prep briefs: {e}")
    })
}

/// Dismiss a pending reply.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn dismiss_pending_reply(
    id: String,
    app_handle: AppHandle,
    state: State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let output = call_python_cli(
        &["dismiss-pending-reply", "--id", &id],
        &state.project_root,
    )
    .await?;
    let parsed: serde_json::Value = serde_json::from_str(&output).map_err(|e| {
        format!("Failed to parse dismiss response: {e}")
    })?;
    if let Err(e) = app_handle.emit("arandu:proactive-refreshed", ()) {
        eprintln!(
            "[dismiss_pending_reply] WARNING: failed to emit proactive-refreshed event: {e}"
        );
    }
    Ok(parsed)
}

/// Dismiss an actionable event.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn dismiss_actionable_event(
    id: String,
    state: State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let output = call_python_cli(
        &["dismiss-actionable-event", "--id", &id],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output).map_err(|e| {
        format!("Failed to parse dismiss response: {e}")
    })
}

// ---------------------------------------------------------------------------
// Tasks / Goals / Habits / Schedule commands
// ---------------------------------------------------------------------------

/// List goals.
///
/// # sensitivity_tier: 2
#[tauri::command]
pub async fn list_goals(
    status: Option<String>,
    category: Option<String>,
    state: State<'_, AppState>,
) -> Result<Vec<types::Goal>, String> {
    let mut args: Vec<String> = vec!["goals-list".to_string()];
    if let Some(s) = status {
        args.push("--status".to_string());
        args.push(s);
    }
    if let Some(c) = category {
        args.push("--category".to_string());
        args.push(c);
    }
    let str_args: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    let output = call_python_cli(&str_args, &state.project_root).await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse goals: {e}"))
}

/// Create a goal.
///
/// # sensitivity_tier: 2
#[derive(serde::Deserialize)]
pub struct GoalCreatePayload {
    pub title: String,
    pub category: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub horizon: Option<String>,
    #[serde(default)]
    pub target_date: Option<String>,
    #[serde(default)]
    pub importance: Option<i32>,
    #[serde(default)]
    pub why: String,
}

#[tauri::command]
pub async fn create_goal(
    payload: GoalCreatePayload,
    state: State<'_, AppState>,
) -> Result<types::Goal, String> {
    let horizon = payload.horizon.unwrap_or_else(|| "medium".to_string());
    let importance = payload.importance.unwrap_or(5).to_string();
    let mut args: Vec<String> = vec![
        "goals-create".to_string(),
        "--title".to_string(), payload.title,
        "--category".to_string(), payload.category,
        "--description".to_string(), payload.description,
        "--horizon".to_string(), horizon,
        "--importance".to_string(), importance,
        "--why".to_string(), payload.why,
    ];
    if let Some(td) = payload.target_date {
        args.push("--target-date".to_string());
        args.push(td);
    }
    let str_args: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    let output = call_python_cli(&str_args, &state.project_root).await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse goal: {e}"))
}

/// Update mutable fields on a goal.
///
/// # sensitivity_tier: 2
#[tauri::command]
pub async fn update_goal(
    id: String,
    patch: serde_json::Value,
    state: State<'_, AppState>,
) -> Result<Option<types::Goal>, String> {
    let patch_str = patch.to_string();
    let output = call_python_cli(
        &["goals-update", "--id", &id, "--patch", &patch_str],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse goal update: {e}"))
}

/// Run the goal extractor over recent evidence; returns newly
/// inserted goals.
///
/// # sensitivity_tier: 3
#[tauri::command]
pub async fn mine_goals(
    state: State<'_, AppState>,
) -> Result<Vec<types::Goal>, String> {
    let output = call_python_cli(
        &["goals-mine"],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse mined goals: {e}"))
}

/// List projects.
///
/// # sensitivity_tier: 2
#[tauri::command]
pub async fn list_projects(
    status: Option<String>,
    category: Option<String>,
    state: State<'_, AppState>,
) -> Result<Vec<types::Project>, String> {
    let mut args: Vec<String> = vec!["projects-list".to_string()];
    if let Some(s) = status {
        args.push("--status".to_string());
        args.push(s);
    }
    if let Some(c) = category {
        args.push("--category".to_string());
        args.push(c);
    }
    let str_args: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    let output = call_python_cli(&str_args, &state.project_root).await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse projects: {e}"))
}

/// Create a project.
///
/// # sensitivity_tier: 2
#[derive(serde::Deserialize)]
pub struct ProjectCreatePayload {
    pub name: String,
    #[serde(default)]
    pub category: Option<String>,
    #[serde(default)]
    pub goal_id: Option<String>,
}

#[tauri::command]
pub async fn create_project(
    payload: ProjectCreatePayload,
    state: State<'_, AppState>,
) -> Result<types::Project, String> {
    let category = payload.category.unwrap_or_else(|| "personal".to_string());
    let mut args: Vec<String> = vec![
        "projects-create".to_string(),
        "--name".to_string(), payload.name,
        "--category".to_string(), category,
    ];
    if let Some(g) = payload.goal_id {
        args.push("--goal-id".to_string());
        args.push(g);
    }
    let str_args: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    let output = call_python_cli(&str_args, &state.project_root).await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse project: {e}"))
}

/// Archive a project.
///
/// # sensitivity_tier: 2
#[tauri::command]
pub async fn archive_project(
    id: String,
    state: State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let output = call_python_cli(
        &["projects-archive", "--id", &id],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse archive response: {e}"))
}

/// List tasks.
///
/// # sensitivity_tier: 2
#[tauri::command]
pub async fn list_tasks(
    status: Option<String>,
    project_id: Option<String>,
    goal_id: Option<String>,
    parent_task_id: Option<String>,
    state: State<'_, AppState>,
) -> Result<Vec<types::Task>, String> {
    let mut args: Vec<String> = vec!["tasks-list".to_string()];
    if let Some(s) = status {
        args.push("--status".to_string());
        args.push(s);
    }
    if let Some(p) = project_id {
        args.push("--project-id".to_string());
        args.push(p);
    }
    if let Some(g) = goal_id {
        args.push("--goal-id".to_string());
        args.push(g);
    }
    if let Some(pt) = parent_task_id {
        args.push("--parent-task-id".to_string());
        args.push(pt);
    }
    let str_args: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    let output = call_python_cli(&str_args, &state.project_root).await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse tasks: {e}"))
}

/// Create a task.
///
/// # sensitivity_tier: 2
#[derive(serde::Deserialize)]
pub struct TaskCreatePayload {
    pub title: String,
    #[serde(default)]
    pub project_id: Option<String>,
    #[serde(default)]
    pub parent_task_id: Option<String>,
    #[serde(default)]
    pub goal_id: Option<String>,
    #[serde(default)]
    pub notes: String,
    #[serde(default)]
    pub importance: Option<i32>,
    #[serde(default)]
    pub due_at: Option<String>,
}

#[tauri::command]
pub async fn create_task(
    payload: TaskCreatePayload,
    state: State<'_, AppState>,
) -> Result<types::Task, String> {
    let importance = payload.importance.unwrap_or(5).to_string();
    let mut args: Vec<String> = vec![
        "tasks-create".to_string(),
        "--title".to_string(), payload.title,
        "--importance".to_string(), importance,
        "--notes".to_string(), payload.notes,
    ];
    if let Some(p) = payload.project_id {
        args.push("--project-id".to_string());
        args.push(p);
    }
    if let Some(pt) = payload.parent_task_id {
        args.push("--parent-task-id".to_string());
        args.push(pt);
    }
    if let Some(g) = payload.goal_id {
        args.push("--goal-id".to_string());
        args.push(g);
    }
    if let Some(d) = payload.due_at {
        args.push("--due-at".to_string());
        args.push(d);
    }
    let str_args: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    let output = call_python_cli(&str_args, &state.project_root).await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse task: {e}"))
}

/// Update mutable fields on a task.
///
/// # sensitivity_tier: 2
#[tauri::command]
pub async fn update_task(
    id: String,
    patch: serde_json::Value,
    state: State<'_, AppState>,
) -> Result<Option<types::Task>, String> {
    let patch_str = patch.to_string();
    let output = call_python_cli(
        &["tasks-update", "--id", &id, "--patch", &patch_str],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse task update: {e}"))
}

/// Toggle a task's done state.
///
/// # sensitivity_tier: 2
#[tauri::command]
pub async fn toggle_task_done(
    id: String,
    note: Option<String>,
    state: State<'_, AppState>,
) -> Result<Option<types::Task>, String> {
    let mut args: Vec<String> = vec![
        "tasks-toggle".to_string(),
        "--id".to_string(), id,
    ];
    if let Some(n) = note {
        args.push("--note".to_string());
        args.push(n);
    }
    let str_args: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    let output = call_python_cli(&str_args, &state.project_root).await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse toggle: {e}"))
}

/// Delete a task.
///
/// # sensitivity_tier: 2
#[tauri::command]
pub async fn delete_task(
    id: String,
    state: State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let output = call_python_cli(
        &["tasks-delete", "--id", &id],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse delete: {e}"))
}

/// List habits.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn list_habits(
    status: Option<String>,
    goal_id: Option<String>,
    state: State<'_, AppState>,
) -> Result<Vec<types::Habit>, String> {
    let mut args: Vec<String> = vec!["habits-list".to_string()];
    if let Some(s) = status {
        args.push("--status".to_string());
        args.push(s);
    }
    if let Some(g) = goal_id {
        args.push("--goal-id".to_string());
        args.push(g);
    }
    let str_args: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    let output = call_python_cli(&str_args, &state.project_root).await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse habits: {e}"))
}

/// Create a habit (must be anchored to a goal).
///
/// # sensitivity_tier: 1
#[derive(serde::Deserialize)]
pub struct HabitCreatePayload {
    pub title: String,
    pub goal_id: String,
    #[serde(default)]
    pub cadence: Option<String>,
    #[serde(default)]
    pub days_of_week: Vec<String>,
    #[serde(default)]
    pub preferred_window: Option<String>,
    #[serde(default)]
    pub why: String,
}

#[tauri::command]
pub async fn create_habit(
    payload: HabitCreatePayload,
    state: State<'_, AppState>,
) -> Result<types::Habit, String> {
    let cadence = payload.cadence.unwrap_or_else(|| "daily".to_string());
    let window = payload.preferred_window.unwrap_or_else(|| "any".to_string());
    let days_json = serde_json::to_string(&payload.days_of_week)
        .map_err(|e| format!("days_of_week serialise failed: {e}"))?;
    let args: Vec<String> = vec![
        "habits-create".to_string(),
        "--title".to_string(), payload.title,
        "--goal-id".to_string(), payload.goal_id,
        "--cadence".to_string(), cadence,
        "--days-of-week".to_string(), days_json,
        "--preferred-window".to_string(), window,
        "--why".to_string(), payload.why,
    ];
    let str_args: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    let output = call_python_cli(&str_args, &state.project_root).await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse habit: {e}"))
}

/// Toggle a habit between active and paused.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn toggle_habit(
    id: String,
    state: State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let output = call_python_cli(
        &["habits-toggle", "--id", &id],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse toggle: {e}"))
}

/// Delete a habit.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn delete_habit(
    id: String,
    state: State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let output = call_python_cli(
        &["habits-delete", "--id", &id],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse delete: {e}"))
}

/// Regenerate brain-sourced habits from the current goal set.
///
/// # sensitivity_tier: 2
#[tauri::command]
pub async fn regenerate_habits(
    state: State<'_, AppState>,
) -> Result<Vec<types::Habit>, String> {
    let output = call_python_cli(
        &["habits-regenerate"],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse habits: {e}"))
}

/// Get the persisted schedule for a date (defaults to today).
///
/// # sensitivity_tier: 2
#[tauri::command]
pub async fn get_daily_schedule(
    schedule_date: Option<String>,
    state: State<'_, AppState>,
) -> Result<Option<types::DailySchedule>, String> {
    let mut args: Vec<String> = vec!["schedule-get".to_string()];
    if let Some(d) = schedule_date {
        args.push("--date".to_string());
        args.push(d);
    }
    let str_args: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    let output = call_python_cli(&str_args, &state.project_root).await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse schedule: {e}"))
}

/// Regenerate the daily schedule via the daily_scheduler agent.
///
/// # sensitivity_tier: 2
#[tauri::command]
pub async fn regenerate_daily_schedule(
    schedule_date: Option<String>,
    state: State<'_, AppState>,
) -> Result<Option<types::DailySchedule>, String> {
    let mut args: Vec<String> = vec!["schedule-regenerate".to_string()];
    if let Some(d) = schedule_date {
        args.push("--date".to_string());
        args.push(d);
    }
    let str_args: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    let output = call_python_cli(&str_args, &state.project_root).await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse schedule: {e}"))
}

// ---------------------------------------------------------------------------
// Notification commands
// ---------------------------------------------------------------------------

/// Get notification preferences.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn get_notification_preferences(
    state: State<'_, AppState>,
) -> Result<Vec<types::NotificationPreference>, String> {
    let output = call_python_cli(
        &["notification-prefs-get"],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output).map_err(|e| {
        format!("Failed to parse notification preferences: {e}")
    })
}

/// Update a notification preference.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn update_notification_preference(
    category: String,
    enabled: bool,
    state: State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let enabled_str = if enabled { "true" } else { "false" };
    let output = call_python_cli(
        &[
            "notification-prefs-set",
            "--category",
            &category,
            "--enabled",
            enabled_str,
        ],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output).map_err(|e| {
        format!("Failed to parse notification pref update: {e}")
    })
}

/// Mute all notifications.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn mute_all_notifications(
    until: Option<String>,
    state: State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let mut args = vec!["notification-prefs-mute-all"];
    if let Some(ref ts) = until {
        args.extend_from_slice(&["--until", ts]);
    }
    let output =
        call_python_cli(&args, &state.project_root).await?;
    serde_json::from_str(&output).map_err(|e| {
        format!("Failed to parse mute response: {e}")
    })
}

/// Get notification log.
///
/// # sensitivity_tier: 2
#[tauri::command]
pub async fn get_notification_log(
    limit: Option<i32>,
    offset: Option<i32>,
    state: State<'_, AppState>,
) -> Result<Vec<types::NotificationRecord>, String> {
    let limit_str = limit.unwrap_or(20).to_string();
    let offset_str = offset.unwrap_or(0).to_string();
    let output = call_python_cli(
        &[
            "notification-log",
            "--limit",
            &limit_str,
            "--offset",
            &offset_str,
        ],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output).map_err(|e| {
        format!("Failed to parse notification log: {e}")
    })
}

/// Infer user profile from available data (contacts, WhatsApp phone, etc.).
///
/// # sensitivity_tier: 2
#[tauri::command]
pub async fn infer_user_profile(
    state: State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let output = call_python_cli(
        &["infer-profile"],
        &state.project_root,
    )
    .await?;
    let result: serde_json::Value = serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse infer-profile JSON: {e}"))?;

    // If fields were applied, reload settings from disk so Rust state is in sync.
    let applied = result.get("applied").and_then(|v| v.as_object());
    if applied.is_some_and(|m| !m.is_empty()) {
        let fresh = tokio::task::spawn_blocking(load_settings_from_disk)
            .await
            .map_err(|e| format!("Task join error: {e}"))?
            .map_err(|e| format!("Settings reload error: {e}"))?;
        let mut current = state.settings.lock().await;
        *current = fresh;
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Learned facts commands
// ---------------------------------------------------------------------------

/// Get active learned facts.
///
/// # sensitivity_tier: 2
#[tauri::command]
pub async fn get_learned_facts(
    state: State<'_, AppState>,
) -> Result<Vec<types::LearnedFact>, String> {
    let output = call_python_cli(
        &["get-learned-facts"],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse learned facts: {e}"))
}

/// Get facts pending user review.
///
/// # sensitivity_tier: 2
#[tauri::command]
pub async fn get_facts_for_review(
    state: State<'_, AppState>,
) -> Result<Vec<types::LearnedFact>, String> {
    let output = call_python_cli(
        &["get-facts-for-review"],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse facts for review: {e}"))
}

/// Get fact statistics.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn get_fact_stats(
    state: State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let output = call_python_cli(
        &["get-fact-stats"],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse fact stats: {e}"))
}

/// Confirm a learned fact.
///
/// # sensitivity_tier: 2
#[tauri::command]
pub async fn confirm_fact(
    state: State<'_, AppState>,
    fact_id: String,
) -> Result<serde_json::Value, String> {
    let output = call_python_cli(
        &["confirm-fact", "--fact-id", &fact_id],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse confirm response: {e}"))
}

/// Dismiss a learned fact.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn dismiss_fact(
    state: State<'_, AppState>,
    fact_id: String,
) -> Result<serde_json::Value, String> {
    let output = call_python_cli(
        &["dismiss-fact", "--fact-id", &fact_id],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse dismiss response: {e}"))
}

/// Edit a learned fact's content.
///
/// # sensitivity_tier: 2
#[tauri::command]
pub async fn edit_fact(
    state: State<'_, AppState>,
    fact_id: String,
    new_content: String,
) -> Result<serde_json::Value, String> {
    let output = call_python_cli(
        &[
            "edit-fact",
            "--fact-id", &fact_id,
            "--content", &new_content,
        ],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse edit response: {e}"))
}

// ---------------------------------------------------------------------------
// Voice transcription commands
// ---------------------------------------------------------------------------

/// Transcribe audio data to text using local Whisper model.
///
/// Accepts base64-encoded audio data from the frontend (recorded via
/// MediaRecorder). Returns transcribed text, detected language, and
/// duration. An optional ISO-639-1 ``language_hint`` skips auto-detection
/// — useful for bilingual users whose preferred language is in Settings.
///
/// # sensitivity_tier: 3
#[tauri::command]
pub async fn transcribe_audio(
    state: State<'_, AppState>,
    audio_base64: String,
    language_hint: Option<String>,
) -> Result<types::TranscriptionResult, String> {
    let mut args: Vec<&str> = vec!["transcribe-audio", &audio_base64];
    if let Some(code) = language_hint.as_deref() {
        if !code.is_empty() {
            args.push("--language");
            args.push(code);
        }
    }
    let output = call_python_cli(&args, &state.project_root).await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse transcription result: {e}"))
}

// ---------------------------------------------------------------------------
// Background task status
// ---------------------------------------------------------------------------

/// List all currently-running background tasks.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn get_active_tasks(
    state: State<'_, AppState>,
) -> Result<Vec<BackgroundTask>, String> {
    let tasks = state.active_tasks.lock().await;
    Ok(tasks.values().cloned().collect())
}

// ---------------------------------------------------------------------------
// Mission Control dashboard
// ---------------------------------------------------------------------------

/// Return today's synthesized brief (server-cached, optional force).
///
/// The Python CLI caches the LLM output keyed by (date, last pipeline
/// `completed_at`). `force = true` bypasses the cache and regenerates.
///
/// # sensitivity_tier: 3
#[tauri::command]
pub async fn get_daily_brief(
    force: Option<bool>,
    state: State<'_, AppState>,
) -> Result<DailyBrief, String> {
    let mut args: Vec<&str> = vec!["get-daily-brief"];
    if force.unwrap_or(false) {
        args.push("--force");
    }
    let output = call_python_cli(&args, &state.project_root).await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse daily brief JSON: {e}"))
}

/// Return cross-source threads of attention for the dashboard.
///
/// # sensitivity_tier: 3
#[tauri::command]
pub async fn get_active_threads(
    limit: Option<u32>,
    state: State<'_, AppState>,
) -> Result<Vec<Thread>, String> {
    let limit_value = limit.unwrap_or(10).to_string();
    let output = call_python_cli(
        &["get-active-threads", "--limit", &limit_value],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse active threads JSON: {e}"))
}

/// Return live agent activity for the Mission Control panel.
///
/// Merges:
///   - `running`: the Rust-side `active_tasks` map (live, no subprocess).
///   - `awaiting_review` + `recently_completed`: from the Python CLI.
///
/// # sensitivity_tier: 2
#[tauri::command]
pub async fn get_agent_stream(
    state: State<'_, AppState>,
) -> Result<AgentStream, String> {
    let output = call_python_cli(
        &["get-agent-stream"],
        &state.project_root,
    )
    .await?;
    let mut stream: AgentStream = serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse agent stream JSON: {e}"))?;

    let tasks = state.active_tasks.lock().await;
    stream.running = tasks
        .values()
        .cloned()
        .map(|task| AgentRunning {
            task_id: task.id,
            agent_name: task.label.clone(),
            label: task.label,
            progress: None,
            started_at: task.started_at,
        })
        .collect();

    Ok(stream)
}

/// Return Command Bar suggestion chips derived from current state.
///
/// Deterministic template generation in Python — no LLM call.
///
/// # sensitivity_tier: 2
#[tauri::command]
pub async fn get_suggested_actions(
    limit: Option<u32>,
    state: State<'_, AppState>,
) -> Result<SuggestedActions, String> {
    let limit_value = limit.unwrap_or(3).to_string();
    let output = call_python_cli(
        &["get-suggested-actions", "--limit", &limit_value],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse suggested actions JSON: {e}"))
}

/// Return today's items + open loops for one life domain.
///
/// `domain` must be one of `work`, `personal`, `health`. Backed by
/// the corresponding `mart_{domain}` view + pending replies whose
/// domain bucket matches.
///
/// # sensitivity_tier: 3
#[tauri::command]
pub async fn get_domain_summary(
    domain: String,
    state: State<'_, AppState>,
) -> Result<DomainSummary, String> {
    let output = call_python_cli(
        &["get-domain-summary", "--domain", &domain],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse domain summary JSON: {e}"))
}

/// Return the unified LifeBoard — goals + today's actions + domain
/// items for each of work / personal / health, in one fetch.
///
/// Replaces the prior need to call `list_goals` + `get_domain_summary`
/// separately and combine on the client. Goals come pre-sorted by
/// `urgency_score`.
///
/// # sensitivity_tier: 3
#[tauri::command]
pub async fn get_life_board(
    state: State<'_, AppState>,
) -> Result<types::LifeBoard, String> {
    let output = call_python_cli(
        &["get-life-board"],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse life board JSON: {e}"))
}

/// Return the dashboard's prioritized "Today" board.
///
/// Slices the persisted daily schedule into a Now / Up Next / Loops
/// shape and stitches in the highest-importance pending replies so the
/// dashboard renders one canonical "what now" surface.
///
/// # sensitivity_tier: 3
#[tauri::command]
pub async fn get_today_board(
    state: State<'_, AppState>,
) -> Result<types::TodayBoard, String> {
    let output = call_python_cli(
        &["today-board"],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse today board JSON: {e}"))
}

/// Return progress + today's moves for a single goal.
///
/// # sensitivity_tier: 3
#[tauri::command]
pub async fn get_goal_progress(
    goal_id: String,
    state: State<'_, AppState>,
) -> Result<types::GoalProgress, String> {
    let output = call_python_cli(
        &["goal-progress", "--id", &goal_id],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse goal progress JSON: {e}"))
}

/// Return the canonical inbox of pending replies, optionally scoped to
/// a single domain (work / personal / health).
///
/// # sensitivity_tier: 3
#[tauri::command]
pub async fn list_inbox(
    domain: Option<String>,
    topic: Option<String>,
    state: State<'_, AppState>,
) -> Result<types::Inbox, String> {
    let mut args: Vec<String> = vec!["list-inbox".to_string()];
    if let Some(d) = domain {
        args.push("--domain".to_string());
        args.push(d);
    }
    if let Some(t) = topic {
        args.push("--topic".to_string());
        args.push(t);
    }
    let str_args: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    let output = call_python_cli(&str_args, &state.project_root).await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse inbox JSON: {e}"))
}

// ---------------------------------------------------------------------------
// Agents page (Phase 4) — Pydantic AI agent registry handlers
// ---------------------------------------------------------------------------

/// List every registered Pydantic AI agent with its resolved config.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn list_pydantic_agents(
    state: State<'_, AppState>,
) -> Result<PydanticAgentListResponse, String> {
    let output = call_python_cli(&["agents-list"], &state.project_root).await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse agents-list JSON: {e}"))
}

/// Fetch one agent's definition + resolved config by id.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn get_agent_config(
    agent_id: String,
    state: State<'_, AppState>,
) -> Result<PydanticAgentResponse, String> {
    let output = call_python_cli(
        &["agents-get", "--agent-id", &agent_id],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse agents-get JSON: {e}"))
}

/// Apply a config patch to one editable agent.
///
/// Rejected by the Python layer when the agent is non-editable (brain
/// or firewall).
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn update_agent_config(
    agent_id: String,
    patch: PydanticAgentPatch,
    state: State<'_, AppState>,
) -> Result<PydanticAgentResponse, String> {
    let patch_json = serde_json::to_string(&patch)
        .map_err(|e| format!("Failed to serialise patch: {e}"))?;
    let output = call_python_cli(
        &["agents-update", "--agent-id", &agent_id, "--patch", &patch_json],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse agents-update JSON: {e}"))
}

/// List the model ids exposed by an agent route's endpoint.
///
/// Backs the model-override picker on the Agents page. Returns a
/// chat-first-sorted list of model ids the configured endpoint
/// exposes. Errors return a result with an empty `models` list and
/// an `error` string so the UI can fall back to a free-text input.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn list_available_models(
    route: String,
    state: State<'_, AppState>,
) -> Result<AvailableModels, String> {
    let output = call_python_cli(
        &["agents-list-models", "--route", &route],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse agents-list-models JSON: {e}"))
}

/// Drop the user override row for one agent and return the default config.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn reset_agent_config(
    agent_id: String,
    state: State<'_, AppState>,
) -> Result<PydanticAgentResponse, String> {
    let output = call_python_cli(
        &["agents-reset", "--agent-id", &agent_id],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse agents-reset JSON: {e}"))
}

/// Run one agent's eval suite synchronously and return the resulting row.
///
/// Used by the "Run eval" button on Brain + firewall cards, and from
/// the editor for editable agents that want an immediate run instead
/// of the background auto-trigger.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn run_agent_eval(
    agent_id: String,
    state: State<'_, AppState>,
) -> Result<AgentEvalRunResponse, String> {
    let output = call_python_cli(
        &["agents-run-eval", "--agent-id", &agent_id, "--trigger", "manual"],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse agents-run-eval JSON: {e}"))
}

/// Evaluate a candidate model override without persisting the change.
///
/// Backs the gate on the Agents page model picker: the frontend calls
/// this with the proposed override, inspects the returned `run.status`,
/// and only calls `update_agent_config` when status == "passed".
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn run_agent_eval_proposal(
    agent_id: String,
    proposed_override: String,
    state: State<'_, AppState>,
) -> Result<AgentEvalProposalResponse, String> {
    let _guard = state.cli_write_lock.lock().await;
    let output = call_python_cli(
        &[
            "agents-run-eval-proposal",
            "--agent-id",
            &agent_id,
            "--override",
            &proposed_override,
        ],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse agents-run-eval-proposal JSON: {e}"))
}

/// Flip the privacy mode (`local_inference_for_sensitive`).
///
/// Backs the toggle in the Settings page. When `enabled=true` this
/// runs every registered agent's eval suite against the user's
/// configured local model and only commits the flag (and reloads the
/// egress firewall) when every run came back `"passed"`. On any
/// failure the response carries `status="eval_failed"` plus a
/// per-agent `failures` list so the UI can show the user which agent
/// blocked the toggle.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn set_local_inference_for_sensitive(
    enabled: bool,
    state: State<'_, AppState>,
) -> Result<LocalInferenceToggleResponse, String> {
    let _guard = state.cli_write_lock.lock().await;
    let flag = if enabled { "true" } else { "false" };
    let output = call_python_cli(
        &["set-local-inference-for-sensitive", "--enabled", flag],
        &state.project_root,
    )
    .await?;
    let parsed: LocalInferenceToggleResponse = serde_json::from_str(&output)
        .map_err(|e| format!(
            "Failed to parse set-local-inference-for-sensitive JSON: {e}"
        ))?;
    // Mirror the committed flag into the cached AppSettings so the
    // UI doesn't have to round-trip through Python on every read.
    if parsed.status == "ok" {
        let mut settings = state.settings.lock().await;
        settings.local_inference_for_sensitive = parsed.enabled;
    }
    Ok(parsed)
}

/// Fetch the latest eval row (+ optional history) for one agent.
///
/// The Agents page polls this after an edit so the auto-triggered
/// eval result lands on the card.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn get_agent_eval_status(
    agent_id: String,
    limit: Option<i32>,
    state: State<'_, AppState>,
) -> Result<AgentEvalStatusResponse, String> {
    let limit = limit.unwrap_or(1);
    let limit_str = limit.to_string();
    let output = call_python_cli(
        &["agents-eval-status", "--agent-id", &agent_id, "--limit", &limit_str],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse agents-eval-status JSON: {e}"))
}

/// Fetch the most-recent input/output entries for one agent.
///
/// Backs the "Recent runs" panel on the Agents page. `limit` defaults
/// to 100 and is clamped server-side to the per-agent ring size
/// (currently 1000). Newest entries come first.
///
/// # sensitivity_tier: varies
#[tauri::command]
pub async fn get_agent_activity(
    agent_id: String,
    limit: Option<i32>,
    state: State<'_, AppState>,
) -> Result<AgentActivityResponse, String> {
    let limit = limit.unwrap_or(100);
    let limit_str = limit.to_string();
    let output = call_python_cli(
        &["agents-activity", "--agent-id", &agent_id, "--limit", &limit_str],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse agents-activity JSON: {e}"))
}

/// Return the eval dataset YAML for one agent (built-in or user).
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn get_agent_eval_dataset(
    agent_id: String,
    state: State<'_, AppState>,
) -> Result<AgentEvalDataset, String> {
    let output = call_python_cli(
        &["agents-eval-dataset", "--agent-id", &agent_id],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse agents-eval-dataset JSON: {e}"))
}

/// Validate a user-uploaded eval dataset YAML and persist on success.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn upload_user_eval_dataset(
    agent_id: String,
    content: String,
    state: State<'_, AppState>,
) -> Result<DatasetValidationResponse, String> {
    let output = call_python_cli(
        &[
            "agents-validate-dataset",
            "--agent-id",
            &agent_id,
            "--content",
            &content,
        ],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse agents-validate-dataset JSON: {e}"))
}

/// Propose a starter eval dataset for a user agent.
///
/// Pass exactly one of `agent_id` (resolves a saved user agent's row)
/// or `unsaved_spec` (the in-flight create-modal fields). The returned
/// `DatasetSuggestion` carries either a YAML proposal or a refusal +
/// `improvement_hints` the user can apply to clarify the agent.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn suggest_eval_dataset(
    agent_id: Option<String>,
    unsaved_spec: Option<UnsavedAgentSpec>,
    state: State<'_, AppState>,
) -> Result<DatasetSuggestionResponse, String> {
    let mut args: Vec<String> = vec!["agents-suggest-dataset".into()];
    let spec_json: String;
    match (&agent_id, &unsaved_spec) {
        (Some(id), None) => {
            args.push("--agent-id".into());
            args.push(id.clone());
        }
        (None, Some(spec)) => {
            spec_json = serde_json::to_string(spec).map_err(|e| {
                format!("Failed to serialise unsaved_spec: {e}")
            })?;
            args.push("--unsaved-spec".into());
            args.push(spec_json);
        }
        _ => {
            return Err(
                "exactly one of agent_id or unsaved_spec must be provided"
                    .into(),
            );
        }
    }
    let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
    let output = call_python_cli(&arg_refs, &state.project_root).await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse agents-suggest-dataset JSON: {e}"))
}

/// Recommend best-overall + cost-effective models for an agent spec.
///
/// The wizard / edit row submits the in-flight form values; the Python
/// CLI fetches the live `/models` lists itself so the LLM only sees
/// real ids. The returned `ModelRecommendation` either carries two
/// model options or refuses with `improvement_hints`.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn suggest_agent_model(
    spec: ModelPickerSpec,
    state: State<'_, AppState>,
) -> Result<ModelRecommendationResponse, String> {
    let spec_json = serde_json::to_string(&spec)
        .map_err(|e| format!("Failed to serialise spec: {e}"))?;
    let output = call_python_cli(
        &["agents-suggest-model", "--unsaved-spec", &spec_json],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse agents-suggest-model JSON: {e}"))
}

/// Rewrite a user agent's system prompt + description for clarity,
/// expected output, language pinning, format strictness, scope, and
/// safety. Returns a `PromptSuggestion` with both a full rewrite and
/// a surgical-additions list — the UI lets the user pick either path.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn suggest_prompt_improvements(
    spec: PromptEngineerSpec,
    state: State<'_, AppState>,
) -> Result<PromptSuggestionResponse, String> {
    let spec_json = serde_json::to_string(&spec)
        .map_err(|e| format!("Failed to serialise spec: {e}"))?;
    let output = call_python_cli(
        &[
            "agents-suggest-prompt-improvements",
            "--unsaved-spec",
            &spec_json,
        ],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output).map_err(|e| {
        format!("Failed to parse agents-suggest-prompt-improvements JSON: {e}")
    })
}

/// Apply a prompt-engineer rewrite to a user-authored agent. The
/// Python handler snapshots the current `system_prompt` +
/// `description` into the `pre_ai_*` columns and mirrors the new
/// prompt into the config overlay so the editor reads it.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn apply_prompt_engineer_edit(
    agent_id: String,
    system_prompt: String,
    description: String,
    state: State<'_, AppState>,
) -> Result<UserAgentResponse, String> {
    let payload = serde_json::to_string(&serde_json::json!({
        "system_prompt": system_prompt,
        "description": description,
    }))
    .map_err(|e| format!("Failed to serialise payload: {e}"))?;
    let output = call_python_cli(
        &[
            "agents-user-apply-prompt-edit",
            "--agent-id",
            &agent_id,
            "--payload",
            &payload,
        ],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output).map_err(|e| {
        format!("Failed to parse agents-user-apply-prompt-edit JSON: {e}")
    })
}

/// Restore a user-authored agent's `system_prompt` + `description`
/// from the pre-AI-edit snapshot. Refuses for built-in agents.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn revert_user_agent_ai_edit(
    agent_id: String,
    state: State<'_, AppState>,
) -> Result<UserAgentResponse, String> {
    let output = call_python_cli(
        &["agents-user-revert-ai-edit", "--agent-id", &agent_id],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output).map_err(|e| {
        format!("Failed to parse agents-user-revert-ai-edit JSON: {e}")
    })
}

/// Create a new user-authored agent and mount it in the registry.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn create_user_agent(
    input: UserAgentInput,
    state: State<'_, AppState>,
) -> Result<UserAgentResponse, String> {
    let payload = serde_json::to_string(&input)
        .map_err(|e| format!("Failed to serialise input: {e}"))?;
    let output = call_python_cli(
        &["agents-create", "--payload", &payload],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse agents-create JSON: {e}"))
}

/// Update an existing user-authored agent.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn update_user_agent(
    agent_id: String,
    input: UserAgentInput,
    state: State<'_, AppState>,
) -> Result<UserAgentResponse, String> {
    let payload = serde_json::to_string(&input)
        .map_err(|e| format!("Failed to serialise input: {e}"))?;
    let output = call_python_cli(
        &[
            "agents-user-update",
            "--agent-id",
            &agent_id,
            "--payload",
            &payload,
        ],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse agents-user-update JSON: {e}"))
}

/// Remove a user-authored agent from SQLite and the registry.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn delete_user_agent(
    agent_id: String,
    state: State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let output = call_python_cli(
        &["agents-delete", "--agent-id", &agent_id],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse agents-delete JSON: {e}"))
}

/// Set the schedule cron + enabled flag for a user-authored agent.
///
/// Source / callable / delivery tool bindings live in
/// `enabled_mcp_tools` + `delivery_tools` on the row itself; this
/// handler is now a pure cron+enabled write.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn set_user_agent_schedule(
    agent_id: String,
    cron: Option<String>,
    enabled: bool,
    state: State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let enabled_arg = if enabled { "true" } else { "false" };
    let cron_value = cron.unwrap_or_default();
    let args: Vec<&str> = vec![
        "agents-set-schedule",
        "--agent-id",
        &agent_id,
        "--cron",
        &cron_value,
        "--enabled",
        enabled_arg,
    ];
    let output = call_python_cli(&args, &state.project_root).await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse agents-set-schedule JSON: {e}"))
}

/// Run a user-authored agent immediately (manual "Run now" button).
///
/// Agents whose `enabled_mcp_tools` includes at least one catalog
/// `data` tool process all unprocessed items from their bound
/// sources (batch path); others fire once with the generic
/// Portuguese trigger.
///
/// # sensitivity_tier: 2
#[tauri::command]
pub async fn run_user_agent_now(
    agent_id: String,
    state: State<'_, AppState>,
) -> Result<BatchRunSummary, String> {
    let output = call_python_cli(
        &["agents-run-now", "--agent-id", &agent_id],
        &state.project_root,
    )
    .await?;
    let envelope: serde_json::Value = serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse agents-run-now JSON: {e}"))?;
    if let Some(err) = envelope.get("error").and_then(|v| v.as_str()) {
        return Err(err.to_string());
    }
    let summary = envelope.get("summary").ok_or_else(|| {
        "agents-run-now response missing 'summary' field".to_string()
    })?;
    serde_json::from_value(summary.clone()).map_err(|e| {
        format!("Failed to deserialise BatchRunSummary: {e}")
    })
}

/// Read scheduling + last/next/pending status for one user agent.
///
/// Powers the schedule strip on the Agents page card.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn get_user_agent_status(
    agent_id: String,
    state: State<'_, AppState>,
) -> Result<UserAgentStatus, String> {
    let output = call_python_cli(
        &["agents-user-status", "--agent-id", &agent_id],
        &state.project_root,
    )
    .await?;
    let envelope: serde_json::Value = serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse agents-user-status JSON: {e}"))?;
    if let Some(err) = envelope.get("error").and_then(|v| v.as_str()) {
        return Err(err.to_string());
    }
    let status = envelope.get("status").ok_or_else(|| {
        "agents-user-status response missing 'status' field".to_string()
    })?;
    serde_json::from_value(status.clone()).map_err(|e| {
        format!("Failed to deserialise UserAgentStatus: {e}")
    })
}

/// List MCP action tools from enabled connectors.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn list_mcp_action_tools(
    state: State<'_, AppState>,
) -> Result<McpActionToolListResponse, String> {
    let output = call_python_cli(
        &["agents-list-mcp-tools"],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse agents-list-mcp-tools JSON: {e}"))
}


// ---------------------------------------------------------------------------
// Skills v2 — SKILL.md-based commands
// ---------------------------------------------------------------------------

/// List all SKILL.md-based skills (L1 metadata).
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn list_skills_v2(state: State<'_, AppState>) -> Result<Vec<serde_json::Value>, String> {
    let output = call_python_cli(&["skills-list-v2"], &state.project_root).await?;
    serde_json::from_str(&output).map_err(|e| format!("Failed to parse skills-list-v2 JSON: {e}"))
}

/// Return one SKILL.md skill's full content (L2).
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn get_skill_detail_v2(
    skill_id: String,
    state: State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let output = call_python_cli(
        &["skills-get-v2", "--skill-id", &skill_id],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse skills-get-v2 JSON: {e}"))
}

/// Create a new SKILL.md-based skill.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn create_skill_v2(
    name: String,
    content: String,
    state: State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let output = call_python_cli(
        &["skills-create-v2", "--name", &name, "--content", &content],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse skills-create-v2 JSON: {e}"))
}

/// Update a SKILL.md-based skill's content.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn update_skill_v2(
    skill_id: String,
    content: String,
    state: State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let output = call_python_cli(
        &["skills-update-v2", "--skill-id", &skill_id, "--content", &content],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse skills-update-v2 JSON: {e}"))
}

/// Delete a SKILL.md-based skill.
///
/// # sensitivity_tier: 1
#[tauri::command]
pub async fn delete_skill_v2(
    skill_id: String,
    state: State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let output = call_python_cli(
        &["skills-delete-v2", "--skill-id", &skill_id],
        &state.project_root,
    )
    .await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse skills-delete-v2 JSON: {e}"))
}

/// Export all local user data to an inspectable JSON zip archive.
///
/// Returns `{ ok, path, tables, total_rows, bytes }`; the frontend
/// surfaces `path` so the user can find and inspect the archive.
///
/// # sensitivity_tier: 3
#[tauri::command]
pub async fn export_user_data(
    state: State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let output = call_python_cli(&["export-data"], &state.project_root).await?;
    serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse export-data JSON: {e}"))
}

/// Permanently delete all local user data — the SQLite/Kuzu/Chroma
/// stores, learned facts, goals & tasks, the audit chain, sessions,
/// and derived caches. App preferences are preserved.
///
/// Order matters for a *complete* wipe:
/// 1. Kill live background writers first (pipeline, sync, insights,
///    scheduled agents, proactive eval, WhatsApp listener) so none of
///    them recreates DB/WAL files inside the freshly emptied directory
///    with data they were holding in memory.
/// 2. Wipe the data directory (the Python CLI).
/// 3. Re-run `init` so Kuzu's schema exists again — a bare webview
///    reload would otherwise leave the app stuck on "DB issue" (Kuzu
///    DDL only runs from `init`/`reset`, not on first write). The
///    frontend reloads after this returns.
///
/// # sensitivity_tier: 3
#[tauri::command]
pub async fn delete_all_user_data(
    state: State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    // 1. Stop live writers. Mirrors the startup cleanup in lib.rs; the
    //    WhatsApp listener is included here (unlike at startup) because
    //    a full wipe removes its session, so it must not keep the DB
    //    open or write past the deletion.
    let writer_patterns = [
        "src\\.core\\.cli startup-sync",
        "src\\.core\\.cli sync-all-stale",
        "src\\.core\\.cli sync-connector",
        "src\\.core\\.cli generate-insights",
        "src\\.core\\.cli run-scheduled-agents",
        "src\\.core\\.cli evaluate-proactive",
        "src\\.core\\.cli evaluate-messages",
        "src\\.core\\.cli whatsapp-listener-run",
        "src\\.pipeline\\.worker run",
    ];
    for pattern in writer_patterns {
        let _ = tokio::process::Command::new("pkill")
            .args(["-f", pattern])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .await;
    }

    // 2. Wipe.
    let output =
        call_python_cli(&["delete-all-data"], &state.project_root).await?;
    let value: serde_json::Value = serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse delete-all-data JSON: {e}"))?;
    if !value.get("ok").and_then(|v| v.as_bool()).unwrap_or(false) {
        let err = value
            .get("error")
            .and_then(|v| v.as_str())
            .unwrap_or("delete failed")
            .to_string();
        return Err(err);
    }

    // 3. Self-heal the (now-empty) DB schema. Best-effort: the wipe
    //    already succeeded, and the startup hook re-runs `init` on the
    //    next launch regardless, so a failure here is non-fatal.
    if let Err(e) = call_python_cli(&["init"], &state.project_root).await {
        eprintln!("[delete_all_user_data] post-wipe init failed: {e}");
    }

    Ok(value)
}

/// Stream a question to a specific agent (Brain or a user-authored agent).
///
/// Mirrors :command:`ask_brain_stream` but accepts an ``agentId``. The
/// emitted event name is the same (``brain-stream``) so the existing
/// frontend listener handles either case.
///
/// # sensitivity_tier: 3
#[tauri::command]
pub async fn ask_agent_stream(
    question: String,
    agent_id: String,
    reply_context: Option<types::ReplyContext>,
    app_handle: AppHandle,
    state: State<'_, AppState>,
) -> Result<(), String> {
    let session_id = ensure_active_session(&state).await?;
    let reply_ctx_json = match &reply_context {
        Some(ctx) => Some(
            serde_json::to_string(ctx)
                .map_err(|e| format!("Failed to serialize reply_context: {e}"))?,
        ),
        None => None,
    };
    let mut args: Vec<&str> = vec![
        "ask-stream",
        &question,
        "--agent-id",
        &agent_id,
        "--session-id",
        &session_id,
    ];
    if let Some(json) = reply_ctx_json.as_deref() {
        args.push("--reply-context");
        args.push(json);
    }
    bridge::call_python_cli_stream(
        &args,
        &state.project_root,
        &app_handle,
        "brain-stream",
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_appsettings_parses_from_empty_object() {
        // Every field now has a serde default, so a subset (or empty)
        // object must deserialize — the basis for lenient recovery.
        let s: AppSettings = serde_json::from_str("{}").expect("empty parses");
        assert_eq!(s.llm_model, "llama3.1:70b");
        assert_eq!(s.max_sensitivity_tier, 2);
        assert!(!s.onboarding_completed);
    }

    #[test]
    fn test_recover_keeps_healthy_fields_and_drops_bad_one() {
        // A single malformed field (max_sensitivity_tier as a string)
        // must NOT reset siblings: onboarding_completed and the provider
        // choice survive; only the bad field falls back to its default.
        let contents = r#"{
            "llm_model": "arandu-pro",
            "llm_provider": "arandu_cloud",
            "onboarding_completed": true,
            "max_sensitivity_tier": "not-a-number",
            "user_name": "Vinicius"
        }"#;
        let s = recover_settings_leniently(contents).expect("recovers");
        assert_eq!(s.llm_model, "arandu-pro");
        assert_eq!(s.llm_provider, "arandu_cloud");
        assert!(s.onboarding_completed);
        assert_eq!(s.user_name.as_deref(), Some("Vinicius"));
        // The malformed field fell back to its default, not the whole file.
        assert_eq!(s.max_sensitivity_tier, 2);
    }

    #[test]
    fn test_recover_tolerates_unknown_and_null_fields() {
        // Unknown future/Pro fields and stray nulls must not nuke config.
        let contents = r#"{
            "onboarding_completed": true,
            "some_future_field": {"nested": 1},
            "eval_window_hours": null
        }"#;
        let s = recover_settings_leniently(contents).expect("recovers");
        assert!(s.onboarding_completed);
    }

    #[test]
    fn test_recover_returns_none_for_non_object() {
        assert!(recover_settings_leniently("[1,2,3]").is_none());
        assert!(recover_settings_leniently("not json").is_none());
    }

    #[tokio::test]
    async fn test_app_state_defaults() {
        let state = AppState::new();
        let active = state.active_chat_session.lock().await;
        assert!(active.is_none());

        // Settings may come from disk (~/.arandu/settings.json) or defaults.
        // Exact field values are tested in test_settings_default (pure defaults).
        let settings = state.settings.lock().await;
        assert!(!settings.llm_model.is_empty());
    }

    #[tokio::test]
    async fn test_pipeline_worker_initially_none() {
        let state = AppState::new();
        let worker = state.pipeline_worker.lock().await;
        assert!(worker.is_none());

        let running = state.pipeline_running.lock().await;
        assert!(running.is_none());
    }

    #[tokio::test]
    async fn test_reap_orphan_worker_clears_slot_when_empty() {
        // No worker stored — reap should be a no-op and leave the
        // slot None.
        let state = AppState::new();
        reap_orphan_worker(&state).await;
        let worker = state.pipeline_worker.lock().await;
        assert!(worker.is_none());
    }

    #[tokio::test]
    async fn test_reap_orphan_worker_clears_stored_handle() {
        // Simulate the bug-state: a `PipelineWorkerHandle` lingering
        // in state after its cleanup task died (or after the 1-h
        // `PIPELINE_MAX_AGE` auto-reset cleared only the flag).
        // `reap_orphan_worker` must clear the slot.  We pass
        // `pid: None` so the test doesn't actually shell out to
        // `kill(1)` — the slot-clearing path is what we're verifying.
        let state = AppState::new();
        {
            let mut worker = state.pipeline_worker.lock().await;
            *worker = Some(PipelineWorkerHandle {
                run_id: "orphan-run".to_string(),
                pid: None,
            });
        }

        reap_orphan_worker(&state).await;

        let worker = state.pipeline_worker.lock().await;
        assert!(
            worker.is_none(),
            "orphan handle should have been cleared",
        );
    }

    #[tokio::test]
    async fn test_is_pipeline_flag_set_resets_and_reaps_after_max_age() {
        // `pipeline_running` set long ago + orphan worker stored.
        // `is_pipeline_flag_set` should reset both.
        let state = AppState::new();
        {
            let mut running = state.pipeline_running.lock().await;
            *running = Some(
                Instant::now()
                    .checked_sub(PIPELINE_MAX_AGE * 2)
                    .expect("subtraction underflows on this host"),
            );
        }
        {
            let mut worker = state.pipeline_worker.lock().await;
            *worker = Some(PipelineWorkerHandle {
                run_id: "stale-run".to_string(),
                pid: None,
            });
        }

        assert!(!is_pipeline_flag_set(&state).await);

        let running = state.pipeline_running.lock().await;
        assert!(running.is_none(), "flag should have been reset");
        let worker = state.pipeline_worker.lock().await;
        assert!(worker.is_none(), "orphan worker should have been reaped");
    }

    #[test]
    fn test_settings_default() {
        let settings = AppSettings::default();
        assert_eq!(settings.theme, "light");
        assert_eq!(settings.data_dir, "~/.arandu/data");
        assert!(!settings.onboarding_completed);
        assert!(settings.onboarding_completed_at.is_none());
        assert!(settings.initial_connectors.is_empty());
        assert!(settings.skipped_connectors.is_empty());
        assert!(!settings.ollama_configured);
        assert!(settings.dismissed_nudges.is_empty());
    }
}
