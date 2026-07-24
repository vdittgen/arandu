use serde::{Deserialize, Serialize};

/// Database statistics from all three embedded engines.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stats {
    pub healthy: bool,
    pub sqlite: serde_json::Value,
    pub kuzu_nodes: serde_json::Value,
    pub chromadb: serde_json::Value,
    pub total_sqlite_rows: i64,
    pub total_kuzu_nodes: i64,
    pub total_chroma_docs: i64,
}

/// Today's summary: events, recent messages, and note count.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TodaySummary {
    pub date: String,
    pub events: Vec<serde_json::Value>,
    pub recent_messages: Vec<serde_json::Value>,
    pub notes_count: i64,
}

// ---------------------------------------------------------------------------
// Generic table browser types
// ---------------------------------------------------------------------------

/// Column metadata from information_schema.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnInfo {
    pub name: String,
    #[serde(rename = "type")]
    pub column_type: String,
}

/// A table entry with metadata for the table catalog.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableInfo {
    pub table_name: String,
    pub row_count: i64,
    pub column_count: i32,
    pub columns: Vec<ColumnInfo>,
}

/// A pipeline model entry from the manifest — used to surface registered
/// models in the Data Models page even when their SQLite table hasn't
/// been materialized yet.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineModel {
    pub name: String,
    pub layer: String,
    pub model_type: String,
    pub depends_on: Vec<String>,
}

/// Sample data from a queried table.
///
/// # sensitivity_tier: 2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableSample {
    pub table_name: String,
    pub total_rows: i64,
    pub columns: Vec<ColumnInfo>,
    pub rows: Vec<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Graph explorer types
// ---------------------------------------------------------------------------

/// A node type with its count from the Kuzu graph.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNodeTypeInfo {
    pub name: String,
    pub count: i64,
}

/// A relationship type with its count from the Kuzu graph.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphRelTypeInfo {
    pub name: String,
    pub count: i64,
}

/// Summary of all node and relationship types in the Kuzu graph.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSummary {
    pub nodes: Vec<GraphNodeTypeInfo>,
    pub relationships: Vec<GraphRelTypeInfo>,
    pub total_nodes: i64,
    pub total_relationships: i64,
}

/// Sample nodes of a given type from the Kuzu graph.
///
/// # sensitivity_tier: 2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNodeSample {
    pub node_type: String,
    pub total: i64,
    pub nodes: Vec<serde_json::Value>,
}

/// Sample relationships of a given type from the Kuzu graph.
///
/// # sensitivity_tier: 2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphRelSample {
    pub rel_type: String,
    pub total: i64,
    pub relationships: Vec<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Vector explorer types
// ---------------------------------------------------------------------------

/// A sample document from a ChromaDB collection.
///
/// # sensitivity_tier: 2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorDocSample {
    pub id: String,
    pub document: String,
    pub metadata: serde_json::Value,
}

/// A ChromaDB collection with count and sample documents.
///
/// # sensitivity_tier: 2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorCollectionInfo {
    pub name: String,
    pub count: i64,
    pub samples: Vec<VectorDocSample>,
}

/// Summary of all ChromaDB collections.
///
/// # sensitivity_tier: 2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorSummary {
    pub collections: Vec<VectorCollectionInfo>,
}

/// A message from the raw_messages table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: String,
    pub source: String,
    pub sender: String,
    pub sender_name: Option<String>,
    pub recipient: Option<String>,
    pub content: String,
    pub timestamp: String,
    pub chat_name: Option<String>,
    pub is_group: Option<bool>,
    pub sensitivity_tier: i32,
}

/// A calendar event from raw_calendar_events.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub id: String,
    pub title: String,
    pub description: Option<String>,
    pub start_time: String,
    pub end_time: String,
    pub location: Option<String>,
    pub attendees: Option<serde_json::Value>,
    pub sensitivity_tier: i32,
}

/// A contact from raw_contacts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contact {
    pub id: String,
    pub name: String,
    pub email: Option<String>,
    pub phone: Option<String>,
    pub relationship: Option<String>,
    pub birthday: Option<String>,
    pub address: Option<String>,
    pub notes: Option<String>,
    pub sensitivity_tier: i32,
}

/// A note from the raw_notes table.
///
/// # sensitivity_tier: 2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Note {
    pub id: String,
    pub title: String,
    pub content: String,
    pub source: String,
    pub created_at: String,
    pub updated_at: String,
    pub tags: Option<serde_json::Value>,
    pub sensitivity_tier: i32,
}

/// An email from the raw_emails table.
///
/// # sensitivity_tier: 2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Email {
    pub id: String,
    pub subject: Option<String>,
    pub from_address: Option<String>,
    pub to_addresses: Option<serde_json::Value>,
    pub date: Option<String>,
    pub body_preview: Option<String>,
    pub is_read: Option<bool>,
    pub folder: Option<String>,
    pub sensitivity_tier: i32,
}

/// Response from the Brain Agent after asking a question.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainResponse {
    pub answer: String,
    pub sources: Vec<serde_json::Value>,
    pub context_summary: String,
    pub model: String,
    pub latency_ms: f64,
    #[serde(default)]
    pub parts: Vec<serde_json::Value>,
}

/// A single chat message in the conversation history.
///
/// `parts` carries typed artifacts (markdown, code, chart specs, HTML,
/// images…) so the renderer registry on the frontend can mount the
/// right component per part. Legacy entries without parts fall back to
/// rendering `content` as markdown. `sources`, `latency_ms`, `model`,
/// and `thinking` are populated for assistant messages loaded from
/// the persisted chat store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    pub timestamp: String,
    #[serde(default)]
    pub parts: Vec<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sources: Option<Vec<serde_json::Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
}

/// Summary of a persisted chat session, used by the sessions panel.
///
/// # sensitivity_tier: 3
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatSessionSummary {
    pub id: String,
    pub title: String,
    pub created_at: String,
    pub updated_at: String,
    pub message_count: i64,
    #[serde(default)]
    pub preview: Option<String>,
}

/// List of chat sessions plus the currently active session pointer.
///
/// # sensitivity_tier: 3
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatSessionListResponse {
    pub sessions: Vec<ChatSessionSummary>,
    #[serde(default)]
    pub active_session_id: Option<String>,
}

/// Full message list returned when the frontend opens a session.
///
/// # sensitivity_tier: 3
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadSessionResponse {
    pub session_id: String,
    pub messages: Vec<ChatMessage>,
}

/// LLM provider and model status.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaStatus {
    #[serde(default = "default_llm_provider")]
    pub provider: String,
    pub server_reachable: bool,
    pub chat_model: String,
    pub chat_model_status: String,
    #[serde(default)]
    pub embed_model: String,
    #[serde(default)]
    pub embed_model_status: String,
    #[serde(default)]
    pub server_version: String,
    /// Set only when the status probe itself could not run (e.g. the Python
    /// CLI failed to spawn or returned unparseable output). Lets the UI tell
    /// a crashed backend apart from a genuinely-unreachable Ollama server.
    /// Absent (`None`) for every status that came from a real CLI response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub probe_error: Option<String>,
}

/// Live progress of an in-flight `ollama pull`, published by the Python
/// pull loop to `~/.arandu/data/ollama_pull_progress.json`. Absent file
/// means no pull is in flight.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPullProgress {
    pub model: String,
    pub status: String,
    pub completed: u64,
    pub total: u64,
    pub percent: f64,
    pub updated_at: String,
}

impl OllamaStatus {
    /// Return a safe offline fallback — used when the CLI ran and reported
    /// the server as unreachable.
    pub fn offline() -> Self {
        Self {
            provider: default_llm_provider(),
            server_reachable: false,
            chat_model: String::new(),
            chat_model_status: "offline".to_string(),
            embed_model: String::new(),
            embed_model_status: "offline".to_string(),
            server_version: String::new(),
            probe_error: None,
        }
    }

    /// Return a backend-error status — used when the status probe could not
    /// run at all (CLI spawn failure, empty stdout, or unparseable output).
    /// Distinct from [`offline`] so the UI can guide the user toward a setup
    /// problem rather than implying their Ollama server is down.
    pub fn backend_error(detail: String) -> Self {
        Self {
            provider: default_llm_provider(),
            server_reachable: false,
            chat_model: String::new(),
            chat_model_status: "backend_error".to_string(),
            embed_model: String::new(),
            embed_model_status: "backend_error".to_string(),
            server_version: String::new(),
            probe_error: Some(detail),
        }
    }
}

/// Memory usage and database file size report.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsage {
    pub rss_mb: f64,
    pub db_sizes_mb: std::collections::HashMap<String, f64>,
    pub total_db_mb: f64,
    pub warning: Option<String>,
}

/// Pipeline status: last run, staleness, pending changes, estimate.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStatus {
    pub last_run: Option<PipelineRunSummary>,
    pub is_stale: bool,
    pub pending_changes: std::collections::HashMap<String, i64>,
    pub estimated_refresh_time: f64,
}

/// Summary of a single pipeline run.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineRunSummary {
    pub run_id: String,
    pub started_at: String,
    pub completed_at: String,
    pub duration_seconds: f64,
    pub status: String,
    pub models_processed: Vec<String>,
    pub rows_processed: std::collections::HashMap<String, i64>,
    pub rows_changed: std::collections::HashMap<String, i64>,
    pub trigger: String,
    pub error: Option<String>,
    #[serde(default)]
    pub plan_summary: Option<String>,
    /// Vector index outcome after marts completed ("success"/"error").
    #[serde(default)]
    pub vector_index_status: Option<String>,
    /// Graph index outcome after marts completed ("success"/"error").
    #[serde(default)]
    pub graph_index_status: Option<String>,
    /// Verbatim re-index error (e.g. embedding dimension mismatch).
    #[serde(default)]
    pub index_error: Option<String>,
}

/// A single model selected for execution with its priority.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannedModel {
    pub name: String,
    pub priority: String,
    pub reason: String,
}

/// A model excluded from execution.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkippedModel {
    pub name: String,
    pub reason: String,
}

/// Prioritized pipeline refresh plan from the Pipeline Brain.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefreshPlan {
    pub models: Vec<PlannedModel>,
    pub skipped: Vec<SkippedModel>,
    pub estimated_duration_seconds: f64,
    pub full_duration_seconds: f64,
    pub summary: String,
}

/// Returned immediately when a background pipeline run is triggered.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineRunStarted {
    pub run_id: String,
    pub status: String,
}

/// Result of polling for a background pipeline run.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineRunResult {
    pub run_id: String,
    pub status: String,
    pub result: Option<PipelineRunSummary>,
}

/// Progress event emitted during streaming pipeline execution.
///
/// Each JSON line from `pipeline-run-stream` is deserialized into this
/// struct and re-emitted as a Tauri event (`pipeline-progress`).
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineProgressEvent {
    #[serde(rename = "type")]
    pub event_type: String,
    pub model_name: Option<String>,
    pub step_index: i32,
    pub total_steps: i32,
    pub status: String,
    pub elapsed_seconds: f64,
    pub rows_processed: Option<i64>,
    pub run_id: Option<String>,
    pub duration_seconds: Option<f64>,
    pub error: Option<String>,
}

/// Application-wide settings persisted to disk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppSettings {
    #[serde(default = "default_llm_model")]
    pub llm_model: String,
    #[serde(default = "default_llm_host")]
    pub llm_host: String,
    #[serde(default = "default_max_sensitivity_tier")]
    pub max_sensitivity_tier: u8,
    #[serde(default = "default_theme")]
    pub theme: String,
    #[serde(default = "default_data_dir")]
    pub data_dir: String,
    #[serde(default = "default_auto_refresh_enabled")]
    pub auto_refresh_enabled: bool,
    #[serde(default = "default_refresh_interval")]
    pub auto_refresh_interval_minutes: u32,
    #[serde(default)]
    pub refresh_on_launch: bool,

    // --- Onboarding ---
    #[serde(default)]
    pub onboarding_completed: bool,
    #[serde(default)]
    pub onboarding_completed_at: Option<String>,
    #[serde(default)]
    pub initial_connectors: Vec<String>,
    #[serde(default)]
    pub skipped_connectors: Vec<String>,
    #[serde(default)]
    pub ollama_configured: bool,
    /// True while the post-onboarding follow-up (enabling each connector
    /// in `initial_connectors`) is still pending. Set by the wizard at
    /// finish-time and cleared by the Dashboard's follow-up hook once
    /// every connector has been toggled. Persisted so the work resumes
    /// if the user closes the app mid-flow.
    #[serde(default)]
    pub onboarding_followup_pending: bool,

    // --- Nudges ---
    #[serde(default)]
    pub dismissed_nudges: Vec<String>,

    // --- Interest profile ---
    /// Manual interest priority overrides (domain -> rank).
    #[serde(default)]
    pub interest_overrides: std::collections::HashMap<String, i32>,

    // --- Notifications ---
    #[serde(default)]
    pub notifications_enabled: bool,
    #[serde(default)]
    pub whatsapp_notification_phone: Option<String>,

    // --- User Profile ---
    /// sensitivity_tier: 2
    #[serde(default)]
    pub user_name: Option<String>,
    #[serde(default)]
    pub user_birthday: Option<String>,
    #[serde(default)]
    pub user_location: Option<String>,
    #[serde(default)]
    pub user_timezone: Option<String>,
    #[serde(default)]
    pub user_language: Option<String>,
    #[serde(default)]
    pub user_bio: Option<String>,

    // --- Voice & Audio ---
    #[serde(default)]
    pub voice_transcription_enabled: bool,
    #[serde(default = "default_whisper_model_size")]
    pub whisper_model_size: String,
    #[serde(default)]
    pub transcribe_whatsapp_audio: bool,

    // --- LLM Provider ---
    #[serde(default = "default_llm_provider")]
    pub llm_provider: String,
    #[serde(default)]
    pub anthropic_api_key: Option<String>,
    #[serde(default = "default_anthropic_model")]
    pub anthropic_model: String,
    #[serde(default)]
    pub external_llm_consent: bool,
    /// API key for an OpenAI-compatible remote server (vLLM, llama.cpp,
    /// TGI, LM Studio).  Many self-hosted servers ignore the key but
    /// expect *something*; ``None`` is converted to ``"EMPTY"`` server-side.
    #[serde(default)]
    pub llm_api_key: Option<String>,
    /// Max concurrent in-flight LLM requests when using the OpenAI-compat
    /// provider.  Has no effect on Ollama or Anthropic providers.
    #[serde(default = "default_llm_max_parallel")]
    pub llm_max_parallel: u32,

    // --- Fresh-restart bounds (sensitivity_tier: 1) ---
    /// ISO-8601 timestamp; ingestion adapters drop records older than this.
    /// Set by `scripts/fresh_restart.sh`; unset/absent disables the filter.
    #[serde(default)]
    pub ingest_cutoff_iso: Option<String>,
    /// Override for the MessageEvaluator recency window (hours).
    #[serde(default)]
    pub eval_window_hours: Option<f64>,
    /// Override for the conversation digest window (hours).
    #[serde(default)]
    pub digest_window_hours: Option<f64>,
    /// Override for the ProactiveIntelligence pre-filter window (hours).
    #[serde(default)]
    pub proactive_window_hours: Option<f64>,
    /// Override for the insight-pattern question-history window (days).
    #[serde(default)]
    pub insight_window_days: Option<u32>,
    /// Delay (minutes) before the proactive + insight loops fire on app
    /// start.  Used after a fresh restart to avoid burning tokens before
    /// the user has approved anything.  ``None``/``0`` keeps the defaults.
    #[serde(default)]
    pub first_run_grace_minutes: Option<u32>,

    // --- Keep Arandu running (sensitivity_tier: 1) ---
    #[serde(default)]
    pub prevent_sleep: bool,
    #[serde(default)]
    pub prevent_sleep_on_battery: bool,
    #[serde(default)]
    pub launch_at_login: bool,
    #[serde(default)]
    pub menu_bar_mode: bool,
    /// Set to `true` after the user has seen the post-onboarding "Keep me
    /// awake" modal once. Frontend reads this on boot to avoid re-showing
    /// the modal on every app launch.
    #[serde(default)]
    pub keep_awake_modal_seen: bool,

    // --- Privacy mode (sensitivity_tier: 1) ---
    /// In Arandu every tier already runs on local Ollama; this
    /// flag exists so the eval gate runs when the user switches
    /// local models. Mutated via the
    /// `set_local_inference_for_sensitive` Tauri command rather than
    /// by writing settings directly, because the flip has to run the
    /// gate before committing.
    #[serde(default)]
    pub local_inference_for_sensitive: bool,
}

fn default_refresh_interval() -> u32 {
    60
}

// The five core fields below used to have no serde default, so a
// settings file missing (or malformed in) ANY one of them failed to
// deserialize as a whole — which, combined with the lenient recovery in
// `load_settings_from_disk`, must never reset the entire config. Giving
// them defaults lets a partial/recovered object always deserialize.
fn default_llm_model() -> String {
    "llama3.1:70b".to_string()
}

fn default_llm_host() -> String {
    "http://localhost:11434".to_string()
}

fn default_max_sensitivity_tier() -> u8 {
    2
}

fn default_theme() -> String {
    "light".to_string()
}

fn default_data_dir() -> String {
    "~/.arandu/data".to_string()
}

fn default_auto_refresh_enabled() -> bool {
    true
}

fn default_whisper_model_size() -> String {
    "base".to_string()
}

fn default_llm_provider() -> String {
    "ollama".to_string()
}

fn default_anthropic_model() -> String {
    "claude-sonnet-4-20250514".to_string()
}

fn default_llm_max_parallel() -> u32 {
    4
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            // Default to the recommended 70B model. Arandu only delivers
            // acceptable results at this tier; smaller models are user-selectable
            // (onboarding wizard / Settings) but not guaranteed. Requires a
            // capable machine (Apple Silicon Ultra, 64–128 GB RAM) — on weaker
            // hardware it can starve the OS, hence the warnings in the wizard.
            llm_model: default_llm_model(),
            llm_host: default_llm_host(),
            max_sensitivity_tier: default_max_sensitivity_tier(),
            theme: default_theme(),
            data_dir: default_data_dir(),
            auto_refresh_enabled: default_auto_refresh_enabled(),
            auto_refresh_interval_minutes: default_refresh_interval(),
            refresh_on_launch: false,
            onboarding_completed: false,
            onboarding_completed_at: None,
            initial_connectors: vec![],
            skipped_connectors: vec![],
            ollama_configured: false,
            onboarding_followup_pending: false,
            dismissed_nudges: vec![],
            interest_overrides: std::collections::HashMap::new(),
            notifications_enabled: false,
            whatsapp_notification_phone: None,
            user_name: None,
            user_birthday: None,
            user_location: None,
            user_timezone: None,
            user_language: None,
            user_bio: None,
            voice_transcription_enabled: true,
            whisper_model_size: default_whisper_model_size(),
            transcribe_whatsapp_audio: false,
            llm_provider: default_llm_provider(),
            anthropic_api_key: None,
            anthropic_model: default_anthropic_model(),
            external_llm_consent: false,
            llm_api_key: None,
            llm_max_parallel: default_llm_max_parallel(),
            ingest_cutoff_iso: None,
            eval_window_hours: None,
            digest_window_hours: None,
            proactive_window_hours: None,
            insight_window_days: None,
            first_run_grace_minutes: None,
            prevent_sleep: false,
            prevent_sleep_on_battery: false,
            launch_at_login: false,
            menu_bar_mode: false,
            keep_awake_modal_seen: false,
            local_inference_for_sensitive: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Connector catalog types
// ---------------------------------------------------------------------------

/// Sync statistics for a connector.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectorSyncStats {
    pub records_synced: i64,
    pub last_sync: Option<String>,
    /// When rows last actually flowed (distinct from last attempt).
    #[serde(default)]
    pub last_success: Option<String>,
    /// Error from the most recent sync attempt, if it failed.
    #[serde(default)]
    pub error: Option<String>,
    pub next_sync: Option<String>,
}

/// A missing requirement for enabling a connector.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectorMissingRequirement {
    #[serde(rename = "type")]
    pub requirement_type: String,
    pub key: String,
    pub label: String,
    pub action: String,
}

/// A connector entry in the catalog with its current status.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectorCatalogEntry {
    pub connector_id: String,
    pub name: String,
    pub icon: String,
    pub description: String,
    pub category: String,
    pub enabled: bool,
    pub status: String,
    pub stats: ConnectorSyncStats,
    pub missing_requirements: Vec<ConnectorMissingRequirement>,
    pub tools_available: i32,
    pub default_schedule: String,
    pub note: Option<String>,
}

/// Result of toggling a connector on.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectorEnableResult {
    pub status: String,
    pub connector_id: String,
    #[serde(default)]
    pub records_synced: i64,
    #[serde(default)]
    pub tools_available: i32,
    pub next_sync_at: Option<String>,
    #[serde(default)]
    pub missing: Vec<ConnectorMissingRequirement>,
    pub error: Option<String>,
}

/// Result of toggling a connector off.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectorDisableResult {
    pub status: String,
    pub connector_id: String,
    #[serde(default)]
    pub data_preserved: bool,
    pub error: Option<String>,
}

/// Result of triggering a connector sync.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectorSyncResult {
    pub connector_id: String,
    pub status: String,
    pub rows_synced: i64,
    pub duration_seconds: f64,
    pub error: Option<String>,
}

/// Full details for a single connector.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectorDetails {
    #[serde(flatten)]
    pub data: serde_json::Value,
}

// ---------------------------------------------------------------------------
// Extension installer types
// ---------------------------------------------------------------------------

/// Preview of a discovered tool from an MCP server.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolPreview {
    pub tool_name: String,
    pub tool_type: String,
    pub target_table: Option<String>,
    #[serde(default)]
    pub is_new_table: bool,
    #[serde(default)]
    pub field_count: i32,
    #[serde(default)]
    pub sensitivity_tiers: std::collections::HashMap<String, i32>,
    #[serde(default)]
    pub confidence: f64,
    #[serde(default)]
    pub warnings: Vec<String>,
}

/// Complete preview of an MCP server extension install.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstallPreview {
    pub server_name: String,
    pub command: String,
    pub args: Vec<String>,
    pub tools: Vec<ToolPreview>,
    #[serde(default)]
    pub data_tools: i32,
    #[serde(default)]
    pub action_tools: i32,
    #[serde(default)]
    pub new_tables: Vec<String>,
    #[serde(default)]
    pub existing_tables: Vec<String>,
    #[serde(default)]
    pub overall_confidence: f64,
    #[serde(default)]
    pub warnings: Vec<String>,
}

/// Result of confirming an extension install.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstallConfirmResult {
    pub status: String,
    pub connector_id: String,
    #[serde(default)]
    pub tables_created: Vec<String>,
    #[serde(default)]
    pub tools_registered: i32,
    #[serde(default)]
    pub models_staged: i32,
    pub error: Option<String>,
}

// ---------------------------------------------------------------------------
// Model generator types
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Agent runner types
// ---------------------------------------------------------------------------

/// Full agent status including manifest info and last run result.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentFullStatus {
    pub agent_id: String,
    pub name: String,
    pub description: String,
    pub category: String,
    pub status: String,
    pub builtin: bool,
    pub triggers: Vec<String>,
    pub max_sensitivity_tier: u8,
    pub last_run_at: Option<String>,
    pub last_result: Option<String>,
    pub error: Option<String>,
}

/// Result of running an agent.
///
/// # sensitivity_tier: 2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRunResult {
    pub agent_id: String,
    pub status: String,
    pub output: String,
    #[serde(default)]
    pub tables_written: Vec<String>,
    #[serde(default)]
    pub rows_written: i64,
    #[serde(default)]
    pub llm_calls: i32,
    #[serde(default)]
    pub duration_ms: f64,
    pub error: Option<String>,
}

/// Information about a registered skill.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillInfo {
    pub id: String,
    pub name: String,
    pub description: String,
    pub category: String,
    #[serde(default)]
    pub uses_llm: bool,
    #[serde(default)]
    pub builtin: bool,
    #[serde(default)]
    pub parameters: std::collections::HashMap<String, String>,
}

/// Preview of a single generated SQLMesh model.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedModelPreview {
    pub model_name: String,
    pub layer: String,
    pub filename: String,
    #[serde(default)]
    pub sensitivity_summary: std::collections::HashMap<String, i32>,
    #[serde(default)]
    pub depends_on: Vec<String>,
}

/// Kuzu graph schema extension preview.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphExtensionPreview {
    #[serde(default)]
    pub node_names: Vec<String>,
    #[serde(default)]
    pub relationship_names: Vec<String>,
}

/// ChromaDB collection mapping preview.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChromaDBMappingPreview {
    pub collection_name: String,
    pub domain: String,
    #[serde(default)]
    pub indexing_fields: Vec<String>,
}

/// Complete preview of generated pipeline models for user review.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPreview {
    pub connector_id: String,
    pub strategy: String,
    #[serde(default)]
    pub models: Vec<GeneratedModelPreview>,
    pub graph_extension: Option<GraphExtensionPreview>,
    pub chromadb_mapping: Option<ChromaDBMappingPreview>,
    #[serde(default)]
    pub total_models: i32,
    #[serde(default)]
    pub sensitivity_summary: std::collections::HashMap<String, i32>,
    #[serde(default)]
    pub warnings: Vec<String>,
}

/// Result of approving generated models.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelApproveResult {
    pub status: String,
    #[serde(default)]
    pub models_installed: i32,
    #[serde(default)]
    pub files_created: Vec<String>,
    #[serde(default)]
    pub pipeline_models_added: Vec<String>,
    #[serde(default)]
    pub graph_extensions_applied: bool,
    pub error: Option<String>,
}

// ---------------------------------------------------------------------------
// Action tool types
// ---------------------------------------------------------------------------

/// An available MCP action tool from an enabled connector.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvailableAction {
    pub connector_id: String,
    pub connector_name: String,
    pub tool_name: String,
    pub display_name: String,
    pub description: String,
    #[serde(default)]
    pub input_schema: serde_json::Value,
}

/// A proposed action for user confirmation before execution.
///
/// # sensitivity_tier: 3
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionProposal {
    pub proposal_id: String,
    pub connector_id: String,
    pub connector_name: String,
    pub tool_name: String,
    pub display_name: String,
    #[serde(default)]
    pub arguments: serde_json::Value,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub missing_params: Vec<String>,
    #[serde(default)]
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
}

/// A ranked recipient candidate surfaced by the disambiguation card.
///
/// Fields mirror `ContactCandidate` in
/// `src/agents/brain/recipient_resolver.py`.
///
/// # sensitivity_tier: 3
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContactCandidate {
    pub name: String,
    pub handle: Option<String>,
    #[serde(default)]
    pub relationship: String,
    #[serde(default)]
    pub active_topic: String,
    #[serde(default)]
    pub topic_importance: i64,
    #[serde(default)]
    pub notification_priority: i64,
    #[serde(default)]
    pub source: String,
}

/// A pending recipient-choice that blocks a messaging action proposal.
///
/// # sensitivity_tier: 3
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecipientDisambiguationProposal {
    pub proposal_id: String,
    pub connector_id: String,
    pub connector_name: String,
    pub tool_name: String,
    pub display_name: String,
    pub channel: String,
    pub original_name: String,
    #[serde(default)]
    pub candidates: Vec<ContactCandidate>,
    #[serde(default)]
    pub draft_arguments: serde_json::Value,
    #[serde(default)]
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub question: String,
    #[serde(default)]
    pub context_text: String,
}

/// Result of executing a confirmed MCP action.
///
/// # sensitivity_tier: 3
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionResult {
    pub proposal_id: String,
    pub status: String,
    pub output: String,
    #[serde(default)]
    pub raw_result: Vec<serde_json::Value>,
    pub error: Option<String>,
    /// Post-action connector re-sync result (if action succeeded).
    pub post_sync: Option<PostSyncResult>,
}

/// Result of the post-action connector re-sync.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostSyncResult {
    pub status: String,
    #[serde(default)]
    pub rows_synced: i64,
    pub error: Option<String>,
}

// ---------------------------------------------------------------------------
// Extension management types
// ---------------------------------------------------------------------------

/// Result of uninstalling an extension (connector + models + data).
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UninstallResult {
    pub status: String,
    pub connector_id: String,
    #[serde(default)]
    pub tables_removed: Vec<String>,
    #[serde(default)]
    pub data_preserved: bool,
    pub error: Option<String>,
}

/// A single sync history entry for a connector.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectorHistoryEntry {
    pub sync_id: String,
    pub started_at: String,
    pub completed_at: String,
    pub rows_synced: i64,
    pub duration_seconds: f64,
    pub status: String,
    pub error: Option<String>,
}

/// Log output from an extension.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionLogOutput {
    pub extension_id: String,
    pub lines: Vec<String>,
}

// ---------------------------------------------------------------------------
// Health check types
// ---------------------------------------------------------------------------

/// Status of a single system component.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentStatus {
    pub component: String,
    pub ok: bool,
    pub detail: Option<String>,
    pub error: Option<String>,
}

/// Aggregate health check result.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResult {
    pub ok: bool,
    pub checks: Vec<ComponentStatus>,
}

// ---------------------------------------------------------------------------
// Interest profile types
// ---------------------------------------------------------------------------

/// A single interest area from the user's profile.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterestArea {
    pub domain: String,
    pub label: String,
    pub description: String,
    pub weight: f64,
    pub query_count: i64,
    pub queries_per_week: f64,
    pub trending: bool,
    pub explicit: bool,
    pub raw_tables: Vec<String>,
    pub mart: Option<String>,
}

/// Per-domain query statistics.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainStats {
    pub domain: String,
    pub total_queries: i64,
    pub last_queried_at: Option<String>,
    pub trend: String,
    pub queries_last_7d: i64,
    pub queries_last_30d: i64,
}

/// A proactive insight generated from question patterns.
///
/// # sensitivity_tier: 3
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Insight {
    pub id: String,
    pub domain: String,
    pub title: String,
    pub content: String,
    #[serde(default)]
    pub sources: Vec<serde_json::Value>,
    pub trigger: String,
    pub pattern: Option<String>,
    pub generated_at: String,
    #[serde(default)]
    pub sensitivity_tier: i32,
    pub suggested_followup: Option<String>,
}

// ---------------------------------------------------------------------------
// Learned facts types
// ---------------------------------------------------------------------------

/// A learned personal fact from conversations.
///
/// # sensitivity_tier: 2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedFact {
    pub id: String,
    pub category: String,
    pub subject: String,
    pub predicate: String,
    pub content: String,
    #[serde(default)]
    pub confidence: f64,
    pub source_type: String,
    pub extracted_at: String,
    pub confirmed_at: Option<String>,
    pub sensitivity_tier: i32,
    #[serde(default)]
    pub times_used: i32,
}

// ---------------------------------------------------------------------------
// Voice transcription types
// ---------------------------------------------------------------------------

/// Result of transcribing an audio file via local Whisper model.
///
/// # sensitivity_tier: 3
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResult {
    pub text: String,
    pub language: String,
    pub duration: f64,
    #[serde(default)]
    pub segments: Vec<TranscriptionSegment>,
}

/// A timestamped segment of transcribed text.
///
/// # sensitivity_tier: 3
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionSegment {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

// ---------------------------------------------------------------------------
// Notification types
// ---------------------------------------------------------------------------

/// A per-category notification preference.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationPreference {
    pub category: String,
    pub enabled: bool,
    pub muted_until: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

/// A notification log entry (decision + delivery).
///
/// # sensitivity_tier: 2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationRecord {
    pub id: String,
    pub dedupe_key: String,
    pub category: String,
    pub importance_score: f64,
    pub decision: String,
    pub delivery_status: String,
    pub message: String,
    #[serde(default)]
    pub opt_out_text: String,
    pub error: Option<String>,
    pub source_type: String,
    pub source_id: String,
    pub created_at: String,
}

// ---------------------------------------------------------------------------
// Proactive intelligence types
// ---------------------------------------------------------------------------

/// A message identified as needing the user's reply.
///
/// # sensitivity_tier: 2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingReply {
    pub id: String,
    pub message_id: String,
    pub source: String,
    pub contact_name: String,
    pub domain: String,
    pub preview: String,
    pub importance: i32,
    pub reason: String,
    pub message_at: String,
    pub detected_at: String,
    #[serde(default = "default_sensitivity_tier_2")]
    pub sensitivity_tier: i32,
}

/// Origin of a "Draft reply" click from any reply-suggesting surface
/// (Today's Loops, Inbox, domain Open Loops). Carries the source
/// channel and original `raw_messages.id` so the Brain can hard-lock
/// the proposed reply to the correct platform and resolve contact
/// info (WhatsApp JID, email address, iMessage handle).
///
/// # sensitivity_tier: 2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplyContext {
    pub source: String,
    pub message_id: String,
    #[serde(default)]
    pub contact_name: Option<String>,
}

/// Context for a "Work on this" click on a task in the Inbox.
/// Seeds the task + goal details into the Chat prompt so the
/// assistant can help the user complete it.
///
/// # sensitivity_tier: 2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskContext {
    pub task_id: String,
    #[serde(default)]
    pub goal_id: Option<String>,
}

/// Aggregated context for an important contact.
///
/// # sensitivity_tier: 3
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContactContext {
    pub contact_id: String,
    pub contact_name: String,
    pub phone: Option<String>,
    pub email: Option<String>,
    #[serde(default)]
    pub total_messages: i32,
    #[serde(default)]
    pub messages_7d: i32,
    pub last_message_at: Option<String>,
    pub last_message_preview: Option<String>,
    #[serde(default)]
    pub total_events: i32,
    pub next_event_at: Option<String>,
    pub next_event_title: Option<String>,
    pub active_context: Option<String>,
    #[serde(default)]
    pub context_domains: Vec<String>,
    #[serde(default)]
    pub context_priority: i32,
    pub birthday: Option<String>,
    #[serde(default)]
    pub has_upcoming_birthday: bool,
    pub updated_at: String,
}

/// A calendar event or birthday needing user action.
///
/// # sensitivity_tier: 2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionableEvent {
    pub id: String,
    pub event_id: String,
    pub event_type: String,
    pub title: String,
    pub event_date: String,
    pub contact_name: Option<String>,
    pub action_needed: String,
    #[serde(default = "default_importance")]
    pub importance: i32,
    pub detected_at: String,
    #[serde(default = "default_sensitivity_tier_2")]
    pub sensitivity_tier: i32,
}

/// One known attendee's standing going into a meeting.
///
/// # sensitivity_tier: 3
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingPrepAttendee {
    pub contact_name: String,
    pub situation: Option<String>,
    #[serde(default)]
    pub domains: Vec<String>,
    pub last_message_at: Option<String>,
    pub last_message_preview: Option<String>,
    #[serde(default)]
    pub open_loops: Vec<serde_json::Value>,
}

/// A prep pack for an upcoming meeting with known attendees.
///
/// # sensitivity_tier: 3
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingPrepBrief {
    pub event_id: String,
    pub title: String,
    pub start_time: String,
    pub location: Option<String>,
    #[serde(default)]
    pub attendees: Vec<MeetingPrepAttendee>,
    #[serde(default)]
    pub generated_at: String,
}

/// A single entry in the topic digest notification.
///
/// # sensitivity_tier: 3
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicDigestEntry {
    pub contact_name: String,
    pub topic: String,
    #[serde(default)]
    pub description: String,
    #[serde(default = "default_importance")]
    pub importance: i32,
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub change_type: String,
    pub previous_importance: Option<i32>,
}

/// Combined result of all four proactive evaluation pillars.
///
/// # sensitivity_tier: 3
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProactiveResult {
    #[serde(default)]
    pub pending_replies: Vec<PendingReply>,
    #[serde(default)]
    pub contact_contexts: Vec<ContactContext>,
    #[serde(default)]
    pub actionable_events: Vec<ActionableEvent>,
    #[serde(default)]
    pub topic_digest: Vec<TopicDigestEntry>,
    pub evaluated_at: String,
}

// ---------------------------------------------------------------------------
// Background task status types
// ---------------------------------------------------------------------------

/// A currently-running background task visible in the sidebar.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackgroundTask {
    pub id: String,
    pub label: String,
    pub started_at: String,
}

fn default_sensitivity_tier_2() -> i32 {
    2
}

fn default_importance() -> i32 {
    5
}

// ---------------------------------------------------------------------------
// Agents page (Phase 4) — Pydantic AI agent registry surface
// ---------------------------------------------------------------------------

/// Resolved configuration for one agent (default ⊕ user override).
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PydanticAgentConfig {
    pub system_prompt: String,
    pub model_route: String,
    #[serde(default)]
    pub model_override: Option<String>,
    /// Concrete LLM model name that would be used right now (override
    /// when set, otherwise the route's configured model).
    #[serde(default)]
    pub resolved_model: Option<String>,
    #[serde(default)]
    pub enabled_tools: Vec<String>,
    #[serde(default)]
    pub enabled_skills: Vec<String>,
    pub version: i32,
    /// Action-typed tool ids the runner dispatches as a post-batch
    /// delivery hook. Populated for user-authored agents only;
    /// always `[]` for built-ins.
    #[serde(default)]
    pub delivery_tools: Vec<String>,
}

/// One row in the Agents page — registry metadata + resolved config.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PydanticAgentRow {
    pub agent_id: String,
    pub name: String,
    pub description: String,
    pub category: String,
    #[serde(default)]
    pub parent_agent: Option<String>,
    pub tier: String,
    pub max_sensitivity_tier: i32,
    pub editable: bool,
    pub pattern: String,
    pub output_schema: String,
    #[serde(default)]
    pub available_tools: Vec<String>,
    #[serde(default)]
    pub available_skills: Vec<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    /// Sub-agent ids this agent delegates to. Empty for single-pattern
    /// agents; non-empty for orchestrators/deep agents. Lets the UI
    /// render the architecture without instantiating the factory.
    #[serde(default)]
    pub subagents: Vec<String>,
    pub config: PydanticAgentConfig,
}

/// `agents-list` response envelope.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PydanticAgentListResponse {
    pub agents: Vec<PydanticAgentRow>,
}

/// `agents-get` / `agents-update` / `agents-reset` response envelope.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PydanticAgentResponse {
    pub agent: PydanticAgentRow,
}

/// One failed case in an eval run, displayed in the editor's
/// expanding failures panel.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentEvalFailedCase {
    pub case: String,
    pub evaluator: String,
    pub reason: String,
}

/// One persisted eval row for an agent — the row the Agents page
/// reads to render the status banner.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentEvalRun {
    pub run_id: String,
    pub agent_id: String,
    #[serde(default)]
    pub suite: Option<String>,
    pub trigger: String,
    pub started_at: String,
    #[serde(default)]
    pub finished_at: Option<String>,
    pub status: String,
    pub cases_total: i32,
    pub cases_passed: i32,
    pub cases_failed: i32,
    #[serde(default)]
    pub failed_cases: Vec<AgentEvalFailedCase>,
    #[serde(default)]
    pub error: Option<String>,
}

/// `agents-run-eval` response envelope.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentEvalRunResponse {
    pub run: AgentEvalRun,
}

/// `agents-run-eval-proposal` response envelope.
///
/// Backs the "Test & save" flow on the Agents page: the model
/// override is applied via a ContextVar scope, the eval suite runs
/// against the proposed model, and the persisted run is returned so
/// the UI can decide whether to call `update_agent_config`.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentEvalProposalResponse {
    pub run: AgentEvalRun,
    pub proposed_override: String,
}

/// `agents-eval-status` response envelope.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentEvalStatusResponse {
    #[serde(default)]
    pub latest: Option<AgentEvalRun>,
    #[serde(default)]
    pub history: Vec<AgentEvalRun>,
}

/// One per-agent eval result from the local-inference gate.
///
/// Returned inside `LocalInferenceToggleResponse` so the UI can show
/// the user exactly which agents failed when the toggle aborts.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalInferenceEvalFailure {
    pub agent_id: String,
    pub status: String,
    #[serde(default)]
    pub failed_cases: Vec<serde_json::Value>,
    #[serde(default)]
    pub error: Option<String>,
}

/// One per-agent eval result from the local-inference gate.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalInferenceEvalResult {
    pub agent_id: String,
    pub status: String,
    #[serde(default)]
    pub cases_total: i32,
    #[serde(default)]
    pub cases_passed: i32,
    #[serde(default)]
    pub cases_failed: i32,
}

/// `set-local-inference-for-sensitive` response envelope.
///
/// `status` is `"ok"` when the flag committed, or `"eval_failed"`
/// when one or more agents' eval suites did not return `"passed"`
/// (the toggle is left disabled in that case). `failures` is empty
/// on `"ok"`.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalInferenceToggleResponse {
    pub status: String,
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub failures: Vec<LocalInferenceEvalFailure>,
    #[serde(default)]
    pub results: Vec<LocalInferenceEvalResult>,
}

/// One row of the per-agent input/output run log.
///
/// `input` is the raw prompt string the agent received;
/// `output` is the JSON-serialized structured output (or `None` for
/// errored runs). `route` is `"remote"` / `"local"`. `status` is
/// `"ok"` / `"error"`. Newest entries come first when emitted by
/// `agents-activity`.
///
/// # sensitivity_tier: varies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRunLogEntry {
    pub id: i64,
    pub agent_id: String,
    pub ts: String,
    #[serde(default)]
    pub input: Option<String>,
    #[serde(default)]
    pub output: Option<String>,
    #[serde(default)]
    pub duration_ms: Option<f64>,
    #[serde(default)]
    pub route: Option<String>,
    pub status: String,
    #[serde(default)]
    pub error: Option<String>,
}

/// `agents-activity` response envelope.
///
/// # sensitivity_tier: varies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentActivityResponse {
    pub agent_id: String,
    #[serde(default)]
    pub entries: Vec<AgentRunLogEntry>,
}

/// Patch shape sent from the Agents page editor.
///
/// All fields are optional — only present keys are applied.
///
/// Result of `agents-list-models` — the model ids exposed by one route's
/// endpoint (chat-family ids sorted first). `error` is populated when the
/// underlying call failed; in that case `models` is empty and the
/// dropdown should fall back to a free-text input.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvailableModels {
    pub route: String,
    pub models: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PydanticAgentPatch {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_route: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_override: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled_tools: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled_skills: Option<Vec<String>>,
    /// Post-batch delivery hook tool ids. Only valid for user-authored
    /// agents — the handler rejects this field for built-ins.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delivery_tools: Option<Vec<String>>,
}

// ---------------------------------------------------------------------------
// Agents page (Phase 4b) — eval datasets + user agents + user skills
// ---------------------------------------------------------------------------

/// One row in the parsed-cases preview shown next to the raw YAML.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentEvalDatasetCase {
    pub name: String,
    pub inputs: String,
    #[serde(default)]
    pub expected_output: Option<String>,
    #[serde(default)]
    pub evaluators: Vec<String>,
}

/// `agents-eval-dataset` response — raw YAML + parsed cases.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentEvalDataset {
    pub agent_id: String,
    #[serde(default)]
    pub suite: Option<String>,
    /// One of `builtin`, `user`, `none`.
    pub source: String,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub parsed_cases: Vec<AgentEvalDatasetCase>,
    pub exists: bool,
}

/// Verdict for a user-uploaded eval dataset.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetValidationReport {
    pub valid: bool,
    #[serde(default)]
    pub errors: Vec<String>,
    #[serde(default)]
    pub proposals: Vec<String>,
    /// One of `allow`, `warn`, `block`.
    pub firewall_verdict: String,
}

/// `agents-validate-dataset` response envelope.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetValidationResponse {
    pub report: DatasetValidationReport,
    pub persisted: bool,
}

/// In-flight create-modal spec passed to `suggest_eval_dataset` when
/// the agent has not been persisted yet.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnsavedAgentSpec {
    pub name: String,
    pub description: String,
    pub system_prompt: String,
    pub max_sensitivity_tier: i32,
    #[serde(default)]
    pub output_schema: Option<String>,
    #[serde(default)]
    pub available_tools: Vec<String>,
}

/// Proposed eval dataset for a user agent. Mirrors the Python
/// `DatasetSuggestion` model.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetSuggestion {
    pub can_create: bool,
    #[serde(default)]
    pub reason_if_not: Option<String>,
    #[serde(default)]
    pub purpose_summary: String,
    /// One of `structured`, `prose`, `classification`, `mixed`, `unknown`.
    #[serde(default)]
    pub output_shape: String,
    /// One of `deterministic`, `llm_judge`, `hybrid`.
    #[serde(default)]
    pub eval_strategy: String,
    #[serde(default)]
    pub dataset_yaml: String,
    #[serde(default)]
    pub case_count: i32,
    #[serde(default)]
    pub confidence: f64,
    #[serde(default)]
    pub notes: Vec<String>,
    #[serde(default)]
    pub improvement_hints: Vec<String>,
    /// One-line additions the user should append to their system
    /// prompt so the LLM produces the tokens the dataset expects
    /// (closed vocabularies, output language, format strictness).
    #[serde(default)]
    pub system_prompt_additions: Vec<String>,
}

/// `agents-suggest-dataset` response envelope.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetSuggestionResponse {
    pub suggestion: DatasetSuggestion,
    #[serde(default)]
    pub existing_case_names: Vec<String>,
    #[serde(default)]
    pub has_existing_dataset: bool,
}

/// One eval case that failed on a previously-tested model, fed back
/// into the next picker round.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPickerFailedCase {
    pub name: String,
    pub evaluator: String,
    pub reason: String,
}

/// A previously-tested model and the cases it failed on. The picker
/// uses the failed-case reasons to infer which capability gap to
/// close on the next pick.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPickerPriorAttempt {
    pub model_id: String,
    /// `"remote"` or `"local"`.
    pub route: String,
    #[serde(default)]
    pub failed_cases: Vec<ModelPickerFailedCase>,
}

/// Live spec sent to `agents-suggest-model` from the agent wizard or
/// edit row. The remote/local model lists are NOT included — the CLI
/// handler fetches them server-side from the live endpoint.
///
/// `excluded_models` and `prior_attempts` carry iteration feedback
/// from earlier suggest → use → eval rounds: the picker must NOT
/// re-suggest excluded ids and should reason about why the prior
/// attempts failed.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPickerSpec {
    pub name: String,
    pub description: String,
    pub system_prompt: String,
    pub max_sensitivity_tier: i32,
    #[serde(default)]
    pub output_schema: Option<String>,
    #[serde(default)]
    pub enabled_skills: Vec<String>,
    #[serde(default)]
    pub enabled_mcp_tools: Vec<String>,
    #[serde(default)]
    pub agent_id: Option<String>,
    #[serde(default)]
    pub excluded_models: Vec<String>,
    #[serde(default)]
    pub prior_attempts: Vec<ModelPickerPriorAttempt>,
}

/// One model recommendation. Mirrors the Python `ModelOption`.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelOption {
    pub model_id: String,
    /// `"remote"` or `"local"`.
    pub route: String,
    #[serde(default)]
    pub rationale: String,
}

/// A model-picker suggestion for a user agent. Mirrors the Python
/// `ModelRecommendation` model.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRecommendation {
    pub can_recommend: bool,
    #[serde(default)]
    pub reason_if_not: Option<String>,
    #[serde(default)]
    pub purpose_summary: String,
    #[serde(default)]
    pub best_overall: Option<ModelOption>,
    #[serde(default)]
    pub cost_effective: Option<ModelOption>,
    #[serde(default)]
    pub notes: Vec<String>,
    #[serde(default)]
    pub improvement_hints: Vec<String>,
    #[serde(default)]
    pub confidence: f64,
}

/// `agents-suggest-model` response envelope.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRecommendationResponse {
    pub recommendation: ModelRecommendation,
    #[serde(default)]
    pub available_remote_models: Vec<String>,
    #[serde(default)]
    pub available_local_models: Vec<String>,
}

/// Live spec sent to `agents-suggest-prompt-improvements` from the
/// agent wizard or edit row.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptEngineerSpec {
    pub name: String,
    pub description: String,
    pub system_prompt: String,
    pub max_sensitivity_tier: i32,
    #[serde(default)]
    pub output_schema: Option<String>,
    #[serde(default)]
    pub available_tools: Vec<String>,
    #[serde(default)]
    pub available_skills: Vec<String>,
    #[serde(default)]
    pub enabled_mcp_tools: Vec<String>,
    #[serde(default)]
    pub agent_id: Option<String>,
    #[serde(default)]
    pub has_dataset: bool,
    #[serde(default)]
    pub prior_eval_failures: Vec<PromptEngineerEvalFailure>,
}

/// One failed eval case fed back into the prompt engineer.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptEngineerEvalFailure {
    pub name: String,
    pub evaluator: String,
    pub reason: String,
}

/// One categorised edit the prompt engineer recommends. Mirrors the
/// Python `Improvement` model.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptImprovement {
    /// `clarity`, `expected_output`, `language`, `format`, `scope`, `safety`.
    pub category: String,
    #[serde(default)]
    pub original_snippet: String,
    pub suggested_replacement: String,
    pub rationale: String,
    #[serde(default = "default_improvement_target")]
    pub target: String,
}

fn default_improvement_target() -> String {
    "system_prompt".into()
}

/// A prompt-engineer rewrite of a user agent's prompt + description.
/// Mirrors the Python `PromptSuggestion` model.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptSuggestion {
    pub can_improve: bool,
    #[serde(default)]
    pub reason_if_not: Option<String>,
    #[serde(default)]
    pub improved_system_prompt: String,
    #[serde(default)]
    pub improved_description: String,
    #[serde(default)]
    pub system_prompt_additions: Vec<String>,
    #[serde(default)]
    pub improvements: Vec<PromptImprovement>,
    #[serde(default)]
    pub change_summary: String,
    #[serde(default)]
    pub confidence: f64,
    #[serde(default)]
    pub notes: Vec<String>,
}

/// `agents-suggest-prompt-improvements` response envelope.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptSuggestionResponse {
    pub suggestion: PromptSuggestion,
}

/// One row of `user_agents` as seen by the UI.
///
/// `pre_ai_system_prompt` / `pre_ai_description` are non-null when a
/// prompt-engineer rewrite has been applied and not yet reverted. The
/// UI uses their presence to surface a "Revert prompt-engineer edits"
/// button.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserAgentRow {
    pub agent_id: String,
    pub name: String,
    pub description: String,
    pub system_prompt: String,
    pub model_route: String,
    #[serde(default)]
    pub model_override: Option<String>,
    #[serde(default)]
    pub enabled_skills: Vec<String>,
    /// Catalog tool ids (`connector_id:tool_name`) the agent is wired
    /// to. Includes BOTH data tools (sources) and action tools
    /// (LLM-callable mid-run); the runner uses catalog `tool_type` at
    /// dispatch time to decide which list each entry belongs to.
    #[serde(default)]
    pub enabled_mcp_tools: Vec<String>,
    pub brain_access: bool,
    pub max_sensitivity_tier: i32,
    #[serde(default)]
    pub schedule_cron: Option<String>,
    pub schedule_enabled: bool,
    pub created_at: String,
    pub updated_at: String,
    pub version: i32,
    #[serde(default)]
    pub pre_ai_system_prompt: Option<String>,
    #[serde(default)]
    pub pre_ai_description: Option<String>,
    /// Action-typed tool ids the runner invokes as a post-batch
    /// delivery hook. Independent of `enabled_mcp_tools` — delivery
    /// is never exposed to the LLM during per-item runs.
    #[serde(default)]
    pub delivery_tools: Vec<String>,
}

/// Input payload for `create_user_agent` / `update_user_agent`.
///
/// `pattern` defaults to `"single"`; set it to `"orchestrator"` plus a
/// non-empty `subagents` list to register an agent that delegates to
/// other registered agents via the SBOrchestrator base class. Both
/// fields are optional in the wire format so older clients keep working.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserAgentInput {
    pub name: String,
    pub description: String,
    pub system_prompt: String,
    pub model_route: String,
    #[serde(default)]
    pub model_override: Option<String>,
    #[serde(default)]
    pub enabled_skills: Vec<String>,
    #[serde(default)]
    pub enabled_mcp_tools: Vec<String>,
    pub brain_access: bool,
    pub max_sensitivity_tier: i32,
    #[serde(default)]
    pub schedule_cron: Option<String>,
    pub schedule_enabled: bool,
    #[serde(default = "default_pattern")]
    pub pattern: String,
    #[serde(default)]
    pub subagents: Vec<String>,
    #[serde(default)]
    pub delivery_tools: Vec<String>,
    /// When true, mark all existing source items as already processed
    /// so the agent only sees new data from creation time onward.
    #[serde(default)]
    pub skip_backfill: bool,
}

fn default_pattern() -> String {
    "single".to_string()
}

/// Scheduling + run-time status for one user agent.
///
/// Powers the Agents page schedule strip — `next_run_at` is computed
/// from the cron expression at status-read time, not stored on disk.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserAgentStatus {
    pub agent_id: String,
    #[serde(default)]
    pub schedule_cron: Option<String>,
    pub schedule_enabled: bool,
    /// Data-typed entries of `enabled_mcp_tools` (`connector_id:tool_name`)
    /// that drive the batch runner each tick.
    #[serde(default)]
    pub enabled_data_tools: Vec<String>,
    /// Tool ids dispatched by the post-batch delivery hook.
    #[serde(default)]
    pub delivery_tools: Vec<String>,
    #[serde(default)]
    pub last_run_at: Option<String>,
    #[serde(default)]
    pub last_status: Option<String>,
    #[serde(default)]
    pub last_error: Option<String>,
    #[serde(default)]
    pub next_run_at: Option<String>,
    pub pending_count: i64,
}

/// Result of one `run_user_agent_now` invocation.
///
/// # sensitivity_tier: 2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchRunSummary {
    pub agent_id: String,
    /// "batch" | "generic"
    pub mode: String,
    pub checked: i64,
    pub processed: i64,
    pub errors: i64,
    pub skipped: i64,
    #[serde(default)]
    pub run_ids: Vec<String>,
    #[serde(default)]
    pub error_messages: Vec<String>,
    /// Post-batch delivery hook results — one entry per
    /// `delivery_tools` invocation. Empty when no delivery tools are
    /// configured or the hook was skipped (no items processed).
    #[serde(default)]
    pub delivery_calls: Vec<DeliveryCallRecord>,
}

/// One post-batch delivery dispatch outcome.
///
/// # sensitivity_tier: 2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeliveryCallRecord {
    pub tool_id: String,
    /// "success" | "error"
    pub status: String,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub result_preview: Option<String>,
}

/// `create_user_agent` / `update_user_agent` response envelope.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserAgentResponse {
    pub agent: PydanticAgentRow,
    pub user_row: UserAgentRow,
}

/// One MCP tool (data or action), surfaced to the unified picker.
///
/// `tool_type` is `"action"` for LLM-callable tools, `"data"` for
/// poller tools that feed `target_table`. The frontend uses these to
/// split each connector card into sources / callable / delivery rows.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolEntry {
    pub connector_id: String,
    pub connector_name: String,
    pub tool_name: String,
    pub display_name: String,
    pub description: String,
    pub tool_type: String,
    #[serde(default)]
    pub target_table: Option<String>,
    #[serde(default)]
    pub input_schema: serde_json::Value,
}

/// `list_mcp_action_tools` response envelope.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpActionToolListResponse {
    pub tools: Vec<McpToolEntry>,
}

/// Full skill detail, including the prompt template, for the Inspect
/// panel + the edit form.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillDetail {
    pub skill_id: String,
    pub name: String,
    pub description: String,
    pub category: String,
    #[serde(default)]
    pub prompt_template: String,
    #[serde(default)]
    pub parameters: std::collections::HashMap<String, String>,
    #[serde(default)]
    pub uses_llm: bool,
    #[serde(default)]
    pub builtin: bool,
}

/// `get_skill` response envelope.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillDetailResponse {
    pub skill: SkillDetail,
}

/// Input payload for `create_user_skill` / `update_user_skill`.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserSkillInput {
    pub name: String,
    pub description: String,
    pub category: String,
    pub prompt_template: String,
    #[serde(default)]
    pub parameters: std::collections::HashMap<String, String>,
    pub uses_llm: bool,
}

/// `create_user_skill` / `update_user_skill` response envelope.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserSkillMutationResponse {
    pub skill: SkillDetail,
}

// ---------------------------------------------------------------------------
// Mission Control dashboard (Phase 1)
// ---------------------------------------------------------------------------

/// Today's synthesized brief — narrative produced by `BrainAgentV2`.
///
/// Cached server-side keyed by (date, pipeline.completed_at). The
/// Dashboard's "Regenerate" button passes `force: true` to bypass
/// the cache.
///
/// # sensitivity_tier: 3
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyBrief {
    pub brief: String,
    pub generated_at: String,
    pub source_counts: BriefSourceCounts,
}

/// Counts of source items that fed the brief synthesis.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BriefSourceCounts {
    #[serde(default)]
    pub events: u32,
    #[serde(default)]
    pub threads: u32,
    #[serde(default)]
    pub pending_replies: u32,
    #[serde(default)]
    pub actionable_events: u32,
}

/// A cross-source unit of attention surfaced on the dashboard.
///
/// One Thread aggregates signals across messages / events / contacts
/// into a single named, status-tagged item with inline actions.
///
/// # sensitivity_tier: 3
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Thread {
    pub id: String,
    pub kind: String, // "conversation" | "event"
    pub title: String,
    #[serde(default)]
    pub subtitle: Option<String>,
    pub status: String, // "waiting" | "soon" | "healthy" | "quiet"
    #[serde(default)]
    pub sources: Vec<String>,
    #[serde(default)]
    pub last_activity: Option<String>,
    #[serde(default)]
    pub suggested_actions: Vec<ThreadAction>,
}

/// One inline action button on a Thread row.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadAction {
    pub label: String,
    pub intent: String,
    #[serde(default)]
    pub payload: serde_json::Value,
}

/// Live agent activity for the Mission Control "Agents at Work" panel.
///
/// `running` is filled from the Rust-side `active_tasks` map (the
/// Python CLI returns `[]`). The other two come from the CLI.
///
/// # sensitivity_tier: 2
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AgentStream {
    #[serde(default)]
    pub running: Vec<AgentRunning>,
    #[serde(default)]
    pub awaiting_review: Vec<AgentReview>,
    #[serde(default)]
    pub recently_completed: Vec<AgentCompleted>,
}

/// One agent currently running.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRunning {
    pub task_id: String,
    pub agent_name: String,
    pub label: String,
    #[serde(default)]
    pub progress: Option<AgentProgress>,
    pub started_at: String,
}

/// Optional progress for a running agent.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentProgress {
    pub current: u32,
    pub total: u32,
    #[serde(default)]
    pub eta_seconds: Option<u32>,
}

/// An item the user should review (drafted reply, surfaced insight).
///
/// # sensitivity_tier: 2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentReview {
    pub id: String,
    pub agent_name: String,
    pub summary: String,
    pub kind: String, // "reply" | "insight"
    pub payload_ref: String,
}

/// An agent run that finished in the last 24h.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCompleted {
    pub id: String,
    pub agent_name: String,
    pub summary: String,
    pub finished_at: String,
}

/// Command Bar suggestion chips — derived from the user's current state.
///
/// # sensitivity_tier: 2
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SuggestedActions {
    #[serde(default)]
    pub chips: Vec<SuggestedChip>,
}

/// One Command Bar chip.
///
/// # sensitivity_tier: 2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuggestedChip {
    pub label: String,
    pub prefilled_prompt: String,
}

/// Per-life-domain summary (work / personal / health).
///
/// Phase 2 of the Mission Control redesign — surfaces what's moving
/// in one area of the user's life right now. Backed by the existing
/// `mart_work` / `mart_personal` / `mart_health` tables.
///
/// # sensitivity_tier: 3
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainSummary {
    pub domain: String,
    #[serde(default)]
    pub items: Vec<DomainItem>,
    #[serde(default)]
    pub open_loops: Vec<DomainOpenLoop>,
}

/// One noteworthy item in a domain — an event (work/personal) or a
/// metric (health). `kind` discriminates so the UI can choose icon
/// + layout.
///
/// # sensitivity_tier: 3
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainItem {
    pub id: String,
    pub kind: String, // "event" | "metric" | "note"
    pub title: String,
    #[serde(default)]
    pub subtitle: Option<String>,
    #[serde(default)]
    pub when: Option<String>,
    #[serde(default)]
    pub badge: Option<String>,
    #[serde(default)]
    pub contact: Option<String>,
    // "personal" | "team_awareness" | "subscribed"; only meaningful when
    // kind == "event". Drives the dashboard's split into the user's own
    // meetings vs. shared-calendar awareness events.
    #[serde(default)]
    pub event_origin: Option<String>,
}

/// A commitment / unanswered message waiting on the user inside a
/// domain bucket. Phase 2 only surfaces pending replies; commitment
/// extraction from notes is a follow-up.
///
/// # sensitivity_tier: 2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainOpenLoop {
    pub id: String,
    pub kind: String, // "reply"
    pub label: String,
    pub context: String,
    pub age_days: u32,
    #[serde(default)]
    pub suggested_action: Option<String>,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub message_id: Option<String>,
    #[serde(default)]
    pub contact_name: Option<String>,
}

/// Result of `rebuild-vector-index`. Wraps the Phase 2 migration
/// CLI so the Settings UI can trigger a model swap without shelling
/// out to a terminal. `progress` is the human-readable narrative
/// emitted by the CLI (model details, drop counts, reindex summary)
/// so the UI can render it verbatim.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RebuildVectorIndexResult {
    pub ok: bool,
    #[serde(default)]
    pub exit_code: Option<i32>,
    #[serde(default)]
    pub target_model: Option<String>,
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub dry_run: Option<bool>,
    #[serde(default)]
    pub progress: Option<String>,
    #[serde(default)]
    pub error: Option<String>,
}

// ---------------------------------------------------------------------------
// Tasks / Goals / Habits / Schedule types
//
// Mirror src/agents/tasks/models.py exactly — the Python CLI handlers
// `asdict()` those dataclasses and the Tauri layer deserialises here.
// Keep field names and types in lock-step.
// ---------------------------------------------------------------------------

fn default_sensitivity_tier_1() -> i32 {
    1
}

/// A user-level goal (work / life / personal) aggregated by the Brain
/// or entered manually.
///
/// # sensitivity_tier: 2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Goal {
    pub id: String,
    pub title: String,
    #[serde(default)]
    pub description: String,
    #[serde(default = "default_category")]
    pub category: String,
    #[serde(default = "default_horizon")]
    pub horizon: String,
    pub target_date: Option<String>,
    #[serde(default = "default_active_status")]
    pub status: String,
    #[serde(default = "default_importance")]
    pub importance: i32,
    #[serde(default)]
    pub why: String,
    #[serde(default = "default_user_source")]
    pub source: String,
    pub source_ref: Option<String>,
    #[serde(default)]
    pub created_at: String,
    #[serde(default)]
    pub updated_at: String,
    pub last_confirmed_at: Option<String>,
    #[serde(default = "default_sensitivity_tier_2")]
    pub sensitivity_tier: i32,
    /// Derived at list-time by ``cmd_goals_list``; not persisted on the
    /// row. Blends tasks_today, overdue work, target-date proximity
    /// and horizon so the dashboard can rank goals by what's pressing.
    #[serde(default)]
    pub urgency_score: i32,
    /// Decay verdict for a brain-mined goal whose evidence has gone
    /// stale (`fresh` / `fading` / `stale`). Always `fresh` for
    /// user-entered goals. Derived at list-time; not persisted.
    #[serde(default = "default_fresh_decay")]
    pub decay_state: String,
    /// Days since the goal's evidence anchor (`last_confirmed_at`, else
    /// `created_at`). Derived at list-time; not persisted.
    #[serde(default)]
    pub days_since_confirmed: i32,
}

/// A grouping of tasks. May roll up under a goal or topic.
///
/// # sensitivity_tier: 2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Project {
    pub id: String,
    pub name: String,
    #[serde(default = "default_category")]
    pub category: String,
    pub topic_id: Option<String>,
    pub goal_id: Option<String>,
    #[serde(default = "default_active_status")]
    pub status: String,
    pub color: Option<String>,
    #[serde(default)]
    pub created_at: String,
    #[serde(default)]
    pub updated_at: String,
    #[serde(default = "default_sensitivity_tier_2")]
    pub sensitivity_tier: i32,
}

/// A single tracked unit of work.
///
/// Subtask iff `parent_task_id` is non-null.
///
/// # sensitivity_tier: 2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: String,
    pub title: String,
    pub project_id: Option<String>,
    pub parent_task_id: Option<String>,
    pub goal_id: Option<String>,
    #[serde(default)]
    pub notes: String,
    #[serde(default = "default_todo_status")]
    pub status: String,
    #[serde(default = "default_importance")]
    pub importance: i32,
    pub due_at: Option<String>,
    pub scheduled_for: Option<String>,
    #[serde(default = "default_user_source")]
    pub source: String,
    pub source_ref: Option<String>,
    pub completion_note: Option<String>,
    pub completion_evidence_id: Option<String>,
    pub completed_at: Option<String>,
    #[serde(default)]
    pub created_at: String,
    #[serde(default)]
    pub updated_at: String,
    #[serde(default = "default_sensitivity_tier_2")]
    pub sensitivity_tier: i32,
}

/// A recurring practice anchored to a goal (atomic-habits style).
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Habit {
    pub id: String,
    pub title: String,
    pub goal_id: String,
    #[serde(default = "default_daily_cadence")]
    pub cadence: String,
    #[serde(default)]
    pub days_of_week: Vec<String>,
    #[serde(default = "default_any_window")]
    pub preferred_window: String,
    #[serde(default)]
    pub why: String,
    #[serde(default = "default_user_source")]
    pub source: String,
    #[serde(default = "default_active_status")]
    pub status: String,
    #[serde(default)]
    pub created_at: String,
    #[serde(default = "default_sensitivity_tier_1")]
    pub sensitivity_tier: i32,
}

/// One slot inside a persisted day plan.
///
/// # sensitivity_tier: 2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduleSlot {
    pub kind: String,
    pub ref_id: String,
    pub title: String,
    pub start: String,
    pub end: String,
    #[serde(default)]
    pub why: String,
    pub category: Option<String>,
    pub goal_id: Option<String>,
    #[serde(default)]
    pub goal_title: Option<String>,
}

/// A persisted daily plan.
///
/// # sensitivity_tier: 2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailySchedule {
    pub schedule_date: String,
    #[serde(default)]
    pub slots: Vec<ScheduleSlot>,
    #[serde(default)]
    pub unscheduled_overflow: Vec<String>,
    #[serde(default)]
    pub rationale: String,
    #[serde(default)]
    pub category_balance: std::collections::HashMap<String, i32>,
    #[serde(default)]
    pub generated_at: String,
    #[serde(default = "default_sensitivity_tier_2")]
    pub sensitivity_tier: i32,
}

fn default_category() -> String {
    "personal".to_string()
}
fn default_horizon() -> String {
    "medium".to_string()
}
fn default_active_status() -> String {
    "active".to_string()
}
fn default_todo_status() -> String {
    "todo".to_string()
}
fn default_user_source() -> String {
    "user".to_string()
}
fn default_fresh_decay() -> String {
    "fresh".to_string()
}
fn default_daily_cadence() -> String {
    "daily".to_string()
}
fn default_any_window() -> String {
    "any".to_string()
}

// ---------------------------------------------------------------------------
// Today board / Inbox / Goal progress (dashboard refocus)
// ---------------------------------------------------------------------------

/// One row in the "Today's loops" column of the Today board.
///
/// # sensitivity_tier: 2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TodayLoop {
    pub id: String,
    pub kind: String,
    pub label: String,
    #[serde(default)]
    pub context: String,
    pub importance: i32,
    #[serde(default)]
    pub age_days: i32,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub message_id: Option<String>,
    #[serde(default)]
    pub contact_name: Option<String>,
}

/// Dashboard's prioritized "Today" surface — Now, Up Next, Loops.
///
/// # sensitivity_tier: 3
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TodayBoard {
    #[serde(default)]
    pub now: Vec<ScheduleSlot>,
    #[serde(default)]
    pub up_next: Vec<ScheduleSlot>,
    #[serde(default)]
    pub todays_loops: Vec<TodayLoop>,
    #[serde(default)]
    pub rationale: String,
    pub schedule_date: Option<String>,
}

/// A topic rolled up under a goal via _projects.topic_id.
///
/// # sensitivity_tier: 3
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalTopic {
    pub topic_id: String,
    pub title: String,
    #[serde(default)]
    pub importance: i32,
    pub last_activity: Option<String>,
    pub contact_name: Option<String>,
}

/// Progress + today's moves for a single goal.
///
/// # sensitivity_tier: 3
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalProgress {
    pub goal_id: String,
    #[serde(default)]
    pub rolled_up_topics: Vec<GoalTopic>,
    #[serde(default)]
    pub tasks_today: Vec<Task>,
    #[serde(default)]
    pub tasks_open: i32,
    #[serde(default)]
    pub overdue_tasks: i32,
    #[serde(default)]
    pub tasks_done_7d: i32,
    #[serde(default)]
    pub habits_today: Vec<Habit>,
    #[serde(default)]
    pub habit_streak_days: i32,
    pub last_evidence_at: Option<String>,
}

/// A task due/overdue/scheduled today, surfaced in the unified inbox.
///
/// # sensitivity_tier: 2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InboxTask {
    pub id: String,
    pub title: String,
    #[serde(default)]
    pub goal_id: Option<String>,
    #[serde(default)]
    pub goal_title: Option<String>,
    #[serde(default)]
    pub category: Option<String>,
    #[serde(default)]
    pub importance: i32,
    #[serde(default)]
    pub due_at: Option<String>,
    #[serde(default)]
    pub scheduled_for: Option<String>,
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub notes: Option<String>,
    #[serde(default)]
    pub source: Option<String>,
}

/// An active habit due today, surfaced in the unified inbox.
///
/// # sensitivity_tier: 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InboxHabit {
    pub id: String,
    pub title: String,
    pub goal_id: String,
    #[serde(default)]
    pub goal_title: Option<String>,
    #[serde(default)]
    pub category: Option<String>,
    #[serde(default)]
    pub preferred_window: Option<String>,
    #[serde(default)]
    pub cadence: Option<String>,
}

/// Unified inbox: pending replies + tasks + habits, optionally scoped.
///
/// # sensitivity_tier: 3
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Inbox {
    #[serde(default)]
    pub replies: Vec<PendingReply>,
    #[serde(default)]
    pub tasks: Vec<InboxTask>,
    #[serde(default)]
    pub habits: Vec<InboxHabit>,
    #[serde(default)]
    pub topics: Vec<GoalTopic>,
}

/// One goal-anchored move scheduled (or due) today inside a domain.
/// Surfaces in the unified LifeBoard so the day's concrete actions
/// — generated from a user's goals — appear next to that domain's
/// calendar / metrics.
///
/// # sensitivity_tier: 3
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TodayAction {
    pub id: String,
    pub kind: String, // "task" | "habit"
    pub title: String,
    pub goal_id: String,
    pub goal_title: String,
    #[serde(default)]
    pub when: Option<String>,
    #[serde(default)]
    pub preferred_window: Option<String>,
}

/// Today's task progress for a single goal — total scheduled today vs.
/// completed today. Surfaced separately from `today_actions` (which
/// only carries pending work) so the UI can render a fully-filled bar
/// when every today-scheduled task has been completed.
///
/// # sensitivity_tier: 3
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TodayProgress {
    pub total: u32,
    pub done: u32,
}

/// One column of the unified LifeBoard — goals + their tasks/habits
/// for today, plus the existing domain shape (events / metrics).
///
/// # sensitivity_tier: 3
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifeBoardDomain {
    pub domain: String,
    #[serde(default)]
    pub goals: Vec<Goal>,
    #[serde(default)]
    pub today_actions: Vec<TodayAction>,
    #[serde(default)]
    pub today_progress: std::collections::HashMap<String, TodayProgress>,
    #[serde(default)]
    pub items: Vec<DomainItem>,
    #[serde(default)]
    pub open_loops: Vec<DomainOpenLoop>,
}

/// The full LifeBoard — one entry per domain, in canonical
/// work / personal / health order.
///
/// # sensitivity_tier: 3
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifeBoard {
    #[serde(default)]
    pub domains: Vec<LifeBoardDomain>,
}
