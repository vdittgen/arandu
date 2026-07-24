pub mod commands;
pub mod firewall;
pub mod keep_awake;
pub mod ollama_supervisor;
pub mod whatsapp_supervisor;

use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;
use tauri::Manager;

use keep_awake::CaffeinateHandle;
use ollama_supervisor::OllamaSupervisor;
use whatsapp_supervisor::Supervisor;

#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! Welcome to Arandu.", name)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Arandu owns the Ollama server lifecycle via `OllamaSupervisor`. Mark the
    // process environment so child Python (which inherits it) does not spawn a
    // second, un-owned `ollama serve` from `ollama_manager.ensure_running()`.
    // Set before any threads/children are spawned so the write is sound.
    std::env::set_var("ARANDU_OLLAMA_MANAGED", "1");

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_process::init())
        .manage(commands::AppState::new())
        .invoke_handler(tauri::generate_handler![
            greet,
            // Data commands
            commands::get_database_stats,
            commands::get_today_summary,
            commands::get_recent_messages,
            commands::get_upcoming_events,
            commands::get_contacts,
            commands::get_notes,
            commands::get_emails,
            // Table browser commands
            commands::list_tables,
            commands::list_pipeline_models,
            commands::query_table,
            // Graph explorer commands
            commands::graph_summary,
            commands::query_graph_nodes,
            commands::query_graph_relationships,
            // Vector explorer commands
            commands::vector_summary,
            // Chat commands
            commands::ask_brain,
            commands::ask_brain_internal,
            commands::get_chat_history,
            commands::clear_chat_history,
            commands::list_chat_sessions,
            commands::load_chat_session,
            commands::new_chat_session,
            commands::delete_chat_session,
            // Streaming & Ollama commands
            commands::ask_brain_stream,
            commands::stop_research,
            commands::get_ollama_status,
            commands::preload_ollama_model,
            commands::get_model_pull_progress,
            // Monitor commands
            commands::get_memory_usage,
            // Pipeline commands
            commands::get_pipeline_status,
            commands::trigger_pipeline_run,
            commands::trigger_pipeline_run_stream,
            commands::get_pipeline_run_result,
            commands::get_pipeline_run_history,
            commands::cancel_pipeline_run,
            commands::is_pipeline_running,
            // Firewall commands
            commands::get_audit_log,
            commands::verify_audit_chain,
            commands::get_redaction_detail,
            // Settings commands
            commands::get_settings,
            commands::update_settings,
            commands::rebuild_vector_index,
            // Connector commands
            commands::get_connector_catalog,
            commands::get_system_health,
            commands::open_macos_permission_settings,
            commands::toggle_connector,
            commands::sync_connector_now,
            commands::get_connector_details,
            commands::get_whatsapp_listener_status,
            // Extension installer commands
            commands::install_extension_discover,
            commands::install_extension_confirm,
            // Model generator commands
            commands::generate_models,
            commands::approve_models,
            commands::reject_models,
            // Agent runner commands
            commands::list_agents,
            commands::run_agent,
            commands::get_agent_result,
            commands::list_skills,
            // Agents page (Phase 4): Pydantic AI registry
            commands::list_pydantic_agents,
            commands::get_agent_config,
            commands::update_agent_config,
            commands::list_available_models,
            commands::run_agent_eval_proposal,
            commands::reset_agent_config,
            // Agents page (Phase 5b): eval status
            commands::run_agent_eval,
            commands::get_agent_eval_status,
            commands::get_agent_activity,
            // Privacy v2: local-inference opt-in (gated by per-agent evals)
            commands::set_local_inference_for_sensitive,
            // Agents page (Phase 5c): eval datasets + user agents + user skills
            commands::get_agent_eval_dataset,
            commands::upload_user_eval_dataset,
            commands::suggest_eval_dataset,
            commands::suggest_agent_model,
            commands::suggest_prompt_improvements,
            commands::apply_prompt_engineer_edit,
            commands::revert_user_agent_ai_edit,
            commands::create_user_agent,
            commands::update_user_agent,
            commands::delete_user_agent,
            commands::set_user_agent_schedule,
            commands::run_user_agent_now,
            commands::get_user_agent_status,
            commands::list_mcp_action_tools,
            // Skills v2 (SKILL.md-based)
            commands::list_skills_v2,
            commands::get_skill_detail_v2,
            commands::create_skill_v2,
            commands::update_skill_v2,
            commands::delete_skill_v2,
            commands::ask_agent_stream,
            // Data ownership (Settings › Privacy)
            commands::export_user_data,
            commands::delete_all_user_data,
            // Action tool commands
            commands::get_available_actions,
            commands::confirm_action,
            commands::cancel_action,
            commands::resume_action_with_recipient,
            commands::search_recipient_candidates,
            // Extension management commands
            commands::uninstall_extension,
            commands::get_connector_history,
            commands::get_extension_logs,
            // Health check
            commands::health_check,
            // Interest profile
            commands::get_interest_profile,
            commands::get_domain_stats,
            // Pipeline brain
            commands::get_refresh_plan,
            // Insight commands
            commands::get_insights,
            commands::generate_insights,
            commands::dismiss_insight,
            commands::follow_up_insight,
            // Notification commands
            commands::get_notification_preferences,
            commands::update_notification_preference,
            commands::mute_all_notifications,
            commands::get_notification_log,
            // Proactive intelligence commands
            commands::evaluate_proactive,
            commands::get_pending_replies,
            commands::get_contact_contexts,
            commands::get_actionable_events,
            commands::get_meeting_prep_briefs,
            commands::dismiss_pending_reply,
            commands::dismiss_actionable_event,
            // User profile inference
            commands::infer_user_profile,
            // Voice transcription
            commands::transcribe_audio,
            // Learned facts
            commands::get_learned_facts,
            commands::get_facts_for_review,
            commands::get_fact_stats,
            commands::confirm_fact,
            commands::dismiss_fact,
            commands::edit_fact,
            // Background task status
            commands::get_active_tasks,
            // Mission Control dashboard
            commands::get_daily_brief,
            commands::get_active_threads,
            commands::get_agent_stream,
            commands::get_suggested_actions,
            commands::get_domain_summary,
            commands::get_life_board,
            commands::get_today_board,
            commands::get_goal_progress,
            commands::list_inbox,
            // Tasks / Goals / Habits / Schedule
            commands::list_goals,
            commands::create_goal,
            commands::update_goal,
            commands::mine_goals,
            commands::list_projects,
            commands::create_project,
            commands::archive_project,
            commands::list_tasks,
            commands::create_task,
            commands::update_task,
            commands::toggle_task_done,
            commands::delete_task,
            commands::list_habits,
            commands::create_habit,
            commands::toggle_habit,
            commands::delete_habit,
            commands::regenerate_habits,
            commands::get_daily_schedule,
            commands::regenerate_daily_schedule,
            // First-launch setup (bundled .app only)
            commands::setup::get_setup_status,
            commands::setup::run_first_launch_setup,
        ])
        .setup(|app| {
            // Read first-run grace once at startup.  After fresh_restart.sh,
            // this is 60 minutes; in steady state it's absent (0) and the
            // proactive + insight loops fire on the original schedule.
            let grace_minutes: u64 = commands::load_settings_from_disk()
                .ok()
                .and_then(|s| s.first_run_grace_minutes.map(u64::from))
                .unwrap_or(0);
            let grace = std::time::Duration::from_secs(grace_minutes * 60);

            // Kill stale CLI/worker processes from previous app sessions.
            // When the app exits ungracefully (SIGKILL, crash, force quit),
            // kill_on_drop doesn't fire and children become orphans that
            // hold SQLite write locks indefinitely, blocking all new writes.
            // WhatsApp listener and ollama-preload are excluded — listener
            // has its own lifecycle, preload is fast and harmless.
            // Runs before other spawns (1s head start) to avoid killing
            // newly spawned processes.
            tauri::async_runtime::spawn(async {
                let stale_patterns = [
                    "src\\.core\\.cli startup-sync",
                    "src\\.core\\.cli sync-all-stale",
                    "src\\.core\\.cli generate-insights",
                    "src\\.core\\.cli run-scheduled-agents",
                    "src\\.core\\.cli evaluate-proactive",
                    "src\\.pipeline\\.worker run",
                ];
                for pattern in stale_patterns {
                    let _ = tokio::process::Command::new("pkill")
                        .args(["-f", pattern])
                        .stdout(Stdio::null())
                        .stderr(Stdio::null())
                        .status()
                        .await;
                }
                eprintln!("[lib] stale process cleanup done");
            });

            // Ensure DB schemas exist before any IPC poll touches them.
            // SQLite and ChromaDB self-bootstrap on first write, but Kuzu's
            // DDL (Person/Event/Place node tables, etc.) only runs from
            // `cli init` or `cli reset`. On a fresh install — or after the
            // user wipes ~/.arandu/ — the AmbientBar's read-only stats
            // poll would otherwise flash "DB issue" until something opens
            // Kuzu read-write. The CLI is idempotent (CREATE NODE TABLE
            // IF NOT EXISTS), so running it on every launch is cheap.
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                let state = handle.state::<commands::AppState>();
                if let Err(e) = commands::bridge::call_python_cli_with_timeout(
                    &["init"], &state.project_root, 30,
                ).await {
                    eprintln!("[lib] startup init failed (non-fatal): {e}");
                }
            });

            // Preload the Ollama model in the background after a short delay.
            // This eliminates cold-start latency on the first chat message.
            // No write lock needed — does not touch the database.
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                let state = handle.state::<commands::AppState>();
                commands::register_task(&state, "ollama-preload", "Preparing model").await;
                // Generous timeout: the configured model is pulled here if it
                // isn't present yet, and a large model (e.g. llama3.1:70b,
                // ~43 GB) can take a long time to download before it loads.
                let _ = commands::bridge::call_python_cli_with_timeout(
                    &["ollama-preload"], &state.project_root, 3600,
                ).await;
                commands::unregister_task(&state, "ollama-preload").await;
            });

            // Keep-awake: manage the caffeinate process handle.
            let caffeinate = Arc::new(CaffeinateHandle::new());
            app.manage(caffeinate.clone());

            // Boot-time: apply keep-awake settings from disk.
            {
                let boot_settings = app.state::<commands::AppState>()
                    .settings
                    .blocking_lock()
                    .clone();
                if boot_settings.prevent_sleep {
                    let caf = caffeinate.clone();
                    tauri::async_runtime::spawn(async move {
                        keep_awake::apply_caffeinate(
                            &caf,
                            true,
                            boot_settings.prevent_sleep_on_battery,
                        )
                        .await;
                    });
                }
                if boot_settings.launch_at_login {
                    keep_awake::apply_launch_at_login(true);
                }
                if boot_settings.menu_bar_mode {
                    let _ = keep_awake::build_tray(app.handle());
                }
            }

            // WhatsApp listener supervisor — owns the subprocess lifecycle.
            // Kills orphans from prior sessions, spawns the listener when the
            // connector is enabled, monitors it, and respawns on crash. The
            // RunEvent::ExitRequested handler below signals it to terminate
            // the child gracefully on app shutdown.
            let project_root: PathBuf = app.state::<commands::AppState>()
                .project_root
                .clone()
                .into();
            let supervisor = Supervisor::spawn(project_root);
            app.manage(supervisor);

            // Ollama server supervisor — ties Ollama's lifecycle to Arandu.
            // Takes over any pre-existing/orphaned `ollama serve`, spawns a
            // Arandu-owned one, and respawns it on crash. The
            // RunEvent::ExitRequested handler below reaps it on app shutdown.
            let ollama = OllamaSupervisor::spawn();
            app.manage(ollama);

            // Startup sync: sync stale connectors and refresh pipeline.
            // Acquires the write lock to prevent concurrent DB writers.
            let handle2 = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                let state = handle2.state::<commands::AppState>();
                let _guard = state.cli_write_lock.lock().await;
                commands::register_task(&state, "startup-sync", "Syncing data").await;
                eprintln!("[lib] starting startup-sync (write lock acquired)");
                match commands::bridge::call_python_cli_with_timeout(
                    &["startup-sync"], &state.project_root, 600,
                ).await {
                    Ok(_) => {
                        eprintln!("[lib] startup-sync completed");
                        // Auto-infer user profile if not yet populated.
                        let needs_inference = {
                            let s = state.settings.lock().await;
                            s.user_name.is_none()
                        };
                        if needs_inference {
                            commands::unregister_task(&state, "startup-sync").await;
                            commands::register_task(&state, "profile-inference", "Inferring profile").await;
                            eprintln!("[lib] inferring user profile");
                            match commands::bridge::call_python_cli_with_timeout(
                                &["infer-profile"], &state.project_root, 60,
                            ).await {
                                Ok(out) => {
                                    eprintln!("[lib] profile inference done: {}", &out[..out.len().min(200)]);
                                    if let Ok(Ok(s)) = tokio::task::spawn_blocking(
                                        commands::load_settings_from_disk,
                                    ).await {
                                        let mut current = state.settings.lock().await;
                                        *current = s;
                                    }
                                }
                                Err(e) => eprintln!("[lib] profile inference failed: {e}"),
                            }
                            commands::unregister_task(&state, "profile-inference").await;
                        }
                    }
                    Err(e) => eprintln!("[lib] startup-sync failed: {e}"),
                }
                commands::unregister_task(&state, "startup-sync").await;
                drop(_guard);
            });

            // Periodic sync: sync all enabled connectors every 15 minutes.
            // Acquires write lock; skips cycle if pipeline worker is running.
            let handle3 = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                let interval = std::time::Duration::from_secs(900);
                loop {
                    tokio::time::sleep(interval).await;
                    let state = handle3.state::<commands::AppState>();
                    if commands::is_pipeline_flag_set(&state).await {
                        eprintln!("[lib] skipping sync-all-stale: pipeline running");
                        continue;
                    }
                    let _guard = state.cli_write_lock.lock().await;
                    commands::register_task(&state, "periodic-sync", "Syncing connectors").await;
                    eprintln!("[lib] starting periodic sync-all-stale");
                    match commands::bridge::call_python_cli_with_timeout(
                        &["sync-all-stale"], &state.project_root, 600,
                    ).await {
                        Ok(_) => eprintln!("[lib] sync-all-stale completed"),
                        Err(e) => eprintln!("[lib] sync-all-stale failed: {e}"),
                    }
                    commands::unregister_task(&state, "periodic-sync").await;
                    drop(_guard);
                }
            });

            // Background insight generation: run every 4 hours.
            // First run after 5 min (avoid colliding with startup-sync LLM calls),
            // plus any first_run_grace_minutes set in settings.json.
            let handle4 = app.handle().clone();
            let insights_grace = grace;
            tauri::async_runtime::spawn(async move {
                tokio::time::sleep(
                    std::time::Duration::from_secs(300) + insights_grace,
                ).await;
                let interval = std::time::Duration::from_secs(4 * 3600);
                loop {
                    let state = handle4.state::<commands::AppState>();
                    if commands::is_pipeline_flag_set(&state).await {
                        eprintln!("[lib] skipping generate-insights: pipeline running");
                    } else {
                        let _guard = state.cli_write_lock.lock().await;
                        commands::register_task(&state, "generate-insights", "Generating insights").await;
                        let _ = commands::bridge::call_python_cli_with_timeout(
                            &["generate-insights"], &state.project_root, 300,
                        ).await;
                        commands::unregister_task(&state, "generate-insights").await;
                        drop(_guard);
                    }
                    tokio::time::sleep(interval).await;
                }
            });

            // Scheduled agent runner: check cron schedules every 60 seconds.
            // Uses try_lock to skip if another write-mode task is running.
            let handle5 = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_secs(60)).await;
                let interval = std::time::Duration::from_secs(60);
                loop {
                    let state = handle5.state::<commands::AppState>();
                    if let Ok(_guard) = state.cli_write_lock.try_lock() {
                        commands::register_task(&state, "scheduled-agents", "Running agents").await;
                        let _ = commands::bridge::call_python_cli_with_timeout(
                            &["run-scheduled-agents"], &state.project_root, 300,
                        ).await;
                        commands::unregister_task(&state, "scheduled-agents").await;
                        drop(_guard);
                    } else {
                        eprintln!("[lib] skipping run-scheduled-agents: write lock busy");
                    }
                    tokio::time::sleep(interval).await;
                }
            });

            // Proactive intelligence: evaluate every 2 hours.
            // First run after 7 min (after insights at +5min finishes), plus
            // any first_run_grace_minutes set in settings.json.
            // Uses try_lock — skips if another LLM-using task is running
            // to avoid Ollama lock contention on single-GPU hardware.
            let handle6 = app.handle().clone();
            let proactive_grace = grace;
            tauri::async_runtime::spawn(async move {
                tokio::time::sleep(
                    std::time::Duration::from_secs(420) + proactive_grace,
                ).await;
                let interval = std::time::Duration::from_secs(2 * 3600);
                loop {
                    let state = handle6.state::<commands::AppState>();
                    if commands::is_pipeline_flag_set(&state).await {
                        eprintln!("[lib] skipping proactive-eval: pipeline running");
                    } else if let Ok(_guard) = state.cli_write_lock.try_lock() {
                        commands::register_task(&state, "proactive-eval", "Proactive evaluation").await;
                        eprintln!("[lib] starting proactive intelligence evaluation");
                        match commands::bridge::call_python_cli_with_timeout(
                            &["evaluate-proactive"], &state.project_root, 900,
                        ).await {
                            Ok(_) => eprintln!("[lib] proactive eval completed"),
                            Err(e) => eprintln!("[lib] proactive eval failed: {e}"),
                        }
                        commands::unregister_task(&state, "proactive-eval").await;
                        drop(_guard);
                    } else {
                        eprintln!("[lib] skipping proactive-eval: write lock busy");
                    }
                    tokio::time::sleep(interval).await;
                }
            });

            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                let app = window.app_handle();
                let menu_bar_mode = app
                    .try_state::<commands::AppState>()
                    .map(|s| s.settings.blocking_lock().menu_bar_mode)
                    .unwrap_or(false);
                if menu_bar_mode {
                    api.prevent_close();
                    let _ = window.hide();
                }
            }
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app_handle, event| {
            if let tauri::RunEvent::ExitRequested { .. } = &event {
                // Stop caffeinate on exit.
                if let Some(caf) = app_handle.try_state::<Arc<CaffeinateHandle>>() {
                    let caf = caf.inner().clone();
                    tauri::async_runtime::block_on(async move {
                        keep_awake::stop_caffeinate(&caf).await;
                    });
                }
                // Terminate the WhatsApp listener cleanly so Baileys can
                // flush state and release its fcntl lock before the app
                // process exits.
                if let Some(supervisor) = app_handle.try_state::<Arc<Supervisor>>() {
                    let sup = supervisor.inner().clone();
                    tauri::async_runtime::block_on(async move {
                        sup.shutdown(Duration::from_secs(6)).await;
                    });
                }
                // Reap the Ollama server so it doesn't outlive Arandu.
                if let Some(ollama) = app_handle.try_state::<Arc<OllamaSupervisor>>() {
                    let sup = ollama.inner().clone();
                    tauri::async_runtime::block_on(async move {
                        sup.shutdown(Duration::from_secs(6)).await;
                    });
                }
                // Terminate any in-flight pipeline worker.  kill_on_drop
                // only covers handle drops inside a live runtime; without
                // this, quitting mid-run strands a Python worker holding
                // the SQLite write lock until the next-launch orphan sweep.
                if let Some(state) = app_handle.try_state::<commands::AppState>() {
                    tauri::async_runtime::block_on(async {
                        commands::reap_orphan_worker(state.inner()).await;
                    });
                }
            }
        });
}
