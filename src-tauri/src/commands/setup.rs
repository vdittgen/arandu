//! First-launch setup for the installed `.app` bundle.
//!
//! When the app runs from a packaged `.app` bundle (vs `cargo tauri dev`),
//! the Python source ships inside `Contents/Resources/arandu_app/` and the
//! relocatable Python interpreter ships at `Contents/Resources/python_runtime/`.
//! On first launch we use the bundled interpreter to create a user-persistent
//! venv at `~/.arandu/venv/` and pip-install the bundled app + deps. After
//! that, all Python invocations go through `~/.arandu/venv/bin/python3`.
//!
//! Subsequent launches detect the marker file and skip setup entirely.
//!
//! sensitivity_tier: 1

use std::process::Stdio;

use serde::Serialize;
use tauri::{AppHandle, Emitter};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

use super::bridge::{
    bundled_app_dir, bundled_python_runtime, expected_marker, is_bundled,
    is_venv_ready, user_venv_dir, user_venv_python,
};

#[derive(Serialize, Clone)]
pub struct SetupStatus {
    pub is_bundled: bool,
    pub venv_ready: bool,
    pub needs_setup: bool,
}

#[tauri::command]
pub fn get_setup_status() -> SetupStatus {
    let bundled = is_bundled();
    let ready = is_venv_ready();
    SetupStatus {
        is_bundled: bundled,
        venv_ready: ready,
        needs_setup: bundled && !ready,
    }
}

#[derive(Serialize, Clone)]
struct SetupProgressPayload {
    stage: String,
    message: String,
    done: bool,
    error: Option<String>,
}

fn emit(app: &AppHandle, stage: &str, message: &str) {
    let _ = app.emit(
        "setup-progress",
        SetupProgressPayload {
            stage: stage.to_string(),
            message: message.to_string(),
            done: false,
            error: None,
        },
    );
}

fn emit_done(app: &AppHandle, message: &str) {
    let _ = app.emit(
        "setup-progress",
        SetupProgressPayload {
            stage: "complete".to_string(),
            message: message.to_string(),
            done: true,
            error: None,
        },
    );
}

fn emit_error(app: &AppHandle, error: &str) {
    let _ = app.emit(
        "setup-progress",
        SetupProgressPayload {
            stage: "error".to_string(),
            message: error.to_string(),
            done: true,
            error: Some(error.to_string()),
        },
    );
}

/// Create `~/.arandu/venv/` using the bundled python-build-standalone runtime,
/// then `pip install` the bundled arandu app (which pulls in pyproject.toml
/// dependencies). Emits `setup-progress` events line-by-line.
#[tauri::command]
pub async fn run_first_launch_setup(app: AppHandle) -> Result<(), String> {
    if !is_bundled() {
        return Err("first-launch setup is only valid in bundled mode".to_string());
    }
    if is_venv_ready() {
        emit_done(&app, "Already set up.");
        return Ok(());
    }

    let bundled_py = bundled_python_runtime()
        .ok_or_else(|| "bundled python runtime not found".to_string())?;
    let app_dir =
        bundled_app_dir().ok_or_else(|| "bundled app dir not found".to_string())?;
    let venv_dir = user_venv_dir()
        .ok_or_else(|| "could not resolve ~/.arandu/venv".to_string())?;
    let venv_py = user_venv_python().expect("user_venv_python paired with user_venv_dir");
    let venv_pip = venv_dir.join("bin").join("pip");

    if let Some(parent) = venv_dir.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("create ~/.arandu: {e}"))?;
    }

    if venv_dir.exists() {
        emit(&app, "preparing", "Cleaning up previous setup attempt…");
        std::fs::remove_dir_all(&venv_dir)
            .map_err(|e| format!("wipe stale venv: {e}"))?;
    }

    emit(
        &app,
        "creating-venv",
        "Creating Python environment (~30s)…",
    );
    let status = Command::new(&bundled_py)
        .args([
            "-m",
            "venv",
            venv_dir.to_str().expect("venv path utf-8"),
        ])
        .status()
        .await
        .map_err(|e| {
            let msg = format!("venv create spawn failed: {e}");
            emit_error(&app, &msg);
            msg
        })?;
    if !status.success() {
        let msg = format!("venv create failed: exit {status}");
        emit_error(&app, &msg);
        return Err(msg);
    }

    emit(&app, "installing-deps", "Updating pip…");
    let _ = Command::new(&venv_py)
        .args(["-m", "pip", "install", "--upgrade", "pip", "--quiet"])
        .status()
        .await;

    emit(
        &app,
        "installing-deps",
        "Installing dependencies (this may take 1–2 minutes)…",
    );

    let mut child = Command::new(&venv_pip)
        .args([
            "install",
            "--disable-pip-version-check",
            app_dir.to_str().expect("app dir utf-8"),
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| {
            let msg = format!("pip spawn failed: {e}");
            emit_error(&app, &msg);
            msg
        })?;

    let stdout = child.stdout.take().expect("piped stdout");
    let stderr = child.stderr.take().expect("piped stderr");

    let app_out = app.clone();
    let stdout_task = tokio::spawn(async move {
        let mut lines = BufReader::new(stdout).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            emit(&app_out, "installing-deps", &line);
        }
    });

    let app_err = app.clone();
    let stderr_task = tokio::spawn(async move {
        let mut lines = BufReader::new(stderr).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            // pip routes "Collecting X" / "Building wheel" to stderr in many configs
            emit(&app_err, "installing-deps", &line);
        }
    });

    let status = child.wait().await.map_err(|e| {
        let msg = format!("pip wait failed: {e}");
        emit_error(&app, &msg);
        msg
    })?;
    let _ = stdout_task.await;
    let _ = stderr_task.await;

    if !status.success() {
        let msg = format!("pip install failed: exit {status}");
        emit_error(&app, &msg);
        return Err(msg);
    }

    // Record what was actually installed, not just the version — see
    // `bridge::expected_marker`. Written last, so a crashed setup leaves
    // no marker and the next launch retries.
    let marker = venv_dir.join(".arandu_setup_complete");
    std::fs::write(&marker, expected_marker())
        .map_err(|e| format!("write setup marker: {e}"))?;

    emit_done(&app, "Setup complete.");
    Ok(())
}
