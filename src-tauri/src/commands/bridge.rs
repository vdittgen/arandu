use std::path::{Path, PathBuf};
use std::process::Stdio;

use tauri::{AppHandle, Emitter};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};

/// True if the executable lives inside a `.app/Contents/MacOS/` directory.
///
/// Used to distinguish a packaged install (icon double-click, Spotlight, Dock)
/// from `cargo tauri dev` running out of the repo's `target/`.
pub fn is_bundled() -> bool {
    std::env::current_exe()
        .ok()
        .map(|p| p.to_string_lossy().contains(".app/Contents/MacOS/"))
        .unwrap_or(false)
}

/// Resources directory of the running `.app` bundle, e.g.
/// `/Applications/Arandu.app/Contents/Resources/`. None in dev mode.
fn bundle_resources_dir() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let macos = exe.parent()?; // .../Contents/MacOS
    let contents = macos.parent()?; // .../Contents
    Some(contents.join("Resources"))
}

/// Path to the bundled python-build-standalone runtime inside the .app.
/// Used by the first-launch setup to create the user's venv.
pub fn bundled_python_runtime() -> Option<PathBuf> {
    bundle_resources_dir().map(|r| r.join("python_runtime").join("bin").join("python3"))
}

/// Path to the bundled app dir (contains src/ and pyproject.toml).
/// Used as the install source for `pip install <path>` at first launch.
/// Must match the `resources` key in `tauri.conf.json`.
pub fn bundled_app_dir() -> Option<PathBuf> {
    bundle_resources_dir().map(|r| r.join("arandu_app"))
}

/// Path to the user-persistent venv at `~/.arandu/venv/`.
pub fn user_venv_dir() -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join(".arandu").join("venv"))
}

/// Path to the python interpreter inside the user-persistent venv.
pub fn user_venv_python() -> Option<PathBuf> {
    user_venv_dir().map(|v| v.join("bin").join("python3"))
}

/// True if the user's venv exists AND the setup-complete marker matches
/// the Python payload this bundle would install.
///
/// The marker is written by `setup_venv` only after `pip install`
/// succeeds, so a partially-created venv from a crashed setup won't be
/// falsely accepted. Comparing the content — not just existence — makes
/// app updates rebuild the venv: the updater replaces the .app, but the
/// Python code lives in `~/.arandu/venv`, and without this an updated
/// install would keep executing the previous build's Python forever.
pub fn is_venv_ready() -> bool {
    let py = match user_venv_python() {
        Some(p) => p,
        None => return false,
    };
    let marker = match user_venv_dir() {
        Some(d) => d.join(".arandu_setup_complete"),
        None => return false,
    };
    if !py.exists() {
        return false;
    }
    match std::fs::read_to_string(&marker) {
        Ok(contents) => marker_is_current(&contents),
        Err(_) => false,
    }
}

/// Whether a setup-complete marker matches what this bundle would
/// install. Stale, empty, and unreadable markers all mean the venv must
/// be rebuilt; `setup_venv` wipes and recreates it.
fn marker_is_current(contents: &str) -> bool {
    contents.trim() == expected_marker()
}

/// The value a freshly-completed setup should record in its marker.
///
/// Prefers a fingerprint of the bundled Python payload, falling back to
/// the app version when nothing is bundled (dev mode, or an unreadable
/// Resources dir).
pub fn expected_marker() -> String {
    payload_fingerprint().unwrap_or_else(|| env!("CARGO_PKG_VERSION").to_string())
}

/// Fingerprint of the Python payload this bundle installs into the venv.
///
/// The marker used to hold the app *version*, which misses the case it
/// exists to catch: Python changing **within** a version. A rebuild that
/// fixes Python but keeps the version number leaves every existing
/// install running the old code — setup sees a matching marker, skips,
/// and the stale venv survives reinstalls indefinitely. Nothing surfaces
/// it, because the Rust and UI halves *do* update.
///
/// Hashes each payload file's path, size and mtime rather than its
/// contents: equally sensitive to a payload that actually changed, at a
/// fraction of the cost (~2.8k files, tens of milliseconds, once per
/// launch, gating a multi-minute rebuild).
///
/// Because mtimes move when a bundle is copied, reinstalling the *same*
/// build also rebuilds the venv. That is the deliberate direction to err
/// in: a redundant rebuild costs minutes, while a missed one leaves the
/// app silently running last build's Python.
///
/// `python_runtime/` is skipped — it is large, static, and only moves
/// with a release, which the version component already covers. The app
/// version is mixed in so a version bump alone still invalidates.
///
/// Returns `None` when nothing is bundled (dev mode).
fn payload_fingerprint() -> Option<String> {
    use sha2::{Digest, Sha256};

    let resources = bundle_resources_dir()?;
    let mut entries: Vec<String> = Vec::new();
    collect_payload_entries(&resources, &resources, &mut entries).ok()?;
    if entries.is_empty() {
        return None;
    }
    // Directory order is not guaranteed across filesystems; sort so the
    // same payload always hashes identically.
    entries.sort_unstable();

    let mut hasher = Sha256::new();
    hasher.update(env!("CARGO_PKG_VERSION").as_bytes());
    for entry in &entries {
        hasher.update(entry.as_bytes());
        hasher.update(b"\n");
    }
    Some(format!("{:x}", hasher.finalize()))
}

/// Recursively record `relative_path:size:mtime` for every payload file.
fn collect_payload_entries(
    root: &Path,
    dir: &Path,
    out: &mut Vec<String>,
) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        // The interpreter is deliberately out of scope — see
        // `payload_fingerprint`.
        if path.file_name().is_some_and(|n| n == "python_runtime") {
            continue;
        }
        let meta = entry.metadata()?;
        if meta.is_dir() {
            collect_payload_entries(root, &path, out)?;
        } else if meta.is_file() {
            let rel = path.strip_prefix(root).unwrap_or(&path);
            let mtime = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            out.push(format!("{}:{}:{}", rel.display(), meta.len(), mtime));
        }
    }
    Ok(())
}

/// Resolve the Python executable.
///
/// Bundled mode: returns the user's venv python at `~/.arandu/venv/bin/python3`.
/// Dev mode: prefers `<project_root>/.venv/bin/python3`, falls back to `python3`.
///
/// # sensitivity_tier: N/A
pub(crate) fn resolve_python(project_root: &str) -> String {
    if is_bundled() {
        if let Some(p) = user_venv_python() {
            if p.exists() {
                return p.to_string_lossy().to_string();
            }
        }
        // Fallback to bundled runtime if user venv is missing — caller will
        // typically have gated on is_venv_ready, so this should not be hit
        // in normal operation.
        if let Some(p) = bundled_python_runtime() {
            if p.exists() {
                return p.to_string_lossy().to_string();
            }
        }
    }
    let venv_python = Path::new(project_root)
        .join(".venv")
        .join("bin")
        .join("python3");
    if venv_python.exists() {
        return venv_python.to_string_lossy().to_string();
    }
    "python3".to_string()
}

/// Execute a Python CLI command asynchronously and return its JSON stdout output.
///
/// Runs `python3 -m src.core.cli <args>` in the project root directory
/// on a background thread, freeing the Tauri main/UI thread.
/// Returns the raw stdout string on success, or an error message on failure.
///
/// # sensitivity_tier: varies (depends on the CLI command invoked)
pub async fn call_python_cli(args: &[&str], project_root: &str) -> Result<String, String> {
    let label = args.first().unwrap_or(&"unknown").to_string();
    let python = resolve_python(project_root);
    eprintln!("[bridge] call_python_cli '{}' starting", label);
    let child = Command::new(&python)
        .arg("-m")
        .arg("src.core.cli")
        .args(args)
        .current_dir(project_root)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| format!("Failed to spawn {python}: {e}"))?;

    let output = child
        .wait_with_output()
        .await
        .map_err(|e| format!("Failed to wait for {python}: {e}"))?;

    eprintln!(
        "[bridge] call_python_cli '{}' exited with code {}",
        label,
        output.status.code().unwrap_or(-1),
    );

    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        Ok(stdout.trim().to_string())
    } else {
        // Some commands write error JSON to stdout (not stderr) so the
        // Rust side can parse structured error messages.  Try stdout first.
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !stdout.is_empty() && stdout.starts_with('{') {
            // If the JSON has an "error" key, extract it and return as Err
            // so the Tauri command handler propagates it to the frontend.
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&stdout) {
                if let Some(error_msg) = json.get("error").and_then(|e| e.as_str()) {
                    return Err(error_msg.to_string());
                }
            }
            // Otherwise it's valid structured output — return as success.
            return Ok(stdout);
        }
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        Err(format!(
            "Python CLI exited with code {}: {}",
            output.status.code().unwrap_or(-1),
            stderr.trim()
        ))
    }
}

/// Execute a Python CLI command with a timeout.
///
/// Wraps `call_python_cli` with `tokio::time::timeout`. If the timeout
/// fires, the spawned child is killed automatically via `kill_on_drop`.
/// Used for background tasks that should not block indefinitely.
///
/// # sensitivity_tier: varies
pub async fn call_python_cli_with_timeout(
    args: &[&str],
    project_root: &str,
    timeout_secs: u64,
) -> Result<String, String> {
    match tokio::time::timeout(
        std::time::Duration::from_secs(timeout_secs),
        call_python_cli(args, project_root),
    )
    .await
    {
        Ok(result) => result,
        Err(_) => {
            let label = args.first().unwrap_or(&"unknown");
            eprintln!(
                "[bridge] call_python_cli '{}' TIMED OUT after {}s",
                label, timeout_secs
            );
            Err(format!(
                "CLI command '{}' timed out after {}s",
                label, timeout_secs
            ))
        }
    }
}

/// Spawn a Python CLI command as a detached background process.
///
/// A background thread calls `wait()` on the child to reap it when it
/// exits, preventing zombie processes.  Zombies retain their PID entry
/// which causes DuckDB to believe the write lock is still held.
///
/// # sensitivity_tier: varies
pub fn spawn_background_cli(args: &[&str], project_root: &str) -> Result<(), String> {
    let python = resolve_python(project_root);
    let label = args.first().unwrap_or(&"unknown").to_string();
    let mut child = std::process::Command::new(&python)
        .arg("-m")
        .arg("src.core.cli")
        .args(args)
        .current_dir(project_root)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("Failed to spawn background CLI: {e}"))?;

    eprintln!("[bridge] spawned background '{}' (pid {})", label, child.id());

    // Reap the child in a background thread to prevent zombie processes.
    std::thread::spawn(move || {
        match child.wait() {
            Ok(status) => eprintln!(
                "[bridge] background '{}' exited with {}",
                label,
                status.code().unwrap_or(-1),
            ),
            Err(e) => eprintln!(
                "[bridge] background '{}' wait error: {}",
                label, e,
            ),
        }
    });

    Ok(())
}

/// Execute a Python CLI command and stream JSON lines as Tauri events.
///
/// Spawns the subprocess with piped stdout, reads lines as they arrive,
/// parses each line as JSON, and emits a Tauri event for each parsed value.
/// Non-JSON lines (e.g. Python logging) are silently skipped.
///
/// # sensitivity_tier: varies (depends on the CLI command invoked)
pub async fn call_python_cli_stream(
    args: &[&str],
    project_root: &str,
    app_handle: &AppHandle,
    event_name: &str,
) -> Result<(), String> {
    call_python_cli_stream_with_observer(
        args,
        project_root,
        app_handle,
        event_name,
        |_| {},
    )
    .await
}

/// Same as [`call_python_cli_stream`] but invokes `observer` for each
/// parsed JSON chunk before emitting it. Lets the caller collect
/// stream state (e.g. assistant message parts) without re-parsing the
/// event stream on the frontend.
///
/// # sensitivity_tier: varies
pub async fn call_python_cli_stream_with_observer<F>(
    args: &[&str],
    project_root: &str,
    app_handle: &AppHandle,
    event_name: &str,
    mut observer: F,
) -> Result<(), String>
where
    F: FnMut(&serde_json::Value),
{
    let python = resolve_python(project_root);
    let mut child = Command::new(&python)
        .arg("-m")
        .arg("src.core.cli")
        .args(args)
        .current_dir(project_root)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| format!("Failed to spawn python3: {e}"))?;

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "Failed to capture stdout".to_string())?;

    let mut reader = BufReader::new(stdout).lines();

    while let Some(line) = reader
        .next_line()
        .await
        .map_err(|e| format!("IO error reading stdout: {e}"))?
    {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        // Parse JSON — skip non-JSON lines (Python logging output)
        if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(trimmed) {
            observer(&json_val);
            let _ = app_handle.emit(event_name, json_val);
        }
    }

    let status = child
        .wait()
        .await
        .map_err(|e| format!("Failed to wait for process: {e}"))?;

    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "Python CLI stream exited with code {}",
            status.code().unwrap_or(-1)
        ))
    }
}

/// Spawn the pipeline worker as an isolated subprocess with lowered CPU priority.
///
/// On macOS/Linux, uses `nice -n 10` to give the UI and LLM higher CPU priority.
/// Returns the `Child` handle so the caller can store it for cancellation.
/// A background tokio task reads stdout JSON lines and emits them as Tauri events.
///
/// # sensitivity_tier: 1
pub async fn spawn_pipeline_worker(
    trigger: &str,
    mode: &str,
    project_root: &str,
    app_handle: &AppHandle,
    event_name: &str,
) -> Result<Child, String> {
    let python = resolve_python(project_root);
    let mut child = if cfg!(unix) {
        Command::new("nice")
            .args([
                "-n",
                "10",
                &python,
                "-m",
                "src.pipeline.worker",
                "run",
                "--trigger",
                trigger,
                "--mode",
                mode,
            ])
            .current_dir(project_root)
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| format!("Failed to spawn pipeline worker: {e}"))?
    } else {
        Command::new(&python)
            .args(["-m", "src.pipeline.worker", "run", "--trigger", trigger, "--mode", mode])
            .current_dir(project_root)
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| format!("Failed to spawn pipeline worker: {e}"))?
    };

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "Failed to capture worker stdout".to_string())?;

    // Spawn a reader task that emits each JSON line as a Tauri event.
    let handle = app_handle.clone();
    let ename = event_name.to_string();
    tokio::spawn(async move {
        let mut reader = BufReader::new(stdout).lines();
        while let Ok(Some(line)) = reader.next_line().await {
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(trimmed) {
                    let _ = handle.emit(&ename, json_val);
                }
            }
        }
    });

    Ok(child)
}

/// Resolve the project root directory for the Python bridge.
///
/// Bundled mode: returns the .app's `Contents/Resources/arandu_app/`
/// (contains src/ and pyproject.toml shipped inside the bundle).
/// Dev mode: walks up from the exe to find the repo containing src-tauri/ + src/.
/// Falls back to the current working directory.
///
/// # sensitivity_tier: N/A
pub fn resolve_project_root() -> String {
    if is_bundled() {
        if let Some(p) = bundled_app_dir() {
            if p.is_dir() {
                return p.to_string_lossy().to_string();
            }
        }
    }

    // In Tauri dev, the binary runs from src-tauri/
    // The project root (with src/core/cli.py) is one level up
    if let Ok(exe_path) = std::env::current_exe() {
        // During development, the exe is deep in target/debug/
        // Walk up to find the directory containing src-tauri/
        let mut dir = exe_path.as_path();
        for _ in 0..10 {
            if let Some(parent) = dir.parent() {
                if parent.join("src-tauri").is_dir() && parent.join("src").is_dir() {
                    return parent.to_string_lossy().to_string();
                }
                dir = parent;
            }
        }
    }

    // Fallback: try current working directory
    if let Ok(cwd) = std::env::current_dir() {
        // Check if we're in src-tauri/
        if cwd.join("src").join("core").join("cli.py").exists() {
            return cwd.to_string_lossy().to_string();
        }
        if let Some(parent) = cwd.parent() {
            if parent.join("src").join("core").join("cli.py").exists() {
                return parent.to_string_lossy().to_string();
            }
        }
    }

    // Last resort: assume current directory
    std::env::current_dir()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| ".".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_project_root_returns_string() {
        let root = resolve_project_root();
        assert!(!root.is_empty());
    }

    #[test]
    fn test_marker_matching_expected_is_current() {
        assert!(marker_is_current(&expected_marker()));
        // Tolerate a trailing newline from manual writes.
        assert!(marker_is_current(&format!("{}\n", expected_marker())));
    }

    #[test]
    fn test_marker_from_another_build_is_stale() {
        // A venv built by a previous (or future) build must read as
        // not-ready so updates rebuild it.
        assert!(!marker_is_current("0.0.1-previous"));
        assert!(!marker_is_current(""));
        assert!(!marker_is_current("   "));
    }

    #[test]
    fn test_expected_marker_falls_back_to_version_unbundled() {
        // Nothing is bundled under `cargo test`, so the fallback is what
        // dev builds record — and it must be stable, not empty.
        assert_eq!(expected_marker(), env!("CARGO_PKG_VERSION"));
    }

    /// Scratch dir helper — avoids taking a `tempfile` dev-dependency
    /// for three tests.
    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir()
            .join(format!("arandu-fp-{}-{}", tag, std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        dir
    }

    #[test]
    fn test_payload_entries_track_content_changes() {
        // The regression this whole mechanism exists for: Python changed
        // but the app version did not. Keying the marker on the version
        // alone left every existing install running the old code, since
        // setup saw a matching marker and skipped. A changed payload must
        // produce a different manifest even with the version untouched.
        let root = scratch("content");
        std::fs::write(root.join("mod.py"), "print('v1')").unwrap();

        let mut before = Vec::new();
        collect_payload_entries(&root, &root, &mut before).unwrap();

        // Longer body => different size => different entry.
        std::fs::write(root.join("mod.py"), "print('v2 is longer')").unwrap();
        let mut after = Vec::new();
        collect_payload_entries(&root, &root, &mut after).unwrap();

        assert_eq!(before.len(), 1);
        assert_ne!(before, after, "a changed payload must change the manifest");

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn test_payload_entries_recurse_and_skip_python_runtime() {
        let root = scratch("skip");
        std::fs::create_dir_all(root.join("arandu_app/src")).unwrap();
        std::fs::write(root.join("arandu_app/src/deep.py"), "x").unwrap();
        // The interpreter is huge, static, and versioned with releases —
        // walking it every launch buys nothing.
        std::fs::create_dir_all(root.join("python_runtime/bin")).unwrap();
        std::fs::write(root.join("python_runtime/bin/python3"), "elf").unwrap();

        let mut entries = Vec::new();
        collect_payload_entries(&root, &root, &mut entries).unwrap();

        assert_eq!(entries.len(), 1, "expected only the nested payload file");
        assert!(entries[0].starts_with("arandu_app/src/deep.py:"));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn test_payload_entries_are_relative_and_sortable() {
        // Paths must be bundle-relative: an absolute path would embed the
        // install location, so the same build would fingerprint
        // differently from /Applications vs a test copy.
        let root = scratch("relative");
        std::fs::write(root.join("b.py"), "b").unwrap();
        std::fs::write(root.join("a.py"), "a").unwrap();

        let mut entries = Vec::new();
        collect_payload_entries(&root, &root, &mut entries).unwrap();
        entries.sort_unstable();

        assert_eq!(entries.len(), 2);
        assert!(entries[0].starts_with("a.py:"), "got {:?}", entries[0]);
        assert!(entries[1].starts_with("b.py:"), "got {:?}", entries[1]);
        assert!(
            !entries.iter().any(|e| e.contains(root.to_str().unwrap())),
            "entries must not embed the absolute install path",
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn test_call_python_cli_invalid_command() {
        // This should fail because there's no such CLI subcommand
        let result = call_python_cli(&["nonexistent-command"], ".").await;
        assert!(result.is_err());
    }
}
