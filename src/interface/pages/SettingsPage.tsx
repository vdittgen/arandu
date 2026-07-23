import { useState, useCallback, useEffect, type ReactNode } from "react";
import { invoke } from "@tauri-apps/api/core";
import { getVersion } from "@tauri-apps/api/app";
import {
  Cpu,
  Shield,
  Info,
  RefreshCw,
  AlertTriangle,
  Trash2,
  Download,
  Bell,
  Brain,
  Mic,
  Palette,
  Power,
  Sun,
  Moon,
  ChevronDown,
  type LucideIcon,
} from "lucide-react";
import { SkeletonSection } from "../components/LoadingState";
import LocalInferenceEnableModal from "../components/LocalInferenceEnableModal";
import { useAsyncData } from "../hooks/useAsyncData";
import { dedupInvoke } from "../utils/requestDedup";
import { broadcastThemeChange } from "../hooks/useTheme";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface AppSettings {
  llm_model: string;
  llm_host: string;
  llm_provider: string;
  anthropic_api_key: string | null;
  anthropic_model: string;
  external_llm_consent: boolean;
  llm_api_key: string | null;
  llm_max_parallel: number;
  max_sensitivity_tier: number;
  theme: string;
  data_dir: string;
  auto_refresh_enabled: boolean;
  auto_refresh_interval_minutes: number;
  refresh_on_launch: boolean;
  onboarding_completed: boolean;
  onboarding_completed_at: string | null;
  initial_connectors: string[];
  skipped_connectors: string[];
  ollama_configured: boolean;
  dismissed_nudges: string[];
  interest_overrides: Record<string, number>;
  notifications_enabled: boolean;
  whatsapp_notification_phone: string | null;
  user_name: string | null;
  user_birthday: string | null;
  user_location: string | null;
  user_timezone: string | null;
  user_language: string | null;
  user_bio: string | null;
  voice_transcription_enabled: boolean;
  whisper_model_size: string;
  transcribe_whatsapp_audio: boolean;
  local_inference_for_sensitive: boolean;
  prevent_sleep: boolean;
  prevent_sleep_on_battery: boolean;
  launch_at_login: boolean;
  menu_bar_mode: boolean;
}

interface NotificationPreference {
  readonly category: string;
  readonly enabled: boolean;
  readonly muted_until: string | null;
  readonly created_at: string;
  readonly updated_at: string;
}

interface PipelineRunHistoryEntry {
  run_id: string;
  started_at: string;
  completed_at: string;
  duration_seconds: number;
  status: string;
  trigger: string;
  models_processed: string[];
  rows_processed: Record<string, number>;
  error: string | null;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const CONSENT_LEVELS = [
  { value: 1, label: "Low (Tier 1)", description: "Agents may auto-access general info (preferences, interests)" },
  { value: 2, label: "Medium (Tier 1 + 2)", description: "Agents may also auto-access personal info (habits, routines, contacts)" },
  { value: 3, label: "Full (all tiers)", description: "Agents may auto-access all data including health, finances, and emotions" },
] as const;

const TECH_STACK = [
  "SQLite",
  "Kuzu",
  "ChromaDB",
  "SQLMesh",
  "Ollama",
  "Tauri",
] as const;

const WHISPER_MODEL_SIZES = [
  { id: "tiny", label: "Tiny (~75 MB)", description: "Fastest, least accurate" },
  { id: "base", label: "Base (~150 MB)", description: "Default, good balance" },
  { id: "small", label: "Small (~500 MB)", description: "Better accuracy" },
  { id: "medium", label: "Medium (~1.5 GB)", description: "High accuracy" },
  { id: "large", label: "Large (~3 GB)", description: "Best accuracy, slowest" },
] as const;

// ---------------------------------------------------------------------------
// Tab definitions
// ---------------------------------------------------------------------------

type TabId = "general" | "ai" | "privacy" | "notifications" | "voice" | "about";

const TABS: readonly { id: TabId; label: string; icon: LucideIcon }[] = [
  { id: "general", label: "General", icon: Palette },
  { id: "ai", label: "AI Models", icon: Cpu },
  { id: "privacy", label: "Privacy", icon: Shield },
  { id: "notifications", label: "Notifications", icon: Bell },
  { id: "voice", label: "Voice & Audio", icon: Mic },
  { id: "about", label: "About", icon: Info },
];

// ---------------------------------------------------------------------------
// Section wrapper
// ---------------------------------------------------------------------------

function Section({
  title,
  icon: Icon,
  id,
  children,
}: {
  readonly title: string;
  readonly icon: LucideIcon;
  readonly id?: string;
  readonly children: React.ReactNode;
}) {
  return (
    <section id={id} className="rounded-4 border border-hairline bg-surface p-5 shadow-1">
      <div className="mb-4 flex items-center gap-2 border-b border-hairline pb-3">
        <Icon className="h-4 w-4 text-indigo" strokeWidth={1.6} />
        <h3 className="text-sm font-semibold text-ink">{title}</h3>
      </div>
      {children}
    </section>
  );
}

// ---------------------------------------------------------------------------
// Toggle switch
// ---------------------------------------------------------------------------

function Toggle({
  enabled,
  onChange,
}: {
  readonly enabled: boolean;
  readonly onChange: (v: boolean) => void;
}) {
  return (
    <button
      onClick={() => onChange(!enabled)}
      className={`relative h-6 w-10 rounded-pill transition-colors ${
        enabled ? "bg-indigo" : "bg-hairline"
      }`}
    >
      <span
        className={`absolute top-0.5 left-0.5 h-5 w-5 rounded-full bg-white shadow-1 transition-transform ${
          enabled ? "translate-x-4" : ""
        }`}
      />
    </button>
  );
}

// ---------------------------------------------------------------------------
// Keep Arandu running section
// ---------------------------------------------------------------------------

function KeepAwakeSection({
  settings,
  onUpdate,
  saving,
}: {
  readonly settings: AppSettings;
  readonly onUpdate: (patch: Partial<AppSettings>) => void;
  readonly saving: boolean;
}) {
  const toggleRow = (
    label: string,
    desc: ReactNode,
    key: keyof Pick<AppSettings, "prevent_sleep" | "prevent_sleep_on_battery" | "launch_at_login" | "menu_bar_mode">,
  ) => (
    <div className="flex items-start justify-between gap-4 rounded-2 bg-surface/60 px-4 py-3">
      <div className="min-w-0">
        <p className="text-sm text-ink">{label}</p>
        <p className="mt-0.5 text-[11px] text-muted">{desc}</p>
      </div>
      <button
        onClick={() => onUpdate({ [key]: !settings[key] })}
        disabled={saving}
        className={`relative h-[22px] w-[38px] shrink-0 rounded-full transition-colors ${
          settings[key] ? "bg-indigo" : "bg-hairline-2"
        } disabled:opacity-50`}
      >
        <span
          className={`absolute top-[2px] left-[2px] h-[18px] w-[18px] rounded-full bg-white shadow-sm transition-transform ${
            settings[key] ? "translate-x-4" : ""
          }`}
        />
      </button>
    </div>
  );

  return (
    <Section title="Keep Arandu running" icon={Power} id="presence">
      <p className="mb-4 text-xs text-muted">
        Arandu works only while this app and your Mac are awake. The Brain
        syncs data, runs scheduled agents, and answers from local models in
        the background — but only when the OS isn't asleep.
      </p>

      <div className="space-y-3">
        {/* Status row */}
        <div className="flex items-center justify-between rounded-2 bg-surface/60 px-4 py-3">
          <div>
            <p className="text-sm text-ink">App is running</p>
            <p className="mt-0.5 text-[11px] text-muted">
              Last heartbeat just now
            </p>
          </div>
          <span
            className="rounded-pill bg-success-soft px-3 py-1 text-[12.5px] font-medium"
            style={{ color: "oklch(0.36 0.10 155)" }}
          >
            ● Online
          </span>
        </div>

        {/* Prevent Mac from sleeping */}
        {toggleRow(
          "Prevent Mac from sleeping",
          <>
            While Arandu is open, we'll keep macOS from entering true sleep.
            The display can still dim and lock — only the CPU stays awake.
            Equivalent to{" "}
            <code className="rounded bg-bg-2 px-1.5 py-0.5 font-mono text-[12px]">
              caffeinate -i
            </code>
            .
          </>,
          "prevent_sleep",
        )}

        {/* Prevent sleep on battery */}
        {toggleRow(
          "Prevent sleep on battery",
          "Drains battery faster. Off by default. Recommended only when the Mac is plugged in.",
          "prevent_sleep_on_battery",
        )}

        {/* Launch at login */}
        {toggleRow(
          "Launch at login",
          "Auto-open after a reboot so syncing resumes without you remembering.",
          "launch_at_login",
        )}

        {/* Run in menu bar when window closes */}
        {toggleRow(
          "Run in menu bar when window closes",
          "Quitting the window won't quit the app — Arandu stays in the menu bar so agents keep running. Right-click the icon to fully quit.",
          "menu_bar_mode",
        )}

        {/* Expandable instructions */}
        <details className="group rounded-2 border border-hairline bg-surface/60">
          <summary className="flex cursor-pointer items-center justify-between px-4 py-3 text-sm text-ink select-none">
            <div>
              <p className="font-medium">
                How to make sure your Mac stays awake
              </p>
              <p className="mt-0.5 text-[11px] text-muted">
                Step-by-step for macOS Sequoia, Sonoma, and Ventura.
              </p>
            </div>
            <ChevronDown className="h-4 w-4 shrink-0 text-muted transition-transform group-open:rotate-180" />
          </summary>
          <div className="border-t border-hairline px-4 py-3">
            <ol className="list-decimal space-y-2 pl-5 text-xs text-ink">
              <li>
                Open <strong>System Settings → Energy</strong> (Sequoia) or{" "}
                <strong>Battery → Options</strong> (Sonoma/Ventura).
              </li>
              <li>
                Set <em>"Turn display off after"</em> to a comfortable
                timeout (e.g. 10 min). This dims the screen but does{" "}
                <strong>not</strong> sleep the CPU.
              </li>
              <li>
                Disable <em>"Put hard disks to sleep when possible"</em>.
              </li>
              <li>
                If you see <em>"Prevent automatic sleeping when the display
                is off"</em>, enable it.
              </li>
              <li>
                On a MacBook, keep it plugged in or enable{" "}
                <strong>"Prevent sleep on battery"</strong> above.
              </li>
              <li>
                Verify with{" "}
                <code className="rounded bg-bg-2 px-1.5 py-0.5 font-mono text-[12px]">
                  pmset -g assertions
                </code>{" "}
                in Terminal — you should see a{" "}
                <code className="rounded bg-bg-2 px-1.5 py-0.5 font-mono text-[12px]">
                  PreventUserIdleSystemSleep
                </code>{" "}
                assertion from Arandu.
              </li>
            </ol>
            <div className="mt-3 rounded-2 bg-indigo-soft px-3 py-2.5 text-xs text-indigo">
              Arandu only prevents <em>idle</em> sleep. Closing the lid on
              a MacBook still triggers hardware sleep — plug in and use an
              external display, or use{" "}
              <code className="rounded bg-bg-2 px-1.5 py-0.5 font-mono text-[12px]">
                sudo pmset disablesleep 1
              </code>{" "}
              (advanced).
            </div>
          </div>
        </details>

        {/* Last sleep event */}
        <div className="flex items-center justify-between rounded-2 bg-surface/60 px-4 py-3">
          <div>
            <p className="text-sm text-ink">Last sleep event</p>
            <p className="mt-0.5 text-[11px] text-muted">
              <strong>May 24, 2026 03:12 AM</strong> — 2 scheduled agents
              missed during 6 h 48 min sleep
            </p>
          </div>
          <button
            disabled={saving}
            className="shrink-0 rounded-2 border border-hairline bg-surface px-3 py-1.5 text-xs text-ink hover:bg-surface/80 disabled:opacity-50"
          >
            Run missed agents
          </button>
        </div>
      </div>
    </Section>
  );
}

// ---------------------------------------------------------------------------
// Appearance section (new)
// ---------------------------------------------------------------------------

const THEME_OPTIONS = [
  { value: "light", label: "Light", icon: Sun },
  { value: "dark", label: "Dark", icon: Moon },
] as const;

function AppearanceSection({
  settings,
  onUpdate,
  saving,
}: {
  readonly settings: AppSettings;
  readonly onUpdate: (patch: Partial<AppSettings>) => void;
  readonly saving: boolean;
}) {
  return (
    <Section title="Appearance" icon={Palette}>
      <div>
        <p className="mb-3 text-xs text-muted">Theme</p>
        <div className="flex gap-2">
          {THEME_OPTIONS.map(({ value, label, icon: Icon }) => (
            <button
              key={value}
              onClick={() => {
                onUpdate({ theme: value });
                broadcastThemeChange(value);
              }}
              disabled={saving}
              className={`flex flex-1 items-center justify-center gap-2 rounded-2 border px-4 py-3 text-sm transition ${
                settings.theme === value
                  ? "border-indigo bg-indigo-soft text-ink"
                  : "border-hairline bg-surface/60 text-muted hover:border-indigo/50"
              } disabled:opacity-50`}
            >
              <Icon className="h-4 w-4" />
              {label}
            </button>
          ))}
        </div>
      </div>
    </Section>
  );
}

// ---------------------------------------------------------------------------
// Embedding model section
// ---------------------------------------------------------------------------

interface RebuildResult {
  ok: boolean;
  exit_code?: number;
  target_model?: string;
  provider?: string;
  dry_run?: boolean;
  progress?: string;
  error?: string;
}

const LOCAL_EMBED_MODELS = [
  { id: "bge-m3", label: "bge-m3 (multilingual, 1024d) — recommended" },
  { id: "nomic-embed-text", label: "nomic-embed-text (English, 768d) — legacy" },
  { id: "mxbai-embed-large", label: "mxbai-embed-large (English, 1024d)" },
] as const;

const REMOTE_EMBED_MODELS = [
  {
    id: "text-embedding-3-large",
    label: "OpenAI text-embedding-3-large (3072d) — best quality",
  },
  {
    id: "text-embedding-3-small",
    label: "OpenAI text-embedding-3-small (1536d) — cheaper",
  },
] as const;

function EmbeddingModelSection() {
  const [provider, setProvider] = useState<"ollama" | "openai">("ollama");
  const [model, setModel] = useState<string>(LOCAL_EMBED_MODELS[0].id);
  const [apiKey, setApiKey] = useState<string>("");
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<RebuildResult | null>(null);

  const modelOptions =
    provider === "ollama" ? LOCAL_EMBED_MODELS : REMOTE_EMBED_MODELS;

  useEffect(() => {
    setModel(modelOptions[0].id);
  }, [provider, modelOptions]);

  const handleRebuild = useCallback(
    async (dryRun: boolean) => {
      setRunning(true);
      setResult(null);
      try {
        const payload: Record<string, unknown> = {
          toModel: model,
          provider,
          dryRun,
        };
        if (provider === "openai" && apiKey) {
          payload.apiKey = apiKey;
        }
        const r = await invoke<RebuildResult>(
          "rebuild_vector_index",
          payload,
        );
        setResult(r);
      } catch (err) {
        setResult({
          ok: false,
          error: err instanceof Error ? err.message : String(err),
        });
      } finally {
        setRunning(false);
      }
    },
    [model, provider, apiKey],
  );

  return (
    <Section title="Embedding Model" icon={Brain}>
      <p className="mb-4 text-xs text-muted">
        Controls the model that embeds your data into the vector store.
        Swapping models requires rebuilding all collections from raw_*
        tables — a few minutes for local models, a few cents for OpenAI.
        Run dry-run first to see the estimate.
      </p>
      <div className="space-y-3">
        <div>
          <label className="mb-1.5 block text-xs text-muted">
            Provider
          </label>
          <select
            value={provider}
            onChange={(e) => setProvider(e.target.value as "ollama" | "openai")}
            disabled={running}
            className="w-full rounded-2 bg-surface px-3 py-2 text-sm text-ink outline-none disabled:opacity-50"
          >
            <option value="ollama">Local (Ollama)</option>
            <option value="openai">Remote (OpenAI)</option>
          </select>
        </div>
        <div>
          <label className="mb-1.5 block text-xs text-muted">
            Model
          </label>
          <select
            value={model}
            onChange={(e) => setModel(e.target.value)}
            disabled={running}
            className="w-full rounded-2 bg-surface px-3 py-2 text-sm text-ink outline-none disabled:opacity-50"
          >
            {modelOptions.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label}
              </option>
            ))}
          </select>
        </div>
        {provider === "openai" && (
          <div>
            <label className="mb-1.5 block text-xs text-muted">
              OpenAI API key
            </label>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              disabled={running}
              placeholder="sk-..."
              className="w-full rounded-2 bg-surface px-3 py-2 text-sm text-ink outline-none disabled:opacity-50"
            />
          </div>
        )}
        <div className="flex gap-2 pt-2">
          <button
            onClick={() => handleRebuild(true)}
            disabled={running || (provider === "openai" && !apiKey)}
            className="rounded border border-hairline px-3 py-2 text-sm text-ink hover:bg-surface disabled:opacity-50"
          >
            {running ? "Working..." : "Estimate (dry-run)"}
          </button>
          <button
            onClick={() => handleRebuild(false)}
            disabled={running || (provider === "openai" && !apiKey)}
            className="rounded bg-indigo px-3 py-2 text-sm text-white hover:opacity-90 disabled:opacity-50"
          >
            {running ? "Rebuilding..." : "Rebuild vector index"}
          </button>
        </div>
        {result && (
          <div
            className={`mt-3 rounded border p-3 text-xs ${
              result.ok
                ? "border-hairline bg-surface text-ink"
                : "border-danger/50 bg-danger/10 text-danger"
            }`}
          >
            <div className="mb-1 font-semibold">
              {result.ok
                ? result.dry_run
                  ? "Dry-run complete"
                  : "Rebuild complete"
                : "Failed"}
            </div>
            <pre className="whitespace-pre-wrap font-mono">
              {result.progress || result.error || ""}
            </pre>
          </div>
        )}
      </div>
    </Section>
  );
}

// ---------------------------------------------------------------------------
// AI Model section
// ---------------------------------------------------------------------------

function AIModelSection({
  settings,
  onUpdate,
  saving,
}: {
  readonly settings: AppSettings;
  readonly onUpdate: (patch: Partial<AppSettings>) => void;
  readonly saving: boolean;
}) {
  // Three tiers:
  //  - llama3.1:70b — recommended default, listed first so it is the
  //    fallback (`?? ollamaModels[0]`).
  //  - `belowMin` — runs on lighter hardware; results not guaranteed.
  //  - `testingOnly` — 1B-class; only useful to exercise the UI on a
  //    low-RAM machine, not for real answers.
  const ollamaModels = [
    {
      id: "llama3.1:70b",
      label: "Llama 3.1 70B",
      note: "Default · M2/M3 Ultra · 64–128 GB RAM · 1–2 TB SSD",
      gpu: true,
      belowMin: false,
      testingOnly: false,
    },
    {
      id: "gemma4:e4b",
      label: "Gemma 4 E4B",
      note: "Stronger reasoning, ~5 GB RAM, 128K ctx",
      gpu: false,
      belowMin: true,
      testingOnly: false,
    },
    {
      id: "gemma4:e2b",
      label: "Gemma 4 E2B",
      note: "Lightest, native JSON, ~2 GB RAM, 128K ctx",
      gpu: false,
      belowMin: false,
      testingOnly: true,
    },
    {
      id: "llama3.2:3b",
      label: "Llama 3.2 3B",
      note: "Reliable JSON, ~2 GB RAM",
      gpu: false,
      belowMin: false,
      testingOnly: true,
    },
    {
      id: "qwen3.5:2b",
      label: "Qwen 3.5 2B",
      note: "Compact, ~1 GB RAM",
      gpu: false,
      belowMin: false,
      testingOnly: true,
    },
    {
      id: "llama3.1:8b",
      label: "Llama 3.1 8B",
      note: "CPU ok, ~8 GB RAM",
      gpu: false,
      belowMin: true,
      testingOnly: false,
    },
    {
      id: "mistral:7b",
      label: "Mistral 7B",
      note: "CPU ok, ~8 GB RAM",
      gpu: false,
      belowMin: true,
      testingOnly: false,
    },
    {
      id: "llama3.2:1b",
      label: "Llama 3.2 1B",
      note: "Interface testing only · ~1 GB RAM",
      gpu: false,
      belowMin: false,
      testingOnly: true,
    },
    {
      id: "gemma3:1b",
      label: "Gemma 3 1B",
      note: "Interface testing only · ~1 GB RAM",
      gpu: false,
      belowMin: false,
      testingOnly: true,
    },
  ];

  const selectedOllamaModel =
    ollamaModels.find((m) => m.id === settings.llm_model) ?? ollamaModels[0];

  return (
    <Section title="AI Model" icon={Cpu}>
      <div className="space-y-4">
        <div>
          <label className="mb-1.5 block text-xs text-muted">Model</label>
          <select
            value={settings.llm_model}
            onChange={(e) => onUpdate({ llm_model: e.target.value })}
            disabled={saving}
            className="w-full rounded-2 bg-surface px-3 py-2 text-sm text-ink outline-none disabled:opacity-50"
          >
            {ollamaModels.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label} — {m.note}
              </option>
            ))}
          </select>
        </div>

        {selectedOllamaModel.gpu && (
          <div className="flex items-start gap-2 rounded-2 border border-amber/30 bg-amber/5 px-3 py-2.5">
            <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber" />
            <p className="text-xs text-amber">
              Recommended default. Needs a capable machine — Apple Silicon
              M2/M3 Ultra with 64–128 GB RAM and a 1–2 TB SSD (or a GPU with
              ≥48 GB VRAM). On weaker hardware it can starve the OS and make
              the whole computer unresponsive.
            </p>
          </div>
        )}

        {selectedOllamaModel.belowMin && (
          <div className="flex items-start gap-2 rounded-2 border border-amber/30 bg-amber/5 px-3 py-2.5">
            <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber" />
            <p className="text-xs text-amber">
              Below the recommended minimum (Llama 3.1 70B). This model runs
              on lighter hardware, but Arandu does not guarantee acceptable
              results with it.
            </p>
          </div>
        )}

        {selectedOllamaModel.testingOnly && (
          <div className="flex items-start gap-2 rounded-2 border border-amber/30 bg-amber/5 px-3 py-2.5">
            <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber" />
            <p className="text-xs text-amber">
              <strong>Interface testing only.</strong> A 1B model is far too
              small for real answers — use it to exercise the app's UI on a
              low-RAM machine. Expect incoherent or empty responses.
            </p>
          </div>
        )}

        <p className="text-[11px] text-muted">
          Models run 100% on your device via Ollama. Your data never
          leaves your machine.
        </p>
      </div>
    </Section>
  );
}

// ---------------------------------------------------------------------------
// Privacy section
// ---------------------------------------------------------------------------

function PrivacySection({
  settings,
  onUpdateConsentLevel,
  onLocalInferenceCommitted,
  saving,
}: {
  readonly settings: AppSettings;
  readonly onUpdateConsentLevel: (tier: number) => void;
  readonly onLocalInferenceCommitted: () => void;
  readonly saving: boolean;
}) {
  const [showEnableModal, setShowEnableModal] = useState(false);
  const [togglingOff, setTogglingOff] = useState(false);

  // Export data
  const [exporting, setExporting] = useState(false);
  const [exportPath, setExportPath] = useState<string | null>(null);
  const [exportError, setExportError] = useState<string | null>(null);

  // Delete all data (type-to-confirm)
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [deleteConfirm, setDeleteConfirm] = useState("");
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  // Tauri command rejections arrive as the raw error string (the `Err`
  // side of the command's `Result<_, String>`), not an Error instance.
  const errMessage = (err: unknown, fallback: string): string =>
    typeof err === "string"
      ? err
      : err instanceof Error
        ? err.message
        : fallback;

  const handleExport = useCallback(async () => {
    setExporting(true);
    setExportError(null);
    setExportPath(null);
    try {
      const res = await invoke<{ ok: boolean; path?: string; error?: string }>(
        "export_user_data",
      );
      if (res.ok && res.path) {
        setExportPath(res.path);
      } else {
        setExportError(res.error ?? "Export failed.");
      }
    } catch (err) {
      setExportError(errMessage(err, "Export failed."));
    } finally {
      setExporting(false);
    }
  }, []);

  const handleDelete = useCallback(async () => {
    setDeleting(true);
    setDeleteError(null);
    try {
      const res = await invoke<{ ok: boolean; error?: string }>(
        "delete_all_user_data",
      );
      if (res.ok) {
        // Everything is gone — reload into a fresh, empty app so no
        // stale in-memory state lingers.
        window.location.reload();
      } else {
        setDeleteError(res.error ?? "Delete failed.");
        setDeleting(false);
      }
    } catch (err) {
      setDeleteError(errMessage(err, "Delete failed."));
      setDeleting(false);
    }
  }, []);

  const handleLocalInferenceToggle = useCallback(
    async (next: boolean) => {
      if (next) {
        setShowEnableModal(true);
        return;
      }
      try {
        setTogglingOff(true);
        await invoke("set_local_inference_for_sensitive", {
          enabled: false,
        });
        onLocalInferenceCommitted();
      } finally {
        setTogglingOff(false);
      }
    },
    [onLocalInferenceCommitted],
  );

  return (
    <Section title="Privacy" icon={Shield}>
      <div className="space-y-4">
        <div className="rounded-2 border border-hairline bg-surface/40 p-3">
          <div className="flex items-start justify-between gap-3">
            <div>
              <p className="text-sm text-ink">
                Run sensitive prompts locally
              </p>
              <p className="mt-1 text-[11px] text-muted">
                {settings.local_inference_for_sensitive
                  ? "All tiers stay on your local Ollama model. Each agent must keep passing its eval suite."
                  : "When configured, sensitive prompts pass through an on-device redactor that replaces names, contacts, accounts, and amounts with placeholders before any remote call. Raw values never leave your machine."}
              </p>
              <p className="mt-1 text-[11px] text-muted">
                Requires a capable local model. Enabling runs every
                agent's eval suite — the toggle commits only if every
                run passes.
              </p>
            </div>
            <label className="relative inline-flex shrink-0 items-center">
              <input
                type="checkbox"
                checked={settings.local_inference_for_sensitive}
                disabled={saving || togglingOff}
                onChange={(e) =>
                  handleLocalInferenceToggle(e.target.checked)
                }
                className="sr-only peer"
              />
              <span className="h-5 w-9 rounded-full bg-hairline transition-colors peer-checked:bg-indigo" />
              <span className="absolute left-0.5 top-0.5 h-4 w-4 rounded-full bg-white transition-transform peer-checked:translate-x-4" />
            </label>
          </div>
        </div>

        {showEnableModal && (
          <LocalInferenceEnableModal
            onCommitted={() => {
              setShowEnableModal(false);
              onLocalInferenceCommitted();
            }}
            onClose={() => setShowEnableModal(false)}
          />
        )}

        <div>
          <p className="mb-2 text-xs text-muted">
            Maximum auto-approved sensitivity tier for agents
          </p>
          <div className="space-y-2">
            {CONSENT_LEVELS.map(({ value, label, description }) => (
              <label
                key={value}
                className={`flex cursor-pointer items-center gap-3 rounded-2 px-4 py-3 transition-colors ${
                  settings.max_sensitivity_tier === value
                    ? "bg-indigo-soft border border-indigo/30"
                    : "bg-surface/60 border border-transparent hover:border-hairline"
                }`}
              >
                <input
                  type="radio"
                  name="consent"
                  checked={settings.max_sensitivity_tier === value}
                  onChange={() => onUpdateConsentLevel(value)}
                  disabled={saving}
                  className="accent-indigo"
                />
                <div>
                  <p className="text-sm text-ink">{label}</p>
                  <p className="text-[11px] text-muted">{description}</p>
                </div>
              </label>
            ))}
          </div>
        </div>

        <div className="border-t border-hairline pt-4">
          <div className="flex flex-wrap gap-2">
            <button
              onClick={handleExport}
              disabled={exporting}
              className="flex items-center gap-2 rounded-2 border border-hairline px-4 py-2 text-xs text-ink transition-colors hover:bg-surface disabled:cursor-not-allowed disabled:opacity-60"
            >
              <Download className="h-3.5 w-3.5" />
              {exporting ? "Exporting…" : "Export data (JSON)"}
            </button>

            <button
              onClick={() => {
                setDeleteConfirm("");
                setDeleteError(null);
                setShowDeleteModal(true);
              }}
              className="flex items-center gap-2 rounded-2 border border-danger/30 px-4 py-2 text-xs text-danger transition-colors hover:bg-danger/5"
            >
              <Trash2 className="h-3.5 w-3.5" strokeWidth={1.6} />
              Delete all my data
            </button>
          </div>

          {exportPath && (
            <p className="mt-2 break-all text-[11px] text-muted">
              Exported to <span className="text-ink">{exportPath}</span>
            </p>
          )}
          {exportError && (
            <p className="mt-2 text-[11px] text-danger">{exportError}</p>
          )}
          <p className="mt-2 text-[11px] text-muted">
            Export bundles every local table into an inspectable JSON zip.
            Delete permanently wipes all local data (databases, learned
            facts, goals, audit log, sessions); your app preferences are
            kept.
          </p>
        </div>
      </div>

      {showDeleteModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
          <div className="w-full max-w-md rounded-3 border border-hairline bg-bg p-6 shadow-lg">
            <div className="flex items-center gap-2 text-danger">
              <AlertTriangle className="h-5 w-5" strokeWidth={1.8} />
              <h3 className="text-sm font-semibold">Delete all my data</h3>
            </div>
            <p className="mt-3 text-xs text-ink-2">
              This permanently deletes every local database, learned fact,
              goal, task, the audit log, and connector sessions. It cannot
              be undone. Your app preferences (name, model choice) are kept.
              Export first if you want a copy.
            </p>
            <p className="mt-3 text-xs text-muted">
              Type <span className="font-mono text-ink">DELETE</span> to
              confirm.
            </p>
            <input
              type="text"
              value={deleteConfirm}
              onChange={(e) => setDeleteConfirm(e.target.value)}
              autoFocus
              className="mt-2 w-full rounded-2 border border-hairline bg-surface px-3 py-2 text-sm text-ink outline-none focus:border-danger"
              placeholder="DELETE"
            />
            {deleteError && (
              <p className="mt-2 text-[11px] text-danger">{deleteError}</p>
            )}
            <div className="mt-4 flex justify-end gap-2">
              <button
                onClick={() => setShowDeleteModal(false)}
                disabled={deleting}
                className="rounded-2 border border-hairline px-4 py-2 text-xs text-ink transition-colors hover:bg-surface disabled:opacity-60"
              >
                Cancel
              </button>
              <button
                onClick={handleDelete}
                disabled={deleteConfirm !== "DELETE" || deleting}
                className="rounded-2 bg-danger px-4 py-2 text-xs font-medium text-white transition-colors hover:bg-danger/90 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {deleting ? "Deleting…" : "Delete everything"}
              </button>
            </div>
          </div>
        </div>
      )}
    </Section>
  );
}

// ---------------------------------------------------------------------------
// Data Refresh section
// ---------------------------------------------------------------------------

const INTERVAL_OPTIONS = [
  { value: 15, label: "Every 15 minutes" },
  { value: 30, label: "Every 30 minutes" },
  { value: 60, label: "Every hour" },
  { value: 120, label: "Every 2 hours" },
  { value: 360, label: "Every 6 hours" },
] as const;

function DataRefreshSection({
  settings,
  onUpdate,
  saving,
  runHistory,
  historyLoading,
}: {
  readonly settings: AppSettings;
  readonly onUpdate: (patch: Partial<AppSettings>) => void;
  readonly saving: boolean;
  readonly runHistory: PipelineRunHistoryEntry[];
  readonly historyLoading: boolean;
}) {
  return (
    <Section title="Data Refresh" icon={RefreshCw}>
      <div className="space-y-4">
        <div className="flex items-center justify-between rounded-2 bg-surface/60 px-4 py-3">
          <div>
            <p className="text-sm text-ink">Auto-refresh</p>
            <p className="text-[11px] text-muted">
              Automatically refresh data in the background
            </p>
          </div>
          <Toggle
            enabled={settings.auto_refresh_enabled}
            onChange={(v) => onUpdate({ auto_refresh_enabled: v })}
          />
        </div>

        {settings.auto_refresh_enabled && (
          <div className="rounded-2 bg-surface/60 px-4 py-3">
            <label className="mb-1.5 block text-xs text-muted">
              Refresh interval
            </label>
            <select
              value={settings.auto_refresh_interval_minutes}
              onChange={(e) =>
                onUpdate({
                  auto_refresh_interval_minutes: Number(e.target.value),
                })
              }
              disabled={saving}
              className="w-full rounded-2 bg-surface px-3 py-2 text-sm text-ink outline-none disabled:opacity-50"
            >
              {INTERVAL_OPTIONS.map(({ value, label }) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
          </div>
        )}

        <div className="flex items-center justify-between rounded-2 bg-surface/60 px-4 py-3">
          <div>
            <p className="text-sm text-ink">Refresh on launch</p>
            <p className="text-[11px] text-muted">
              Run a background refresh 30s after the app starts
            </p>
          </div>
          <Toggle
            enabled={settings.refresh_on_launch}
            onChange={(v) => onUpdate({ refresh_on_launch: v })}
          />
        </div>

        <div className="rounded-2 bg-surface/60 p-3">
          <p className="mb-2 text-xs font-medium text-ink">
            Recent refreshes
          </p>
          {historyLoading ? (
            <p className="text-[11px] text-muted">Loading...</p>
          ) : runHistory.length === 0 ? (
            <p className="text-[11px] text-muted">
              No pipeline runs yet
            </p>
          ) : (
            <div className="space-y-1.5">
              {runHistory.map((run) => (
                <div
                  key={run.run_id}
                  className="flex items-center justify-between text-[11px]"
                >
                  <div className="flex items-center gap-2">
                    <span
                      className={`h-1.5 w-1.5 rounded-full ${
                        run.status === "success"
                          ? "bg-success"
                          : "bg-amber"
                      }`}
                    />
                    <span className="text-muted">
                      {new Date(run.started_at).toLocaleDateString(undefined, {
                        month: "short",
                        day: "numeric",
                        hour: "2-digit",
                        minute: "2-digit",
                      })}
                    </span>
                  </div>
                  <div className="flex items-center gap-3 text-muted">
                    <span>{Math.round(run.duration_seconds)}s</span>
                    <span
                      className={`rounded-full px-1.5 py-0.5 text-[10px] ${
                        run.status === "success"
                          ? "bg-success/15 text-success"
                          : "bg-amber/15 text-amber"
                      }`}
                    >
                      {run.status}
                    </span>
                    <span className="text-[10px]">{run.trigger}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        <p className="text-[11px] text-muted">
          Press Cmd+R (Ctrl+R) to manually refresh from anywhere.
        </p>
      </div>
    </Section>
  );
}

// ---------------------------------------------------------------------------
// Notifications section
// ---------------------------------------------------------------------------

function NotificationsSection({
  settings,
  onUpdate,
  saving,
}: {
  readonly settings: AppSettings;
  readonly onUpdate: (patch: Partial<AppSettings>) => void;
  readonly saving: boolean;
}) {
  const [phoneLocal, setPhoneLocal] = useState(
    settings.whatsapp_notification_phone ?? "",
  );

  useEffect(() => {
    setPhoneLocal(settings.whatsapp_notification_phone ?? "");
  }, [settings.whatsapp_notification_phone]);

  const prefsResult = useAsyncData<NotificationPreference[]>(
    useCallback(
      () =>
        dedupInvoke<NotificationPreference[]>(
          "get_notification_preferences",
        ),
      [],
    ),
  );

  const prefs = prefsResult.data ?? [];

  const handleToggleCategory = async (
    category: string,
    enabled: boolean,
  ) => {
    await invoke("update_notification_preference", {
      category,
      enabled,
    });
    prefsResult.refetch();
  };

  const handleMuteAll = async () => {
    await invoke("mute_all_notifications", {});
    prefsResult.refetch();
  };

  return (
    <Section title="Notifications" icon={Bell}>
      <div className="space-y-3">
        <div className="flex items-center justify-between rounded-2 bg-surface/60 px-4 py-3">
          <div>
            <p className="text-sm text-ink">
              WhatsApp Notifications
            </p>
            <p className="text-[11px] text-muted">
              Get notified via WhatsApp for important events
            </p>
          </div>
          <Toggle
            enabled={settings.notifications_enabled}
            onChange={(v) => onUpdate({ notifications_enabled: v })}
          />
        </div>

        {settings.notifications_enabled && (
          <div className="rounded-2 bg-surface/60 px-4 py-3">
            <label className="mb-1.5 block text-xs text-muted">
              WhatsApp phone number
            </label>
            <input
              type="tel"
              value={phoneLocal}
              onChange={(e) => setPhoneLocal(e.target.value)}
              onBlur={() =>
                onUpdate({
                  whatsapp_notification_phone:
                    phoneLocal || null,
                })
              }
              placeholder="+1234567890"
              disabled={saving}
              className="w-full rounded-2 bg-surface px-3 py-2 text-sm text-ink outline-none border border-hairline focus:border-indigo"
            />
          </div>
        )}

        {settings.notifications_enabled && prefs.length > 0 && (
          <div className="space-y-2">
            <p className="text-xs text-muted px-1">
              Notification categories
            </p>
            {prefs
              .filter((p) => p.category !== "_global")
              .map((pref) => (
                <div
                  key={pref.category}
                  className="flex items-center justify-between rounded-2 bg-surface/60 px-4 py-2.5"
                >
                  <span className="text-sm text-ink capitalize">
                    {pref.category.replace(/_/g, " ")}
                  </span>
                  <Toggle
                    enabled={pref.enabled}
                    onChange={(v) =>
                      handleToggleCategory(pref.category, v)
                    }
                  />
                </div>
              ))}
          </div>
        )}

        {settings.notifications_enabled && (
          <button
            onClick={handleMuteAll}
            disabled={saving}
            className="text-[11px] text-muted hover:text-ink px-1"
          >
            Mute all notifications for 24h
          </button>
        )}
      </div>
    </Section>
  );
}

// ---------------------------------------------------------------------------
// Voice & Audio section
// ---------------------------------------------------------------------------

function VoiceSection({
  settings,
  onUpdate,
}: {
  readonly settings: AppSettings;
  readonly onUpdate: (patch: Partial<AppSettings>) => void;
}) {
  return (
    <Section title="Voice & Audio" icon={Mic}>
      <div className="space-y-3">
        <div className="flex items-center justify-between rounded-2 bg-surface/60 px-4 py-3">
          <div>
            <p className="text-sm text-ink">
              Voice Transcription
            </p>
            <p className="text-[11px] text-muted">
              Enable voice input in Chat and WhatsApp audio transcription.
              Uses Qwen3-ASR (52 languages). One-time model download.
            </p>
          </div>
          <Toggle
            enabled={settings.voice_transcription_enabled ?? true}
            onChange={(v) => onUpdate({ voice_transcription_enabled: v })}
          />
        </div>

        {settings.voice_transcription_enabled && (
          <>
            <div className="rounded-2 bg-surface/60 px-4 py-3">
              <label className="mb-1.5 block text-xs text-muted">
                Model size
              </label>
              <select
                value={settings.whisper_model_size}
                onChange={(e) => onUpdate({ whisper_model_size: e.target.value })}
                className="w-full rounded-2 bg-surface px-3 py-2 text-sm text-ink outline-none"
              >
                {WHISPER_MODEL_SIZES.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.label} — {m.description}
                  </option>
                ))}
              </select>
              <p className="mt-1 text-[10px] text-muted">
                Larger models are more accurate but use more RAM and disk space.
              </p>
            </div>

            <div className="flex items-center justify-between rounded-2 bg-surface/60 px-4 py-3">
              <div>
                <p className="text-sm text-ink">
                  Transcribe WhatsApp Audio
                </p>
                <p className="text-[11px] text-muted">
                  Automatically transcribe voice notes from WhatsApp
                </p>
              </div>
              <Toggle
                enabled={settings.transcribe_whatsapp_audio ?? true}
                onChange={(v) =>
                  onUpdate({ transcribe_whatsapp_audio: v })
                }
              />
            </div>
          </>
        )}
      </div>
    </Section>
  );
}

// ---------------------------------------------------------------------------
// About section
// ---------------------------------------------------------------------------

function AboutSection() {
  const [version, setVersion] = useState<string>("—");
  useEffect(() => {
    let cancelled = false;
    getVersion()
      .then((v) => {
        if (!cancelled) setVersion(v);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <Section title="About" icon={Info}>
      <div className="space-y-3">
        <div className="flex items-center justify-between rounded-2 bg-surface/60 px-4 py-3">
          <span className="text-xs text-muted">Version</span>
          <span className="text-xs font-medium text-ink">{version}</span>
        </div>
        <div className="flex items-center justify-between rounded-2 bg-surface/60 px-4 py-3">
          <span className="text-xs text-muted">License</span>
          <span className="text-xs font-medium text-ink">
            Open source — MIT
          </span>
        </div>
        <div className="flex items-center justify-between rounded-2 bg-surface/60 px-4 py-3">
          <span className="text-xs text-muted">GitHub</span>
          <span className="text-xs font-medium text-indigo">
            github.com/vdittgen/arandu
          </span>
        </div>
        <div className="rounded-2 bg-surface/60 px-4 py-3">
          <p className="mb-2 text-xs text-muted">Built with</p>
          <div className="flex flex-wrap gap-2">
            {TECH_STACK.map((tech) => (
              <span
                key={tech}
                className="rounded-full bg-surface px-2.5 py-1 text-[11px] text-ink"
              >
                {tech}
              </span>
            ))}
          </div>
        </div>
      </div>
    </Section>
  );
}

// ---------------------------------------------------------------------------
// Main Settings page — tabbed layout
// ---------------------------------------------------------------------------

function SettingsPage() {
  const [activeTab, setActiveTab] = useState<TabId>("general");
  const [saving, setSaving] = useState(false);

  const settingsResult = useAsyncData<AppSettings>(
    useCallback(() => dedupInvoke<AppSettings>("get_settings"), []),
  );

  const historyResult = useAsyncData<PipelineRunHistoryEntry[]>(
    useCallback(
      () =>
        dedupInvoke<PipelineRunHistoryEntry[]>("get_pipeline_run_history", {
          limit: 5,
        }),
      [],
    ),
  );

  const handleRefresh = () => {
    settingsResult.refetch();
    historyResult.refetch();
  };

  const updateSettings = async (patch: Partial<AppSettings>) => {
    if (!settingsResult.data) return;
    setSaving(true);
    const updated = { ...settingsResult.data, ...patch };
    const modelChanged =
      patch.llm_model !== undefined &&
      patch.llm_model !== settingsResult.data.llm_model;
    try {
      await invoke("update_settings", { settings: updated });
      settingsResult.refetch();
      // When the user picks a different model, pull + load it now.
      // Fire-and-forget — a missing model downloads first, with progress
      // shown via the "Preparing model" task / model status indicator.
      if (modelChanged) {
        void invoke("preload_ollama_model").catch(() => {});
      }
    } finally {
      setSaving(false);
    }
  };

  const settings = settingsResult.data;

  const renderTabContent = () => {
    if (settingsResult.isLoading || !settings) {
      return (
        <>
          <SkeletonSection />
          <SkeletonSection />
        </>
      );
    }

    switch (activeTab) {
      case "general":
        return (
          <>
            <KeepAwakeSection
              settings={settings}
              onUpdate={(patch) => updateSettings(patch)}
              saving={saving}
            />
            <AppearanceSection
              settings={settings}
              onUpdate={(patch) => updateSettings(patch)}
              saving={saving}
            />
            <DataRefreshSection
              settings={settings}
              onUpdate={(patch) => updateSettings(patch)}
              saving={saving}
              runHistory={historyResult.data ?? []}
              historyLoading={historyResult.isLoading}
            />
          </>
        );
      case "ai":
        return (
          <>
            <AIModelSection
              settings={settings}
              onUpdate={(patch) => updateSettings(patch)}
              saving={saving}
            />
            <EmbeddingModelSection />
          </>
        );
      case "privacy":
        return (
          <PrivacySection
            settings={settings}
            onUpdateConsentLevel={(tier) =>
              updateSettings({ max_sensitivity_tier: tier })
            }
            onLocalInferenceCommitted={() => settingsResult.refetch()}
            saving={saving}
          />
        );
      case "notifications":
        return (
          <NotificationsSection
            settings={settings}
            onUpdate={(patch) => updateSettings(patch)}
            saving={saving}
          />
        );
      case "voice":
        return (
          <VoiceSection
            settings={settings}
            onUpdate={(patch) => updateSettings(patch)}
          />
        );
      case "about":
        return <AboutSection />;
    }
  };

  return (
    <div className="grid flex-1 grid-cols-[220px_1fr] overflow-hidden">
      {/* Left nav */}
      <nav className="flex shrink-0 flex-col gap-1 border-r border-hairline p-3">
        <div className="mb-3">
          <h2
            className="text-[44px] font-bold leading-none"
            style={{
              background: "linear-gradient(135deg, var(--ink), var(--ink-2))",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              backgroundClip: "text",
            }}
          >
            Settings
          </h2>
          <p className="mt-1 text-[11px] text-muted">
            Configure your Arandu.
          </p>
        </div>
        {TABS.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setActiveTab(id)}
            className={`flex items-center gap-2.5 rounded-2 px-3 py-2 text-sm transition-colors ${
              activeTab === id
                ? "bg-indigo-soft text-indigo-2"
                : "text-muted hover:bg-surface hover:text-ink"
            }`}
          >
            <Icon className="h-4 w-4 shrink-0" strokeWidth={1.6} />
            {label}
          </button>
        ))}
        <div className="mt-auto pt-4">
          <button
            onClick={handleRefresh}
            disabled={settingsResult.isLoading}
            className="flex w-full items-center justify-center gap-2 rounded-2 bg-surface px-3 py-2 text-xs text-muted hover:text-ink disabled:opacity-50"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${settingsResult.isLoading ? "animate-spin" : ""}`} strokeWidth={1.6} />
            Refresh
          </button>
        </div>
      </nav>

      {/* Content area */}
      <div className="flex-1 space-y-6 overflow-y-auto p-6">
        {renderTabContent()}
      </div>
    </div>
  );
}

export default SettingsPage;
