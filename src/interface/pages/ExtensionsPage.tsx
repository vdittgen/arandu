/**
 * Connectors page — manage data source integrations + discover new ones.
 *
 * Two tabs: My Connectors | Discover
 *
 * sensitivity_tier: 1 (catalog/registry metadata is infrastructure)
 */

import { useState, useCallback, useMemo, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import {
  Plug,
  Compass,
  Search,
  RefreshCw,
  ChevronDown,
  ChevronRight,
  Trash2,
  Settings,
  ExternalLink,
  Clock,
  Database,
  Loader2,
  Terminal,
  Lock,
  Key,
  FileText,
  Download,
  AlertTriangle,
  MessageSquare,
  Briefcase,
  Heart,
  Code2,
  Globe,
  CreditCard,
  Plus,
  type LucideIcon,
} from "lucide-react";
import { Skeleton, SkeletonSection, SkeletonTable } from "../components/LoadingState";
import { useAsyncData } from "../hooks/useAsyncData";
import { dedupInvoke } from "../utils/requestDedup";
import { formatRelativeTime } from "../utils/timeFormat";
import InstallExtensionModal, { type EnvRequirement } from "../components/InstallExtensionModal";
import ConnectorConfigModal from "../components/ConnectorConfigModal";
import { WhatsAppPairingPanel } from "../components/WhatsAppPairingPanel";

// ---------------------------------------------------------------------------
// Types (match Rust DTOs in src-tauri/src/commands/types.rs)
// ---------------------------------------------------------------------------

// sensitivity_tier: 1
interface ConnectorSyncStats {
  readonly records_synced: number;
  readonly last_sync: string | null;
  readonly next_sync: string | null;
}

// sensitivity_tier: 1
interface ConnectorMissingRequirement {
  readonly type: string;
  readonly key: string;
  readonly label: string;
  readonly action: string;
}

// sensitivity_tier: 1
interface ConnectorCatalogEntry {
  readonly connector_id: string;
  readonly name: string;
  readonly icon: string;
  readonly description: string;
  readonly category: string;
  readonly enabled: boolean;
  readonly status: string;
  readonly stats: ConnectorSyncStats;
  readonly missing_requirements: readonly ConnectorMissingRequirement[];
  readonly tools_available: number;
  readonly default_schedule: string;
  readonly note: string | null;
}

// sensitivity_tier: 1
interface ConnectorToolDetail {
  readonly tool_name: string;
  readonly tool_type: string;
  readonly target_table: string | null;
  readonly field_count: number;
  readonly dedup_key: readonly string[];
  readonly fields?: readonly {
    readonly source: string;
    readonly target: string;
    readonly type: string;
    readonly tier: number;
    readonly transform: string | null;
  }[];
}

// sensitivity_tier: 1
interface ConnectorDetailData {
  readonly connector_id: string;
  readonly name: string;
  readonly tools: readonly ConnectorToolDetail[];
  readonly stats: ConnectorSyncStats & { readonly error: string | null };
  readonly schedule: {
    readonly interval_seconds: number;
    readonly next_sync: string | null;
  } | null;
  readonly default_schedule: string;
  readonly [key: string]: unknown;
}

// sensitivity_tier: 1
interface ConnectorHistoryEntry {
  readonly sync_id: string;
  readonly started_at: string;
  readonly completed_at: string;
  readonly rows_synced: number;
  readonly duration_seconds: number;
  readonly status: string;
  readonly error: string | null;
}

// sensitivity_tier: 1
interface ExtensionLogOutput {
  readonly extension_id: string;
  readonly lines: readonly string[];
}

// sensitivity_tier: 1
interface AppSettings {
  readonly notifications_enabled: boolean;
  readonly whatsapp_notification_phone: string | null;
  readonly [key: string]: unknown;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// Fallback permission used to pre-open System Settings when the user
// toggles a connector ON before the manager has reported its missing
// requirements. All apple-* connectors read protected SQLite databases
// under ~/Library directly, so they all gate on Full Disk Access (the
// catalog declares this too — keep these in sync).
const CONNECTOR_PERMISSION_BY_ID: Readonly<Record<string, string>> = {
  "apple-calendar": "Full Disk Access",
  "apple-contacts": "Full Disk Access",
  "apple-notes": "Full Disk Access",
  "apple-mail": "Full Disk Access",
  "apple-messages": "Full Disk Access",
};

const WHATSAPP_CONNECTOR_ID = "whatsapp";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function statusColor(status: string): { dot: string; text: string; bg: string } {
  switch (status) {
    case "connected":
      return { dot: "bg-success", text: "text-success", bg: "bg-success/15" };
    case "running":
      return { dot: "bg-success animate-pulse", text: "text-success", bg: "bg-success/15" };
    case "error":
      return { dot: "bg-danger", text: "text-danger", bg: "bg-amber/15" };
    case "needs_setup":
    case "scheduled":
      return { dot: "bg-amber", text: "text-amber", bg: "bg-amber/15" };
    case "idle":
      return { dot: "bg-muted", text: "text-muted", bg: "bg-surface" };
    default:
      return { dot: "bg-gray-500", text: "text-gray-500", bg: "bg-surface" };
  }
}

function statusLabel(status: string): string {
  switch (status) {
    case "connected":
      return "Running";
    case "needs_setup":
      return "Setup Required";
    case "error":
      return "Error";
    case "disabled":
      return "Stopped";
    case "running":
      return "Running";
    case "idle":
      return "Idle";
    case "scheduled":
      return "Scheduled";
    default:
      return status;
  }
}

function requirementIcon(type: string) {
  switch (type) {
    case "permission": return Lock;
    case "oauth": return Key;
    case "env": return FileText;
    case "app": return Download;
    default: return AlertTriangle;
  }
}

function tierDotColor(tier: number): string {
  switch (tier) {
    case 1: return "bg-success";
    case 2: return "bg-amber";
    case 3: return "bg-danger";
    default: return "bg-gray-500";
  }
}

function formatTimestamp(ts: string): string {
  try {
    const d = new Date(ts);
    return d.toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return ts;
  }
}

function normalizeWhatsappPhone(value: string): string {
  return value.replace(/\s+/g, "").trim();
}

// ---------------------------------------------------------------------------
// StatusBadge
// ---------------------------------------------------------------------------

function StatusBadge({ status }: { readonly status: string }) {
  const c = statusColor(status);
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${c.bg} ${c.text}`}>
      <span className={`h-1.5 w-1.5 rounded-full ${c.dot}`} />
      {statusLabel(status)}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Toggle
// ---------------------------------------------------------------------------

function Toggle({
  enabled,
  onChange,
  disabled,
  pending,
}: {
  readonly enabled: boolean;
  readonly onChange: (v: boolean) => void;
  readonly disabled?: boolean;
  readonly pending?: boolean;
}) {
  return (
    <button
      onClick={(e) => {
        e.stopPropagation();
        if (!disabled && !pending) onChange(!enabled);
      }}
      disabled={disabled || pending}
      className={`relative h-6 w-10 shrink-0 rounded-full transition-colors ${
        pending
          ? "animate-pulse bg-indigo/50"
          : enabled
            ? "bg-indigo"
            : "bg-hairline"
      } ${disabled ? "cursor-not-allowed opacity-50" : "cursor-pointer"}`}
    >
      <span
        className={`absolute top-0.5 left-0.5 h-5 w-5 rounded-full bg-white transition-transform ${
          enabled || pending ? "translate-x-4" : ""
        }`}
      />
    </button>
  );
}

// ---------------------------------------------------------------------------
// TierDot
// ---------------------------------------------------------------------------

function TierDot({ tier }: { readonly tier: number }) {
  return <span className={`h-2 w-2 shrink-0 rounded-full ${tierDotColor(tier)}`} />;
}

// ---------------------------------------------------------------------------
// ConnectorDetailPanel — expanded view with sub-tabs
// ---------------------------------------------------------------------------

type DetailTab = "mapping" | "history" | "tools" | "logs";

function ConnectorDetailPanel({
  connectorId,
  details,
  detailsLoading,
}: {
  readonly connectorId: string;
  readonly details: ConnectorDetailData | null;
  readonly detailsLoading: boolean;
}) {
  const [activeDetailTab, setActiveDetailTab] = useState<DetailTab>("mapping");
  const [history, setHistory] = useState<ConnectorHistoryEntry[] | null>(null);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [logs, setLogs] = useState<readonly string[] | null>(null);
  const [logsLoading, setLogsLoading] = useState(false);

  // Fetch history on tab switch
  useEffect(() => {
    if (activeDetailTab !== "history") return;
    let cancelled = false;
    setHistoryLoading(true);
    dedupInvoke<ConnectorHistoryEntry[]>("get_connector_history", {
      connectorId,
      limit: 10,
    })
      .then((data) => { if (!cancelled) setHistory(data); })
      .catch(() => { if (!cancelled) setHistory([]); })
      .finally(() => { if (!cancelled) setHistoryLoading(false); });
    return () => { cancelled = true; };
  }, [activeDetailTab, connectorId]);

  // Fetch logs on tab switch
  useEffect(() => {
    if (activeDetailTab !== "logs") return;
    let cancelled = false;
    setLogsLoading(true);
    dedupInvoke<ExtensionLogOutput>("get_extension_logs", {
      extensionId: connectorId,
      lines: 50,
    })
      .then((data) => { if (!cancelled) setLogs(data.lines); })
      .catch(() => { if (!cancelled) setLogs([]); })
      .finally(() => { if (!cancelled) setLogsLoading(false); });
    return () => { cancelled = true; };
  }, [activeDetailTab, connectorId]);

  if (detailsLoading) {
    return (
      <div className="mt-2 space-y-2 rounded-2 border border-hairline bg-surface/60 px-4 py-3">
        <Skeleton className="h-4 w-48" />
        <Skeleton className="h-4 w-32" />
        <Skeleton className="h-8 w-full" />
      </div>
    );
  }

  if (!details) return null;

  const detailTabs: readonly { readonly id: DetailTab; readonly label: string }[] = [
    { id: "mapping", label: "Mapping" },
    { id: "history", label: "History" },
    { id: "tools", label: "Tools" },
    { id: "logs", label: "Logs" },
  ];

  return (
    <div className="mt-2 space-y-3 rounded-2 border border-hairline bg-surface/60 px-4 py-3">
      {/* Sub-tab bar */}
      <div className="flex gap-1 border-b border-hairline pb-2">
        {detailTabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveDetailTab(tab.id)}
            className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
              activeDetailTab === tab.id
                ? "bg-indigo-soft text-indigo"
                : "text-muted hover:text-ink"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Mapping tab */}
      {activeDetailTab === "mapping" && (
        <div className="overflow-x-auto">
          {details.tools.length === 0 ? (
            <p className="py-4 text-center text-xs text-muted">No field mappings available.</p>
          ) : (
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-hairline text-[11px] uppercase tracking-wider text-muted">
                  <th className="px-3 py-2">Source</th>
                  <th className="px-3 py-2">Target</th>
                  <th className="px-3 py-2">Type</th>
                  <th className="px-3 py-2">Tier</th>
                </tr>
              </thead>
              <tbody>
                {details.tools.flatMap((tool) =>
                  (tool.fields ?? []).map((field) => (
                    <tr
                      key={`${tool.tool_name}-${field.source}`}
                      className="border-b border-hairline/30"
                    >
                      <td className="px-3 py-2 font-mono text-muted">{field.source}</td>
                      <td className="px-3 py-2 font-mono text-ink">{field.target}</td>
                      <td className="px-3 py-2 text-muted">{field.type}</td>
                      <td className="px-3 py-2">
                        <span className="flex items-center gap-1.5">
                          <TierDot tier={field.tier} />
                          <span className="text-muted">{field.tier}</span>
                        </span>
                      </td>
                    </tr>
                  )),
                )}
              </tbody>
            </table>
          )}
        </div>
      )}

      {/* History tab */}
      {activeDetailTab === "history" && (
        <div className="overflow-x-auto">
          {historyLoading ? (
            <SkeletonTable rows={3} />
          ) : !history || history.length === 0 ? (
            <p className="py-4 text-center text-xs text-muted">No sync history yet.</p>
          ) : (
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-hairline text-[11px] uppercase tracking-wider text-muted">
                  <th className="px-3 py-2">Time</th>
                  <th className="px-3 py-2">Rows</th>
                  <th className="px-3 py-2">Duration</th>
                  <th className="px-3 py-2">Status</th>
                </tr>
              </thead>
              <tbody>
                {history.map((entry) => (
                  <tr key={entry.sync_id} className="border-b border-hairline/30">
                    <td className="whitespace-nowrap px-3 py-2 text-muted">
                      {formatTimestamp(entry.started_at)}
                    </td>
                    <td className="px-3 py-2 text-ink">{entry.rows_synced.toLocaleString()}</td>
                    <td className="px-3 py-2 text-muted">{entry.duration_seconds.toFixed(1)}s</td>
                    <td className="px-3 py-2">
                      <span
                        className={`text-xs font-medium ${
                          entry.status === "success"
                            ? "text-success"
                            : entry.status === "error"
                              ? "text-danger"
                              : "text-muted"
                        }`}
                      >
                        {entry.status}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      {/* Tools tab */}
      {activeDetailTab === "tools" && (
        <div className="space-y-2">
          {details.tools.length === 0 ? (
            <p className="py-4 text-center text-xs text-muted">No tools discovered.</p>
          ) : (
            details.tools.map((tool) => (
              <div
                key={tool.tool_name}
                className="flex items-center justify-between rounded-2 bg-surface/30 px-3 py-2"
              >
                <div className="flex items-center gap-2">
                  <Terminal className="h-3.5 w-3.5 text-muted" />
                  <span className="font-mono text-xs text-ink">{tool.tool_name}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span
                    className={`rounded px-2 py-0.5 text-[10px] font-semibold uppercase ${
                      tool.tool_type === "data"
                        ? "bg-indigo-soft text-indigo"
                        : "bg-amber/15 text-amber"
                    }`}
                  >
                    {tool.tool_type}
                  </span>
                  {tool.target_table && (
                    <span className="flex items-center gap-1 text-[11px] text-muted">
                      <Database className="h-3 w-3" />
                      {tool.target_table}
                    </span>
                  )}
                  <span className="text-[11px] text-muted">{tool.field_count} fields</span>
                </div>
              </div>
            ))
          )}
        </div>
      )}

      {/* Logs tab */}
      {activeDetailTab === "logs" && (
        <div>
          {logsLoading ? (
            <Skeleton className="h-40 w-full" />
          ) : !logs || logs.length === 0 ? (
            <p className="py-4 text-center text-xs text-muted">No logs available.</p>
          ) : (
            <div className="max-h-60 overflow-y-auto rounded-2 bg-bg p-3">
              <pre className="whitespace-pre-wrap font-mono text-[11px] leading-relaxed text-muted">
                {logs.join("\n")}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// ConnectorRow
// ---------------------------------------------------------------------------

function ConnectorRow({
  connector,
  expanded,
  details,
  detailsLoading,
  onToggle,
  onToggleWithInputs,
  onExpand,
  onSyncNow,
  onSettings,
  onRemove,
  onOpenPermissionSettings,
  togglePending,
  syncPending,
  removePending,
  whatsappPhoneValue,
  whatsappPhonePrompt,
  whatsappPhoneSaving,
  onWhatsappPhoneChange,
  onSaveWhatsappPhone,
}: {
  readonly connector: ConnectorCatalogEntry;
  readonly expanded: boolean;
  readonly details: ConnectorDetailData | null;
  readonly detailsLoading: boolean;
  readonly onToggle: (enabled: boolean) => void;
  readonly onToggleWithInputs: (inputs: Record<string, string>) => void;
  readonly onExpand: () => void;
  readonly onSyncNow: () => void;
  readonly onSettings: () => void;
  readonly onRemove: () => void;
  readonly onOpenPermissionSettings: (permission: string) => Promise<void>;
  readonly togglePending: boolean;
  readonly syncPending: boolean;
  readonly removePending: boolean;
  readonly whatsappPhoneValue?: string;
  readonly whatsappPhonePrompt?: boolean;
  readonly whatsappPhoneSaving?: boolean;
  readonly onWhatsappPhoneChange?: (value: string) => void;
  readonly onSaveWhatsappPhone?: () => void;
}) {
  const isWhatsApp = connector.connector_id === WHATSAPP_CONNECTOR_ID;
  const isConnected = connector.status === "connected";
  const needsSetup = connector.status === "needs_setup";
  const hasError = connector.status === "error";
  const hasWhatsappPhone = normalizeWhatsappPhone(whatsappPhoneValue ?? "").length > 0;
  const [envValues, setEnvValues] = useState<Record<string, string>>({});
  const [confirmRemove, setConfirmRemove] = useState(false);

  return (
    <div className="rounded-4 border border-hairline bg-surface">
      {/* Main row */}
      <div
        onClick={isConnected ? onExpand : undefined}
        className={`flex items-center justify-between px-4 py-3 transition-colors ${
          isConnected ? "cursor-pointer hover:bg-surface/80" : "cursor-default"
        }`}
      >
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-3">
            <span className="text-lg">{connector.icon}</span>
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium text-ink">{connector.name}</span>
                <StatusBadge status={connector.status} />
              </div>
              <p className="mt-0.5 text-[11px] text-muted">{connector.description}</p>
            </div>
          </div>

          {/* Stats line */}
          {isConnected && (
            <div className="mt-1.5 flex flex-wrap items-center gap-3 pl-9 text-[11px] text-muted">
              {connector.stats.records_synced > 0 && (
                <span className="flex items-center gap-1">
                  <Database className="h-3 w-3" />
                  {connector.stats.records_synced.toLocaleString()} synced
                </span>
              )}
              {connector.stats.last_sync && (
                <span className="flex items-center gap-1">
                  <Clock className="h-3 w-3" />
                  {formatRelativeTime(connector.stats.last_sync)}
                </span>
              )}
              {connector.stats.next_sync && (
                <span className="flex items-center gap-1">
                  Next: {formatRelativeTime(connector.stats.next_sync)}
                </span>
              )}
              <span>{connector.tools_available} tools</span>
            </div>
          )}

          {/* Error */}
          {hasError && (
            <p className="mt-1 pl-9 text-[11px] text-amber">
              Connection error. Try toggling off and on.
            </p>
          )}
        </div>

        {/* Right side controls */}
        <div className="flex items-center gap-2">
          {isConnected && (
            <>
              <button
                onClick={(e) => { e.stopPropagation(); onSyncNow(); }}
                disabled={syncPending}
                className="flex items-center gap-1 rounded-2 px-2.5 py-1.5 text-xs text-muted transition-colors hover:bg-hairline hover:text-ink"
                title="Sync Now"
              >
                <RefreshCw className={`h-3 w-3 ${syncPending ? "animate-spin" : ""}`} />
              </button>
              <button
                onClick={(e) => { e.stopPropagation(); onSettings(); }}
                className="flex items-center gap-1 rounded-2 px-2.5 py-1.5 text-xs text-muted transition-colors hover:bg-hairline hover:text-ink"
                title="Settings"
              >
                <Settings className="h-3 w-3" />
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  if (confirmRemove) onRemove();
                  else setConfirmRemove(true);
                }}
                disabled={removePending}
                className={`flex items-center gap-1 rounded-2 px-2.5 py-1.5 text-xs transition-colors ${
                  confirmRemove
                    ? "bg-amber/15 text-amber"
                    : "text-muted hover:bg-hairline hover:text-ink"
                }`}
                title={confirmRemove ? "Click again to confirm" : "Remove"}
                onBlur={() => setConfirmRemove(false)}
              >
                {removePending ? (
                  <Loader2 className="h-3 w-3 animate-spin" />
                ) : (
                  <Trash2 className="h-3 w-3" strokeWidth={1.6} />
                )}
              </button>
            </>
          )}
          <Toggle
            enabled={connector.enabled}
            onChange={onToggle}
            pending={togglePending}
          />
          {isConnected && (
            expanded
              ? <ChevronDown className="h-4 w-4 text-muted" />
              : <ChevronRight className="h-4 w-4 text-muted" />
          )}
        </div>
      </div>

      {/* Requirement card (needs_setup) */}
      {needsSetup && connector.missing_requirements.length > 0 && (
        <div className="border-t border-hairline px-4 py-3">
          <div className="space-y-3">
            {connector.missing_requirements.map((req) => {
              const Icon = requirementIcon(req.type);
              return (
                <div key={req.key} className="space-y-1">
                  <div className="flex items-center gap-2 text-xs text-ink">
                    <Icon className="h-3.5 w-3.5 shrink-0 text-muted" />
                    <span>{req.label}</span>
                  </div>
                  {req.type === "permission" && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        void onOpenPermissionSettings(req.key);
                      }}
                      className="ml-5 inline-flex items-center gap-1 text-xs text-indigo transition-colors hover:text-indigo/80"
                    >
                      Open Privacy Settings
                      <ExternalLink className="h-3 w-3" />
                    </button>
                  )}
                  {req.type === "oauth" && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        // `start_oauth` is what actually launches the
                        // PKCE flow — see
                        // ConnectionManager._hydrate_oauth_requirements.
                        // Without it the retry below only re-runs the
                        // requirements check, which finds the same
                        // missing token and leaves the connector stuck
                        // on "Setup Required" forever.
                        onToggleWithInputs({
                          ...envValues,
                          start_oauth: "true",
                          oauth_provider: req.key,
                        });
                      }}
                      className="ml-5 inline-flex items-center gap-1 text-xs text-indigo transition-colors hover:text-indigo/80"
                    >
                      {req.label}
                      <ExternalLink className="h-3 w-3" />
                    </button>
                  )}
                  {req.type === "env" && (
                    <div className="flex items-center gap-2 pl-5">
                      <input
                        type="text"
                        placeholder={req.label}
                        value={envValues[req.key] ?? ""}
                        onChange={(e) =>
                          setEnvValues((prev) => ({ ...prev, [req.key]: e.target.value }))
                        }
                        className="flex-1 rounded-2 bg-bg px-3 py-1.5 text-xs text-ink placeholder-muted outline-none ring-1 ring-hairline focus:ring-indigo"
                      />
                    </div>
                  )}
                </div>
              );
            })}
            <button
              onClick={() => onToggleWithInputs(envValues)}
              className="rounded-2 bg-indigo px-3 py-1.5 text-xs font-medium text-white transition-colors hover:bg-indigo/90"
            >
              Retry Connection
            </button>
          </div>
        </div>
      )}

      {isWhatsApp && (
        <div className="border-t border-hairline px-4 py-3">
          <div className="space-y-2">
            <div className="flex items-center justify-between gap-2">
              <p className="text-xs font-medium text-ink">
                WhatsApp notification number
              </p>
              <span
                className={`rounded-full px-2 py-0.5 text-[10px] font-medium ${
                  hasWhatsappPhone
                    ? "bg-success/15 text-success"
                    : "bg-amber/15 text-amber"
                }`}
              >
                {hasWhatsappPhone ? "Saved" : "Required on first enable"}
              </span>
            </div>

            <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
              <input
                type="tel"
                value={whatsappPhoneValue ?? ""}
                onChange={(e) => onWhatsappPhoneChange?.(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault();
                    onSaveWhatsappPhone?.();
                  }
                }}
                placeholder="+1234567890"
                className="flex-1 rounded-2 bg-bg px-3 py-1.5 text-xs text-ink placeholder-muted outline-none ring-1 ring-hairline focus:ring-indigo"
              />
              <button
                onClick={() => onSaveWhatsappPhone?.()}
                disabled={whatsappPhoneSaving || !hasWhatsappPhone}
                className="inline-flex items-center justify-center rounded-2 bg-indigo px-3 py-1.5 text-xs font-medium text-white transition-colors hover:bg-indigo/90 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {whatsappPhoneSaving ? "Saving..." : "Save Number"}
              </button>
            </div>

            {whatsappPhonePrompt && (
              <p className="text-[11px] text-amber">
                Add your number and save it to finish turning WhatsApp on.
              </p>
            )}

            <WhatsAppPairingPanel enabled={hasWhatsappPhone} />
          </div>
        </div>
      )}

      {/* Expanded detail panel */}
      {expanded && isConnected && (
        <div className="border-t border-hairline px-4 py-3">
          <ConnectorDetailPanel
            connectorId={connector.connector_id}
            details={details}
            detailsLoading={detailsLoading}
          />
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// ConnectorsTab
// ---------------------------------------------------------------------------

function ConnectorsTab({
  onOpenConfig,
}: {
  readonly onOpenConfig: (connectorId: string) => void;
}) {
  const [search, setSearch] = useState("");
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [pendingToggles, setPendingToggles] = useState<Set<string>>(() => new Set());
  const [pendingSyncs, setPendingSyncs] = useState<Set<string>>(() => new Set());
  const [pendingRemoves, setPendingRemoves] = useState<Set<string>>(() => new Set());
  const [error, setError] = useState<string | null>(null);
  const [whatsappPhoneDraft, setWhatsappPhoneDraft] = useState<string | null>(null);
  const [whatsappPhonePrompt, setWhatsappPhonePrompt] = useState(false);
  const [whatsappPhoneSaving, setWhatsappPhoneSaving] = useState(false);
  const [queuedWhatsappEnable, setQueuedWhatsappEnable] = useState(false);

  const [details, setDetails] = useState<ConnectorDetailData | null>(null);
  const [detailsLoading, setDetailsLoading] = useState(false);

  const {
    data: catalog,
    error: catalogError,
    refetch: refetchCatalog,
    isLoading,
  } = useAsyncData<ConnectorCatalogEntry[]>(
    useCallback(() => dedupInvoke<ConnectorCatalogEntry[]>("get_connector_catalog"), []),
  );

  const settingsResult = useAsyncData<AppSettings>(
    useCallback(() => dedupInvoke<AppSettings>("get_settings"), []),
  );

  // Fetch details on expand
  useEffect(() => {
    if (!expandedId) {
      setDetails(null);
      return;
    }
    let cancelled = false;
    setDetailsLoading(true);
    invoke<ConnectorDetailData>("get_connector_details", { connectorId: expandedId })
      .then((result) => { if (!cancelled) { setDetails(result); setDetailsLoading(false); } })
      .catch(() => { if (!cancelled) { setDetails(null); setDetailsLoading(false); } });
    return () => { cancelled = true; };
  }, [expandedId]);

  const filtered = useMemo(() => {
    if (!catalog) return [];
    const q = search.toLowerCase();
    if (!q) return catalog;
    return catalog.filter(
      (c) => c.name.toLowerCase().includes(q) || c.description.toLowerCase().includes(q),
    );
  }, [catalog, search]);

  const persistedWhatsappPhone = settingsResult.data?.whatsapp_notification_phone ?? "";
  const whatsappPhoneValue = whatsappPhoneDraft ?? persistedWhatsappPhone;

  // sensitivity_tier: 1
  const openPermissionSettings = useCallback(
    async (permission: string) => {
      try {
        await invoke("open_macos_permission_settings", { permission });
      } catch (err) {
        console.error("open_macos_permission_settings failed:", err);
      }
    },
    [],
  );

  // sensitivity_tier: 1
  const performToggle = useCallback(
    async (connector: ConnectorCatalogEntry, enabled: boolean) => {
      const connectorId = connector.connector_id;
      setError(null);
      setPendingToggles((prev) => new Set(prev).add(connectorId));
      try {
        if (enabled) {
          const permissionRequirement = connector.missing_requirements.find((r) => r.type === "permission");
          const permission =
            permissionRequirement?.key ??
            CONNECTOR_PERMISSION_BY_ID[connectorId];
          if (permission) {
            await openPermissionSettings(permission);
          }
        }
        const result = await invoke<{ status?: string; error?: string }>(
          "toggle_connector",
          { connectorId, enabled },
        );
        if (result?.status === "error" && result.error) {
          setError(result.error);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setPendingToggles((prev) => { const n = new Set(prev); n.delete(connectorId); return n; });
        refetchCatalog();
      }
    },
    [openPermissionSettings, refetchCatalog],
  );

  // sensitivity_tier: 1
  const saveWhatsappPhone = useCallback(async (): Promise<boolean> => {
    const normalized = normalizeWhatsappPhone(whatsappPhoneValue);
    if (!normalized) {
      setError("Enter a WhatsApp phone number in international format (example: +1234567890).");
      return false;
    }

    setError(null);
    setWhatsappPhoneSaving(true);
    try {
      const currentSettings = settingsResult.data ?? await invoke<AppSettings>("get_settings");
      await invoke("update_settings", {
        settings: {
          ...currentSettings,
          whatsapp_notification_phone: normalized,
        },
      });

      setWhatsappPhoneDraft(normalized);
      setWhatsappPhonePrompt(false);
      void settingsResult.refetch();

      if (queuedWhatsappEnable) {
        const whatsappConnector = catalog?.find(
          (entry) => entry.connector_id === WHATSAPP_CONNECTOR_ID,
        );
        setQueuedWhatsappEnable(false);
        if (whatsappConnector && !whatsappConnector.enabled) {
          await performToggle(whatsappConnector, true);
        }
      }
      return true;
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      return false;
    } finally {
      setWhatsappPhoneSaving(false);
    }
  }, [
    catalog,
    performToggle,
    queuedWhatsappEnable,
    settingsResult,
    whatsappPhoneValue,
  ]);

  // sensitivity_tier: 1
  const handleToggle = useCallback(
    async (connector: ConnectorCatalogEntry, enabled: boolean) => {
      if (
        enabled
        && connector.connector_id === WHATSAPP_CONNECTOR_ID
        && !normalizeWhatsappPhone(whatsappPhoneValue)
      ) {
        setError("WhatsApp needs your notification number before it can be enabled.");
        setQueuedWhatsappEnable(true);
        setWhatsappPhonePrompt(true);
        return;
      }

      if (connector.connector_id === WHATSAPP_CONNECTOR_ID) {
        setQueuedWhatsappEnable(false);
        setWhatsappPhonePrompt(false);
      }

      await performToggle(connector, enabled);
    },
    [performToggle, whatsappPhoneValue],
  );

  // sensitivity_tier: 1
  const handleToggleWithInputs = useCallback(
    async (connectorId: string, userInputs: Record<string, string>) => {
      setError(null);
      setPendingToggles((prev) => new Set(prev).add(connectorId));
      try {
        const result = await invoke<{ status?: string; error?: string }>(
          "toggle_connector",
          { connectorId, enabled: true, userInputs },
        );
        if (result?.status === "error" && result.error) {
          setError(result.error);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setPendingToggles((prev) => { const n = new Set(prev); n.delete(connectorId); return n; });
        refetchCatalog();
      }
    },
    [refetchCatalog],
  );

  // sensitivity_tier: 1
  const handleSyncNow = useCallback(
    async (connectorId: string) => {
      setPendingSyncs((prev) => new Set(prev).add(connectorId));
      try {
        await invoke("sync_connector_now", { connectorId });
      } catch (err) {
        console.error("sync_connector_now failed:", err);
      } finally {
        setPendingSyncs((prev) => { const n = new Set(prev); n.delete(connectorId); return n; });
        refetchCatalog();
      }
    },
    [refetchCatalog],
  );

  // sensitivity_tier: 1
  const handleRemove = useCallback(
    async (connectorId: string) => {
      setPendingRemoves((prev) => new Set(prev).add(connectorId));
      try {
        await invoke("uninstall_extension", { connectorId, preserveData: true });
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setPendingRemoves((prev) => { const n = new Set(prev); n.delete(connectorId); return n; });
        setExpandedId(null);
        refetchCatalog();
      }
    },
    [refetchCatalog],
  );

  const connectedCount = catalog ? catalog.filter((c) => c.status === "connected").length : 0;

  return (
    <div className="space-y-4">
      {/* Actions bar */}
      <div className="flex items-center gap-3">
        <div className="flex flex-1 items-center gap-2 rounded-2 bg-surface px-3 py-2">
          <Search className="h-4 w-4 shrink-0 text-muted" strokeWidth={1.6} />
          <input
            type="text"
            placeholder="Search connectors..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="flex-1 bg-transparent text-sm text-ink placeholder-muted outline-none"
          />
          {search && (
            <button onClick={() => setSearch("")} className="text-xs text-muted hover:text-ink">
              &times;
            </button>
          )}
        </div>
      </div>

      {/* Error banner */}
      {error && (
        <div className="flex items-center justify-between rounded-2 border border-amber/30 bg-amber/5 px-4 py-2 text-xs text-amber">
          <span>{error}</span>
          <button onClick={() => setError(null)} className="ml-2 text-muted hover:text-ink">&times;</button>
        </div>
      )}

      {/* Connected summary */}
      {!isLoading && catalog && (
        <p className="text-xs text-muted">
          {connectedCount} of {catalog.length} connectors active
        </p>
      )}

      {/* Connector list */}
      {isLoading && !catalog ? (
        <div className="space-y-3">
          <SkeletonSection />
          <SkeletonSection />
        </div>
      ) : catalogError && !catalog ? (
        <div className="rounded-4 border border-amber/30 bg-amber/5 px-4 py-3 text-sm text-amber">
          {catalogError}
        </div>
      ) : filtered.length === 0 && search ? (
        <p className="py-8 text-center text-sm text-muted">
          No connectors match &ldquo;{search}&rdquo;
        </p>
      ) : (
        <div className="space-y-3">
          {filtered.map((connector) => (
            <ConnectorRow
              key={connector.connector_id}
              connector={connector}
              expanded={expandedId === connector.connector_id}
              details={expandedId === connector.connector_id ? details : null}
              detailsLoading={expandedId === connector.connector_id && detailsLoading}
              onToggle={(enabled) => handleToggle(connector, enabled)}
              onToggleWithInputs={(inputs) => handleToggleWithInputs(connector.connector_id, inputs)}
              onExpand={() => setExpandedId(expandedId === connector.connector_id ? null : connector.connector_id)}
              onSyncNow={() => handleSyncNow(connector.connector_id)}
              onSettings={() => onOpenConfig(connector.connector_id)}
              onRemove={() => handleRemove(connector.connector_id)}
              onOpenPermissionSettings={openPermissionSettings}
              togglePending={pendingToggles.has(connector.connector_id)}
              syncPending={pendingSyncs.has(connector.connector_id)}
              removePending={pendingRemoves.has(connector.connector_id)}
              whatsappPhoneValue={connector.connector_id === WHATSAPP_CONNECTOR_ID ? whatsappPhoneValue : undefined}
              whatsappPhonePrompt={
                connector.connector_id === WHATSAPP_CONNECTOR_ID
                  ? whatsappPhonePrompt
                  : undefined
              }
              whatsappPhoneSaving={
                connector.connector_id === WHATSAPP_CONNECTOR_ID
                  ? whatsappPhoneSaving
                  : undefined
              }
              onWhatsappPhoneChange={
                connector.connector_id === WHATSAPP_CONNECTOR_ID
                  ? (value) => setWhatsappPhoneDraft(value)
                  : undefined
              }
              onSaveWhatsappPhone={
                connector.connector_id === WHATSAPP_CONNECTOR_ID
                  ? () => { void saveWhatsappPhone(); }
                  : undefined
              }
            />
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Discover tab — browse and install MCP connectors by category
// ---------------------------------------------------------------------------

interface DiscoverConnector {
  readonly name: string;
  readonly command: string;
  readonly description: string;
  readonly icon: string;
  readonly note?: string;
  readonly requiresEnv?: ReadonlyArray<EnvRequirement>;
}

interface DiscoverCategory {
  readonly id: string;
  readonly label: string;
  readonly description: string;
  readonly icon: React.ReactNode;
  readonly connectors: readonly DiscoverConnector[];
}

// Catalog verified May 2026. Many packages from the original list were
// removed because their npm package never existed (the `@anthropic/*`
// scope) or was archived by upstream. Where a credible community
// replacement existed, the command and env requirements were updated.
const DISCOVER_CATEGORIES: readonly DiscoverCategory[] = [
  {
    id: "communication",
    label: "Communication",
    description: "Email, messaging, and chat",
    icon: <MessageSquare className="h-4 w-4" />,
    connectors: [
      {
        name: "Slack",
        command: "npx -y slack-mcp-server@latest --transport stdio",
        description: "Messages, channels, and workspace search",
        icon: "\u{1F4AC}",
        note: "Maintained fork by korotovsky (the official package was archived)",
        requiresEnv: [
          {
            key: "SLACK_MCP_XOXP_TOKEN",
            label: "Slack OAuth user token (xoxp-…)",
            helpUrl: "https://github.com/korotovsky/slack-mcp-server/blob/master/docs/03-authentication-setup.md",
          },
        ],
      },
    ],
  },
  {
    id: "productivity",
    label: "Work & Productivity",
    description: "Project management, notes, and CRM",
    icon: <Briefcase className="h-4 w-4" />,
    connectors: [
      {
        name: "Notion",
        command: "npx -y @notionhq/notion-mcp-server",
        description: "Pages, databases, and workspace content",
        icon: "\u{1F4D3}",
        note: "Official Notion package",
        requiresEnv: [
          {
            key: "OPENAPI_MCP_HEADERS",
            label: 'JSON headers, e.g. {"Authorization":"Bearer ntn_…","Notion-Version":"2022-06-28"}',
            helpUrl: "https://developers.notion.com/docs/get-started-with-mcp",
          },
        ],
      },
      {
        name: "Linear",
        command: "npx -y @tacticlaunch/mcp-linear",
        description: "Issues, projects, and team workflows",
        icon: "\u{1F4D0}",
        requiresEnv: [
          {
            key: "LINEAR_API_KEY",
            label: "Personal Linear API key",
            helpUrl: "https://linear.app/settings/api",
          },
        ],
      },
      {
        name: "Jira",
        command: "npx -y @aashari/mcp-server-atlassian-jira",
        description: "Boards, issues, sprints, and JQL search",
        icon: "\u{1F4CB}",
        requiresEnv: [
          {
            key: "ATLASSIAN_SITE_NAME",
            label: "e.g. your-team (from your-team.atlassian.net)",
          },
          {
            key: "ATLASSIAN_USER_EMAIL",
            label: "Your Atlassian account email",
          },
          {
            key: "ATLASSIAN_API_TOKEN",
            label: "API token",
            helpUrl: "https://id.atlassian.com/manage-profile/security/api-tokens",
          },
        ],
      },
      {
        name: "HubSpot",
        command: "npx -y @hubspot/mcp-server",
        description: "CRM contacts, deals, and pipelines",
        icon: "\u{1F91D}",
        note: "Official HubSpot package (public beta)",
        requiresEnv: [
          {
            key: "PRIVATE_APP_ACCESS_TOKEN",
            label: "HubSpot Private App access token",
            helpUrl: "https://developers.hubspot.com/docs/api/private-apps",
          },
        ],
      },
      {
        name: "Airtable",
        command: "npx -y airtable-mcp-server",
        description: "Bases, tables, and records",
        icon: "\u{1F4CA}",
        note: "Community package by domdomegg",
        requiresEnv: [
          {
            key: "AIRTABLE_API_KEY",
            label: "Personal access token",
            helpUrl: "https://airtable.com/create/tokens",
          },
        ],
      },
    ],
  },
  {
    id: "health",
    label: "Health & Fitness",
    description: "Wellness tracking and health data",
    icon: <Heart className="h-4 w-4" />,
    connectors: [
      {
        name: "Strava",
        command: "npx -y @r-huijts/strava-mcp-server",
        description: "Activities, routes, and training data",
        icon: "\u{1F3C3}",
        requiresEnv: [
          { key: "STRAVA_CLIENT_ID", label: "API application client ID" },
          { key: "STRAVA_CLIENT_SECRET", label: "API application client secret" },
          {
            key: "STRAVA_REFRESH_TOKEN",
            label: "OAuth refresh token",
            helpUrl: "https://www.strava.com/settings/api",
          },
        ],
      },
      {
        name: "Oura Ring",
        command: "npx -y oura-ring-mcp",
        description: "Sleep, readiness, and activity scores",
        icon: "\u{1F48D}",
        requiresEnv: [
          {
            key: "OURA_ACCESS_TOKEN",
            label: "Personal access token",
            helpUrl: "https://cloud.ouraring.com/personal-access-tokens",
          },
        ],
      },
    ],
  },
  {
    id: "developer",
    label: "Developer Tools",
    description: "Code, CI/CD, and infrastructure",
    icon: <Code2 className="h-4 w-4" />,
    connectors: [
      {
        name: "GitLab",
        command: "npx -y @davidfei/gitlab-mcp",
        description: "Projects, merge requests, and CI",
        icon: "\u{1F98A}",
        requiresEnv: [
          {
            key: "GITLAB_PERSONAL_ACCESS_TOKEN",
            label: "Personal access token with read_api scope",
            helpUrl: "https://docs.gitlab.com/ee/user/profile/personal_access_tokens.html",
          },
          {
            key: "GITLAB_API_URL",
            label: "Optional — defaults to https://gitlab.com/api/v4",
          },
        ],
      },
      {
        name: "Sentry",
        command: "npx -y @sentry/mcp-server",
        description: "Error tracking and performance",
        icon: "\u{1F50D}",
        note: "Official Sentry package",
        requiresEnv: [
          {
            key: "SENTRY_AUTH_TOKEN",
            label: "User auth token",
            helpUrl: "https://sentry.io/orgredirect/organizations/:orgslug/settings/auth-tokens/",
          },
        ],
      },
      {
        name: "Playwright",
        command: "npx -y @playwright/mcp",
        description: "Browser automation and testing",
        icon: "\u{1F3AD}",
        note: "Official Microsoft Playwright MCP",
      },
    ],
  },
  {
    id: "data",
    label: "Data & Storage",
    description: "Databases, files, and cloud storage",
    icon: <Database className="h-4 w-4" />,
    connectors: [
      {
        name: "Supabase",
        command: "npx -y @supabase/mcp-server-supabase",
        description: "Database, auth, and edge functions",
        icon: "⚡",
        requiresEnv: [
          {
            key: "SUPABASE_ACCESS_TOKEN",
            label: "Personal access token",
            helpUrl: "https://supabase.com/dashboard/account/tokens",
          },
        ],
      },
    ],
  },
  {
    id: "finance",
    label: "Finance",
    description: "Payments, banking, and markets",
    icon: <CreditCard className="h-4 w-4" />,
    connectors: [
      {
        name: "Stripe",
        command: "npx -y @stripe/mcp --tools=all",
        description: "Payments, invoices, and subscriptions",
        icon: "\u{1F4B3}",
        note: "Official Stripe package",
        requiresEnv: [
          {
            key: "STRIPE_SECRET_KEY",
            label: "Secret API key (sk_test_… or sk_live_…)",
            helpUrl: "https://dashboard.stripe.com/apikeys",
          },
        ],
      },
      {
        name: "PayPal",
        command: "npx -y @paypal/mcp --tools=all",
        description: "Transactions and account management",
        icon: "\u{1F4B0}",
        note: "Official PayPal package",
        requiresEnv: [
          { key: "PAYPAL_ACCESS_TOKEN", label: "OAuth access token" },
          {
            key: "PAYPAL_ENVIRONMENT",
            label: "SANDBOX or LIVE",
          },
        ],
      },
    ],
  },
  {
    id: "web",
    label: "Web & Search",
    description: "Web scraping, search, and content",
    icon: <Globe className="h-4 w-4" />,
    connectors: [
      {
        name: "Firecrawl",
        command: "npx -y firecrawl-mcp",
        description: "Scrape and extract structured web content",
        icon: "\u{1F525}",
        requiresEnv: [
          {
            key: "FIRECRAWL_API_KEY",
            label: "API key",
            helpUrl: "https://www.firecrawl.dev/app/api-keys",
          },
        ],
      },
      {
        name: "Tavily",
        command: "npx -y tavily-mcp",
        description: "AI-powered web search and extraction",
        icon: "\u{1F50E}",
        requiresEnv: [
          {
            key: "TAVILY_API_KEY",
            label: "API key",
            helpUrl: "https://app.tavily.com/home",
          },
        ],
      },
    ],
  },
];

interface DiscoverInstallTarget {
  /**
   * Preset command to pre-fill and auto-discover. Omitted for the
   * free-form path, where the modal opens with an empty, editable
   * command field — the "paste any MCP command" flow. The modal always
   * supported this (`initialCommand` is optional, and gates only the
   * auto-discover), but nothing ever opened it that way, so the
   * advertised escape hatch had no entry point at all once the orphaned
   * DataSourcesPage was removed.
   */
  readonly command?: string;
  readonly requiresEnv?: ReadonlyArray<EnvRequirement>;
}

function DiscoverConnectorCard({
  connector,
  onInstall,
}: {
  readonly connector: DiscoverConnector;
  readonly onInstall: (target: DiscoverInstallTarget) => void;
}) {
  return (
    <div className="flex items-center justify-between rounded-2 border border-hairline bg-surface px-3 py-2.5 transition-colors hover:border-indigo/30">
      <div className="flex items-center gap-3">
        <span className="text-lg">{connector.icon}</span>
        <div>
          <p className="text-xs font-medium text-ink">{connector.name}</p>
          <p className="text-[10px] text-muted">{connector.description}</p>
          {connector.note && (
            <p className="text-[9px] text-indigo">{connector.note}</p>
          )}
        </div>
      </div>
      <button
        onClick={() =>
          onInstall({
            command: connector.command,
            requiresEnv: connector.requiresEnv,
          })
        }
        className="flex shrink-0 items-center gap-1 rounded-md bg-surface border border-hairline-2 px-2.5 py-1 text-[11px] font-medium text-ink shadow-1 transition-colors hover:bg-bg-2"
      >
        <Plug className="h-3 w-3" />
        Install
      </button>
    </div>
  );
}

function DiscoverTab({
  onInstall,
  onAddCustom,
}: {
  readonly onInstall: (target: DiscoverInstallTarget) => void;
  readonly onAddCustom: () => void;
}) {
  const [search, setSearch] = useState("");

  const filtered = useMemo(() => {
    if (!search) return DISCOVER_CATEGORIES;
    const q = search.toLowerCase();
    return DISCOVER_CATEGORIES
      .map((cat) => ({
        ...cat,
        connectors: cat.connectors.filter(
          (c) =>
            c.name.toLowerCase().includes(q) ||
            c.description.toLowerCase().includes(q),
        ),
      }))
      .filter((cat) => cat.connectors.length > 0);
  }, [search]);

  return (
    <div className="space-y-4">
      {/* Search */}
      <div className="flex items-center gap-3">
        <div className="flex flex-1 items-center gap-2 rounded-2 bg-surface px-3 py-2">
          <Search className="h-4 w-4 shrink-0 text-muted" strokeWidth={1.6} />
          <input
            type="text"
            placeholder="Search connectors..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="flex-1 bg-transparent text-sm text-ink placeholder-muted outline-none"
          />
          {search && (
            <button onClick={() => setSearch("")} className="text-xs text-muted hover:text-ink">
              &times;
            </button>
          )}
        </div>
        <button
          onClick={onAddCustom}
          className="flex shrink-0 items-center gap-1.5 rounded-2 border border-hairline px-3 py-2 text-xs font-medium text-ink transition-colors hover:border-indigo/40 hover:bg-surface"
        >
          <Plus className="h-3.5 w-3.5" strokeWidth={1.6} />
          Add custom
        </button>
        <a
          href="https://mcp.so"
          target="_blank"
          rel="noopener noreferrer"
          className="flex shrink-0 items-center gap-1.5 rounded-2 border border-hairline px-3 py-2 text-xs text-muted transition-colors hover:border-indigo/40 hover:text-ink"
        >
          Browse all
          <ExternalLink className="h-3 w-3" />
        </a>
      </div>

      {/* Categories */}
      {filtered.length === 0 ? (
        <p className="py-8 text-center text-sm text-muted">
          No connectors match &ldquo;{search}&rdquo;
        </p>
      ) : (
        <div className="space-y-5">
          {filtered.map((cat) => (
            <section key={cat.id}>
              <div className="mb-2 flex items-center gap-2">
                <span className="text-indigo">{cat.icon}</span>
                <h3 className="text-sm font-medium text-ink">{cat.label}</h3>
                <span className="text-[10px] text-muted">{cat.description}</span>
              </div>
              <div className="grid grid-cols-1 gap-2 lg:grid-cols-2">
                {cat.connectors.map((c) => (
                  <DiscoverConnectorCard key={c.name} connector={c} onInstall={onInstall} />
                ))}
              </div>
            </section>
          ))}
        </div>
      )}

      <div className="rounded-2 border border-dashed border-hairline bg-surface/30 p-4">
        <p className="text-xs text-muted">
          Don&rsquo;t see what you need? Any MCP-compatible server works —{" "}
          <button
            onClick={onAddCustom}
            className="font-medium text-indigo-2 underline decoration-dotted underline-offset-2 hover:text-indigo"
          >
            add a custom connector
          </button>{" "}
          and paste its command. Arandu probes the server, infers a schema from
          a sample of its output, and shows you the mapping before anything is
          stored.
        </p>
        <p className="mt-2 text-[10px] text-muted">
          Google Calendar and Gmail are available as native connectors above —
          no MCP server needed. Third-party servers that run an interactive
          OAuth browser flow on first launch may time out during discovery; run
          them once via their own CLI to cache credentials, then add them here.
        </p>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------

type TabId = "my-connectors" | "discover";

const TABS: readonly { readonly id: TabId; readonly label: string; readonly icon: LucideIcon }[] = [
  { id: "my-connectors", label: "My Connectors", icon: Plug },
  { id: "discover", label: "Discover", icon: Compass },
];

// ---------------------------------------------------------------------------
// ConnectorsPage — main component
// ---------------------------------------------------------------------------

function ConnectorsPage() {
  const [activeTab, setActiveTab] = useState<TabId>("my-connectors");
  const [installTarget, setInstallTarget] =
    useState<DiscoverInstallTarget | null>(null);
  const [configConnectorId, setConfigConnectorId] = useState<string | null>(null);

  const handleDiscoverInstall = useCallback(
    (target: DiscoverInstallTarget) => {
      setInstallTarget(target);
    },
    [],
  );

  // Free-form path: open the same modal with no preset, so the command
  // field starts empty and editable and nothing auto-discovers.
  const handleAddCustom = useCallback(() => {
    setInstallTarget({});
  }, []);

  return (
    <div className="flex-1 space-y-6 overflow-y-auto p-6">
      {/* Header */}
      <div>
        <h2 className="text-[44px] font-bold leading-none" style={{ background: "linear-gradient(135deg, var(--ink), var(--ink-2))", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text" }}>Connectors</h2>
        <p className="mt-1 text-sm text-muted">
          Manage data source integrations. Everything runs locally on your device.
        </p>
      </div>

      {/* Tab bar */}
      <div className="flex gap-1 border-b border-hairline pb-0">
        {TABS.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setActiveTab(id)}
            className={`flex items-center gap-2 border-b-2 px-4 py-2.5 text-sm font-medium transition-colors ${
              activeTab === id
                ? "border-indigo text-indigo-2"
                : "border-transparent text-muted hover:text-ink"
            }`}
          >
            <Icon className="h-4 w-4" strokeWidth={1.6} />
            {label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {activeTab === "my-connectors" && (
        <ConnectorsTab onOpenConfig={(id) => setConfigConnectorId(id)} />
      )}
      {activeTab === "discover" && (
        <DiscoverTab
          onInstall={handleDiscoverInstall}
          onAddCustom={handleAddCustom}
        />
      )}

      {/* Modals */}
      {installTarget !== null && (
        <InstallExtensionModal
          open
          initialCommand={installTarget.command}
          initialRequiresEnv={installTarget.requiresEnv}
          onClose={() => setInstallTarget(null)}
          onInstalled={() => {
            setInstallTarget(null);
            setActiveTab("my-connectors");
          }}
        />
      )}
      {configConnectorId && (
        <ConnectorConfigModal
          connectorId={configConnectorId}
          open={!!configConnectorId}
          onClose={() => setConfigConnectorId(null)}
          onSaved={() => setConfigConnectorId(null)}
        />
      )}
    </div>
  );
}

export default ConnectorsPage;
