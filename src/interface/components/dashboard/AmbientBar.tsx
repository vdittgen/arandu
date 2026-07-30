/**
 * Ambient system-state bar — collapsed pipeline + DB health.
 *
 * Collapsed by default: a single low-contrast row.
 * Expanded: database health row. Persists open/closed in localStorage
 * so the choice survives reloads.
 *
 * Also surfaces proactive-loop freshness ("eval 2h ago") so a silently
 * starving evaluation loop is visible instead of masquerading as an
 * empty dashboard. Stale state (never evaluated, or older than 2× the
 * 2h cycle) is flagged in amber.
 *
 * sensitivity_tier: 1
 */

import { useCallback, useEffect, useState } from "react";
import { ChevronDown, ChevronUp, Loader2 } from "lucide-react";
import { listen } from "@tauri-apps/api/event";
import { dedupInvoke } from "../../utils/requestDedup";
import { useAsyncData } from "../../hooks/useAsyncData";
import { useOnboardingFollowupContext } from "../../App";

const STORAGE_KEY = "dashboard:ambient_expanded";

/** Proactive cycle is 2h; older than 2× that (or never) = starving. */
const STALE_AFTER_MS = 4 * 3600 * 1000;

interface DbStats {
  readonly healthy: boolean;
  readonly total_sqlite_rows: number;
  readonly total_kuzu_nodes: number;
  readonly total_chroma_docs: number;
}

interface ProactiveStatus {
  readonly last_evaluated_at: string | null;
  readonly pending_replies: number;
  readonly contact_contexts: number;
  readonly actionable_events: number;
}

function formatAge(iso: string): string {
  const ms = Date.now() - new Date(iso).getTime();
  if (Number.isNaN(ms) || ms < 0) return "just now";
  const minutes = Math.floor(ms / 60000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 48) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

function AmbientBar() {
  const [expanded, setExpanded] = useState(() => {
    try {
      return window.localStorage.getItem(STORAGE_KEY) === "1";
    } catch {
      return false;
    }
  });

  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEY, expanded ? "1" : "0");
    } catch {
      // localStorage may be unavailable in some embeds — non-fatal.
    }
  }, [expanded]);

  const statsFetcher = useCallback(
    () => dedupInvoke<DbStats>("get_database_stats"),
    [],
  );
  const stats = useAsyncData<DbStats>(statsFetcher);
  const followup = useOnboardingFollowupContext();

  const proactiveFetcher = useCallback(
    () => dedupInvoke<ProactiveStatus>("get_proactive_status"),
    [],
  );
  const proactive = useAsyncData<ProactiveStatus>(proactiveFetcher);

  // Refetch freshness immediately after any proactive evaluation —
  // otherwise a just-completed cycle would stay flagged stale for the
  // whole poll interval.
  useEffect(() => {
    const unlisten = listen("arandu:proactive-refreshed", () => {
      void proactive.refetch();
    });
    return () => {
      void unlisten.then((fn) => fn());
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const dbHealthy = stats.data?.healthy ?? true;
  const totalRecords = stats.data
    ? stats.data.total_sqlite_rows +
      stats.data.total_kuzu_nodes +
      stats.data.total_chroma_docs
    : 0;

  const lastEval = proactive.data?.last_evaluated_at ?? null;
  const evalStale =
    lastEval === null ||
    Date.now() - new Date(lastEval).getTime() > STALE_AFTER_MS;
  const evalLabel =
    lastEval === null ? "eval: never" : `eval ${formatAge(lastEval)}`;

  return (
    <div className="rounded-2 border border-hairline bg-surface">
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        aria-expanded={expanded}
        className="flex w-full items-center justify-between px-4 py-2 text-[11px] text-muted transition-colors hover:text-ink"
      >
        <span className="flex items-center gap-3">
          {followup?.running && (
            <>
              <span className="flex items-center gap-1.5 text-indigo">
                <Loader2 className="h-3 w-3 animate-spin" strokeWidth={1.6} />
                Setting up {followup.done}/{followup.total}
                {followup.current ? ` · ${followup.current}` : ""}
              </span>
              <span>·</span>
            </>
          )}
          <span className={dbHealthy ? "text-muted" : "text-danger"}>
            DB {dbHealthy ? "ok" : "issue"}
          </span>
          {totalRecords > 0 && (
            <>
              <span>·</span>
              <span>{totalRecords.toLocaleString()} records</span>
            </>
          )}
          <span>·</span>
          <span
            className={evalStale ? "text-amber" : "text-muted"}
            title={
              lastEval === null
                ? "Proactive evaluation has never completed — replies, contexts and events may be missing."
                : `Last proactive evaluation: ${lastEval}`
            }
          >
            {evalLabel}
          </span>
        </span>
        {expanded ? (
          <ChevronUp className="h-3.5 w-3.5" strokeWidth={1.6} />
        ) : (
          <ChevronDown className="h-3.5 w-3.5" strokeWidth={1.6} />
        )}
      </button>

      {expanded && stats.data && (
        <div className="space-y-3 border-t border-hairline px-4 py-3">
          <div className="text-[11px] text-muted">
            {totalRecords.toLocaleString()} records ·{" "}
            {stats.data.total_sqlite_rows.toLocaleString()} rows,{" "}
            {stats.data.total_kuzu_nodes.toLocaleString()} graph nodes,{" "}
            {stats.data.total_chroma_docs.toLocaleString()} embeddings
          </div>
          {proactive.data && (
            <div className="text-[11px] text-muted">
              Proactive eval:{" "}
              <span className={evalStale ? "text-amber" : undefined}>
                {evalLabel}
              </span>{" "}
              · {proactive.data.pending_replies} pending replies ·{" "}
              {proactive.data.actionable_events} events ·{" "}
              {proactive.data.contact_contexts} contexts
              {evalStale && (
                <>
                  {" "}
                  — the evaluation loop may be starved; it catches up
                  automatically once background work settles.
                </>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default AmbientBar;
