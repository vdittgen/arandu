/**
 * LifeBoard — the dashboard's unified Work / Personal / Health canvas.
 *
 * Merges what used to be two separate widgets (the Goals widget and
 * the Your Life snapshot) into one card so the user sees, per domain:
 *   - the goals that matter today, ordered by `urgency_score`;
 *   - today's concrete goal-anchored moves (tasks + habits) — the
 *     same data that used to be hidden behind a Goals-page drill-down;
 *   - the domain's "today's shape" — calendar events for work/personal
 *     or latest metrics for health (existing `DomainTimeline`).
 *
 * Top goal in each column carries a 2px accent border-left to signal
 * urgency without introducing a new color token. Goal chips and
 * action rows route into their existing detail surfaces.
 *
 * sensitivity_tier: 3 (aggregates Tier 3 goals + actions + events)
 */

import { useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Briefcase,
  Flame,
  Heart,
  Home as HomeIcon,
  Layers,
  Target,
} from "lucide-react";
import { Skeleton } from "../../LoadingState";
import DomainTimeline from "../domains/DomainTimeline";
import {
  useLifeBoard,
  type LifeBoardDomain,
  type LifeBoardDomainName,
} from "../../../hooks/useLifeBoard";
import type { Goal } from "../../../hooks/useGoals";

interface DomainSpec {
  readonly id: LifeBoardDomainName;
  readonly label: string;
  readonly icon: typeof Briefcase;
  readonly tint: string;
  readonly inkColor: string;
  readonly inkVar: string;
}

const DOMAINS: ReadonlyArray<DomainSpec> = [
  { id: "work", label: "Work", icon: Briefcase, tint: "var(--work-tint)", inkColor: "text-work-ink", inkVar: "var(--work-ink)" },
  { id: "personal", label: "Personal", icon: HomeIcon, tint: "var(--personal-tint)", inkColor: "text-personal-ink", inkVar: "var(--personal-ink)" },
  { id: "health", label: "Health", icon: Heart, tint: "var(--health-tint)", inkColor: "text-health-ink", inkVar: "var(--health-ink)" },
];

const MAX_GOALS_PER_COLUMN = 3;

function HorizonBadge({ horizon }: { readonly horizon: Goal["horizon"] }) {
  const labels: Record<Goal["horizon"], string> = {
    short: "≤3mo",
    medium: "3-12mo",
    long: "12mo+",
  };
  return (
    <span className="rounded-1 bg-bg-2 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-muted">
      {labels[horizon]}
    </span>
  );
}

function GoalChip({
  goal,
  emphasis,
  todayProgress,
}: {
  readonly goal: Goal;
  readonly emphasis: boolean;
  readonly todayProgress?: { total: number; done: number };
}) {
  const navigate = useNavigate();
  return (
    <button
      type="button"
      onClick={() =>
        navigate("/goals", { state: { selectedId: goal.id } })
      }
      className={`flex w-full flex-col gap-1 rounded-2 border bg-bg/60 px-2 py-1.5 text-left transition-colors hover:border-indigo ${
        emphasis
          ? "border-hairline border-l-2 border-l-indigo"
          : "border-hairline"
      }`}
    >
      <div className="flex w-full items-center justify-between gap-2">
        <p className="min-w-0 flex-1 truncate text-[11px] font-medium text-ink">
          {goal.title}
        </p>
        <div className="flex shrink-0 items-center gap-1.5">
          {goal.urgency_score > 0 && (
            <span
              className="flex items-center gap-0.5 text-[10px] text-amber"
              title="Urgency score"
            >
              <Flame className="h-2.5 w-2.5" strokeWidth={1.6} />
              {goal.urgency_score}
            </span>
          )}
          <HorizonBadge horizon={goal.horizon} />
        </div>
      </div>
      {todayProgress && todayProgress.total > 0 && (
        <div className="flex w-full items-center gap-2">
          <div className="h-1 flex-1 rounded-pill bg-hairline">
            <div
              className="h-1 rounded-pill bg-indigo transition-all"
              style={{
                width: `${Math.round(
                  (todayProgress.done / todayProgress.total) * 100,
                )}%`,
              }}
            />
          </div>
          <span className="text-[10px] text-muted">
            {todayProgress.done}/{todayProgress.total} today
          </span>
        </div>
      )}
    </button>
  );
}

interface ColumnProps {
  readonly spec: DomainSpec;
  readonly data: LifeBoardDomain | null;
  readonly onRefetch: () => void;
}

function Column({ spec, data }: ColumnProps) {
  const navigate = useNavigate();
  const Icon = spec.icon;
  const goals = data?.goals ?? [];
  const actions = data?.today_actions ?? [];
  const progressByGoal = data?.today_progress ?? {};
  const items = data?.items ?? [];

  const totalToday = actions.length;
  const isEmpty = goals.length === 0 && items.length === 0;

  return (
    <section
      className="relative min-w-0 flex-1 overflow-hidden rounded-4 border border-hairline bg-surface p-5 shadow-1"
    >
      <div
        className="pointer-events-none absolute inset-0 h-20 opacity-70"
        style={{ background: `linear-gradient(180deg, ${spec.tint}, transparent)` }}
      />
      <header className="relative mb-3 flex items-center gap-1.5">
        <Icon className="h-4 w-4" strokeWidth={1.6} style={{ color: spec.inkVar }} />
        <span className="text-sm font-semibold" style={{ color: spec.inkVar }}>
          {spec.label}
        </span>
        {totalToday > 0 && (
          <span className="rounded-pill bg-indigo-soft px-1.5 py-0.5 text-[10px] font-medium text-indigo">
            {totalToday} today
          </span>
        )}
      </header>

      {isEmpty ? (
        <p className="py-2 text-sm italic text-muted">
          Nothing on the radar today.
        </p>
      ) : (
        <>
          {/* Goals strip — urgency-sorted, top one accented. */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-1 text-[10px] uppercase tracking-[0.06em] text-muted">
                <Target className="h-2.5 w-2.5" strokeWidth={1.6} />
                Goals
              </span>
              <button
                type="button"
                onClick={() => navigate("/goals")}
                className="text-[10px] text-muted hover:text-ink"
              >
                Open →
              </button>
            </div>
            {goals.length === 0 ? (
              <p className="text-[11px] text-faint">
                No active goals here.
              </p>
            ) : (
              <ul className="space-y-1">
                {goals.slice(0, MAX_GOALS_PER_COLUMN).map((g, idx) => (
                  <li key={g.id}>
                    <GoalChip
                      goal={g}
                      emphasis={idx === 0}
                      todayProgress={progressByGoal[g.id]}
                    />
                  </li>
                ))}
              </ul>
            )}
          </div>

          {/* Today's shape — events / metrics from the domain mart. */}
          <div className="mt-3 space-y-1.5">
            <span className="text-[10px] uppercase tracking-[0.06em] text-muted">
              Today's shape
            </span>
            {items.length === 0 ? (
              <p className="py-1 text-[11px] text-faint">
                {spec.id === "health"
                  ? "No health connector yet — steps, sleep, and workouts will appear here once one ships."
                  : "No events in this domain today."}
              </p>
            ) : (
              <DomainTimeline items={items} />
            )}
          </div>
        </>
      )}
    </section>
  );
}

function LifeBoard() {
  const board = useLifeBoard();
  const { refetch } = board;
  const [refreshKey, setRefreshKey] = useState(0);

  const triggerRefetch = useCallback(() => {
    setRefreshKey((k) => k + 1);
    refetch();
  }, [refetch]);

  // The whole board reflects pipeline + proactive state — refetch when
  // either signals a refresh, same pattern as the previous widgets.
  useEffect(() => {
    const handler = () => refetch();
    window.addEventListener("arandu:pipeline-refreshed", handler);
    window.addEventListener("arandu:proactive-refreshed", handler);
    return () => {
      window.removeEventListener("arandu:pipeline-refreshed", handler);
      window.removeEventListener("arandu:proactive-refreshed", handler);
    };
  }, [refetch]);

  const data = board.data;

  if (board.isLoading && !data) {
    return (
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <Layers className="h-5 w-5 text-indigo" strokeWidth={1.6} />
          <h3 className="text-[22px] font-semibold text-ink">Your Life</h3>
        </div>
        <div className="grid grid-cols-1 gap-3.5 md:grid-cols-3">
          {DOMAINS.map((spec) => (
            <div key={spec.id} className="space-y-2 rounded-4 border border-hairline bg-surface p-5">
              <Skeleton className="h-4 w-20" />
              <Skeleton className="h-12 w-full" />
              <Skeleton className="h-12 w-full" />
              <Skeleton className="h-12 w-full" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  const byDomain = new Map<LifeBoardDomainName, LifeBoardDomain>();
  for (const d of data?.domains ?? []) {
    byDomain.set(d.domain, d);
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <Layers className="h-5 w-5 text-indigo" strokeWidth={1.6} />
        <h3 className="text-[22px] font-semibold text-ink">Your Life</h3>
      </div>
      <div
        key={refreshKey}
        className="grid grid-cols-1 gap-3.5 md:grid-cols-3"
      >
        {DOMAINS.map((spec) => (
          <Column
            key={spec.id}
            spec={spec}
            data={byDomain.get(spec.id) ?? null}
            onRefetch={triggerRefetch}
          />
        ))}
      </div>
    </div>
  );
}

export default LifeBoard;
