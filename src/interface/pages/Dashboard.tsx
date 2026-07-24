/**
 * Mission Control — the Arandu home page.
 *
 * Composition-only file. Dashboard owns the data hooks; child widgets
 * are presentation-only and receive `data` + `onRefetch` as props.
 *
 * Layout (top → bottom):
 *   - Header (eyebrow kicker + gradient greeting)
 *   - CommandBar (Ask Arandu, with Chat's full tool reach)
 *   - Bento grid (DailyBrief, AgentStream summary, stat cards)
 *   - TodayBoard (Now / Up Next / Today's loops — one canvas)
 *   - LifeBoard (Work / Personal / Health — goals, today's moves, events)
 *   - AgentStream (Running + Recently completed; "Awaiting review" retired)
 *   - AmbientBar (collapsed pipeline / spend / DB health)
 *
 * The standalone Goals widget and Your-Life snapshot fused into one
 * LifeBoard so the day's goal-anchored tasks/habits sit next to the
 * domain they belong to instead of living one drill-down away on the
 * Goals page. Pending replies + topic threads live on /inbox.
 *
 * sensitivity_tier: 3 (aggregates Tier 3 narrative + Tier 2 chips)
 */

import { useCallback, useEffect } from "react";
import { Link } from "react-router-dom";
import { Sun, Sunset, Moon } from "lucide-react";
import CommandBar from "../components/dashboard/CommandBar";
import DailyBrief from "../components/dashboard/DailyBrief";
import TodayBoard from "../components/dashboard/today/TodayBoard";
import MeetingPrepCard from "../components/dashboard/MeetingPrepCard";
import AgentStream from "../components/dashboard/agents/AgentStream";
import LifeBoard from "../components/dashboard/life/LifeBoard";
import AmbientBar from "../components/dashboard/AmbientBar";
import { useAgentStream } from "../hooks/useAgentStream";
import { useSuggestedActions } from "../hooks/useSuggestedActions";
import { useInbox, type InboxReply } from "../hooks/useInbox";
import { useTasks } from "../hooks/useTasks";
import { useGoals, type Goal } from "../hooks/useGoals";
import { formatRelativeTime } from "../utils/timeFormat";

function getGreeting(): { text: string; Icon: typeof Sun } {
  const hour = new Date().getHours();
  if (hour < 12) return { text: "Good morning", Icon: Sun };
  if (hour < 18) return { text: "Good afternoon", Icon: Sunset };
  return { text: "Good evening", Icon: Moon };
}

function formatEyebrow(): string {
  const now = new Date();
  const weekday = now
    .toLocaleDateString(undefined, { weekday: "long" })
    .toUpperCase();
  const day = now.getDate();
  const month = now
    .toLocaleDateString(undefined, { month: "short" })
    .toUpperCase();
  const hours = now.getHours().toString().padStart(2, "0");
  const minutes = now.getMinutes().toString().padStart(2, "0");
  return `${weekday} · ${day} ${month} · ${hours}:${minutes}`;
}

function DashboardHeader() {
  const { text } = getGreeting();
  return (
    <div>
      <p className="text-[11px] font-medium uppercase tracking-[0.08em] text-muted">
        {formatEyebrow()}
      </p>
      <h1 className="mt-1 text-3xl font-bold text-ink">
        {text},{" "}
        <span
          style={{
            background:
              "linear-gradient(135deg, var(--indigo) 0%, oklch(0.62 0.18 250) 100%)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
          }}
        >
          Vinicius
        </span>
      </h1>
    </div>
  );
}

function InboxUrgentCard({ replies }: { readonly replies: ReadonlyArray<InboxReply> }) {
  const urgent = replies.filter((r) => r.importance >= 8);
  const count = urgent.length;
  const top = urgent[0];
  const hasUrgent = count > 0;

  return (
    <Link
      to="/inbox"
      className="block h-full rounded-4 border border-hairline p-5 shadow-1 transition-all hover:shadow-2 hover:-translate-y-px"
      style={hasUrgent
        ? { background: "linear-gradient(135deg, var(--danger-soft), var(--surface) 70%)" }
        : { background: "var(--surface)" }
      }
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          {hasUrgent && (
            <span className="h-1.5 w-1.5 rounded-full bg-danger" style={{ boxShadow: "0 0 0 3px var(--danger-soft)" }} />
          )}
          <p className="text-[11px] font-medium uppercase tracking-[0.06em] text-muted">
            Inbox · urgent
          </p>
        </div>
        <span className="text-[12.5px] font-medium text-indigo-2">Open →</span>
      </div>
      <p className={`mt-1 text-[38px] font-semibold leading-none ${hasUrgent ? "text-danger" : "text-ink"}`} style={{ letterSpacing: "-0.03em" }}>
        {count}
      </p>
      {top ? (
        <p className="mt-1 text-[13px]" style={{ color: "oklch(0.40 0.14 25)", fontWeight: 500 }}>
          {top.contact_name} · {formatRelativeTime(top.message_at)}
        </p>
      ) : (
        <p className="mt-1 text-[13px] text-muted">All clear</p>
      )}
    </Link>
  );
}

function TasksTodayCard({ count }: { readonly count: number }) {
  return (
    <Link to="/tasks" className="block h-full rounded-4 border border-hairline bg-surface p-5 shadow-1 transition-all hover:shadow-2 hover:-translate-y-px">
      <div className="flex items-center justify-between">
        <p className="text-[11px] font-medium uppercase tracking-[0.06em] text-muted">
          Tasks today
        </p>
        <span className="text-[12.5px] font-medium text-indigo-2">Open →</span>
      </div>
      <p className="mt-1 text-[38px] font-semibold leading-none text-ink" style={{ letterSpacing: "-0.03em" }}>{count}</p>
      <svg className="mt-1 w-full" height="36" viewBox="0 0 100 36" preserveAspectRatio="none">
        <path d="M0,28 L14,24 L28,26 L42,16 L56,20 L70,12 L84,16 L100,8 L100,36 L0,36 Z" fill="var(--indigo-tint)" />
        <path d="M0,28 L14,24 L28,26 L42,16 L56,20 L70,12 L84,16 L100,8" fill="none" stroke="var(--indigo)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
      <div className="mt-1 text-[13px] text-muted">↑ vs week (avg {Math.max(0, count - 1)})</div>
    </Link>
  );
}

function GoalsActiveCard({ goals }: { readonly goals: ReadonlyArray<Goal> }) {
  const active = goals.filter((g) => g.status === "active");
  const onTrack = active.filter((g) => g.urgency_score < 5).length;
  const needAttention = active.filter((g) => g.urgency_score >= 5 && g.urgency_score < 8).length;
  const dormant = active.filter((g) => g.urgency_score >= 8).length;

  return (
    <Link to="/goals" className="block h-full rounded-4 border border-hairline bg-surface p-5 shadow-1 transition-all hover:shadow-2 hover:-translate-y-px">
      <div className="flex items-center justify-between">
        <p className="text-[11px] font-medium uppercase tracking-[0.06em] text-muted">
          Goals · active
        </p>
        <span className="text-[12.5px] font-medium text-indigo-2">Open →</span>
      </div>
      <p className="mt-1 text-[38px] font-semibold leading-none text-ink" style={{ letterSpacing: "-0.03em" }}>{active.length}</p>
      <div className="mt-1 flex flex-col gap-0.5 text-[12px] text-muted">
        <div><span className="text-success font-semibold">●</span> {onTrack} on track</div>
        <div><span className="text-amber font-semibold">●</span> {needAttention} need attention</div>
        <div><span className="text-faint font-semibold">●</span> {dormant} dormant</div>
      </div>
    </Link>
  );
}

function Dashboard() {
  const agentStream = useAgentStream();
  const suggested = useSuggestedActions();
  const inbox = useInbox({ domain: null, topic: null });
  const tasks = useTasks();
  const goals = useGoals();

  const replies = inbox.data?.replies ?? [];
  const taskCount = (tasks.data ?? []).filter((t) => t.status !== "done").length;
  const allGoals = goals.data ?? [];

  const refetchLight = useCallback(() => {
    agentStream.refetch();
    suggested.refetch();
    inbox.refetch();
  }, [agentStream, suggested, inbox]);

  useEffect(() => {
    const handler = () => refetchLight();
    window.addEventListener("arandu:pipeline-refreshed", handler);
    window.addEventListener("arandu:proactive-refreshed", handler);
    return () => {
      window.removeEventListener("arandu:pipeline-refreshed", handler);
      window.removeEventListener("arandu:proactive-refreshed", handler);
    };
  }, [refetchLight]);

  return (
    <div className="flex-1 space-y-5 overflow-y-auto">
      <DashboardHeader />
      <CommandBar
        chips={suggested.data?.chips ?? []}
        loading={suggested.isLoading}
      />

      {/* Bento grid */}
      <div className="grid grid-cols-12 gap-3.5">
        <div className="col-span-7 min-h-0">
          <DailyBrief />
        </div>
        <div className="col-span-5 min-h-0">
          <AgentStream
            stream={agentStream.data}
            isLoading={agentStream.isLoading}
          />
        </div>
        <div className="col-span-4">
          <InboxUrgentCard replies={replies} />
        </div>
        <div className="col-span-4">
          <TasksTodayCard count={taskCount} />
        </div>
        <div className="col-span-4">
          <GoalsActiveCard goals={allGoals} />
        </div>
      </div>

      <TodayBoard />
      <MeetingPrepCard />
      <LifeBoard />
      <AmbientBar />
    </div>
  );
}

export default Dashboard;
