/**
 * Goals page.
 *
 * Three columns (Personal / Life / Work) listing the user's active
 * goals. Selecting a goal opens a detail panel with the why, the
 * topics rolled up under it, the habits anchored to it, and status
 * controls. Brain-mined goals carry an accent border so they stand
 * out from user-entered ones; a brain goal whose evidence has gone
 * stale (`decay_state`) fades and shows a "Dormant" badge, and once
 * auto-archived it moves to the collapsible Archived review section
 * where it can be restored.
 *
 * sensitivity_tier: 3
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import {
  Archive,
  Flame,
  Plus,
  RefreshCw,
  Sparkles,
  Target,
} from "lucide-react";
import Card from "../components/dashboard/Card";
import { Skeleton } from "../components/LoadingState";
import {
  createGoal,
  mineGoals,
  updateGoal,
  useGoals,
  type Goal,
  type GoalCategory,
  type GoalHorizon,
} from "../hooks/useGoals";
import { useHabits, toggleHabit } from "../hooks/useHabits";
import { toggleTaskDone } from "../hooks/useTasks";
import { useGoalProgress } from "../hooks/useGoalProgress";
import { formatRelativeTime } from "../utils/timeFormat";

const CATEGORIES: ReadonlyArray<{
  readonly key: GoalCategory;
  readonly label: string;
}> = [
  { key: "personal", label: "Personal" },
  { key: "life", label: "Life" },
  { key: "work", label: "Work" },
];

interface GoalFormState {
  title: string;
  category: GoalCategory;
  horizon: GoalHorizon;
  target_date: string;
  importance: number;
  why: string;
}

const DEFAULT_FORM: GoalFormState = {
  title: "",
  category: "personal",
  horizon: "medium",
  target_date: "",
  importance: 5,
  why: "",
};

function ImportanceDots({ value }: { readonly value: number }) {
  const dots = Math.max(1, Math.min(10, Math.round(value)));
  return (
    <div className="flex gap-0.5">
      {Array.from({ length: 10 }, (_, i) => (
        <span
          key={i}
          className={`h-1.5 w-1.5 rounded-full ${
            i < dots ? "bg-indigo" : "bg-hairline"
          }`}
        />
      ))}
    </div>
  );
}

function HorizonBadge({ horizon }: { readonly horizon: GoalHorizon }) {
  const labels: Record<GoalHorizon, string> = {
    short: "≤ 3 mo",
    medium: "3-12 mo",
    long: "12+ mo",
  };
  return (
    <span className="shrink-0 whitespace-nowrap rounded bg-bg-2 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-muted">
      {labels[horizon]}
    </span>
  );
}

interface GoalCardProps {
  readonly goal: Goal;
  readonly selected: boolean;
  readonly onSelect: () => void;
}

function GoalCard({ goal, selected, onSelect }: GoalCardProps) {
  const isBrain = goal.source === "brain";
  // A brain goal the miner stopped re-confirming fades toward archival.
  const dormant = goal.decay_state !== "fresh";
  return (
    <button
      type="button"
      onClick={onSelect}
      className={`w-full rounded-2 border p-3 text-left transition-colors ${
        selected
          ? "border-indigo bg-indigo/5"
          : "border-hairline bg-surface hover:bg-bg-2"
      } ${isBrain ? "border-l-4 border-l-indigo" : ""} ${
        dormant ? "opacity-60" : ""
      }`}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="truncate text-sm font-medium text-ink">
          {goal.title}
        </span>
        <HorizonBadge horizon={goal.horizon} />
      </div>
      {goal.why && (
        <p className="mt-1 italic text-xs text-muted line-clamp-2">
          {goal.why}
        </p>
      )}
      <div className="mt-2 flex items-center justify-between">
        <ImportanceDots value={goal.importance} />
        <div className="flex items-center gap-2">
          {dormant && (
            <span
              className="rounded-pill bg-bg-2 px-1.5 py-0.5 text-[10px] text-muted"
              title={`No fresh evidence for ${goal.days_since_confirmed} days`}
            >
              Dormant
            </span>
          )}
          {isBrain && (
            <span className="flex items-center gap-1 text-[10px] text-indigo">
              <Sparkles className="h-3 w-3" /> Brain
            </span>
          )}
        </div>
      </div>
    </button>
  );
}

interface ColumnProps {
  readonly category: GoalCategory;
  readonly label: string;
  readonly goals: ReadonlyArray<Goal>;
  readonly selectedId: string | null;
  readonly onSelect: (g: Goal) => void;
}

function Column({ category, label, goals, selectedId, onSelect }: ColumnProps) {
  const filtered = goals.filter((g) => g.category === category);
  return (
    <Card title={label} icon={<Target className="h-4 w-4 text-indigo" />}>
      {filtered.length === 0 ? (
        <p className="py-6 text-center text-xs text-muted">
          No active goals in this category.
        </p>
      ) : (
        <div className="space-y-2">
          {filtered.map((g) => (
            <GoalCard
              key={g.id}
              goal={g}
              selected={g.id === selectedId}
              onSelect={() => onSelect(g)}
            />
          ))}
        </div>
      )}
    </Card>
  );
}

interface DetailPanelProps {
  readonly goal: Goal;
  readonly onRefetch: () => void;
}

function ProgressStat({
  label,
  value,
  emphasis = false,
}: {
  readonly label: string;
  readonly value: string;
  readonly emphasis?: boolean;
}) {
  return (
    <div className="rounded border border-hairline bg-bg-2 p-2">
      <p className="text-[10px] uppercase tracking-wide text-muted">
        {label}
      </p>
      <p
        className={`text-base font-semibold ${
          emphasis ? "text-indigo" : "text-ink"
        }`}
      >
        {value}
      </p>
    </div>
  );
}

function DetailPanel({ goal, onRefetch }: DetailPanelProps) {
  const navigate = useNavigate();
  const habits = useHabits({ goal_id: goal.id });
  const progress = useGoalProgress(goal.id);

  const onSetStatus = useCallback(
    async (status: Goal["status"]) => {
      await updateGoal(goal.id, { status });
      onRefetch();
    },
    [goal.id, onRefetch],
  );

  const onToggleTask = useCallback(
    async (taskId: string) => {
      await toggleTaskDone(taskId);
      progress.refetch();
    },
    [progress],
  );

  const onToggleHabit = useCallback(
    async (habitId: string) => {
      await toggleHabit(habitId);
      progress.refetch();
      habits.refetch();
    },
    [habits, progress],
  );

  const data = progress.data;

  return (
    <Card title={goal.title}>
      <div className="space-y-4 text-sm">
        {goal.description && (
          <p className="text-ink">{goal.description}</p>
        )}
        {goal.why && (
          <p className="border-l-2 border-indigo pl-3 italic text-muted">
            {goal.why}
          </p>
        )}
        <div className="flex items-center gap-3 text-xs text-muted">
          <HorizonBadge horizon={goal.horizon} />
          {goal.target_date && <span>Target {goal.target_date}</span>}
          <span>Source: {goal.source}</span>
        </div>

        <section>
          <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted">
            Progress
          </h4>
          {progress.isLoading && !data ? (
            <Skeleton className="h-14 w-full" />
          ) : data ? (
            <div className="grid grid-cols-2 gap-2 md:grid-cols-5">
              <ProgressStat
                label="Open"
                value={String(data.tasks_open)}
              />
              <ProgressStat
                label="Today"
                value={String(data.tasks_today.length)}
                emphasis={data.tasks_today.length > 0}
              />
              <ProgressStat
                label="Done 7d"
                value={String(data.tasks_done_7d)}
              />
              <ProgressStat
                label="Habits"
                value={String(data.habits_today.length)}
              />
              <ProgressStat
                label="Streak"
                value={`${data.habit_streak_days}d`}
                emphasis={data.habit_streak_days > 0}
              />
            </div>
          ) : null}
          {data?.last_evidence_at && (
            <p className="mt-2 text-[11px] text-muted">
              Last evidence {formatRelativeTime(data.last_evidence_at)}
            </p>
          )}
        </section>

        {data && data.rolled_up_topics.length > 0 && (
          <section>
            <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted">
              Topics rolled up here
            </h4>
            <ul className="space-y-1">
              {data.rolled_up_topics.map((t) => (
                <li key={t.topic_id}>
                  <button
                    type="button"
                    onClick={() =>
                      navigate(`/inbox?topic=${encodeURIComponent(t.title)}`)
                    }
                    className="flex w-full items-center justify-between gap-2 rounded border border-hairline bg-bg-2 px-2 py-1.5 text-left text-xs hover:border-indigo"
                  >
                    <span className="truncate text-ink">
                      {t.title}
                      {t.contact_name && (
                        <span className="text-muted">
                          {" "}— {t.contact_name}
                        </span>
                      )}
                    </span>
                    {t.importance > 0 && (
                      <span className="flex items-center gap-1 text-[10px] text-muted">
                        <Flame className="h-2.5 w-2.5" />
                        {t.importance}
                      </span>
                    )}
                  </button>
                </li>
              ))}
            </ul>
          </section>
        )}

        {data &&
          (data.tasks_today.length > 0 || data.habits_today.length > 0) && (
          <section>
            <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted">
              Today's moves
            </h4>
            <ul className="space-y-1.5">
              {data.tasks_today.map((t) => (
                <li
                  key={`task-${t.id}`}
                  className="flex items-start gap-2 rounded border border-hairline bg-bg-2 p-2"
                >
                  <button
                    type="button"
                    onClick={() => onToggleTask(t.id)}
                    className="mt-0.5 h-3.5 w-3.5 shrink-0 rounded border border-hairline bg-bg-2"
                    title="Toggle done"
                  />
                  <span className="flex-1 text-xs text-ink">
                    {t.title}
                  </span>
                  {t.due_at && (
                    <span className="text-[10px] text-muted">
                      {new Date(t.due_at).toLocaleTimeString([], {
                        hour: "2-digit",
                        minute: "2-digit",
                      })}
                    </span>
                  )}
                </li>
              ))}
              {data.habits_today.map((h) => (
                <li
                  key={`habit-${h.id}`}
                  className="flex items-start gap-2 rounded border border-hairline bg-bg-2 p-2"
                >
                  <button
                    type="button"
                    onClick={() => onToggleHabit(h.id)}
                    className="mt-0.5 h-3.5 w-3.5 shrink-0 rounded border border-hairline bg-bg-2"
                    title="Toggle habit"
                  />
                  <span className="flex-1 text-xs text-ink">
                    {h.title}
                  </span>
                  <span className="text-[10px] text-muted">
                    {h.preferred_window}
                  </span>
                </li>
              ))}
            </ul>
          </section>
        )}

        <section>
          <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted">
            Habits anchored to this goal
          </h4>
          {habits.isLoading ? (
            <Skeleton className="h-8 w-full" />
          ) : (habits.data ?? []).length === 0 ? (
            <p className="text-xs text-muted">
              No habits yet. Run "Mine goals" then "Regenerate habits"
              on the Goals page to surface atomic habits anchored here.
            </p>
          ) : (
            <ul className="space-y-1">
              {(habits.data ?? []).map((h) => (
                <li key={h.id} className="text-xs text-ink">
                  • {h.title}{" "}
                  <span className="text-muted">
                    — {h.preferred_window}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </section>

        <div className="flex flex-wrap gap-2">
          {(["active", "paused", "achieved", "abandoned"] as const).map((s) => (
            <button
              key={s}
              type="button"
              disabled={goal.status === s}
              onClick={() => onSetStatus(s)}
              className={`rounded border px-2 py-1 text-xs ${
                goal.status === s
                  ? "border-indigo bg-indigo-soft text-indigo"
                  : "border-hairline text-muted hover:text-ink"
              }`}
            >
              {s}
            </button>
          ))}
        </div>
      </div>
    </Card>
  );
}

function NewGoalModal({
  open,
  onClose,
  onCreated,
}: {
  readonly open: boolean;
  readonly onClose: () => void;
  readonly onCreated: (g: Goal) => void;
}) {
  const [form, setForm] = useState<GoalFormState>(DEFAULT_FORM);
  const [submitting, setSubmitting] = useState(false);

  if (!open) return null;

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!form.title.trim()) return;
    setSubmitting(true);
    try {
      const goal = await createGoal({
        title: form.title.trim(),
        category: form.category,
        horizon: form.horizon,
        target_date: form.target_date || null,
        importance: form.importance,
        why: form.why,
      });
      onCreated(goal);
      setForm(DEFAULT_FORM);
      onClose();
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <form
        onSubmit={onSubmit}
        className="w-full max-w-md space-y-3 rounded-4 border border-hairline bg-surface p-5"
      >
        <h3 className="text-sm font-semibold text-ink">New goal</h3>
        <input
          autoFocus
          placeholder="Title"
          value={form.title}
          onChange={(e) => setForm({ ...form, title: e.target.value })}
          className="w-full rounded border border-hairline bg-bg-2 px-3 py-2 text-sm text-ink"
        />
        <textarea
          placeholder="Why does this goal matter?"
          value={form.why}
          onChange={(e) => setForm({ ...form, why: e.target.value })}
          className="w-full rounded border border-hairline bg-bg-2 px-3 py-2 text-sm text-ink"
          rows={2}
        />
        <div className="flex gap-2">
          <select
            value={form.category}
            onChange={(e) =>
              setForm({ ...form, category: e.target.value as GoalCategory })
            }
            className="flex-1 rounded border border-hairline bg-bg-2 px-2 py-1.5 text-xs text-ink"
          >
            {CATEGORIES.map((c) => (
              <option key={c.key} value={c.key}>
                {c.label}
              </option>
            ))}
          </select>
          <select
            value={form.horizon}
            onChange={(e) =>
              setForm({ ...form, horizon: e.target.value as GoalHorizon })
            }
            className="flex-1 rounded border border-hairline bg-bg-2 px-2 py-1.5 text-xs text-ink"
          >
            <option value="short">Short (≤ 3 mo)</option>
            <option value="medium">Medium (3-12 mo)</option>
            <option value="long">Long (12+ mo)</option>
          </select>
        </div>
        <div className="flex gap-2">
          <input
            type="date"
            value={form.target_date}
            onChange={(e) => setForm({ ...form, target_date: e.target.value })}
            className="flex-1 rounded border border-hairline bg-bg-2 px-2 py-1.5 text-xs text-ink"
          />
          <input
            type="number"
            min={1}
            max={10}
            value={form.importance}
            onChange={(e) =>
              setForm({ ...form, importance: Number(e.target.value) })
            }
            className="w-20 rounded border border-hairline bg-bg-2 px-2 py-1.5 text-xs text-ink"
          />
        </div>
        <div className="flex justify-end gap-2">
          <button
            type="button"
            onClick={onClose}
            className="rounded border border-hairline px-3 py-1.5 text-xs text-muted hover:text-ink"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={submitting || !form.title.trim()}
            className="rounded bg-indigo px-3 py-1.5 text-xs text-white disabled:opacity-60"
          >
            {submitting ? "Saving…" : "Create"}
          </button>
        </div>
      </form>
    </div>
  );
}

function ArchivedReview({
  onRestored,
}: {
  readonly onRestored: () => void;
}) {
  const archived = useGoals({ status: "archived" });
  const [open, setOpen] = useState(false);
  const [restoring, setRestoring] = useState<string | null>(null);

  const rows = archived.data ?? [];
  if (rows.length === 0) return null;

  const onRestore = async (id: string) => {
    setRestoring(id);
    try {
      await updateGoal(id, { status: "active" });
      archived.refetch();
      onRestored();
    } finally {
      setRestoring(null);
    }
  };

  return (
    <div className="rounded-2 border border-hairline bg-surface/60">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between px-4 py-2 text-xs text-muted hover:text-ink"
      >
        <span className="flex items-center gap-2">
          <Archive className="h-3.5 w-3.5" />
          Archived goals ({rows.length})
        </span>
        <span className="text-[10px]">{open ? "Hide" : "Review"}</span>
      </button>
      {open && (
        <ul className="space-y-1 border-t border-hairline px-4 py-3">
          {rows.map((g) => (
            <li
              key={g.id}
              className="flex items-center justify-between gap-2 text-xs"
            >
              <span className="min-w-0 truncate text-ink-2">
                {g.title}
                <span className="ml-2 text-muted">
                  dormant {g.days_since_confirmed}d
                </span>
              </span>
              <button
                type="button"
                onClick={() => onRestore(g.id)}
                disabled={restoring === g.id}
                className="shrink-0 rounded border border-hairline px-2 py-1 text-[11px] text-indigo hover:bg-indigo/5 disabled:opacity-60"
              >
                {restoring === g.id ? "Restoring…" : "Restore"}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function Goals() {
  const location = useLocation();
  const initialSelected =
    (location.state as { selectedId?: string } | null)?.selectedId ?? null;
  const goals = useGoals({ status: "active" });
  const [selectedId, setSelectedId] = useState<string | null>(initialSelected);
  const [showNew, setShowNew] = useState(false);
  const [mining, setMining] = useState(false);

  // If we landed here from a dashboard click-through, sync the
  // selection once the goals list loads — the goal may not have
  // existed when the location.state was captured.
  useEffect(() => {
    if (initialSelected && goals.data && !selectedId) {
      setSelectedId(initialSelected);
    }
  }, [initialSelected, goals.data, selectedId]);

  const selected = useMemo(
    () => (goals.data ?? []).find((g) => g.id === selectedId) ?? null,
    [goals.data, selectedId],
  );

  const onMine = async () => {
    setMining(true);
    try {
      await mineGoals();
      goals.refetch();
    } finally {
      setMining(false);
    }
  };

  return (
    <div className="flex-1 space-y-4 overflow-y-auto p-5">
      <header className="flex items-center justify-between">
        <div>
          <h2
            className="text-[44px] font-bold leading-none"
            style={{
              background: "linear-gradient(135deg, var(--ink), var(--ink-2))",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              backgroundClip: "text",
            }}
          >
            Goals
          </h2>
          <p className="mt-1 text-sm text-muted">
            Goals carry a horizon and a why. Tasks and habits roll up here.
          </p>
        </div>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={onMine}
            disabled={mining}
            className="flex items-center gap-1 rounded border border-hairline bg-surface px-3 py-1.5 text-xs text-muted hover:text-ink disabled:opacity-60"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${mining ? "animate-spin" : ""}`} />
            Mine goals from sources
          </button>
          <button
            type="button"
            onClick={() => setShowNew(true)}
            className="flex items-center gap-1 rounded bg-indigo px-3 py-1.5 text-xs text-white"
          >
            <Plus className="h-3.5 w-3.5" strokeWidth={1.6} /> New goal
          </button>
        </div>
      </header>

      <div>
        {goals.isLoading && goals.data === null ? (
          <Skeleton className="h-40 w-full" />
        ) : selected ? (
          <DetailPanel goal={selected} onRefetch={goals.refetch} />
        ) : (
          <Card title="Pick a goal">
            <p className="py-6 text-center text-xs text-muted">
              Select a goal to see its why, target date, and the habits
              anchored to it.
            </p>
          </Card>
        )}
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        {CATEGORIES.map((c) => (
          <Column
            key={c.key}
            category={c.key}
            label={c.label}
            goals={goals.data ?? []}
            selectedId={selectedId}
            onSelect={(g) => setSelectedId(g.id)}
          />
        ))}
      </div>

      <ArchivedReview onRestored={goals.refetch} />

      <NewGoalModal
        open={showNew}
        onClose={() => setShowNew(false)}
        onCreated={(g) => {
          goals.refetch();
          setSelectedId(g.id);
        }}
      />
    </div>
  );
}

export default Goals;
