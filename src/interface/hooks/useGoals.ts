/**
 * Hooks for goals: list, mutate, mine from sources.
 *
 * Type definitions mirror `src-tauri/src/commands/types.rs::Goal`
 * exactly — the Tauri serializer/deserializer is the only contract
 * between the two layers. Keep field names and types in lock-step.
 *
 * sensitivity_tier: 2
 */

import { useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import { dedupInvoke } from "../utils/requestDedup";
import { useAsyncData, type AsyncDataResult } from "./useAsyncData";

export type GoalCategory = "personal" | "life" | "work";
export type GoalHorizon = "short" | "medium" | "long";
export type GoalStatus =
  | "active"
  | "paused"
  | "achieved"
  | "abandoned"
  | "archived";
export type GoalSource = "user" | "brain";

export interface Goal {
  readonly id: string;
  readonly title: string;
  readonly description: string;
  readonly category: GoalCategory;
  readonly horizon: GoalHorizon;
  readonly target_date: string | null;
  readonly status: GoalStatus;
  readonly importance: number;
  readonly why: string;
  readonly source: GoalSource;
  readonly source_ref: string | null;
  readonly created_at: string;
  readonly updated_at: string;
  readonly last_confirmed_at: string | null;
  readonly sensitivity_tier: number;
  /**
   * Derived at list-time by `cmd_goals_list`. Higher = more pressing
   * (tasks due today, overdue work, near target date). Use this for
   * dashboard ordering; the Goals page can still group by category.
   */
  readonly urgency_score: number;
  /**
   * Decay verdict for a brain-mined goal whose evidence has gone stale
   * ("fading"/"stale"). Always "fresh" for user goals. Derived at
   * list-time. Use to fade or badge dormant mined goals.
   */
  readonly decay_state: "fresh" | "fading" | "stale";
  /** Days since the goal's evidence anchor. Derived at list-time. */
  readonly days_since_confirmed: number;
}

export interface GoalCreatePayload {
  readonly title: string;
  readonly category: GoalCategory;
  readonly description?: string;
  readonly horizon?: GoalHorizon;
  readonly target_date?: string | null;
  readonly importance?: number;
  readonly why?: string;
}

export interface UseGoalsFilters {
  readonly status?: GoalStatus | null;
  readonly category?: GoalCategory | null;
}

export function useGoals(
  filters: UseGoalsFilters = { status: "active" },
): AsyncDataResult<Goal[]> {
  const { status, category } = filters;
  const fetcher = useCallback(
    () =>
      dedupInvoke<Goal[]>("list_goals", {
        status: status ?? null,
        category: category ?? null,
      }),
    [status, category],
  );
  return useAsyncData<Goal[]>(fetcher);
}

export async function createGoal(payload: GoalCreatePayload): Promise<Goal> {
  return invoke<Goal>("create_goal", { payload });
}

export async function updateGoal(
  id: string,
  patch: Record<string, unknown>,
): Promise<Goal | null> {
  return invoke<Goal | null>("update_goal", { id, patch });
}

export async function mineGoals(): Promise<Goal[]> {
  return invoke<Goal[]>("mine_goals");
}
