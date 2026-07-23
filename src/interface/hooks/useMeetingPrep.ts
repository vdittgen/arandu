/**
 * Meeting prep briefs — "before your 2pm with X, here's where things
 * stand" packs for upcoming meetings with known attendees.
 *
 * Composed server-side (Python CLI) from already-cached data — the
 * enriched-events mart, cached contact contexts, and per-contact
 * topics — with no LLM, so this is cheap on page load. Types mirror
 * `src-tauri/src/commands/types.rs::MeetingPrepBrief`.
 *
 * sensitivity_tier: 3
 */

import { useCallback } from "react";
import { dedupInvoke } from "../utils/requestDedup";
import { useAsyncData, type AsyncDataResult } from "./useAsyncData";

export interface MeetingPrepOpenLoop {
  readonly topic: string;
  readonly description: string;
  readonly importance: number;
}

export interface MeetingPrepAttendee {
  readonly contact_name: string;
  readonly situation: string | null;
  readonly domains: readonly string[];
  readonly last_message_at: string | null;
  readonly last_message_preview: string | null;
  readonly open_loops: readonly MeetingPrepOpenLoop[];
}

export interface MeetingPrepBrief {
  readonly event_id: string;
  readonly title: string;
  readonly start_time: string;
  readonly location: string | null;
  readonly attendees: readonly MeetingPrepAttendee[];
  readonly generated_at: string;
}

export function useMeetingPrepBriefs(
  withinHours = 24,
): AsyncDataResult<MeetingPrepBrief[]> {
  const fetcher = useCallback(
    () =>
      dedupInvoke<MeetingPrepBrief[]>("get_meeting_prep_briefs", {
        withinHours,
      }),
    [withinHours],
  );
  return useAsyncData<MeetingPrepBrief[]>(fetcher);
}
