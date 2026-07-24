/**
 * Meeting prep card — a pre-event pack for upcoming meetings.
 *
 * For each meeting in the next N hours with a known attendee, shows
 * where things stand: the attendee's current situation, their open
 * loops (per-contact topics), and their last message. Composed
 * server-side from already-cached data (no LLM), so it renders
 * instantly and stays quiet (renders nothing) when there's nothing
 * upcoming.
 *
 * sensitivity_tier: 3
 */

import { CalendarClock, Flame, MapPin } from "lucide-react";
import Card from "./Card";
import {
  useMeetingPrepBriefs,
  type MeetingPrepAttendee,
  type MeetingPrepBrief,
} from "../../hooks/useMeetingPrep";

function formatStart(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, {
    weekday: "short",
    hour: "numeric",
    minute: "2-digit",
  });
}

function AttendeeBlock({ attendee }: { readonly attendee: MeetingPrepAttendee }) {
  return (
    <div className="rounded-2 border border-hairline bg-bg-2 p-3">
      <div className="flex items-center justify-between gap-2">
        <span className="text-sm font-medium text-ink">
          {attendee.contact_name}
        </span>
        {attendee.domains.length > 0 && (
          <span className="text-[10px] uppercase tracking-wide text-muted">
            {attendee.domains.join(" · ")}
          </span>
        )}
      </div>

      {attendee.situation && (
        <p className="mt-1 text-xs text-ink-2">{attendee.situation}</p>
      )}

      {attendee.open_loops.length > 0 && (
        <ul className="mt-2 space-y-1">
          {attendee.open_loops.map((loop) => (
            <li
              key={loop.topic}
              className="flex items-start gap-1.5 text-[11px] text-muted"
            >
              <Flame className="mt-0.5 h-2.5 w-2.5 shrink-0 text-indigo" />
              <span>
                <span className="text-ink">{loop.topic}</span>
                {loop.description && <> — {loop.description}</>}
              </span>
            </li>
          ))}
        </ul>
      )}

      {attendee.last_message_preview && (
        <p className="mt-2 truncate text-[11px] italic text-faint">
          Last: “{attendee.last_message_preview}”
        </p>
      )}
    </div>
  );
}

function BriefBlock({ brief }: { readonly brief: MeetingPrepBrief }) {
  return (
    <div className="space-y-2">
      <div className="flex items-baseline justify-between gap-2">
        <h4 className="truncate text-sm font-semibold text-ink">
          {brief.title}
        </h4>
        <span className="shrink-0 text-[11px] text-muted">
          {formatStart(brief.start_time)}
        </span>
      </div>
      {brief.location && (
        <p className="flex items-center gap-1 text-[11px] text-muted">
          <MapPin className="h-3 w-3" /> {brief.location}
        </p>
      )}
      <div className="space-y-2">
        {brief.attendees.map((a) => (
          <AttendeeBlock key={a.contact_name} attendee={a} />
        ))}
      </div>
    </div>
  );
}

function MeetingPrepCard() {
  const { data } = useMeetingPrepBriefs();
  const briefs = data ?? [];

  // Stay quiet when there's nothing upcoming — no empty-state clutter.
  if (briefs.length === 0) return null;

  return (
    <Card
      title="Meeting prep"
      icon={<CalendarClock className="h-4 w-4 text-indigo" />}
      meta={
        <span className="text-[11px] text-muted">
          next {briefs.length === 1 ? "meeting" : `${briefs.length} meetings`}
        </span>
      }
    >
      <div className="space-y-4">
        {briefs.map((b) => (
          <BriefBlock key={b.event_id} brief={b} />
        ))}
      </div>
    </Card>
  );
}

export default MeetingPrepCard;
