/*
    Daily aggregation across all staging models — foundation for the
    Daily Dashboard.

    Aggregates messages, events, notes, emails, reminders, and health
    metrics by calendar date.

    Column Sensitivity Tiers:
        summary_date:        tier 1
        message_count:       tier 1
        avg_message_length:  tier 1
        event_count:         tier 1
        total_meeting_hours: tier 1
        notes_created:       tier 1
        email_count:         tier 1
        reminder_count:      tier 1
        overdue_reminders:   tier 1
        latest_heart_rate:   tier 3
        latest_steps:        tier 3
        latest_sleep_hours:  tier 3
        latest_weight_kg:    tier 3
        sensitivity_tier:    tier 1
        _loaded_at:          tier 1
*/

WITH daily_messages AS (
    SELECT
        DATE(timestamp)             AS summary_date,
        COUNT(*)                    AS message_count,
        AVG(message_length)         AS avg_message_length
    FROM stg_messages
    GROUP BY 1
),
daily_events AS (
    SELECT
        DATE(start_time)            AS summary_date,
        COUNT(*)                    AS event_count,
        SUM(duration_minutes) / 60.0 AS total_meeting_hours
    FROM stg_calendar_events
    GROUP BY 1
),
daily_notes AS (
    SELECT
        DATE(created_at)            AS summary_date,
        COUNT(*)                    AS notes_created
    FROM stg_notes
    GROUP BY 1
),
daily_emails AS (
    SELECT
        DATE(date)                  AS summary_date,
        COUNT(*)                    AS email_count
    FROM stg_emails
    GROUP BY 1
),
daily_reminders AS (
    SELECT
        DATE(due_date)              AS summary_date,
        COUNT(*)                    AS reminder_count,
        SUM(CASE WHEN is_overdue = 1 THEN 1 ELSE 0 END) AS overdue_reminders
    FROM stg_reminders
    WHERE due_date IS NOT NULL
    GROUP BY 1
),
daily_health AS (
    SELECT
        DATE(recorded_at)           AS summary_date,
        MAX(CASE WHEN metric_type = 'heart_rate' THEN value END)  AS latest_heart_rate,
        MAX(CASE WHEN metric_type = 'steps' THEN value END)       AS latest_steps,
        MAX(CASE WHEN metric_type = 'sleep_hours' THEN value END) AS latest_sleep_hours,
        MAX(CASE WHEN metric_type = 'weight_kg' THEN value END)   AS latest_weight_kg
    FROM stg_health_metrics
    GROUP BY 1
),
all_dates AS (
    SELECT summary_date FROM daily_messages
    UNION
    SELECT summary_date FROM daily_events
    UNION
    SELECT summary_date FROM daily_notes
    UNION
    SELECT summary_date FROM daily_emails
    UNION
    SELECT summary_date FROM daily_reminders
    UNION
    SELECT summary_date FROM daily_health
)
SELECT
    d.summary_date,
    COALESCE(m.message_count, 0)        AS message_count,
    COALESCE(m.avg_message_length, 0.0) AS avg_message_length,
    COALESCE(e.event_count, 0)          AS event_count,
    COALESCE(e.total_meeting_hours, 0.0) AS total_meeting_hours,
    COALESCE(n.notes_created, 0)        AS notes_created,
    COALESCE(em.email_count, 0)         AS email_count,
    COALESCE(rm.reminder_count, 0)      AS reminder_count,
    COALESCE(rm.overdue_reminders, 0)   AS overdue_reminders,
    h.latest_heart_rate,
    h.latest_steps,
    h.latest_sleep_hours,
    h.latest_weight_kg,
    3                                   AS sensitivity_tier,
    datetime('now')                     AS _loaded_at
FROM all_dates d
LEFT JOIN daily_messages m ON d.summary_date = m.summary_date
LEFT JOIN daily_events e   ON d.summary_date = e.summary_date
LEFT JOIN daily_notes n      ON d.summary_date = n.summary_date
LEFT JOIN daily_emails em    ON d.summary_date = em.summary_date
LEFT JOIN daily_reminders rm ON d.summary_date = rm.summary_date
LEFT JOIN daily_health h     ON d.summary_date = h.summary_date
