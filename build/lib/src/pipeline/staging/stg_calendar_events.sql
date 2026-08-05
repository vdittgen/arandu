/*
    Staged calendar events with parsed attendees and computed duration.

    Column Sensitivity Tiers:
        id:                       tier 1
        title:                    tier 1
        description:              tier 2
        start_time:               tier 2
        end_time:                 tier 2
        location:                 tier 2
        attendees:                tier 2
        sensitivity_tier:         tier 1
        attendees_count:          tier 1
        duration_minutes:         tier 1
        calendar_name:            tier 1
        calendar_owner_email:     tier 2
        is_shared_calendar:       tier 1
        is_subscribed_calendar:   tier 1
        self_response_status:     tier 2
        event_origin:             tier 1
        _loaded_at:               tier 1
*/

SELECT
    CAST(id AS TEXT)                                        AS id,
    CAST(TRIM(title) AS TEXT)                               AS title,
    CAST(description AS TEXT)                               AS description,
    start_time,
    end_time,
    CAST(TRIM(location) AS TEXT)                            AS location,
    attendees,
    CAST(sensitivity_tier AS INTEGER)                       AS sensitivity_tier,
    CAST(json_array_length(attendees) AS INTEGER)           AS attendees_count,
    CAST((julianday(end_time) - julianday(start_time)) * 1440 AS INTEGER) AS duration_minutes,
    CAST(calendar_name AS TEXT)                             AS calendar_name,
    CAST(calendar_owner_email AS TEXT)                      AS calendar_owner_email,
    CAST(COALESCE(is_shared_calendar, 0) AS INTEGER)        AS is_shared_calendar,
    CAST(COALESCE(is_subscribed_calendar, 0) AS INTEGER)    AS is_subscribed_calendar,
    CAST(self_response_status AS TEXT)                      AS self_response_status,
    CAST(COALESCE(event_origin, 'personal') AS TEXT)        AS event_origin,
    datetime('now')                                         AS _loaded_at
FROM raw_calendar_events
