/*
    Today mart — primary data source for the Daily Dashboard.

    Combines today's calendar events, recent messages, and notes created
    today into a single feed. Includes a placeholder coaching_phrase column
    to be populated by the LLM agent.

    Column Sensitivity Tiers:
        item_type:        tier 1
        id:               tier 1
        title:            tier 2
        detail:           tier 3
        occurred_at:      tier 2
        category:         tier 1
        duration_minutes: tier 1
        sensitivity_tier: tier 1
        event_origin:     tier 1
        coaching_phrase:  tier 2
        _loaded_at:       tier 1

    "Today" means the user's local calendar day, not UTC.

    `DATE('now')` is SQLite's UTC clock. Filtering on it meant that for
    a user at UTC-3 the feed rolled over to tomorrow from 21:00 local:
    the rest of today's events vanished and tomorrow's appeared. At
    UTC+14 it was wrong the other way for ten hours a day. Only UTC
    users ever saw the right day for a full 24 hours.

    Both sides of each comparison therefore need care, because the
    stored values are not all the same kind of thing:

      - A value carrying an explicit zone ("...Z" or "...+05:30")
        denotes an *instant*. Which local day it falls on depends on
        the reader, so it must be converted: DATE(v, 'localtime').
      - A bare date ("2026-08-03" — all-day events, some reminder due
        dates) is already a calendar day. Converting it would shift it
        a day backwards for every negative UTC offset, so it must be
        left alone.
      - A naive datetime has no zone to convert from, and is treated as
        already being in the reader's frame.

    Hence `local_day()` below, spelled out inline because these models
    are executed as plain SQL against SQLite with no UDFs registered.

    The reference clock is the machine's timezone. For a local-first
    desktop app that is the clock the user is reading; `user_timezone`
    in settings is a declared preference used for prompt context, and
    routing it here would mean parameterising the mart models.
*/

SELECT
    'event'                                 AS item_type,
    e.id,
    e.title,
    e.description                           AS detail,
    e.start_time                            AS occurred_at,
    e.event_category                        AS category,
    e.duration_minutes,
    e.sensitivity_tier,
    e.event_origin                          AS event_origin,
    CAST(NULL AS TEXT)                      AS coaching_phrase,
    datetime('now')                         AS _loaded_at
FROM int_events_enriched e
WHERE CASE
           WHEN e.start_time LIKE '%Z'
             OR e.start_time GLOB '*[+-][0-9][0-9]:[0-9][0-9]'
           THEN DATE(e.start_time, 'localtime')
           ELSE DATE(e.start_time)
       END = DATE('now', 'localtime')

UNION ALL

SELECT
    'message'                               AS item_type,
    m.id,
    m.sender || ': ' || SUBSTR(m.content, 1, 80) AS title,
    m.content                               AS detail,
    m.timestamp                             AS occurred_at,
    m.message_category                      AS category,
    CAST(NULL AS INTEGER)                   AS duration_minutes,
    m.sensitivity_tier,
    CAST(NULL AS TEXT)                      AS event_origin,
    CAST(NULL AS TEXT)                      AS coaching_phrase,
    datetime('now')                         AS _loaded_at
FROM int_personal_enriched m
WHERE CASE
           WHEN m.timestamp LIKE '%Z'
             OR m.timestamp GLOB '*[+-][0-9][0-9]:[0-9][0-9]'
           THEN DATE(m.timestamp, 'localtime')
           ELSE DATE(m.timestamp)
       END = DATE('now', 'localtime')

UNION ALL

SELECT
    'note'                                  AS item_type,
    n.id,
    n.title,
    n.content                               AS detail,
    n.created_at                            AS occurred_at,
    'note'                                  AS category,
    CAST(NULL AS INTEGER)                   AS duration_minutes,
    n.sensitivity_tier,
    CAST(NULL AS TEXT)                      AS event_origin,
    CAST(NULL AS TEXT)                      AS coaching_phrase,
    datetime('now')                         AS _loaded_at
FROM stg_notes n
WHERE CASE
           WHEN n.created_at LIKE '%Z'
             OR n.created_at GLOB '*[+-][0-9][0-9]:[0-9][0-9]'
           THEN DATE(n.created_at, 'localtime')
           ELSE DATE(n.created_at)
       END = DATE('now', 'localtime')

UNION ALL

SELECT
    'email'                                 AS item_type,
    e.id,
    e.subject                               AS title,
    e.body_preview                          AS detail,
    e.date                                  AS occurred_at,
    CASE
        WHEN e.from_address LIKE '%@company.com'
        THEN 'work'
        ELSE 'other'
    END                                     AS category,
    CAST(NULL AS INTEGER)                   AS duration_minutes,
    e.sensitivity_tier,
    CAST(NULL AS TEXT)                      AS event_origin,
    CAST(NULL AS TEXT)                      AS coaching_phrase,
    datetime('now')                         AS _loaded_at
FROM stg_emails e
WHERE CASE
           WHEN e.date LIKE '%Z'
             OR e.date GLOB '*[+-][0-9][0-9]:[0-9][0-9]'
           THEN DATE(e.date, 'localtime')
           ELSE DATE(e.date)
       END = DATE('now', 'localtime')

UNION ALL

SELECT
    'reminder'                              AS item_type,
    r.id,
    r.title,
    r.notes                                 AS detail,
    COALESCE(r.due_date,
             datetime('now', 'localtime'))  AS occurred_at,
    COALESCE(r.list_name, 'default')        AS category,
    CAST(NULL AS INTEGER)                   AS duration_minutes,
    r.sensitivity_tier,
    CAST(NULL AS TEXT)                      AS event_origin,
    CAST(NULL AS TEXT)                      AS coaching_phrase,
    datetime('now')                         AS _loaded_at
FROM stg_reminders r
WHERE (CASE
           WHEN r.due_date LIKE '%Z'
             OR r.due_date GLOB '*[+-][0-9][0-9]:[0-9][0-9]'
           THEN DATE(r.due_date, 'localtime')
           ELSE DATE(r.due_date)
       END = DATE('now', 'localtime')
       OR r.due_date IS NULL)
  AND (r.completed IS NULL OR r.completed = 0)
