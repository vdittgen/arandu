/*
    Work domain mart — work messages, meetings, and work-related notes.

    Combines work messages, meeting events, and work notes into a single
    domain table. Includes a placeholder topic column for future
    project/topic grouping by ML.

    Column Sensitivity Tiers:
        item_type:        tier 1
        id:               tier 1
        title:            tier 1
        detail:           tier 2
        occurred_at:      tier 2
        contact_name:     tier 2
        sensitivity_tier: tier 1
        topic:            tier 1
        _loaded_at:       tier 1
*/

SELECT
    'message'                               AS item_type,
    m.id,
    m.sender || ': ' || SUBSTR(m.content, 1, 80) AS title,
    m.content                               AS detail,
    m.timestamp                             AS occurred_at,
    m.contact_name,
    m.sensitivity_tier,
    CAST(NULL AS TEXT)                       AS topic,
    datetime('now')                         AS _loaded_at
FROM int_personal_enriched m
WHERE m.message_category = 'work'

UNION ALL

SELECT
    'event'                                 AS item_type,
    e.id,
    e.title,
    e.description                           AS detail,
    e.start_time                            AS occurred_at,
    e.known_attendee_names                  AS contact_name,
    e.sensitivity_tier,
    CAST(NULL AS TEXT)                       AS topic,
    datetime('now')                         AS _loaded_at
FROM int_events_enriched e
WHERE e.event_category = 'meeting'
  AND COALESCE(e.event_origin, 'personal') = 'personal'

UNION ALL

SELECT
    'note'                                  AS item_type,
    n.id,
    n.title,
    n.content                               AS detail,
    n.created_at                            AS occurred_at,
    CAST(NULL AS TEXT)                       AS contact_name,
    n.sensitivity_tier,
    CAST(NULL AS TEXT)                       AS topic,
    datetime('now')                         AS _loaded_at
FROM stg_notes n
WHERE n.tags_csv LIKE '%work%'
   OR n.tags_csv LIKE '%meetings%'
   OR n.tags_csv LIKE '%planning%'
   OR n.tags_csv LIKE '%architecture%'
   OR n.tags_csv LIKE '%arandu%'
