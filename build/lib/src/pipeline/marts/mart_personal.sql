/*
    Personal domain mart — non-work messages, social events, and personal notes.

    Combines personal/health messages, social events, and personal notes
    into a single domain table filtered to non-work content. Includes a
    placeholder emotional_label column for future ML enrichment.

    Column Sensitivity Tiers:
        item_type:        tier 1
        id:               tier 1
        title:            tier 2
        detail:           tier 3
        occurred_at:      tier 2
        contact_name:     tier 2
        sensitivity_tier: tier 1
        emotional_label:  tier 3
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
    CAST(NULL AS TEXT)                       AS emotional_label,
    datetime('now')                         AS _loaded_at
FROM int_personal_enriched m
WHERE m.message_category IN ('personal', 'health')

UNION ALL

SELECT
    'event'                                 AS item_type,
    e.id,
    e.title,
    e.description                           AS detail,
    e.start_time                            AS occurred_at,
    e.known_attendee_names                  AS contact_name,
    e.sensitivity_tier,
    CAST(NULL AS TEXT)                       AS emotional_label,
    datetime('now')                         AS _loaded_at
FROM int_events_enriched e
WHERE e.event_category IN ('social', 'health')
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
    CAST(NULL AS TEXT)                       AS emotional_label,
    datetime('now')                         AS _loaded_at
FROM stg_notes n
WHERE n.tags_csv LIKE '%personal%'
   OR n.tags_csv LIKE '%journal%'
   OR n.tags_csv LIKE '%gratitude%'
   OR n.tags_csv LIKE '%friends%'
   OR n.tags_csv LIKE '%social%'
   OR n.tags_csv LIKE '%health%'
