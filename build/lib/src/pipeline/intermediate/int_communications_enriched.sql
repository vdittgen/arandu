/*
    Unified communications: messages + emails enriched with contact info.

    UNIONs stg_messages and stg_emails into a common schema, then
    LEFT JOINs with stg_contacts to resolve sender identity and
    classify as work/personal/health/other.

    Column Sensitivity Tiers:
        id:               tier 1
        channel_type:     tier 1
        sender:           tier 2
        content_preview:  tier 3
        occurred_at:      tier 2
        sensitivity_tier: tier 1
        content_length:   tier 1
        contact_name:     tier 2
        relationship:     tier 2
        comm_category:    tier 1
        _loaded_at:       tier 1
*/

WITH unified AS (
    SELECT
        id,
        source                              AS channel_type,
        sender,
        SUBSTR(content, 1, 200)             AS content_preview,
        timestamp                           AS occurred_at,
        sensitivity_tier,
        message_length                      AS content_length
    FROM stg_messages

    UNION ALL

    SELECT
        id,
        'email'                             AS channel_type,
        from_address                        AS sender,
        body_preview                        AS content_preview,
        date                                AS occurred_at,
        sensitivity_tier,
        body_length                         AS content_length
    FROM stg_emails
)
SELECT
    u.id,
    u.channel_type,
    u.sender,
    u.content_preview,
    u.occurred_at,
    u.sensitivity_tier,
    u.content_length,
    c.name                                  AS contact_name,
    c.relationship,
    CASE
        WHEN u.channel_type = 'slack' THEN 'work'
        WHEN u.sender LIKE '%@company.com' THEN 'work'
        WHEN c.relationship IN ('doctor', 'therapist') THEN 'health'
        WHEN c.relationship IN ('family', 'friend') THEN 'personal'
        WHEN u.channel_type = 'imessage' THEN 'personal'
        ELSE 'other'
    END                                     AS comm_category,
    datetime('now')                         AS _loaded_at
FROM unified u
LEFT JOIN stg_contacts c
    ON u.sender = c.email
    OR LOWER(u.sender) = LOWER(c.name)
