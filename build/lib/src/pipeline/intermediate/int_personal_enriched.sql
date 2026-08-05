/*
    Messages enriched with contact info and a domain category sourced
    from :mod:`src.pipeline.intermediate.int_labeled_messages`.

    The legacy CASE expression keyword-matched on sender domains and
    source channels, which left every WhatsApp message (i.e. nearly
    all of them) tagged ``"other"`` and starved the Work / Personal
    marts. We now defer to :class:`LabelerAgent` — its ``domain``
    field is already an LLM verdict over the message body, so we
    don't pay for the same call twice.

    Mapping from labeller domain → mart category:
        work       → work
        health     → health
        social     → personal   (mart_personal handles social events
                                 too, so collapse here)
        spiritual  → personal   (not a first-class mart bucket)
        personal   → personal
        anything else, or NULL → personal (safe default for messages
                                 that haven't been labelled yet, which
                                 is the dominant case in real data)

    Column Sensitivity Tiers:
        id:               tier 1
        source:           tier 1
        sender:           tier 2
        recipient:        tier 2
        content:          tier 3
        timestamp:        tier 2
        sensitivity_tier: tier 1
        message_length:   tier 1
        contact_name:     tier 2
        relationship:     tier 2
        message_category: tier 1
        _loaded_at:       tier 1
*/

SELECT
    m.id,
    m.source,
    m.sender,
    m.recipient,
    m.content,
    m.timestamp,
    m.sensitivity_tier,
    m.message_length,
    c.name                                  AS contact_name,
    c.relationship,
    CASE LOWER(COALESCE(lm.domain, 'personal'))
        WHEN 'work'      THEN 'work'
        WHEN 'health'    THEN 'health'
        WHEN 'social'    THEN 'personal'
        WHEN 'spiritual' THEN 'personal'
        WHEN 'personal'  THEN 'personal'
        ELSE 'personal'
    END                                     AS message_category,
    datetime('now')                         AS _loaded_at
FROM stg_messages m
LEFT JOIN stg_contacts c
    ON m.sender = c.email
    OR LOWER(m.sender) = LOWER(c.name)
LEFT JOIN int_labeled_messages lm
    ON lm.message_id = m.id
