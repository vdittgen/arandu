/*
    Staged messages with type casting, computed fields, and sensitivity metadata.

    Column Sensitivity Tiers:
        id:               tier 1
        source:           tier 1
        sender:           tier 2
        recipient:        tier 2
        content:          tier 3
        timestamp:        tier 2
        metadata:         tier 2
        sensitivity_tier: tier 1
        message_length:   tier 1
        _loaded_at:       tier 1
*/

SELECT
    CAST(id AS TEXT)                        AS id,
    CAST(TRIM(source) AS TEXT)              AS source,
    CAST(TRIM(sender) AS TEXT)              AS sender,
    CAST(TRIM(recipient) AS TEXT)           AS recipient,
    CAST(content AS TEXT)                   AS content,
    timestamp,
    metadata,
    CAST(sensitivity_tier AS INTEGER)       AS sensitivity_tier,
    CAST(LENGTH(content) AS INTEGER)        AS message_length,
    datetime('now')                         AS _loaded_at
FROM raw_messages
