/*
    Staged emails with computed recipient count and body length.

    Column Sensitivity Tiers:
        id:               tier 1
        source:           tier 1
        message_id:       tier 1
        subject:          tier 2
        from_address:     tier 2
        to_addresses:     tier 2
        date:             tier 2
        body_preview:     tier 3
        is_read:          tier 1
        folder:           tier 1
        labels:           tier 1
        sensitivity_tier: tier 1
        recipient_count:  tier 1
        body_length:      tier 1
        labels_csv:       tier 1
        _loaded_at:       tier 1
*/

SELECT
    CAST(id AS TEXT)                        AS id,
    CAST(TRIM(source) AS TEXT)              AS source,
    CAST(message_id AS TEXT)                AS message_id,
    CAST(TRIM(subject) AS TEXT)             AS subject,
    CAST(TRIM(from_address) AS TEXT)        AS from_address,
    to_addresses,
    date,
    CAST(body_preview AS TEXT)              AS body_preview,
    CAST(is_read AS INTEGER)               AS is_read,
    CAST(TRIM(folder) AS TEXT)             AS folder,
    labels,
    CAST(sensitivity_tier AS INTEGER)       AS sensitivity_tier,
    CAST(
        json_array_length(to_addresses) AS INTEGER
    )                                       AS recipient_count,
    CAST(LENGTH(body_preview) AS INTEGER)  AS body_length,
    REPLACE(
        REPLACE(
            REPLACE(CAST(labels AS TEXT), '"', ''),
            '[', ''
        ),
        ']', ''
    )                                       AS labels_csv,
    datetime('now')                         AS _loaded_at
FROM raw_emails
