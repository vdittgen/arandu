/*
    Staged notes with computed word count and parsed tags.

    Column Sensitivity Tiers:
        id:               tier 1
        title:            tier 1
        content:          tier 2
        source:           tier 1
        created_at:       tier 1
        updated_at:       tier 1
        tags:             tier 1
        sensitivity_tier: tier 1
        word_count:       tier 1
        tags_csv:         tier 1
        _loaded_at:       tier 1
*/

SELECT
    CAST(id AS TEXT)                        AS id,
    CAST(TRIM(title) AS TEXT)               AS title,
    CAST(content AS TEXT)                   AS content,
    CAST(TRIM(source) AS TEXT)              AS source,
    created_at,
    updated_at,
    tags,
    CAST(sensitivity_tier AS INTEGER)       AS sensitivity_tier,
    CASE
        WHEN TRIM(content) = '' THEN 0
        ELSE LENGTH(TRIM(content)) - LENGTH(REPLACE(TRIM(content), ' ', '')) + 1
    END                                     AS word_count,
    REPLACE(
        REPLACE(
            REPLACE(CAST(tags AS TEXT), '"', ''),
            '[', ''
        ),
        ']', ''
    )                                       AS tags_csv,
    datetime('now')                         AS _loaded_at
FROM raw_notes
