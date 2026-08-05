/*
    Staged contacts with computed days since last contact.

    Column Sensitivity Tiers:
        id:                      tier 1
        name:                    tier 2
        email:                   tier 3
        phone:                   tier 3
        relationship:            tier 2
        notes:                   tier 2
        last_contact:            tier 2
        sensitivity_tier:        tier 1
        days_since_last_contact: tier 2
        _loaded_at:              tier 1
*/

SELECT
    CAST(id AS TEXT)                        AS id,
    CAST(TRIM(name) AS TEXT)                AS name,
    CAST(TRIM(email) AS TEXT)               AS email,
    CAST(TRIM(phone) AS TEXT)               AS phone,
    CAST(TRIM(relationship) AS TEXT)        AS relationship,
    CAST(notes AS TEXT)                     AS notes,
    last_contact,
    CAST(sensitivity_tier AS INTEGER)       AS sensitivity_tier,
    CASE
        WHEN last_contact IS NOT NULL
        THEN CAST(julianday(DATE('now')) - julianday(DATE(last_contact)) AS INTEGER)
        ELSE NULL
    END                                     AS days_since_last_contact,
    datetime('now')                         AS _loaded_at
FROM raw_contacts
