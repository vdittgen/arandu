/*
    Staged reminders with overdue flag and days until due.

    Column Sensitivity Tiers:
        id:               tier 1
        source:           tier 1
        title:            tier 1
        due_date:         tier 1
        notes:            tier 2
        completed:        tier 1
        list_name:        tier 1
        sensitivity_tier: tier 1
        is_overdue:       tier 1
        days_until_due:   tier 1
        _loaded_at:       tier 1
*/

SELECT
    CAST(id AS TEXT)                        AS id,
    CAST(TRIM(source) AS TEXT)              AS source,
    CAST(TRIM(title) AS TEXT)               AS title,
    due_date,
    CAST(notes AS TEXT)                     AS notes,
    CAST(completed AS INTEGER)              AS completed,
    CAST(TRIM(list_name) AS TEXT)           AS list_name,
    CAST(sensitivity_tier AS INTEGER)       AS sensitivity_tier,
    CASE
        WHEN due_date IS NOT NULL
         AND due_date < datetime('now')
         AND (completed IS NULL OR completed = 0)
        THEN 1
        ELSE 0
    END                                     AS is_overdue,
    CASE
        WHEN due_date IS NOT NULL
        THEN CAST(
            julianday(DATE(due_date)) - julianday(DATE('now'))
            AS INTEGER
        )
        ELSE NULL
    END                                     AS days_until_due,
    datetime('now')                         AS _loaded_at
FROM raw_reminders
