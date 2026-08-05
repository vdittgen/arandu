/*
    Staged health metrics with normalized float values.

    Column Sensitivity Tiers:
        id:               tier 1
        metric_type:      tier 2
        value:            tier 3
        unit:             tier 1
        recorded_at:      tier 2
        source:           tier 1
        sensitivity_tier: tier 1
        _loaded_at:       tier 1
*/

SELECT
    CAST(id AS TEXT)                        AS id,
    CAST(TRIM(metric_type) AS TEXT)         AS metric_type,
    CAST(value AS REAL)                     AS value,
    CAST(TRIM(unit) AS TEXT)                AS unit,
    recorded_at,
    CAST(TRIM(source) AS TEXT)              AS source,
    CAST(sensitivity_tier AS INTEGER)       AS sensitivity_tier,
    datetime('now')                         AS _loaded_at
FROM raw_health_metrics
