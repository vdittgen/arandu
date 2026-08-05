/*
    Health domain mart — latest metrics with 7-day trends and anomaly flags.

    Computes rolling 7-day average and standard deviation per metric type,
    then flags values exceeding 2 standard deviations from the rolling mean
    as potential anomalies. All data is tier 3 (high sensitivity).

    SQLite does not have STDDEV_SAMP as a window function, so we compute it
    manually using SUM(v*v), SUM(v), COUNT(v) windows and the population
    variance formula adjusted for sample size (Bessel's correction).

    Column Sensitivity Tiers:
        id:               tier 3
        metric_type:      tier 3
        value:            tier 3
        unit:             tier 3
        recorded_at:      tier 3
        source:           tier 3
        sensitivity_tier: tier 3
        avg_7d:           tier 3
        stddev_7d:        tier 3
        is_anomaly:       tier 3
        is_latest:        tier 3
        _loaded_at:       tier 3
*/

WITH metrics_with_windows AS (
    SELECT
        id,
        metric_type,
        value,
        unit,
        recorded_at,
        source,
        sensitivity_tier,
        AVG(value) OVER (
            PARTITION BY metric_type
            ORDER BY recorded_at
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        )                                       AS avg_7d,
        SUM(value * value) OVER (
            PARTITION BY metric_type
            ORDER BY recorded_at
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        )                                       AS sum_sq,
        SUM(value) OVER (
            PARTITION BY metric_type
            ORDER BY recorded_at
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        )                                       AS sum_v,
        COUNT(value) OVER (
            PARTITION BY metric_type
            ORDER BY recorded_at
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        )                                       AS cnt,
        ROW_NUMBER() OVER (
            PARTITION BY metric_type
            ORDER BY recorded_at DESC
        )                                       AS _rn
    FROM stg_health_metrics
),
metrics_with_stats AS (
    SELECT
        *,
        CASE
            WHEN cnt > 1
            THEN SQRT(
                (sum_sq - sum_v * sum_v / cnt) / (cnt - 1)
            )
            ELSE NULL
        END                                     AS stddev_7d
    FROM metrics_with_windows
)
SELECT
    id,
    metric_type,
    value,
    unit,
    recorded_at,
    source,
    sensitivity_tier,
    ROUND(avg_7d, 2)                        AS avg_7d,
    ROUND(stddev_7d, 2)                     AS stddev_7d,
    CASE
        WHEN stddev_7d IS NOT NULL
         AND stddev_7d > 0
         AND ABS(value - avg_7d) > 2 * stddev_7d
        THEN 1
        ELSE 0
    END                                     AS is_anomaly,
    CASE WHEN _rn = 1 THEN 1 ELSE 0 END    AS is_latest,
    datetime('now')                         AS _loaded_at
FROM metrics_with_stats
