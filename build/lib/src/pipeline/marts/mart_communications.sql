/*
    Communications analytics mart — daily volume by channel and category.

    Aggregates from int_communications_enriched to produce daily
    communication statistics with top sender per group.

    Column Sensitivity Tiers:
        summary_date:       tier 1
        channel_type:       tier 1
        comm_category:      tier 1
        message_count:      tier 1
        avg_content_length: tier 1
        top_sender:         tier 2
        sensitivity_tier:   tier 1
        _loaded_at:         tier 1
*/

WITH ranked_senders AS (
    SELECT
        DATE(occurred_at)                   AS summary_date,
        channel_type,
        comm_category,
        sender,
        COUNT(*)                            AS sender_count,
        ROW_NUMBER() OVER (
            PARTITION BY
                DATE(occurred_at),
                channel_type,
                comm_category
            ORDER BY COUNT(*) DESC
        )                                   AS rn
    FROM int_communications_enriched
    GROUP BY 1, 2, 3, 4
)
SELECT
    agg.summary_date,
    agg.channel_type,
    agg.comm_category,
    agg.message_count,
    agg.avg_content_length,
    rs.sender                               AS top_sender,
    2                                       AS sensitivity_tier,
    datetime('now')                         AS _loaded_at
FROM (
    SELECT
        DATE(occurred_at)                   AS summary_date,
        channel_type,
        comm_category,
        COUNT(*)                            AS message_count,
        ROUND(AVG(content_length), 1)       AS avg_content_length
    FROM int_communications_enriched
    GROUP BY 1, 2, 3
) agg
LEFT JOIN ranked_senders rs
    ON agg.summary_date = rs.summary_date
    AND agg.channel_type = rs.channel_type
    AND agg.comm_category = rs.comm_category
    AND rs.rn = 1
