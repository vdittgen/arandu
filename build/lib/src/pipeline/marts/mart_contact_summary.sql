/*
    Contact summary mart — per-contact aggregation with communication
    stats, channels, recency, relationship context, and contextual topics.

    Aggregates direct (1:1) messages from raw_messages per contact,
    then enriches with:
      - Apple Contacts metadata via three matching strategies
        (exact name → phone digit → first name)
      - Per-contact contextual topics from int_contact_topics (LLM-extracted)
        e.g., "hiring a psychologist for Repensar", "father's cancer treatment"
      - Topic importance weights for proactive guidance and notification priority

    Topic columns degrade gracefully when LLM is unavailable — they
    will auto-populate once the topic extraction pipeline runs.

    Column Sensitivity Tiers:
        contact_name:           tier 2
        whatsapp_name:          tier 2
        relationship:           tier 2
        email:                  tier 2
        phone:                  tier 2
        total_messages:         tier 1
        messages_sent:          tier 1
        messages_received:      tier 1
        messages_7d:            tier 1
        messages_30d:           tier 1
        first_message_at:       tier 2
        last_message_at:        tier 2
        days_since_last:        tier 1
        primary_channel:        tier 1
        avg_message_length:     tier 1
        active_topics_json:     tier 3
        top_topic:              tier 3
        max_topic_importance:   tier 2
        topic_count:            tier 1
        notification_priority:  tier 1
        sensitivity_tier:       tier 1
        _loaded_at:             tier 1
*/

WITH contact_msgs AS (
    SELECT
        COALESCE(
            NULLIF(NULLIF(NULLIF(m.sender_name, ''), 'Unknown'), 'me'),
            m.chat_name,
            m.sender
        )                                           AS contact_name,
        -- Keep the raw JID for phone matching later
        CASE
            WHEN m.is_from_me = 1 THEN m.chat_name
            ELSE m.sender
        END                                         AS raw_jid,
        m.source                                    AS channel,
        m.timestamp,
        m.is_from_me,
        LENGTH(m.content)                           AS msg_len
    FROM raw_messages m
    WHERE m.is_group = 0
      AND m.sender IS NOT NULL
      AND m.sender != ''
),
-- Lookup: chat_name JID → best known contact name
jid_names AS (
    SELECT
        m.chat_name                                 AS jid,
        m.sender_name                               AS known_name,
        COUNT(*)                                    AS cnt
    FROM raw_messages m
    WHERE m.is_group = 0
      AND m.is_from_me = 0
      AND m.sender_name IS NOT NULL
      AND m.sender_name != ''
      AND m.sender_name != 'Unknown'
    GROUP BY m.chat_name, m.sender_name
),
best_jid_name AS (
    SELECT jid, known_name
    FROM (
        SELECT jid, known_name,
               ROW_NUMBER() OVER (PARTITION BY jid ORDER BY cnt DESC) AS rn
        FROM jid_names
    )
    WHERE rn = 1
),
-- Re-resolve names and keep raw_jid for phone matching
resolved AS (
    SELECT
        COALESCE(bn.known_name, cm.contact_name)   AS contact_name,
        cm.raw_jid,
        cm.channel,
        cm.timestamp,
        cm.is_from_me,
        cm.msg_len
    FROM contact_msgs cm
    LEFT JOIN best_jid_name bn
        ON cm.contact_name = bn.jid
),
contact_agg AS (
    SELECT
        contact_name,
        -- Keep one representative JID for phone matching
        MIN(raw_jid)                                AS sample_jid,
        COUNT(*)                                    AS total_messages,
        SUM(CASE WHEN is_from_me = 1 THEN 1 ELSE 0 END)
                                                    AS messages_sent,
        SUM(CASE WHEN is_from_me = 0 THEN 1 ELSE 0 END)
                                                    AS messages_received,
        SUM(CASE
            WHEN DATE(timestamp) >= DATE('now', '-7 days')
            THEN 1 ELSE 0
        END)                                        AS messages_7d,
        SUM(CASE
            WHEN DATE(timestamp) >= DATE('now', '-30 days')
            THEN 1 ELSE 0
        END)                                        AS messages_30d,
        MIN(timestamp)                              AS first_message_at,
        MAX(timestamp)                              AS last_message_at,
        CAST(AVG(msg_len) AS INTEGER)               AS avg_message_length
    FROM resolved
    WHERE contact_name != 'me'
    GROUP BY contact_name
    HAVING COUNT(*) >= 3
),
channel_rank AS (
    SELECT
        COALESCE(bn.known_name, cm.contact_name)   AS contact_name,
        cm.channel,
        ROW_NUMBER() OVER (
            PARTITION BY COALESCE(bn.known_name, cm.contact_name)
            ORDER BY COUNT(*) DESC
        ) AS rn
    FROM contact_msgs cm
    LEFT JOIN best_jid_name bn
        ON cm.contact_name = bn.jid
    WHERE COALESCE(bn.known_name, cm.contact_name) != 'me'
    GROUP BY 1, 2
),
-- ---------------------------------------------------------------
-- Contextual topics from int_contact_topics (LLM-extracted)
-- Aggregate per contact: JSON array of active topics, top topic,
-- max importance, topic count
-- ---------------------------------------------------------------
active_topics AS (
    SELECT
        contact_name,
        topic,
        description,
        importance,
        status
    FROM int_contact_topics
    WHERE status = 'active'
),
topics_agg AS (
    SELECT
        contact_name,
        -- JSON array of all active topics with importance
        '[' || GROUP_CONCAT(
            '{"topic":' || json_quote(topic)
            || ',"description":' || json_quote(description)
            || ',"importance":' || CAST(importance AS TEXT)
            || '}',
            ','
        ) || ']'                                    AS active_topics_json,
        -- Highest importance topic
        MAX(importance)                             AS max_topic_importance,
        COUNT(*)                                    AS topic_count
    FROM active_topics
    GROUP BY contact_name
),
-- Top topic per contact (highest importance, tie-break by name)
top_topic AS (
    SELECT
        contact_name,
        topic AS top_topic
    FROM (
        SELECT
            contact_name,
            topic,
            ROW_NUMBER() OVER (
                PARTITION BY contact_name
                ORDER BY importance DESC, topic
            ) AS rn
        FROM active_topics
    )
    WHERE rn = 1
),
-- ---------------------------------------------------------------
-- Notification priority score (0-100)
-- Factors: recency, volume trend, topic importance, message volume
-- ---------------------------------------------------------------
priority_score AS (
    SELECT
        ca.contact_name,
        CAST(MIN(100, (
            -- Recency: 0-30 points (more recent = higher)
            CASE
                WHEN ca.messages_7d > 0 THEN 30
                WHEN ca.messages_30d > 0 THEN 18
                ELSE MAX(0, 10 - CAST(
                    julianday(DATE('now')) - julianday(DATE(ca.last_message_at))
                    AS INTEGER) / 10)
            END
            -- Volume trend: 0-20 points (7d vs 30d ratio)
            + CASE
                WHEN ca.messages_30d > 0
                THEN MIN(20, CAST(
                    ca.messages_7d * 4.0 / MAX(ca.messages_30d, 1) * 20
                    AS INTEGER))
                ELSE 0
            END
            -- Topic importance: 0-35 points (most important factor)
            + MIN(35, COALESCE(ta.max_topic_importance, 0) * 3
                      + COALESCE(ta.topic_count, 0) * 2)
            -- Message volume: 0-15 points
            + MIN(15, ca.total_messages / 50)
        )) AS INTEGER)                              AS notification_priority
    FROM contact_agg ca
    LEFT JOIN topics_agg ta
        ON ca.contact_name = ta.contact_name
),
-- ---------------------------------------------------------------
-- Contact matching: normalize phones for digit-based matching
-- ---------------------------------------------------------------
contact_phones AS (
    SELECT
        name,
        relationship,
        email,
        phone,
        -- Strip formatting: +, spaces, parens, dashes, dots
        REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(
            phone, '+', ''), ' ', ''), '(', ''), ')', ''), '-', ''), '.', ''
        ) AS phone_digits
    FROM stg_contacts
    WHERE phone IS NOT NULL AND phone != ''
),
-- Strategy 1: exact name match
match_name AS (
    SELECT
        ca.contact_name                             AS agg_name,
        c.name, c.relationship, c.email, c.phone,
        ROW_NUMBER() OVER (
            PARTITION BY ca.contact_name ORDER BY c.name
        ) AS rn
    FROM contact_agg ca
    INNER JOIN stg_contacts c
        ON LOWER(ca.contact_name) = LOWER(c.name)
),
-- Strategy 2: phone-digit match (last 8 digits of JID vs contact phone)
match_phone AS (
    SELECT
        ca.contact_name                             AS agg_name,
        cp.name, cp.relationship, cp.email, cp.phone,
        ROW_NUMBER() OVER (
            PARTITION BY ca.contact_name ORDER BY cp.name
        ) AS rn
    FROM contact_agg ca
    INNER JOIN contact_phones cp
        ON LENGTH(cp.phone_digits) >= 8
        AND SUBSTR(
            REPLACE(
                CASE
                    WHEN ca.sample_jid LIKE '%@%'
                    THEN SUBSTR(ca.sample_jid, 1, INSTR(ca.sample_jid, '@') - 1)
                    ELSE REPLACE(ca.contact_name, '+', '')
                END,
                '+', ''
            ),
            -8
        ) = SUBSTR(cp.phone_digits, -8)
    WHERE ca.contact_name NOT IN (SELECT agg_name FROM match_name WHERE rn = 1)
),
-- Strategy 3: first-name match (first word, min 3 chars)
match_first_name AS (
    SELECT
        ca.contact_name                             AS agg_name,
        c.name, c.relationship, c.email, c.phone,
        ROW_NUMBER() OVER (
            PARTITION BY ca.contact_name ORDER BY LENGTH(c.name)
        ) AS rn
    FROM contact_agg ca
    INNER JOIN stg_contacts c
        ON LENGTH(ca.contact_name) >= 3
        AND ca.contact_name NOT LIKE '+%'
        AND LOWER(
            CASE
                WHEN INSTR(ca.contact_name, ' ') > 0
                THEN SUBSTR(ca.contact_name, 1, INSTR(ca.contact_name, ' ') - 1)
                ELSE ca.contact_name
            END
        ) = LOWER(
            CASE
                WHEN INSTR(c.name, ' ') > 0
                THEN SUBSTR(c.name, 1, INSTR(c.name, ' ') - 1)
                ELSE c.name
            END
        )
        AND LENGTH(
            CASE
                WHEN INSTR(ca.contact_name, ' ') > 0
                THEN SUBSTR(ca.contact_name, 1, INSTR(ca.contact_name, ' ') - 1)
                ELSE ca.contact_name
            END
        ) >= 3
    WHERE ca.contact_name NOT IN (SELECT agg_name FROM match_name WHERE rn = 1)
      AND ca.contact_name NOT IN (SELECT agg_name FROM match_phone WHERE rn = 1)
),
-- Combine all matches (priority: name > phone > first_name)
best_match AS (
    SELECT agg_name, name, relationship, email, phone FROM match_name WHERE rn = 1
    UNION ALL
    SELECT agg_name, name, relationship, email, phone FROM match_phone WHERE rn = 1
    UNION ALL
    SELECT agg_name, name, relationship, email, phone FROM match_first_name WHERE rn = 1
)
SELECT
    -- Use Apple Contact name when matched, otherwise keep WhatsApp name
    COALESCE(bm.name, ca.contact_name)             AS contact_name,
    -- Keep original WhatsApp name for reference when it differs
    CASE
        WHEN bm.name IS NOT NULL AND bm.name != ca.contact_name
        THEN ca.contact_name
        ELSE NULL
    END                                             AS whatsapp_name,
    bm.relationship,
    bm.email,
    COALESCE(bm.phone,
        CASE
            WHEN ca.contact_name LIKE '+%' THEN ca.contact_name
            ELSE NULL
        END
    )                                               AS phone,
    ca.total_messages,
    ca.messages_sent,
    ca.messages_received,
    ca.messages_7d,
    ca.messages_30d,
    ca.first_message_at,
    ca.last_message_at,
    CAST(
        julianday(DATE('now')) - julianday(DATE(ca.last_message_at))
        AS INTEGER
    )                                               AS days_since_last,
    cr.channel                                      AS primary_channel,
    ca.avg_message_length,
    -- Contextual topics (populated when int_contact_topics has LLM data)
    ta.active_topics_json,
    tt.top_topic,
    COALESCE(ta.max_topic_importance, 0)            AS max_topic_importance,
    COALESCE(ta.topic_count, 0)                     AS topic_count,
    -- Notification priority: higher = more important for proactive alerts
    COALESCE(ps.notification_priority, 0)           AS notification_priority,
    2                                               AS sensitivity_tier,
    datetime('now')                                 AS _loaded_at
FROM contact_agg ca
LEFT JOIN channel_rank cr
    ON ca.contact_name = cr.contact_name
    AND cr.rn = 1
LEFT JOIN best_match bm
    ON ca.contact_name = bm.agg_name
LEFT JOIN topics_agg ta
    ON ca.contact_name = ta.contact_name
LEFT JOIN top_topic tt
    ON ca.contact_name = tt.contact_name
LEFT JOIN priority_score ps
    ON ca.contact_name = ps.contact_name
ORDER BY ca.total_messages DESC
