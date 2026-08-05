"""Kuzu graph schema for Arandu.

Defines node tables (Person, Event, Place, Emotion, Idea, Topic) and
relationship tables that connect them.  All relationship tables carry
``weight DOUBLE``, ``timestamp TIMESTAMP``, and ``sensitivity_tier INT64``
so the firewall can filter graph traversals by sensitivity at query time.

All DDL is idempotent (``IF NOT EXISTS``).

Data Sensitivity Tiers
----------------------
- Tier 1 (low):    general preferences, interests
- Tier 2 (medium): habits, routines, people names
- Tier 3 (high):   health, finances, emotions, traumas
"""

from __future__ import annotations

from src.core.kuzu.engine import GraphEngine

# ---------------------------------------------------------------------------
# Node table DDL
# ---------------------------------------------------------------------------

# sensitivity_tier default 2 — people nodes reveal relationships
NODE_PERSON = """
CREATE NODE TABLE IF NOT EXISTS Person (
    id               STRING,
    name             STRING,
    relationship     STRING,
    sensitivity_tier INT64,
    PRIMARY KEY (id)
)
"""

# sensitivity_tier default 2 — event titles/attendees reveal social patterns
NODE_EVENT = """
CREATE NODE TABLE IF NOT EXISTS Event (
    id               STRING,
    title            STRING,
    event_type       STRING,
    date             TIMESTAMP,
    sensitivity_tier INT64,
    PRIMARY KEY (id)
)
"""

# sensitivity_tier default 1 — place names are generally low-sensitivity
NODE_PLACE = """
CREATE NODE TABLE IF NOT EXISTS Place (
    id               STRING,
    name             STRING,
    place_type       STRING,
    latitude         DOUBLE,
    longitude        DOUBLE,
    sensitivity_tier INT64,
    PRIMARY KEY (id)
)
"""

# sensitivity_tier default 3 — emotions reveal mental/emotional state
NODE_EMOTION = """
CREATE NODE TABLE IF NOT EXISTS Emotion (
    id               STRING,
    name             STRING,
    valence          DOUBLE,
    arousal          DOUBLE,
    sensitivity_tier INT64,
    PRIMARY KEY (id)
)
"""

# sensitivity_tier default 1 — ideas / concepts are low-sensitivity
NODE_IDEA = """
CREATE NODE TABLE IF NOT EXISTS Idea (
    id               STRING,
    title            STRING,
    description      STRING,
    domain           STRING,
    sensitivity_tier INT64,
    PRIMARY KEY (id)
)
"""

# sensitivity_tier default 1 — topics are general concepts
NODE_TOPIC = """
CREATE NODE TABLE IF NOT EXISTS Topic (
    id               STRING,
    name             STRING,
    category         STRING,
    sensitivity_tier INT64,
    PRIMARY KEY (id)
)
"""

# sensitivity_tier default 2 — singleton representing the user
NODE_SELF = """
CREATE NODE TABLE IF NOT EXISTS Self (
    id               STRING,
    name             STRING,
    bio              STRING,
    sensitivity_tier INT64,
    PRIMARY KEY (id)
)
"""

# ---------------------------------------------------------------------------
# Relationship table DDL
# All edges share: weight DOUBLE, timestamp TIMESTAMP, sensitivity_tier INT64
# ---------------------------------------------------------------------------

# Person ↔ Person social graph
REL_KNOWS = """
CREATE REL TABLE IF NOT EXISTS KNOWS (
    FROM Person TO Person,
    weight           DOUBLE,
    timestamp        TIMESTAMP,
    sensitivity_tier INT64
)
"""

# Person attended an Event
REL_PARTICIPATED_IN = """
CREATE REL TABLE IF NOT EXISTS PARTICIPATED_IN (
    FROM Person TO Event,
    weight           DOUBLE,
    timestamp        TIMESTAMP,
    sensitivity_tier INT64
)
"""

# Person felt / Event triggered an Emotion
REL_FELT = """
CREATE REL TABLE IF NOT EXISTS FELT (
    FROM Person TO Emotion,
    FROM Event  TO Emotion,
    weight           DOUBLE,
    timestamp        TIMESTAMP,
    sensitivity_tier INT64
)
"""

# Event took place at a Place
REL_LOCATED_AT = """
CREATE REL TABLE IF NOT EXISTS LOCATED_AT (
    FROM Event TO Place,
    weight           DOUBLE,
    timestamp        TIMESTAMP,
    sensitivity_tier INT64
)
"""

# Idea is related to a Topic, or another Idea
REL_RELATED_TO = """
CREATE REL TABLE IF NOT EXISTS RELATED_TO (
    FROM Idea TO Topic,
    FROM Idea TO Idea,
    weight           DOUBLE,
    timestamp        TIMESTAMP,
    sensitivity_tier INT64
)
"""

# Person / Topic is mentioned in an Event
REL_MENTIONED_IN = """
CREATE REL TABLE IF NOT EXISTS MENTIONED_IN (
    FROM Person TO Event,
    FROM Topic  TO Event,
    weight           DOUBLE,
    timestamp        TIMESTAMP,
    sensitivity_tier INT64
)
"""

# Event / Idea is tagged with a Topic
REL_TAGGED_WITH = """
CREATE REL TABLE IF NOT EXISTS TAGGED_WITH (
    FROM Event TO Topic,
    FROM Idea  TO Topic,
    weight           DOUBLE,
    timestamp        TIMESTAMP,
    sensitivity_tier INT64
)
"""

# Self → Person (learned relationship context from conversations)
REL_HAS_RELATIONSHIP = """
CREATE REL TABLE IF NOT EXISTS HAS_RELATIONSHIP (
    FROM Self TO Person,
    context          STRING,
    relationship_type STRING,
    weight           DOUBLE,
    timestamp        TIMESTAMP,
    sensitivity_tier INT64
)
"""

# Self → Topic (learned interests and preferences)
REL_INTERESTED_IN = """
CREATE REL TABLE IF NOT EXISTS INTERESTED_IN (
    FROM Self TO Topic,
    context          STRING,
    weight           DOUBLE,
    timestamp        TIMESTAMP,
    sensitivity_tier INT64
)
"""

# Self → Place (learned location connections)
REL_CONNECTED_TO = """
CREATE REL TABLE IF NOT EXISTS CONNECTED_TO (
    FROM Self TO Place,
    context          STRING,
    weight           DOUBLE,
    timestamp        TIMESTAMP,
    sensitivity_tier INT64
)
"""

# Person → Topic (contact discusses a topic, from pipeline int_contact_topics)
REL_DISCUSSES = """
CREATE REL TABLE IF NOT EXISTS DISCUSSES (
    FROM Person TO Topic,
    weight           DOUBLE,
    timestamp        TIMESTAMP,
    sensitivity_tier INT64
)
"""

# ---------------------------------------------------------------------------
# Registries for test introspection
# ---------------------------------------------------------------------------

ALL_NODE_TABLES: list[str] = [
    "Person",
    "Event",
    "Place",
    "Emotion",
    "Idea",
    "Topic",
    "Self",
]

ALL_REL_TABLES: list[str] = [
    "KNOWS",
    "PARTICIPATED_IN",
    "FELT",
    "LOCATED_AT",
    "RELATED_TO",
    "MENTIONED_IN",
    "TAGGED_WITH",
    "HAS_RELATIONSHIP",
    "INTERESTED_IN",
    "CONNECTED_TO",
    "DISCUSSES",
]

_NODE_DDL: list[str] = [
    NODE_PERSON,
    NODE_EVENT,
    NODE_PLACE,
    NODE_EMOTION,
    NODE_IDEA,
    NODE_TOPIC,
    NODE_SELF,
]

_REL_DDL: list[str] = [
    REL_KNOWS,
    REL_PARTICIPATED_IN,
    REL_FELT,
    REL_LOCATED_AT,
    REL_RELATED_TO,
    REL_MENTIONED_IN,
    REL_TAGGED_WITH,
    REL_HAS_RELATIONSHIP,
    REL_INTERESTED_IN,
    REL_CONNECTED_TO,
    REL_DISCUSSES,
]


def create_schema(engine: GraphEngine) -> None:
    """Create all node and relationship tables (idempotent).

    Args:
        engine: An open GraphEngine instance to run the DDL against.
    """
    for ddl in _NODE_DDL + _REL_DDL:
        engine.execute(ddl)
