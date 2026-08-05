"""Kuzu graph indexer — populates the knowledge graph from DuckDB raw tables.

Reads from ``raw_contacts``, ``raw_calendar_events``, and ``raw_notes`` and
creates Person, Event, Idea, and Topic nodes plus PARTICIPATED_IN, KNOWS,
and TAGGED_WITH edges.

All writes use ``MERGE`` for idempotency — safe to call repeatedly without
duplicating nodes.  Edges use MATCH + CREATE with a prior delete-all step
on full reindex, because Kuzu has no MERGE support for relationship tables.

Data Sensitivity Tiers
----------------------
- Person nodes: tier 2  (names / relationships)
- Event nodes:  tier 2  (titles / attendees)
- Idea nodes:   tier 1  (notes are generally low)
- Topic nodes:  tier 1  (general concepts)
- Edges inherit the higher tier of their endpoints.

sensitivity_tier: 2
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from src.core.kuzu.engine import GraphEngine
from src.core.sqlite.engine import DatabaseEngine

logger = logging.getLogger(__name__)

# Standard edge property fragment for Cypher CREATE statements.
_EDGE_PROPS = "{weight: $w, timestamp: timestamp($ts), sensitivity_tier: $tier}"


class GraphIndexer:
    """Populate the Kuzu knowledge graph from DuckDB raw tables.

    Follows the same facade pattern as ``src.core.chromadb.indexer.Indexer``:
    accepts a DuckDB engine for reads and a Kuzu engine for writes.

    Args:
        duckdb: An open DatabaseEngine for reading raw tables.
        kuzu: An open GraphEngine for writing nodes and edges.

    sensitivity_tier: 2
    """

    def __init__(self, duckdb: DatabaseEngine, kuzu: GraphEngine) -> None:
        self._duck = duckdb
        self._kuzu = kuzu

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def full_reindex(self) -> dict[str, int]:
        """Clear the entire graph and rebuild from DuckDB.

        Returns:
            Dict mapping entity type to count of items created.

        sensitivity_tier: 2
        """
        self._clear_all()
        return self._index_all()

    def incremental_index(
        self, since: datetime | None = None,
    ) -> dict[str, int]:
        """Index only records created after *since*.

        When *since* is ``None``, indexes all records (additive — uses
        MERGE so existing nodes are updated in place).

        Returns:
            Dict mapping entity type to count of items processed.

        sensitivity_tier: 2
        """
        return self._index_all(since=since)

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def _index_all(self, since: datetime | None = None) -> dict[str, int]:
        """Run all indexers and return aggregate counts.

        Clears all edges before recreating them so that repeated
        incremental calls don't accumulate duplicates (Kuzu has no
        MERGE for relationship tables).

        sensitivity_tier: 2
        """
        for rel in (
            "PARTICIPATED_IN", "TAGGED_WITH", "DISCUSSES",
            "HAS_RELATIONSHIP", "INTERESTED_IN", "CONNECTED_TO",
            "KNOWS", "LOCATED_AT", "FELT", "MENTIONED_IN",
        ):
            self._clear_edges(rel)

        counts: dict[str, int] = {}
        # Nodes
        counts["persons"] = self._index_persons(since)
        counts["events"] = self._index_events(since)
        counts["ideas"] = self._index_ideas(since)
        counts["topics"] = self._index_topics(since)
        counts["pipeline_topics"] = self._index_pipeline_topics(since)
        counts["places"] = self._index_places(since)
        counts["emotions"] = self._index_emotions(since)
        counts["self_node"] = self._index_self_node()
        # Edges
        counts["participated_in"] = self._index_participated_in(since)
        counts["tagged_with"] = self._index_tagged_with(since)
        counts["discusses"] = self._index_discusses(since)
        counts["knows"] = self._index_knows(since)
        counts["located_at"] = self._index_located_at(since)
        counts["felt"] = self._index_felt(since)
        counts["mentioned_in"] = self._index_mentioned_in(since)
        counts["self_relationships"] = (
            self._index_self_relationships()
        )
        counts["learned_relationships"] = (
            self._index_learned_relationships()
        )
        total = sum(counts.values())
        logger.info("Graph indexer complete: %d items (%s)", total, counts)
        return counts

    # ------------------------------------------------------------------
    # Node indexers
    # ------------------------------------------------------------------

    def _index_persons(self, since: datetime | None = None) -> int:
        """Create/update Person nodes from ``raw_contacts``.

        sensitivity_tier: 2
        """
        sql = "SELECT id, name, relationship, sensitivity_tier FROM raw_contacts"
        sql += self._since_clause(since)

        rows = self._query_duck(sql)
        cypher = (
            "MERGE (p:Person {id: $id}) "
            "SET p.name = $name, p.relationship = $rel, "
            "p.sensitivity_tier = $tier"
        )
        for row in rows:
            self._kuzu.execute(cypher, {
                "id": str(row["id"]),
                "name": str(row["name"]),
                "rel": str(row.get("relationship") or "unknown"),
                "tier": int(row.get("sensitivity_tier") or 2),
            })
        return len(rows)

    def _index_events(self, since: datetime | None = None) -> int:
        """Create/update Event nodes from ``raw_calendar_events``.

        sensitivity_tier: 2
        """
        sql = (
            "SELECT id, title, start_time, location, sensitivity_tier "
            "FROM raw_calendar_events"
        )
        sql += self._since_clause(since)

        rows = self._query_duck(sql)
        cypher = (
            "MERGE (e:Event {id: $id}) "
            "SET e.title = $title, e.event_type = $etype, "
            "e.date = timestamp($date), e.sensitivity_tier = $tier"
        )
        for row in rows:
            date_val = row.get("start_time")
            date_str = (
                date_val.isoformat()
                if hasattr(date_val, "isoformat")
                else str(date_val)
            )
            self._kuzu.execute(cypher, {
                "id": str(row["id"]),
                "title": str(row["title"]),
                "etype": "calendar",
                "date": date_str,
                "tier": int(row.get("sensitivity_tier") or 2),
            })
        return len(rows)

    def _index_ideas(self, since: datetime | None = None) -> int:
        """Create/update Idea nodes from ``raw_notes``.

        sensitivity_tier: 1
        """
        sql = (
            "SELECT id, title, SUBSTRING(content, 1, 200) AS description, "
            "sensitivity_tier FROM raw_notes"
        )
        sql += self._since_clause(since)

        rows = self._query_duck(sql)
        cypher = (
            "MERGE (i:Idea {id: $id}) "
            "SET i.title = $title, i.description = $description, "
            "i.domain = $domain, i.sensitivity_tier = $tier"
        )
        for row in rows:
            self._kuzu.execute(cypher, {
                "id": str(row["id"]),
                "title": str(row["title"]),
                "description": str(row.get("description") or ""),
                "domain": "notes",
                "tier": int(row.get("sensitivity_tier") or 1),
            })
        return len(rows)

    def _index_topics(self, since: datetime | None = None) -> int:
        """Create Topic nodes from contact relationship types and note tags.

        Topics are derived, not directly stored.  We extract unique
        relationship types from contacts and tags from notes.

        sensitivity_tier: 1
        """
        topics_created = 0

        # From contact relationship types
        sql = (
            "SELECT DISTINCT relationship FROM raw_contacts "
            "WHERE relationship IS NOT NULL"
        )
        if since:
            sql += f" AND created_at >= '{since.isoformat()}'"
        rel_rows = self._query_duck(sql)

        cypher = (
            "MERGE (t:Topic {id: $id}) "
            "SET t.name = $name, t.category = $cat, t.sensitivity_tier = $tier"
        )
        for row in rel_rows:
            rel = str(row["relationship"])
            topic_id = f"tp-rel-{rel.lower().replace(' ', '-')}"
            self._kuzu.execute(cypher, {
                "id": topic_id,
                "name": rel.title(),
                "cat": "relationship",
                "tier": 1,
            })
            topics_created += 1

        # From note tags (JSON array column)
        sql = "SELECT DISTINCT tags FROM raw_notes WHERE tags IS NOT NULL"
        if since:
            sql += f" AND created_at >= '{since.isoformat()}'"
        tag_rows = self._query_duck(sql)

        seen_tags: set[str] = set()
        for row in tag_rows:
            tags = self._parse_json_array(row["tags"])
            for tag in tags:
                tag_lower = tag.lower().strip()
                if tag_lower and tag_lower not in seen_tags:
                    seen_tags.add(tag_lower)
                    topic_id = f"tp-tag-{tag_lower.replace(' ', '-')}"
                    self._kuzu.execute(cypher, {
                        "id": topic_id,
                        "name": tag.strip(),
                        "cat": "tag",
                        "tier": 1,
                    })
                    topics_created += 1

        return topics_created

    def _index_places(self, since: datetime | None = None) -> int:
        """Create Place nodes from calendar event locations.

        sensitivity_tier: 2
        """
        sql = (
            "SELECT DISTINCT location FROM raw_calendar_events "
            "WHERE location IS NOT NULL AND TRIM(location) != ''"
        )
        if since:
            sql += f" AND created_at >= '{since.isoformat()}'"

        rows = self._query_duck(sql)
        cypher = (
            "MERGE (p:Place {id: $id}) "
            "SET p.name = $name, p.place_type = $ptype, "
            "p.sensitivity_tier = $tier"
        )
        count = 0
        for row in rows:
            loc = str(row["location"]).strip()
            if not loc:
                continue
            slug = loc.lower().replace(" ", "-")[:60]
            place_id = f"pl-{slug}"
            self._kuzu.execute(cypher, {
                "id": place_id,
                "name": loc,
                "ptype": "venue",
                "tier": 2,
            })
            count += 1
        return count

    def _index_emotions(self, since: datetime | None = None) -> int:
        """Create Emotion nodes from ``int_labeled_messages``.

        sensitivity_tier: 3
        """
        tables = self._query_duck(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name = 'int_labeled_messages'",
        )
        if not tables:
            return 0

        sql = (
            "SELECT DISTINCT primary_emotion FROM int_labeled_messages "
            "WHERE primary_emotion NOT IN ('unlabeled', 'trust')"
        )
        rows = self._query_duck(sql)
        cypher = (
            "MERGE (e:Emotion {id: $id}) "
            "SET e.name = $name, e.sensitivity_tier = $tier"
        )
        count = 0
        for row in rows:
            emotion = str(row["primary_emotion"]).strip()
            if not emotion:
                continue
            self._kuzu.execute(cypher, {
                "id": f"em-{emotion.lower()}",
                "name": emotion.title(),
                "tier": 3,
            })
            count += 1
        return count

    # ------------------------------------------------------------------
    # Edge indexers
    # ------------------------------------------------------------------

    def _index_participated_in(self, since: datetime | None = None) -> int:
        """Create PARTICIPATED_IN edges from calendar event attendees.

        Parses the ``attendees`` JSON column in ``raw_calendar_events``
        and creates Person → Event edges.  Also MERGE-creates Person
        nodes for attendees not yet in the graph.

        sensitivity_tier: 2
        """
        sql = (
            "SELECT id, title, start_time, attendees, sensitivity_tier "
            "FROM raw_calendar_events WHERE attendees IS NOT NULL"
        )
        sql += self._since_clause(since, prefix=" AND")

        rows = self._query_duck(sql)
        edge_count = 0

        person_cypher = (
            "MERGE (p:Person {id: $id}) "
            "SET p.name = $name, p.relationship = $rel, "
            "p.sensitivity_tier = $tier"
        )
        edge_cypher = (
            "MATCH (p:Person {id: $from_id}), (e:Event {id: $to_id}) "
            f"CREATE (p)-[:PARTICIPATED_IN {_EDGE_PROPS}]->(e)"
        )

        for row in rows:
            attendees = self._parse_json_array(row["attendees"])
            event_id = str(row["id"])
            tier = int(row.get("sensitivity_tier") or 2)
            ts = row.get("start_time")
            ts_str = (
                ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            )

            for attendee in attendees:
                attendee_name = str(attendee).strip()
                if not attendee_name:
                    continue
                person_id = f"p-{attendee_name.lower().replace(' ', '-')}"

                # Ensure attendee Person node exists
                self._kuzu.execute(person_cypher, {
                    "id": person_id,
                    "name": attendee_name,
                    "rel": "attendee",
                    "tier": tier,
                })

                # Create the edge
                self._kuzu.execute(edge_cypher, {
                    "from_id": person_id,
                    "to_id": event_id,
                    "w": 1.0,
                    "ts": ts_str,
                    "tier": tier,
                })
                edge_count += 1

        return edge_count

    def _index_tagged_with(self, since: datetime | None = None) -> int:
        """Create TAGGED_WITH edges from note tags (Idea → Topic).

        sensitivity_tier: 1
        """
        sql = (
            "SELECT id, tags FROM raw_notes "
            "WHERE tags IS NOT NULL"
        )
        sql += self._since_clause(since, prefix=" AND")

        rows = self._query_duck(sql)
        edge_count = 0

        edge_cypher = (
            "MATCH (i:Idea {id: $from_id}), (t:Topic {id: $to_id}) "
            f"CREATE (i)-[:TAGGED_WITH {_EDGE_PROPS}]->(t)"
        )

        for row in rows:
            tags = self._parse_json_array(row["tags"])
            idea_id = str(row["id"])

            for tag in tags:
                tag_lower = tag.lower().strip()
                if not tag_lower:
                    continue
                topic_id = f"tp-tag-{tag_lower.replace(' ', '-')}"

                self._kuzu.execute(edge_cypher, {
                    "from_id": idea_id,
                    "to_id": topic_id,
                    "w": 1.0,
                    "ts": datetime.now().isoformat(),
                    "tier": 1,
                })
                edge_count += 1

        return edge_count

    # ------------------------------------------------------------------
    # Pipeline-enriched topic indexers
    # ------------------------------------------------------------------

    def _index_pipeline_topics(
        self, since: datetime | None = None,
    ) -> int:
        """Create Topic nodes from the ``int_contact_topics`` pipeline table.

        The pipeline extracts per-contact conversation topics via LLM
        (e.g. "father's cancer treatment", "clinic financial recovery").
        Each active topic becomes a Topic node in the graph.

        sensitivity_tier: 3
        """
        tables = self._query_duck(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name = 'int_contact_topics'",
        )
        if not tables:
            return 0

        sql = (
            "SELECT DISTINCT topic, description, category, "
            "sensitivity_tier FROM int_contact_topics "
            "WHERE status != 'resolved'"
        )
        sql += self._since_clause(since, prefix=" AND")

        rows = self._query_duck(sql)
        cypher = (
            "MERGE (t:Topic {id: $id}) "
            "SET t.name = $name, t.category = $cat, "
            "t.sensitivity_tier = $tier"
        )
        seen: set[str] = set()
        for row in rows:
            topic = str(row["topic"]).strip()
            if not topic:
                continue
            slug = topic.lower().replace(" ", "-")[:60]
            topic_id = f"tp-ct-{slug}"
            if topic_id in seen:
                continue
            seen.add(topic_id)

            cat = row.get("category") or "conversation"
            self._kuzu.execute(cypher, {
                "id": topic_id,
                "name": topic,
                "cat": str(cat),
                "tier": int(row.get("sensitivity_tier") or 3),
            })
        return len(seen)

    def _index_discusses(
        self, since: datetime | None = None,
    ) -> int:
        """Create DISCUSSES edges from Person → Topic via ``int_contact_topics``.

        Links each contact to the topics they discuss.

        sensitivity_tier: 3
        """
        tables = self._query_duck(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name = 'int_contact_topics'",
        )
        if not tables:
            return 0

        sql = (
            "SELECT contact_name, topic, importance, "
            "last_seen, sensitivity_tier "
            "FROM int_contact_topics "
            "WHERE status != 'resolved'"
        )
        sql += self._since_clause(since, prefix=" AND")

        rows = self._query_duck(sql)
        edge_count = 0

        person_cypher = (
            "MERGE (p:Person {id: $id}) "
            "SET p.name = $name, p.sensitivity_tier = $tier"
        )
        edge_cypher = (
            "MATCH (p:Person {id: $from_id}), (t:Topic {id: $to_id}) "
            f"CREATE (p)-[:DISCUSSES {_EDGE_PROPS}]->(t)"
        )

        for row in rows:
            contact = str(row["contact_name"]).strip()
            topic = str(row["topic"]).strip()
            if not contact or not topic:
                continue

            person_id = f"p-{contact.lower().replace(' ', '-')}"
            slug = topic.lower().replace(" ", "-")[:60]
            topic_id = f"tp-ct-{slug}"

            importance = row.get("importance")
            try:
                weight = max(0.1, min(1.0, int(importance) / 10.0))
            except (TypeError, ValueError):
                weight = 0.5

            ts = row.get("last_seen") or datetime.now().isoformat()
            tier = int(row.get("sensitivity_tier") or 3)

            self._kuzu.execute(person_cypher, {
                "id": person_id,
                "name": contact,
                "tier": tier,
            })
            self._kuzu.execute(edge_cypher, {
                "from_id": person_id,
                "to_id": topic_id,
                "w": weight,
                "ts": str(ts),
                "tier": tier,
            })
            edge_count += 1

        return edge_count

    # ------------------------------------------------------------------
    # Derived edge indexers
    # ------------------------------------------------------------------

    def _index_knows(self, since: datetime | None = None) -> int:
        """Create KNOWS edges between people who co-occur in group chats.

        Also creates Self→Person KNOWS for direct contacts.

        sensitivity_tier: 2
        """
        tables = self._query_duck(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name = 'raw_messages'",
        )
        if not tables:
            return 0

        sql = (
            "SELECT chat_name, sender, sender_name "
            "FROM raw_messages "
            "WHERE is_group = 1 "
            "AND sender IS NOT NULL AND sender != '' "
            "AND sender != 'me'"
        )
        sql += self._since_clause(since, prefix=" AND")
        rows = self._query_duck(sql)
        if not rows:
            return 0

        groups: dict[str, set[tuple[str, str]]] = {}
        for row in rows:
            chat = str(row.get("chat_name") or "")
            sender = str(row["sender"]).strip()
            name = str(row.get("sender_name") or sender).strip()
            if not chat or not sender:
                continue
            if chat not in groups:
                groups[chat] = set()
            pid = f"p-{name.lower().replace(' ', '-')}"
            groups[chat].add((pid, name))

        person_cypher = (
            "MERGE (p:Person {id: $id}) "
            "SET p.name = $name, p.sensitivity_tier = $tier"
        )
        edge_cypher = (
            "MATCH (a:Person {id: $from_id}), "
            "(b:Person {id: $to_id}) "
            f"CREATE (a)-[:KNOWS {_EDGE_PROPS}]->(b)"
        )

        seen_pairs: set[tuple[str, str]] = set()
        edge_count = 0
        ts = datetime.now().isoformat()

        for members in groups.values():
            member_list = list(members)
            for i, (pid_a, name_a) in enumerate(member_list):
                self._kuzu.execute(person_cypher, {
                    "id": pid_a, "name": name_a, "tier": 2,
                })
                for pid_b, name_b in member_list[i + 1:]:
                    pair = (min(pid_a, pid_b), max(pid_a, pid_b))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    self._kuzu.execute(person_cypher, {
                        "id": pid_b, "name": name_b, "tier": 2,
                    })
                    self._kuzu.execute(edge_cypher, {
                        "from_id": pid_a, "to_id": pid_b,
                        "w": 1.0, "ts": ts, "tier": 2,
                    })
                    edge_count += 1

        return edge_count

    def _index_located_at(self, since: datetime | None = None) -> int:
        """Create LOCATED_AT edges from Event → Place.

        sensitivity_tier: 2
        """
        sql = (
            "SELECT id, location FROM raw_calendar_events "
            "WHERE location IS NOT NULL AND TRIM(location) != ''"
        )
        sql += self._since_clause(since, prefix=" AND")

        rows = self._query_duck(sql)
        edge_cypher = (
            "MATCH (e:Event {id: $from_id}), (p:Place {id: $to_id}) "
            f"CREATE (e)-[:LOCATED_AT {_EDGE_PROPS}]->(p)"
        )
        edge_count = 0
        ts = datetime.now().isoformat()
        for row in rows:
            loc = str(row["location"]).strip()
            if not loc:
                continue
            slug = loc.lower().replace(" ", "-")[:60]
            self._kuzu.execute(edge_cypher, {
                "from_id": str(row["id"]),
                "to_id": f"pl-{slug}",
                "w": 1.0, "ts": ts, "tier": 2,
            })
            edge_count += 1
        return edge_count

    def _index_felt(self, since: datetime | None = None) -> int:
        """Create FELT edges from Person → Emotion via ``int_labeled_messages``.

        Joins labeled messages back to raw_messages to find the sender,
        then creates edges weighted by emotional intensity.

        sensitivity_tier: 3
        """
        for tbl in ("int_labeled_messages", "raw_messages"):
            check = self._query_duck(
                f"SELECT name FROM sqlite_master "
                f"WHERE type = 'table' AND name = '{tbl}'",
            )
            if not check:
                return 0

        sql = (
            "SELECT m.sender, m.sender_name, "
            "l.primary_emotion, AVG(CAST(l.intensity AS REAL)) AS avg_i, "
            "COUNT(*) AS cnt "
            "FROM int_labeled_messages l "
            "JOIN raw_messages m ON l.message_id = m.id "
            "WHERE l.primary_emotion NOT IN ('unlabeled', 'trust') "
            "AND m.sender IS NOT NULL AND m.sender != '' "
            "AND m.sender != 'me' "
            "GROUP BY m.sender, m.sender_name, l.primary_emotion "
            "HAVING cnt >= 2"
        )
        rows = self._query_duck(sql)

        person_cypher = (
            "MERGE (p:Person {id: $id}) "
            "SET p.name = $name, p.sensitivity_tier = $tier"
        )
        edge_cypher = (
            "MATCH (p:Person {id: $from_id}), "
            "(e:Emotion {id: $to_id}) "
            f"CREATE (p)-[:FELT {_EDGE_PROPS}]->(e)"
        )
        edge_count = 0
        ts = datetime.now().isoformat()

        for row in rows:
            name = str(row.get("sender_name") or row["sender"]).strip()
            if not name:
                continue
            pid = f"p-{name.lower().replace(' ', '-')}"
            emotion = str(row["primary_emotion"]).strip().lower()

            try:
                weight = max(0.1, min(1.0, float(row["avg_i"])))
            except (TypeError, ValueError):
                weight = 0.5

            self._kuzu.execute(person_cypher, {
                "id": pid, "name": name, "tier": 2,
            })
            self._kuzu.execute(edge_cypher, {
                "from_id": pid, "to_id": f"em-{emotion}",
                "w": weight, "ts": ts, "tier": 3,
            })
            edge_count += 1
        return edge_count

    def _index_mentioned_in(self, since: datetime | None = None) -> int:
        """Create MENTIONED_IN edges for Person/Topic names in event titles.

        Scans calendar event titles for Person names and pipeline Topic
        names, creating Person→Event and Topic→Event edges.

        sensitivity_tier: 2
        """
        sql = "SELECT id, title FROM raw_calendar_events"
        sql += self._since_clause(since)
        events = self._query_duck(sql)
        if not events:
            return 0

        persons = self._kuzu.query(
            "MATCH (p:Person) RETURN p.id AS id, p.name AS name",
        )
        topics = self._kuzu.query(
            "MATCH (t:Topic) RETURN t.id AS id, t.name AS name",
        )

        person_entries = [
            (str(p["name"]).lower(), str(p["id"]))
            for p in persons if p.get("name")
        ]
        topic_entries = [
            (str(t["name"]).lower(), str(t["id"]))
            for t in topics if t.get("name")
        ]

        person_cypher = (
            "MATCH (p:Person {id: $from_id}), "
            "(e:Event {id: $to_id}) "
            f"CREATE (p)-[:MENTIONED_IN {_EDGE_PROPS}]->(e)"
        )
        topic_cypher = (
            "MATCH (t:Topic {id: $from_id}), "
            "(e:Event {id: $to_id}) "
            f"CREATE (t)-[:MENTIONED_IN {_EDGE_PROPS}]->(e)"
        )

        edge_count = 0
        ts = datetime.now().isoformat()

        for event in events:
            title = str(event.get("title") or "").lower()
            if not title:
                continue
            event_id = str(event["id"])
            title_words = set(title.split())

            for name_lower, nid in person_entries:
                if self._name_matches(name_lower, title, title_words):
                    self._kuzu.execute(person_cypher, {
                        "from_id": nid, "to_id": event_id,
                        "w": 1.0, "ts": ts, "tier": 2,
                    })
                    edge_count += 1

            for topic_lower, tid in topic_entries:
                if self._name_matches(topic_lower, title, title_words):
                    self._kuzu.execute(topic_cypher, {
                        "from_id": tid, "to_id": event_id,
                        "w": 1.0, "ts": ts, "tier": 2,
                    })
                    edge_count += 1

        return edge_count

    @staticmethod
    def _name_matches(
        name: str, text: str, text_words: set[str],
    ) -> bool:
        """Check if *name* is mentioned in *text*.

        Tries full substring match first, then falls back to matching
        any individual word of *name* that is >= 4 characters.

        sensitivity_tier: N/A
        """
        if len(name) < 3:
            return False
        if name in text:
            return True
        for word in name.split():
            if len(word) >= 4 and word in text_words:
                return True
        return False

    def _index_self_relationships(self) -> int:
        """Create Self → Person/Topic/Place edges from communication data.

        - HAS_RELATIONSHIP: Self → Person for contacts the user messages
        - INTERESTED_IN: Self → Topic for user's active conversation topics
        - CONNECTED_TO: Self → Place for locations the user visits

        sensitivity_tier: 2
        """
        edge_count = 0

        try:
            self._kuzu.execute(
                "MERGE (s:Self {id: 'self'}) "
                "SET s.sensitivity_tier = 2", {},
            )
        except Exception:  # noqa: BLE001
            return 0

        # HAS_RELATIONSHIP: contacts the user actively messages
        tables = self._query_duck(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name = 'int_contact_topics'",
        )
        if tables:
            contacts = self._query_duck(
                "SELECT DISTINCT contact_name "
                "FROM int_contact_topics "
                "WHERE status != 'resolved'",
            )
            person_cypher = (
                "MERGE (p:Person {id: $id}) "
                "SET p.name = $name, p.sensitivity_tier = $tier"
            )
            edge_cypher = (
                "MATCH (s:Self {id: 'self'}), "
                "(p:Person {id: $pid}) "
                "CREATE (s)-[:HAS_RELATIONSHIP {"
                "context: 'active contact', "
                "relationship_type: 'communicates_with', "
                "weight: 1.0, "
                "timestamp: timestamp($ts), "
                "sensitivity_tier: $tier}]->(p)"
            )
            ts = datetime.now().isoformat()
            for row in contacts:
                name = str(row["contact_name"]).strip()
                if not name:
                    continue
                pid = f"p-{name.lower().replace(' ', '-')}"
                self._kuzu.execute(person_cypher, {
                    "id": pid, "name": name, "tier": 2,
                })
                self._kuzu.execute(edge_cypher, {
                    "pid": pid, "ts": ts, "tier": 2,
                })
                edge_count += 1

        # INTERESTED_IN: user's active topics
        if tables:
            topics = self._query_duck(
                "SELECT DISTINCT topic, category "
                "FROM int_contact_topics "
                "WHERE status = 'active'",
            )
            edge_cypher = (
                "MATCH (s:Self {id: 'self'}), "
                "(t:Topic {id: $tid}) "
                "CREATE (s)-[:INTERESTED_IN {"
                "context: $ctx, weight: 1.0, "
                "timestamp: timestamp($ts), "
                "sensitivity_tier: $tier}]->(t)"
            )
            ts = datetime.now().isoformat()
            seen: set[str] = set()
            for row in topics:
                topic = str(row["topic"]).strip()
                if not topic:
                    continue
                slug = topic.lower().replace(" ", "-")[:60]
                tid = f"tp-ct-{slug}"
                if tid in seen:
                    continue
                seen.add(tid)
                cat = str(row.get("category") or "conversation")
                self._kuzu.execute(edge_cypher, {
                    "tid": tid, "ctx": cat,
                    "ts": ts, "tier": 2,
                })
                edge_count += 1

        # CONNECTED_TO: places the user has visited
        places = self._kuzu.query(
            "MATCH (p:Place) RETURN p.id AS id",
        )
        if places:
            edge_cypher = (
                "MATCH (s:Self {id: 'self'}), "
                "(p:Place {id: $pid}) "
                "CREATE (s)-[:CONNECTED_TO {"
                "context: 'visited', weight: 1.0, "
                "timestamp: timestamp($ts), "
                "sensitivity_tier: $tier}]->(p)"
            )
            ts = datetime.now().isoformat()
            for row in places:
                self._kuzu.execute(edge_cypher, {
                    "pid": str(row["id"]),
                    "ts": ts, "tier": 2,
                })
                edge_count += 1

        return edge_count

    # ------------------------------------------------------------------
    # Self node + learned facts indexers
    # ------------------------------------------------------------------

    def _index_self_node(self) -> int:
        """Create or update the singleton Self node from user settings.

        Reads ``user_name`` and ``user_bio`` from settings to populate
        the Self node.  Returns 1 if created/updated, 0 if no data.

        sensitivity_tier: 2
        """
        try:
            from src.models.llm_provider import load_llm_settings

            settings = load_llm_settings()
            name = settings.get("user_name", "")
            bio = settings.get("user_bio", "")
            if not name:
                return 0

            self._kuzu.execute(
                "MERGE (s:Self {id: $id}) "
                "SET s.name = $name, s.bio = $bio, "
                "s.sensitivity_tier = $tier",
                {"id": "self", "name": name, "bio": bio or "", "tier": 2},
            )
            return 1
        except Exception:  # noqa: BLE001
            logger.debug("Failed to index Self node", exc_info=True)
            return 0

    def _index_learned_relationships(self) -> int:
        """Create edges from Self to Person/Topic/Place from learned facts.

        Reads ``_learned_facts`` and creates HAS_RELATIONSHIP,
        INTERESTED_IN, or CONNECTED_TO edges based on fact category.

        sensitivity_tier: 2
        """
        edge_count = 0
        try:
            # Check if _learned_facts table exists
            tables = self._duck.query(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name = '_learned_facts'"
            )
            if not tables:
                return 0

            facts = self._duck.query(
                "SELECT id, category, subject, predicate, content, "
                "  sensitivity_tier "
                "FROM _learned_facts "
                "WHERE dismissed_at IS NULL "
                "AND superseded_by IS NULL "
                "AND confidence >= 0.5"
            )
            if not facts:
                return 0

            # Ensure Self node exists
            self._kuzu.execute(
                "MERGE (s:Self {id: 'self'}) "
                "SET s.sensitivity_tier = 2",
                {},
            )

            for fact in facts:
                subject = str(fact["subject"])
                category = str(fact["category"])
                content = str(fact["content"])
                tier = int(fact.get("sensitivity_tier") or 2)

                if category == "relationship" and subject != "self":
                    # Create Person node + HAS_RELATIONSHIP edge
                    person_id = (
                        f"p-{subject.lower().replace(' ', '-')}"
                    )
                    self._kuzu.execute(
                        "MERGE (p:Person {id: $id}) "
                        "SET p.name = $name, p.sensitivity_tier = $tier",
                        {"id": person_id, "name": subject, "tier": tier},
                    )
                    self._kuzu.execute(
                        "MATCH (s:Self {id: 'self'}), "
                        "(p:Person {id: $pid}) "
                        "CREATE (s)-[:HAS_RELATIONSHIP {"
                        "context: $ctx, relationship_type: $rtype, "
                        "weight: 1.0, "
                        "timestamp: timestamp($ts), "
                        "sensitivity_tier: $tier}]->(p)",
                        {
                            "pid": person_id,
                            "ctx": content,
                            "rtype": str(fact["predicate"]),
                            "ts": datetime.now().isoformat(),
                            "tier": tier,
                        },
                    )
                    edge_count += 1

                elif category in ("preference", "habit", "opinion"):
                    # Create Topic node + INTERESTED_IN edge
                    pred = str(fact["predicate"])
                    topic_id = f"tp-fact-{pred.replace(' ', '-')}"
                    self._kuzu.execute(
                        "MERGE (t:Topic {id: $id}) "
                        "SET t.name = $name, t.category = $cat, "
                        "t.sensitivity_tier = $tier",
                        {
                            "id": topic_id,
                            "name": pred.replace("_", " ").title(),
                            "cat": "learned",
                            "tier": tier,
                        },
                    )
                    self._kuzu.execute(
                        "MATCH (s:Self {id: 'self'}), "
                        "(t:Topic {id: $tid}) "
                        "CREATE (s)-[:INTERESTED_IN {"
                        "context: $ctx, weight: 1.0, "
                        "timestamp: timestamp($ts), "
                        "sensitivity_tier: $tier}]->(t)",
                        {
                            "tid": topic_id,
                            "ctx": content,
                            "ts": datetime.now().isoformat(),
                            "tier": tier,
                        },
                    )
                    edge_count += 1

                elif category == "location":
                    # Create Place node + CONNECTED_TO edge
                    pred = str(fact["predicate"])
                    place_id = f"pl-fact-{pred.replace(' ', '-')}"
                    self._kuzu.execute(
                        "MERGE (p:Place {id: $id}) "
                        "SET p.name = $name, p.place_type = $ptype, "
                        "p.sensitivity_tier = $tier",
                        {
                            "id": place_id,
                            "name": content,
                            "ptype": "learned",
                            "tier": tier,
                        },
                    )
                    self._kuzu.execute(
                        "MATCH (s:Self {id: 'self'}), "
                        "(p:Place {id: $pid}) "
                        "CREATE (s)-[:CONNECTED_TO {"
                        "context: $ctx, weight: 1.0, "
                        "timestamp: timestamp($ts), "
                        "sensitivity_tier: $tier}]->(p)",
                        {
                            "pid": place_id,
                            "ctx": content,
                            "ts": datetime.now().isoformat(),
                            "tier": tier,
                        },
                    )
                    edge_count += 1

        except Exception:  # noqa: BLE001
            logger.debug(
                "Failed to index learned relationships", exc_info=True,
            )

        return edge_count

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _clear_edges(self, rel_type: str) -> None:
        """Delete all edges of a specific relationship type.

        sensitivity_tier: N/A
        """
        try:
            self._kuzu.execute(
                f"MATCH ()-[r:{rel_type}]->() DELETE r",
            )
        except Exception:  # noqa: BLE001
            logger.debug("No %s edges to delete", rel_type)

    def _clear_all(self) -> None:
        """Delete all edges and nodes from the graph.

        Edges must be deleted first because Kuzu enforces referential
        integrity.

        sensitivity_tier: N/A
        """
        try:
            self._kuzu.execute("MATCH ()-[r]->() DELETE r")
        except Exception:  # noqa: BLE001
            logger.debug("No edges to delete (empty graph)")
        try:
            self._kuzu.execute("MATCH (n) DELETE n")
        except Exception:  # noqa: BLE001
            logger.debug("No nodes to delete (empty graph)")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _query_duck(self, sql: str) -> list[dict[str, Any]]:
        """Execute a read query against DuckDB and return rows as dicts.

        sensitivity_tier: 2
        """
        try:
            return self._duck.query(sql)
        except Exception:  # noqa: BLE001
            logger.warning("DuckDB query failed: %s", sql[:120])
            return []

    @staticmethod
    def _since_clause(
        since: datetime | None,
        prefix: str = " WHERE",
    ) -> str:
        """Build an optional ``WHERE created_at >= ...`` clause.

        sensitivity_tier: N/A
        """
        if since is None:
            return ""
        return f"{prefix} created_at >= '{since.isoformat()}'"

    @staticmethod
    def _parse_json_array(value: Any) -> list[str]:
        """Parse a DuckDB JSON column into a list of strings.

        Handles both raw JSON strings and already-parsed Python lists.

        sensitivity_tier: N/A
        """
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v) for v in value]
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [str(v) for v in parsed]
            except (json.JSONDecodeError, TypeError):
                pass
        return []
