"""Two-pass schema discovery for MCP server output.

Analyzes sample records from MCP tools and auto-generates field mappings,
sensitivity tiers, and SQLite table assignments.

Pass 1: Rule-based structural analysis (fast, deterministic, always runs).
Pass 2: LLM enhancement via Ollama (only when Pass 1 confidence is low).

sensitivity_tier: 1 (analyzes schema structure, not user data content)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from src.extensions.models import FieldTemplate, ToolTemplate

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_HOST = "http://localhost:11434"
CONFIDENCE_THRESHOLD = 0.6
JACCARD_MATCH_THRESHOLD = 0.3

# ---------------------------------------------------------------------------
# Keyword sets for field-name-based sensitivity classification
# ---------------------------------------------------------------------------

_TIER_3_FIELD_KEYWORDS: frozenset[str] = frozenset({
    "health", "medication", "diagnosis", "therapy", "trauma",
    "salary", "income", "bank", "ssn", "password", "secret",
    "emotion", "mood", "anxiety", "depression", "fear",
    "heart_rate", "blood", "cholesterol", "insulin", "glucose",
    "calories", "body_fat", "weight", "bmi", "temperature",
    "oxygen", "heart", "medical", "prescription",
})

_TIER_2_FIELD_KEYWORDS: frozenset[str] = frozenset({
    "name", "email", "phone", "address", "birthday", "location",
    "attendee", "participant", "contact", "sender", "recipient",
    "habit", "routine", "schedule", "latitude", "longitude",
    "from", "to", "person", "user",
})

# Regex patterns for detecting sensitive content in sample values
_TIER_3_VALUE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\$[\d,]+\.?\d*"),          # Dollar amounts: $1,250.00
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),   # SSN pattern: 123-45-6789
]

_TIER_2_VALUE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),  # Dates: 3/12/2025
    re.compile(r"\+?\d[\d\-\s]{8,}\d"),            # Phone numbers
]

# ---------------------------------------------------------------------------
# Timestamp / ID / content detection
# ---------------------------------------------------------------------------

_TIMESTAMP_SUFFIXES: frozenset[str] = frozenset({
    "_at", "_time", "_date", "_timestamp",
})

_TIMESTAMP_NAMES: frozenset[str] = frozenset({
    "date", "timestamp", "created", "modified", "updated",
    "sent", "received", "played", "recorded", "start", "end",
})

_ISO_8601_PATTERN: re.Pattern[str] = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}",
)

_UNIX_TS_RANGE = (946684800, 4102444800)  # 2000-01-01 to 2100-01-01

# ---------------------------------------------------------------------------
# Domain keyword sets
# ---------------------------------------------------------------------------

_DOMAIN_KEYWORDS: dict[str, frozenset[str]] = {
    "messages": frozenset({
        "sender", "recipient", "content", "message", "chat",
        "is_from_me", "chat_name", "is_group",
    }),
    "calendar": frozenset({
        "event", "start_time", "end_time", "attendee", "calendar",
        "is_all_day", "duration",
    }),
    "health": frozenset({
        "heart_rate", "steps", "calories", "workout", "sleep",
        "health", "metric_type", "blood_pressure", "bmi",
    }),
    "files": frozenset({
        "filepath", "filename", "filetype", "size", "directory",
        "size_bytes", "content_preview",
    }),
    "contacts": frozenset({
        "contact", "phone", "relationship", "birthday",
    }),
    "notes": frozenset({
        "note", "title", "content", "tag", "notebook", "parent_page",
    }),
    "email": frozenset({
        "subject", "folder", "labels", "body_preview", "is_read",
    }),
    "music": frozenset({
        "track", "artist", "album", "played_at", "track_name",
        "artist_name", "album_name",
    }),
    "browser": frozenset({
        "url", "visit", "domain", "browser", "bookmark",
        "visit_count", "last_visited",
    }),
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldMapping:
    """A discovered mapping from a source field to a SQLite column.

    sensitivity_tier: 1
    """

    source_name: str
    target_column: str
    source_type: str
    target_type: str
    sensitivity_tier: int
    confidence: float
    tier_source: str  # "keyword_match", "value_scan", "llm_classification", "default"
    transform: str | None = None
    is_new_column: bool = False


@dataclass(frozen=True)
class DiscoveredMapping:
    """Complete schema discovery result for a single MCP tool.

    sensitivity_tier: 1
    """

    tool_name: str
    target_table: str
    is_new_table: bool
    domain: str
    confidence: float
    analysis_method: str  # "rules_only" or "rules_plus_llm"
    fields: tuple[FieldMapping, ...]
    dedup_key: tuple[str, ...]
    suggested_schedule: str
    unmapped_fields: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Pure helper functions (Pass 1 building blocks)
# ---------------------------------------------------------------------------


def _infer_source_type(values: list[Any]) -> str:
    """Infer the JSON source type from a list of sample values.

    Uses majority vote on non-None values. Falls back to "string".

    sensitivity_tier: N/A
    """
    non_none = [v for v in values if v is not None]
    if not non_none:
        return "string"

    type_counts: dict[str, int] = {
        "string": 0, "number": 0, "boolean": 0, "array": 0, "object": 0,
    }
    for v in non_none:
        if isinstance(v, bool):
            type_counts["boolean"] += 1
        elif isinstance(v, (int, float)):
            type_counts["number"] += 1
        elif isinstance(v, list):
            type_counts["array"] += 1
        elif isinstance(v, dict):
            type_counts["object"] += 1
        else:
            type_counts["string"] += 1

    best = max(type_counts, key=lambda k: type_counts[k])
    if type_counts[best] == 0:
        return "string"
    return best


def _normalize_column_name(name: str) -> str:
    """Convert a field name to snake_case SQLite column convention.

    Examples: "startTime" -> "start_time", "body-preview" -> "body_preview"

    sensitivity_tier: N/A
    """
    # Insert underscore before uppercase letters (camelCase split)
    result = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    # Replace hyphens and dots with underscores
    result = re.sub(r"[-.]", "_", result)
    # Collapse multiple underscores
    result = re.sub(r"_+", "_", result)
    # Strip leading/trailing underscores and lowercase
    return result.strip("_").lower()


def _looks_like_timestamp(field_name: str, values: list[Any]) -> bool:
    """Check if a string field looks like a timestamp.

    sensitivity_tier: N/A
    """
    lower = field_name.lower()
    name_match = any(lower.endswith(s) for s in _TIMESTAMP_SUFFIXES) or any(
        kw in lower for kw in _TIMESTAMP_NAMES
    )
    if not name_match:
        return False

    str_values = [v for v in values if isinstance(v, str)]
    if not str_values:
        return False

    return any(_ISO_8601_PATTERN.match(v) for v in str_values)


def _looks_like_unix_timestamp(values: list[Any]) -> bool:
    """Check if numeric values look like Unix timestamps.

    sensitivity_tier: N/A
    """
    nums = [v for v in values if isinstance(v, (int, float)) and v is not True]
    if not nums:
        return False
    return all(_UNIX_TS_RANGE[0] <= v <= _UNIX_TS_RANGE[1] for v in nums)


def _infer_target_type(
    source_type: str,
    field_name: str,
    values: list[Any],
) -> tuple[str, str | None]:
    """Map source type + field name to a SQLite type and optional transform.

    Returns (target_type, transform_or_None).

    sensitivity_tier: N/A
    """
    if source_type == "string":
        if _looks_like_timestamp(field_name, values):
            return ("TEXT", "iso_to_timestamp")
        str_values = [v for v in values if isinstance(v, str)]
        if str_values and sum(len(v) for v in str_values) / len(str_values) > 500:
            return ("TEXT", None)
        return ("VARCHAR", None)

    if source_type == "number":
        if _looks_like_unix_timestamp(values):
            return ("TEXT", "unix_to_timestamp")
        nums = [v for v in values if isinstance(v, (int, float)) and v is not True]
        if nums and all(isinstance(v, int) for v in nums):
            if any(abs(v) > 2**31 for v in nums):
                return ("BIGINT", None)
            return ("INTEGER", None)
        return ("DOUBLE", None)

    if source_type == "boolean":
        return ("BOOLEAN", None)

    if source_type == "array":
        return ("JSON", "json_array")

    if source_type == "object":
        return ("JSON", "flatten_object")

    return ("VARCHAR", None)


def _classify_field_sensitivity(
    field_name: str,
    values: list[Any],
) -> tuple[int, str]:
    """Classify a single field's sensitivity tier.

    Returns (tier, source) where source is "keyword_match", "value_scan",
    or "default".

    sensitivity_tier: N/A
    """
    lower = field_name.lower()
    name_tier = 1
    name_source = "default"

    # Check field name against keyword sets
    for keyword in _TIER_3_FIELD_KEYWORDS:
        if keyword in lower:
            name_tier = 3
            name_source = "keyword_match"
            break

    if name_tier < 3:
        for keyword in _TIER_2_FIELD_KEYWORDS:
            if keyword in lower:
                name_tier = 2
                name_source = "keyword_match"
                break

    # Scan sample values for sensitive patterns and keywords
    value_tier = 1
    value_source = "default"
    str_values = [str(v) for v in values if v is not None]
    for text in str_values:
        text_lower = text.lower()
        # Check tier-3 keywords in values
        for keyword in _TIER_3_FIELD_KEYWORDS:
            if keyword in text_lower:
                value_tier = 3
                value_source = "value_scan"
                break
        if value_tier == 3:
            break
        # Check tier-3 regex patterns
        for pattern in _TIER_3_VALUE_PATTERNS:
            if pattern.search(text):
                value_tier = 3
                value_source = "value_scan"
                break
        if value_tier == 3:
            break
        # Check tier-2 keywords in values
        for keyword in _TIER_2_FIELD_KEYWORDS:
            if keyword in text_lower:
                value_tier = 2
                value_source = "value_scan"
                break
        if value_tier >= 2:
            continue
        # Check tier-2 regex patterns
        for pattern in _TIER_2_VALUE_PATTERNS:
            if pattern.search(text):
                value_tier = 2
                value_source = "value_scan"
                break

    # Conservative: take the higher tier
    if value_tier > name_tier:
        return (value_tier, value_source)
    return (name_tier, name_source)


def _compute_jaccard(set_a: set[str], set_b: set[str]) -> float:
    """Compute Jaccard similarity between two string sets.

    sensitivity_tier: N/A
    """
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def _find_best_table_match(
    source_fields: set[str],
    existing_tables: dict[str, list[str]],
) -> tuple[str | None, float]:
    """Find the existing table with highest Jaccard similarity.

    Returns (table_name_or_None, similarity_score).
    Returns (None, 0.0) if best score < JACCARD_MATCH_THRESHOLD.

    sensitivity_tier: N/A
    """
    best_table: str | None = None
    best_score = 0.0

    normalized_source = {_normalize_column_name(f) for f in source_fields}

    for table_name, columns in existing_tables.items():
        col_set = {c.lower() for c in columns}
        score = _compute_jaccard(normalized_source, col_set)
        if score > best_score:
            best_score = score
            best_table = table_name

    if best_score < JACCARD_MATCH_THRESHOLD:
        return (None, best_score)
    return (best_table, best_score)


def _detect_nested_fields(
    sample_records: list[dict[str, Any]],
) -> list[tuple[str, list[str]]]:
    """Detect nested objects suitable for flattening.

    Returns list of (parent_key, [child_keys]) for consistent 1-level nests.

    sensitivity_tier: N/A
    """
    result: list[tuple[str, list[str]]] = []

    # Find all dict-valued fields
    dict_fields: dict[str, list[set[str]]] = {}
    for record in sample_records:
        for key, value in record.items():
            if isinstance(value, dict):
                if key not in dict_fields:
                    dict_fields[key] = []
                dict_fields[key].append(set(value.keys()))

    for parent_key, key_sets in dict_fields.items():
        if not key_sets:
            continue
        # Check if nested keys are consistent (same keys in >50% of records)
        all_keys = set()
        for ks in key_sets:
            all_keys |= ks
        # Check none of the child values are themselves dicts (only flatten 1 level)
        has_deep_nesting = False
        for record in sample_records:
            val = record.get(parent_key)
            if isinstance(val, dict):
                if any(isinstance(v, dict) for v in val.values()):
                    has_deep_nesting = True
                    break
        if has_deep_nesting:
            continue

        # Consistent if the intersection of all key sets is >50% of the union
        common_keys = key_sets[0]
        for ks in key_sets[1:]:
            common_keys = common_keys & ks
        if len(common_keys) >= len(all_keys) * 0.5:
            result.append((parent_key, sorted(all_keys)))

    return result


def _detect_dedup_key(
    field_names: list[str],
    sample_records: list[dict[str, Any]],
) -> tuple[str, ...]:
    """Heuristically detect the dedup key for a set of records.

    sensitivity_tier: N/A
    """
    normalized = {_normalize_column_name(f): f for f in field_names}

    # Look for an "id" field
    id_field: str | None = None
    for norm, orig in normalized.items():
        if norm == "id" or norm.endswith("_id"):
            values = [r.get(orig) for r in sample_records if r.get(orig) is not None]
            if values and len(set(str(v) for v in values)) == len(values):
                id_field = orig
                break

    if id_field is not None:
        # Check for a "source" field to combine with
        for norm, orig in normalized.items():
            if norm == "source":
                return (id_field, orig)
        return (id_field,)

    # Fallback: look for a unique string field + timestamp
    ts_field: str | None = None
    unique_field: str | None = None
    for norm, orig in normalized.items():
        values = [r.get(orig) for r in sample_records if r.get(orig) is not None]
        if not values:
            continue
        if ts_field is None and any(
            norm.endswith(s) for s in _TIMESTAMP_SUFFIXES
        ):
            ts_field = orig
        if unique_field is None and all(isinstance(v, str) for v in values):
            if len(set(values)) == len(values):
                unique_field = orig

    parts: list[str] = []
    if unique_field:
        parts.append(unique_field)
    if ts_field:
        parts.append(ts_field)
    return tuple(parts)


def _classify_domain(
    tool_name: str,
    field_names: list[str],
    tool_description: str,
) -> str:
    """Classify the tool's data domain from field/tool names.

    sensitivity_tier: N/A
    """
    # Combine tool name, description, and field names into a searchable corpus
    lower_fields = {f.lower() for f in field_names}
    lower_tool = tool_name.lower()
    lower_desc = tool_description.lower()

    best_domain = "general"
    best_score = 0

    for domain, keywords in _DOMAIN_KEYWORDS.items():
        score = 0
        for kw in keywords:
            if kw in lower_tool or kw in lower_desc:
                score += 2
            normalized_fields = {_normalize_column_name(f) for f in lower_fields}
            if kw in normalized_fields or kw in lower_fields:
                score += 1
        if score > best_score:
            best_score = score
            best_domain = domain

    return best_domain


def _suggest_schedule(domain: str) -> str:
    """Suggest a sync schedule based on the data domain.

    sensitivity_tier: N/A
    """
    fast_domains = {"messages", "calendar", "email"}
    medium_domains = {"health", "music", "browser", "contacts"}

    if domain in fast_domains:
        return "every_15min"
    if domain in medium_domains:
        return "hourly"
    return "daily"


# ---------------------------------------------------------------------------
# LLM prompt + response parsing live in
# :mod:`src.agents.schema_discovery.agent` (pydantic-ai); the legacy
# string-prompt builders here were retired in Phase F1.5.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# SchemaDiscoveryAgent
# ---------------------------------------------------------------------------


class SchemaDiscoveryAgent:
    """Analyzes MCP tool output and generates field mappings + sensitivity tiers.

    Two-pass analysis:
      Pass 1: Rule-based structural analysis (fast, deterministic)
      Pass 2: LLM-assisted via Ollama (only when Pass 1 confidence < threshold)

    sensitivity_tier: 1
    """

    def __init__(
        self,
        existing_tables: dict[str, list[str]] | None = None,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ) -> None:
        self._existing_tables = existing_tables or {}
        self._model = model
        self._host = host
        self._confidence_threshold = confidence_threshold

    def discover(
        self,
        tool_name: str,
        sample_records: list[dict[str, Any]],
        tool_description: str = "",
    ) -> DiscoveredMapping:
        """Analyze MCP tool output and generate a complete field mapping.

        Args:
            tool_name: Name of the MCP tool that produced the data.
            sample_records: 1+ sample output records from the tool.
            tool_description: Optional human-readable tool description.

        Returns:
            DiscoveredMapping with field-level mappings and sensitivity tiers.

        Raises:
            ValueError: If sample_records is empty.

        sensitivity_tier: 1
        """
        if not sample_records:
            msg = "sample_records must contain at least one record"
            raise ValueError(msg)

        rule_result = self._run_rule_based_pass(
            tool_name, sample_records, tool_description,
        )

        if rule_result.confidence >= self._confidence_threshold:
            return rule_result

        # Pass 2: LLM enhancement
        llm_result = self._call_llm(tool_name, tool_description, sample_records)
        if llm_result is None:
            warnings = list(rule_result.warnings)
            warnings.append("LLM unavailable, using rule-based analysis only")
            return DiscoveredMapping(
                tool_name=rule_result.tool_name,
                target_table=rule_result.target_table,
                is_new_table=rule_result.is_new_table,
                domain=rule_result.domain,
                confidence=rule_result.confidence,
                analysis_method="rules_only",
                fields=rule_result.fields,
                dedup_key=rule_result.dedup_key,
                suggested_schedule=rule_result.suggested_schedule,
                unmapped_fields=rule_result.unmapped_fields,
                warnings=tuple(warnings),
            )

        return self._merge_llm_result(rule_result, llm_result)

    # ------------------------------------------------------------------
    # Pass 1: Rule-based analysis
    # ------------------------------------------------------------------

    def _run_rule_based_pass(
        self,
        tool_name: str,
        sample_records: list[dict[str, Any]],
        tool_description: str,
    ) -> DiscoveredMapping:
        """Execute the full rule-based analysis pass.

        sensitivity_tier: 1
        """
        # 1. Collect all field names (union across all records)
        all_field_names: list[str] = []
        seen: set[str] = set()
        for record in sample_records:
            for key in record:
                if key not in seen:
                    all_field_names.append(key)
                    seen.add(key)

        # 2. Gather sample values per field
        field_values: dict[str, list[Any]] = {
            f: [r.get(f) for r in sample_records] for f in all_field_names
        }

        # 3. Detect and handle nested objects
        nested = _detect_nested_fields(sample_records)
        nested_parents = {parent for parent, _ in nested}

        warnings: list[str] = []
        flattened_fields: list[tuple[str, list[Any]]] = []

        for field_name in all_field_names:
            if field_name in nested_parents:
                # Flatten this nested object
                for parent, children in nested:
                    if parent == field_name:
                        for child in children:
                            flat_name = f"{parent}_{child}"
                            child_values = []
                            for record in sample_records:
                                parent_val = record.get(parent)
                                if isinstance(parent_val, dict):
                                    child_values.append(parent_val.get(child))
                                else:
                                    child_values.append(None)
                            flattened_fields.append((flat_name, child_values))
            else:
                values = field_values[field_name]
                # Check for dict values that aren't flattenable
                source_type = _infer_source_type(values)
                if source_type == "object":
                    warnings.append(
                        f"Field '{field_name}' contains nested objects stored as JSON"
                    )
                flattened_fields.append((field_name, values))

        # 4. Check for arrays of objects
        for field_name, values in flattened_fields:
            non_none = [v for v in values if v is not None]
            if non_none and all(isinstance(v, list) for v in non_none):
                inner = [item for v in non_none for item in v if isinstance(item, dict)]
                if inner:
                    warnings.append(
                        f"Field '{field_name}' contains arrays of objects — "
                        "consider separate table"
                    )

        # 5. Build field mappings
        field_mappings: list[FieldMapping] = []
        normalized_names = [
            _normalize_column_name(name)
            for name, _ in flattened_fields
        ]

        # Find best table match
        source_field_set = set(normalized_names)
        matched_table, match_score = _find_best_table_match(
            source_field_set, self._existing_tables,
        )

        existing_columns: set[str] = set()
        if matched_table:
            existing_columns = {
                c.lower() for c in self._existing_tables.get(matched_table, [])
            }

        for (field_name, values), norm_name in zip(
            flattened_fields, normalized_names, strict=True,
        ):
            source_type = _infer_source_type(values)
            target_type, transform = _infer_target_type(
                source_type, field_name, values,
            )
            tier, tier_source = _classify_field_sensitivity(field_name, values)

            # Determine if column exists in target table
            is_new_column = norm_name not in existing_columns

            # Confidence for this field
            if not is_new_column and matched_table:
                field_confidence = 1.0
            elif matched_table:
                field_confidence = 0.5
            else:
                field_confidence = 0.6

            # Check for long text → suggest ChromaDB indexing
            str_values = [v for v in values if isinstance(v, str)]
            if str_values:
                avg_len = sum(len(v) for v in str_values) / len(str_values)
                if avg_len > 1000:
                    warnings.append(
                        f"Field '{field_name}' has long text (avg {avg_len:.0f} chars)"
                        " — consider ChromaDB indexing"
                    )

            # Check for foreign key pattern
            if norm_name.endswith("_id") and norm_name != "id":
                warnings.append(
                    f"Field '{field_name}' looks like a foreign key"
                    " — consider Kuzu graph edge"
                )

            field_mappings.append(FieldMapping(
                source_name=field_name,
                target_column=norm_name,
                source_type=source_type,
                target_type=target_type,
                sensitivity_tier=tier,
                confidence=field_confidence,
                tier_source=tier_source,
                transform=transform,
                is_new_column=is_new_column,
            ))

        # 6. Classify domain and determine table
        domain = _classify_domain(tool_name, all_field_names, tool_description)

        is_new_table = matched_table is None
        target_table = matched_table or f"raw_{domain}"

        # 7. Detect dedup key
        dedup_field_names = [name for name, _ in flattened_fields]
        dedup_key = _detect_dedup_key(dedup_field_names, sample_records)

        # 8. Suggest schedule
        suggested_schedule = _suggest_schedule(domain)

        # 9. Collect unmapped fields (new columns in an existing table)
        unmapped: list[str] = []
        if matched_table:
            for fm in field_mappings:
                if fm.is_new_column:
                    unmapped.append(fm.source_name)

        # 10. Warn if too many fields
        if len(field_mappings) > 30:
            warnings.append(
                f"Tool has {len(field_mappings)} fields"
                " — consider filtering unused fields"
            )

        # 11. Compute overall confidence
        field_confidences = [fm.confidence for fm in field_mappings]
        avg_field_conf = (
            sum(field_confidences) / len(field_confidences)
            if field_confidences
            else 0.0
        )
        table_weight = match_score if matched_table else 0.3
        dedup_weight = 0.2 if dedup_key else 0.0
        overall_confidence = table_weight * 0.4 + avg_field_conf * 0.4 + dedup_weight

        return DiscoveredMapping(
            tool_name=tool_name,
            target_table=target_table,
            is_new_table=is_new_table,
            domain=domain,
            confidence=overall_confidence,
            analysis_method="rules_only",
            fields=tuple(field_mappings),
            dedup_key=dedup_key,
            suggested_schedule=suggested_schedule,
            unmapped_fields=tuple(unmapped),
            warnings=tuple(warnings),
        )

    # ------------------------------------------------------------------
    # Pass 2: LLM enhancement
    # ------------------------------------------------------------------

    def _call_llm(
        self,
        tool_name: str,
        tool_description: str,  # noqa: ARG002 (kept for API stability)
        sample_records: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Delegate to :class:`SchemaDiscoveryAgent` (pydantic-ai).

        Returns the legacy dict shape so ``_merge_llm_result`` keeps
        working unchanged.

        sensitivity_tier: 1
        """
        from src.agents.schema_discovery.agent import (
            SchemaDiscoveryAgent as _Agent,
        )

        try:
            draft = _Agent().discover(
                tool_name=tool_name,
                sample_records=sample_records,
                known_tables=list(self._existing_tables.keys()),
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "SchemaDiscoveryAgent failed", exc_info=True,
            )
            return None
        if draft is None:
            return None
        return {
            "target_table": draft.target_table,
            "is_new_table": draft.is_new_table,
            "domain": draft.domain,
            "fields": [
                {
                    "source_name": f.source_name,
                    "target_column": f.target_column,
                    "target_type": f.target_type,
                    "sensitivity_tier": f.sensitivity_tier,
                    "transform": f.transform,
                }
                for f in draft.fields
            ],
            "dedup_key": list(draft.dedup_key),
        }

    def _merge_llm_result(
        self,
        rule_based: DiscoveredMapping,
        llm_result: dict[str, Any],
    ) -> DiscoveredMapping:
        """Merge LLM refinements into the rule-based result.

        Principles:
        - Sensitivity tiers can only be raised, never lowered (conservative).
        - Fields omitted by LLM are kept from rule-based with a warning.

        sensitivity_tier: 1
        """
        warnings = list(rule_based.warnings)

        # Target table: accept LLM's if valid
        llm_table = llm_result.get("target_table", rule_based.target_table)
        if not isinstance(llm_table, str) or not llm_table:
            llm_table = rule_based.target_table
        is_new_table = llm_table not in self._existing_tables

        # Domain
        llm_domain = llm_result.get("domain", rule_based.domain)
        valid_domains = set(_DOMAIN_KEYWORDS.keys()) | {"general"}
        if llm_domain not in valid_domains:
            llm_domain = rule_based.domain

        # Fields: merge by source_name
        llm_fields_by_name: dict[str, dict[str, Any]] = {}
        for f in llm_result.get("fields", []):
            if isinstance(f, dict) and "source_name" in f:
                llm_fields_by_name[f["source_name"]] = f

        merged_fields: list[FieldMapping] = []
        for rule_field in rule_based.fields:
            llm_field = llm_fields_by_name.pop(rule_field.source_name, None)
            if llm_field is None:
                merged_fields.append(rule_field)
                continue

            # Merge: use LLM's target_column and target_type if provided
            target_col = llm_field.get("target_column", rule_field.target_column)
            if not isinstance(target_col, str) or not target_col:
                target_col = rule_field.target_column

            target_type = llm_field.get("target_type", rule_field.target_type)
            if not isinstance(target_type, str) or not target_type:
                target_type = rule_field.target_type

            # Sensitivity: max(rule, llm) — never lower
            llm_tier = llm_field.get("sensitivity_tier", 1)
            if not isinstance(llm_tier, int) or llm_tier < 1 or llm_tier > 3:
                llm_tier = 1
            final_tier = max(rule_field.sensitivity_tier, llm_tier)
            tier_source = (
                "llm_classification"
                if llm_tier > rule_field.sensitivity_tier
                else rule_field.tier_source
            )

            transform = llm_field.get("transform", rule_field.transform)

            existing_columns = set()
            if llm_table in self._existing_tables:
                existing_columns = {
                    c.lower() for c in self._existing_tables[llm_table]
                }

            merged_fields.append(FieldMapping(
                source_name=rule_field.source_name,
                target_column=_normalize_column_name(target_col),
                source_type=rule_field.source_type,
                target_type=target_type,
                sensitivity_tier=final_tier,
                confidence=max(rule_field.confidence, 0.7),
                tier_source=tier_source,
                transform=transform,
                is_new_column=(
                    _normalize_column_name(target_col)
                    not in existing_columns
                ),
            ))

        # Fields the LLM added that rule-based didn't have
        for source_name, llm_field in llm_fields_by_name.items():
            warnings.append(f"LLM added field '{source_name}' not found in samples")

        # Dedup key
        llm_dedup = llm_result.get("dedup_key", [])
        if isinstance(llm_dedup, list) and llm_dedup:
            field_names = {fm.source_name for fm in merged_fields}
            if all(k in field_names for k in llm_dedup):
                dedup_key = tuple(llm_dedup)
            else:
                dedup_key = rule_based.dedup_key
        else:
            dedup_key = rule_based.dedup_key

        # Schedule
        llm_schedule = llm_result.get(
            "suggested_schedule", rule_based.suggested_schedule,
        )
        valid_schedules = {"every_15min", "hourly", "daily"}
        if llm_schedule not in valid_schedules:
            llm_schedule = rule_based.suggested_schedule

        # LLM warnings
        llm_warnings = llm_result.get("warnings", [])
        if isinstance(llm_warnings, list):
            warnings.extend(str(w) for w in llm_warnings)

        # Confidence: boosted but capped
        confidence = min(max(rule_based.confidence, 0.7), 0.95)

        return DiscoveredMapping(
            tool_name=rule_based.tool_name,
            target_table=llm_table,
            is_new_table=is_new_table,
            domain=llm_domain,
            confidence=confidence,
            analysis_method="rules_plus_llm",
            fields=tuple(merged_fields),
            dedup_key=dedup_key,
            suggested_schedule=llm_schedule,
            unmapped_fields=rule_based.unmapped_fields,
            warnings=tuple(warnings),
        )


# ---------------------------------------------------------------------------
# Converter to ToolTemplate
# ---------------------------------------------------------------------------


def to_tool_template(mapping: DiscoveredMapping) -> ToolTemplate:
    """Convert a DiscoveredMapping to a ToolTemplate for the extension system.

    sensitivity_tier: 1
    """
    return ToolTemplate(
        tool_name=mapping.tool_name,
        tool_type="data",
        target_table=mapping.target_table,
        fields=tuple(
            FieldTemplate(
                source_name=f.source_name,
                target_column=f.target_column,
                source_type=f.source_type,
                target_type=f.target_type,
                sensitivity_tier=f.sensitivity_tier,
                transform=f.transform,
            )
            for f in mapping.fields
        ),
        dedup_key=mapping.dedup_key,
    )
