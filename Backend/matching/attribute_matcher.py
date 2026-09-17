from __future__ import annotations
from matching.asme_rules import is_critical_attribute
from enum import Enum
from matching.asme_rules import is_critical_attribute

class AttributeMatchStatus(str, Enum):
    MATCH = "MATCH"
    CONFLICT = "CONFLICT"
    MISSING = "MISSING"
    UNKNOWN = "UNKNOWN"


def compare_attribute(
    value_a: object,
    value_b: object,
) -> AttributeMatchStatus:
    """
    Compare two already-canonicalized attribute values.

    This function performs deterministic comparison only.
    It does not:
    - use raw descriptions
    - use canonical_id
    - use extracted_* fields
    - perform fuzzy matching
    - perform semantic matching
    - make Tier 1/2/3 decisions
    """

    if value_a is None or value_b is None:
        return AttributeMatchStatus.MISSING

    if not isinstance(value_a, str) or not isinstance(value_b, str):
        return AttributeMatchStatus.UNKNOWN

    value_a = value_a.strip()
    value_b = value_b.strip()

    if not value_a or not value_b:
        return AttributeMatchStatus.MISSING

    if value_a == value_b:
        return AttributeMatchStatus.MATCH

    return AttributeMatchStatus.CONFLICT

ATTRIBUTE_NAMES = (
    "item_type",
    "size",
    "pressure_rating",
    "metallurgy",
    "facing_end",
    "standard",
)


def compare_attributes(
    attributes_a: dict[str, object],
    attributes_b: dict[str, object],
) -> dict[str, AttributeMatchStatus]:
    """
    Compare all supported material attributes.

    Only deterministic attribute comparison is performed.
    """

    return {
        attribute: compare_attribute(
            attributes_a.get(attribute),
            attributes_b.get(attribute),
        )
        for attribute in ATTRIBUTE_NAMES
    }

def summarize_attribute_matches(
    results: dict[str, AttributeMatchStatus],
) -> dict[str, list[str]]:
    """
    Group attribute comparison results into explainable evidence buckets.
    """

    return {
        "matched_attributes": [
            attribute
            for attribute, status in results.items()
            if status == AttributeMatchStatus.MATCH
        ],
        "conflicting_attributes": [
            attribute
            for attribute, status in results.items()
            if status == AttributeMatchStatus.CONFLICT
        ],
        "missing_attributes": [
            attribute
            for attribute, status in results.items()
            if status == AttributeMatchStatus.MISSING
        ],
        "unknown_attributes": [
            attribute
            for attribute, status in results.items()
            if status == AttributeMatchStatus.UNKNOWN
        ],
    }

def find_critical_conflicts(
    results: dict[str, AttributeMatchStatus],
) -> list[str]:
    """
    Return attributes that are both conflicting and engineering-critical.
    """

    return [
        attribute
        for attribute, status in results.items()
        if (
            status == AttributeMatchStatus.CONFLICT
            and is_critical_attribute(attribute)
        )
    ]