from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from matching.asme_rules import is_critical_attribute
from matching.attribute_matcher import AttributeMatchStatus


class MatchTier(str, Enum):
    TIER_1 = "TIER_1"
    TIER_2 = "TIER_2"
    TIER_3 = "TIER_3"


@dataclass(frozen=True)
class TierDecision:
    """
    Final action classification for a material comparison.

    The decision is based on the technical evidence produced by
    MatchAssessment.
    """

    tier: MatchTier
    reason: str


def find_missing_critical_attributes(
    comparison_results: dict[str, AttributeMatchStatus],
) -> list[str]:
    """
    Return attributes that are both missing and engineering-critical.
    """
    return [
        attribute
        for attribute, status in comparison_results.items()
        if (
            status == AttributeMatchStatus.MISSING
            and is_critical_attribute(attribute)
        )
    ]


def find_unknown_critical_attributes(
    comparison_results: dict[str, AttributeMatchStatus],
) -> list[str]:
    """
    Return attributes that are both unknown and engineering-critical.
    """
    return [
        attribute
        for attribute, status in comparison_results.items()
        if (
            status == AttributeMatchStatus.UNKNOWN
            and is_critical_attribute(attribute)
        )
    ]


def decide_tier(
    comparison_results: dict[str, AttributeMatchStatus],
) -> TierDecision:
    """
    Assign a match tier from deterministic technical evidence.

    Decision order:
    1. Critical conflict -> TIER_3
    2. Missing critical information -> TIER_2
    3. Unknown critical information -> TIER_2
    4. Otherwise -> TIER_1
    """
    critical_conflicts = [
        attribute
        for attribute, status in comparison_results.items()
        if (
            status == AttributeMatchStatus.CONFLICT
            and is_critical_attribute(attribute)
        )
    ]

    if critical_conflicts:
        return TierDecision(
            tier=MatchTier.TIER_3,
            reason=(
                "Critical technical conflict: "
                + ", ".join(critical_conflicts)
            ),
        )

    missing_critical = find_missing_critical_attributes(
        comparison_results
    )

    if missing_critical:
        return TierDecision(
            tier=MatchTier.TIER_2,
            reason=(
                "Critical technical information missing: "
                + ", ".join(missing_critical)
            ),
        )

    unknown_critical = find_unknown_critical_attributes(
        comparison_results
    )

    if unknown_critical:
        return TierDecision(
            tier=MatchTier.TIER_2,
            reason=(
                "Critical technical information could not be determined: "
                + ", ".join(unknown_critical)
            ),
        )

    return TierDecision(
        tier=MatchTier.TIER_1,
        reason="No critical technical conflicts or missing critical information",
    )