from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MatchAssessment:
    """
    Explainable technical evidence for a material comparison.

    This object describes the evidence only.
    It does not assign Tier 1, Tier 2, or Tier 3.
    """

    matched_attributes: tuple[str, ...]
    conflicting_attributes: tuple[str, ...]
    missing_attributes: tuple[str, ...]
    unknown_attributes: tuple[str, ...]
    critical_conflicts: tuple[str, ...]

from matching.attribute_matcher import (
    AttributeMatchStatus,
    find_critical_conflicts,
    summarize_attribute_matches,
)


def build_match_assessment(
    comparison_results: dict[str, AttributeMatchStatus],
) -> MatchAssessment:
    """
    Build an explainable assessment from attribute comparison results.

    This function interprets comparison evidence but does not
    assign Tier 1, Tier 2, or Tier 3.
    """

    summary = summarize_attribute_matches(comparison_results)

    critical_conflicts = find_critical_conflicts(comparison_results)

    return MatchAssessment(
        matched_attributes=tuple(summary["matched_attributes"]),
        conflicting_attributes=tuple(summary["conflicting_attributes"]),
        missing_attributes=tuple(summary["missing_attributes"]),
        unknown_attributes=tuple(summary["unknown_attributes"]),
        critical_conflicts=tuple(critical_conflicts),
    )