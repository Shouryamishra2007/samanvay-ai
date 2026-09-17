from matching.attribute_matcher import (
    AttributeMatchStatus,
    compare_attribute,
    compare_attributes,
    summarize_attribute_matches,
    find_critical_conflicts,
)
from matching.asme_rules import is_critical_attribute
from matching.match_assessment import (
    MatchAssessment,
    build_match_assessment,
)
from matching.tier_engine import (
    MatchTier,
    TierDecision,
    find_missing_critical_attributes,
    find_unknown_critical_attributes,
    decide_tier,
)

def test_compare_attribute_matching_values():
    assert (
        compare_attribute("150MM", "150MM")
        == AttributeMatchStatus.MATCH
    )


def test_compare_attribute_conflicting_values():
    assert (
        compare_attribute("150MM", "200MM")
        == AttributeMatchStatus.CONFLICT
    )


def test_compare_attribute_missing_first_value():
    assert (
        compare_attribute(None, "150MM")
        == AttributeMatchStatus.MISSING
    )


def test_compare_attribute_missing_second_value():
    assert (
        compare_attribute("150MM", None)
        == AttributeMatchStatus.MISSING
    )


def test_compare_attribute_empty_value():
    assert (
        compare_attribute("", "150MM")
        == AttributeMatchStatus.MISSING
    )


def test_compare_attribute_invalid_value_type():
    assert (
        compare_attribute(150, "150MM")
        == AttributeMatchStatus.UNKNOWN
    )


def test_compare_attribute_does_not_apply_engineering_equivalence():
    assert (
        compare_attribute("PN150", "CLASS 150")
        == AttributeMatchStatus.CONFLICT
    )

def test_compare_attributes_returns_status_for_all_attributes():
    attributes_a = {
        "item_type": "GATE VALVE",
        "size": "150MM",
        "pressure_rating": "CLASS 150",
        "metallurgy": "316 STAINLESS STEEL",
        "facing_end": "RAISED FACE",
        "standard": "ASME B16.5",
    }

    attributes_b = {
        "item_type": "GATE VALVE",
        "size": "150MM",
        "pressure_rating": "CLASS 300",
        "metallurgy": "316 STAINLESS STEEL",
        "facing_end": "RAISED FACE",
        "standard": "ASME B16.5",
    }

    result = compare_attributes(attributes_a, attributes_b)

    assert result == {
        "item_type": AttributeMatchStatus.MATCH,
        "size": AttributeMatchStatus.MATCH,
        "pressure_rating": AttributeMatchStatus.CONFLICT,
        "metallurgy": AttributeMatchStatus.MATCH,
        "facing_end": AttributeMatchStatus.MATCH,
        "standard": AttributeMatchStatus.MATCH,
    }


def test_compare_attributes_handles_missing_attributes():
    attributes_a = {
        "item_type": "GATE VALVE",
        "size": "150MM",
        "pressure_rating": None,
        "metallurgy": "316 STAINLESS STEEL",
        "facing_end": "RAISED FACE",
        "standard": None,
    }

    attributes_b = {
        "item_type": "GATE VALVE",
        "size": "150MM",
        "pressure_rating": "CLASS 150",
        "metallurgy": "316 STAINLESS STEEL",
        "facing_end": "RAISED FACE",
        "standard": "ASME B16.5",
    }

    result = compare_attributes(attributes_a, attributes_b)

    assert result["item_type"] == AttributeMatchStatus.MATCH
    assert result["size"] == AttributeMatchStatus.MATCH
    assert result["pressure_rating"] == AttributeMatchStatus.MISSING
    assert result["metallurgy"] == AttributeMatchStatus.MATCH
    assert result["facing_end"] == AttributeMatchStatus.MATCH
    assert result["standard"] == AttributeMatchStatus.MISSING

def test_summarize_attribute_matches_groups_results():
    results = {
        "item_type": AttributeMatchStatus.MATCH,
        "size": AttributeMatchStatus.MATCH,
        "pressure_rating": AttributeMatchStatus.CONFLICT,
        "metallurgy": AttributeMatchStatus.MATCH,
        "facing_end": AttributeMatchStatus.MATCH,
        "standard": AttributeMatchStatus.MISSING,
    }

    summary = summarize_attribute_matches(results)

    assert summary == {
        "matched_attributes": [
            "item_type",
            "size",
            "metallurgy",
            "facing_end",
        ],
        "conflicting_attributes": [
            "pressure_rating",
        ],
        "missing_attributes": [
            "standard",
        ],
        "unknown_attributes": [],
    }


def test_summarize_attribute_matches_handles_unknown():
    results = {
        "item_type": AttributeMatchStatus.UNKNOWN,
        "size": AttributeMatchStatus.MATCH,
    }

    summary = summarize_attribute_matches(results)

    assert summary["matched_attributes"] == ["size"]
    assert summary["conflicting_attributes"] == []
    assert summary["missing_attributes"] == []
    assert summary["unknown_attributes"] == ["item_type"]\

def test_critical_attributes():
    assert is_critical_attribute("item_type")
    assert is_critical_attribute("size")
    assert is_critical_attribute("pressure_rating")
    assert is_critical_attribute("metallurgy")
    assert is_critical_attribute("facing_end")


def test_standard_is_not_yet_critical():
    assert not is_critical_attribute("standard")


def test_unknown_attribute_is_not_critical():
    assert not is_critical_attribute("unknown_attribute")


def test_find_critical_conflicts():
    results = {
        "item_type": AttributeMatchStatus.MATCH,
        "size": AttributeMatchStatus.CONFLICT,
        "pressure_rating": AttributeMatchStatus.CONFLICT,
        "metallurgy": AttributeMatchStatus.MATCH,
        "facing_end": AttributeMatchStatus.MATCH,
        "standard": AttributeMatchStatus.CONFLICT,
    }

    assert find_critical_conflicts(results) == [
        "size",
        "pressure_rating",
    ]


def test_find_critical_conflicts_returns_empty_when_no_critical_conflict():
    results = {
        "item_type": AttributeMatchStatus.MATCH,
        "size": AttributeMatchStatus.MATCH,
        "pressure_rating": AttributeMatchStatus.MATCH,
        "metallurgy": AttributeMatchStatus.MATCH,
        "facing_end": AttributeMatchStatus.MATCH,
        "standard": AttributeMatchStatus.CONFLICT,
    }

    assert find_critical_conflicts(results) == []

def test_match_assessment_stores_explainable_evidence():
    assessment = MatchAssessment(
        matched_attributes=("item_type", "size", "metallurgy"),
        conflicting_attributes=("pressure_rating",),
        missing_attributes=("standard",),
        unknown_attributes=(),
        critical_conflicts=("pressure_rating",),
    )

    assert assessment.matched_attributes == (
        "item_type",
        "size",
        "metallurgy",
    )
    assert assessment.conflicting_attributes == ("pressure_rating",)
    assert assessment.missing_attributes == ("standard",)
    assert assessment.unknown_attributes == ()
    assert assessment.critical_conflicts == ("pressure_rating",)

def test_build_match_assessment():
    comparison_results = {
        "item_type": AttributeMatchStatus.MATCH,
        "size": AttributeMatchStatus.MATCH,
        "pressure_rating": AttributeMatchStatus.CONFLICT,
        "metallurgy": AttributeMatchStatus.MATCH,
        "facing_end": AttributeMatchStatus.MATCH,
        "standard": AttributeMatchStatus.MISSING,
    }

    assessment = build_match_assessment(comparison_results)

    assert assessment == MatchAssessment(
        matched_attributes=(
            "item_type",
            "size",
            "metallurgy",
            "facing_end",
        ),
        conflicting_attributes=("pressure_rating",),
        missing_attributes=("standard",),
        unknown_attributes=(),
        critical_conflicts=("pressure_rating",),
    )


def test_build_match_assessment_with_no_conflicts():
    comparison_results = {
        "item_type": AttributeMatchStatus.MATCH,
        "size": AttributeMatchStatus.MATCH,
        "pressure_rating": AttributeMatchStatus.MATCH,
        "metallurgy": AttributeMatchStatus.MATCH,
        "facing_end": AttributeMatchStatus.MATCH,
        "standard": AttributeMatchStatus.MATCH,
    }

    assessment = build_match_assessment(comparison_results)

    assert assessment.critical_conflicts == ()
    assert assessment.conflicting_attributes == ()
    assert assessment.missing_attributes == ()
    assert assessment.unknown_attributes == ()

def test_tier_decision_stores_tier_and_reason():
    decision = TierDecision(
        tier=MatchTier.TIER_3,
        reason="Critical technical conflict",
    )

    assert decision.tier == MatchTier.TIER_3
    assert decision.reason == "Critical technical conflict"

from matching.tier_engine import (
    MatchTier,
    TierDecision,
    find_missing_critical_attributes,
)

def test_find_missing_critical_attributes():
    results = {
        "item_type": AttributeMatchStatus.MATCH,
        "size": AttributeMatchStatus.MISSING,
        "pressure_rating": AttributeMatchStatus.MISSING,
        "metallurgy": AttributeMatchStatus.MATCH,
        "facing_end": AttributeMatchStatus.MATCH,
        "standard": AttributeMatchStatus.MISSING,
    }

    assert find_missing_critical_attributes(results) == [
        "size",
        "pressure_rating",
    ]


def test_find_missing_critical_attributes_ignores_noncritical_missing():
    results = {
        "item_type": AttributeMatchStatus.MATCH,
        "size": AttributeMatchStatus.MATCH,
        "pressure_rating": AttributeMatchStatus.MATCH,
        "metallurgy": AttributeMatchStatus.MATCH,
        "facing_end": AttributeMatchStatus.MATCH,
        "standard": AttributeMatchStatus.MISSING,
    }

    assert find_missing_critical_attributes(results) == []

def test_decide_tier_returns_tier_3_for_critical_conflict():
    results = {
        "item_type": AttributeMatchStatus.MATCH,
        "size": AttributeMatchStatus.CONFLICT,
        "pressure_rating": AttributeMatchStatus.MATCH,
        "metallurgy": AttributeMatchStatus.MATCH,
        "facing_end": AttributeMatchStatus.MATCH,
        "standard": AttributeMatchStatus.MATCH,
    }

    decision = decide_tier(results)

    assert decision.tier == MatchTier.TIER_3
    assert "size" in decision.reason


def test_decide_tier_returns_tier_2_for_missing_critical_attribute():
    results = {
        "item_type": AttributeMatchStatus.MATCH,
        "size": AttributeMatchStatus.MATCH,
        "pressure_rating": AttributeMatchStatus.MISSING,
        "metallurgy": AttributeMatchStatus.MATCH,
        "facing_end": AttributeMatchStatus.MATCH,
        "standard": AttributeMatchStatus.MATCH,
    }

    decision = decide_tier(results)

    assert decision.tier == MatchTier.TIER_2
    assert "pressure_rating" in decision.reason


def test_decide_tier_returns_tier_1_when_critical_attributes_agree():
    results = {
        "item_type": AttributeMatchStatus.MATCH,
        "size": AttributeMatchStatus.MATCH,
        "pressure_rating": AttributeMatchStatus.MATCH,
        "metallurgy": AttributeMatchStatus.MATCH,
        "facing_end": AttributeMatchStatus.MATCH,
        "standard": AttributeMatchStatus.MATCH,
    }

    decision = decide_tier(results)

    assert decision.tier == MatchTier.TIER_1

def test_find_unknown_critical_attributes():
    results = {
        "item_type": AttributeMatchStatus.MATCH,
        "size": AttributeMatchStatus.UNKNOWN,
        "pressure_rating": AttributeMatchStatus.MATCH,
        "metallurgy": AttributeMatchStatus.UNKNOWN,
        "facing_end": AttributeMatchStatus.MATCH,
        "standard": AttributeMatchStatus.UNKNOWN,
    }

    assert find_unknown_critical_attributes(results) == [
        "size",
        "metallurgy",
    ]


def test_find_unknown_critical_attributes_ignores_noncritical_unknown():
    results = {
        "item_type": AttributeMatchStatus.MATCH,
        "size": AttributeMatchStatus.MATCH,
        "pressure_rating": AttributeMatchStatus.MATCH,
        "metallurgy": AttributeMatchStatus.MATCH,
        "facing_end": AttributeMatchStatus.MATCH,
        "standard": AttributeMatchStatus.UNKNOWN,
    }

    assert find_unknown_critical_attributes(results) == []

def test_decide_tier_returns_tier_2_for_unknown_critical_attribute():
    results = {
        "item_type": AttributeMatchStatus.MATCH,
        "size": AttributeMatchStatus.UNKNOWN,
        "pressure_rating": AttributeMatchStatus.MATCH,
        "metallurgy": AttributeMatchStatus.MATCH,
        "facing_end": AttributeMatchStatus.MATCH,
        "standard": AttributeMatchStatus.MATCH,
    }

    decision = decide_tier(results)

    assert decision.tier == MatchTier.TIER_2
    assert "size" in decision.reason


def test_decide_tier_ignores_noncritical_unknown_attribute():
    results = {
        "item_type": AttributeMatchStatus.MATCH,
        "size": AttributeMatchStatus.MATCH,
        "pressure_rating": AttributeMatchStatus.MATCH,
        "metallurgy": AttributeMatchStatus.MATCH,
        "facing_end": AttributeMatchStatus.MATCH,
        "standard": AttributeMatchStatus.UNKNOWN,
    }

    decision = decide_tier(results)

    assert decision.tier == MatchTier.TIER_1