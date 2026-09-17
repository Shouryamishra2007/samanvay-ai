from __future__ import annotations


def is_within_absolute_tolerance(
    value_a: float,
    value_b: float,
    tolerance: float,
) -> bool:
    """
    Return whether two numeric values differ by no more than
    the explicitly supplied absolute tolerance.

    This function performs only deterministic numeric comparison.

    It does not:
    - infer engineering equivalence
    - decide material compatibility
    - assign MatchTier
    - use canonical_id
    - use extracted_* fields
    - perform semantic matching
    """
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    return abs(value_a - value_b) <= tolerance