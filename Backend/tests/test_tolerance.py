import pytest

from matching.tolerance import is_within_absolute_tolerance


def test_values_within_tolerance_return_true():
    assert is_within_absolute_tolerance(150, 151, 1) is True


def test_values_outside_tolerance_return_false():
    assert is_within_absolute_tolerance(150, 152, 1) is False


def test_exact_match_returns_true():
    assert is_within_absolute_tolerance(150, 150, 0) is True


def test_boundary_value_returns_true():
    assert is_within_absolute_tolerance(150, 151, 1) is True


def test_negative_tolerance_raises_error():
    with pytest.raises(ValueError):
        is_within_absolute_tolerance(150, 151, -1)