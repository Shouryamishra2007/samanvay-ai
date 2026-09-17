from __future__ import annotations


CRITICAL_ATTRIBUTES = {
    "item_type",
    "size",
    "pressure_rating",
    "metallurgy",
    "facing_end",
}

def is_critical_attribute(attribute: str) -> bool:
    """Return whether an attribute is currently treated as engineering-critical."""
    return attribute in CRITICAL_ATTRIBUTES