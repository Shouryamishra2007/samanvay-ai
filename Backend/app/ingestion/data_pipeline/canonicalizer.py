"""
Deterministic canonicalization of NER-predicted material attributes.

Production input:
    Predicted attributes from NER inference.

This module:
    - uses only predicted attribute values
    - applies explicit deterministic mappings
    - preserves unknown values
    - does not use canonical_id or extracted_* fields
    - does not perform semantic matching or engineering equivalence decisions
"""

from __future__ import annotations

import re
from typing import Mapping


# ---------------------------------------------------------------------------
# Explicit canonical mappings
# ---------------------------------------------------------------------------

ITEM_TYPE_MAP = {
    "GATE VLV": "GATE VALVE",
    "VLV-GT": "GATE VALVE",
    "GATE": "GATE VALVE",
}


METALLURGY_MAP = {
    "SS316": "316 STAINLESS STEEL",
    "AISI 316": "316 STAINLESS STEEL",
    "SS 316": "316 STAINLESS STEEL",
}


PRESSURE_RATING_MAP = {
    "CL150": "CLASS 150",
    "CLASS150": "CLASS 150",
    "CL-150": "CLASS 150",
    "150#": "CLASS 150",

    "CL300": "CLASS 300",
    "CLASS300": "CLASS 300",
    "CL-300": "CLASS 300",
    "300#": "CLASS 300",

    "CL600": "CLASS 600",
    "CLASS600": "CLASS 600",
    "CL-600": "CLASS 600",
    "600#": "CLASS 600",

    "CL900": "CLASS 900",
    "CLASS900": "CLASS 900",
    "CL-900": "CLASS 900",
    "900#": "CLASS 900",

    "CL1500": "CLASS 1500",
    "CLASS1500": "CLASS 1500",
    "CL-1500": "CLASS 1500",
    "1500#": "CLASS 1500",

    "CL2500": "CLASS 2500",
    "CLASS2500": "CLASS 2500",
    "CL-2500": "CLASS 2500",
    "2500#": "CLASS 2500",
}


FACING_END_MAP = {
    "RF": "RAISED FACE",
    "RF FACING": "RAISED FACE",
    "RAISED FACE": "RAISED FACE",
}


STANDARD_MAP = {
    "ASMEB16.5": "ASME B16.5",
    "ASME B16.5": "ASME B16.5",
    "ASME-B16.5": "ASME B16.5",

    "API6D": "API 6D",
    "API 6D": "API 6D",

    "BS1873": "BS 1873",
    "BS 1873": "BS 1873",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _prepare_value(value: object) -> str | None:
    """
    Prepare an attribute value for deterministic lookup.

    This performs only representation-level cleanup.
    It does not infer engineering meaning.
    """

    if not isinstance(value, str):
        return None

    value = value.strip()

    if not value:
        return None

    value = re.sub(r"\s+", " ", value)
    return value.upper()


def _canonicalize(
    value: object,
    mapping: Mapping[str, str],
) -> str | None:
    """
    Apply an explicit deterministic mapping.

    Unknown values are returned unchanged.
    """

    prepared = _prepare_value(value)

    if prepared is None:
        return None

    return mapping.get(prepared, prepared)


# ---------------------------------------------------------------------------
# Attribute-specific canonicalizers
# ---------------------------------------------------------------------------

def canonicalize_item_type(value: object) -> str | None:
    """Canonicalize the item type."""
    return _canonicalize(value, ITEM_TYPE_MAP)


def canonicalize_size(value: object) -> str | None:
    """
    Canonicalize size representation.

    Currently performs safe representation cleanup only.
    Engineering conversions such as MM -> DN are intentionally excluded.
    """
    return _canonicalize(value, {})


def canonicalize_pressure_rating(value: object) -> str | None:
    """Canonicalize pressure-rating notation."""
    return _canonicalize(value, PRESSURE_RATING_MAP)


def canonicalize_metallurgy(value: object) -> str | None:
    """Canonicalize metallurgy notation."""
    return _canonicalize(value, METALLURGY_MAP)


def canonicalize_facing_end(value: object) -> str | None:
    """Canonicalize facing/end notation."""
    return _canonicalize(value, FACING_END_MAP)


def canonicalize_standard(value: object) -> str | None:
    """Canonicalize engineering-standard notation."""
    return _canonicalize(value, STANDARD_MAP)


# ---------------------------------------------------------------------------
# Complete attribute canonicalization
# ---------------------------------------------------------------------------

def canonicalize_attributes(
    attributes: Mapping[str, object],
) -> dict[str, str | None]:
    """
    Canonicalize the six NER-predicted material attributes.

    Expected input keys:
        item_type
        size
        pressure_rating
        metallurgy
        facing_end
        standard

    Missing or invalid values become None.

    Extra input keys are ignored intentionally so that this function
    operates only on the supported material-attribute schema.
    """

    return {
        "item_type": canonicalize_item_type(
            attributes.get("item_type")
        ),
        "size": canonicalize_size(
            attributes.get("size")
        ),
        "pressure_rating": canonicalize_pressure_rating(
            attributes.get("pressure_rating")
        ),
        "metallurgy": canonicalize_metallurgy(
            attributes.get("metallurgy")
        ),
        "facing_end": canonicalize_facing_end(
            attributes.get("facing_end")
        ),
        "standard": canonicalize_standard(
            attributes.get("standard")
        ),
    }