"""
Tests for deterministic material-attribute canonicalization.
"""

from app.ingestion.data_pipeline.canonicalizer import (
    canonicalize_attributes,
    canonicalize_facing_end,
    canonicalize_item_type,
    canonicalize_metallurgy,
    canonicalize_pressure_rating,
    canonicalize_size,
    canonicalize_standard,
)


class TestItemTypeCanonicalization:

    def test_gate_valve_variants(self):
        assert canonicalize_item_type("GATE VLV") == "GATE VALVE"
        assert canonicalize_item_type("VLV-GT") == "GATE VALVE"
        assert canonicalize_item_type("GATE") == "GATE VALVE"

    def test_unknown_item_type_is_preserved(self):
        assert canonicalize_item_type("BUTTERFLY VALVE") == "BUTTERFLY VALVE"


class TestSizeCanonicalization:

    def test_size_is_preserved(self):
        assert canonicalize_size("150MM") == "150MM"
        assert canonicalize_size("M10") == "M10"

    def test_size_does_not_make_engineering_conversion(self):
        assert canonicalize_size("150MM") != "DN150"


class TestPressureRatingCanonicalization:

    def test_class_variants(self):
        assert canonicalize_pressure_rating("CL150") == "CLASS 150"
        assert canonicalize_pressure_rating("CLASS150") == "CLASS 150"
        assert canonicalize_pressure_rating("CL-150") == "CLASS 150"
        assert canonicalize_pressure_rating("150#") == "CLASS 150"

    def test_multiple_pressure_classes(self):
        assert canonicalize_pressure_rating("CL300") == "CLASS 300"
        assert canonicalize_pressure_rating("CL600") == "CLASS 600"
        assert canonicalize_pressure_rating("CL900") == "CLASS 900"

    def test_pressure_value_is_not_converted_to_another_system(self):
        assert canonicalize_pressure_rating("PN150") == "PN150"


class TestMetallurgyCanonicalization:

    def test_stainless_steel_variants(self):
        assert canonicalize_metallurgy("SS316") == "316 STAINLESS STEEL"
        assert canonicalize_metallurgy("SS 316") == "316 STAINLESS STEEL"
        assert canonicalize_metallurgy("AISI 316") == "316 STAINLESS STEEL"

    def test_unknown_metallurgy_is_preserved(self):
        assert canonicalize_metallurgy("A105") == "A105"


class TestFacingEndCanonicalization:

    def test_raised_face_variants(self):
        assert canonicalize_facing_end("RF") == "RAISED FACE"
        assert canonicalize_facing_end("RF FACING") == "RAISED FACE"
        assert canonicalize_facing_end("RAISED FACE") == "RAISED FACE"

    def test_unknown_facing_end_is_preserved(self):
        assert canonicalize_facing_end("BW ENDS") == "BW ENDS"


class TestStandardCanonicalization:

    def test_asme_standard_variants(self):
        assert canonicalize_standard("ASMEB16.5") == "ASME B16.5"
        assert canonicalize_standard("ASME B16.5") == "ASME B16.5"
        assert canonicalize_standard("ASME-B16.5") == "ASME B16.5"

    def test_api_standard_variants(self):
        assert canonicalize_standard("API6D") == "API 6D"
        assert canonicalize_standard("API 6D") == "API 6D"

    def test_unknown_standard_is_preserved(self):
        assert canonicalize_standard("ISO 9001") == "ISO 9001"


class TestInvalidValues:

    def test_none_returns_none(self):
        assert canonicalize_item_type(None) is None
        assert canonicalize_size(None) is None
        assert canonicalize_pressure_rating(None) is None
        assert canonicalize_metallurgy(None) is None
        assert canonicalize_facing_end(None) is None
        assert canonicalize_standard(None) is None

    def test_empty_string_returns_none(self):
        assert canonicalize_item_type("") is None
        assert canonicalize_size("   ") is None

    def test_non_string_returns_none(self):
        assert canonicalize_item_type(123) is None
        assert canonicalize_size(150) is None


class TestRepresentationCleanup:

    def test_whitespace_is_normalized(self):
        assert canonicalize_item_type("  gate   vlv  ") == "GATE VALVE"
        assert canonicalize_metallurgy("  ss316  ") == "316 STAINLESS STEEL"

    def test_lowercase_is_normalized(self):
        assert canonicalize_item_type("gate vlv") == "GATE VALVE"
        assert canonicalize_standard("asme b16.5") == "ASME B16.5"


class TestCompleteAttributeCanonicalization:

    def test_complete_material(self):
        attributes = {
            "item_type": "GATE VLV",
            "size": "150MM",
            "pressure_rating": "CL150",
            "metallurgy": "SS316",
            "facing_end": "RF FACING",
            "standard": "ASMEB16.5",
        }

        result = canonicalize_attributes(attributes)

        assert result == {
            "item_type": "GATE VALVE",
            "size": "150MM",
            "pressure_rating": "CLASS 150",
            "metallurgy": "316 STAINLESS STEEL",
            "facing_end": "RAISED FACE",
            "standard": "ASME B16.5",
        }

    def test_missing_attributes_remain_none(self):
        attributes = {
            "item_type": "GATE VLV",
            "size": "150MM",
            "pressure_rating": None,
            "metallurgy": "SS316",
            "facing_end": "RF",
            "standard": None,
        }

        result = canonicalize_attributes(attributes)

        assert result == {
            "item_type": "GATE VALVE",
            "size": "150MM",
            "pressure_rating": None,
            "metallurgy": "316 STAINLESS STEEL",
            "facing_end": "RAISED FACE",
            "standard": None,
        }

    def test_extra_fields_are_ignored(self):
        attributes = {
            "item_type": "GATE VLV",
            "size": "150MM",
            "pressure_rating": "CL150",
            "metallurgy": "SS316",
            "facing_end": "RF",
            "standard": "API6D",
            "canonical_id": "SHOULD_NOT_BE_USED",
            "extracted_item_type": "SHOULD_NOT_BE_USED",
        }

        result = canonicalize_attributes(attributes)

        assert result == {
            "item_type": "GATE VALVE",
            "size": "150MM",
            "pressure_rating": "CLASS 150",
            "metallurgy": "316 STAINLESS STEEL",
            "facing_end": "RAISED FACE",
            "standard": "API 6D",
        }