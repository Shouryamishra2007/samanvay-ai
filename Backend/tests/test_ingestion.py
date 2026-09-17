"""
backend/tests/ingestion/data_pipeline/test_normalizer_validation.py

Comprehensive validation tests for normalizer.py

Tests verify:
1. Technical distinction preservation
2. Data leakage prevention (no influence from canonical_id or extracted_*)
3. Missing value handling (None, NaN, pd.NA, pd.NaT, empty strings)
4. Idempotence
5. Original data preservation
6. Hyphen handling decisions
"""

import pandas as pd
import pytest
import numpy as np
from typing import Optional

from app.ingestion.data_pipeline.normalizer import (
    normalize_text,
    normalize_material_record,
    normalize_material_frame,
    NORMALIZED_DESCRIPTION_COLUMN,
)
from app.ingestion.data_pipeline.loader import LoadedMaterialData


# ============================================================================
# TEST 1: TECHNICAL DISTINCTION PRESERVATION
# ============================================================================

class TestTechnicalDistinctionPreservation:
    """
    Verify that normalization does NOT collapse important technical 
    distinctions that must be preserved for material classification.
    
    These tests ensure critical material specs remain distinguishable.
    """

    def test_fastener_size_m10_vs_m12(self):
        """M10 vs M12 must produce different normalized outputs."""
        norm_m10 = normalize_text("M10 BOLT")
        norm_m12 = normalize_text("M12 BOLT")
        
        assert norm_m10 == "M10 BOLT"
        assert norm_m12 == "M12 BOLT"
        assert norm_m10 != norm_m12, \
            "M10 and M12 must remain distinguishable; normalization must not collapse them"

    def test_inch_size_2_vs_2_5(self):
        """2 INCH vs 2.5 INCH must produce different outputs."""
        norm_2in = normalize_text("FLANGE 2 INCH")
        norm_2_5in = normalize_text("FLANGE 2.5 INCH")
        
        assert norm_2in == "FLANGE 2 INCH"
        assert norm_2_5in == "FLANGE 2.5 INCH"
        assert norm_2in != norm_2_5in, \
            "2 INCH and 2.5 INCH must remain distinguishable"

    def test_pressure_class_150_vs_300(self):
        """CLASS 150 vs CLASS 300 must produce different outputs."""
        norm_150 = normalize_text("VALVE CLASS 150")
        norm_300 = normalize_text("VALVE CLASS 300")
        
        assert norm_150 == "VALVE CLASS 150"
        assert norm_300 == "VALVE CLASS 300"
        assert norm_150 != norm_300, \
            "CLASS 150 and CLASS 300 must remain distinguishable"

    def test_stainless_steel_grade_ss304_vs_ss316(self):
        """SS304 vs SS316 must produce different outputs."""
        norm_304 = normalize_text("SS304 PIPE")
        norm_316 = normalize_text("SS316 PIPE")
        
        assert norm_304 == "SS304 PIPE"
        assert norm_316 == "SS316 PIPE"
        assert norm_304 != norm_316, \
            "SS304 and SS316 must remain distinguishable"

    def test_flange_facing_rf_vs_rtj(self):
        """RF vs RTJ facing must produce different outputs."""
        norm_rf = normalize_text("RF FLANGE")
        norm_rtj = normalize_text("RTJ FLANGE")
        
        assert norm_rf == "RF FLANGE"
        assert norm_rtj == "RTJ FLANGE"
        assert norm_rf != norm_rtj, \
            "RF and RTJ facing types must remain distinguishable"

    def test_joint_type_bw_vs_rf(self):
        """BW vs RF joint type must produce different outputs."""
        norm_bw = normalize_text("BUTT WELD CONNECTION")
        norm_rf = normalize_text("RAISED FACE CONNECTION")
        
        assert norm_bw == "BUTT WELD CONNECTION"
        assert norm_rf == "RAISED FACE CONNECTION"
        assert norm_bw != norm_rf, \
            "BW and RF must remain distinguishable"

    def test_astm_standard_a105_vs_a182(self):
        """ASTM A105 vs ASTM A182 must produce different outputs."""
        norm_a105 = normalize_text("ASTM A105 FORGING")
        norm_a182 = normalize_text("ASTM A182 FORGING")
        
        assert norm_a105 == "ASTM A105 FORGING"
        assert norm_a182 == "ASTM A182 FORGING"
        assert norm_a105 != norm_a182, \
            "ASTM A105 and ASTM A182 must remain distinguishable"

    def test_api_standard_600_vs_6d(self):
        """API 600 vs API 6D must produce different outputs."""
        norm_600 = normalize_text("API 600 VALVE")
        norm_6d = normalize_text("API 6D VALVE")
        
        assert norm_600 == "API 600 VALVE"
        assert norm_6d == "API 6D VALVE"
        assert norm_600 != norm_6d, \
            "API 600 and API 6D must remain distinguishable"

    def test_material_grade_with_suffix_wcb_vs_f316l(self):
        """ASTM A216 WCB vs ASTM A182 F316L must produce different outputs."""
        norm_wcb = normalize_text("ASTM A216 WCB BODY")
        norm_f316l = normalize_text("ASTM A182 F316L BODY")
        
        assert norm_wcb == "ASTM A216 WCB BODY"
        assert norm_f316l == "ASTM A182 F316L BODY"
        assert norm_wcb != norm_f316l, \
            "ASTM A216 WCB and ASTM A182 F316L must remain distinguishable"

    def test_codes_with_hyphens_and_digits(self):
        """
        Hyphens adjacent to digits (technical codes) must be preserved.
        
        These are NOT converted by the letter-hyphen-letter rule because
        digits do not match [A-Z].
        """
        # ASTM A-105 (letter-hyphen-digit pattern)
        norm_a105 = normalize_text("ASTM A-105 FORGING")
        # The "A-1" part: A matches [A-Z] but 1 doesn't match [A-Z]
        # So the hyphen is PRESERVED
        assert "-" in norm_a105 or norm_a105 == "ASTM A-105 FORGING"

        # F-316L (letter-hyphen-letter-digit pattern)
        norm_f316l = normalize_text("MATERIAL F-316L")
        # The "F-3" part: F matches [A-Z] but 3 doesn't match [A-Z]
        # So the hyphen is PRESERVED
        assert "-" in norm_f316l or "F-316L" in norm_f316l

        # M12-65 (letter-digit-hyphen-digit pattern)
        norm_m12 = normalize_text("BOLT M12-65")
        # The "2-6" part: 2 doesn't match [A-Z], so hyphen is PRESERVED
        assert "-" in norm_m12 or "M12-65" in norm_m12

    def test_letter_hyphen_letter_patterns_not_present_in_cpse_data(self):
        """
        Verify that the letter-hyphen-letter transformation is not applied
        (since real CPSE data doesn't contain such patterns).
        
        This test documents the decision to remove that transformation.
        """
        # These patterns don't appear in real CPSE descriptions
        # (it would be "GATE VALVE" not "GATE-VALVE" in the CSV)
        # But test the behavior anyway:
        
        # If such a pattern somehow appeared, the current implementation
        # does NOT separate it (because we removed that rule)
        result = normalize_text("EXAMPLE-PATTERN")
        # It will be uppercased and whitespace-collapsed, but hyphen preserved
        assert "EXAMPLE-PATTERN" == result or result == "EXAMPLE-PATTERN"


# ============================================================================
# TEST 2: DATA LEAKAGE PREVENTION
# ============================================================================

class TestDataLeakagePrevention:
    """
    Verify that normalized production output depends ONLY on raw_description.
    
    These tests ensure that canonical_id and extracted_* fields cannot
    influence the normalized text, preventing information leakage from
    ground-truth annotations.
    """

    def test_identical_raw_description_produces_identical_normalized_output(self):
        """
        Two records with identical raw_description but completely different
        ground truth must produce identical normalized_description.
        """
        record_a = {
            "raw_description": "GATE VALVE 2 INCH CLASS 150",
            "canonical_id": "CANONICAL_A",
            "extracted_item_type": "GATE_VALVE",
            "extracted_size_nb_mm": 50.8,
            "extracted_pressure_class": 150,
            "extracted_metallurgy": "CAST_STEEL",
            "extracted_facing": "RF",
        }
        
        record_b = {
            "raw_description": "GATE VALVE 2 INCH CLASS 150",
            "canonical_id": "CANONICAL_B_COMPLETELY_DIFFERENT",
            "extracted_item_type": "COMPLETELY_DIFFERENT_TYPE",
            "extracted_size_nb_mm": 25.4,
            "extracted_pressure_class": 300,
            "extracted_metallurgy": "SS316",
            "extracted_facing": "RTJ",
        }
        
        norm_a = normalize_material_record(record_a)
        norm_b = normalize_material_record(record_b)
        
        assert norm_a[NORMALIZED_DESCRIPTION_COLUMN] == \
               norm_b[NORMALIZED_DESCRIPTION_COLUMN], \
            "Identical raw_description must produce identical normalized_description " \
            "regardless of canonical_id or extracted_* fields"

    def test_changing_canonical_id_does_not_change_normalized_description(self):
        """Modifying canonical_id must not affect normalized_description."""
        base_record = {
            "raw_description": "BALL VALVE 3 INCH",
            "canonical_id": "ID_VERSION_1",
        }
        
        modified_record = dict(base_record)
        modified_record["canonical_id"] = "ID_VERSION_2_COMPLETELY_DIFFERENT"
        
        norm_base = normalize_material_record(base_record)
        norm_modified = normalize_material_record(modified_record)
        
        assert norm_base[NORMALIZED_DESCRIPTION_COLUMN] == \
               norm_modified[NORMALIZED_DESCRIPTION_COLUMN], \
            "Changing canonical_id must not change normalized_description"

    def test_changing_extracted_item_type_does_not_change_normalized(self):
        """Modifying extracted_item_type must not affect normalization."""
        record_a = {
            "raw_description": "FLANGE 2 INCH",
            "extracted_item_type": "FLANGE",
        }
        
        record_b = {
            "raw_description": "FLANGE 2 INCH",
            "extracted_item_type": "COMPLETELY_DIFFERENT_TYPE",
        }
        
        assert normalize_material_record(record_a)[NORMALIZED_DESCRIPTION_COLUMN] == \
               normalize_material_record(record_b)[NORMALIZED_DESCRIPTION_COLUMN], \
            "extracted_item_type must not influence normalized_description"

    def test_changing_extracted_size_does_not_change_normalized(self):
        """Modifying extracted_size_nb_mm must not affect normalization."""
        record_a = {
            "raw_description": "PIPE 50.8MM",
            "extracted_size_nb_mm": 50.8,
        }
        
        record_b = {
            "raw_description": "PIPE 50.8MM",
            "extracted_size_nb_mm": 99999.9,
        }
        
        assert normalize_material_record(record_a)[NORMALIZED_DESCRIPTION_COLUMN] == \
               normalize_material_record(record_b)[NORMALIZED_DESCRIPTION_COLUMN], \
            "extracted_size_nb_mm must not influence normalized_description"

    def test_changing_extracted_pressure_class_does_not_change_normalized(self):
        """Modifying extracted_pressure_class must not affect normalization."""
        record_a = {
            "raw_description": "VALVE CLASS 150",
            "extracted_pressure_class": 150,
        }
        
        record_b = {
            "raw_description": "VALVE CLASS 150",
            "extracted_pressure_class": 300,
        }
        
        assert normalize_material_record(record_a)[NORMALIZED_DESCRIPTION_COLUMN] == \
               normalize_material_record(record_b)[NORMALIZED_DESCRIPTION_COLUMN], \
            "extracted_pressure_class must not influence normalized_description"

    def test_changing_extracted_metallurgy_does_not_change_normalized(self):
        """Modifying extracted_metallurgy must not affect normalization."""
        record_a = {
            "raw_description": "BODY SS304",
            "extracted_metallurgy": "SS304",
        }
        
        record_b = {
            "raw_description": "BODY SS304",
            "extracted_metallurgy": "SS316",
        }
        
        assert normalize_material_record(record_a)[NORMALIZED_DESCRIPTION_COLUMN] == \
               normalize_material_record(record_b)[NORMALIZED_DESCRIPTION_COLUMN], \
            "extracted_metallurgy must not influence normalized_description"

    def test_changing_extracted_facing_does_not_change_normalized(self):
        """Modifying extracted_facing must not affect normalization."""
        record_a = {
            "raw_description": "FLANGE RF",
            "extracted_facing": "RF",
        }
        
        record_b = {
            "raw_description": "FLANGE RF",
            "extracted_facing": "RTJ",
        }
        
        assert normalize_material_record(record_a)[NORMALIZED_DESCRIPTION_COLUMN] == \
               normalize_material_record(record_b)[NORMALIZED_DESCRIPTION_COLUMN], \
            "extracted_facing must not influence normalized_description"

    def test_multiple_extracted_fields_changed_does_not_change_normalized(self):
        """Changing multiple extracted_* fields simultaneously must not affect normalization."""
        record_a = {
            "raw_description": "DOUBLE BLOCK BLEED VALVE 2 INCH",
            "extracted_item_type": "DBB_VALVE",
            "extracted_size_nb_mm": 50.8,
            "extracted_pressure_class": 600,
            "extracted_metallurgy": "ASTM_A182_F316L",
            "extracted_facing": "RF",
        }
        
        record_b = {
            "raw_description": "DOUBLE BLOCK BLEED VALVE 2 INCH",
            "extracted_item_type": "OTHER",
            "extracted_size_nb_mm": 25.4,
            "extracted_pressure_class": 150,
            "extracted_metallurgy": "CAST_STEEL",
            "extracted_facing": "RTJ",
        }
        
        assert normalize_material_record(record_a)[NORMALIZED_DESCRIPTION_COLUMN] == \
               normalize_material_record(record_b)[NORMALIZED_DESCRIPTION_COLUMN], \
            "Multiple extracted_* field changes must not affect normalized_description"


# ============================================================================
# TEST 3: MISSING VALUE HANDLING
# ============================================================================

class TestMissingValueHandling:
    """
    Test that missing/invalid inputs are handled safely.
    
    Missing descriptions must return None, not fabricated text.
    """

    def test_none_input_returns_none(self):
        """None input must return None."""
        result = normalize_text(None)
        assert result is None, "None input must return None, not fabricated text"

    def test_float_nan_input_returns_none(self):
        """float('nan') must return None."""
        result = normalize_text(float("nan"))
        assert result is None, "float('nan') must return None"

    def test_numpy_nan_input_returns_none(self):
        """np.nan must return None."""
        result = normalize_text(np.nan)
        assert result is None, "np.nan must return None"

    def test_pandas_na_input_returns_none(self):
        """pd.NA must return None."""
        result = normalize_text(pd.NA)
        assert result is None, "pd.NA must return None"

    def test_pandas_nat_input_returns_none(self):
        """pd.NaT must return None."""
        result = normalize_text(pd.NaT)
        assert result is None, "pd.NaT must return None"

    def test_empty_string_returns_none(self):
        """Empty string must return None."""
        result = normalize_text("")
        assert result is None, "Empty string must return None"

    def test_whitespace_only_returns_none(self):
        """Whitespace-only string must return None."""
        result = normalize_text("   \t\n  ")
        assert result is None, "Whitespace-only string must return None"

    def test_non_string_non_missing_returns_none(self):
        """Non-string, non-missing inputs must return None."""
        assert normalize_text(12345) is None
        assert normalize_text([1, 2, 3]) is None
        assert normalize_text({"key": "value"}) is None

    def test_valid_string_is_normalized(self):
        """Valid text must be normalized, not returned as None."""
        result = normalize_text("valid text")
        assert result == "VALID TEXT"
        assert result is not None


# ============================================================================
# TEST 4: IDEMPOTENCE
# ============================================================================

class TestIdempotence:
    """
    Test idempotence property:
    normalize_text(normalize_text(x)) == normalize_text(x)
    
    Applying normalization twice produces the same result as applying it once.
    """

    @pytest.mark.parametrize("input_text", [
        "GATE VALVE 2 INCH CLASS 150",
        "gate valve 2 inch class 150",
        "Gate Valve 2 Inch Class 150",
        "BALL-CHECK VALVE",
        "SS316 MATERIAL",
        "ASTM A-105 FORGING",
        "API 600 STANDARD",
        "2.5 INCH FLANGE",
        "  WHITESPACE  TEXT  ",
        "CasE MiXeD tExT",
        "FLANGE\t10\nINCH",
        "Unicode–Dash–Variants—Test",
    ])
    def test_idempotence_on_various_inputs(self, input_text):
        """
        Verify normalize_text(normalize_text(x)) == normalize_text(x)
        for representative descriptions.
        """
        first_pass = normalize_text(input_text)
        
        if first_pass is None:
            # If first pass returns None, second pass must also return None
            second_pass = normalize_text(first_pass)
            assert first_pass == second_pass == None, \
                f"Not idempotent for: {input_text}"
        else:
            # If first pass returns a string, second pass must return the same string
            second_pass = normalize_text(first_pass)
            assert first_pass == second_pass, \
                f"Not idempotent for: {input_text}\n" \
                f"First pass:  {first_pass}\n" \
                f"Second pass: {second_pass}"


# ============================================================================
# TEST 5: ORIGINAL DATA PRESERVATION
# ============================================================================

class TestOriginalDataPreservation:
    """
    Verify that raw_description is never modified.
    """

    def test_raw_description_unchanged_in_record(self):
        """raw_description must not be overwritten in normalize_material_record."""
        original = "GATE-VALVE 2 INCH CLASS 150"
        record = {"raw_description": original}
        
        result = normalize_material_record(record)
        
        assert "raw_description" in result
        assert result["raw_description"] == original, \
            "raw_description must not be modified"

    def test_raw_description_unchanged_in_frame(self):
        """raw_description must not be overwritten in normalize_material_frame."""
        df = pd.DataFrame({
            "raw_description": [
                "GATE-VALVE 2 INCH",
                "BALL VALVE 3 INCH",
                "CHECK VALVE 1 INCH",
            ]
        })
        
        loaded = LoadedMaterialData(
            frame=df.copy(),
            inference_columns=("raw_description",),
            ground_truth_columns=(),
            metadata_columns=(),
        )
        
        normalized = normalize_material_frame(loaded)
        
        # raw_description must be unchanged
        pd.testing.assert_series_equal(
            normalized.frame["raw_description"],
            df["raw_description"],
            check_names=True
        )

    def test_normalized_description_added_not_replacing(self):
        """normalized_description must be added, not replace raw_description."""
        df = pd.DataFrame({
            "raw_description": ["TEST VALVE"],
        })
        
        loaded = LoadedMaterialData(
            frame=df.copy(),
            inference_columns=("raw_description",),
            ground_truth_columns=(),
            metadata_columns=(),
        )
        
        normalized = normalize_material_frame(loaded)
        
        # Both columns must exist
        assert "raw_description" in normalized.frame.columns
        assert NORMALIZED_DESCRIPTION_COLUMN in normalized.frame.columns
        
        # They must be different (normalized is uppercase, stripped, etc.)
        assert normalized.frame["raw_description"].iloc[0] == "TEST VALVE"


# ============================================================================
# TEST 6: UNICODE AND SPECIAL CHARACTER HANDLING
# ============================================================================

class TestUnicodeAndSpecialCharacters:
    """
    Test safe handling of unicode and special characters.
    """

    def test_unicode_dash_variants_normalized(self):
        """Unicode dash variants should be converted to ASCII hyphen."""
        # Various unicode dashes
        test_cases = [
            ("2–5 INCH", "2-5 INCH"),  # en dash (U+2013)
            ("2—5 INCH", "2-5 INCH"),  # em dash (U+2014)
            ("2‐5 INCH", "2-5 INCH"),  # hyphen (U+2010)
        ]
        
        for input_str, expected in test_cases:
            result = normalize_text(input_str)
            assert result == expected, \
                f"Unicode dash normalization failed for: {input_str}"

    def test_nfkc_normalization_safe(self):
        """NFKC normalization should not break technical tokens."""
        # Standard ASCII tokens
        result = normalize_text("ASTM A105 SS304 API 600")
        assert result == "ASTM A105 SS304 API 600"
        assert "ASTM" in result
        assert "A105" in result
        assert "SS304" in result
        assert "API" in result
        assert "600" in result

    def test_uppercasing_preserves_technical_tokens(self):
        """Uppercasing should preserve technical distinction."""
        result_lower = normalize_text("ss304 material")
        result_upper = normalize_text("SS304 MATERIAL")
        result_mixed = normalize_text("Ss304 Material")
        
        assert result_lower == result_upper == result_mixed == "SS304 MATERIAL"
        assert result_lower != "SS304" in result_lower  # Technical token preserved


# ============================================================================
# TEST 7: DATAFRAME INTEGRATION
# ============================================================================

class TestDataFrameIntegration:
    """
    Test normalize_material_frame integration with LoadedMaterialData.
    """

    def test_frame_normalization_preserves_all_original_columns(self):
        """All original columns must be preserved after normalization."""
        df = pd.DataFrame({
            "raw_description": ["VALVE 1", "VALVE 2"],
            "canonical_id": ["ID1", "ID2"],
            "extracted_item_type": ["VALVE", "VALVE"],
            "metadata_field": ["M1", "M2"],
        })
        
        loaded = LoadedMaterialData(
            frame=df,
            inference_columns=("raw_description",),
            ground_truth_columns=("canonical_id", "extracted_item_type"),
            metadata_columns=("metadata_field",),
        )
        
        normalized = normalize_material_frame(loaded)
        
        # All original columns must be preserved
        for col in df.columns:
            assert col in normalized.frame.columns, \
                f"Original column {col} should be preserved"

    def test_normalized_description_added_to_frame(self):
        """normalized_description column must be added to frame."""
        df = pd.DataFrame({
            "raw_description": ["GATE-VALVE 2 INCH"],
        })
        
        loaded = LoadedMaterialData(
            frame=df,
            inference_columns=("raw_description",),
            ground_truth_columns=(),
            metadata_columns=(),
        )
        
        normalized = normalize_material_frame(loaded)
        
        assert NORMALIZED_DESCRIPTION_COLUMN in normalized.frame.columns

    def test_inference_frame_contains_raw_and_normalized(self):
        """inference_frame must contain both raw_description and normalized_description."""
        df = pd.DataFrame({
            "raw_description": ["GATE-VALVE 2 INCH"],
        })
        
        loaded = LoadedMaterialData(
            frame=df,
            inference_columns=("raw_description",),
            ground_truth_columns=(),
            metadata_columns=(),
        )
        
        normalized = normalize_material_frame(loaded)
        inference = normalized.inference_frame
        
        assert "raw_description" in inference.columns
        assert NORMALIZED_DESCRIPTION_COLUMN in inference.columns

    def test_ground_truth_frame_unchanged_from_loader(self):
        """ground_truth_frame should match original ground truth columns."""
        df = pd.DataFrame({
            "raw_description": ["VALVE"],
            "canonical_id": ["ID1"],
            "extracted_item_type": ["VALVE"],
        })
        
        original_truth = df[["canonical_id", "extracted_item_type"]].copy()
        
        loaded = LoadedMaterialData(
            frame=df,
            inference_columns=("raw_description",),
            ground_truth_columns=("canonical_id", "extracted_item_type"),
            metadata_columns=(),
        )
        
        normalized = normalize_material_frame(loaded)
        result_truth = normalized.ground_truth_frame
        
        pd.testing.assert_frame_equal(original_truth, result_truth)

    def test_multiple_rows_independently_normalized(self):
        """Each row must be normalized independently."""
        df = pd.DataFrame({
            "raw_description": [
                "GATE-VALVE",
                "BALL VALVE",
                "CHECK VALVE",
            ]
        })
        
        loaded = LoadedMaterialData(
            frame=df,
            inference_columns=("raw_description",),
            ground_truth_columns=(),
            metadata_columns=(),
        )
        
        normalized = normalize_material_frame(loaded)
        
        assert len(normalized.frame) == 3
        for idx, normalized_desc in enumerate(normalized.frame[NORMALIZED_DESCRIPTION_COLUMN]):
            # Each should be uppercase and properly normalized
            assert normalized_desc is not None
            assert normalized_desc.isupper() or normalized_desc is None


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
    