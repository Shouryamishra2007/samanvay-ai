
"""
test_ner_dataset.py

Unit tests for backend/app/ner_dataset.py's char-span -> BIO alignment.

Tokenizer note: see _local_tokenizer.py -- this sandbox cannot reach
huggingface.co, so tests run against a locally-trained BERT-style WordPiece
tokenizer (same normalizer/pre-tokenizer/post-processor family as
distilbert-base-cased) rather than the real distilbert-base-cased vocab.
ner_dataset.py itself is tokenizer-agnostic; swapping in the real tokenizer
requires no code change.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add backend/ to the Python path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Add tests/ml/ to the Python path so _local_tokenizer can be imported
# without requiring tests/ or tests/ml/ to be Python packages.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.ner_dataset import (  # noqa: E402
    IGNORE_LABEL_ID,
    LABEL_TO_ID,
    NerAnnotationError,
    NerAlignmentError,
    CharSpan,
    NerRecord,
    align_record,
    iter_ner_records,
    load_ner_records,
    prepare_ner_dataset,
)

from _local_tokenizer import build_tokenizer_from_ner_train  # noqa: E402


NER_TRAIN_PATH = (
    Path(__file__).resolve().parents[2]
    / "app"
    / "ner_train.jsonl"
)


@pytest.fixture(scope="module")
def tokenizer():
    """A real fast tokenizer, trained locally on the actual corpus so
    subword-splitting behavior is realistic for domain abbreviations."""
    return build_tokenizer_from_ner_train(
        NER_TRAIN_PATH,
        vocab_size=3000,
    )


def _record(
    text: str,
    spans: list[tuple[int, int, str]],
    index: int = 0,
) -> NerRecord:
    return NerRecord(
        index=index,
        text=text,
        spans=tuple(
            CharSpan(start=s, end=e, label=l)
            for s, e, l in spans
        ),
    )


def _decode_non_special(
    example,
    tokenizer,
) -> list[tuple[str, str]]:
    """Helper: (token_string, BIO_label_string) pairs for non-special
    tokens only, for readable assertions."""
    tokens = tokenizer.convert_ids_to_tokens(example.input_ids)
    id_to_label = {v: k for k, v in LABEL_TO_ID.items()}

    out = []

    for tok, label_id in zip(tokens, example.labels):
        if label_id == IGNORE_LABEL_ID:
            continue

        out.append((tok, id_to_label[label_id]))

    return out


# ---------------------------------------------------------------------------
# Structural validation (parse-time, tokenizer-independent)
# ---------------------------------------------------------------------------


class TestStructuralValidation:
    def test_valid_multiword_entity_parses(self):
        rec = _record(
            "VALVE, GLOBE 100MM CLASS 300 A105 RAISED FACE",
            [
                (0, 12, "ITEM_TYPE"),
                (13, 18, "SIZE"),
                (19, 28, "PRESSURE_RATING"),
            ],
        )

        assert rec.spans[0].label == "ITEM_TYPE"

    def test_invalid_span_start_negative_raises(self):
        from app.ner_dataset import _validate_spans

        with pytest.raises(
            NerAnnotationError,
            match="out of range",
        ):
            _validate_spans(
                0,
                "GATE VALVE",
                [
                    {
                        "start": -1,
                        "end": 4,
                        "label": "ITEM_TYPE",
                    }
                ],
            )

    def test_invalid_span_end_past_text_length_raises(self):
        from app.ner_dataset import _validate_spans

        with pytest.raises(
            NerAnnotationError,
            match="out of range",
        ):
            _validate_spans(
                0,
                "GATE VALVE",
                [
                    {
                        "start": 0,
                        "end": 999,
                        "label": "ITEM_TYPE",
                    }
                ],
            )

    def test_invalid_span_start_ge_end_raises(self):
        from app.ner_dataset import _validate_spans

        with pytest.raises(
            NerAnnotationError,
            match="out of range",
        ):
            _validate_spans(
                0,
                "GATE VALVE",
                [
                    {
                        "start": 5,
                        "end": 5,
                        "label": "ITEM_TYPE",
                    }
                ],
            )

    def test_unknown_label_raises(self):
        from app.ner_dataset import _validate_spans

        with pytest.raises(
            NerAnnotationError,
            match="unknown entity label",
        ):
            _validate_spans(
                0,
                "GATE VALVE",
                [
                    {
                        "start": 0,
                        "end": 4,
                        "label": "NOT_A_LABEL",
                    }
                ],
            )

    def test_overlapping_spans_raise(self):
        from app.ner_dataset import _validate_spans

        with pytest.raises(
            NerAnnotationError,
            match="overlapping spans",
        ):
            _validate_spans(
                0,
                "GATE VALVE DN 100",
                [
                    {
                        "start": 0,
                        "end": 10,
                        "label": "ITEM_TYPE",
                    },
                    {
                        "start": 5,
                        "end": 17,
                        "label": "SIZE",
                    },
                ],
            )

    def test_error_message_includes_index_text_and_span(self):
        from app.ner_dataset import _validate_spans

        with pytest.raises(NerAnnotationError) as exc_info:
            _validate_spans(
                42,
                "BAD TEXT HERE",
                [
                    {
                        "start": 0,
                        "end": 999,
                        "label": "ITEM_TYPE",
                    }
                ],
            )

        message = str(exc_info.value)

        assert "record #42" in message
        assert "BAD TEXT HERE" in message
        assert "start=0" in message
        assert "end=999" in message


# ---------------------------------------------------------------------------
# Loading the real file
# ---------------------------------------------------------------------------


class TestLoadRealFile:
    def test_load_ner_records_from_actual_file(self):
        records = load_ner_records(NER_TRAIN_PATH)

        assert len(records) == 5000
        assert all(isinstance(r, NerRecord) for r in records)

    def test_missing_file_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_ner_records(
                "/nonexistent/path/ner_train.jsonl"
            )


# ---------------------------------------------------------------------------
# Alignment: the cases requested
# ---------------------------------------------------------------------------


class TestAlignment:
    def test_normal_multiword_entity(self, tokenizer):
        text = "VALVE, GLOBE 100MM CLASS 300 A105 RAISED FACE"

        rec = _record(
            text,
            [
                (0, 12, "ITEM_TYPE"),
                (13, 18, "SIZE"),
                (19, 28, "PRESSURE_RATING"),
                (29, 33, "METALLURGY"),
                (34, 45, "FACING_END"),
            ],
        )

        example = align_record(rec, tokenizer)
        pairs = _decode_non_special(
            example,
            tokenizer,
        )

        bio_labels = {
            label
            for _, label in pairs
        }

        for entity in (
            "ITEM_TYPE",
            "SIZE",
            "PRESSURE_RATING",
            "METALLURGY",
            "FACING_END",
        ):
            assert f"B-{entity}" in bio_labels

    def test_hyphenated_entity_flg_wn(self, tokenizer):
        text = "FLG-WN RTJ DN 300 300 LBS SS304"

        rec = _record(
            text,
            [
                (0, 6, "ITEM_TYPE"),
                (7, 10, "FACING_END"),
                (11, 17, "SIZE"),
                (18, 25, "PRESSURE_RATING"),
                (26, 31, "METALLURGY"),
            ],
        )

        example = align_record(
            rec,
            tokenizer,
        )

        assert LABEL_TO_ID["B-ITEM_TYPE"] in example.labels

    def test_inch_fractional_size(self, tokenizer):
        text = 'GLB VLV, 12", CL-1500, ASTM A216 WCB, RF, BS 1873'

        rec = _record(
            text,
            [
                (0, 7, "ITEM_TYPE"),
                (9, 12, "SIZE"),
                (14, 21, "PRESSURE_RATING"),
                (23, 36, "METALLURGY"),
                (38, 40, "FACING_END"),
                (42, 49, "STANDARD"),
            ],
        )

        example = align_record(
            rec,
            tokenizer,
        )

        assert LABEL_TO_ID["B-SIZE"] in example.labels
        assert LABEL_TO_ID["B-STANDARD"] in example.labels

    def test_fractional_slash_size(self, tokenizer):
        text = "FLG BLD RF FACING 1/2IN 300LB A350-LF2"

        rec = _record(
            text,
            [
                (0, 7, "ITEM_TYPE"),
                (8, 17, "FACING_END"),
                (18, 23, "SIZE"),
                (24, 29, "PRESSURE_RATING"),
                (30, 38, "METALLURGY"),
            ],
        )

        example = align_record(
            rec,
            tokenizer,
        )

        assert LABEL_TO_ID["B-SIZE"] in example.labels

    def test_cl1500_pn100_hash_pressure_forms(self, tokenizer):
        cases = [
            (
                'VLV-GT-2"-CLASS 1500-ASTM A182 F316-RF-API6D',
                (10, 20, "PRESSURE_RATING"),
            ),
            (
                "FLG-WN RTJ FACING 200MM PN20 LF2 ANSI/ASME B16.5",
                (24, 28, "PRESSURE_RATING"),
            ),
            (
                "GT VLV DN 300 900# A105 BW ENDS",
                (14, 18, "PRESSURE_RATING"),
            ),
        ]

        for text, (s, e, label) in cases:
            rec = _record(
                text,
                [(s, e, label)],
            )

            example = align_record(
                rec,
                tokenizer,
            )

            assert LABEL_TO_ID[f"B-{label}"] in example.labels, text

    def test_a216_wcb_and_l7_gr7_metallurgy_forms(self, tokenizer):
        cases = [
            (
                'VLV BL-4"-150#-CAST STEEL WCB-RAISED FACE',
                (15, 29, "METALLURGY"),
            ),
            (
                "BLT-STUD-DN200-150#-L7-GR7-THRD-ASME B18.2.1",
                (20, 26, "METALLURGY"),
            ),
        ]

        for text, (s, e, label) in cases:
            rec = _record(
                text,
                [(s, e, label)],
            )

            example = align_record(
                rec,
                tokenizer,
            )

            assert LABEL_TO_ID[f"B-{label}"] in example.labels, text

    def test_asme_b16_5_standard_with_space(self, tokenizer):
        text = (
            "VALVE, GLOBE 100MM CLASS 300 A105 "
            "RAISED FACE ANSI/ASME B16.34"
        )

        rec = _record(
            text,
            [(46, 62, "STANDARD")],
        )

        example = align_record(
            rec,
            tokenizer,
        )

        assert LABEL_TO_ID["B-STANDARD"] in example.labels

    def test_record_without_standard_has_no_standard_labels(
        self,
        tokenizer,
    ):
        text = "FLG WN-1.5 INCH-PN100-SS304-RF"

        rec = _record(
            text,
            [
                (0, 6, "ITEM_TYPE"),
                (7, 15, "SIZE"),
                (16, 21, "PRESSURE_RATING"),
                (22, 27, "METALLURGY"),
                (28, 30, "FACING_END"),
            ],
        )

        example = align_record(
            rec,
            tokenizer,
        )

        assert LABEL_TO_ID["B-STANDARD"] not in example.labels
        assert LABEL_TO_ID["I-STANDARD"] not in example.labels

    def test_subword_split_entity_still_aligns(self, tokenizer):
        text = "STUDBLT-50MM-CLASS 600-B7-2H-THRD"

        rec = _record(
            text,
            [
                (0, 7, "ITEM_TYPE"),
                (8, 12, "SIZE"),
                (13, 22, "PRESSURE_RATING"),
                (23, 28, "METALLURGY"),
                (29, 33, "FACING_END"),
            ],
        )

        example = align_record(
            rec,
            tokenizer,
        )

        pairs = _decode_non_special(
            example,
            tokenizer,
        )

        item_type_tokens = [
            t
            for t, l in pairs
            if l.endswith("ITEM_TYPE")
        ]

        item_type_labels = [
            l
            for _, l in pairs
            if l.endswith("ITEM_TYPE")
        ]

        assert item_type_labels[0] == "B-ITEM_TYPE"

        assert all(
            l == "I-ITEM_TYPE"
            for l in item_type_labels[1:]
        )

        assert len(item_type_tokens) >= 1

    def test_all_five_thousand_real_records_align_without_error(
        self,
        tokenizer,
    ):
        """Integration checkpoint: every real record in ner_train.jsonl
        must align cleanly against a real fast tokenizer with no errors."""
        records = load_ner_records(
            NER_TRAIN_PATH
        )

        examples = prepare_ner_dataset(
            records,
            tokenizer,
            max_length=64,
        )

        assert len(examples) == len(records)


# ---------------------------------------------------------------------------
# Invalid / overlapping spans caught before alignment is attempted
# ---------------------------------------------------------------------------


class TestInvalidSpans:
    def test_out_of_range_span_raises_before_tokenization(
        self,
        tokenizer,
    ):
        from app.ner_dataset import _validate_spans

        with pytest.raises(NerAnnotationError):
            _validate_spans(
                0,
                "GATE VALVE",
                [
                    {
                        "start": 0,
                        "end": 500,
                        "label": "ITEM_TYPE",
                    }
                ],
            )

    def test_negative_start_raises(self):
        from app.ner_dataset import _validate_spans

        with pytest.raises(NerAnnotationError):
            _validate_spans(
                0,
                "GATE VALVE",
                [
                    {
                        "start": -5,
                        "end": 4,
                        "label": "ITEM_TYPE",
                    }
                ],
            )

    def test_overlapping_spans_raise_before_tokenization(self):
        from app.ner_dataset import _validate_spans

        with pytest.raises(
            NerAnnotationError,
            match="overlapping",
        ):
            _validate_spans(
                0,
                "FLG-WN RTJ DN 300",
                [
                    {
                        "start": 0,
                        "end": 10,
                        "label": "ITEM_TYPE",
                    },
                    {
                        "start": 7,
                        "end": 17,
                        "label": "FACING_END",
                    },
                ],
            )

    def test_full_jsonl_parse_error_includes_context(
        self,
        tmp_path,
    ):
        bad_file = (
            tmp_path
            / "bad_ner_train.jsonl"
        )

        bad_file.write_text(
            '{"text": "GATE VALVE", '
            '"labels": [{"start": 0, "end": 999, '
            '"label": "ITEM_TYPE"}]}\n',
            encoding="utf-8",
        )

        with pytest.raises(
            NerAnnotationError
        ) as exc_info:
            load_ner_records(
                bad_file
            )

        message = str(exc_info.value)

        assert "record #0" in message
        assert "GATE VALVE" in message


# ---------------------------------------------------------------------------
# Alignment-layer failure modes (require a tokenizer)
# ---------------------------------------------------------------------------


class TestAlignmentFailureModes:
    def test_truncation_cutting_into_span_raises_alignment_error(
        self,
        tokenizer,
    ):
        text = (
            "GATE VALVE DN 100 CL1500 F316 "
            "RF ASME B16.34"
        )

        span_start = text.index(
            "ASME B16.34"
        )

        span_end = (
            span_start
            + len("ASME B16.34")
        )

        rec = _record(
            text,
            [
                (
                    span_start,
                    span_end,
                    "STANDARD",
                )
            ],
        )

        with pytest.raises(
            NerAlignmentError,
            match=(
                "did not align to any non-special token"
                "|boundary mismatch"
            ),
        ):
            align_record(
                rec,
                tokenizer,
                max_length=6,
            )

    def test_span_with_no_covering_token_raises(
        self,
        tokenizer,
    ):
        text = "GATE VALVE"

        rec = _record(
            text,
            [
                (
                    4,
                    5,
                    "ITEM_TYPE",
                )
            ],
        )

        with pytest.raises(
            NerAlignmentError,
            match="did not align to any non-special token",
        ):
            align_record(
                rec,
                tokenizer,
            )

    def test_non_fast_tokenizer_rejected(self):
        class _FakeSlowTokenizer:
            is_fast = False

        rec = _record(
            "GATE VALVE",
            [
                (
                    0,
                    4,
                    "ITEM_TYPE",
                )
            ],
        )

        with pytest.raises(
            ValueError,
            match="fast tokenizer",
        ):
            align_record(
                rec,
                _FakeSlowTokenizer(),
            )

