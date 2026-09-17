"""
Unit tests for production NER inference.

These tests verify that raw_description is converted into predicted
material attributes without using canonical_id, extracted_* fields,
or gold annotations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # backend/

from app.ner_inference import (  # noqa: E402
    PredictedEntity,
    predict_attributes,
)
class FakeEncoding(dict):
    """Minimal tokenizer output compatible with predict_entities()."""

    def __init__(self, input_ids, attention_mask, offset_mapping, special_tokens_mask):
        super().__init__(
            input_ids=torch.tensor([input_ids], dtype=torch.long),
            attention_mask=torch.tensor([attention_mask], dtype=torch.long),
            offset_mapping=torch.tensor([offset_mapping], dtype=torch.long),
            special_tokens_mask=torch.tensor(
                [special_tokens_mask], dtype=torch.long
            ),
        )


class FakeTokenizer:
    """Character-level tokenizer for deterministic inference tests."""

    def __call__(
        self,
        text,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    ):
        offsets = [(i, i + 1) for i in range(len(text))]

        return FakeEncoding(
            input_ids=list(range(len(text))),
            attention_mask=[1] * len(text),
            offset_mapping=offsets,
            special_tokens_mask=[0] * len(text),
        )


class FakeOutput:
    """Minimal model output containing token-classification logits."""

    def __init__(self, logits):
        self.logits = logits


class FakeModel:
    """Deterministic model that emits predefined BIO labels."""

    def __init__(self, labels):
        self.labels = labels
        self.device = torch.device("cpu")

    def eval(self):
        return self

    def __call__(self, **model_inputs):
        sequence_length = model_inputs["input_ids"].shape[1]

        logits = torch.full(
            (1, sequence_length, 13),
            -1000.0,
            dtype=torch.float32,
        )

        for index, label_id in enumerate(self.labels):
            logits[0, index, label_id] = 1000.0

        return FakeOutput(logits)

def _bio_labels_for_entities(text, entities):
    """Create token-level BIO label IDs for character-level fake tokens."""
    from app.ner_dataset import LABEL_TO_ID

    labels = [LABEL_TO_ID["O"]] * len(text)

    for start, end, entity_type in entities:
        labels[start] = LABEL_TO_ID[f"B-{entity_type}"]

        for index in range(start + 1, end):
            labels[index] = LABEL_TO_ID[f"I-{entity_type}"]

    return labels

def test_predict_attributes_complete_description():
    """All attributes present in the description should be extracted."""
    raw_description = "Gate Valve, 150mm, PN150, AISI 316, BW Ends"

    entities = [
        (0, 10, "ITEM_TYPE"),
        (12, 17, "SIZE"),
        (19, 24, "PRESSURE_RATING"),
        (26, 34, "METALLURGY"),
        (36, 43, "FACING_END"),
    ]

    tokenizer = FakeTokenizer()
    model = FakeModel(
        _bio_labels_for_entities(raw_description.upper(), entities)
    )

    attributes = predict_attributes(
        raw_description=raw_description,
        model=model,
        tokenizer=tokenizer,
        max_length=64,
    )

    assert attributes == {
        "item_type": "GATE VALVE",
        "size": "150MM",
        "pressure_rating": "PN150",
        "metallurgy": "AISI 316",
        "facing_end": "BW ENDS",
        "standard": None,
    }

def test_predict_attributes_missing_pressure_rating():
    """Missing attributes should remain None."""
    raw_description = "Gate Valve, 150mm, AISI 316, BW Ends"

    entities = [
        (0, 10, "ITEM_TYPE"),
        (12, 17, "SIZE"),
        (19, 27, "METALLURGY"),
        (29, 36, "FACING_END"),
    ]

    normalized_text = raw_description.upper()

    tokenizer = FakeTokenizer()
    model = FakeModel(
        _bio_labels_for_entities(normalized_text, entities)
    )

    attributes = predict_attributes(
        raw_description=raw_description,
        model=model,
        tokenizer=tokenizer,
        max_length=64,
    )

    assert attributes == {
        "item_type": "GATE VALVE",
        "size": "150MM",
        "pressure_rating": None,
        "metallurgy": "AISI 316",
        "facing_end": "BW ENDS",
        "standard": None,
    }

def test_predict_attributes_with_standard():
    """STANDARD should be extracted when explicitly present."""
    raw_description = "Gate Valve, 150mm, PN150, AISI 316, BW Ends, API 6D"

    entities = [
        (0, 10, "ITEM_TYPE"),
        (12, 17, "SIZE"),
        (19, 24, "PRESSURE_RATING"),
        (26, 34, "METALLURGY"),
        (36, 43, "FACING_END"),
        (45, 51, "STANDARD"),
    ]

    normalized_text = raw_description.upper()

    tokenizer = FakeTokenizer()
    model = FakeModel(
        _bio_labels_for_entities(normalized_text, entities)
    )

    attributes = predict_attributes(
        raw_description=raw_description,
        model=model,
        tokenizer=tokenizer,
        max_length=64,
    )

    assert attributes == {
        "item_type": "GATE VALVE",
        "size": "150MM",
        "pressure_rating": "PN150",
        "metallurgy": "AISI 316",
        "facing_end": "BW ENDS",
        "standard": "API 6D",
    }

@pytest.mark.parametrize(
    "raw_description",
    [
        None,
        "",
        "   ",
        12345,
    ],
)
def test_predict_attributes_invalid_input(raw_description):
    """Invalid or empty descriptions should return empty attributes."""
    attributes = predict_attributes(
        raw_description=raw_description,
        model=None,
        tokenizer=None,
        max_length=64,
    )

    assert attributes == {
        "item_type": None,
        "size": None,
        "pressure_rating": None,
        "metallurgy": None,
        "facing_end": None,
        "standard": None,
    }
def test_predict_attributes_variant_terminology():
    """Variant technical terminology should be returned as predicted."""
    raw_description = "GATE VLV 150MM PN150 SS316 RF FACING API 6D"

    entities = [
        (0, 8, "ITEM_TYPE"),
        (9, 14, "SIZE"),
        (15, 20, "PRESSURE_RATING"),
        (21, 26, "METALLURGY"),
        (27, 36, "FACING_END"),
        (37, 43, "STANDARD"),
    ]

    normalized_text = raw_description.upper()

    tokenizer = FakeTokenizer()
    model = FakeModel(
        _bio_labels_for_entities(normalized_text, entities)
    )

    attributes = predict_attributes(
        raw_description=raw_description,
        model=model,
        tokenizer=tokenizer,
        max_length=64,
    )

    assert attributes == {
        "item_type": "GATE VLV",
        "size": "150MM",
        "pressure_rating": "PN150",
        "metallurgy": "SS316",
        "facing_end": "RF FACING",
        "standard": "API 6D",
    }