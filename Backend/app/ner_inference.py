"""
Production inference for the Samanvay-AI material-attribute NER model.

Production input:
    raw_description
        -> deterministic normalization
        -> fast-tokenizer inference
        -> BIO predictions
        -> predicted character spans
        -> material attributes

This module MUST NOT use:
    - canonical_id
    - extracted_* columns
    - gold NER spans
    - CPSE-specific reference annotations

The trained NER model and tokenizer are supplied by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.ingestion.data_pipeline.normalizer import normalize_text
from app.ner_dataset import ENTITY_TYPES, ID_TO_LABEL


@dataclass(frozen=True)
class PredictedEntity:
    """One entity predicted by the NER model."""

    label: str
    text: str
    start: int
    end: int


def predict_entities(
    raw_description: object,
    model,
    tokenizer,
    max_length: int = 64,
) -> list[PredictedEntity]:
    """
    Predict material-attribute entities from one raw description.

    The production input is raw_description only. The description is first
    passed through the deterministic production normalizer.

    Returns predicted entities with character offsets relative to the
    normalized description.
    """
    normalized_text = normalize_text(raw_description)

    if normalized_text is None:
        return []

    encoding = tokenizer(
        normalized_text,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    offsets = encoding.pop("offset_mapping")[0].tolist()
    special_tokens_mask = encoding.pop("special_tokens_mask")[0].tolist()

    model_inputs = {
        key: value.to(model.device)
        for key, value in encoding.items()
    }

    model.eval()

    with __import__("torch").no_grad():
        outputs = model(**model_inputs)

    predicted_ids = outputs.logits.argmax(dim=-1)[0].tolist()

    entities: list[PredictedEntity] = []
    current_label: str | None = None
    current_start: int | None = None
    current_end: int | None = None

    def close_entity() -> None:
        nonlocal current_label, current_start, current_end

        if (
            current_label is not None
            and current_start is not None
            and current_end is not None
        ):
            entities.append(
                PredictedEntity(
                    label=current_label,
                    text=normalized_text[current_start:current_end],
                    start=current_start,
                    end=current_end,
                )
            )

        current_label = None
        current_start = None
        current_end = None

    for token_idx, predicted_id in enumerate(predicted_ids):
        if special_tokens_mask[token_idx]:
            close_entity()
            continue

        token_start, token_end = offsets[token_idx]

        if token_start == token_end:
            close_entity()
            continue

        label = ID_TO_LABEL[predicted_id]

        if label == "O":
            close_entity()
            continue

        prefix, entity_type = label.split("-", 1)

        if prefix == "B":
            close_entity()

            current_label = entity_type
            current_start = token_start
            current_end = token_end

        elif prefix == "I":
            if current_label == entity_type:
                current_end = token_end
            else:
                close_entity()

                current_label = entity_type
                current_start = token_start
                current_end = token_end

    close_entity()

    return entities


def predict_attributes(
    raw_description: object,
    model,
    tokenizer,
    max_length: int = 64,
) -> dict[str, str | None]:
    """
    Predict the six supported material attributes from raw_description.

    Missing attributes are returned as None.
    """

    entities = predict_entities(
        raw_description=raw_description,
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    attributes: dict[str, str | None] = {
        "item_type": None,
        "size": None,
        "pressure_rating": None,
        "metallurgy": None,
        "facing_end": None,
        "standard": None,
    }

    label_to_key = {
        "ITEM_TYPE": "item_type",
        "SIZE": "size",
        "PRESSURE_RATING": "pressure_rating",
        "METALLURGY": "metallurgy",
        "FACING_END": "facing_end",
        "STANDARD": "standard",
    }

    for entity in entities:
        key = label_to_key[entity.label]

        if attributes[key] is None:
            attributes[key] = entity.text

    return attributes