"""
ner_dataset.py

Dataset preparation for the material-attribute NER model: converts the
character-span annotations in ``ner_train.jsonl`` into BIO token labels
aligned against a Hugging Face *fast* tokenizer's ``offset_mapping``.

Scope
-----
This module does ONLY:
    - parse + structurally validate ner_train.jsonl records
    - tokenize each record with a caller-supplied fast tokenizer
    - align character spans to token positions using offset_mapping
    - produce BIO labels (with -100 for special tokens) ready for
      AutoModelForTokenClassification training

It does NOT:
    - pick or load a specific tokenizer/model (the tokenizer is a
      caller-supplied dependency -- see "Tokenizer dependency" below)
    - train anything
    - read canonical_id, extracted_*, or anything from the CPSE CSVs /
      loader.py / normalizer.py. ner_train.jsonl is the only annotation
      source for this component, exactly as specified.
    - split into train/val/test (a separate, later concern)

Tokenizer dependency
---------------------
`align_record` / `prepare_ner_dataset` take a `tokenizer` argument typed as
`transformers.PreTrainedTokenizerFast`. The alignment logic is written
entirely against the fast-tokenizer contract (`return_offsets_mapping`,
`return_special_tokens_mask`) and does not depend on which vocabulary the
tokenizer holds. Pass the real `AutoTokenizer.from_pretrained(
"distilbert-base-cased", use_fast=True)` wherever this runs with internet
access to the Hugging Face Hub. This module never loads a tokenizer itself.

Why BIO, and why -100
----------------------
BIO ("Beginning/Inside/Outside") lets a token-classification model output a
label per token that a simple decoder can turn back into spans. -100 is the
sentinel `torch.nn.CrossEntropyLoss` (and HF's Trainer) uses to mean "ignore
this position when computing loss" -- used here for [CLS]/[SEP]/[PAD] and
any other special token that isn't part of the actual material description.

Fail-loud policy
-----------------
Every validation in this module raises `NerAlignmentError` with the record
index, the record's text, and the offending span on failure. Nothing is
silently dropped, clipped, or "best-effort" repaired -- a bad annotation or
an alignment mismatch is a data problem to fix at the source, not something
this module should paper over.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Label schema
# ---------------------------------------------------------------------------

ENTITY_TYPES: tuple[str, ...] = (
    "ITEM_TYPE",
    "SIZE",
    "PRESSURE_RATING",
    "METALLURGY",
    "FACING_END",
    "STANDARD",
)

# Sentinel used by HF Trainer / torch.nn.CrossEntropyLoss to mean
# "exclude this token position from the loss".
IGNORE_LABEL_ID: int = -100

_OUTSIDE_LABEL: str = "O"


def build_bio_label_list() -> list[str]:
    """Return the fixed, ordered BIO label vocabulary: O, then B-/I- for
    each entity type. Order is deterministic so label ids are stable across
    runs (important for reproducibility -- a model checkpoint's label ids
    only mean anything if this ordering never silently changes)."""
    labels = [_OUTSIDE_LABEL]
    for entity in ENTITY_TYPES:
        labels.append(f"B-{entity}")
        labels.append(f"I-{entity}")
    return labels


BIO_LABELS: list[str] = build_bio_label_list()
LABEL_TO_ID: dict[str, int] = {label: idx for idx, label in enumerate(BIO_LABELS)}
ID_TO_LABEL: dict[int, str] = {idx: label for label, idx in LABEL_TO_ID.items()}


# ---------------------------------------------------------------------------
# Source annotation model
# ---------------------------------------------------------------------------


class NerAnnotationError(ValueError):
    """Raised when a source record in ner_train.jsonl is structurally
    invalid (bad offsets, overlapping spans, unknown label) -- independent
    of any tokenizer."""


class NerAlignmentError(ValueError):
    """Raised when a structurally valid record cannot be reliably mapped to
    BIO token labels for a given tokenizer (span not covered by any
    non-special token, a token spanning two different labels, truncation
    cutting into a span, or reconstructed boundaries that don't match the
    source annotation)."""


@dataclass(frozen=True)
class CharSpan:
    """A single character-offset entity annotation, exactly as authored in
    ner_train.jsonl."""

    start: int
    end: int
    label: str


@dataclass(frozen=True)
class NerRecord:
    """One structurally validated ner_train.jsonl record."""

    index: int
    text: str
    spans: tuple[CharSpan, ...]


@dataclass(frozen=True)
class AlignedExample:
    """One record after tokenization + BIO alignment, ready to feed a
    token-classification model (after batching/padding, which is a
    training-time collator concern, not this module's)."""

    index: int
    text: str
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]


def _fail(index: int, text: str, message: str, span: CharSpan | None = None) -> None:
    """Raise NerAnnotationError with full context, per the fail-loud policy:
    every error names the record index, its text, and the offending span."""
    span_info = f" span={span!r}" if span is not None else ""
    raise NerAnnotationError(
        f"record #{index}: {message}{span_info} | text={text!r}"
    )


def _validate_spans(index: int, text: str, raw_spans: list[dict]) -> tuple[CharSpan, ...]:
    """Structural validation of one record's spans, independent of any
    tokenizer:

      1. every span has a start/end within [0, len(text)] and start < end
      2. the span's label is one of the six known entity types
      3. no two spans overlap
    """
    spans: list[CharSpan] = []
    for raw in raw_spans:
        try:
            start, end, label = raw["start"], raw["end"], raw["label"]
        except KeyError as exc:
            _fail(index, text, f"span is missing required key {exc}")

        if not isinstance(start, int) or not isinstance(end, int):
            _fail(index, text, f"span start/end must be int, got {raw!r}")

        if start < 0 or end > len(text) or start >= end:
            _fail(
                index,
                text,
                f"span out of range or empty (text length={len(text)})",
                CharSpan(start, end, label),
            )

        if label not in ENTITY_TYPES:
            _fail(
                index,
                text,
                f"unknown entity label {label!r}; expected one of {ENTITY_TYPES}",
                CharSpan(start, end, label),
            )

        spans.append(CharSpan(start=start, end=end, label=label))

    spans.sort(key=lambda s: s.start)
    for prev, curr in zip(spans, spans[1:]):
        if curr.start < prev.end:
            _fail(
                index,
                text,
                f"overlapping spans: {prev!r} overlaps {curr!r}",
                curr,
            )

    return tuple(spans)


def iter_ner_records(path: str | Path) -> Iterator[NerRecord]:
    """Stream-parse ner_train.jsonl, validating each record structurally
    (see `_validate_spans`) before yielding it. Raises `NerAnnotationError`
    immediately on the first invalid record, naming its index/text/span --
    a malformed annotation is never silently skipped or repaired.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"NER training file not found: {path}")

    with path.open(encoding="utf-8") as fh:
        for index, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise NerAnnotationError(
                    f"record #{index}: invalid JSON: {exc} | raw_line={line!r}"
                ) from exc

            if "text" not in raw or "labels" not in raw:
                _fail(index, raw.get("text", ""), "record missing 'text' or 'labels' key")

            text = raw["text"]
            if not isinstance(text, str) or not text:
                _fail(index, str(text), "record 'text' must be a non-empty string")

            spans = _validate_spans(index, text, raw["labels"])
            yield NerRecord(index=index, text=text, spans=spans)


def load_ner_records(path: str | Path) -> list[NerRecord]:
    """Materialize all validated records from ner_train.jsonl into a list.
    Deterministic: preserves file order."""
    records = list(iter_ner_records(path))
    LOGGER.info("Loaded and validated %d NER records from %s", len(records), path)
    return records


# ---------------------------------------------------------------------------
# Character-span -> token-span (BIO) alignment
# ---------------------------------------------------------------------------


def align_record(
    record: NerRecord,
    tokenizer: "PreTrainedTokenizerFast",
    max_length: int = 128,
) -> AlignedExample:
    """Align one record's character-span annotations to BIO token labels
    using the tokenizer's `offset_mapping`. Never infers boundaries from
    whitespace or regex -- entirely offset-driven.

    Validation performed (fail-loud, raises `NerAlignmentError` with
    record index/text/span on any violation):

      3. every entity span aligns to at least one non-special token
      4. no token receives conflicting labels from two different spans
      5. the token(s) assigned to a span reconstruct exactly the original
         character boundaries (start and end) -- this also catches
         truncation silently cutting into a span, since a truncated span's
         trailing tokens simply won't be present to reconstruct `end`.

    (Validations 1 and 2 -- valid spans, no overlaps -- are structural and
    already enforced in `_validate_spans` at parse time.)

    Partial token/span overlap (a token's character range crosses a span
    boundary without being fully inside or fully outside it) is treated as
    an alignment failure rather than assigned to either side, since there
    is no non-arbitrary way to label such a token correctly.
    """
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(
            "align_record requires a fast tokenizer (offset_mapping support); "
            f"got {type(tokenizer).__name__} with is_fast=False."
        )

    encoding = tokenizer(
        record.text,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        truncation=True,
        max_length=max_length,
    )
    offsets: list[tuple[int, int]] = encoding["offset_mapping"]
    special_mask: list[int] = encoding["special_tokens_mask"]
    num_tokens = len(offsets)

    labels: list[int | None] = [None] * num_tokens
    assigned_span_idx: list[int | None] = [None] * num_tokens

    for i in range(num_tokens):
        if special_mask[i] == 1:
            labels[i] = IGNORE_LABEL_ID

    for span_idx, span in enumerate(record.spans):
        covering_token_indices: list[int] = []

        for i in range(num_tokens):
            if special_mask[i] == 1:
                continue
            token_start, token_end = offsets[i]
            if token_start == token_end:
                # A real (non-special) token should never have a zero-width
                # offset for a fast WordPiece tokenizer; treat it the same
                # as "no overlap" rather than guessing.
                continue

            overlaps = not (token_end <= span.start or token_start >= span.end)
            if not overlaps:
                continue

            fully_contained = token_start >= span.start and token_end <= span.end
            if not fully_contained:
                raise NerAlignmentError(
                    f"record #{record.index}: token [{token_start}:{token_end}] "
                    f"partially overlaps {span!r} without being fully contained "
                    f"in it -- cannot assign an unambiguous label | "
                    f"text={record.text!r}"
                )

            if assigned_span_idx[i] is not None and assigned_span_idx[i] != span_idx:
                conflicting_span = record.spans[assigned_span_idx[i]]
                raise NerAlignmentError(
                    f"record #{record.index}: token [{token_start}:{token_end}] "
                    f"would receive conflicting labels from {conflicting_span!r} "
                    f"and {span!r} | text={record.text!r}"
                )

            assigned_span_idx[i] = span_idx
            covering_token_indices.append(i)

        if not covering_token_indices:
            raise NerAlignmentError(
                f"record #{record.index}: {span!r} did not align to any "
                f"non-special token (possibly truncated away by "
                f"max_length={max_length}) | text={record.text!r}"
            )

        reconstructed_start = min(offsets[i][0] for i in covering_token_indices)
        reconstructed_end = max(offsets[i][1] for i in covering_token_indices)
        if reconstructed_start != span.start or reconstructed_end != span.end:
            raise NerAlignmentError(
                f"record #{record.index}: {span!r} reconstructed as "
                f"[{reconstructed_start}:{reconstructed_end}] from assigned "
                f"tokens -- boundary mismatch (likely truncation or a token "
                f"crossing the span edge) | text={record.text!r}"
            )

        covering_token_indices.sort()
        for position, token_idx in enumerate(covering_token_indices):
            prefix = "B" if position == 0 else "I"
            labels[token_idx] = LABEL_TO_ID[f"{prefix}-{span.label}"]

    for i in range(num_tokens):
        if labels[i] is None:
            labels[i] = LABEL_TO_ID[_OUTSIDE_LABEL]

    return AlignedExample(
        index=record.index,
        text=record.text,
        input_ids=encoding["input_ids"],
        attention_mask=encoding["attention_mask"],
        labels=labels,  # type: ignore[arg-type]  # all None slots filled above
    )


def prepare_ner_dataset(
    records: list[NerRecord],
    tokenizer: "PreTrainedTokenizerFast",
    max_length: int = 128,
) -> list[AlignedExample]:
    """Align every record in `records` against `tokenizer`. Fails on the
    first alignment error rather than skipping bad records, consistent
    with the rest of this pipeline's fail-loud data contract."""
    return [align_record(record, tokenizer, max_length=max_length) for record in records]
