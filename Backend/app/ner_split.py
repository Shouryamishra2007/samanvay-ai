"""
ner_split.py

Deterministic, stratified train/validation/test split for the NER dataset.

Scope
-----
This module does ONLY the split. It builds on `ner_dataset.load_ner_records`
(imported, not modified) and produces three disjoint lists of `NerRecord`
plus a statistics report. It does NOT tokenize, align to BIO labels, build
a model, or train anything -- that is `ner_dataset.py` (already validated)
and a later training script respectively.

Stratification
---------------
Stratified on presence/absence of a STANDARD span, since that is the one
structurally non-universal field in this dataset (64.76% of the 5,000
records have a STANDARD span, 35.24% do not -- see prior schema report).
Splitting naively (unstratified) risks a validation or test fold with a
meaningfully different STANDARD-presence rate than training, which would
make STANDARD-specific metrics harder to interpret across folds.

Determinism
------------
A single fixed seed (`DEFAULT_SEED = 42`) drives both stratified splits
(train vs. temp, then temp vs. val/test) via `sklearn.train_test_split`,
which is already installed in this environment -- no new package required.
Given the same input records and the same seed, this function always
returns byte-identical split membership.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from sklearn.model_selection import train_test_split

from app.ml.ner_dataset import ENTITY_TYPES, NerRecord, load_ner_records

LOGGER = logging.getLogger(__name__)

DEFAULT_SEED: int = 42
DEFAULT_TRAIN_FRACTION: float = 0.8
DEFAULT_VAL_FRACTION: float = 0.1
DEFAULT_TEST_FRACTION: float = 0.1


class SplitIntegrityError(ValueError):
    """Raised when the produced split fails one of the required integrity
    checks (record leakage, duplicate text across folds, a label missing
    from training)."""


@dataclass(frozen=True)
class NerSplit:
    """Three disjoint sets of NerRecord. Every input record appears in
    exactly one of these lists."""

    train: list[NerRecord]
    validation: list[NerRecord]
    test: list[NerRecord]


def _has_standard(record: NerRecord) -> bool:
    return any(span.label == "STANDARD" for span in record.spans)


def split_ner_records(
    records: list[NerRecord],
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    seed: int = DEFAULT_SEED,
) -> NerSplit:
    """Deterministic 80/10/10 split, stratified on STANDARD presence.

    Implementation: two sequential stratified splits (both driven by the
    same `seed`), rather than a single three-way split, since
    `train_test_split` is two-way only:
      1. train (80%) vs. temp (20%), stratified on STANDARD presence
      2. temp -> validation (50%) / test (50%), i.e. 10%/10% of the
         original total, again stratified on STANDARD presence

    Every record is split by its own index/identity -- a record is never
    duplicated or dropped, only assigned to one of the three output lists.
    """
    fractions_total = train_fraction + val_fraction + test_fraction
    if abs(fractions_total - 1.0) > 1e-9:
        raise ValueError(
            f"train/val/test fractions must sum to 1.0, got {fractions_total}"
        )

    stratify_labels = [_has_standard(r) for r in records]

    train_records, temp_records, train_strat, temp_strat = train_test_split(
        records,
        stratify_labels,
        train_size=train_fraction,
        random_state=seed,
        stratify=stratify_labels,
        shuffle=True,
    )

    # val_fraction and test_fraction are fractions of the ORIGINAL total,
    # so within `temp_records` (which is (1 - train_fraction) of the
    # total) the validation share is val_fraction / (1 - train_fraction).
    remaining_fraction = 1.0 - train_fraction
    val_share_of_temp = val_fraction / remaining_fraction

    val_records, test_records = train_test_split(
        temp_records,
        train_size=val_share_of_temp,
        random_state=seed,
        stratify=temp_strat,
        shuffle=True,
    )

    return NerSplit(train=train_records, validation=val_records, test=test_records)


# ---------------------------------------------------------------------------
# Integrity checks (fail-loud, consistent with the rest of this pipeline)
# ---------------------------------------------------------------------------


def verify_split_integrity(split: NerSplit, original_records: list[NerRecord]) -> None:
    """Raise `SplitIntegrityError` if any of the required guarantees don't
    hold. Never silently drops or reassigns a record to "fix" a violation.

    Checks:
      - every original record appears in exactly one of the three folds
        (no leakage, no loss, no duplication across folds)
      - no duplicate `text` value crosses a fold boundary
      - every one of the six entity labels appears at least once in the
        training fold
    """
    all_indices = [r.index for r in original_records]
    split_indices = (
        [r.index for r in split.train]
        + [r.index for r in split.validation]
        + [r.index for r in split.test]
    )

    if sorted(split_indices) != sorted(all_indices):
        missing = set(all_indices) - set(split_indices)
        duplicated = [idx for idx, count in Counter(split_indices).items() if count > 1]
        raise SplitIntegrityError(
            f"split does not exactly partition the original records: "
            f"{len(missing)} missing indices, {len(duplicated)} duplicated indices"
        )

    text_to_fold: dict[str, str] = {}
    for fold_name, fold_records in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        for record in fold_records:
            if record.text in text_to_fold and text_to_fold[record.text] != fold_name:
                raise SplitIntegrityError(
                    f"duplicate text crosses splits: {record.text!r} appears in "
                    f"both {text_to_fold[record.text]!r} and {fold_name!r}"
                )
            text_to_fold[record.text] = fold_name

    train_labels_present = {
        span.label for record in split.train for span in record.spans
    }
    missing_labels = set(ENTITY_TYPES) - train_labels_present
    if missing_labels:
        raise SplitIntegrityError(
            f"training fold is missing entity label(s) entirely: {missing_labels}"
        )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FoldStats:
    name: str
    record_count: int
    standard_present_count: int
    standard_present_pct: float
    entity_counts: dict[str, int]


def compute_fold_stats(name: str, fold_records: list[NerRecord]) -> FoldStats:
    n = len(fold_records)
    standard_present = sum(1 for r in fold_records if _has_standard(r))
    entity_counts: Counter[str] = Counter()
    for record in fold_records:
        for span in record.spans:
            entity_counts[span.label] += 1

    return FoldStats(
        name=name,
        record_count=n,
        standard_present_count=standard_present,
        standard_present_pct=(100.0 * standard_present / n) if n else 0.0,
        entity_counts={label: entity_counts.get(label, 0) for label in ENTITY_TYPES},
    )


def build_split_report(split: NerSplit) -> dict[str, FoldStats]:
    return {
        "train": compute_fold_stats("train", split.train),
        "validation": compute_fold_stats("validation", split.validation),
        "test": compute_fold_stats("test", split.test),
    }


def print_split_report(report: dict[str, FoldStats]) -> None:
    for fold_name in ("train", "validation", "test"):
        stats = report[fold_name]
        print(f"\n=== {stats.name.upper()} ===")
        print(f"records: {stats.record_count}")
        print(
            f"STANDARD present: {stats.standard_present_count} "
            f"({stats.standard_present_pct:.2f}%)"
        )
        print("entity counts:")
        for label, count in stats.entity_counts.items():
            print(f"  {label:16s} {count}")


if __name__ == "__main__":
    import sys

    ner_train_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "../data/ml_training/ner_train.jsonl"
    )
    records = load_ner_records(ner_train_path)
    split = split_ner_records(records)
    verify_split_integrity(split, records)
    report = build_split_report(split)
    print_split_report(report)
    print("\nAll integrity checks passed.")
