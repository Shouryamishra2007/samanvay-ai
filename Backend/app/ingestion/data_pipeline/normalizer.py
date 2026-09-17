"""
Deterministic text normalization for Samanvay-AI material descriptions.

This module implements ONLY the normalization stage of the ingestion
pipeline:

    CPSE Material CSV -> Data Loader -> [Text Normalization] -> ...

Design goals
------------
- Normalize FORMATTING, never engineering/technical MEANING.
- Derive `normalized_description` exclusively from `raw_description`, the
  only field the real production system will have available before the
  NER/attribute-extraction stage exists.
- Never read `canonical_id` or any `extracted_*` column. Those are
  evaluation/reference-only (see `loader.GROUND_TRUTH_COLUMNS`) and must
  have zero influence on the normalized production representation.
- Be deterministic, idempotent (where safe), and cheap per row so it scales
  to millions of material records (no per-row model calls, all regexes
  precompiled at module load).

What this module deliberately does NOT do
------------------------------------------
No embeddings, no NER, no fuzzy/semantic matching, no equivalence
decisions between materials, no tolerance/engineering rules, no Tier
classification. Those belong to later pipeline stages.

Hyphen handling rationale
--------------------------
The initial implementation contained a regex to separate letter-hyphen-letter
patterns (e.g., GATE-VALVE -> GATE VALVE). This rule was REMOVED because:

1. Real CPSE material descriptions do not contain letter-hyphen-letter pairs
2. Hyphens that DO appear are adjacent to digits (A-105, F-316L, M12-65)
   and are correctly preserved
3. Removing a potentially destructive transformation is safer than keeping it
   for theoretical cases
4. The normalization priority is preservation of technical meaning, not
   aesthetic uniformity

If letter-hyphen-letter patterns emerge in production data, the decision can
be revisited with empirical evidence.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Final, Mapping

import pandas as pd

from app.ingestion.data_pipeline.loader import (
    GROUND_TRUTH_COLUMNS,
    INFERENCE_COLUMNS,
    METADATA_COLUMNS,
    LoadedMaterialData,
)

LOGGER = logging.getLogger(__name__)


NORMALIZED_DESCRIPTION_COLUMN: Final[str] = "normalized_description"

# Inference columns for the normalization stage: the original raw text plus
# the new normalized text. Ground-truth and metadata partitions are passed
# through unchanged from the loader.
NORMALIZED_INFERENCE_COLUMNS: Final[tuple[str, ...]] = (
    *INFERENCE_COLUMNS,
    NORMALIZED_DESCRIPTION_COLUMN,
)

# --------------------------------------------------------------------------
# Precompiled regexes (compiled once at import time for throughput).
# --------------------------------------------------------------------------

# Collapse any run of whitespace (space, tab, newline, non-breaking space
# after NFKC normalization, etc.) into a single ASCII space.
_WHITESPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")

# Unicode dash/minus variants (en dash, em dash, figure dash, minus sign,
# etc.) -> canonical ASCII hyphen. This is pure formatting canonicalization:
# it does not decide whether the hyphen itself is meaningful.
_UNICODE_DASH_RE: Final[re.Pattern[str]] = re.compile(
    "[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]"
)


def normalize_text(value: object) -> str | None:
    """
    Deterministically normalize a single material description.

    Safe formatting-only transformations applied, in order:
      1. Reject/short-circuit missing or non-string input (see below).
      2. Unicode NFKC normalization (canonicalizes compatibility characters,
         e.g. full-width digits/letters, without altering technical tokens).
      3. Canonicalize unicode dash variants to a plain ASCII hyphen.
      4. Collapse repeated whitespace and strip leading/trailing whitespace.
      5. Uppercase the result for a consistent casing convention.

    Deliberately NOT done here (would risk destroying technical meaning):
      - Removing or rewriting digits, units, dimensions, standards, grades.
      - Stripping punctuation wholesale (e.g. `"`, `.`, `/`).
      - Separating hyphens between letters (letter-hyphen-letter).
        See module docstring for rationale.
      - Fuzzy matching, spelling correction, or synonym collapsing.
      - Any transformation whose safety depends on domain/engineering
        judgement rather than being unconditionally format-preserving.

    Missing / invalid input handling:
      - `None`, NaN/NA-like values, non-string values, and strings that are
        empty or whitespace-only after normalization all resolve to
        `None`. We do not fabricate placeholder text for a missing
        description -- production consumers must treat `None` as "no
        description available" rather than as an empty technical claim.

    Idempotence:
      For any string that survives to a non-None result, calling this
      function again on that result returns the same string, i.e.
      `normalize_text(normalize_text(x)) == normalize_text(x)`.

    Parameters
    ----------
    value:
        The raw value from `raw_description`. Typed as `object` because
        this function must also defensively handle non-string input that
        could arrive from a loosely-validated source (e.g. direct
        dict/record usage outside the loader's dtype-enforced DataFrame).

    Returns
    -------
    The normalized, uppercase description, or `None` if no real
    description text was available.
    """
    if value is None:
        return None

    if isinstance(value, str):
        text = value
    else:
        # Handles pandas/NumPy NA scalars (float NaN, pd.NA, pd.NaT, etc.)
        # without assuming `value` is hashable/comparable in a way that
        # would make `pd.isna` unsafe to call.
        try:
            is_missing = pd.isna(value)
        except (TypeError, ValueError):
            is_missing = False

        if isinstance(is_missing, bool) and is_missing:
            return None

        LOGGER.warning(
            "normalize_text received non-string, non-missing input of "
            "type %s; treating as missing rather than fabricating text.",
            type(value).__name__,
        )
        return None

    # Apply normalization steps in order
    text = unicodedata.normalize("NFKC", text)
    text = _UNICODE_DASH_RE.sub("-", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    text = text.upper()

    return text or None


def normalize_material_record(record: Mapping[str, object]) -> dict[str, object]:
    """
    Normalize a single material record (e.g. one CSV row as a mapping).

    Only `raw_description` is read to compute `normalized_description`.
    Every other key in `record` -- including `canonical_id` and any
    `extracted_*` ground-truth annotation -- is passed through completely
    unchanged and is never inspected. This is what makes the leakage
    guarantee structural rather than merely conventional: there is no code
    path here that reads those keys.

    Parameters
    ----------
    record:
        A mapping representing one material row, expected to contain at
        least a `raw_description` key (missing is tolerated and treated as
        a missing description).

    Returns
    -------
    A new dict equal to `record` plus a `normalized_description` key.
    """
    result: dict[str, object] = dict(record)
    result[NORMALIZED_DESCRIPTION_COLUMN] = normalize_text(record.get("raw_description"))
    return result


@dataclass(frozen=True)
class NormalizedMaterialData:
    """
    Normalized material data with the same explicit feature/reference
    partitioning convention as `loader.LoadedMaterialData`.

    `frame` contains every original source column plus
    `normalized_description`. The partitions below make the safe-usage
    boundary explicit for downstream stages:

    - `inference_frame`: `raw_description` + `normalized_description`.
      This is the only data the production NER/attribute-extraction and
      embedding stages are permitted to consume.
    - `ground_truth_frame`: unchanged from the loader (`canonical_id`,
      `extracted_*`). Evaluation/training only -- never fed to inference.
    - `metadata_frame`: unchanged from the loader (business/source fields).
    """

    frame: pd.DataFrame
    inference_columns: tuple[str, ...] = NORMALIZED_INFERENCE_COLUMNS
    ground_truth_columns: tuple[str, ...] = GROUND_TRUTH_COLUMNS
    metadata_columns: tuple[str, ...] = METADATA_COLUMNS

    @property
    def inference_frame(self) -> pd.DataFrame:
        """Return only fields permitted as production inference inputs."""
        return self.frame.loc[:, self.inference_columns].copy()

    @property
    def ground_truth_frame(self) -> pd.DataFrame:
        """Return gold/reference annotations for training or evaluation."""
        return self.frame.loc[:, self.ground_truth_columns].copy()

    @property
    def metadata_frame(self) -> pd.DataFrame:
        """Return preserved source/business metadata."""
        return self.frame.loc[:, self.metadata_columns].copy()


def normalize_material_frame(loaded: LoadedMaterialData) -> NormalizedMaterialData:
    """
    Apply deterministic normalization across an entire loaded dataset.

    Only the `raw_description` column is read to compute
    `normalized_description`. `canonical_id` and every `extracted_*`
    column are carried through in `frame` unchanged (so they remain
    available for later evaluation) but are never consulted while
    computing the normalized text.

    Parameters
    ----------
    loaded:
        The validated output of `loader.load_material_file` /
        `loader.load_material_sources`.

    Returns
    -------
    `NormalizedMaterialData` wrapping the original frame plus a new
    `normalized_description` column.
    """
    frame = loaded.frame.copy()

    normalized_values = frame["raw_description"].map(normalize_text)
    frame[NORMALIZED_DESCRIPTION_COLUMN] = pd.array(
        normalized_values, dtype="string"
    )

    return NormalizedMaterialData(frame=frame)

@dataclass(frozen=True)
class NormalizedMaterialData:
    """
    Normalized material data with the same explicit feature/reference
    partitioning convention as `loader.LoadedMaterialData`.

    `frame` contains every original source column plus
    `normalized_description`. The partitions below make the safe-usage
    boundary explicit for downstream stages:

    - `inference_frame`: `raw_description` + `normalized_description`.
      This is the only data the production NER/attribute-extraction and
      embedding stages are permitted to consume.
    - `ground_truth_frame`: unchanged from the loader (`canonical_id`,
      `extracted_*`). Evaluation/training only -- never fed to inference.
    - `metadata_frame`: unchanged from the loader (business/source fields).
    """

    frame: pd.DataFrame
    inference_columns: tuple[str, ...] = NORMALIZED_INFERENCE_COLUMNS
    ground_truth_columns: tuple[str, ...] = GROUND_TRUTH_COLUMNS
    metadata_columns: tuple[str, ...] = METADATA_COLUMNS

    @property
    def inference_frame(self) -> pd.DataFrame:
        """Return only fields permitted as production inference inputs."""
        return self.frame.loc[:, self.inference_columns].copy()

    @property
    def ground_truth_frame(self) -> pd.DataFrame:
        """Return gold/reference annotations for training or evaluation."""
        return self.frame.loc[:, self.ground_truth_columns].copy()

    @property
    def metadata_frame(self) -> pd.DataFrame:
        """Return preserved source/business metadata."""
        return self.frame.loc[:, self.metadata_columns].copy()


def normalize_material_frame(loaded: LoadedMaterialData) -> NormalizedMaterialData:
    """
    Apply deterministic normalization across an entire loaded dataset.

    Only the `raw_description` column is read to compute
    `normalized_description`. `canonical_id` and every `extracted_*`
    column are carried through in `frame` unchanged (so they remain
    available for later evaluation) but are never consulted while
    computing the normalized text.

    Parameters
    ----------
    loaded:
        The validated output of `loader.load_material_file` /
        `loader.load_material_sources`.

    Returns
    -------
    `NormalizedMaterialData` wrapping the original frame plus a new
    `normalized_description` column.
    """
    frame = loaded.frame.copy()

    normalized_values = frame["raw_description"].map(normalize_text)
    frame[NORMALIZED_DESCRIPTION_COLUMN] = pd.array(
        normalized_values, dtype="string"
    )

    # Preserve the column partitioning from the incoming LoadedMaterialData.
    # Only append NORMALIZED_DESCRIPTION_COLUMN to inference_columns.
    normalized_inference_columns = (
        *loaded.inference_columns,
        NORMALIZED_DESCRIPTION_COLUMN,
    )

    return NormalizedMaterialData(
        frame=frame,
        inference_columns=normalized_inference_columns,
        ground_truth_columns=loaded.ground_truth_columns,
        metadata_columns=loaded.metadata_columns,
    )