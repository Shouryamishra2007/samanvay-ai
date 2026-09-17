"""
Schema-aware ingestion for Samanvay-AI material-master datasets.

Design goals:
- Preserve every source column.
- Validate the observed CPSE schema before downstream processing.
- Keep inference inputs separate from ground-truth/evaluation annotations.
- Never use canonical_id or extracted_* annotations as production inference features.
- Fail loudly on malformed input instead of silently dropping rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import logging
import pandas as pd


LOGGER = logging.getLogger(__name__)


SOURCE_COLUMNS: Final[tuple[str, ...]] = (
    "local_code",
    "cpse",
    "raw_description",
    "canonical_id",
    "depot_id",
    "depot_location",
    "quantity",
    "unit_cost_inr",
    "idle_days",
    "extracted_item_type",
    "extracted_size_nb_mm",
    "extracted_pressure_class",
    "extracted_metallurgy",
    "extracted_facing",
    "extracted_standard",
)

# Production inference is deliberately limited to information available
# before the NER/attribute-extraction stage. Technical attributes are
# re-derived later from raw_description.
INFERENCE_COLUMNS: Final[tuple[str, ...]] = (
    "raw_description",
)

# Gold/reference annotations. These may be used for training/evaluation,
# but must never be included in production inference features.
GROUND_TRUTH_COLUMNS: Final[tuple[str, ...]] = (
    "canonical_id",
    "extracted_item_type",
    "extracted_size_nb_mm",
    "extracted_pressure_class",
    "extracted_metallurgy",
    "extracted_facing",
    "extracted_standard",
)

# Preserved source/business metadata that is not part of the semantic
# inference feature set.
METADATA_COLUMNS: Final[tuple[str, ...]] = (
    "local_code",
    "cpse",
    "depot_id",
    "depot_location",
    "quantity",
    "unit_cost_inr",
    "idle_days",
)

EXPECTED_DTYPES: Final[dict[str, str]] = {
    "local_code": "string",
    "cpse": "string",
    "raw_description": "string",
    "canonical_id": "string",
    "depot_id": "string",
    "depot_location": "string",
    "quantity": "Int64",
    "unit_cost_inr": "Int64",
    "idle_days": "Int64",
    "extracted_item_type": "string",
    "extracted_size_nb_mm": "Float64",
    "extracted_pressure_class": "Int64",
    "extracted_metallurgy": "string",
    "extracted_facing": "string",
    "extracted_standard": "string",
}


class DataContractError(ValueError):
    """Raised when an input file violates the material data contract."""


@dataclass(frozen=True)
class LoadedMaterialData:
    """Validated source data with explicit feature/reference partitions."""

    frame: pd.DataFrame
    inference_columns: tuple[str, ...] = INFERENCE_COLUMNS
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


def _read_csv(path: Path) -> pd.DataFrame:
    """Read CSV without silently skipping malformed records."""
    encodings = ("utf-8-sig", "utf-8", "cp1252")
    last_error: UnicodeDecodeError | None = None

    for encoding in encodings:
        try:
            LOGGER.info("Reading %s with encoding=%s", path, encoding)
            return pd.read_csv(
                path,
                encoding=encoding,
                dtype=EXPECTED_DTYPES,
                on_bad_lines="error",
                low_memory=False,
            )
        except UnicodeDecodeError as exc:
            last_error = exc

    raise DataContractError(
        f"Could not decode {path} using supported encodings: {encodings}"
    ) from last_error


def _read_excel(path: Path) -> pd.DataFrame:
    """Read Excel through the same validation path as CSV."""
    try:
        return pd.read_excel(path, dtype=EXPECTED_DTYPES)
    except ImportError as exc:
        raise DataContractError(
            "Excel input requires an installed Excel engine (typically openpyxl)."
        ) from exc


def _read_source(path: Path) -> pd.DataFrame:
    """Dispatch by file extension while keeping validation centralized."""
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return _read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return _read_excel(path)

    raise DataContractError(
        f"Unsupported file type '{suffix}' for {path}. "
        "Expected .csv, .xlsx, or .xls."
    )


def _validate_schema(df: pd.DataFrame, path: Path) -> None:
    """Validate columns and required field properties without altering data."""
    actual = tuple(df.columns)

    missing = [column for column in SOURCE_COLUMNS if column not in df.columns]
    unexpected = [column for column in df.columns if column not in SOURCE_COLUMNS]

    if missing:
        raise DataContractError(
            f"{path}: missing required columns: {missing}"
        )

    if unexpected:
        raise DataContractError(
            f"{path}: unexpected columns found: {unexpected}. "
            "The loader refuses unknown columns so schema drift cannot pass silently."
        )

    if actual != SOURCE_COLUMNS:
        LOGGER.warning(
            "%s: columns are valid but ordering differs from the data contract.",
            path,
        )

    if len(df) == 0:
        raise DataContractError(f"{path}: file contains zero data rows.")

    null_counts = df.loc[:, SOURCE_COLUMNS].isna().sum()
    null_fields = null_counts[null_counts > 0].to_dict()
    if null_fields:
        raise DataContractError(
            f"{path}: missing values found in required source fields: {null_fields}"
        )

    if df["local_code"].duplicated().any():
        duplicates = int(df["local_code"].duplicated().sum())
        raise DataContractError(
            f"{path}: {duplicates} duplicate local_code values detected "
            "within one CPSE source file."
        )

    observed_cpse = set(df["cpse"].astype("string").str.upper().unique())
    if len(observed_cpse) != 1:
        raise DataContractError(
            f"{path}: expected exactly one CPSE value, found {sorted(observed_cpse)}"
        )


def _validate_cpse(df: pd.DataFrame, expected_cpse: str | None, path: Path) -> None:
    """Validate source CPSE when the caller supplies the expected identity."""
    if expected_cpse is None:
        return

    observed = str(df["cpse"].iloc[0]).strip().upper()
    expected = expected_cpse.strip().upper()

    if observed != expected:
        raise DataContractError(
            f"{path}: expected CPSE '{expected}', but file contains '{observed}'."
        )


def load_material_file(
    path: str | Path,
    *,
    expected_cpse: str | None = None,
) -> LoadedMaterialData:
    """
    Load and validate one CPSE material file.

    All source columns are preserved. Only raw_description is exposed through
    inference_frame. canonical_id and extracted_* are explicitly isolated in
    ground_truth_frame and must not be used by production inference.
    """
    source_path = Path(path)

    if not source_path.is_file():
        raise FileNotFoundError(f"Material source file not found: {source_path}")

    df = _read_source(source_path)

    _validate_schema(df, source_path)
    _validate_cpse(df, expected_cpse, source_path)

    # Reorder only to the verified contract order; no source information is lost.
    df = df.loc[:, SOURCE_COLUMNS].copy()

    LOGGER.info(
        "Loaded %d rows from %s (CPSE=%s)",
        len(df),
        source_path.name,
        df["cpse"].iloc[0],
    )

    return LoadedMaterialData(frame=df)


def load_material_sources(
    sources: dict[str, str | Path],
) -> LoadedMaterialData:
    """
    Load multiple CPSE sources and concatenate them after per-file validation.

    Example:
        load_material_sources({
            "ONGC": "data/raw/ongc_materials.csv",
            "IOCL": "data/raw/iocl_materials.csv",
            "BPCL": "data/raw/bpcl_materials.csv",
        })

    Every source is validated independently before concatenation.
    """
    if not sources:
        raise ValueError("At least one material source is required.")

    frames: list[pd.DataFrame] = []

    for expected_cpse, path in sources.items():
        loaded = load_material_file(path, expected_cpse=expected_cpse)
        frames.append(loaded.frame)

    combined = pd.concat(frames, axis=0, ignore_index=True)

    if combined["local_code"].duplicated().any():
        # local_code is only required to be unique within a CPSE. This check
        # therefore uses the composite source identity.
        duplicate_pairs = combined.duplicated(
            subset=["cpse", "local_code"]
        )
        if duplicate_pairs.any():
            count = int(duplicate_pairs.sum())
            raise DataContractError(
                f"Combined sources contain {count} duplicate (cpse, local_code) keys."
            )

    return LoadedMaterialData(frame=combined)
