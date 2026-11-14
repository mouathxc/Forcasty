"""Loading of canonical cleaned datasets for exploratory analysis.

The loader is intentionally small and strict.  It reads one already-cleaned
CSV produced by :mod:`forcasty.data.cleaning` and attaches the scientific
metadata that later analysis phases must not guess:

* daily  -- LST, timezone-naive calendar dates, GHI in ``kW-hr/m^2/day``;
* hourly -- UTC, timezone-aware start-of-hour timestamps, GHI in ``Wh/m^2``.

The returned frame keeps the file's columns and values exactly as read.  The
parsed time axis lives beside it in :attr:`AnalysisDataset.time_index`, so no
source column is silently rewritten.  The loader never writes files, sorts
rows, fills gaps, converts units, or changes time conventions.  Structurally
invalid files are rejected with an actionable exception instead of repaired.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timezone
from pathlib import Path

import pandas as pd

from forcasty.config import SITE_ID_PATTERN, Site
from forcasty.data.validation import (
    DAILY,
    DAILY_REQUIRED_COLUMNS,
    FREQUENCIES,
    GHI_UNITS,
    HOURLY,
    HOURLY_REQUIRED_COLUMNS,
    NUMERIC_COLUMNS,
    TIME_STANDARD,
    TIMESTAMP_COLUMN,
)
from forcasty.paths import PATHS, ProjectPaths

__all__ = [
    "AnalysisDataError",
    "AnalysisDataset",
    "DatasetNotFoundError",
    "DatasetSchemaError",
    "TemporalConventionError",
    "UnsupportedFrequencyError",
    "assert_units",
    "load_cleaned_dataset",
]

# Canonical forms written by M2 cleaning.  Parsing with an explicit format
# rejects silently coerced dates and non-UTC hourly offsets.
TIMESTAMP_FORMAT = {
    DAILY: "%Y-%m-%d",
    HOURLY: "%Y-%m-%dT%H:%M:%SZ",
}

TIMESTAMP_EXAMPLE = {
    DAILY: "2020-01-01",
    HOURLY: "2020-01-01T00:00:00Z",
}


class AnalysisDataError(ValueError):
    """Base class for dataset-loading failures in the analysis package."""


class UnsupportedFrequencyError(AnalysisDataError):
    """Raised when a frequency other than daily or hourly is requested."""


class DatasetNotFoundError(AnalysisDataError, FileNotFoundError):
    """Raised when the expected cleaned dataset file does not exist."""


class DatasetSchemaError(AnalysisDataError):
    """Raised when a cleaned dataset does not match the expected schema."""


class TemporalConventionError(DatasetSchemaError):
    """Raised when a dataset violates its frequency's temporal contract."""


@dataclass(frozen=True, eq=False)
class AnalysisDataset:
    """One loaded dataset plus the metadata its analysis must respect.

    ``frame`` is the CSV content exactly as read; callers must treat it as
    read-only.  ``time_index`` is the parsed axis, positionally aligned with
    ``frame``.  Daily indexes are timezone-naive LST dates; hourly indexes are
    timezone-aware UTC timestamps at the start of each hour.
    """

    site: Site
    frequency: str
    frame: pd.DataFrame
    time_index: pd.DatetimeIndex
    timestamp_column: str
    time_standard: str
    ghi_units: str
    source_path: Path

    @property
    def row_count(self) -> int:
        """Number of rows in the loaded frame."""

        return int(len(self.frame))

    @property
    def columns(self) -> tuple[str, ...]:
        """Column names in file order."""

        return tuple(self.frame.columns)

    @property
    def first_timestamp(self) -> pd.Timestamp | None:
        """First parsed timestamp, or ``None`` for an empty dataset."""

        return None if len(self.time_index) == 0 else self.time_index[0]

    @property
    def last_timestamp(self) -> pd.Timestamp | None:
        """Last parsed timestamp, or ``None`` for an empty dataset."""

        return None if len(self.time_index) == 0 else self.time_index[-1]


def _assert_timestamp_correspondence(
    dataset: AnalysisDataset, frequency: str
) -> None:
    """Check that ``time_index`` matches the frame's canonical timestamps.

    The frame column is re-parsed with the same explicit format used by the
    loader.  Malformed values raise instead of being coerced to ``NaT``.
    """

    try:
        reparsed = pd.to_datetime(
            dataset.frame[dataset.timestamp_column],
            format=TIMESTAMP_FORMAT[frequency],
            errors="raise",
            utc=(frequency == HOURLY),
        )
    except (TypeError, ValueError) as exc:
        raise TemporalConventionError(
            f"{frequency} column {dataset.timestamp_column!r} contains "
            "values that do not match the canonical format "
            f"{TIMESTAMP_FORMAT[frequency]!r}; the time index cannot be "
            "verified against it."
        ) from exc

    parsed_index = pd.DatetimeIndex(reparsed)
    differences = parsed_index != dataset.time_index
    if bool(differences.any()):
        first = int(differences.argmax())
        raise TemporalConventionError(
            f"{frequency} time_index does not correspond to column "
            f"{dataset.timestamp_column!r} at position {first}: "
            f"index={dataset.time_index[first]!r}, "
            f"frame={parsed_index[first]!r}."
        )


def assert_units(dataset: AnalysisDataset) -> None:
    """Enforce frequency-specific units and temporal conventions.

    Raises
    ------
    UnsupportedFrequencyError
        If ``dataset.frequency`` is not ``daily`` or ``hourly``.
    DatasetSchemaError
        If the GHI unit, column names or frame/time-axis lengths disagree with
        the declared frequency.
    TemporalConventionError
        If daily times are timezone-aware or not exactly midnight, hourly
        times are naive, not explicitly UTC, or not at the start of an hour,
        the time index does not correspond to the frame's timestamp column,
        or timestamps are duplicated or unsorted.

    The function only checks; it never sorts, converts, repairs or mutates
    anything.
    """

    frequency = dataset.frequency
    if frequency not in FREQUENCIES:
        raise UnsupportedFrequencyError(
            f"frequency must be one of {FREQUENCIES}; got {frequency!r}."
        )

    if dataset.ghi_units != GHI_UNITS[frequency]:
        raise DatasetSchemaError(
            f"{frequency} GHI units must be {GHI_UNITS[frequency]!r}; "
            f"got {dataset.ghi_units!r}."
        )

    if dataset.time_standard != TIME_STANDARD[frequency]:
        raise TemporalConventionError(
            f"{frequency} data must use {TIME_STANDARD[frequency]!r}; "
            f"got {dataset.time_standard!r}."
        )

    if dataset.timestamp_column != TIMESTAMP_COLUMN[frequency]:
        raise DatasetSchemaError(
            f"{frequency} data must use the timestamp column "
            f"{TIMESTAMP_COLUMN[frequency]!r}; got "
            f"{dataset.timestamp_column!r}."
        )

    if dataset.timestamp_column not in dataset.frame.columns:
        raise DatasetSchemaError(
            f"frame is missing the timestamp column "
            f"{dataset.timestamp_column!r}."
        )

    if len(dataset.time_index) != len(dataset.frame):
        raise DatasetSchemaError(
            "time_index and frame have different lengths: "
            f"{len(dataset.time_index)} != {len(dataset.frame)}."
        )

    index = dataset.time_index
    if not isinstance(index, pd.DatetimeIndex):
        raise TemporalConventionError(
            "time_index must be a pandas DatetimeIndex; got "
            f"{type(index).__name__}."
        )

    if frequency == DAILY:
        if index.tz is not None:
            raise TemporalConventionError(
                "daily data are LST calendar dates and their time index must "
                f"be timezone-naive; got tz={index.tz}."
            )
        not_midnight = (
            (index.hour != 0)
            | (index.minute != 0)
            | (index.second != 0)
            | (index.microsecond != 0)
            | (index.nanosecond != 0)
        )
        if bool(not_midnight.any()):
            raise TemporalConventionError(
                "daily timestamps must be exactly midnight (00:00:00); "
                f"{int(not_midnight.sum())} timestamp(s) are not."
            )
    else:
        if index.tz is None or (
            index.tz is not timezone.utc and str(index.tz) != "UTC"
        ):
            raise TemporalConventionError(
                "hourly data are UTC and their time index must be "
                f"timezone-aware and explicitly UTC; got tz={index.tz!r}."
            )
        off_hour = (
            (index.minute != 0)
            | (index.second != 0)
            | (index.microsecond != 0)
            | (index.nanosecond != 0)
        )
        if bool(off_hour.any()):
            raise TemporalConventionError(
                "hourly timestamps must be at the start of the hour; "
                f"{int(off_hour.sum())} timestamp(s) are not."
            )

    _assert_timestamp_correspondence(dataset, frequency)

    if index.has_duplicates:
        raise TemporalConventionError(
            f"{int(index.duplicated().sum())} duplicate timestamp(s) found; "
            "the loader does not discard records."
        )
    if not index.is_monotonic_increasing:
        raise TemporalConventionError(
            "timestamps are not strictly increasing; the loader does not "
            "sort records."
        )


def _validate_site(site: Site) -> None:
    """Reject sites whose id could escape the expected file name."""

    if not isinstance(site, Site):
        raise TypeError(
            "site must be a forcasty.config.Site instance; got "
            f"{type(site).__name__}."
        )
    if not SITE_ID_PATTERN.fullmatch(site.site_id):
        raise AnalysisDataError(
            f"site_id must match {SITE_ID_PATTERN.pattern!r}; "
            f"got {site.site_id!r}."
        )


def _cleaned_dataset_path(
    site: Site, frequency: str, paths: ProjectPaths
) -> Path:
    if frequency == HOURLY:
        return paths.hourly_clean_dataset_path(site.site_id)
    return paths.clean_dataset_path(site.site_id)


def load_cleaned_dataset(
    site: Site,
    frequency: str,
    *,
    paths: ProjectPaths = PATHS,
) -> AnalysisDataset:
    """Load one canonical cleaned dataset for ``site``.

    Parameters
    ----------
    site:
        Validated site from :func:`forcasty.config.load_config`.
    frequency:
        ``"daily"`` or ``"hourly"``.
    paths:
        Injectable project layout.  Defaults to the repository ``PATHS``.

    Returns
    -------
    AnalysisDataset
        The unmodified frame, its parsed time axis, and its scientific
        metadata.

    Raises
    ------
    UnsupportedFrequencyError
        If ``frequency`` is not supported.
    TypeError, AnalysisDataError
        If ``site`` is not a ``Site`` or has an unsafe id.
    DatasetNotFoundError
        If the cleaned file is absent.
    DatasetSchemaError, TemporalConventionError
        If the file is empty, lacks required columns, has non-numeric
        variables, uses non-canonical timestamps, or violates its temporal
        convention.
    """

    if frequency not in FREQUENCIES:
        raise UnsupportedFrequencyError(
            f"frequency must be one of {FREQUENCIES}; got {frequency!r}."
        )

    _validate_site(site)
    source_path = _cleaned_dataset_path(site, frequency, paths)

    # Defense in depth: the legacy data path helpers interpolate the site id
    # verbatim, so re-check that the result still sits directly in processed/.
    if source_path.parent != paths.processed_dir:
        raise AnalysisDataError(
            f"site_id {site.site_id!r} does not resolve to a file directly "
            f"inside {paths.processed_dir}."
        )

    if not source_path.is_file():
        command = (
            f"python -m forcasty.data.cleaning {frequency} "
            f"--site {site.site_id}"
        )
        raise DatasetNotFoundError(
            f"cleaned {frequency} dataset not found: {source_path}. "
            f"Generate it first with: {command}"
        )

    try:
        frame = pd.read_csv(source_path)
    except pd.errors.EmptyDataError as exc:
        raise DatasetSchemaError(
            f"{source_path} is empty; expected a header and at least one row."
        ) from exc
    except OSError as exc:
        raise AnalysisDataError(
            f"could not read cleaned dataset {source_path}: {exc}"
        ) from exc

    required = (
        DAILY_REQUIRED_COLUMNS
        if frequency == DAILY
        else HOURLY_REQUIRED_COLUMNS
    )
    missing_columns = [
        column for column in required if column not in frame.columns
    ]
    if missing_columns:
        raise DatasetSchemaError(
            f"{source_path.name} is missing required column(s): "
            f"{missing_columns}."
        )

    if frame.empty:
        raise DatasetSchemaError(
            f"{source_path.name} contains no rows; an empty cleaned dataset "
            "cannot be analysed."
        )

    non_numeric = [
        f"{column} ({frame[column].dtype})"
        for column in NUMERIC_COLUMNS
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        raise DatasetSchemaError(
            f"{source_path.name} has non-numeric value column(s): "
            f"{non_numeric}. Re-run canonical cleaning before analysis."
        )

    timestamp_column = TIMESTAMP_COLUMN[frequency]
    try:
        parsed = pd.to_datetime(
            frame[timestamp_column],
            format=TIMESTAMP_FORMAT[frequency],
            errors="raise",
            utc=(frequency == HOURLY),
        )
    except (TypeError, ValueError) as exc:
        raise DatasetSchemaError(
            f"{frequency} column {timestamp_column!r} must use the canonical "
            f"format {TIMESTAMP_FORMAT[frequency]!r}, for example "
            f"{TIMESTAMP_EXAMPLE[frequency]!r}."
        ) from exc

    dataset = AnalysisDataset(
        site=site,
        frequency=frequency,
        frame=frame,
        time_index=pd.DatetimeIndex(parsed),
        timestamp_column=timestamp_column,
        time_standard=TIME_STANDARD[frequency],
        ghi_units=GHI_UNITS[frequency],
        source_path=source_path,
    )
    assert_units(dataset)
    return dataset
