"""Descriptive profiling of a loaded analysis dataset.

Profiling here is descriptive and deliberately non-interpretive.  It reports
counts, missingness and distributional statistics; it does not classify
day/night, judge physical validity, flag outliers, or infer data quality from
the numbers.  In particular:

* a GHI value of exactly zero is a recorded observation, not "night";
* a missing GHI value is missing, not zero;
* no observation is removed, replaced, interpolated or capped.

All returned values are native Python objects so the result can be written to
JSON without a custom encoder.  Undefined statistics (empty or all-missing
columns, a single observation's standard deviation, non-finite results) are
returned as ``None``.  Nothing is written to disk in this phase.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from forcasty.analysis.datasets import (
    AnalysisDataset,
    DatasetSchemaError,
    assert_units,
)
from forcasty.data.validation import NUMERIC_COLUMNS, format_timestamp

__all__ = [
    "describe_numeric",
    "profile_dataset",
    "summarize_ghi",
    "summarize_missing",
]

GHI_COLUMN = "ALLSKY_SFC_SW_DWN"


def _native_float(value: Any) -> float | None:
    """Return a finite Python ``float``, or ``None`` when undefined."""

    if value is None or pd.isna(value):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _percentage(part: int, total: int) -> float:
    """Return ``part`` as a percentage of ``total`` (0.0 when empty)."""

    if total <= 0:
        return 0.0
    return 100.0 * part / total


def describe_numeric(series: pd.Series) -> dict[str, Any]:
    """Return descriptive statistics for one numeric series.

    Missing values are counted and excluded from the statistics.  ``count`` is
    the number of non-missing observations.  Standard deviation uses pandas'
    sample convention (``ddof=1``); it is ``None`` when fewer than two values
    are available.  A fully missing column is reported safely even when pandas
    inferred an object dtype for it.
    """

    total = int(len(series))
    missing = int(series.isna().sum())
    values = series.dropna()

    # An all-missing column may be inferred as object; it is safe to report as
    # empty rather than raising.  Any non-missing value must be numeric.
    if not values.empty and not pd.api.types.is_numeric_dtype(series):
        raise TypeError(
            "describe_numeric requires a numeric dtype; "
            f"got {series.dtype}. Load data through load_cleaned_dataset."
        )

    if values.empty:
        mean = median = std = minimum = maximum = None
    else:
        mean = _native_float(values.mean())
        median = _native_float(values.median())
        std = _native_float(values.std())
        minimum = _native_float(values.min())
        maximum = _native_float(values.max())

    return {
        "count": int(values.size),
        "missing": missing,
        "missing_percentage": _percentage(missing, total),
        "mean": mean,
        "median": median,
        "std": std,
        "min": minimum,
        "max": maximum,
    }


def summarize_missing(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Return missing counts and percentages for every column."""

    total = int(len(frame))
    counts = frame.isna().sum()
    return {
        str(column): {
            "count": int(count),
            "percentage": _percentage(int(count), total),
        }
        for column, count in counts.items()
    }


def summarize_ghi(series: pd.Series) -> dict[str, int]:
    """Count missing, zero, positive and negative GHI observations.

    The four categories are mutually exclusive and together account for every
    row.  Zero is reported as an observed value; it is **not** relabelled as
    night, and missing is never folded into zero.

    A fully missing column is accepted even if pandas inferred an object
    dtype.  Any non-missing value must be numeric; non-numeric values raise
    :class:`DatasetSchemaError` rather than being coerced or dropped.
    """

    values = series.dropna()
    if not values.empty and not pd.api.types.is_numeric_dtype(series):
        raise DatasetSchemaError(
            "GHI values must be numeric; got dtype "
            f"{series.dtype} with non-missing values. Load data through "
            "load_cleaned_dataset."
        )

    return {
        "missing": int(len(series) - len(values)),
        "zero": int((values == 0).sum()),
        "positive": int((values > 0).sum()),
        "negative": int((values < 0).sum()),
        "non_null": int(len(values)),
    }


def _format_boundary(
    value: pd.Timestamp | None, frequency: str
) -> str | None:
    if value is None:
        return None
    return format_timestamp(value, frequency)


def profile_dataset(dataset: AnalysisDataset) -> dict[str, Any]:
    """Return a JSON-serialisable descriptive profile of ``dataset``.

    The profile contains dataset metadata, the timestamp range, missing counts
    per column, non-null descriptive statistics per numeric variable, and the
    GHI category counts.  It contains no temporal aggregation, no day/night
    classification and no quality judgement.
    """

    assert_units(dataset)
    frame = dataset.frame

    if GHI_COLUMN not in frame.columns:
        raise DatasetSchemaError(
            f"frame is missing the GHI column {GHI_COLUMN!r}."
        )

    return {
        "site_id": dataset.site.site_id,
        "site_name": dataset.site.name,
        "frequency": dataset.frequency,
        "time_standard": dataset.time_standard,
        "ghi_units": dataset.ghi_units,
        "timestamp_column": dataset.timestamp_column,
        "row_count": dataset.row_count,
        "column_count": int(len(frame.columns)),
        "columns": [str(column) for column in frame.columns],
        "first_timestamp": _format_boundary(
            dataset.first_timestamp, dataset.frequency
        ),
        "last_timestamp": _format_boundary(
            dataset.last_timestamp, dataset.frequency
        ),
        "missing_by_column": summarize_missing(frame),
        "descriptive_statistics": {
            str(column): describe_numeric(frame[column])
            for column in frame.columns
            if column in NUMERIC_COLUMNS
            or pd.api.types.is_numeric_dtype(frame[column])
        },
        "ghi_counts": summarize_ghi(frame[GHI_COLUMN]),
    }
