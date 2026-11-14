"""Descriptive temporal aggregation of a loaded analysis dataset.

Phase 2 groups GHI by calendar components, using each dataset's own temporal
convention and nothing else:

* daily  -- LST calendar dates: calendar year, calendar month and year-month;
* hourly -- UTC start-of-hour timestamps: hour of day, month and year-month.

Daily LST dates and hourly UTC dates are never combined or treated as the same
calendar day, and the hourly index is never converted to local time.

Missing GHI stays missing.  It is counted per group but excluded from every
descriptive statistic; it is never replaced by zero.  A zero GHI value is an
observed value, not evidence of nighttime.  Groups with no valid observation
report ``None`` for all statistics, and a group with one valid observation has
no sample standard deviation.

The functions are read-only and return JSON-native dictionaries.  They do not
write files, interpolate, fill, sort, convert units or classify day/night.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from forcasty.analysis.datasets import (
    AnalysisDataset,
    DatasetSchemaError,
    UnsupportedFrequencyError,
    assert_units,
)
from forcasty.data.validation import DAILY, HOURLY

__all__ = [
    "analyze_daily_temporal",
    "analyze_hourly_temporal",
]

# Matches ``profiling.GHI_COLUMN`` and the first entry of the validation
# module's ``NUMERIC_COLUMNS``.
GHI_COLUMN = "ALLSKY_SFC_SW_DWN"


def _native_float(value: Any) -> float | None:
    """Return a finite Python ``float``, or ``None`` when undefined."""

    if value is None or pd.isna(value):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _summarize(values: pd.Series) -> dict[str, Any]:
    """Return total, valid, missing and descriptive statistics for a group.

    ``total`` counts every row, ``valid`` counts non-missing rows, and the
    statistics use only non-missing values.  Sample standard deviation uses
    ``ddof=1`` and is ``None`` when fewer than two valid values exist.
    """

    total = int(len(values))
    present = values.dropna()
    valid = int(len(present))

    if present.empty:
        mean = median = std = minimum = maximum = None
    else:
        mean = _native_float(present.mean())
        median = _native_float(present.median())
        std = _native_float(present.std())
        minimum = _native_float(present.min())
        maximum = _native_float(present.max())

    return {
        "total": total,
        "valid": valid,
        "missing": total - valid,
        "mean": mean,
        "median": median,
        "std": std,
        "min": minimum,
        "max": maximum,
    }


def _key_value(key: Any) -> int | str:
    """Return a group key as a native JSON scalar."""

    if isinstance(key, str):
        return str(key)
    return int(key)


def _group_records(
    keys: pd.Index, values: pd.Series
) -> list[dict[str, Any]]:
    """Return one summary record per distinct key, ordered by key.

    ``keys`` comes from ``dataset.time_index`` and is positionally aligned
    with ``values``.  No group is invented and no missing interval is filled.
    """

    grouped = pd.DataFrame(
        {"key": list(keys), "value": values.reset_index(drop=True)}
    )
    records: list[dict[str, Any]] = []
    for key, chunk in grouped.groupby("key", sort=True):
        record: dict[str, Any] = {"key": _key_value(key)}
        record.update(_summarize(chunk["value"]))
        records.append(record)
    return records


def _ghi_series(dataset: AnalysisDataset) -> pd.Series:
    """Return the frame's GHI column after checking it is usable."""

    if GHI_COLUMN not in dataset.frame.columns:
        raise DatasetSchemaError(
            f"frame is missing the GHI column {GHI_COLUMN!r}."
        )

    series = dataset.frame[GHI_COLUMN]
    if not series.dropna().empty and not pd.api.types.is_numeric_dtype(series):
        raise DatasetSchemaError(
            "GHI values must be numeric; got dtype "
            f"{series.dtype} with non-missing values. Load data through "
            "load_cleaned_dataset."
        )
    return series


def _validate(dataset: AnalysisDataset, frequency: str) -> pd.Series:
    """Validate the dataset for temporal analysis and return GHI values."""

    assert_units(dataset)
    if dataset.frequency != frequency:
        raise UnsupportedFrequencyError(
            f"{frequency} temporal analysis requires a {frequency} dataset; "
            f"got {dataset.frequency!r}. Daily LST and hourly UTC data are "
            "not interchangeable."
        )
    return _ghi_series(dataset)


def _base_result(dataset: AnalysisDataset) -> dict[str, Any]:
    """Return the metadata shared by both temporal analyses."""

    return {
        "site_id": dataset.site.site_id,
        "site_name": dataset.site.name,
        "frequency": dataset.frequency,
        "time_standard": dataset.time_standard,
        "ghi_units": dataset.ghi_units,
        "ghi_column": GHI_COLUMN,
        "row_count": dataset.row_count,
    }


def analyze_daily_temporal(dataset: AnalysisDataset) -> dict[str, Any]:
    """Group a daily LST dataset by year, month and year-month.

    The grouping keys come from the timezone-naive LST calendar dates in
    ``dataset.time_index``.  ``by_year`` and ``by_month`` keys are integers;
    ``by_year_month`` keys are ``"YYYY-MM"`` strings.  Each group reports
    ``total``, ``valid``, ``missing``, ``mean``, ``median``, ``std``, ``min``
    and ``max``.  Incomplete groups keep their valid count visible instead of
    being masked by an average.
    """

    values = _validate(dataset, DAILY)
    index = dataset.time_index

    result = _base_result(dataset)
    result["by_year"] = _group_records(index.year, values)
    result["by_month"] = _group_records(index.month, values)
    result["by_year_month"] = _group_records(
        index.strftime("%Y-%m"), values
    )
    return result


def analyze_hourly_temporal(dataset: AnalysisDataset) -> dict[str, Any]:
    """Group an hourly UTC dataset by hour of day, month and year-month.

    The grouping keys come from the timezone-aware UTC timestamps in
    ``dataset.time_index`` and are never converted to local time.
    ``by_utc_hour`` keys are integers 0-23; ``by_month`` keys are integers
    1-12; ``by_year_month`` keys are ``"YYYY-MM"`` strings.  Each group
    reports ``total``, ``valid``, ``missing``, ``mean``, ``median``, ``std``,
    ``min`` and ``max``.
    """

    values = _validate(dataset, HOURLY)
    index = dataset.time_index

    result = _base_result(dataset)
    result["by_utc_hour"] = _group_records(index.hour, values)
    result["by_month"] = _group_records(index.month, values)
    result["by_year_month"] = _group_records(
        index.strftime("%Y-%m"), values
    )
    return result
