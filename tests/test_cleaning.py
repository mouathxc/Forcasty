"""Offline tests for canonical cleaning, reports and file processing."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from forcasty.config import Site
from forcasty.data.cleaning import (
    build_processing_report,
    clean_daily,
    clean_dataset,
    clean_hourly,
    process_dataset,
)
from forcasty.data.metadata import sha256_file
from forcasty.data.validation import ValidationError, validate_daily
from forcasty.paths import PATHS, ProjectPaths

SITE = Site(
    site_id="tunis",
    name="Tunis",
    latitude=36.8065,
    longitude=10.1815,
)
COLUMNS = ["ALLSKY_SFC_SW_DWN", "T2M", "RH2M", "WS2M"]
FIXED_NOW = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def daily_frame(rows: int = 5) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=rows, freq="D")
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "ALLSKY_SFC_SW_DWN": [1.0 + index for index in range(rows)],
            "T2M": [10.0 + index for index in range(rows)],
            "RH2M": [60.0] * rows,
            "WS2M": [2.0] * rows,
        }
    )


def hourly_frame(hours: int = 6) -> pd.DataFrame:
    stamps = pd.date_range("2020-01-01T00:00:00Z", periods=hours, freq="h")
    return pd.DataFrame(
        {
            "timestamp": stamps.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ALLSKY_SFC_SW_DWN": [index * 100.0 for index in range(hours)],
            "T2M": [10.0 + index for index in range(hours)],
            "RH2M": [60.0] * hours,
            "WS2M": [2.0] * hours,
        }
    )


def test_clean_replaces_fill_but_never_interpolates() -> None:
    frame = daily_frame()
    frame.loc[2, "ALLSKY_SFC_SW_DWN"] = -999

    result = clean_daily(frame)
    cleaned = result.frame

    assert pd.isna(cleaned.loc[2, "ALLSKY_SFC_SW_DWN"])
    # Neighbouring observations are untouched; no interpolation occurred.
    assert cleaned.loc[1, "ALLSKY_SFC_SW_DWN"] == frame.loc[1, "ALLSKY_SFC_SW_DWN"]
    assert cleaned.loc[3, "ALLSKY_SFC_SW_DWN"] == frame.loc[3, "ALLSKY_SFC_SW_DWN"]
    assert result.operations[2]["counts"]["ALLSKY_SFC_SW_DWN"] == 1
    assert result.operations[4]["interpolation"] is False


def test_clean_does_not_mutate_the_input_frame() -> None:
    frame = daily_frame()
    frame.loc[1, "ALLSKY_SFC_SW_DWN"] = -999
    original = frame.copy(deep=True)

    clean_daily(frame)

    pd.testing.assert_frame_equal(frame, original)


def test_clean_is_deterministic_and_idempotent() -> None:
    frame = hourly_frame()
    frame.loc[2, "T2M"] = -999

    first = clean_hourly(frame).frame
    second = clean_hourly(frame).frame
    third = clean_hourly(first).frame

    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(first, third)


def test_clean_sorts_chronologically_without_dropping_rows() -> None:
    frame = daily_frame().iloc[[3, 0, 4, 1, 2]].reset_index(drop=True)

    result = clean_daily(frame)

    assert result.frame["date"].tolist() == [
        "2020-01-01",
        "2020-01-02",
        "2020-01-03",
        "2020-01-04",
        "2020-01-05",
    ]
    assert len(result.frame) == len(frame)
    assert result.operations[3]["changed"] is True


def test_clean_rejects_duplicate_timestamps() -> None:
    frame = daily_frame()
    frame.loc[2, "date"] = frame.loc[1, "date"]

    with pytest.raises(ValidationError, match="DUPLICATE_TIMESTAMPS"):
        clean_daily(frame)


def test_clean_rejects_invalid_timestamps() -> None:
    frame = daily_frame()
    frame.loc[0, "date"] = "yesterday"

    with pytest.raises(ValidationError, match="INVALID_TIMESTAMPS"):
        clean_daily(frame)


def test_clean_rejects_invalid_numeric_strings() -> None:
    frame = hourly_frame()
    frame["WS2M"] = frame["WS2M"].astype(object)
    frame.loc[1, "WS2M"] = "breeze"

    with pytest.raises(ValidationError, match="INVALID_NUMERIC_VALUES"):
        clean_hourly(frame)


def test_clean_preserves_physical_anomalies_and_extra_columns() -> None:
    frame = daily_frame()
    frame["station_note"] = ["a", "b", "c", "d", "e"]
    frame.loc[0, "ALLSKY_SFC_SW_DWN"] = -1.0
    frame.loc[1, "RH2M"] = 101.0

    result = clean_daily(frame)

    assert result.frame.loc[0, "ALLSKY_SFC_SW_DWN"] == -1.0
    assert result.frame.loc[1, "RH2M"] == 101.0
    assert "station_note" in result.frame.columns
    assert len(result.frame) == len(frame)


def test_daily_and_hourly_timestamp_formats_are_canonical() -> None:
    daily = clean_daily(daily_frame()).frame
    hourly = clean_hourly(hourly_frame()).frame

    assert daily["date"].tolist()[0] == "2020-01-01"
    assert hourly["timestamp"].tolist()[0] == "2020-01-01T00:00:00Z"
    assert hourly.loc[0, "timestamp"].endswith("Z")


def test_missing_values_survive_cleaning() -> None:
    frame = daily_frame()
    frame.loc[1, "T2M"] = None
    frame.loc[2, "RH2M"] = -999

    cleaned = clean_daily(frame).frame

    assert pd.isna(cleaned.loc[1, "T2M"])
    assert pd.isna(cleaned.loc[2, "RH2M"])
    assert int(cleaned["T2M"].isna().sum()) == 1


def test_clean_dataset_rejects_unknown_frequency() -> None:
    with pytest.raises(ValueError, match="frequency"):
        clean_dataset(daily_frame(), "monthly")


def test_process_dataset_writes_output_and_report_without_touching_raw(
    tmp_path: Path,
) -> None:
    paths = ProjectPaths(tmp_path)
    raw_path = paths.raw_dataset_path(SITE.site_id)
    raw_path.parent.mkdir(parents=True)
    frame = daily_frame()
    frame.loc[3, "ALLSKY_SFC_SW_DWN"] = -999
    frame.to_csv(raw_path, index=False)
    raw_hash = sha256_file(raw_path)

    processed = process_dataset(SITE, "daily", paths=paths, now=FIXED_NOW)

    assert sha256_file(raw_path) == raw_hash
    assert processed.output_path == paths.clean_dataset_path(SITE.site_id)
    assert processed.output_path.is_file()
    assert processed.report_path == paths.processing_report_path(
        SITE.site_id, "daily"
    )
    assert processed.report_path.is_file()

    cleaned = pd.read_csv(processed.output_path)
    assert pd.isna(cleaned.loc[3, "ALLSKY_SFC_SW_DWN"])
    assert len(cleaned) == len(frame)

    report = json.loads(processed.report_path.read_text(encoding="utf-8"))
    assert report["site_id"] == "tunis"
    assert report["site_name"] == "Tunis"
    assert report["frequency"] == "daily"
    assert report["timestamp_convention"] == "LST"
    assert report["ghi_units"] == "kW-hr/m^2/day"
    assert report["input_row_count"] == 5
    assert report["output_row_count"] == 5
    assert report["timestamp_range"] == {
        "first": "2020-01-01",
        "last": "2020-01-05",
    }
    assert report["fill_value_counts_by_column"]["ALLSKY_SFC_SW_DWN"] == 1
    assert report["missing_values_by_column"]["ALLSKY_SFC_SW_DWN"] == 1
    assert report["duplicate_timestamp_count"] == 0
    assert report["invalid_timestamp_count"] == 0
    assert report["missing_interval_count"] == 0
    assert report["irregular_spacing_count"] == 0
    assert report["unsorted_timestamps"] is False
    assert report["source_sha256"] == raw_hash
    assert report["output_sha256"] == sha256_file(processed.output_path)
    assert report["processing_timestamp_utc"] == "2026-01-01T12:00:00Z"
    assert report["software_version"]
    assert "validation_errors" in report
    assert isinstance(report["cleaning_operations"], list)


def test_process_hourly_dataset_and_report(tmp_path: Path) -> None:
    paths = ProjectPaths(tmp_path)
    raw_path = paths.hourly_dataset_path(SITE.site_id)
    raw_path.parent.mkdir(parents=True)
    frame = hourly_frame().drop(index=2).reset_index(drop=True)
    frame.to_csv(raw_path, index=False)

    processed = process_dataset(SITE, "hourly", paths=paths, now=FIXED_NOW)

    assert processed.output_path == paths.hourly_clean_dataset_path(
        SITE.site_id
    )
    report = json.loads(processed.report_path.read_text(encoding="utf-8"))
    assert report["frequency"] == "hourly"
    assert report["timestamp_convention"] == "UTC"
    assert report["ghi_units"] == "Wh/m^2"
    assert report["missing_interval_count"] == 1
    assert report["timestamp_range"]["first"] == "2020-01-01T00:00:00Z"
    assert report["timestamp_range"]["last"] == "2020-01-01T05:00:00Z"
    assert processed.frame["timestamp"].str.endswith("Z").all()


def test_report_serializes_irregular_spacing_and_unsorted_flags(
    tmp_path: Path,
) -> None:
    paths = ProjectPaths(tmp_path)
    raw_path = paths.hourly_dataset_path(SITE.site_id)
    raw_path.parent.mkdir(parents=True)
    frame = hourly_frame(hours=3).iloc[[1, 0, 2]].reset_index(drop=True)
    frame.loc[2, "timestamp"] = "2020-01-01T02:30:00Z"
    frame.to_csv(raw_path, index=False)

    processed = process_dataset(SITE, "hourly", paths=paths, now=FIXED_NOW)
    report = json.loads(processed.report_path.read_text(encoding="utf-8"))

    assert report["unsorted_timestamps"] is True
    assert report["irregular_spacing_count"] == 1
    assert report["missing_interval_count"] == 1
    # Cleaning sorted the rows without dropping the off-grid observation.
    assert processed.frame["timestamp"].tolist() == [
        "2020-01-01T00:00:00Z",
        "2020-01-01T01:00:00Z",
        "2020-01-01T02:30:00Z",
    ]


def test_reports_are_reproducible_apart_from_the_timestamp(tmp_path: Path) -> None:
    paths = ProjectPaths(tmp_path)
    raw_path = paths.raw_dataset_path(SITE.site_id)
    raw_path.parent.mkdir(parents=True)
    daily_frame().to_csv(raw_path, index=False)

    first = process_dataset(SITE, "daily", paths=paths, now=FIXED_NOW)
    second = process_dataset(SITE, "daily", paths=paths, now=FIXED_NOW)

    assert first.report == second.report
    assert first.report["output_sha256"] == second.report["output_sha256"]


def test_build_report_is_deterministic_with_fixed_timestamp() -> None:
    frame = daily_frame()
    validation = validate_daily(frame)
    common = dict(
        site=SITE,
        frequency="daily",
        source_path=PATHS.raw_dataset_path("tunis"),
        source_sha256="a" * 64,
        output_path=PATHS.clean_dataset_path("tunis"),
        output_sha256="b" * 64,
        input_row_count=5,
        output_row_count=5,
        validation=validation,
        operations=(),
        now=FIXED_NOW,
    )
    first = build_processing_report(paths=PATHS, **common)
    second = build_processing_report(paths=PATHS, **common)

    assert first == second
    assert first["timestamp_range"]["first"] == "2020-01-01"


def test_missing_raw_file_raises(tmp_path: Path) -> None:
    paths = ProjectPaths(tmp_path)

    with pytest.raises(FileNotFoundError, match="Raw dataset not found"):
        process_dataset(SITE, "daily", paths=paths)
