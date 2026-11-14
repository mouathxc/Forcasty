"""Offline tests for daily/hourly validation contracts."""

from __future__ import annotations

import pandas as pd
import pytest

from forcasty.data.validation import (
    DAILY,
    HOURLY,
    ValidationError,
    validate_daily,
    validate_dataset,
    validate_hourly,
)

COLUMNS = ["ALLSKY_SFC_SW_DWN", "T2M", "RH2M", "WS2M"]


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
    stamps = pd.date_range(
        "2020-01-01T00:00:00Z", periods=hours, freq="h"
    )
    return pd.DataFrame(
        {
            "timestamp": stamps.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ALLSKY_SFC_SW_DWN": [index * 100.0 for index in range(hours)],
            "T2M": [10.0 + index for index in range(hours)],
            "RH2M": [60.0] * hours,
            "WS2M": [2.0] * hours,
        }
    )


def test_valid_daily_dataset() -> None:
    report = validate_daily(daily_frame())

    assert report.is_valid
    assert report.frequency == DAILY
    assert report.time_standard == "LST"
    assert report.ghi_units == "kW-hr/m^2/day"
    assert report.row_count == 5
    assert report.first_timestamp == "2020-01-01"
    assert report.last_timestamp == "2020-01-05"
    assert report.missing_interval_count == 0
    assert report.duplicate_timestamp_count == 0
    assert report.invalid_timestamp_count == 0
    assert report.missing_value_counts == {column: 0 for column in COLUMNS}
    assert report.errors == []


def test_valid_hourly_dataset() -> None:
    report = validate_hourly(hourly_frame())

    assert report.is_valid
    assert report.frequency == HOURLY
    assert report.time_standard == "UTC"
    assert report.ghi_units == "Wh/m^2"
    assert report.first_timestamp == "2020-01-01T00:00:00Z"
    assert report.last_timestamp == "2020-01-01T05:00:00Z"
    assert report.missing_interval_count == 0
    assert report.duplicate_timestamp_count == 0


def test_empty_dataset_is_an_error() -> None:
    report = validate_daily(pd.DataFrame(columns=["date", *COLUMNS]))

    assert not report.is_valid
    assert report.errors[0].code == "EMPTY_DATASET"
    with pytest.raises(ValidationError, match="EMPTY_DATASET"):
        report.raise_if_invalid()


@pytest.mark.parametrize("column", ["date", "ALLSKY_SFC_SW_DWN", "T2M", "RH2M", "WS2M"])
def test_missing_required_daily_columns(column: str) -> None:
    frame = daily_frame().drop(columns=[column])

    report = validate_daily(frame)

    assert not report.is_valid
    assert report.errors[0].code == "MISSING_COLUMNS"
    assert column in report.errors[0].message


@pytest.mark.parametrize(
    "column", ["timestamp", "ALLSKY_SFC_SW_DWN", "T2M", "RH2M", "WS2M"]
)
def test_missing_required_hourly_columns(column: str) -> None:
    frame = hourly_frame().drop(columns=[column])

    report = validate_hourly(frame)

    assert not report.is_valid
    assert report.errors[0].code == "MISSING_COLUMNS"


def test_invalid_timestamps_are_counted() -> None:
    frame = daily_frame()
    frame.loc[2, "date"] = "not-a-date"

    report = validate_daily(frame)

    assert not report.is_valid
    assert report.invalid_timestamp_count == 1
    assert any(issue.code == "INVALID_TIMESTAMPS" for issue in report.errors)


def test_duplicate_timestamps_are_errors() -> None:
    frame = hourly_frame()
    frame.loc[3, "timestamp"] = frame.loc[2, "timestamp"]

    report = validate_hourly(frame)

    assert not report.is_valid
    assert report.duplicate_timestamp_count == 1
    assert any(issue.code == "DUPLICATE_TIMESTAMPS" for issue in report.errors)


def test_unsorted_timestamps_are_a_warning() -> None:
    frame = daily_frame()
    frame = frame.iloc[[2, 0, 1, 3, 4]].reset_index(drop=True)

    report = validate_daily(frame)

    assert report.is_valid
    assert report.unsorted_timestamps
    assert any(issue.code == "UNSORTED_TIMESTAMPS" for issue in report.warnings)


def test_missing_daily_dates_are_reported_not_filled() -> None:
    frame = daily_frame().drop(index=2).reset_index(drop=True)

    report = validate_daily(frame)

    assert report.is_valid
    assert report.missing_interval_count == 1
    assert any(issue.code == "MISSING_INTERVALS" for issue in report.warnings)
    assert report.first_timestamp == "2020-01-01"
    assert report.last_timestamp == "2020-01-05"


def test_missing_hourly_intervals_are_reported_not_filled() -> None:
    frame = hourly_frame().drop(index=2).reset_index(drop=True)

    report = validate_hourly(frame)

    assert report.is_valid
    assert report.missing_interval_count == 1


def test_irregular_hourly_spacing_is_distinguished_from_gaps() -> None:
    frame = hourly_frame(hours=3)
    frame.loc[2, "timestamp"] = "2020-01-01T02:30:00Z"

    report = validate_hourly(frame)

    assert report.is_valid
    assert report.missing_interval_count == 1
    assert report.irregular_spacing_count == 1
    assert any(
        issue.code == "IRREGULAR_TIMESTAMP_SPACING"
        for issue in report.warnings
    )


def test_nasa_fill_values_are_reported_separately_from_missing() -> None:
    frame = daily_frame()
    frame.loc[1, "ALLSKY_SFC_SW_DWN"] = -999
    frame.loc[2, "T2M"] = None

    report = validate_daily(frame)

    assert report.is_valid
    assert report.fill_value_counts["ALLSKY_SFC_SW_DWN"] == 1
    assert report.fill_value_counts["T2M"] == 0
    assert report.missing_value_counts["ALLSKY_SFC_SW_DWN"] == 1
    assert report.missing_value_counts["T2M"] == 1
    assert report.missing_value_total == 2
    assert any(issue.code == "NASA_FILL_VALUES" for issue in report.warnings)
    assert any(issue.code == "MISSING_VALUES" for issue in report.warnings)


def test_negative_ghi_is_flagged_but_not_rejected() -> None:
    frame = daily_frame()
    frame.loc[1, "ALLSKY_SFC_SW_DWN"] = -5.0

    report = validate_daily(frame)

    assert report.is_valid
    assert report.invalid_observation_counts["negative_ghi"] == 1
    assert any(issue.code == "NEGATIVE_GHI" for issue in report.warnings)


def test_relative_humidity_bounds_are_checked() -> None:
    frame = hourly_frame()
    frame.loc[1, "RH2M"] = -1.0
    frame.loc[2, "RH2M"] = 101.0

    report = validate_hourly(frame)

    assert report.is_valid
    assert report.invalid_observation_counts["relative_humidity_below_0"] == 1
    assert report.invalid_observation_counts["relative_humidity_above_100"] == 1
    assert any(
        issue.code == "RELATIVE_HUMIDITY_OUT_OF_RANGE"
        for issue in report.warnings
    )


def test_negative_wind_speed_is_flagged() -> None:
    frame = hourly_frame()
    frame.loc[0, "WS2M"] = -0.5

    report = validate_hourly(frame)

    assert report.is_valid
    assert report.invalid_observation_counts["negative_wind_speed"] == 1


@pytest.mark.parametrize("value", [-91.0, 61.0])
def test_implausible_temperature_is_flagged(value: float) -> None:
    frame = daily_frame()
    frame.loc[0, "T2M"] = value

    report = validate_daily(frame)

    assert report.is_valid
    assert report.invalid_observation_counts["temperature_out_of_range"] == 1


def test_invalid_numeric_strings_block_cleaning() -> None:
    frame = daily_frame()
    frame["T2M"] = frame["T2M"].astype(object)
    frame.loc[2, "T2M"] = "warm"

    report = validate_daily(frame)

    assert not report.is_valid
    assert report.invalid_numeric_counts["T2M"] == 1
    assert any(
        issue.code == "INVALID_NUMERIC_VALUES" for issue in report.errors
    )


def test_hourly_timestamp_off_the_hour_is_a_warning() -> None:
    frame = hourly_frame()
    frame.loc[1, "timestamp"] = "2020-01-01T01:30:00Z"

    report = validate_hourly(frame)

    assert report.is_valid
    assert any(
        issue.code == "HOURLY_NOT_START_OF_HOUR" for issue in report.warnings
    )


def test_daily_time_component_is_normalised_and_detected() -> None:
    frame = daily_frame()
    frame.loc[0, "date"] = "2020-01-01T12:00:00"

    report = validate_daily(frame)

    assert report.is_valid
    assert report.first_timestamp == "2020-01-01"
    assert any(
        issue.code == "DAILY_TIME_COMPONENT" for issue in report.warnings
    )


def test_unknown_frequency_is_rejected() -> None:
    with pytest.raises(ValueError, match="frequency"):
        validate_dataset(daily_frame(), "monthly")


def test_report_to_dict_contains_counts_and_units() -> None:
    payload = validate_hourly(hourly_frame()).to_dict()

    assert payload["frequency"] == "hourly"
    assert payload["time_standard"] == "UTC"
    assert payload["ghi_units"] == "Wh/m^2"
    assert payload["row_count"] == 6
    assert payload["is_valid"] is True
    assert "missing_value_counts" in payload
    assert "fill_value_counts" in payload
