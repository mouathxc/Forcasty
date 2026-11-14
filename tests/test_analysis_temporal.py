"""Offline tests for descriptive temporal aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from forcasty.analysis.datasets import (
    AnalysisDataset,
    UnsupportedFrequencyError,
)
from forcasty.analysis.temporal import (
    analyze_daily_temporal,
    analyze_hourly_temporal,
)
from forcasty.config import Site

SITE = Site(
    site_id="tunis",
    name="Tunis",
    latitude=36.8065,
    longitude=10.1815,
)


def daily_frame(dates: list[str], ghi: list[float | None]) -> pd.DataFrame:
    count = len(dates)
    return pd.DataFrame(
        {
            "date": dates,
            "ALLSKY_SFC_SW_DWN": ghi,
            "T2M": [20.0] * count,
            "RH2M": [50.0] * count,
            "WS2M": [2.0] * count,
        }
    )


def hourly_frame(
    timestamps: list[str], ghi: list[float | None]
) -> pd.DataFrame:
    count = len(timestamps)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "ALLSKY_SFC_SW_DWN": ghi,
            "T2M": [20.0] * count,
            "RH2M": [50.0] * count,
            "WS2M": [2.0] * count,
        }
    )


def empty_daily_frame() -> pd.DataFrame:
    return daily_frame([], []).astype(
        {"date": "object", "ALLSKY_SFC_SW_DWN": "float64"}
    )


def make_dataset(
    frame: pd.DataFrame,
    frequency: str = "daily",
) -> AnalysisDataset:
    if frequency == "daily":
        index = pd.DatetimeIndex(
            pd.to_datetime(frame["date"], format="%Y-%m-%d")
        )
        timestamp_column = "date"
        time_standard = "LST"
        ghi_units = "kW-hr/m^2/day"
    else:
        index = pd.DatetimeIndex(
            pd.to_datetime(
                frame["timestamp"],
                format="%Y-%m-%dT%H:%M:%SZ",
                utc=True,
            )
        )
        timestamp_column = "timestamp"
        time_standard = "UTC"
        ghi_units = "Wh/m^2"

    return AnalysisDataset(
        site=SITE,
        frequency=frequency,
        frame=frame,
        time_index=index,
        timestamp_column=timestamp_column,
        time_standard=time_standard,
        ghi_units=ghi_units,
        source_path=Path("data/processed/example.csv"),
    )


def groups_by_key(records: list[dict]) -> dict:
    return {record["key"]: record for record in records}


def test_daily_yearly_aggregation() -> None:
    frame = daily_frame(
        ["2020-12-31", "2021-01-01", "2021-01-02"],
        [1.0, 2.0, 4.0],
    )

    records = analyze_daily_temporal(make_dataset(frame))["by_year"]

    assert [record["key"] for record in records] == [2020, 2021]
    by_year = groups_by_key(records)
    assert by_year[2020]["total"] == 1
    assert by_year[2020]["valid"] == 1
    assert by_year[2020]["mean"] == pytest.approx(1.0)
    assert by_year[2020]["std"] is None
    assert by_year[2021]["total"] == 2
    assert by_year[2021]["mean"] == pytest.approx(3.0)
    assert by_year[2021]["median"] == pytest.approx(3.0)
    assert by_year[2021]["std"] == pytest.approx(1.4142135623730951)
    assert by_year[2021]["min"] == 2.0
    assert by_year[2021]["max"] == 4.0


def test_daily_calendar_month_aggregation() -> None:
    frame = daily_frame(
        ["2020-01-05", "2020-02-10", "2021-01-15"],
        [1.0, 2.0, 6.0],
    )

    by_month = groups_by_key(
        analyze_daily_temporal(make_dataset(frame))["by_month"]
    )

    assert sorted(by_month) == [1, 2]
    assert by_month[1]["total"] == 2
    assert by_month[1]["mean"] == pytest.approx(3.5)
    assert by_month[1]["median"] == pytest.approx(3.5)
    assert by_month[1]["std"] == pytest.approx(3.5355339059327378)
    assert by_month[1]["min"] == 1.0
    assert by_month[1]["max"] == 6.0
    assert by_month[2]["total"] == 1
    assert by_month[2]["mean"] == pytest.approx(2.0)
    assert by_month[2]["std"] is None


def test_daily_year_month_aggregation() -> None:
    frame = daily_frame(
        ["2020-01-05", "2020-01-06", "2020-02-10"],
        [1.0, 3.0, 5.0],
    )

    records = analyze_daily_temporal(make_dataset(frame))["by_year_month"]

    assert [record["key"] for record in records] == ["2020-01", "2020-02"]
    by_month = groups_by_key(records)
    assert by_month["2020-01"]["total"] == 2
    assert by_month["2020-01"]["mean"] == pytest.approx(2.0)
    assert by_month["2020-01"]["std"] == pytest.approx(
        1.4142135623730951
    )
    assert by_month["2020-02"]["mean"] == pytest.approx(5.0)


def test_hourly_utc_hour_of_day_aggregation() -> None:
    frame = hourly_frame(
        [
            "2020-01-01T00:00:00Z",
            "2020-01-01T01:00:00Z",
            "2020-01-02T00:00:00Z",
            "2020-01-02T01:00:00Z",
        ],
        [0.0, 10.0, 20.0, 30.0],
    )

    records = analyze_hourly_temporal(make_dataset(frame, "hourly"))[
        "by_utc_hour"
    ]

    assert [record["key"] for record in records] == [0, 1]
    by_hour = groups_by_key(records)
    assert by_hour[0]["total"] == 2
    assert by_hour[0]["mean"] == pytest.approx(10.0)
    assert by_hour[0]["median"] == pytest.approx(10.0)
    assert by_hour[0]["std"] == pytest.approx(14.142135623730951)
    assert by_hour[0]["min"] == 0.0
    assert by_hour[0]["max"] == 20.0
    assert by_hour[1]["mean"] == pytest.approx(20.0)
    assert by_hour[1]["min"] == 10.0
    assert by_hour[1]["max"] == 30.0


def test_hourly_month_aggregation_uses_utc() -> None:
    frame = hourly_frame(
        [
            "2020-01-31T23:00:00Z",
            "2020-02-01T00:00:00Z",
            "2021-01-01T00:00:00Z",
        ],
        [1.0, 2.0, 6.0],
    )

    by_month = groups_by_key(
        analyze_hourly_temporal(make_dataset(frame, "hourly"))["by_month"]
    )

    assert sorted(by_month) == [1, 2]
    assert by_month[1]["total"] == 2
    assert by_month[1]["mean"] == pytest.approx(3.5)
    assert by_month[2]["total"] == 1
    assert by_month[2]["mean"] == pytest.approx(2.0)
    assert by_month[2]["std"] is None


def test_hourly_year_month_aggregation() -> None:
    frame = hourly_frame(
        [
            "2020-01-01T00:00:00Z",
            "2020-01-02T00:00:00Z",
            "2020-02-01T00:00:00Z",
        ],
        [1.0, 3.0, 5.0],
    )

    records = analyze_hourly_temporal(make_dataset(frame, "hourly"))[
        "by_year_month"
    ]

    assert [record["key"] for record in records] == ["2020-01", "2020-02"]
    by_month = groups_by_key(records)
    assert by_month["2020-01"]["total"] == 2
    assert by_month["2020-01"]["mean"] == pytest.approx(2.0)
    assert by_month["2020-02"]["mean"] == pytest.approx(5.0)


def test_missing_ghi_is_counted_and_excluded_from_statistics() -> None:
    frame = daily_frame(
        ["2020-01-01", "2020-01-02", "2020-01-03"],
        [1.0, None, 3.0],
    )

    record = analyze_daily_temporal(make_dataset(frame))["by_year"][0]

    assert record["total"] == 3
    assert record["valid"] == 2
    assert record["missing"] == 1
    assert record["mean"] == pytest.approx(2.0)
    assert record["median"] == pytest.approx(2.0)
    assert record["std"] == pytest.approx(1.4142135623730951)


def test_all_missing_group_reports_none_statistics() -> None:
    frame = daily_frame(
        ["2020-01-01", "2020-01-02"],
        [None, None],
    )

    record = analyze_daily_temporal(make_dataset(frame))["by_year"][0]

    assert record["total"] == 2
    assert record["valid"] == 0
    assert record["missing"] == 2
    assert record["mean"] is None
    assert record["median"] is None
    assert record["std"] is None
    assert record["min"] is None
    assert record["max"] is None


def test_single_valid_observation_has_no_standard_deviation() -> None:
    frame = daily_frame(
        ["2020-01-01", "2020-01-02", "2020-01-03"],
        [5.0, None, None],
    )

    record = analyze_daily_temporal(make_dataset(frame))["by_year"][0]

    assert record["valid"] == 1
    assert record["missing"] == 2
    assert record["mean"] == pytest.approx(5.0)
    assert record["std"] is None


def test_zero_ghi_is_an_observation_not_missing() -> None:
    frame = daily_frame(
        ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
        [0.0, None, 0.0, 5.0],
    )

    record = analyze_daily_temporal(make_dataset(frame))["by_year"][0]

    assert record["total"] == 4
    assert record["valid"] == 3
    assert record["missing"] == 1
    assert record["min"] == 0.0
    assert record["max"] == 5.0
    assert record["mean"] == pytest.approx(5.0 / 3.0)


def test_result_reports_units_and_time_standard() -> None:
    daily = analyze_daily_temporal(
        make_dataset(daily_frame(["2020-01-01"], [1.0]))
    )
    hourly = analyze_hourly_temporal(
        make_dataset(
            hourly_frame(["2020-01-01T00:00:00Z"], [1.0]),
            "hourly",
        )
    )

    assert daily["frequency"] == "daily"
    assert daily["time_standard"] == "LST"
    assert daily["ghi_units"] == "kW-hr/m^2/day"
    assert daily["ghi_column"] == "ALLSKY_SFC_SW_DWN"
    assert hourly["frequency"] == "hourly"
    assert hourly["time_standard"] == "UTC"
    assert hourly["ghi_units"] == "Wh/m^2"
    assert hourly["ghi_column"] == "ALLSKY_SFC_SW_DWN"


def test_wrong_frequency_is_rejected() -> None:
    hourly = make_dataset(
        hourly_frame(["2020-01-01T00:00:00Z"], [1.0]),
        "hourly",
    )
    daily = make_dataset(daily_frame(["2020-01-01"], [1.0]))

    with pytest.raises(UnsupportedFrequencyError, match="daily"):
        analyze_daily_temporal(hourly)
    with pytest.raises(UnsupportedFrequencyError, match="hourly"):
        analyze_hourly_temporal(daily)


def test_result_is_strictly_json_serializable() -> None:
    frame = daily_frame(
        ["2020-01-01", "2020-01-02", "2021-01-01"],
        [1.0, None, 3.0],
    )

    result = analyze_daily_temporal(make_dataset(frame))

    encoded = json.dumps(result, allow_nan=False)
    assert json.loads(encoded) == result


def test_analysis_does_not_mutate_frame_or_index() -> None:
    dataset = make_dataset(
        daily_frame(["2020-01-01", "2020-01-02"], [1.0, None])
    )
    frame_before = dataset.frame.copy(deep=True)
    index_before = dataset.time_index.copy()

    analyze_daily_temporal(dataset)

    pd.testing.assert_frame_equal(dataset.frame, frame_before)
    assert dataset.time_index.equals(index_before)


def test_empty_dataset_is_analyzed_safely() -> None:
    result = analyze_daily_temporal(make_dataset(empty_daily_frame()))

    assert result["row_count"] == 0
    assert result["by_year"] == []
    assert result["by_month"] == []
    assert result["by_year_month"] == []
