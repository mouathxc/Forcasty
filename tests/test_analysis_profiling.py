"""Offline tests for descriptive dataset profiling."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from forcasty.analysis.datasets import AnalysisDataset, DatasetSchemaError
from forcasty.analysis.profiling import (
    describe_numeric,
    profile_dataset,
    summarize_ghi,
    summarize_missing,
)
from forcasty.config import Site

SITE = Site(
    site_id="tunis",
    name="Tunis",
    latitude=36.8065,
    longitude=10.1815,
)
COLUMNS = ["ALLSKY_SFC_SW_DWN", "T2M", "RH2M", "WS2M"]


def daily_frame(rows: int = 4) -> pd.DataFrame:
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


def hourly_frame(hours: int = 4) -> pd.DataFrame:
    stamps = pd.date_range("2020-01-01T00:00:00Z", periods=hours, freq="h")
    return pd.DataFrame(
        {
            "timestamp": stamps.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ALLSKY_SFC_SW_DWN": [100.0 * index for index in range(hours)],
            "T2M": [10.0 + index for index in range(hours)],
            "RH2M": [60.0] * hours,
            "WS2M": [2.0] * hours,
        }
    )


def empty_daily_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {column: pd.Series([], dtype="float64") for column in COLUMNS}
    )
    frame.insert(0, "date", pd.Series([], dtype="object"))
    return frame


def make_dataset(
    frame: pd.DataFrame,
    frequency: str = "daily",
    source_path: Path = Path("data/processed/example.csv"),
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
        source_path=source_path,
    )


def test_profile_reports_dataset_metadata() -> None:
    profile = profile_dataset(make_dataset(daily_frame(rows=3)))

    assert profile["site_id"] == "tunis"
    assert profile["site_name"] == "Tunis"
    assert profile["frequency"] == "daily"
    assert profile["time_standard"] == "LST"
    assert profile["ghi_units"] == "kW-hr/m^2/day"
    assert profile["timestamp_column"] == "date"
    assert profile["row_count"] == 3
    assert profile["column_count"] == 5
    assert profile["columns"] == ["date", *COLUMNS]


def test_profile_reports_daily_timestamp_range() -> None:
    profile = profile_dataset(make_dataset(daily_frame(rows=3)))

    assert profile["first_timestamp"] == "2020-01-01"
    assert profile["last_timestamp"] == "2020-01-03"


def test_profile_reports_hourly_metadata_and_timestamp_range() -> None:
    profile = profile_dataset(
        make_dataset(hourly_frame(hours=3), frequency="hourly")
    )

    assert profile["frequency"] == "hourly"
    assert profile["time_standard"] == "UTC"
    assert profile["ghi_units"] == "Wh/m^2"
    assert profile["timestamp_column"] == "timestamp"
    assert profile["first_timestamp"] == "2020-01-01T00:00:00Z"
    assert profile["last_timestamp"] == "2020-01-01T02:00:00Z"


def test_missing_counts_and_percentages() -> None:
    frame = daily_frame(rows=5)
    frame.loc[1, "T2M"] = None
    frame.loc[2, "RH2M"] = None
    frame.loc[3, "RH2M"] = None

    profile = profile_dataset(make_dataset(frame))
    missing = profile["missing_by_column"]

    assert missing["date"] == {"count": 0, "percentage": 0.0}
    assert missing["T2M"] == {"count": 1, "percentage": 20.0}
    assert missing["RH2M"]["count"] == 2
    assert missing["RH2M"]["percentage"] == pytest.approx(40.0)


def test_ghi_categories_are_distinct_and_exhaustive() -> None:
    frame = daily_frame(rows=6)
    frame["ALLSKY_SFC_SW_DWN"] = [
        -2.0,
        0.0,
        0.0,
        5.0,
        float("nan"),
        7.0,
    ]

    profile = profile_dataset(make_dataset(frame))
    counts = profile["ghi_counts"]

    assert counts["missing"] == 1
    assert counts["zero"] == 2
    assert counts["positive"] == 2
    assert counts["negative"] == 1
    assert counts["non_null"] == 5
    assert (
        counts["missing"]
        + counts["zero"]
        + counts["positive"]
        + counts["negative"]
        == profile["row_count"]
    )


def test_missing_ghi_is_not_counted_as_zero() -> None:
    frame = daily_frame(rows=2)
    frame["ALLSKY_SFC_SW_DWN"] = [float("nan"), 0.0]

    counts = profile_dataset(make_dataset(frame))["ghi_counts"]

    assert counts == {
        "missing": 1,
        "zero": 1,
        "positive": 0,
        "negative": 0,
        "non_null": 1,
    }


def test_zero_ghi_is_not_labelled_as_night() -> None:
    frame = daily_frame(rows=2)
    frame["ALLSKY_SFC_SW_DWN"] = [0.0, 4.0]

    profile = profile_dataset(make_dataset(frame))

    assert set(profile["ghi_counts"]) == {
        "missing",
        "zero",
        "positive",
        "negative",
        "non_null",
    }
    assert "night" not in json.dumps(profile).lower()


def test_descriptive_statistics_are_correct_on_known_values() -> None:
    frame = daily_frame(rows=4)
    frame["T2M"] = [1.0, 2.0, 3.0, 4.0]

    stats = profile_dataset(make_dataset(frame))["descriptive_statistics"][
        "T2M"
    ]

    assert stats["count"] == 4
    assert stats["missing"] == 0
    assert stats["missing_percentage"] == 0.0
    assert stats["mean"] == pytest.approx(2.5)
    assert stats["median"] == pytest.approx(2.5)
    assert stats["std"] == pytest.approx(1.2909944487358056)
    assert stats["min"] == 1.0
    assert stats["max"] == 4.0


def test_missing_values_are_excluded_from_statistics() -> None:
    frame = daily_frame(rows=4)
    frame["T2M"] = [1.0, None, 3.0, None]

    stats = profile_dataset(make_dataset(frame))["descriptive_statistics"][
        "T2M"
    ]

    assert stats["count"] == 2
    assert stats["missing"] == 2
    assert stats["missing_percentage"] == pytest.approx(50.0)
    assert stats["mean"] == pytest.approx(2.0)
    assert stats["median"] == pytest.approx(2.0)
    assert stats["std"] == pytest.approx(1.4142135623730951)


def test_all_missing_numeric_column_is_safe() -> None:
    frame = daily_frame(rows=4)
    frame["RH2M"] = [None, None, None, None]

    profile = profile_dataset(make_dataset(frame))
    stats = profile["descriptive_statistics"]["RH2M"]

    assert stats["count"] == 0
    assert stats["missing"] == 4
    assert stats["missing_percentage"] == pytest.approx(100.0)
    assert stats["mean"] is None
    assert stats["median"] is None
    assert stats["std"] is None
    assert stats["min"] is None
    assert stats["max"] is None
    assert profile["missing_by_column"]["RH2M"] == {
        "count": 4,
        "percentage": 100.0,
    }


def test_empty_dataset_is_profiled_safely() -> None:
    dataset = make_dataset(empty_daily_frame())

    profile = profile_dataset(dataset)

    assert profile["row_count"] == 0
    assert profile["column_count"] == 5
    assert profile["first_timestamp"] is None
    assert profile["last_timestamp"] is None
    assert profile["ghi_counts"] == {
        "missing": 0,
        "zero": 0,
        "positive": 0,
        "negative": 0,
        "non_null": 0,
    }
    assert profile["missing_by_column"]["T2M"] == {
        "count": 0,
        "percentage": 0.0,
    }
    stats = profile["descriptive_statistics"]["T2M"]
    assert stats["count"] == 0
    assert stats["missing"] == 0
    assert stats["missing_percentage"] == 0.0
    assert stats["mean"] is None
    assert stats["std"] is None


def test_single_observation_has_no_standard_deviation() -> None:
    frame = daily_frame(rows=4)
    frame["T2M"] = [5.0, None, None, None]

    stats = profile_dataset(make_dataset(frame))["descriptive_statistics"][
        "T2M"
    ]

    assert stats["count"] == 1
    assert stats["mean"] == pytest.approx(5.0)
    assert stats["median"] == pytest.approx(5.0)
    assert stats["std"] is None


def test_all_missing_ghi_counts_only_missing() -> None:
    frame = daily_frame(rows=3)
    frame["ALLSKY_SFC_SW_DWN"] = [None, None, None]

    counts = profile_dataset(make_dataset(frame))["ghi_counts"]

    assert counts == {
        "missing": 3,
        "zero": 0,
        "positive": 0,
        "negative": 0,
        "non_null": 0,
    }


def test_profile_is_strictly_json_serializable() -> None:
    frame = daily_frame(rows=4)
    frame.loc[1, "T2M"] = None
    frame.loc[2, "RH2M"] = None

    profile = profile_dataset(make_dataset(frame))

    encoded = json.dumps(profile, allow_nan=False)
    assert json.loads(encoded) == profile
    assert type(profile["row_count"]) is int
    assert type(profile["ghi_counts"]["zero"]) is int
    assert type(
        profile["descriptive_statistics"]["T2M"]["mean"]
    ) is float


def test_profile_does_not_mutate_the_dataset() -> None:
    frame = daily_frame(rows=4)
    frame.loc[1, "T2M"] = None
    frame["station_note"] = ["a", "b", "c", "d"]
    dataset = make_dataset(frame)
    frame_before = dataset.frame.copy(deep=True)
    index_before = dataset.time_index.copy()

    profile_dataset(dataset)

    pd.testing.assert_frame_equal(dataset.frame, frame_before)
    assert dataset.time_index.equals(index_before)


def test_profile_does_not_write_any_files(tmp_path: Path) -> None:
    dataset = make_dataset(
        daily_frame(rows=3), source_path=tmp_path / "example.csv"
    )
    before = sorted(str(path) for path in tmp_path.rglob("*"))

    profile_dataset(dataset)

    after = sorted(str(path) for path in tmp_path.rglob("*"))
    assert after == before


def test_extra_columns_are_reported_and_numeric_ones_profiled() -> None:
    frame = daily_frame(rows=3)
    frame["station_note"] = ["a", "b", "c"]
    frame["cloud_index"] = [0.1, 0.2, 0.3]

    profile = profile_dataset(make_dataset(frame))

    assert "station_note" in profile["columns"]
    assert "station_note" in profile["missing_by_column"]
    assert "station_note" not in profile["descriptive_statistics"]
    assert "cloud_index" in profile["descriptive_statistics"]


def test_describe_numeric_rejects_non_numeric_series() -> None:
    with pytest.raises(TypeError, match="numeric"):
        describe_numeric(pd.Series(["a", "b"]))


def test_summarize_missing_and_ghi_are_pure_functions() -> None:
    frame = daily_frame(rows=3)
    frame.loc[1, "T2M"] = None
    frame["ALLSKY_SFC_SW_DWN"] = [-1.0, 0.0, None]

    assert summarize_missing(frame)["T2M"]["count"] == 1
    assert summarize_ghi(frame["ALLSKY_SFC_SW_DWN"]) == {
        "missing": 1,
        "zero": 1,
        "positive": 0,
        "negative": 1,
        "non_null": 2,
    }


def test_summarize_ghi_rejects_non_numeric_values() -> None:
    series = pd.Series(["0", "sunny", None])

    with pytest.raises(DatasetSchemaError, match="numeric"):
        summarize_ghi(series)


def test_profile_requires_the_ghi_column() -> None:
    frame = daily_frame(rows=3).drop(columns=["ALLSKY_SFC_SW_DWN"])

    with pytest.raises(DatasetSchemaError, match="GHI column"):
        profile_dataset(make_dataset(frame))
