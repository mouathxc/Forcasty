"""Offline tests for analysis dataset loading and scientific contracts."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from forcasty.analysis.datasets import (
    AnalysisDataError,
    AnalysisDataset,
    DatasetNotFoundError,
    DatasetSchemaError,
    TemporalConventionError,
    UnsupportedFrequencyError,
    assert_units,
    load_cleaned_dataset,
)
from forcasty.config import Site
from forcasty.paths import ProjectPaths

SITE = Site(
    site_id="tunis",
    name="Tunis",
    latitude=36.8065,
    longitude=10.1815,
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


def dataset_path(paths: ProjectPaths, frequency: str) -> Path:
    if frequency == "hourly":
        return paths.hourly_clean_dataset_path(SITE.site_id)
    return paths.clean_dataset_path(SITE.site_id)


def write_frame(
    paths: ProjectPaths, frame: pd.DataFrame, frequency: str
) -> Path:
    path = dataset_path(paths, frequency)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def test_load_daily_dataset(tmp_path: Path) -> None:
    paths = ProjectPaths(tmp_path)
    frame = daily_frame()
    source = write_frame(paths, frame, "daily")

    dataset = load_cleaned_dataset(SITE, "daily", paths=paths)

    assert isinstance(dataset, AnalysisDataset)
    assert dataset.site is SITE
    assert dataset.frequency == "daily"
    assert dataset.time_standard == "LST"
    assert dataset.ghi_units == "kW-hr/m^2/day"
    assert dataset.timestamp_column == "date"
    assert dataset.source_path == source
    assert dataset.row_count == 5
    assert dataset.columns == tuple(frame.columns)
    assert dataset.time_index.tz is None
    assert dataset.time_index.tolist() == list(
        pd.date_range("2020-01-01", periods=5, freq="D")
    )
    assert dataset.first_timestamp == pd.Timestamp("2020-01-01")
    assert dataset.last_timestamp == pd.Timestamp("2020-01-05")


def test_load_hourly_dataset(tmp_path: Path) -> None:
    paths = ProjectPaths(tmp_path)
    write_frame(paths, hourly_frame(), "hourly")

    dataset = load_cleaned_dataset(SITE, "hourly", paths=paths)

    assert dataset.frequency == "hourly"
    assert dataset.time_standard == "UTC"
    assert dataset.ghi_units == "Wh/m^2"
    assert dataset.timestamp_column == "timestamp"
    assert dataset.row_count == 4
    assert dataset.time_index.tz is not None
    assert str(dataset.time_index.tz) == "UTC"
    assert dataset.first_timestamp == pd.Timestamp("2020-01-01T00:00:00Z")
    assert dataset.last_timestamp == pd.Timestamp("2020-01-01T03:00:00Z")
    assert bool((dataset.time_index.minute == 0).all())
    assert bool((dataset.time_index.second == 0).all())


def test_loading_preserves_source_columns_and_dtypes(tmp_path: Path) -> None:
    paths = ProjectPaths(tmp_path)
    frame = hourly_frame()
    frame["station_note"] = ["a", "b", "c", "d"]
    write_frame(paths, frame, "hourly")

    dataset = load_cleaned_dataset(SITE, "hourly", paths=paths)

    # The CSV timestamp stays a string; parsing lives in time_index only.
    assert dataset.frame["timestamp"].tolist() == frame["timestamp"].tolist()
    assert not pd.api.types.is_datetime64_any_dtype(
        dataset.frame["timestamp"]
    )
    assert all(
        isinstance(value, str) for value in dataset.frame["timestamp"]
    )
    assert dataset.frame[COLUMNS].dtypes.tolist() == [float] * len(COLUMNS)
    assert "station_note" in dataset.frame.columns
    assert dataset.frame["station_note"].tolist() == ["a", "b", "c", "d"]


def test_loading_does_not_modify_the_source_file(tmp_path: Path) -> None:
    paths = ProjectPaths(tmp_path)
    source = write_frame(paths, daily_frame(), "daily")
    before = hashlib.sha256(source.read_bytes()).hexdigest()

    load_cleaned_dataset(SITE, "daily", paths=paths)

    assert hashlib.sha256(source.read_bytes()).hexdigest() == before


def test_invalid_frequency_is_rejected(tmp_path: Path) -> None:
    paths = ProjectPaths(tmp_path)

    with pytest.raises(UnsupportedFrequencyError, match="frequency"):
        load_cleaned_dataset(SITE, "monthly", paths=paths)


def test_missing_file_raises_an_actionable_error(tmp_path: Path) -> None:
    paths = ProjectPaths(tmp_path)
    expected = dataset_path(paths, "daily")

    with pytest.raises(DatasetNotFoundError) as info:
        load_cleaned_dataset(SITE, "daily", paths=paths)

    assert isinstance(info.value, FileNotFoundError)
    assert str(expected) in str(info.value)
    assert "forcasty.data.cleaning" in str(info.value)


def test_missing_required_column_is_rejected(tmp_path: Path) -> None:
    paths = ProjectPaths(tmp_path)
    frame = daily_frame().drop(columns=["T2M"])
    write_frame(paths, frame, "daily")

    with pytest.raises(DatasetSchemaError, match="T2M"):
        load_cleaned_dataset(SITE, "daily", paths=paths)


def test_non_numeric_value_column_is_rejected(tmp_path: Path) -> None:
    paths = ProjectPaths(tmp_path)
    frame = daily_frame()
    frame["T2M"] = "warm"
    write_frame(paths, frame, "daily")

    with pytest.raises(DatasetSchemaError, match="non-numeric"):
        load_cleaned_dataset(SITE, "daily", paths=paths)


def test_empty_dataset_is_rejected(tmp_path: Path) -> None:
    paths = ProjectPaths(tmp_path)
    write_frame(paths, daily_frame(rows=0), "daily")

    with pytest.raises(DatasetSchemaError, match="no rows"):
        load_cleaned_dataset(SITE, "daily", paths=paths)


def test_completely_empty_file_is_rejected(tmp_path: Path) -> None:
    paths = ProjectPaths(tmp_path)
    path = dataset_path(paths, "daily")
    path.parent.mkdir(parents=True)
    path.write_text("", encoding="utf-8")

    with pytest.raises(DatasetSchemaError, match="empty"):
        load_cleaned_dataset(SITE, "daily", paths=paths)


def test_non_canonical_daily_timestamp_is_rejected(tmp_path: Path) -> None:
    paths = ProjectPaths(tmp_path)
    frame = daily_frame()
    frame.loc[0, "date"] = "01/01/2020"
    write_frame(paths, frame, "daily")

    with pytest.raises(DatasetSchemaError, match="canonical format"):
        load_cleaned_dataset(SITE, "daily", paths=paths)


def test_non_canonical_hourly_timestamp_is_rejected(tmp_path: Path) -> None:
    paths = ProjectPaths(tmp_path)
    frame = hourly_frame()
    frame.loc[0, "timestamp"] = "2020-01-01T00:00:00+00:00"
    write_frame(paths, frame, "hourly")

    with pytest.raises(DatasetSchemaError, match="canonical format"):
        load_cleaned_dataset(SITE, "hourly", paths=paths)


def test_hourly_timestamp_not_at_start_of_hour_is_rejected(
    tmp_path: Path,
) -> None:
    paths = ProjectPaths(tmp_path)
    frame = hourly_frame(hours=3)
    frame.loc[1, "timestamp"] = "2020-01-01T01:30:00Z"
    write_frame(paths, frame, "hourly")

    with pytest.raises(TemporalConventionError, match="start of the hour"):
        load_cleaned_dataset(SITE, "hourly", paths=paths)


def test_duplicate_timestamps_are_rejected(tmp_path: Path) -> None:
    paths = ProjectPaths(tmp_path)
    frame = daily_frame()
    frame.loc[1, "date"] = frame.loc[0, "date"]
    write_frame(paths, frame, "daily")

    with pytest.raises(TemporalConventionError, match="duplicate"):
        load_cleaned_dataset(SITE, "daily", paths=paths)


def test_unsorted_timestamps_are_rejected_not_sorted(tmp_path: Path) -> None:
    paths = ProjectPaths(tmp_path)
    frame = daily_frame().iloc[[2, 0, 1, 3, 4]].reset_index(drop=True)
    write_frame(paths, frame, "daily")

    with pytest.raises(TemporalConventionError, match="strictly increasing"):
        load_cleaned_dataset(SITE, "daily", paths=paths)


def test_assert_units_accepts_both_valid_frequencies(
    tmp_path: Path,
) -> None:
    paths = ProjectPaths(tmp_path)
    write_frame(paths, daily_frame(), "daily")
    write_frame(paths, hourly_frame(), "hourly")

    assert_units(load_cleaned_dataset(SITE, "daily", paths=paths))
    assert_units(load_cleaned_dataset(SITE, "hourly", paths=paths))


def test_assert_units_rejects_daily_index_with_non_midnight_timestamp(
    tmp_path: Path,
) -> None:
    paths = ProjectPaths(tmp_path)
    write_frame(paths, daily_frame(), "daily")
    dataset = load_cleaned_dataset(SITE, "daily", paths=paths)

    midday = dataset.time_index + pd.Timedelta(hours=12)
    with pytest.raises(TemporalConventionError, match="midnight"):
        assert_units(replace(dataset, time_index=midday))


def test_assert_units_rejects_non_utc_hourly_timezone(
    tmp_path: Path,
) -> None:
    paths = ProjectPaths(tmp_path)
    write_frame(paths, hourly_frame(), "hourly")
    dataset = load_cleaned_dataset(SITE, "hourly", paths=paths)

    paris = dataset.time_index.tz_convert("Europe/Paris")
    with pytest.raises(TemporalConventionError, match="UTC"):
        assert_units(replace(dataset, time_index=paris))


def test_assert_units_rejects_daily_index_not_matching_frame_dates(
    tmp_path: Path,
) -> None:
    paths = ProjectPaths(tmp_path)
    write_frame(paths, daily_frame(), "daily")
    dataset = load_cleaned_dataset(SITE, "daily", paths=paths)

    shifted = dataset.time_index + pd.Timedelta(days=1)
    with pytest.raises(TemporalConventionError, match="does not correspond"):
        assert_units(replace(dataset, time_index=shifted))


def test_assert_units_rejects_hourly_index_not_matching_frame_timestamps(
    tmp_path: Path,
) -> None:
    paths = ProjectPaths(tmp_path)
    write_frame(paths, hourly_frame(), "hourly")
    dataset = load_cleaned_dataset(SITE, "hourly", paths=paths)

    shifted = dataset.time_index + pd.Timedelta(hours=1)
    with pytest.raises(TemporalConventionError, match="does not correspond"):
        assert_units(replace(dataset, time_index=shifted))


def test_assert_units_rejects_wrong_ghi_units(tmp_path: Path) -> None:
    paths = ProjectPaths(tmp_path)
    write_frame(paths, daily_frame(), "daily")
    dataset = load_cleaned_dataset(SITE, "daily", paths=paths)

    with pytest.raises(DatasetSchemaError, match="GHI units"):
        assert_units(replace(dataset, ghi_units="Wh/m^2"))


def test_assert_units_rejects_aware_daily_time_index(tmp_path: Path) -> None:
    paths = ProjectPaths(tmp_path)
    write_frame(paths, daily_frame(), "daily")
    dataset = load_cleaned_dataset(SITE, "daily", paths=paths)

    aware = dataset.time_index.tz_localize("UTC")
    with pytest.raises(TemporalConventionError, match="timezone-naive"):
        assert_units(replace(dataset, time_index=aware))


def test_assert_units_rejects_naive_hourly_time_index(tmp_path: Path) -> None:
    paths = ProjectPaths(tmp_path)
    write_frame(paths, hourly_frame(), "hourly")
    dataset = load_cleaned_dataset(SITE, "hourly", paths=paths)

    naive = dataset.time_index.tz_localize(None)
    with pytest.raises(TemporalConventionError, match="timezone-aware"):
        assert_units(replace(dataset, time_index=naive))


def test_assert_units_rejects_wrong_time_standard(tmp_path: Path) -> None:
    paths = ProjectPaths(tmp_path)
    write_frame(paths, hourly_frame(), "hourly")
    dataset = load_cleaned_dataset(SITE, "hourly", paths=paths)

    with pytest.raises(TemporalConventionError, match="UTC"):
        assert_units(replace(dataset, time_standard="LST"))


def test_assert_units_rejects_mismatched_lengths(tmp_path: Path) -> None:
    paths = ProjectPaths(tmp_path)
    write_frame(paths, daily_frame(), "daily")
    dataset = load_cleaned_dataset(SITE, "daily", paths=paths)

    with pytest.raises(DatasetSchemaError, match="different lengths"):
        assert_units(replace(dataset, time_index=dataset.time_index[:3]))


def test_assert_units_rejects_unsupported_frequency() -> None:
    dataset = AnalysisDataset(
        site=SITE,
        frequency="monthly",
        frame=daily_frame(),
        time_index=pd.DatetimeIndex(pd.date_range("2020-01-01", periods=5)),
        timestamp_column="date",
        time_standard="LST",
        ghi_units="kW-hr/m^2/day",
        source_path=Path("data/processed/example.csv"),
    )

    with pytest.raises(UnsupportedFrequencyError, match="frequency"):
        assert_units(dataset)


def test_invalid_site_id_is_rejected_before_reading(tmp_path: Path) -> None:
    paths = ProjectPaths(tmp_path)
    unsafe = Site(
        site_id="../escape",
        name="Escape",
        latitude=0.0,
        longitude=0.0,
    )

    with pytest.raises(AnalysisDataError, match="site_id"):
        load_cleaned_dataset(unsafe, "daily", paths=paths)


def test_non_site_argument_is_rejected(tmp_path: Path) -> None:
    paths = ProjectPaths(tmp_path)

    with pytest.raises(TypeError, match="Site"):
        load_cleaned_dataset(
            "tunis",  # type: ignore[arg-type]
            "daily",
            paths=paths,
        )


def test_loader_defends_against_site_ids_the_legacy_path_helpers_accept(
    tmp_path: Path,
) -> None:
    paths = ProjectPaths(tmp_path)
    unsafe = Site(
        site_id="../escape",
        name="Escape",
        latitude=0.0,
        longitude=0.0,
    )

    # M2's helper interpolates the id verbatim, so the resolved file is no
    # longer directly inside processed/.  The loader must not use it.
    legacy = paths.clean_dataset_path(unsafe.site_id)
    assert legacy.parent != paths.processed_dir

    with pytest.raises(AnalysisDataError, match="site_id"):
        load_cleaned_dataset(unsafe, "daily", paths=paths)
