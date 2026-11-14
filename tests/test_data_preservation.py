"""Regression tests for the committed datasets.

These hashes intentionally fail if a dataset changes.  Only update them
together with an explicit, reviewed data change.
"""

from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from forcasty.paths import PATHS, PROJECT_ROOT

EXPECTED_MD5 = {
    "data/raw/nasa_power_tunis.csv": "b6766b6ec4bfba964f497fb389b6db7a",
    "data/raw/nasa_power_tunis_hourly.csv": "b6e966f1d2e501af7e0bedf05fcc2aa8",
    "data/processed/forcasty_tunis_clean.csv": "b6766b6ec4bfba964f497fb389b6db7a",
    "data/processed/forcasty_tunis_hourly_clean.csv": "b6e966f1d2e501af7e0bedf05fcc2aa8",
    "data/processed/forcasty_tunis_features.csv": "9b0465579c87e63c739103f603805756",
}


@pytest.mark.parametrize(("relative_path", "expected"), EXPECTED_MD5.items())
def test_dataset_checksum_is_unchanged(relative_path: str, expected: str) -> None:
    path = PROJECT_ROOT / relative_path

    assert path.is_file(), f"missing dataset: {relative_path}"
    digest = hashlib.md5(path.read_bytes()).hexdigest()
    assert digest == expected


def test_raw_dataset_structure_and_range() -> None:
    path = PATHS.raw_dataset_path("tunis")
    df = pd.read_csv(path, parse_dates=["date"])

    assert df.shape == (2509, 5)
    assert list(df.columns) == [
        "date",
        "ALLSKY_SFC_SW_DWN",
        "T2M",
        "RH2M",
        "WS2M",
    ]
    assert df["date"].min() == pd.Timestamp("2020-01-01")
    assert df["date"].max() == pd.Timestamp("2026-11-13")
    # The historical days had no missing values; the tail contains NASA
    # POWER fill values that must remain missing, never zero-filled.
    assert int(df.isna().sum().sum()) == 19
    assert not df["date"].duplicated().any()
    assert len(pd.date_range(df["date"].min(), df["date"].max(), freq="D")) == len(df)


def test_hourly_dataset_structure_and_coverage() -> None:
    path = PATHS.hourly_dataset_path("tunis")
    df = pd.read_csv(path)
    stamps = pd.to_datetime(df["timestamp"], format="%Y-%m-%dT%H:%M:%SZ", utc=True)

    assert df.shape == (60216, 5)
    assert list(df.columns) == [
        "timestamp",
        "ALLSKY_SFC_SW_DWN",
        "T2M",
        "RH2M",
        "WS2M",
    ]
    assert stamps.min() == pd.Timestamp("2020-01-01T00:00:00Z")
    assert stamps.max() == pd.Timestamp("2026-11-13T23:00:00Z")
    assert stamps.is_monotonic_increasing
    assert not stamps.duplicated().any()
    assert len(pd.date_range(stamps.min(), stamps.max(), freq="h")) == len(df)
    assert (df["ALLSKY_SFC_SW_DWN"].dropna() >= 0).all()
    assert int(df["ALLSKY_SFC_SW_DWN"].isna().sum()) == 1776
    for column in df.columns[1:]:
        assert pd.api.types.is_float_dtype(df[column])


def test_cleaned_datasets_are_canonical() -> None:
    daily = pd.read_csv(PATHS.clean_dataset_path("tunis"), parse_dates=["date"])
    hourly = pd.read_csv(PATHS.hourly_clean_dataset_path("tunis"))
    stamps = pd.to_datetime(
        hourly["timestamp"], format="%Y-%m-%dT%H:%M:%SZ", utc=True
    )

    assert daily.shape == (2509, 5)
    assert daily["date"].is_monotonic_increasing
    assert not daily["date"].duplicated().any()
    assert int(daily.isna().sum().sum()) == 19

    assert hourly.shape == (60216, 5)
    assert stamps.is_monotonic_increasing
    assert not stamps.duplicated().any()
    assert int(hourly.isna().sum().sum()) == 1920
    # Missing GHI stays missing: no zero-filling happened.
    assert int((hourly["ALLSKY_SFC_SW_DWN"] == 0).sum()) < 60216


def test_feature_dataset_structure_and_causality_columns() -> None:
    path = PATHS.features_dataset_path("tunis")
    df = pd.read_csv(path)

    assert df.shape == (2192, 16)
    for column in (
        "ghi_lag_1",
        "ghi_lag_2",
        "ghi_lag_7",
        "ghi_rolling_7",
        "ghi_rolling_30",
    ):
        assert column in df.columns

    # The first row cannot have any lagged or rolling history.
    assert df["ghi_lag_1"].isna().sum() == 1
    assert df["ghi_lag_7"].isna().sum() == 7
    assert df["ghi_rolling_7"].isna().sum() == 7
    assert df["ghi_rolling_30"].isna().sum() == 30


def test_notebook_still_present() -> None:
    notebook = PROJECT_ROOT / "notebooks" / "01_exploratory_data_analysis.ipynb"

    assert notebook.is_file()
    assert notebook.stat().st_size > 0
