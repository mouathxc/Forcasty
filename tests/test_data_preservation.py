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
    "data/raw/nasa_power_tunis.csv": "ff6c7fc1f87ea909f424f3068af5c15b",
    "data/processed/forcasty_tunis_clean.csv": "ff6c7fc1f87ea909f424f3068af5c15b",
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

    assert df.shape == (2192, 5)
    assert list(df.columns) == [
        "date",
        "ALLSKY_SFC_SW_DWN",
        "T2M",
        "RH2M",
        "WS2M",
    ]
    assert df["date"].min() == pd.Timestamp("2020-01-01")
    assert df["date"].max() == pd.Timestamp("2025-12-31")
    assert int(df.isna().sum().sum()) == 0
    assert not df["date"].duplicated().any()


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
