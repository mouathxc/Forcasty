"""Dataset inspection: shapes, dtypes, ranges and missing values."""

from __future__ import annotations

import argparse

import pandas as pd

from forcasty.config import load_config
from forcasty.paths import PATHS


def inspect_data(df: pd.DataFrame) -> None:
    """Print a structural summary of ``df``."""

    print("\n===== DATASET SHAPE =====")
    print(df.shape)

    print("\n===== COLUMNS =====")
    print(df.columns.tolist())

    print("\n===== DATA TYPES =====")
    print(df.dtypes)

    print("\n===== FIRST 5 ROWS =====")
    print(df.head())

    print("\n===== LAST 5 ROWS =====")
    print(df.tail())

    print("\n===== MISSING VALUES =====")
    print(df.isnull().sum())

    print("\n===== BASIC STATISTICS =====")
    print(df.describe())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect a raw dataset.")
    parser.add_argument("--site", help="Site id from config/sites.yaml.")
    args = parser.parse_args(argv)

    site = load_config().get_site(args.site)
    path = PATHS.raw_dataset_path(site.site_id)
    inspect_data(pd.read_csv(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
