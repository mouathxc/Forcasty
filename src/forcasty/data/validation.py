"""Raw dataset validation.

Behaviour is preserved from the original pipeline: structural problems are
reported, and only a missing/empty dataset or missing required columns raises.
Stronger, failing validation is scheduled for milestone M2.
"""

from __future__ import annotations

import argparse

import pandas as pd

from forcasty.config import load_config
from forcasty.paths import PATHS


def validate_data(df: pd.DataFrame) -> None:
    """Report dataset structure, duplicates, missing values and date range."""

    print("===== DATA VALIDATION =====")

    # 1. Dataset exists and isn't empty
    if df.empty:
        raise ValueError("Dataset is empty.")

    print("[ok] Dataset is not empty")

    # 2. Required columns
    required_columns = [
        "date",
        "ALLSKY_SFC_SW_DWN",
        "T2M",
        "RH2M",
        "WS2M",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    print("[ok] Required columns are present")

    # 3. Duplicate dates
    duplicate_dates = df["date"].duplicated().sum()

    if duplicate_dates > 0:
        print(f"[warning] Found {duplicate_dates} duplicate dates")
    else:
        print("[ok] No duplicate dates")

    # 4. Missing values
    missing_values = df.isnull().sum()

    print("\nMissing values:")
    print(missing_values)

    # 5. Date range
    dates = pd.to_datetime(df["date"])

    print("\nDate range:")
    print(f"Start: {dates.min()}")
    print(f"End:   {dates.max()}")

    # 6. Number of records
    print(f"\nNumber of records: {len(df)}")

    print("\n===== VALIDATION COMPLETE =====")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a raw dataset.")
    parser.add_argument("--site", help="Site id from config/sites.yaml.")
    args = parser.parse_args(argv)

    site = load_config().get_site(args.site)
    path = PATHS.raw_dataset_path(site.site_id)
    validate_data(pd.read_csv(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
