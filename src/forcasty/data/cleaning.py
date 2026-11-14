"""Deterministic cleaning for a raw NASA POWER dataset.

The transformation logic is preserved from the legacy pipeline.  Only path
handling changes: input and output locations derive from configuration, and
writes are constrained to the repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from forcasty.config import Site, load_config
from forcasty.paths import PATHS


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a raw NASA POWER frame and return a new frame."""

    print("===== DATA CLEANING =====")

    df = df.copy()

    # --------------------------------------------------
    # 1. Convert date column
    # --------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    print("[ok] Date column converted")

    # --------------------------------------------------
    # 2. Remove invalid dates
    # --------------------------------------------------

    invalid_dates = df["date"].isna().sum()

    if invalid_dates > 0:
        print(f"[warning] Removing {invalid_dates} invalid dates")

        df = df.dropna(subset=["date"])

    else:
        print("[ok] No invalid dates")

    # --------------------------------------------------
    # 3. Remove duplicate dates
    # --------------------------------------------------

    duplicates = df["date"].duplicated().sum()

    if duplicates > 0:
        print(f"[warning] Removing {duplicates} duplicate dates")

        df = df.drop_duplicates(
            subset=["date"],
            keep="first",
        )

    else:
        print("[ok] No duplicate dates")

    # --------------------------------------------------
    # 4. Sort chronologically
    # --------------------------------------------------

    df = df.sort_values("date")

    print("[ok] Data sorted chronologically")

    # --------------------------------------------------
    # 5. Replace NASA missing-value marker
    # --------------------------------------------------

    df = df.replace(-999, pd.NA)

    print("[ok] NASA missing-value markers handled")

    # --------------------------------------------------
    # 6. Convert numerical columns
    # --------------------------------------------------

    numerical_columns = [
        "ALLSKY_SFC_SW_DWN",
        "T2M",
        "RH2M",
        "WS2M",
    ]

    for column in numerical_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    print("[ok] Numerical columns converted")

    # --------------------------------------------------
    # 7. Final report
    # --------------------------------------------------

    print("\nFinal dataset:")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nMissing values:")
    print(df.isna().sum())

    return df


def clean_output_path(site: Site) -> Path:
    """Return the in-repository cleaned dataset path for ``site``."""

    return PATHS.clean_dataset_path(site.site_id)


def save_data(
    df: pd.DataFrame,
    site: Site,
    output_path: str | Path | None = None,
) -> Path:
    """Write the cleaned dataset inside the repository and return its path."""

    path = (
        PATHS.resolve(output_path)
        if output_path is not None
        else clean_output_path(site)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

    print("\n[ok] Clean dataset saved to:")
    print(path)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clean a raw dataset.")
    parser.add_argument("--site", help="Site id from config/sites.yaml.")
    args = parser.parse_args(argv)

    site = load_config().get_site(args.site)

    print("Loading raw dataset...\n")
    raw_path = PATHS.raw_dataset_path(site.site_id)
    df = pd.read_csv(raw_path)

    df = clean_data(df)
    save_data(df, site)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
