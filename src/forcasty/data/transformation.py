"""Calendar, lag and rolling features.

The causal logic is preserved exactly from the legacy pipeline: every lag uses
a positive ``shift`` and every rolling statistic is computed on ``shift(1)``
first, so no value at or after the current timestamp contributes to its own
features.  Only path handling is configuration-driven.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from forcasty.config import Site, load_config
from forcasty.paths import PATHS


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with calendar, lag and rolling features."""

    df = df.copy()

    print("===== FEATURE ENGINEERING =====")

    # --------------------------------------------------
    # 1. Make sure data is sorted
    # --------------------------------------------------

    df = df.sort_values("date")

    # --------------------------------------------------
    # 2. Calendar features
    # --------------------------------------------------

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_year"] = df["date"].dt.dayofyear
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    print("[ok] Calendar features created")

    # --------------------------------------------------
    # 3. Season
    # --------------------------------------------------

    def get_season(month):

        if month in [12, 1, 2]:
            return "winter"

        elif month in [3, 4, 5]:
            return "spring"

        elif month in [6, 7, 8]:
            return "summer"

        else:
            return "autumn"

    df["season"] = df["month"].apply(get_season)

    print("[ok] Season feature created")

    # --------------------------------------------------
    # 4. Solar radiation lag features
    # --------------------------------------------------

    ghi = "ALLSKY_SFC_SW_DWN"

    df["ghi_lag_1"] = df[ghi].shift(1)
    df["ghi_lag_2"] = df[ghi].shift(2)
    df["ghi_lag_7"] = df[ghi].shift(7)

    print("[ok] Lag features created")

    # --------------------------------------------------
    # 5. Rolling statistics
    # --------------------------------------------------

    df["ghi_rolling_7"] = (
        df[ghi]
        .shift(1)
        .rolling(window=7)
        .mean()
    )

    df["ghi_rolling_30"] = (
        df[ghi]
        .shift(1)
        .rolling(window=30)
        .mean()
    )

    print("[ok] Rolling features created")

    return df


def features_output_path(site: Site) -> Path:
    """Return the in-repository feature dataset path for ``site``."""

    return PATHS.features_dataset_path(site.site_id)


def save_features(
    df: pd.DataFrame,
    site: Site,
    output_path: str | Path | None = None,
) -> Path:
    """Write the feature dataset inside the repository and return its path."""

    path = (
        PATHS.resolve(output_path)
        if output_path is not None
        else features_output_path(site)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

    print("\n[ok] Feature dataset saved to:")
    print(path)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build features from cleaned data.")
    parser.add_argument("--site", help="Site id from config/sites.yaml.")
    args = parser.parse_args(argv)

    site = load_config().get_site(args.site)

    print("Loading cleaned dataset...\n")
    clean_path = PATHS.clean_dataset_path(site.site_id)
    df = pd.read_csv(clean_path)

    df["date"] = pd.to_datetime(df["date"])

    df = create_features(df)
    save_features(df, site)

    print("\n===== FINAL DATASET =====")
    print(df.head())
    print("\nShape:", df.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
