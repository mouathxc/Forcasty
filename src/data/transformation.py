import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forcasty_tunis_clean.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forcasty_tunis_features.csv"
)


def create_features(df):

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

    print("✓ Calendar features created")

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

    print("✓ Season feature created")

    # --------------------------------------------------
    # 4. Solar radiation lag features
    # --------------------------------------------------

    ghi = "ALLSKY_SFC_SW_DWN"

    df["ghi_lag_1"] = df[ghi].shift(1)
    df["ghi_lag_2"] = df[ghi].shift(2)
    df["ghi_lag_7"] = df[ghi].shift(7)

    print("✓ Lag features created")

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

    print("✓ Rolling features created")

    return df


def save_features(df):

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n✓ Feature dataset saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":

    print("Loading cleaned dataset...\n")

    df = pd.read_csv(INPUT_PATH)

    df["date"] = pd.to_datetime(df["date"])

    df = create_features(df)

    save_features(df)

    print("\n===== FINAL DATASET =====")
    print(df.head())
    print("\nShape:", df.shape)