import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "nasa_power_tunis.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "forcasty_tunis_clean.csv"


def clean_data(df):

    print("===== DATA CLEANING =====")

    df = df.copy()

    # --------------------------------------------------
    # 1. Convert date column
    # --------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    print("✓ Date column converted")

    # --------------------------------------------------
    # 2. Remove invalid dates
    # --------------------------------------------------

    invalid_dates = df["date"].isna().sum()

    if invalid_dates > 0:
        print(f"⚠ Removing {invalid_dates} invalid dates")

        df = df.dropna(subset=["date"])

    else:
        print("✓ No invalid dates")

    # --------------------------------------------------
    # 3. Remove duplicate dates
    # --------------------------------------------------

    duplicates = df["date"].duplicated().sum()

    if duplicates > 0:
        print(f"⚠ Removing {duplicates} duplicate dates")

        df = df.drop_duplicates(
            subset=["date"],
            keep="first"
        )

    else:
        print("✓ No duplicate dates")

    # --------------------------------------------------
    # 4. Sort chronologically
    # --------------------------------------------------

    df = df.sort_values("date")

    print("✓ Data sorted chronologically")

    # --------------------------------------------------
    # 5. Replace NASA missing-value marker
    # --------------------------------------------------

    df = df.replace(-999, pd.NA)

    print("✓ NASA missing-value markers handled")

    # --------------------------------------------------
    # 6. Convert numerical columns
    # --------------------------------------------------

    numerical_columns = [
        "ALLSKY_SFC_SW_DWN",
        "T2M",
        "RH2M",
        "WS2M"
    ]

    for column in numerical_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    print("✓ Numerical columns converted")

    # --------------------------------------------------
    # 7. Final report
    # --------------------------------------------------

    print("\nFinal dataset:")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nMissing values:")
    print(df.isna().sum())

    return df


def save_data(df):

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(f"\n✓ Clean dataset saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":

    print("Loading raw dataset...\n")

    df = pd.read_csv(INPUT_PATH)

    df = clean_data(df)

    save_data(df)