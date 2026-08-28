import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "nasa_power_tunis.csv"


def validate_data(df):

    print("===== DATA VALIDATION =====")

    # 1. Dataset exists and isn't empty
    if df.empty:
        raise ValueError("Dataset is empty.")

    print("✓ Dataset is not empty")

    # 2. Required columns
    required_columns = [
        "date",
        "ALLSKY_SFC_SW_DWN",
        "T2M",
        "RH2M",
        "WS2M"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    print("✓ Required columns are present")

    # 3. Duplicate dates
    duplicate_dates = df["date"].duplicated().sum()

    if duplicate_dates > 0:
        print(f"⚠ Found {duplicate_dates} duplicate dates")
    else:
        print("✓ No duplicate dates")

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


if __name__ == "__main__":

    df = pd.read_csv(INPUT_PATH)

    validate_data(df)