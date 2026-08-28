import requests
import pandas as pd
from pathlib import Path


# Tunisia location
LATITUDE = 36.8065
LONGITUDE = 10.1815

START_DATE = "20200101"
END_DATE = "20201231"

OUTPUT_PATH = Path("data/raw/nasa_power_tunis.csv")


def fetch_nasa_power():
    """Download daily solar and meteorological data from NASA POWER."""

    url = "https://power.larc.nasa.gov/api/temporal/daily/point"

    params = {
        "parameters": "ALLSKY_SFC_SW_DWN,T2M,RH2M,WS2M",
        "community": "RE",
        "longitude": LONGITUDE,
        "latitude": LATITUDE,
        "start": START_DATE,
        "end": END_DATE,
        "format": "JSON",
    }

    print("Requesting NASA POWER data...")

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()

    values = data["properties"]["parameter"]

    df = pd.DataFrame(values)

    df = df.T.reset_index()
    df = df.rename(columns={"index": "date"})

    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

    return df


def save_data(df):
    """Save raw dataset to CSV."""

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved dataset to: {OUTPUT_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    df = fetch_nasa_power()
    save_data(df)

    print("\nFirst five rows:")
    print(df.head())