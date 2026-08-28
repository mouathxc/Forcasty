import requests
import pandas as pd
from pathlib import Path


LATITUDE = 36.8065
LONGITUDE = 10.1815

START_DATE = "20200101"
END_DATE = "20251231"

OUTPUT_PATH = Path("C:\\Users\\Mouath\\Desktop\\Forcasty\\data\\raw\\nasa_power_tunis.csv")


def fetch_nasa_power():

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

    parameters = data["properties"]["parameter"]

    # Create dataframe directly from the parameter dictionaries
    df = pd.DataFrame(parameters)

    # Dates are the dictionary index
    df.index.name = "date"

    # Move date from index into a column
    df = df.reset_index()

    # Convert YYYYMMDD → datetime
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

    return df


def save_data(df):

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