"""NASA POWER daily acquisition for a configured site.

The request construction is intentionally unchanged from the original
pipeline (same endpoint, community, parameters and date format).  Only the
location and output path now come from configuration instead of hardcoded
values, and an empty API payload is treated as an error rather than a
successful download of nothing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests

from forcasty.config import Site, load_config
from forcasty.paths import PATHS

NASA_POWER_DAILY_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMETERS = "ALLSKY_SFC_SW_DWN,T2M,RH2M,WS2M"
COMMUNITY = "RE"
DEFAULT_START_DATE = "20200101"
DEFAULT_END_DATE = "20251231"


def build_request_params(
    site: Site,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
) -> dict[str, str | float]:
    """Build the NASA POWER query string for ``site`` and a date range."""

    return {
        "parameters": PARAMETERS,
        "community": COMMUNITY,
        "longitude": site.longitude,
        "latitude": site.latitude,
        "start": start_date,
        "end": end_date,
        "format": "JSON",
    }


def fetch_nasa_power(
    site: Site,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    *,
    session: requests.Session | None = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """Download daily NASA POWER data for one site."""

    params = build_request_params(site, start_date, end_date)
    http = session if session is not None else requests

    print(f"Requesting NASA POWER data for {site.name} ({site.site_id})...")

    response = http.get(NASA_POWER_DAILY_URL, params=params, timeout=timeout)
    response.raise_for_status()

    data = response.json()
    parameters = data["properties"]["parameter"]

    if not parameters or not all(parameters.values()):
        raise RuntimeError(
            "NASA POWER returned no data for "
            f"{site.site_id} ({start_date}-{end_date}). "
            "Check the date range and the API response messages."
        )

    # Dates are the dictionary index; parameters share the same date keys.
    df = pd.DataFrame(parameters)
    df.index.name = "date"
    df = df.reset_index()

    # Convert YYYYMMDD -> datetime.
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

    return df


def raw_output_path(site: Site) -> Path:
    """Return the in-repository raw dataset path for ``site``."""

    return PATHS.raw_dataset_path(site.site_id)


def save_data(df: pd.DataFrame, site: Site, output_path: str | Path | None = None) -> Path:
    """Write the raw dataset inside the repository and return its path."""

    path = PATHS.resolve(output_path) if output_path is not None else raw_output_path(site)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

    print(f"Saved dataset to: {path}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch daily NASA POWER data.")
    parser.add_argument("--site", help="Site id from config/sites.yaml.")
    parser.add_argument("--start", default=DEFAULT_START_DATE, help="YYYYMMDD.")
    parser.add_argument("--end", default=DEFAULT_END_DATE, help="YYYYMMDD.")
    args = parser.parse_args(argv)

    site = load_config().get_site(args.site)
    df = fetch_nasa_power(site, args.start, args.end)
    save_data(df, site)

    print("\nFirst five rows:")
    print(df.head())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
