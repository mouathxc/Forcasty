# Forcasty

Hourly global horizontal irradiance (GHI) forecasting for Tunisia.

**Status:** milestones M0 (repository foundation) and M1 (NASA POWER hourly
acquisition) are complete. The repository now contains a validated hourly UTC
dataset for Tunis from 2020-01-01 through the latest complete day, an extended
daily dataset, provenance metadata, and an offline test suite. No forecasting
model, feature contract, or dashboard exists yet.

## Scope

- Target: hourly GHI (global horizontal irradiance), in `Wh/m²` per hourly
  interval.
- Horizon: 24 hours ahead (methodology milestone, not yet implemented).
- Initial site: Tunis (36.8065 N, 10.1815 E), configured in
  `config/sites.yaml`.
- Future expansion: additional Tunisian sites, one entry per site in the same
  configuration file.
- Forecast target is **irradiance**, not photovoltaic power. GHI forecasting
  and PV power forecasting are different problems.

## Data

### Hourly dataset (primary)

- File: `data/raw/nasa_power_tunis_hourly.csv`.
- Source: NASA POWER hourly point API, `community=RE`,
  `time-standard=UTC`.
- Parameters and units as returned by the API:
  - `ALLSKY_SFC_SW_DWN` — all-sky surface shortwave downward irradiance,
    `Wh/m²` (energy in the hour).
  - `T2M` — temperature at 2 m, `°C`.
  - `RH2M` — relative humidity at 2 m, `%`.
  - `WS2M` — wind speed at 2 m, `m/s`.
- Coverage: 2020-01-01T00:00Z through 2026-11-13T23:00Z, 60,216 rows.
- Timestamps are UTC and represent the **start of each hour**. The column is
  `timestamp` in `YYYY-MM-DDTHH:MM:SSZ` form.
- One row per hour, unique and sorted, numeric columns.

### Daily dataset (extended, not replaced)

- File: `data/raw/nasa_power_tunis.csv`.
- Coverage: 2020-01-01 through 2026-11-13, 2,509 rows.
- Radiation unit is `kW-hr/m²/day`, which is **not** interchangeable with the
  hourly `Wh/m²`. Hourly values are energy per hour; daily values are daily
  insolation totals for the NASA POWER day.
- The original 2,192 historical rows are preserved byte-for-byte. Only the
  missing 2026 tail (317 rows) was appended.
- The daily history was acquired in NASA POWER **Local Solar Time (LST)**.
  LST is neither UTC nor Tunisia civil time. The tail is deliberately
  requested in LST as well; appending UTC days would shift day boundaries.

### Source characteristics

NASA POWER is a gridded product derived from reanalysis and satellite data
(MERRA-2, CERES SYN1deg), not a ground-station measurement. One grid cell
represents the Tunis area, not all of Tunisia. Missing values returned as the
NASA fill value `-999` are stored as empty cells; they are never zero-filled,
interpolated, or fabricated.

Current missing values (preserved as missing):

- Hourly: `ALLSKY_SFC_SW_DWN` 1,776 of 60,216; `T2M`, `RH2M`, `WS2M` 48 each.
- Daily: 19 cells across the appended tail. The 2020–2025 history had none.

## Repository layout

```text
config/sites.yaml          Site registry
src/forcasty/
  paths.py                 Project-root discovery and path containment
  config.py                Configuration loading and validation
  data/
    client.py              NASA POWER HTTP client, retries, payload validation
    acquisition.py         Hourly acquisition and daily tail extension
    metadata.py            SHA-256, UTC timestamps, atomic JSON writes
    inspection.py          Structural inspection
    validation.py          Raw dataset checks (daily; M2 will generalize)
    cleaning.py            Deterministic cleaning (daily; M2)
    transformation.py      Calendar, lag and rolling features (daily; M2)
data/raw/                  Raw downloads (daily and hourly)
data/processed/            Cleaned and feature datasets
data/metadata/             Acquisition provenance (JSON)
notebooks/                 Exploratory analysis
tests/                     Offline pytest suite
```

## Requirements

- Python 3.12 or newer (developed and tested on Python 3.14).
- Runtime libraries are pinned in `pyproject.toml` and
  `requirements.txt`.

No new dependency was introduced for M1; the client uses `requests`.

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -e ".[dev]"
```

If you only need the runtime dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Sites live in `config/sites.yaml`:

```yaml
default_site: tunis
sites:
  tunis:
    name: Tunis
    latitude: 36.8065
    longitude: 10.1815
```

Rules enforced by `forcasty.config`:

- At least one site is required.
- Site ids must match `^[a-z0-9][a-z0-9_-]{0,63}$` (they appear in filenames).
- `name` must be non-empty.
- `latitude` must be within `[-90, 90]` and `longitude` within
  `[-180, 180]`, as finite numbers.
- Unknown fields, duplicate YAML keys and unknown `default_site` values are
  rejected.

Every path written by the package is resolved against the project root and is
rejected if it would leave the repository. The root is found from
`pyproject.toml`, or overridden with the `FORCASTY_ROOT` environment variable.

## Acquisition

```bash
# Hourly UTC dataset (2020-01-01 through the last complete UTC day)
python -m forcasty.data.acquisition hourly --site tunis

# Extend the daily dataset with only the missing tail
python -m forcasty.data.acquisition daily-tail --site tunis

# Both, rebuilding hourly data when it already exists
python -m forcasty.data.acquisition all --site tunis --force
```

Useful options:

- `--start YYYYMMDD` — hourly start date (default `20200101`).
- `--end YYYYMMDD` — end date (default: yesterday in UTC).
- `--parameters A,B,C` — override the parameter list (maximum 15).
- `--max-retries N` — transient-failure retries (default 4).
- `--force` — allow rebuilding an existing hourly file.

Behavior:

- Requests are split into inclusive yearly chunks with no overlap or gap.
- `time-standard` is always sent explicitly and the response value is checked.
- Transient failures (connection errors, 429, 5xx) retry with exponential
  backoff; permanent 4xx responses do not retry.
- Invalid JSON, API error payloads, empty parameter data, missing parameters
  and time-standard mismatches raise instead of producing an empty dataset.
- The hourly CLI refuses to overwrite an existing file without `--force`.
- The daily CLI only appends; it never rewrites historical rows.

Each successful run writes provenance to `data/metadata/`:
`nasa_power_tunis_hourly.json` and `nasa_power_tunis_daily.json`. These record
the endpoint, API version, sources, site and returned grid coordinates,
parameters and units, community, time standard, timestamp convention,
requested and actual ranges, acquisition time, counts, missing and duplicate
counts, fill value, chunk details, file path and SHA-256.

### Legacy daily pipeline

Cleaning, validation and feature generation still operate on the daily
dataset. They are preserved from M0 and will be adapted to the hourly target
in M2:

```bash
python -m forcasty.data.validation --site tunis
python -m forcasty.data.cleaning --site tunis
python -m forcasty.data.transformation --site tunis
```

## Tests

```bash
python -m pytest
```

The suite runs entirely offline. It covers project-root discovery,
configuration and invalid sites, path containment, dataset preservation, and
mocked acquisition: request construction, explicit UTC/LST handling, response
parsing, yearly chunk boundaries, retry and backoff behavior, invalid JSON and
API errors, sentinel handling, metadata and file hashing, daily prefix
preservation, and repository-contained outputs. NASA POWER is never contacted
by a test.

## Roadmap

| Milestone | Scope | Status |
| --- | --- | --- |
| M0 | Repository foundation, config, paths, packaging, tests | Complete |
| M1 | Hourly acquisition, chunking, provenance, timezone handling | Complete |
| M2 | Failing validation contracts and deterministic cleaning | Not started |
| M3+ | Solar geometry, EDA, features, models, evaluation, dashboard | Not started |

## Limitations

- NASA POWER values are modelled/gridded, not measured at a Tunisian station.
- Hourly missing values exist and are left missing; no gap filling is done.
- Daily data are LST-based while hourly data are UTC; they must not be mixed
  without an explicit, documented conversion.
- Daily and hourly radiation units differ and are not interchangeable.
- Validation, cleaning and feature code still target the daily dataset; that
  is M2 work.
- The daily tail may include NASA fill values that reduce usable days.
- There are no trained models and no dashboard.

## Author

Mouath Chouchane — Energy Engineering Student, Renewable Energy & Data
Science.

## License

MIT.
