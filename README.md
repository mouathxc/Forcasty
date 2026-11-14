# Forcasty

Hourly global horizontal irradiance (GHI) forecasting for Tunisia.

**Status:** milestone M0 (repository foundation) is complete. The package
layout, configuration, path handling and tests are in place. The dataset
currently in the repository is the legacy **daily** NASA POWER extract for
Tunis; hourly acquisition is milestone M1 and has not been implemented yet.
No forecasting model or dashboard exists at this stage.

## Scope

- Target: hourly GHI (global horizontal irradiance), in `Wh/m²` per hour.
- Horizon: 24 hours ahead (methodology milestone, not yet implemented).
- Initial site: Tunis (36.8065 N, 10.1815 E), configured in
  `config/sites.yaml`.
- Future expansion: additional Tunisian sites, one entry per site in the same
  configuration file.
- Forecast target is **irradiance**, not photovoltaic power. GHI forecasting
  and PV power forecasting are different problems.

## Data

- Source: NASA POWER, `ALLSKY_SFC_SW_DWN`, `T2M`, `RH2M`, `WS2M`,
  `community=RE`.
- The committed dataset is **daily** (`kW-hr/m²/day` for radiation), covering
  2020-01-01 to 2025-12-31 for the NASA POWER grid cell nearest Tunis.
- NASA POWER is a modelled/reanalysis/satellite product (MERRA-2, CERES
  SYN1deg), not a ground-station measurement.
- NASA POWER reports timestamps in Local Solar Time (LST). LST is not UTC and
  not Tunisia civil time.
- Daily totals cannot be disaggregated into hourly values. Hourly forecasting
  requires the hourly endpoint, which is planned for M1.
- One grid cell represents the Tunis area, not all of Tunisia.

## Repository layout

```text
config/sites.yaml          Site registry
src/forcasty/
  paths.py                 Project-root discovery and path containment
  config.py                Configuration loading and validation
  data/
    acquisition.py         NASA POWER daily download
    inspection.py          Structural inspection
    validation.py          Raw dataset checks
    cleaning.py            Deterministic cleaning
    transformation.py      Calendar, lag and rolling features
data/raw/                  Immutable downloads
data/processed/            Cleaned and feature datasets
notebooks/                 Exploratory analysis
tests/                     Pytest suite
```

## Requirements

- Python 3.12 or newer (developed and tested on Python 3.14).
- Runtime libraries are pinned in `pyproject.toml` and
  `requirements.txt`.

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

## Usage (legacy daily pipeline)

The existing daily pipeline is preserved for reference and will be replaced by
the hourly pipeline in M1.

```bash
python -m forcasty.data.acquisition --site tunis
python -m forcasty.data.validation --site tunis
python -m forcasty.data.cleaning --site tunis
python -m forcasty.data.transformation --site tunis
```

The date range defaults to 2020-01-01 through 2025-12-31 and can be changed
with `--start` and `--end` (`YYYYMMDD`).

## Tests

```bash
python -m pytest
```

The suite runs offline and covers project-root discovery, configuration
loading, invalid site configurations, path containment and dataset
preservation. NASA POWER is not contacted by any test.

## Roadmap

| Milestone | Scope | Status |
| --- | --- | --- |
| M0 | Repository foundation, config, paths, packaging, tests | Complete |
| M1 | Hourly acquisition, chunking, provenance, timezone handling | Not started |
| M2 | Failing validation contracts and deterministic cleaning | Not started |
| M3+ | Solar geometry, EDA, features, models, evaluation, dashboard | Not started |

## Limitations

- The current dataset is daily and cannot support hourly forecasting.
- Validation and cleaning still report progress with `print` and do not yet
  enforce the full physical contract; that is M2.
- There are no trained models and no dashboard.
- Error metrics would describe agreement with a modelled product, not with
  measured irradiance.

## License

MIT.
