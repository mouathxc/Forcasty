"""Validation contracts for the daily and hourly NASA POWER datasets.

The two datasets are validated independently because they use different
temporal conventions and radiation units:

* daily  — calendar days in NASA POWER Local Solar Time (LST), GHI in
  ``kW-hr/m^2/day``;
* hourly — UTC timestamps at the start of each hour, GHI in ``Wh/m^2``.

Policy
------
Structural defects are errors and block cleaning:

* empty dataset, missing required columns;
* timestamps that cannot be parsed;
* duplicate timestamps;
* non-numeric values in numeric columns.

The following are warnings and do not block cleaning:

* unsorted timestamps (canonical cleaning sorts them);
* missing calendar days or hours (never filled);
* timestamps off the expected one-day/one-hour grid;
* physical implausibilities (negative GHI or wind speed, humidity outside
  0–100 %, temperature outside the documented broad plausibility range).

Physical warnings never cause values to be changed. NASA's ``-999`` marker is
reported separately from genuine missing values and from invalid values.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from forcasty.config import load_config
from forcasty.paths import PATHS

DAILY = "daily"
HOURLY = "hourly"
FREQUENCIES = (DAILY, HOURLY)

DAILY_REQUIRED_COLUMNS = (
    "date",
    "ALLSKY_SFC_SW_DWN",
    "T2M",
    "RH2M",
    "WS2M",
)
HOURLY_REQUIRED_COLUMNS = (
    "timestamp",
    "ALLSKY_SFC_SW_DWN",
    "T2M",
    "RH2M",
    "WS2M",
)
NUMERIC_COLUMNS = ("ALLSKY_SFC_SW_DWN", "T2M", "RH2M", "WS2M")
TIMESTAMP_COLUMN = {DAILY: "date", HOURLY: "timestamp"}
TIME_STANDARD = {DAILY: "LST", HOURLY: "UTC"}
GHI_UNITS = {DAILY: "kW-hr/m^2/day", HOURLY: "Wh/m^2"}

NASA_FILL_VALUE = -999.0

# A deliberately broad physical screen, not a Tunisian climatology: the
# lowest/highest surface air temperatures ever observed on Earth are roughly
# -89 C and +57 C. Values outside this range are almost certainly corrupt.
T2M_PLAUSIBLE_MIN = -90.0
T2M_PLAUSIBLE_MAX = 60.0


class ValidationError(ValueError):
    """Raised when a dataset violates a structural validation contract."""

    def __init__(self, message: str, report: "ValidationReport | None" = None):
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class ValidationIssue:
    """One reported validation finding."""

    severity: str
    code: str
    message: str
    count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "count": self.count,
        }


@dataclass
class ValidationReport:
    """Structured validation outcome for one dataset."""

    frequency: str
    row_count: int
    timestamp_column: str
    time_standard: str
    ghi_units: str
    first_timestamp: str | None = None
    last_timestamp: str | None = None
    duplicate_timestamp_count: int = 0
    invalid_timestamp_count: int = 0
    missing_interval_count: int = 0
    irregular_spacing_count: int = 0
    unsorted_timestamps: bool = False
    missing_value_counts: dict[str, int] = field(default_factory=dict)
    fill_value_counts: dict[str, int] = field(default_factory=dict)
    invalid_numeric_counts: dict[str, int] = field(default_factory=dict)
    invalid_observation_counts: dict[str, int] = field(default_factory=dict)
    warnings: list[ValidationIssue] = field(default_factory=list)
    errors: list[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    @property
    def missing_value_total(self) -> int:
        return int(sum(self.missing_value_counts.values()))

    def add_warning(self, code: str, message: str, count: int = 1) -> None:
        self.warnings.append(
            ValidationIssue("warning", code, message, int(count))
        )

    def add_error(self, code: str, message: str, count: int = 1) -> None:
        self.errors.append(ValidationIssue("error", code, message, int(count)))

    def raise_if_invalid(self) -> None:
        if self.errors:
            details = "; ".join(
                f"{issue.code}: {issue.message}" for issue in self.errors
            )
            raise ValidationError(
                f"{self.frequency} dataset failed validation: {details}",
                report=self,
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "frequency": self.frequency,
            "timestamp_column": self.timestamp_column,
            "time_standard": self.time_standard,
            "ghi_units": self.ghi_units,
            "row_count": self.row_count,
            "first_timestamp": self.first_timestamp,
            "last_timestamp": self.last_timestamp,
            "duplicate_timestamp_count": self.duplicate_timestamp_count,
            "invalid_timestamp_count": self.invalid_timestamp_count,
            "missing_interval_count": self.missing_interval_count,
            "irregular_spacing_count": self.irregular_spacing_count,
            "unsorted_timestamps": self.unsorted_timestamps,
            "missing_value_counts": dict(self.missing_value_counts),
            "missing_value_total": self.missing_value_total,
            "fill_value_counts": dict(self.fill_value_counts),
            "invalid_numeric_counts": dict(self.invalid_numeric_counts),
            "invalid_observation_counts": dict(
                self.invalid_observation_counts
            ),
            "warnings": [issue.to_dict() for issue in self.warnings],
            "errors": [issue.to_dict() for issue in self.errors],
            "is_valid": self.is_valid,
        }


def _required_columns(frequency: str) -> tuple[str, ...]:
    return (
        HOURLY_REQUIRED_COLUMNS
        if frequency == HOURLY
        else DAILY_REQUIRED_COLUMNS
    )


def parse_timestamps(
    series: pd.Series, frequency: str
) -> tuple[pd.Series, int]:
    """Return parsed timestamps and the count of daily time components."""

    if frequency == HOURLY:
        parsed = pd.to_datetime(
            series, errors="coerce", format="mixed", utc=True
        )
        return parsed, 0

    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    non_midnight = 0
    if not parsed.isna().all():
        normalized = parsed.dt.normalize()
        non_midnight = int((parsed != normalized).sum())
        parsed = normalized
    return parsed, non_midnight


def format_timestamp(value: Any, frequency: str) -> str:
    stamp = pd.Timestamp(value)
    if frequency == HOURLY:
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize("UTC")
        else:
            stamp = stamp.tz_convert("UTC")
        return stamp.strftime("%Y-%m-%dT%H:%M:%SZ")
    return stamp.strftime("%Y-%m-%d")


def _numeric_columns_report(
    df: pd.DataFrame, report: ValidationReport
) -> dict[str, pd.Series]:
    """Populate numeric/fill/missing counters and return screened columns."""

    screened: dict[str, pd.Series] = {}
    for column in NUMERIC_COLUMNS:
        raw = df[column]
        numeric = pd.to_numeric(raw, errors="coerce")
        unconvertible = numeric.isna() & raw.notna()
        non_finite = numeric.isin([float("inf"), float("-inf")])
        invalid = unconvertible | non_finite
        fill = numeric.eq(NASA_FILL_VALUE)
        missing = (numeric.isna() & ~unconvertible) | fill

        report.invalid_numeric_counts[column] = int(invalid.sum())
        report.fill_value_counts[column] = int(fill.sum())
        report.missing_value_counts[column] = int(missing.sum())
        screened[column] = numeric.mask(fill).mask(invalid)

    total_invalid = int(sum(report.invalid_numeric_counts.values()))
    if total_invalid:
        report.add_error(
            "INVALID_NUMERIC_VALUES",
            "numeric columns contain values that cannot be interpreted "
            f"as numbers: {report.invalid_numeric_counts}",
            total_invalid,
        )

    if report.missing_value_total:
        report.add_warning(
            "MISSING_VALUES",
            f"missing values retained without filling: "
            f"{report.missing_value_counts}",
            report.missing_value_total,
        )

    total_fill = int(sum(report.fill_value_counts.values()))
    if total_fill:
        report.add_warning(
            "NASA_FILL_VALUES",
            f"NASA missing marker {NASA_FILL_VALUE:g} found: "
            f"{report.fill_value_counts}",
            total_fill,
        )

    return screened


def _physical_checks(
    screened: dict[str, pd.Series], report: ValidationReport
) -> None:
    ghi = screened["ALLSKY_SFC_SW_DWN"]
    negative_ghi = int((ghi < 0).sum())
    if negative_ghi:
        report.invalid_observation_counts["negative_ghi"] = negative_ghi
        report.add_warning(
            "NEGATIVE_GHI",
            f"{negative_ghi} GHI value(s) are negative "
            f"(unit: {report.ghi_units}); values were not modified.",
            negative_ghi,
        )

    rh = screened["RH2M"]
    rh_below = int((rh < 0).sum())
    rh_above = int((rh > 100).sum())
    if rh_below:
        report.invalid_observation_counts["relative_humidity_below_0"] = (
            rh_below
        )
    if rh_above:
        report.invalid_observation_counts["relative_humidity_above_100"] = (
            rh_above
        )
    if rh_below or rh_above:
        report.add_warning(
            "RELATIVE_HUMIDITY_OUT_OF_RANGE",
            f"relative humidity outside 0-100 %: {rh_below} below 0, "
            f"{rh_above} above 100; values were not modified.",
            rh_below + rh_above,
        )

    wind = screened["WS2M"]
    negative_wind = int((wind < 0).sum())
    if negative_wind:
        report.invalid_observation_counts["negative_wind_speed"] = (
            negative_wind
        )
        report.add_warning(
            "NEGATIVE_WIND_SPEED",
            f"{negative_wind} wind speed value(s) are negative; "
            "values were not modified.",
            negative_wind,
        )

    temperature = screened["T2M"]
    temperature_out = int(
        ((temperature < T2M_PLAUSIBLE_MIN) | (temperature > T2M_PLAUSIBLE_MAX))
        .fillna(False)
        .sum()
    )
    if temperature_out:
        report.invalid_observation_counts["temperature_out_of_range"] = (
            temperature_out
        )
        report.add_warning(
            "TEMPERATURE_OUT_OF_RANGE",
            f"{temperature_out} temperature value(s) outside the broad "
            f"physical screen [{T2M_PLAUSIBLE_MIN:g}, "
            f"{T2M_PLAUSIBLE_MAX:g}] C; values were not modified.",
            temperature_out,
        )


def validate_dataset(df: pd.DataFrame, frequency: str) -> ValidationReport:
    """Validate a daily or hourly frame without mutating it."""

    if frequency not in FREQUENCIES:
        raise ValueError(
            f"frequency must be 'daily' or 'hourly', got {frequency!r}."
        )

    report = ValidationReport(
        frequency=frequency,
        row_count=int(len(df)),
        timestamp_column=TIMESTAMP_COLUMN[frequency],
        time_standard=TIME_STANDARD[frequency],
        ghi_units=GHI_UNITS[frequency],
    )

    if df.empty:
        report.add_error("EMPTY_DATASET", "dataset contains no rows")
        return report

    required = _required_columns(frequency)
    missing_columns = [column for column in required if column not in df.columns]
    if missing_columns:
        report.add_error(
            "MISSING_COLUMNS",
            f"missing required column(s): {missing_columns}",
            len(missing_columns),
        )
        return report

    timestamp_column = report.timestamp_column
    parsed, non_midnight = parse_timestamps(df[timestamp_column], frequency)

    invalid_count = int(parsed.isna().sum())
    report.invalid_timestamp_count = invalid_count
    if invalid_count:
        report.add_error(
            "INVALID_TIMESTAMPS",
            f"{invalid_count} timestamp(s) could not be parsed",
            invalid_count,
        )

    valid = parsed.dropna()
    if not valid.empty:
        report.first_timestamp = format_timestamp(valid.min(), frequency)
        report.last_timestamp = format_timestamp(valid.max(), frequency)
        sorted_valid = valid.is_monotonic_increasing
        report.unsorted_timestamps = not sorted_valid
        if report.unsorted_timestamps:
            report.add_warning(
                "UNSORTED_TIMESTAMPS",
                "timestamps are not strictly increasing; canonical cleaning "
                "will sort them",
            )

        duplicates = int(valid.duplicated().sum())
        report.duplicate_timestamp_count = duplicates
        if duplicates:
            report.add_error(
                "DUPLICATE_TIMESTAMPS",
                f"{duplicates} duplicate timestamp(s); cleaning does not "
                "silently discard records",
                duplicates,
            )

        unique = pd.DatetimeIndex(sorted(valid.unique()))
        freq = "h" if frequency == HOURLY else "D"
        expected = pd.date_range(unique[0], unique[-1], freq=freq)
        missing_intervals = int(len(expected.difference(unique)))
        report.missing_interval_count = missing_intervals
        if missing_intervals:
            unit = "hour" if frequency == HOURLY else "calendar day"
            report.add_warning(
                "MISSING_INTERVALS",
                f"{missing_intervals} missing {unit}(s) between "
                f"{report.first_timestamp} and {report.last_timestamp}; "
                "they are not filled",
                missing_intervals,
            )

        expected_seconds = 3600.0 if frequency == HOURLY else 86400.0
        if len(unique) > 1:
            seconds = (
                pd.Series(unique).diff().dropna().dt.total_seconds()
            )
            irregular = int(
                ((seconds <= 0) | (seconds % expected_seconds != 0)).sum()
            )
            report.irregular_spacing_count = irregular
            if irregular:
                report.add_warning(
                    "IRREGULAR_TIMESTAMP_SPACING",
                    f"{irregular} timestamp interval(s) are not on the "
                    f"expected {expected_seconds:g}-second grid",
                    irregular,
                )

        if frequency == HOURLY:
            off_hour = int(
                (
                    (valid.dt.minute != 0)
                    | (valid.dt.second != 0)
                    | (valid.dt.microsecond != 0)
                ).sum()
            )
            if off_hour:
                report.add_warning(
                    "HOURLY_NOT_START_OF_HOUR",
                    f"{off_hour} timestamp(s) are not at the start of an hour",
                    off_hour,
                )
        elif non_midnight:
            report.add_warning(
                "DAILY_TIME_COMPONENT",
                f"{non_midnight} daily timestamp(s) carried a time component; "
                "validation treats them as calendar days",
                non_midnight,
            )

    screened = _numeric_columns_report(df, report)
    _physical_checks(screened, report)
    return report


def validate_daily(df: pd.DataFrame) -> ValidationReport:
    """Validate a daily LST frame."""

    return validate_dataset(df, DAILY)


def validate_hourly(df: pd.DataFrame) -> ValidationReport:
    """Validate an hourly UTC frame."""

    return validate_dataset(df, HOURLY)


def print_report(report: ValidationReport) -> None:
    """Print a concise human-readable validation summary."""

    print(f"===== {report.frequency.upper()} VALIDATION =====")
    print(f"Rows: {report.row_count}")
    print(
        "Range: "
        f"{report.first_timestamp or '<none>'} .. "
        f"{report.last_timestamp or '<none>'} "
        f"({report.time_standard}, GHI {report.ghi_units})"
    )
    print(
        "Duplicates: "
        f"{report.duplicate_timestamp_count}; invalid timestamps: "
        f"{report.invalid_timestamp_count}; missing intervals: "
        f"{report.missing_interval_count}"
    )
    print(f"Missing values: {report.missing_value_counts}")
    print(f"NASA -999 markers: {report.fill_value_counts}")

    for issue in report.warnings:
        print(f"[warning] {issue.message}")
    for issue in report.errors:
        print(f"[error] {issue.message}")

    print("Status:", "valid" if report.is_valid else "invalid")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate NASA POWER data.")
    parser.add_argument(
        "frequency",
        nargs="?",
        choices=(DAILY, HOURLY, "all"),
        default=DAILY,
        help="Dataset to validate (default: daily).",
    )
    parser.add_argument("--site", help="Site id from config/sites.yaml.")
    args = parser.parse_args(argv)

    site = load_config().get_site(args.site)
    frequencies = FREQUENCIES if args.frequency == "all" else (
        args.frequency,
    )

    exit_code = 0
    for frequency in frequencies:
        path = (
            PATHS.hourly_dataset_path(site.site_id)
            if frequency == HOURLY
            else PATHS.raw_dataset_path(site.site_id)
        )
        try:
            df = pd.read_csv(path)
        except OSError as exc:
            print(f"[error] could not read {path}: {exc}")
            exit_code = 1
            continue

        report = validate_dataset(df, frequency)
        print_report(report)
        if not report.is_valid:
            exit_code = 1
        print()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
