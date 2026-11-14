"""Canonical cleaning for the daily and hourly NASA POWER datasets.

Cleaning is deliberately narrow.  It parses timestamps, converts numeric
columns, replaces NASA's ``-999`` marker with a genuine missing value, and
sorts chronologically.  It never interpolates, fills, repairs, or drops
physical values, and it never discards duplicate or invalid records: those are
structural validation errors, so the run stops and asks for a decision.

Running the pipeline on an already-cleaned dataset produces the same result.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from forcasty import __version__
from forcasty.config import Site, load_config
from forcasty.data.metadata import iso_z, sha256_file, utc_now, write_json_atomic
from forcasty.data.validation import (
    DAILY,
    FREQUENCIES,
    GHI_UNITS,
    HOURLY,
    NASA_FILL_VALUE,
    NUMERIC_COLUMNS,
    TIME_STANDARD,
    TIMESTAMP_COLUMN,
    ValidationReport,
    format_timestamp,
    parse_timestamps,
    validate_dataset,
)
from forcasty.paths import PATHS, ProjectPaths

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CleaningResult:
    """A cleaned frame plus the transformations that produced it."""

    frame: pd.DataFrame
    frequency: str
    validation: ValidationReport
    operations: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ProcessedDataset:
    """Result of cleaning one source file and writing its artefacts."""

    frame: pd.DataFrame
    frequency: str
    validation: ValidationReport
    operations: tuple[dict[str, Any], ...]
    output_path: Path
    report_path: Path
    report: dict[str, Any]


def clean_dataset(df: pd.DataFrame, frequency: str) -> CleaningResult:
    """Return a canonically cleaned copy of ``df``.

    The input frame is never mutated.  Structural validation runs first; if it
    reports errors (invalid or duplicate timestamps, non-numeric values, empty
    data, missing columns), :class:`ValidationError` is raised.
    """

    if frequency not in FREQUENCIES:
        raise ValueError(
            f"frequency must be 'daily' or 'hourly', got {frequency!r}."
        )

    validation = validate_dataset(df, frequency)
    validation.raise_if_invalid()

    cleaned = df.copy()
    timestamp_column = TIMESTAMP_COLUMN[frequency]

    parsed, _ = parse_timestamps(cleaned[timestamp_column], frequency)
    formatted = parsed.map(lambda value: format_timestamp(value, frequency))
    cleaned[timestamp_column] = formatted

    already_sorted = bool(parsed.is_monotonic_increasing)

    fill_counts: dict[str, int] = {}
    for column in NUMERIC_COLUMNS:
        try:
            numeric = pd.to_numeric(cleaned[column], errors="raise").astype(
                "float64"
            )
        except (TypeError, ValueError) as exc:  # defensive; validation caught it
            raise ValueError(
                f"Column {column!r} could not be converted to numeric values."
            ) from exc
        marker = numeric.eq(NASA_FILL_VALUE)
        fill_counts[column] = int(marker.sum())
        cleaned[column] = numeric.mask(marker)

    ordered = cleaned.assign(
        __forcasty_sort=parsed
    ).sort_values("__forcasty_sort", kind="mergesort")
    ordered = ordered.drop(columns="__forcasty_sort").reset_index(drop=True)

    operations: tuple[dict[str, Any], ...] = (
        {
            "operation": "parse_timestamps",
            "column": timestamp_column,
            "time_standard": TIME_STANDARD[frequency],
            "invalid_records_removed": 0,
        },
        {
            "operation": "convert_numeric",
            "columns": list(NUMERIC_COLUMNS),
            "invalid_values_replaced": 0,
        },
        {
            "operation": "replace_fill_values",
            "marker": NASA_FILL_VALUE,
            "counts": fill_counts,
        },
        {
            "operation": "sort_chronologically",
            "changed": not already_sorted,
        },
        {
            "operation": "preserve_missing_values",
            "interpolation": False,
            "rows_dropped": 0,
        },
    )

    return CleaningResult(
        frame=ordered,
        frequency=frequency,
        validation=validation,
        operations=operations,
    )


def clean_daily(df: pd.DataFrame) -> CleaningResult:
    """Clean a daily LST frame."""

    return clean_dataset(df, DAILY)


def clean_hourly(df: pd.DataFrame) -> CleaningResult:
    """Clean an hourly UTC frame."""

    return clean_dataset(df, HOURLY)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible daily cleaner returning only the frame."""

    return clean_daily(df).frame


def source_dataset_path(
    site: Site, frequency: str, paths: ProjectPaths = PATHS
) -> Path:
    """Return the raw input path for ``frequency``."""

    if frequency == HOURLY:
        return paths.hourly_dataset_path(site.site_id)
    if frequency == DAILY:
        return paths.raw_dataset_path(site.site_id)
    raise ValueError(f"Unknown frequency {frequency!r}.")


def cleaned_dataset_path(
    site: Site, frequency: str, paths: ProjectPaths = PATHS
) -> Path:
    """Return the canonical processed output path for ``frequency``."""

    if frequency == HOURLY:
        return paths.hourly_clean_dataset_path(site.site_id)
    if frequency == DAILY:
        return paths.clean_dataset_path(site.site_id)
    raise ValueError(f"Unknown frequency {frequency!r}.")


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        frame.to_csv(temporary, index=False, lineterminator="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_processing_report(
    *,
    site: Site,
    frequency: str,
    source_path: Path,
    source_sha256: str,
    output_path: Path,
    output_sha256: str,
    input_row_count: int,
    output_row_count: int,
    validation: ValidationReport,
    operations: tuple[dict[str, Any], ...],
    paths: ProjectPaths = PATHS,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build the JSON-serialisable processing report.

    Everything except ``processing_timestamp_utc`` is a deterministic function
    of the input data and configuration.
    """

    return {
        "site_id": site.site_id,
        "site_name": site.name,
        "frequency": frequency,
        "timestamp_convention": validation.time_standard,
        "time_standard": TIME_STANDARD[frequency],
        "ghi_units": GHI_UNITS[frequency],
        "source_file": source_path.relative_to(paths.root).as_posix(),
        "source_sha256": source_sha256,
        "output_file": output_path.relative_to(paths.root).as_posix(),
        "output_sha256": output_sha256,
        "input_row_count": int(input_row_count),
        "output_row_count": int(output_row_count),
        "timestamp_range": {
            "first": validation.first_timestamp,
            "last": validation.last_timestamp,
        },
        "duplicate_timestamp_count": validation.duplicate_timestamp_count,
        "invalid_timestamp_count": validation.invalid_timestamp_count,
        "missing_interval_count": validation.missing_interval_count,
        "irregular_spacing_count": validation.irregular_spacing_count,
        "unsorted_timestamps": validation.unsorted_timestamps,
        "missing_values_by_column": dict(validation.missing_value_counts),
        "fill_value_counts_by_column": dict(validation.fill_value_counts),
        "invalid_numeric_counts": dict(validation.invalid_numeric_counts),
        "invalid_observation_counts": dict(
            validation.invalid_observation_counts
        ),
        "cleaning_operations": list(operations),
        "validation_warnings": [
            issue.to_dict() for issue in validation.warnings
        ],
        "validation_errors": [
            issue.to_dict() for issue in validation.errors
        ],
        "processing_timestamp_utc": iso_z(
            now if now is not None else utc_now()
        ),
        "software_version": __version__,
    }


def process_dataset(
    site: Site,
    frequency: str,
    *,
    paths: ProjectPaths = PATHS,
    now: datetime | None = None,
) -> ProcessedDataset:
    """Clean one raw dataset, write its CSV and its processing report.

    The raw file is only read; its SHA-256 is recorded before writing anything.
    """

    if frequency not in FREQUENCIES:
        raise ValueError(
            f"frequency must be 'daily' or 'hourly', got {frequency!r}."
        )

    source_path = source_dataset_path(site, frequency, paths)
    if not source_path.is_file():
        raise FileNotFoundError(f"Raw dataset not found: {source_path}")

    source_sha256 = sha256_file(source_path)
    raw = pd.read_csv(source_path)

    result = clean_dataset(raw, frequency)

    output_path = cleaned_dataset_path(site, frequency, paths)
    _write_csv_atomic(result.frame, output_path)
    output_sha256 = sha256_file(output_path)

    report = build_processing_report(
        site=site,
        frequency=frequency,
        source_path=source_path,
        source_sha256=source_sha256,
        output_path=output_path,
        output_sha256=output_sha256,
        input_row_count=len(raw),
        output_row_count=len(result.frame),
        validation=result.validation,
        operations=result.operations,
        paths=paths,
        now=now,
    )
    report_path = paths.processing_report_path(site.site_id, frequency)
    write_json_atomic(report, report_path)

    logger.info(
        "Cleaned %s %s: %d rows -> %s",
        frequency,
        source_path.name,
        len(result.frame),
        output_path,
    )

    return ProcessedDataset(
        frame=result.frame,
        frequency=frequency,
        validation=result.validation,
        operations=result.operations,
        output_path=output_path,
        report_path=report_path,
        report=report,
    )


def print_summary(processed: ProcessedDataset) -> None:
    """Print a concise processing summary."""

    validation = processed.validation
    print(f"===== {processed.frequency.upper()} CLEANING =====")
    print(
        f"Rows: {processed.report['input_row_count']} -> "
        f"{processed.report['output_row_count']}"
    )
    print(
        "Range: "
        f"{validation.first_timestamp} .. {validation.last_timestamp} "
        f"({validation.time_standard}, GHI {validation.ghi_units})"
    )
    print(
        f"Missing intervals: {validation.missing_interval_count}; "
        f"duplicates: {validation.duplicate_timestamp_count}; "
        f"invalid timestamps: {validation.invalid_timestamp_count}"
    )
    print(f"Missing values: {validation.missing_value_counts}")
    print(f"Warnings: {len(validation.warnings)}")
    print(f"Output: {processed.output_path}")
    print(f"Report: {processed.report_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clean NASA POWER data.")
    parser.add_argument(
        "frequency",
        nargs="?",
        choices=(DAILY, HOURLY, "all"),
        default=DAILY,
        help="Dataset to clean (default: daily).",
    )
    parser.add_argument("--site", help="Site id from config/sites.yaml.")
    parser.add_argument(
        "--verbose", action="store_true", help="Enable debug logging."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    site = load_config().get_site(args.site)
    frequencies = FREQUENCIES if args.frequency == "all" else (
        args.frequency,
    )

    exit_code = 0
    for frequency in frequencies:
        try:
            processed = process_dataset(site, frequency)
        except (ValidationError, FileNotFoundError, ValueError) as exc:
            print(f"[error] {frequency}: {exc}")
            exit_code = 1
            continue
        print_summary(processed)
        print()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
