"""NASA POWER acquisition for Forcasty.

Two datasets are produced:

``hourly``
    The primary dataset, one row per hour in UTC (``time-standard=UTC``) from
    2020-01-01 through the latest complete UTC day.  Requested in yearly
    chunks so no single call is excessive; the chunks are inclusive and
    contiguous by construction.

``daily``
    The existing daily dataset is extended, never rewritten.  Its history was
    acquired in Local Solar Time (LST), so the tail is requested in LST as
    well; appending UTC days to an LST history would shift day boundaries.
    Only the missing tail after the current last date is fetched, and the
    historical bytes are preserved verbatim.

Nothing is fabricated: ``-999`` becomes missing, values are not filled,
interpolated or replaced with zero, and a request that returns no data raises
instead of producing an empty dataset.
"""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from forcasty.config import Site, load_config
from forcasty.data.client import (
    DAILY_ENDPOINT,
    HOURLY_ENDPOINT,
    MAX_PARAMETERS_PER_REQUEST,
    NasaPowerClient,
    NasaPowerError,
    NasaPowerPayloadError,
    PowerResponse,
    build_request_params,
)
from forcasty.data.metadata import iso_z, sha256_file, utc_now, write_json_atomic
from forcasty.paths import PATHS, ProjectPaths

logger = logging.getLogger(__name__)

COMMUNITY = "RE"
DEFAULT_START_DATE = date(2020, 1, 1)
DEFAULT_HOURLY_PARAMETERS: tuple[str, ...] = (
    "ALLSKY_SFC_SW_DWN",
    "T2M",
    "RH2M",
    "WS2M",
)
HOURLY_TIME_STANDARD = "UTC"
DAILY_TIME_STANDARD = "LST"
HOURLY_TIMESTAMP_FORMAT = "%Y%m%d%H"
DAILY_TIMESTAMP_FORMAT = "%Y%m%d"


class AcquisitionError(RuntimeError):
    """Raised when a dataset cannot be acquired or written safely."""


@dataclass(frozen=True)
class AcquisitionResult:
    """Summary of one completed acquisition."""

    dataset: str
    path: Path
    metadata_path: Path
    records: int
    first_timestamp: str
    last_timestamp: str
    missing_records: int
    duplicate_records: int


def latest_complete_date(now: datetime | None = None) -> date:
    """Return the last complete UTC day (yesterday in UTC).

    NASA POWER can expose a partially populated current day; requesting only
    complete days keeps the dataset honest and reproducible.
    """

    moment = now if now is not None else datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).date() - timedelta(days=1)


def parse_date(value: str | date | datetime) -> date:
    """Parse ``YYYYMMDD`` (or ``YYYY-MM-DD``) into a :class:`date`."""

    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Expected a YYYYMMDD or YYYY-MM-DD date, got {value!r}.")


def nasa_date(value: date) -> str:
    """Format a date the way NASA POWER expects it."""

    return value.strftime("%Y%m%d")


def yearly_chunks(start: date, end: date) -> list[tuple[date, date]]:
    """Split ``start``..``end`` into inclusive, contiguous yearly chunks."""

    if start > end:
        raise ValueError(f"start {start} is after end {end}.")
    chunks: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(date(cursor.year, 12, 31), end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def _hourly_frame(response: PowerResponse) -> pd.DataFrame:
    keys = list(next(iter(response.values.values())).keys())
    try:
        stamps = pd.to_datetime(
            pd.Series(keys), format=HOURLY_TIMESTAMP_FORMAT, utc=True
        )
    except (TypeError, ValueError) as exc:
        raise NasaPowerPayloadError(
            "Hourly timestamps are not YYYYMMDDHH values."
        ) from exc
    frame = pd.DataFrame({"timestamp": stamps})
    _attach_parameters(frame, response, keys)
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    frame["timestamp"] = frame["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return frame


def _daily_frame(response: PowerResponse) -> pd.DataFrame:
    keys = list(next(iter(response.values.values())).keys())
    try:
        stamps = pd.to_datetime(pd.Series(keys), format=DAILY_TIMESTAMP_FORMAT)
    except (TypeError, ValueError) as exc:
        raise NasaPowerPayloadError(
            "Daily timestamps are not YYYYMMDD values."
        ) from exc
    frame = pd.DataFrame({"date": stamps})
    _attach_parameters(frame, response, keys)
    frame = frame.sort_values("date").reset_index(drop=True)
    return frame


def _attach_parameters(
    frame: pd.DataFrame, response: PowerResponse, keys: Sequence[str]
) -> None:
    fill = response.fill_value
    for name, column in response.values.items():
        raw = pd.Series([column[key] for key in keys])
        try:
            numeric = pd.to_numeric(raw, errors="raise")
        except (TypeError, ValueError) as exc:
            raise NasaPowerPayloadError(
                f"Parameter {name!r} contains non-numeric values."
            ) from exc
        # NASA POWER's fill value means "missing"; it is never a measurement.
        frame[name] = numeric.mask(numeric == fill).astype("float64")


def _frame_from_response(response: PowerResponse, *, daily: bool) -> pd.DataFrame:
    return _daily_frame(response) if daily else _hourly_frame(response)


def _request_units(response: PowerResponse, parameters: Sequence[str]) -> dict[str, str]:
    units: dict[str, str] = {}
    for name in parameters:
        info = response.parameters.get(name, {})
        units[name] = str(info.get("units", "unknown"))
    return units


def _request_longnames(
    response: PowerResponse, parameters: Sequence[str]
) -> dict[str, str]:
    names: dict[str, str] = {}
    for name in parameters:
        info = response.parameters.get(name, {})
        names[name] = str(info.get("longname", "unknown"))
    return names


def _fetch_range(
    client: NasaPowerClient,
    *,
    endpoint: str,
    site: Site,
    parameters: Sequence[str],
    chunks: Iterable[tuple[date, date]],
    time_standard: str,
    community: str,
) -> list[PowerResponse]:
    responses: list[PowerResponse] = []
    for first, last in chunks:
        params = build_request_params(
            site,
            parameters,
            nasa_date(first),
            nasa_date(last),
            time_standard=time_standard,
            community=community,
        )
        logger.info(
            "Requesting %s %s..%s (time-standard=%s, %d parameters)",
            endpoint.rsplit("/", 2)[-2],
            first.isoformat(),
            last.isoformat(),
            time_standard,
            len(parameters),
        )
        response = client.fetch(
            endpoint, params, time_standard=time_standard
        )
        rows = next(iter(response.values.values()))
        logger.info("  received %d rows", len(rows))
        responses.append(response)
    return responses


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        frame.to_csv(temporary, index=False, lineterminator="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _append_text_atomic(text: str, existing: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        prefix = existing
        if prefix and not prefix.endswith("\n"):
            prefix += "\n"
        temporary.write_text(prefix + text, encoding="utf-8", newline="")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _format_stamp(frame: pd.DataFrame, column: str, index: int) -> str:
    value = frame[column].iloc[index]
    if column == "date":
        return pd.Timestamp(value).date().isoformat()
    return str(value)


def _metadata_common(
    *,
    dataset: str,
    site: Site,
    parameters: Sequence[str],
    responses: Sequence[PowerResponse],
    start: date,
    end: date,
    time_standard: str,
    output_path: Path,
    project_root: Path,
    frame: pd.DataFrame,
    timestamp_column: str,
    missing_records: int,
    duplicate_records: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    first_response = responses[0]
    api_info = first_response.header.get("api")
    missing_counts = {
        name: int(frame[name].isna().sum()) for name in parameters
    }
    payload: dict[str, Any] = {
        "source": "NASA POWER",
        "product_description": (
            "NASA POWER gridded data derived from reanalysis (MERRA-2) and "
            "satellite products (CERES SYN1deg). These are not ground-station "
            "measurements."
        ),
        "dataset": dataset,
        "endpoint": first_response.endpoint,
        "api_version": first_response.api_version,
        "api_name": str(
            api_info.get("name", "") if isinstance(api_info, dict) else ""
        ),
        "sources": list(first_response.header.get("sources", [])),
        "community": COMMUNITY,
        "time_standard": time_standard,
        "timestamp_convention": (
            "start of hour"
            if timestamp_column == "timestamp"
            else "calendar day"
        ),
        "site": site.to_dict(),
        "requested_coordinates": {
            "latitude": site.latitude,
            "longitude": site.longitude,
        },
        "returned_grid_coordinates": first_response.grid_coordinates,
        "parameters": list(parameters),
        "parameter_units": _request_units(first_response, parameters),
        "parameter_longnames": _request_longnames(first_response, parameters),
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "actual_first_timestamp": _format_stamp(
            frame, timestamp_column, 0
        ),
        "actual_last_timestamp": _format_stamp(
            frame, timestamp_column, -1
        ),
        "acquisition_utc": iso_z(utc_now()),
        "record_count": int(len(frame)),
        "missing_value_counts": missing_counts,
        "missing_value_total": int(sum(missing_counts.values())),
        "missing_record_count": int(missing_records),
        "duplicate_timestamp_count": int(duplicate_records),
        "fill_value": first_response.fill_value,
        "chunks": [
            {
                "requested_start": response.request_params.get("start"),
                "requested_end": response.request_params.get("end"),
                "returned_rows": len(next(iter(response.values.values()))),
            }
            for response in responses
        ],
        "file": output_path.relative_to(project_root).as_posix(),
        "file_sha256": sha256_file(output_path),
        "status": "success",
    }
    if extra:
        payload.update(extra)
    return payload


def acquire_hourly(
    site: Site,
    *,
    start: date | str = DEFAULT_START_DATE,
    end: date | str | None = None,
    parameters: Sequence[str] = DEFAULT_HOURLY_PARAMETERS,
    client: NasaPowerClient | None = None,
    paths: ProjectPaths = PATHS,
    force: bool = False,
    now: datetime | None = None,
) -> AcquisitionResult:
    """Acquire (or explicitly rebuild) the hourly UTC dataset."""

    start_date = parse_date(start)
    end_date = (
        parse_date(end) if end is not None else latest_complete_date(now)
    )
    if start_date > end_date:
        raise ValueError(f"start {start_date} is after end {end_date}.")
    if len(parameters) > MAX_PARAMETERS_PER_REQUEST:
        raise ValueError(
            f"At most {MAX_PARAMETERS_PER_REQUEST} parameters per request."
        )

    output_path = paths.hourly_dataset_path(site.site_id)
    if output_path.exists() and not force:
        raise FileExistsError(
            f"{output_path} already exists; pass force=True (--force) to "
            "rebuild the hourly dataset."
        )

    http = client if client is not None else NasaPowerClient()
    chunks = yearly_chunks(start_date, end_date)
    responses = _fetch_range(
        http,
        endpoint=HOURLY_ENDPOINT,
        site=site,
        parameters=parameters,
        chunks=chunks,
        time_standard=HOURLY_TIME_STANDARD,
        community=COMMUNITY,
    )

    frames = [_hourly_frame(response) for response in responses]
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.sort_values("timestamp").reset_index(drop=True)

    duplicates = int(frame["timestamp"].duplicated().sum())
    if duplicates:
        raise AcquisitionError(
            f"NASA POWER returned {duplicates} duplicate hourly timestamps."
        )

    expected = pd.date_range(
        pd.Timestamp(start_date, tz="UTC"),
        pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(hours=23),
        freq="h",
    )
    actual = pd.to_datetime(frame["timestamp"], utc=True)
    missing = int(len(expected.difference(actual)))
    if missing:
        logger.warning(
            "Hourly coverage is incomplete: %d of %d expected hours missing.",
            missing,
            len(expected),
        )
    logger.info("Hourly dataset contains %d rows.", len(frame))

    _write_csv_atomic(frame, output_path)

    metadata_path = paths.metadata_path(
        f"{output_path.stem.replace('_hourly', '')}_hourly"
    )
    metadata = _metadata_common(
        dataset="hourly",
        site=site,
        parameters=parameters,
        responses=responses,
        start=start_date,
        end=end_date,
        time_standard=HOURLY_TIME_STANDARD,
        output_path=output_path,
        project_root=paths.root,
        frame=frame,
        timestamp_column="timestamp",
        missing_records=missing,
        duplicate_records=duplicates,
        extra={
            "timestamp_format": "YYYY-MM-DDTHH:MM:SSZ",
            "expected_record_count": int(len(expected)),
            "rebuild": force,
        },
    )
    write_json_atomic(metadata, metadata_path)

    return AcquisitionResult(
        dataset="hourly",
        path=output_path,
        metadata_path=metadata_path,
        records=len(frame),
        first_timestamp=str(frame["timestamp"].iloc[0]),
        last_timestamp=str(frame["timestamp"].iloc[-1]),
        missing_records=missing,
        duplicate_records=duplicates,
    )


def _read_daily_history(path: Path) -> tuple[pd.DataFrame, str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Daily history not found at {path}; refusing to create a "
            "replacement history from scratch."
        )
    text = path.read_text(encoding="utf-8")
    history = pd.read_csv(path)
    if "date" not in history.columns:
        raise AcquisitionError(f"{path} has no 'date' column.")
    history["date"] = pd.to_datetime(history["date"], errors="raise")
    if history["date"].duplicated().any():
        raise AcquisitionError(f"{path} contains duplicate dates.")
    if not history["date"].is_monotonic_increasing:
        raise AcquisitionError(f"{path} is not sorted chronologically.")
    return history, text


def acquire_daily_tail(
    site: Site,
    *,
    end: date | str | None = None,
    parameters: Sequence[str] = DEFAULT_HOURLY_PARAMETERS,
    client: NasaPowerClient | None = None,
    paths: ProjectPaths = PATHS,
    now: datetime | None = None,
) -> AcquisitionResult:
    """Extend the daily dataset with only the missing tail.

    The historical rows are preserved byte-for-byte; only new calendar days
    are appended.  The tail is requested in LST to match the existing
    history's time standard.
    """

    output_path = paths.raw_dataset_path(site.site_id)
    history, original_text = _read_daily_history(output_path)
    history_columns = list(history.columns)
    if list(parameters) != [name for name in history_columns if name != "date"]:
        raise AcquisitionError(
            "Requested parameters do not match the existing daily schema: "
            f"history has {history_columns}, request has {list(parameters)}."
        )

    end_date = (
        parse_date(end) if end is not None else latest_complete_date(now)
    )
    last_date = history["date"].max().date()
    tail_start = last_date + timedelta(days=1)

    if tail_start > end_date:
        logger.info(
            "Daily history already reaches %s; nothing to acquire.",
            last_date.isoformat(),
        )
        # Still emit metadata so the no-op run is traceable, but never
        # overwrite provenance from a run that actually downloaded data.
        frame = history.copy()
        frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
        metadata_path = paths.metadata_path(f"{output_path.stem}_daily")
        if not metadata_path.exists():
            metadata = {
                "source": "NASA POWER",
                "dataset": "daily",
                "endpoint": DAILY_ENDPOINT,
                "community": COMMUNITY,
                "status": "up-to-date",
                "site": site.to_dict(),
                "parameters": list(parameters),
                "time_standard": DAILY_TIME_STANDARD,
                "timestamp_convention": "calendar day",
                "requested_start": tail_start.isoformat(),
                "requested_end": end_date.isoformat(),
                "actual_first_timestamp": str(frame["date"].iloc[0]),
                "actual_last_timestamp": str(frame["date"].iloc[-1]),
                "acquisition_utc": iso_z(utc_now()),
                "record_count": int(len(frame)),
                "missing_value_counts": {
                    name: int(frame[name].isna().sum()) for name in parameters
                },
                "duplicate_timestamp_count": int(
                    frame["date"].duplicated().sum()
                ),
                "file": output_path.relative_to(paths.root).as_posix(),
                "file_sha256": sha256_file(output_path),
            }
            write_json_atomic(metadata, metadata_path)
        else:
            logger.info(
                "No new daily days; existing metadata kept at %s.",
                metadata_path,
            )
        return AcquisitionResult(
            dataset="daily",
            path=output_path,
            metadata_path=metadata_path,
            records=len(frame),
            first_timestamp=str(frame["date"].iloc[0]),
            last_timestamp=str(frame["date"].iloc[-1]),
            missing_records=0,
            duplicate_records=0,
        )

    http = client if client is not None else NasaPowerClient()
    chunks = yearly_chunks(tail_start, end_date)
    logger.info(
        "Extending daily history %s -> %s (LST).",
        tail_start.isoformat(),
        end_date.isoformat(),
    )
    responses = _fetch_range(
        http,
        endpoint=DAILY_ENDPOINT,
        site=site,
        parameters=parameters,
        chunks=chunks,
        time_standard=DAILY_TIME_STANDARD,
        community=COMMUNITY,
    )

    tail = pd.concat(
        [_daily_frame(response) for response in responses], ignore_index=True
    )
    tail = tail.sort_values("date").reset_index(drop=True)
    tail["date"] = pd.to_datetime(tail["date"])

    # Defensive: never re-emit a historical day, even if NASA POWER returns it.
    overlap = tail["date"] <= pd.Timestamp(last_date)
    if overlap.any():
        logger.warning(
            "Discarding %d returned row(s) that already exist in history.",
            int(overlap.sum()),
        )
        tail = tail.loc[~overlap].reset_index(drop=True)

    if tail.empty:
        raise AcquisitionError(
            "NASA POWER returned no new daily rows for "
            f"{tail_start}..{end_date}."
        )

    duplicates = int(tail["date"].duplicated().sum())
    if duplicates:
        raise AcquisitionError(
            f"NASA POWER returned {duplicates} duplicate daily timestamps."
        )

    expected = pd.date_range(
        pd.Timestamp(tail_start), pd.Timestamp(end_date), freq="D"
    )
    missing = int(len(expected.difference(pd.DatetimeIndex(tail["date"]))))
    if missing:
        logger.warning(
            "Daily tail is incomplete: %d of %d expected days missing.",
            missing,
            len(expected),
        )

    tail_text = tail.to_csv(index=False, header=False, lineterminator="\n")
    _append_text_atomic(tail_text, original_text, output_path)

    combined, _ = _read_daily_history(output_path)
    combined_gaps = int(
        len(
            pd.date_range(
                combined["date"].min(), combined["date"].max(), freq="D"
            ).difference(pd.DatetimeIndex(combined["date"]))
        )
    )
    if combined_gaps:
        logger.warning(
            "Combined daily dataset has %d gap day(s); they are not filled.",
            combined_gaps,
        )

    metadata_path = paths.metadata_path(f"{output_path.stem}_daily")
    metadata = _metadata_common(
        dataset="daily",
        site=site,
        parameters=parameters,
        responses=responses,
        start=tail_start,
        end=end_date,
        time_standard=DAILY_TIME_STANDARD,
        output_path=output_path,
        project_root=paths.root,
        frame=tail,
        timestamp_column="date",
        missing_records=missing,
        duplicate_records=duplicates,
        extra={
            "timestamp_format": "YYYY-MM-DD",
            "time_standard_note": (
                "The existing daily history was acquired in NASA POWER Local "
                "Solar Time (LST). The tail uses LST as well to keep day "
                "boundaries consistent; hourly data use UTC."
            ),
            "append_only": True,
            "historical_last_date": last_date.isoformat(),
            "historical_rows": int(len(history)),
            "appended_rows": int(len(tail)),
            "combined_rows": int(len(combined)),
            "combined_missing_day_count": combined_gaps,
        },
    )
    write_json_atomic(metadata, metadata_path)

    return AcquisitionResult(
        dataset="daily",
        path=output_path,
        metadata_path=metadata_path,
        records=len(combined),
        first_timestamp=str(combined["date"].min().date()),
        last_timestamp=str(combined["date"].max().date()),
        missing_records=missing,
        duplicate_records=duplicates,
    )


def acquire_all(
    site: Site,
    *,
    start: date | str = DEFAULT_START_DATE,
    end: date | str | None = None,
    parameters: Sequence[str] = DEFAULT_HOURLY_PARAMETERS,
    client: NasaPowerClient | None = None,
    paths: ProjectPaths = PATHS,
    force: bool = False,
    now: datetime | None = None,
) -> tuple[AcquisitionResult, AcquisitionResult]:
    """Acquire the hourly dataset, then extend the daily tail."""

    http = client if client is not None else NasaPowerClient()
    hourly = acquire_hourly(
        site,
        start=start,
        end=end,
        parameters=parameters,
        client=http,
        paths=paths,
        force=force,
        now=now,
    )
    daily = acquire_daily_tail(
        site,
        end=end,
        parameters=parameters,
        client=http,
        paths=paths,
        now=now,
    )
    return hourly, daily


def _parse_parameters(value: str | None) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_HOURLY_PARAMETERS
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    if not names:
        raise argparse.ArgumentTypeError("At least one parameter is required.")
    return names


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Acquire NASA POWER data for a configured site."
    )
    parser.add_argument(
        "dataset",
        choices=("hourly", "daily-tail", "all"),
        help="Which dataset to acquire.",
    )
    parser.add_argument("--site", help="Site id from config/sites.yaml.")
    parser.add_argument(
        "--start",
        default=DEFAULT_START_DATE.strftime("%Y%m%d"),
        help="Hourly start date (YYYYMMDD); daily always resumes after history.",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End date (YYYYMMDD); defaults to the last complete UTC day.",
    )
    parser.add_argument(
        "--parameters",
        default=None,
        help="Comma-separated NASA POWER parameters (max 15).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the hourly dataset if it already exists.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Retries for transient failures (default: 4).",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable debug logging."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    site = load_config().get_site(args.site)
    parameters = _parse_parameters(args.parameters)
    client = NasaPowerClient(max_retries=args.max_retries)
    end = args.end if args.end is not None else latest_complete_date()

    try:
        if args.dataset == "hourly":
            results = [
                acquire_hourly(
                    site,
                    start=args.start,
                    end=end,
                    parameters=parameters,
                    client=client,
                    force=args.force,
                )
            ]
        elif args.dataset == "daily-tail":
            results = [
                acquire_daily_tail(
                    site,
                    end=end,
                    parameters=parameters,
                    client=client,
                )
            ]
        else:
            results = list(
                acquire_all(
                    site,
                    start=args.start,
                    end=end,
                    parameters=parameters,
                    client=client,
                    force=args.force,
                )
            )
    except (NasaPowerError, AcquisitionError, FileExistsError) as exc:
        logger.error("Acquisition failed: %s", exc)
        return 1

    for result in results:
        logger.info(
            "%s dataset: %s (%d rows, %s .. %s); metadata: %s",
            result.dataset,
            result.path,
            result.records,
            result.first_timestamp,
            result.last_timestamp,
            result.metadata_path,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
