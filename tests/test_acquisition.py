"""Offline tests for NASA POWER acquisition.

Every HTTP interaction is faked.  No test contacts NASA POWER, and acquisition
outputs are written to a temporary project root that is still enforced by
:class:`ProjectPaths`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import requests

from forcasty.config import Site
from forcasty.data import acquisition
from forcasty.data.client import (
    DAILY_ENDPOINT,
    HOURLY_ENDPOINT,
    NasaPowerClient,
    NasaPowerPayloadError,
    NasaPowerRequestError,
    PowerResponse,
    build_request_params,
)
from forcasty.data.metadata import sha256_file
from forcasty.paths import PATHS, ProjectPaths

PARAMS = ("ALLSKY_SFC_SW_DWN", "T2M", "RH2M", "WS2M")
UNITS = {
    "ALLSKY_SFC_SW_DWN": "Wh/m^2",
    "T2M": "C",
    "RH2M": "%",
    "WS2M": "m/s",
}
SITE = Site(
    site_id="tunis",
    name="Tunis",
    latitude=36.8065,
    longitude=10.1815,
)


# ---------------------------------------------------------------------------
# Fakes and fixtures
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(
        self,
        status: int = 200,
        payload: Any = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}

    def json(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


class FakeSession:
    def __init__(self, *responses: Any) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, params: Any = None, timeout: Any = None) -> Any:
        self.calls.append({"url": url, "params": dict(params), "timeout": timeout})
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def power_payload(
    start: date,
    end: date,
    *,
    hourly: bool,
    time_standard: str,
    parameters: tuple[str, ...] = PARAMS,
    values: dict[str, list[float]] | None = None,
    fill_value: float = -999.0,
    include_header: bool = True,
) -> dict[str, Any]:
    """Build a realistic NASA POWER response for a date range."""

    if hourly:
        stamps = pd.date_range(
            pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(hours=23),
            freq="h",
        )
        keys = [stamp.strftime("%Y%m%d%H") for stamp in stamps]
    else:
        stamps = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D")
        keys = [stamp.strftime("%Y%m%d") for stamp in stamps]

    series: dict[str, Any] = {}
    for index, name in enumerate(parameters):
        if values is not None and name in values:
            data = values[name]
        elif hourly:
            data = [float((hour % 24) * 10 + index) for hour in range(len(keys))]
        else:
            data = [float((hour % 24) * 10 + index) for hour in range(len(keys))]
        series[name] = dict(zip(keys, data))

    payload: dict[str, Any] = {
        "properties": {"parameter": series},
        "parameters": {
            name: {"units": UNITS.get(name, "unknown"), "longname": name}
            for name in parameters
        },
        "geometry": {
            "coordinates": [10.18, 36.81, 4.0],
            "type": "Point",
        },
        "messages": [],
    }
    if include_header:
        payload["header"] = {
            "title": "NASA/POWER Source Native Resolution Data",
            "api": {"version": "v2.10.4", "name": "POWER API"},
            "sources": ["MERRA2", "SYN1DEG", "POWER"],
            "fill_value": fill_value,
            "time_standard": time_standard,
            "start": start.strftime("%Y%m%d"),
            "end": end.strftime("%Y%m%d"),
        }
    return payload


def power_response(
    start: date,
    end: date,
    *,
    hourly: bool,
    time_standard: str,
    endpoint: str = HOURLY_ENDPOINT,
    **kwargs: Any,
) -> PowerResponse:
    payload = power_payload(
        start, end, hourly=hourly, time_standard=time_standard, **kwargs
    )
    params = {
        "parameters": ",".join(PARAMS),
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "time-standard": time_standard,
    }
    return PowerResponse(
        endpoint=endpoint,
        request_params=params,
        header=payload["header"],
        parameters=payload["parameters"],
        values=payload["properties"]["parameter"],
        geometry=payload["geometry"],
    )


class FakeClient:
    """Client double that records requests and synthesises payloads."""

    def __init__(self, *, hourly: bool, time_standard: str) -> None:
        self.hourly = hourly
        self.time_standard = time_standard
        self.calls: list[tuple[str, dict[str, Any], str]] = []

    def fetch(
        self, endpoint: str, params: dict[str, Any], *, time_standard: str
    ) -> PowerResponse:
        self.calls.append((endpoint, dict(params), time_standard))
        start = datetime.strptime(params["start"], "%Y%m%d").date()
        end = datetime.strptime(params["end"], "%Y%m%d").date()
        return power_response(
            start,
            end,
            hourly=self.hourly,
            time_standard=self.time_standard,
            endpoint=endpoint,
        )


def project_paths(root: Path) -> ProjectPaths:
    root.mkdir(parents=True, exist_ok=True)
    return ProjectPaths(root)


def read_hourly(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


# ---------------------------------------------------------------------------
# Client tests: endpoint, UTC, retries, payload validation
# ---------------------------------------------------------------------------


def test_request_params_target_hourly_endpoint_and_explicit_utc() -> None:
    params = build_request_params(
        SITE, PARAMS, "20200101", "20201231", time_standard="UTC"
    )

    assert HOURLY_ENDPOINT.endswith("/temporal/hourly/point")
    assert params["time-standard"] == "UTC"
    assert params["community"] == "RE"
    assert params["format"] == "JSON"
    assert params["parameters"].split(",") == list(PARAMS)
    assert params["latitude"] == SITE.latitude
    assert params["longitude"] == SITE.longitude
    assert params["start"] == "20200101"
    assert params["end"] == "20201231"


def test_client_rejects_more_than_fifteen_parameters() -> None:
    too_many = tuple(f"PARAM_{index}" for index in range(16))

    with pytest.raises(ValueError, match="at most 15"):
        build_request_params(
            SITE, too_many, "20200101", "20200102", time_standard="UTC"
        )


def test_client_parses_successful_payload_and_records_timeout() -> None:
    payload = power_payload(
        date(2020, 1, 1),
        date(2020, 1, 2),
        hourly=True,
        time_standard="UTC",
    )
    session = FakeSession(FakeResponse(payload=payload))
    client = NasaPowerClient(session=session)
    params = build_request_params(
        SITE, PARAMS, "20200101", "20200102", time_standard="UTC"
    )

    response = client.fetch(HOURLY_ENDPOINT, params, time_standard="UTC")

    assert response.time_standard == "UTC"
    assert response.api_version == "v2.10.4"
    assert set(response.values) == set(PARAMS)
    assert len(next(iter(response.values.values()))) == 48
    assert response.grid_coordinates["latitude"] == pytest.approx(36.81)
    assert session.calls[0]["timeout"] == (10.0, 120.0)


def test_client_retries_transient_status_with_backoff() -> None:
    payload = power_payload(
        date(2020, 1, 1),
        date(2020, 1, 1),
        hourly=True,
        time_standard="UTC",
    )
    session = FakeSession(
        FakeResponse(status=503),
        requests.ConnectionError("temporary network failure"),
        FakeResponse(payload=payload),
    )
    sleeps: list[float] = []
    client = NasaPowerClient(
        session=session, sleep=sleeps.append, backoff_factor=1.0
    )
    params = build_request_params(
        SITE, PARAMS, "20200101", "20200101", time_standard="UTC"
    )

    response = client.fetch(HOURLY_ENDPOINT, params, time_standard="UTC")

    assert response.time_standard == "UTC"
    assert len(session.calls) == 3
    assert sleeps == [1.0, 2.0]


def test_client_honours_retry_after_and_gives_up_after_max_retries() -> None:
    session = FakeSession(
        FakeResponse(status=429, headers={"Retry-After": "7"}),
        FakeResponse(status=429, headers={"Retry-After": "7"}),
    )
    sleeps: list[float] = []
    client = NasaPowerClient(
        session=session, max_retries=1, sleep=sleeps.append
    )
    params = build_request_params(
        SITE, PARAMS, "20200101", "20200101", time_standard="UTC"
    )

    with pytest.raises(NasaPowerRequestError, match="HTTP 429"):
        client.fetch(HOURLY_ENDPOINT, params, time_standard="UTC")

    assert len(session.calls) == 2
    assert sleeps == [7.0]


def test_client_does_not_retry_permanent_client_error() -> None:
    session = FakeSession(
        FakeResponse(
            status=422,
            payload={"messages": ["Invalid parameter: NOT_A_PARAMETER"]},
        )
    )
    client = NasaPowerClient(session=session, sleep=lambda _: None)
    params = build_request_params(
        SITE, PARAMS, "20200101", "20200101", time_standard="UTC"
    )

    with pytest.raises(NasaPowerRequestError, match="NOT_A_PARAMETER"):
        client.fetch(HOURLY_ENDPOINT, params, time_standard="UTC")

    assert len(session.calls) == 1


def test_client_rejects_invalid_json_without_retrying() -> None:
    session = FakeSession(FakeResponse(payload=ValueError("not json")))
    client = NasaPowerClient(session=session, sleep=lambda _: None)
    params = build_request_params(
        SITE, PARAMS, "20200101", "20200101", time_standard="UTC"
    )

    with pytest.raises(NasaPowerPayloadError, match="invalid JSON"):
        client.fetch(HOURLY_ENDPOINT, params, time_standard="UTC")

    assert len(session.calls) == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"header": {"time_standard": "UTC"}, "properties": {"parameter": {}}},
        {
            "header": {"time_standard": "UTC"},
            "properties": {"parameter": {"T2M": {"2020010100": 1.0}}},
        },
        {
            "header": {"time_standard": "LST"},
            "properties": {
                "parameter": {
                    "ALLSKY_SFC_SW_DWN": {"2020010100": 1.0},
                }
            },
        },
        {
            "properties": {
                "parameter": {"ALLSKY_SFC_SW_DWN": {"2020010100": 1.0}}
            }
        },
        {"header": {"time_standard": "UTC"}},
        {"header": {"time_standard": "UTC"}, "properties": {}},
    ],
)
def test_client_rejects_empty_or_malformed_payloads(payload: dict[str, Any]) -> None:
    session = FakeSession(FakeResponse(payload=payload))
    client = NasaPowerClient(session=session, sleep=lambda _: None)
    params = build_request_params(
        SITE, ("ALLSKY_SFC_SW_DWN",), "20200101", "20200101", time_standard="UTC"
    )

    with pytest.raises(NasaPowerPayloadError):
        client.fetch(HOURLY_ENDPOINT, params, time_standard="UTC")


# ---------------------------------------------------------------------------
# Hourly acquisition
# ---------------------------------------------------------------------------


def test_hourly_acquisition_writes_utc_sorted_schema(tmp_path: Path) -> None:
    paths = project_paths(tmp_path)
    client = FakeClient(hourly=True, time_standard="UTC")

    result = acquisition.acquire_hourly(
        SITE,
        start=date(2020, 1, 1),
        end=date(2020, 1, 1),
        client=client,
        paths=paths,
    )

    assert result.records == 24
    assert result.first_timestamp == "2020-01-01T00:00:00Z"
    assert result.last_timestamp == "2020-01-01T23:00:00Z"
    assert result.missing_records == 0
    assert result.duplicate_records == 0

    frame = read_hourly(result.path)
    assert list(frame.columns) == ["timestamp", *PARAMS]
    assert frame["timestamp"].is_monotonic_increasing
    assert not frame["timestamp"].duplicated().any()
    assert frame["timestamp"].dt.tz is not None
    assert (frame["timestamp"].dt.strftime("%Y%m%d%H") == "2020010100").sum() == 1
    for name in PARAMS:
        assert pd.api.types.is_float_dtype(frame[name])

    endpoint, params, standard = client.calls[0]
    assert endpoint == HOURLY_ENDPOINT
    assert standard == "UTC"
    assert params["time-standard"] == "UTC"
    assert params["start"] == "20200101"
    assert params["end"] == "20200101"


def test_multiple_chunks_join_without_duplicate_or_missing_boundary(
    tmp_path: Path,
) -> None:
    paths = project_paths(tmp_path)
    client = FakeClient(hourly=True, time_standard="UTC")

    result = acquisition.acquire_hourly(
        SITE,
        start=date(2020, 12, 31),
        end=date(2021, 1, 1),
        client=client,
        paths=paths,
    )

    # Yearly chunking produces one request per calendar year.
    assert [call[1]["start"] for call in client.calls] == ["20201231", "20210101"]
    assert result.records == 48
    assert result.missing_records == 0

    stamps = read_hourly(result.path)["timestamp"]
    expected = pd.date_range(
        "2020-12-31T00:00Z", "2021-01-01T23:00Z", freq="h"
    )
    assert stamps.tolist() == expected.tolist()
    assert not stamps.duplicated().any()


def test_hourly_missing_sentinel_becomes_nan_without_filling(tmp_path: Path) -> None:
    paths = project_paths(tmp_path)
    start = date(2020, 1, 1)
    payload = power_payload(
        start,
        start,
        hourly=True,
        time_standard="UTC",
        values={
            "ALLSKY_SFC_SW_DWN": [-999.0] * 24,
            "T2M": [22.0] * 24,
            "RH2M": [55.0] * 24,
            "WS2M": [3.0] * 24,
        },
    )
    response = PowerResponse(
        endpoint=HOURLY_ENDPOINT,
        request_params={"parameters": ",".join(PARAMS)},
        header=payload["header"],
        parameters=payload["parameters"],
        values=payload["properties"]["parameter"],
        geometry={"coordinates": [10.18, 36.81, 4.0]},
    )

    class OneResponseClient:
        def fetch(self, endpoint: str, params: Any, *, time_standard: str) -> PowerResponse:
            return response

    result = acquisition.acquire_hourly(
        SITE,
        start=start,
        end=start,
        client=OneResponseClient(),  # type: ignore[arg-type]
        paths=paths,
    )

    frame = read_hourly(result.path)
    assert frame["ALLSKY_SFC_SW_DWN"].isna().all()
    assert (frame["ALLSKY_SFC_SW_DWN"] == 0).sum() == 0
    assert (frame["T2M"] == 22.0).all()

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["missing_value_counts"]["ALLSKY_SFC_SW_DWN"] == 24
    assert metadata["missing_value_total"] == 24


def test_hourly_refuses_to_overwrite_without_force(tmp_path: Path) -> None:
    paths = project_paths(tmp_path)
    client = FakeClient(hourly=True, time_standard="UTC")
    acquisition.acquire_hourly(
        SITE, start=date(2020, 1, 1), end=date(2020, 1, 1),
        client=client, paths=paths,
    )

    with pytest.raises(FileExistsError, match="force"):
        acquisition.acquire_hourly(
            SITE, start=date(2020, 1, 1), end=date(2020, 1, 1),
            client=client, paths=paths,
        )


def test_hourly_metadata_documents_provenance_and_hash(tmp_path: Path) -> None:
    paths = project_paths(tmp_path)
    client = FakeClient(hourly=True, time_standard="UTC")

    result = acquisition.acquire_hourly(
        SITE,
        start=date(2020, 1, 1),
        end=date(2020, 1, 2),
        client=client,
        paths=paths,
    )

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["source"] == "NASA POWER"
    assert metadata["endpoint"] == HOURLY_ENDPOINT
    assert metadata["api_version"] == "v2.10.4"
    assert metadata["time_standard"] == "UTC"
    assert metadata["timestamp_convention"] == "start of hour"
    assert metadata["community"] == "RE"
    assert metadata["parameters"] == list(PARAMS)
    assert metadata["parameter_units"]["ALLSKY_SFC_SW_DWN"] == "Wh/m^2"
    assert metadata["site"]["site_id"] == "tunis"
    assert metadata["requested_start"] == "2020-01-01"
    assert metadata["requested_end"] == "2020-01-02"
    assert metadata["actual_first_timestamp"] == "2020-01-01T00:00:00Z"
    assert metadata["actual_last_timestamp"] == "2020-01-02T23:00:00Z"
    assert metadata["record_count"] == 48
    assert metadata["duplicate_timestamp_count"] == 0
    assert metadata["missing_record_count"] == 0
    assert metadata["status"] == "success"
    assert metadata["file"] == "data/raw/nasa_power_tunis_hourly.csv"
    assert metadata["file_sha256"] == sha256_file(result.path)
    assert metadata["acquisition_utc"].endswith("Z")
    assert metadata["returned_grid_coordinates"]["latitude"] == pytest.approx(36.81)
    assert len(metadata["chunks"]) == 1
    assert "not ground-station" in metadata["product_description"]


def test_hourly_acquisition_leaves_existing_daily_dataset_untouched(
    tmp_path: Path,
) -> None:
    paths = project_paths(tmp_path)
    daily_path = paths.raw_dataset_path(SITE.site_id)
    daily_path.parent.mkdir(parents=True, exist_ok=True)
    daily_path.write_bytes(
        (PATHS.root / "data" / "raw" / "nasa_power_tunis.csv").read_bytes()
    )
    before = hashlib.sha256(daily_path.read_bytes()).hexdigest()

    client = FakeClient(hourly=True, time_standard="UTC")
    acquisition.acquire_hourly(
        SITE,
        start=date(2020, 1, 1),
        end=date(2020, 1, 1),
        client=client,
        paths=paths,
    )

    assert hashlib.sha256(daily_path.read_bytes()).hexdigest() == before


# ---------------------------------------------------------------------------
# Daily tail
# ---------------------------------------------------------------------------


def _write_daily_history(path: Path, dates: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["date,ALLSKY_SFC_SW_DWN,T2M,RH2M,WS2M"]
    for index, day in enumerate(dates):
        lines.append(f"{day},{1.5 + index},9.96,76.92,2.9")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_daily_tail_appends_only_missing_days_and_preserves_history(
    tmp_path: Path,
) -> None:
    paths = project_paths(tmp_path)
    daily_path = paths.raw_dataset_path(SITE.site_id)
    _write_daily_history(
        daily_path, ["2020-01-01", "2020-01-02", "2020-01-03"]
    )
    original = daily_path.read_text(encoding="utf-8")
    client = FakeClient(hourly=False, time_standard="LST")

    result = acquisition.acquire_daily_tail(
        SITE,
        end=date(2020, 1, 5),
        client=client,
        paths=paths,
    )

    updated = daily_path.read_text(encoding="utf-8")
    assert updated.startswith(original)
    frame = pd.read_csv(daily_path, parse_dates=["date"])
    assert frame["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2020-01-01",
        "2020-01-02",
        "2020-01-03",
        "2020-01-04",
        "2020-01-05",
    ]
    assert not frame["date"].duplicated().any()
    assert result.records == 5
    assert result.missing_records == 0
    assert result.duplicate_records == 0

    endpoint, params, standard = client.calls[0]
    assert endpoint == DAILY_ENDPOINT
    assert standard == "LST"
    assert params["time-standard"] == "LST"
    assert params["start"] == "20200104"
    assert params["end"] == "20200105"

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["time_standard"] == "LST"
    assert metadata["append_only"] is True
    assert metadata["historical_rows"] == 3
    assert metadata["appended_rows"] == 2
    assert metadata["combined_rows"] == 5
    assert metadata["timestamp_convention"] == "calendar day"


def test_daily_tail_is_a_no_op_when_history_is_current(tmp_path: Path) -> None:
    paths = project_paths(tmp_path)
    daily_path = paths.raw_dataset_path(SITE.site_id)
    _write_daily_history(daily_path, ["2020-01-01", "2020-01-02"])
    before = daily_path.read_bytes()

    class ExplodingClient:
        def fetch(self, *args: Any, **kwargs: Any) -> Any:
            raise AssertionError("no request should be made")

    result = acquisition.acquire_daily_tail(
        SITE,
        end=date(2020, 1, 2),
        client=ExplodingClient(),  # type: ignore[arg-type]
        paths=paths,
    )

    assert daily_path.read_bytes() == before
    assert result.records == 2
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["status"] == "up-to-date"


# ---------------------------------------------------------------------------
# Paths and helpers
# ---------------------------------------------------------------------------


def test_outputs_stay_inside_the_project_root(tmp_path: Path) -> None:
    paths = project_paths(tmp_path)
    client = FakeClient(hourly=True, time_standard="UTC")

    result = acquisition.acquire_hourly(
        SITE,
        start=date(2020, 1, 1),
        end=date(2020, 1, 1),
        client=client,
        paths=paths,
    )

    root = tmp_path.resolve()
    assert root in result.path.parents
    assert root in result.metadata_path.parents
    assert result.path == paths.hourly_dataset_path(SITE.site_id)
    assert result.metadata_path == paths.metadata_path(
        "nasa_power_tunis_hourly"
    )


def test_latest_complete_date_is_yesterday_utc() -> None:
    now = datetime(2026, 11, 14, 3, 30, tzinfo=acquisition.timezone.utc)

    assert acquisition.latest_complete_date(now) == date(2026, 11, 13)


def test_yearly_chunks_are_contiguous() -> None:
    chunks = acquisition.yearly_chunks(date(2020, 1, 1), date(2022, 6, 15))

    assert chunks == [
        (date(2020, 1, 1), date(2020, 12, 31)),
        (date(2021, 1, 1), date(2021, 12, 31)),
        (date(2022, 1, 1), date(2022, 6, 15)),
    ]
    for (_, first_end), (next_start, _) in zip(chunks, chunks[1:]):
        assert next_start == first_end + timedelta(days=1)
