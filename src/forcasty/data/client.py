"""HTTP access to the NASA POWER point APIs.

The client owns everything that can fail over the network: URL selection,
request parameters, timeouts, retries with exponential backoff, and validation
of the response envelope.  It never writes files and never runs at import time,
so it can be exercised offline with a fake ``requests`` session.

Two conventions are enforced rather than assumed:

* ``time-standard`` is always sent explicitly and the value echoed back by the
  API must match it.
* a payload with no parameters, or missing one of the requested parameters, is
  an error.  A failed request must never look like a successful empty dataset.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import requests

logger = logging.getLogger(__name__)

HOURLY_ENDPOINT = "https://power.larc.nasa.gov/api/temporal/hourly/point"
DAILY_ENDPOINT = "https://power.larc.nasa.gov/api/temporal/daily/point"

MAX_PARAMETERS_PER_REQUEST = 15
DEFAULT_CONNECT_TIMEOUT = 10.0
DEFAULT_READ_TIMEOUT = 120.0
DEFAULT_MAX_RETRIES = 4
DEFAULT_BACKOFF_FACTOR = 1.0
DEFAULT_BACKOFF_MAX = 60.0
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class NasaPowerError(RuntimeError):
    """Base class for NASA POWER acquisition failures."""


class NasaPowerPayloadError(NasaPowerError):
    """The response was retrieved but is not a usable NASA POWER payload."""


class NasaPowerRequestError(NasaPowerError):
    """The request failed after exhausting retries or hit a permanent error."""


@dataclass(frozen=True)
class PowerResponse:
    """A validated NASA POWER payload for one request.

    ``values`` maps each parameter to ``{timestamp_key: value}`` exactly as
    returned by the API.  Parsing to frames happens in the acquisition layer so
    that hourly and daily conventions stay separate.
    """

    endpoint: str
    request_params: Mapping[str, Any]
    header: Mapping[str, Any]
    parameters: Mapping[str, Mapping[str, Any]]
    values: Mapping[str, Mapping[str, float]]
    geometry: Mapping[str, Any]

    @property
    def time_standard(self) -> str:
        return str(self.header.get("time_standard", ""))

    @property
    def api_version(self) -> str:
        api = self.header.get("api")
        if isinstance(api, Mapping):
            return str(api.get("version", ""))
        return ""

    @property
    def fill_value(self) -> float:
        return float(self.header.get("fill_value", -999.0))

    @property
    def grid_coordinates(self) -> dict[str, float]:
        """Return returned grid ``latitude``/``longitude``/``elevation``."""

        coordinates = self.geometry.get("coordinates")
        if not isinstance(coordinates, Sequence) or len(coordinates) < 2:
            return {}
        names = ("longitude", "latitude", "elevation")
        return {
            name: float(value)
            for name, value in zip(names, coordinates)
            if isinstance(value, (int, float))
        }


def build_request_params(
    site: Any,
    parameters: Sequence[str],
    start: str,
    end: str,
    *,
    time_standard: str,
    community: str = "RE",
) -> dict[str, str | float]:
    """Build a NASA POWER query for ``site`` and an inclusive date range.

    ``start`` and ``end`` are ``YYYYMMDD`` strings.  Callers must choose the
    time standard explicitly; there is no implicit default here.
    """

    names = tuple(parameters)
    if not names:
        raise ValueError("At least one NASA POWER parameter is required.")
    if len(names) > MAX_PARAMETERS_PER_REQUEST:
        raise ValueError(
            f"NASA POWER accepts at most {MAX_PARAMETERS_PER_REQUEST} "
            f"parameters per request; got {len(names)}."
        )
    standard = time_standard.strip().upper()
    if standard not in {"UTC", "LST"}:
        raise ValueError(
            f"time_standard must be 'UTC' or 'LST', got {time_standard!r}."
        )

    return {
        "parameters": ",".join(names),
        "community": community,
        "longitude": site.longitude,
        "latitude": site.latitude,
        "start": start,
        "end": end,
        "format": "JSON",
        "time-standard": standard,
    }


def _messages_from(payload: Any) -> list[str]:
    if not isinstance(payload, Mapping):
        return []
    messages = payload.get("messages")
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        return []
    return [str(message) for message in messages]


class NasaPowerClient:
    """Minimal, retrying client for the NASA POWER point endpoints."""

    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        read_timeout: float = DEFAULT_READ_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
        backoff_max: float = DEFAULT_BACKOFF_MAX,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._session = session if session is not None else requests.Session()
        self._timeout = (connect_timeout, read_timeout)
        self._max_retries = max_retries
        self._backoff_factor = backoff_factor
        self._backoff_max = backoff_max
        self._sleep = sleep

    def fetch(
        self,
        endpoint: str,
        params: Mapping[str, Any],
        *,
        time_standard: str,
    ) -> PowerResponse:
        """GET ``endpoint`` and return a validated payload.

        Transient failures (connection errors, 429, 5xx) are retried with
        exponential backoff.  Permanent 4xx responses and unusable payloads
        raise immediately.
        """

        expected_standard = str(params.get("time-standard", "")).upper()
        if expected_standard != time_standard.strip().upper():
            raise ValueError(
                "Request parameters and requested time standard disagree: "
                f"{params.get('time-standard')!r} vs {time_standard!r}."
            )

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._session.get(
                    endpoint, params=dict(params), timeout=self._timeout
                )
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= self._max_retries:
                    break
                delay = self._delay(attempt)
                logger.warning(
                    "NASA POWER request error (%s); retry %d/%d in %.1fs",
                    exc,
                    attempt + 1,
                    self._max_retries,
                    delay,
                )
                self._sleep(delay)
                continue

            status = response.status_code
            if status == 200:
                response.raise_for_status()
                return self._parse(response, endpoint, params, expected_standard)

            if status in RETRYABLE_STATUS and attempt < self._max_retries:
                delay = self._delay(attempt, response)
                logger.warning(
                    "NASA POWER returned HTTP %d; retry %d/%d in %.1fs",
                    status,
                    attempt + 1,
                    self._max_retries,
                    delay,
                )
                self._sleep(delay)
                continue

            # Permanent client error, or a retryable status with no attempts
            # left: expose the HTTP error and any API messages.
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                payload = self._maybe_json(response)
                messages = _messages_from(payload)
                detail = f"; messages: {messages}" if messages else ""
                raise NasaPowerRequestError(
                    f"NASA POWER request failed with HTTP {status} "
                    f"for {endpoint}{detail}"
                ) from exc
            raise NasaPowerRequestError(
                f"NASA POWER request failed with HTTP {status} for {endpoint}"
            )

        raise NasaPowerRequestError(
            f"NASA POWER request to {endpoint} failed after "
            f"{self._max_retries + 1} attempt(s): {last_error}"
        ) from last_error

    def _delay(self, attempt: int, response: Any | None = None) -> float:
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    seconds = float(retry_after)
                except (TypeError, ValueError):
                    seconds = None
                if seconds is not None and seconds >= 0:
                    return min(seconds, self._backoff_max)
        return min(
            self._backoff_factor * (2**attempt),
            self._backoff_max,
        )

    @staticmethod
    def _maybe_json(response: Any) -> Any:
        try:
            return response.json()
        except ValueError:
            return None

    def _parse(
        self,
        response: Any,
        endpoint: str,
        params: Mapping[str, Any],
        expected_standard: str,
    ) -> PowerResponse:
        try:
            payload = response.json()
        except ValueError as exc:
            raise NasaPowerPayloadError(
                f"NASA POWER returned invalid JSON for {endpoint}."
            ) from exc

        if not isinstance(payload, Mapping):
            raise NasaPowerPayloadError(
                "NASA POWER payload must be a JSON object; got "
                f"{type(payload).__name__}."
            )

        messages = _messages_from(payload)
        status = payload.get("status")
        if status is not None and str(status).lower() not in {"ok", "success"}:
            raise NasaPowerPayloadError(
                f"NASA POWER reported status {status!r}; messages: {messages}."
            )

        header = payload.get("header")
        if not isinstance(header, Mapping):
            raise NasaPowerPayloadError(
                "NASA POWER payload is missing the 'header' object."
            )

        actual_standard = str(header.get("time_standard", "")).upper()
        if actual_standard != expected_standard:
            raise NasaPowerPayloadError(
                "NASA POWER did not honour the requested time standard: "
                f"asked {expected_standard}, got {actual_standard or '<missing>'}."
            )

        properties = payload.get("properties")
        if not isinstance(properties, Mapping):
            raise NasaPowerPayloadError(
                "NASA POWER payload is missing the 'properties' object."
            )

        parameter_values = properties.get("parameter")
        if not isinstance(parameter_values, Mapping) or not parameter_values:
            raise NasaPowerPayloadError(
                "NASA POWER returned no parameter data"
                + (f"; messages: {messages}" if messages else ".")
            )

        requested = [
            name.strip()
            for name in str(params.get("parameters", "")).split(",")
            if name.strip()
        ]
        missing = [name for name in requested if name not in parameter_values]
        if missing:
            raise NasaPowerPayloadError(
                f"NASA POWER response is missing requested parameter(s): {missing}."
            )

        values: dict[str, Mapping[str, float]] = {}
        for name in requested:
            column = parameter_values[name]
            if not isinstance(column, Mapping) or not column:
                raise NasaPowerPayloadError(
                    f"NASA POWER returned no values for parameter {name!r}."
                )
            values[name] = MappingProxyType(dict(column))

        lengths = {len(column) for column in values.values()}
        if len(lengths) != 1:
            raise NasaPowerPayloadError(
                "NASA POWER parameters have inconsistent lengths: "
                f"{ {name: len(col) for name, col in values.items()} }."
            )
        key_sets = {frozenset(column) for column in values.values()}
        if len(key_sets) != 1:
            raise NasaPowerPayloadError(
                "NASA POWER parameters do not share the same timestamps."
            )

        metadata = payload.get("parameters")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise NasaPowerPayloadError(
                "NASA POWER 'parameters' metadata must be an object."
            )

        geometry = payload.get("geometry")
        if geometry is None:
            geometry = {}
        elif not isinstance(geometry, Mapping):
            raise NasaPowerPayloadError(
                "NASA POWER 'geometry' must be an object."
            )

        return PowerResponse(
            endpoint=endpoint,
            request_params=MappingProxyType(dict(params)),
            header=MappingProxyType(dict(header)),
            parameters=MappingProxyType(
                {
                    str(name): MappingProxyType(dict(info))
                    for name, info in (metadata or {}).items()
                    if isinstance(info, Mapping)
                }
            ),
            values=MappingProxyType(values),
            geometry=MappingProxyType(dict(geometry)),
        )
