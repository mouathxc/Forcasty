"""Site registry loading and validation.

Forcasty locations live in ``config/sites.yaml``.  The loader is strict on
purpose: unknown fields, duplicate keys, malformed coordinates and unsafe site
ids all raise :class:`ConfigError` instead of being silently ignored.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import yaml

from forcasty.paths import PATHS

SITE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
SITE_FIELDS = frozenset({"name", "latitude", "longitude"})
TOP_LEVEL_FIELDS = frozenset({"default_site", "sites"})


class ConfigError(ValueError):
    """Raised when the site configuration is missing or invalid."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys instead of overwriting."""

    def construct_mapping(self, node, deep=False):
        mapping: dict[object, object] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
                line = key_node.start_mark.line + 1
                raise ConfigError(
                    f"Duplicate key {key!r} in configuration (line {line})."
                )
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


@dataclass(frozen=True)
class Site:
    """A single forecasting location."""

    site_id: str
    name: str
    latitude: float
    longitude: float

    def to_dict(self) -> dict[str, object]:
        return {
            "site_id": self.site_id,
            "name": self.name,
            "latitude": self.latitude,
            "longitude": self.longitude,
        }


@dataclass(frozen=True)
class ForcastyConfig:
    """Validated application configuration."""

    sites: Mapping[str, Site]
    default_site: str

    def get_site(self, site_id: str | None = None) -> Site:
        """Return ``site_id``, or the configured default when omitted."""

        key = self.default_site if site_id is None else site_id
        try:
            return self.sites[key]
        except KeyError:
            available = ", ".join(sorted(self.sites)) or "<none>"
            raise ConfigError(
                f"Unknown site {key!r}. Configured sites: {available}."
            ) from None

    @property
    def site_ids(self) -> tuple[str, ...]:
        return tuple(self.sites)


def _parse_site(site_id: object, payload: object) -> Site:
    if not isinstance(site_id, str) or not SITE_ID_PATTERN.match(site_id):
        raise ConfigError(
            "Site ids must match "
            f"{SITE_ID_PATTERN.pattern!r}; got {site_id!r}."
        )
    if not isinstance(payload, Mapping):
        raise ConfigError(f"Site {site_id!r} must be a mapping of fields.")

    unknown = set(payload) - SITE_FIELDS
    if unknown:
        raise ConfigError(
            f"Site {site_id!r} has unknown field(s): {sorted(unknown)}."
        )
    missing = SITE_FIELDS - set(payload)
    if missing:
        raise ConfigError(
            f"Site {site_id!r} is missing required field(s): {sorted(missing)}."
        )

    name = payload["name"]
    if not isinstance(name, str) or not name.strip():
        raise ConfigError(f"Site {site_id!r} needs a non-empty 'name'.")

    coordinates: dict[str, float] = {}
    for field, limit in (("latitude", 90.0), ("longitude", 180.0)):
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(
                f"Site {site_id!r} field {field!r} must be a number, "
                f"got {value!r}."
            )
        number = float(value)
        if not math.isfinite(number) or abs(number) > limit:
            raise ConfigError(
                f"Site {site_id!r} field {field!r} must be finite and within "
                f"[-{limit:g}, {limit:g}], got {number!r}."
            )
        coordinates[field] = number

    return Site(
        site_id=site_id,
        name=name.strip(),
        latitude=coordinates["latitude"],
        longitude=coordinates["longitude"],
    )


def load_config(path: str | Path | None = None) -> ForcastyConfig:
    """Load and validate ``config/sites.yaml`` (or ``path`` when given)."""

    config_path = Path(path).expanduser() if path is not None else PATHS.config_file
    if not config_path.is_file():
        raise ConfigError(f"Configuration file not found: {config_path}")

    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Could not read {config_path}: {exc}") from exc

    try:
        raw = yaml.load(text, Loader=_UniqueKeyLoader)
    except ConfigError:
        raise
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc

    if raw is None:
        raise ConfigError(f"Configuration file is empty: {config_path}")
    if not isinstance(raw, Mapping):
        raise ConfigError("The configuration root must be a mapping.")

    unknown = set(raw) - TOP_LEVEL_FIELDS
    if unknown:
        raise ConfigError(f"Unknown top-level field(s): {sorted(unknown)}.")

    sites_raw = raw.get("sites")
    if not isinstance(sites_raw, Mapping) or not sites_raw:
        raise ConfigError("The configuration must define at least one site.")

    sites = {
        site_id: _parse_site(site_id, payload)
        for site_id, payload in sites_raw.items()
    }

    default_site = raw.get("default_site")
    if default_site is None:
        if len(sites) != 1:
            raise ConfigError(
                "'default_site' is required when several sites are defined."
            )
        default_site = next(iter(sites))
    if not isinstance(default_site, str) or default_site not in sites:
        raise ConfigError(
            f"'default_site' must reference a configured site; got "
            f"{default_site!r}."
        )

    return ForcastyConfig(
        sites=MappingProxyType(sites),
        default_site=default_site,
    )
