"""Tests for configuration loading and valid site definitions."""

from __future__ import annotations

from pathlib import Path

import pytest

from forcasty.config import ConfigError, Site, load_config


def test_default_config_loads_tunis() -> None:
    config = load_config()

    assert config.default_site == "tunis"
    assert config.site_ids == ("tunis",)

    site = config.get_site()
    assert isinstance(site, Site)
    assert site.site_id == "tunis"
    assert site.name == "Tunis"
    assert site.latitude == pytest.approx(36.8065)
    assert site.longitude == pytest.approx(10.1815)


def test_get_site_by_id_and_unknown_id() -> None:
    config = load_config()

    assert config.get_site("tunis").name == "Tunis"
    with pytest.raises(ConfigError, match="Unknown site"):
        config.get_site("sfax")


def test_sites_mapping_is_read_only() -> None:
    config = load_config()

    with pytest.raises(TypeError):
        config.sites["sfax"] = config.sites["tunis"]  # type: ignore[index]


def test_valid_multi_site_config(tmp_path: Path) -> None:
    path = tmp_path / "sites.yaml"
    path.write_text(
        """
default_site: sfax
sites:
  tunis:
    name: Tunis
    latitude: 36.8065
    longitude: 10.1815
  sfax:
    name: Sfax
    latitude: 34.7406
    longitude: 10.7603
""".strip(),
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.default_site == "sfax"
    assert config.site_ids == ("tunis", "sfax")
    assert config.get_site().name == "Sfax"


def test_single_site_defaults_when_default_site_omitted(tmp_path: Path) -> None:
    path = tmp_path / "sites.yaml"
    path.write_text(
        """
sites:
  tunis:
    name: Tunis
    latitude: 36.8065
    longitude: 10.1815
""".strip(),
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.default_site == "tunis"


def test_site_name_is_trimmed(tmp_path: Path) -> None:
    path = tmp_path / "sites.yaml"
    path.write_text(
        """
sites:
  tunis:
    name: "  Tunis  "
    latitude: 36.8065
    longitude: 10.1815
""".strip(),
        encoding="utf-8",
    )

    assert load_config(path).get_site().name == "Tunis"


def test_site_to_dict_contains_coordinates() -> None:
    site = Site(
        site_id="tunis",
        name="Tunis",
        latitude=36.8065,
        longitude=10.1815,
    )

    assert site.to_dict() == {
        "site_id": "tunis",
        "name": "Tunis",
        "latitude": 36.8065,
        "longitude": 10.1815,
    }
