"""Tests for rejected site configurations."""

from __future__ import annotations

from pathlib import Path

import pytest

from forcasty.config import ConfigError, load_config

VALID_SITE = """
  tunis:
    name: Tunis
    latitude: 36.8065
    longitude: 10.1815
"""


def write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "sites.yaml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("label", "text"),
    [
        ("top-level-list", "- tunis\n- sfax\n"),
        ("no-sites-key", "default_site: tunis\n"),
        ("empty-sites", "sites: {}\n"),
        (
            "missing-latitude",
            "sites:\n  tunis:\n    name: Tunis\n    longitude: 10.1815\n",
        ),
        (
            "missing-name",
            "sites:\n  tunis:\n    latitude: 36.8065\n    longitude: 10.1815\n",
        ),
        (
            "unknown-field",
            "sites:\n  tunis:\n    name: Tunis\n    latitude: 36.8065\n"
            "    longitude: 10.1815\n    elevation: 10\n",
        ),
        (
            "non-numeric-latitude",
            "sites:\n  tunis:\n    name: Tunis\n    latitude: north\n"
            "    longitude: 10.1815\n",
        ),
        (
            "latitude-out-of-range",
            "sites:\n  tunis:\n    name: Tunis\n    latitude: 91\n"
            "    longitude: 10.1815\n",
        ),
        (
            "longitude-out-of-range",
            "sites:\n  tunis:\n    name: Tunis\n    latitude: 36.8065\n"
            "    longitude: -181\n",
        ),
        (
            "latitude-not-finite",
            "sites:\n  tunis:\n    name: Tunis\n    latitude: .nan\n"
            "    longitude: 10.1815\n",
        ),
        (
            "boolean-coordinate",
            "sites:\n  tunis:\n    name: Tunis\n    latitude: true\n"
            "    longitude: 10.1815\n",
        ),
        (
            "unsafe-site-id",
            "sites:\n  \"../evil\":\n    name: Evil\n    latitude: 36.8\n"
            "    longitude: 10.1\n",
        ),
        (
            "uppercase-site-id",
            "sites:\n  Tunis:\n    name: Tunis\n    latitude: 36.8\n"
            "    longitude: 10.1\n",
        ),
        (
            "empty-name",
            "sites:\n  tunis:\n    name: \"  \"\n    latitude: 36.8\n"
            "    longitude: 10.1\n",
        ),
        (
            "unknown-default-site",
            "default_site: sfax\nsites:\n" + VALID_SITE,
        ),
        (
            "several-sites-without-default",
            "sites:\n"
            + VALID_SITE
            + "  sfax:\n    name: Sfax\n    latitude: 34.7\n    longitude: 10.7\n",
        ),
        (
            "duplicate-sites-key",
            "sites:\n" + VALID_SITE + "sites:\n" + VALID_SITE,
        ),
        ("malformed-yaml", "sites: [\n"),
        ("empty-document", "\n"),
    ],
)
def test_invalid_configurations_raise(
    tmp_path: Path, label: str, text: str
) -> None:
    path = write_config(tmp_path, text)

    with pytest.raises(ConfigError):
        load_config(path)


def test_missing_config_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "sites.yaml")


def test_config_must_be_a_mapping(tmp_path: Path) -> None:
    path = write_config(tmp_path, "- just\n- a\n- list\n")

    with pytest.raises(ConfigError, match="mapping"):
        load_config(path)
