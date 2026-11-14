"""Tests that generated paths stay inside the repository."""

from __future__ import annotations

from pathlib import Path

import pytest

from forcasty import paths
from forcasty.config import load_config
from forcasty.paths import (
    PATHS,
    PROJECT_ROOT,
    PathOutsideProjectError,
    ProjectPaths,
    ProjectRootError,
)


def test_raw_dataset_path_is_site_aware_and_inside() -> None:
    site = load_config().get_site()
    path = PATHS.raw_dataset_path(site.site_id)

    assert path == PROJECT_ROOT / "data" / "raw" / "nasa_power_tunis.csv"
    assert PROJECT_ROOT in path.parents


def test_clean_and_feature_paths_are_inside() -> None:
    site = load_config().get_site()

    clean = PATHS.clean_dataset_path(site.site_id)
    features = PATHS.features_dataset_path(site.site_id)

    assert clean == PROJECT_ROOT / "data" / "processed" / "forcasty_tunis_clean.csv"
    assert features == (
        PROJECT_ROOT / "data" / "processed" / "forcasty_tunis_features.csv"
    )
    assert PROJECT_ROOT in clean.parents
    assert PROJECT_ROOT in features.parents


def test_hourly_and_metadata_paths_are_inside() -> None:
    site = load_config().get_site()

    hourly = PATHS.hourly_dataset_path(site.site_id)
    metadata = PATHS.metadata_path("nasa_power_tunis_hourly")

    assert hourly == (
        PROJECT_ROOT / "data" / "raw" / "nasa_power_tunis_hourly.csv"
    )
    assert metadata == (
        PROJECT_ROOT / "data" / "metadata" / "nasa_power_tunis_hourly.json"
    )
    assert PROJECT_ROOT in hourly.parents
    assert PROJECT_ROOT in metadata.parents


def test_relative_path_resolves_against_root() -> None:
    resolved = PATHS.resolve("data/raw/example.csv")

    assert resolved == PROJECT_ROOT / "data" / "raw" / "example.csv"


def test_absolute_path_inside_root_is_allowed() -> None:
    inside = PROJECT_ROOT / "data" / "raw" / "example.csv"

    assert PATHS.resolve(inside) == inside.resolve()


def test_parent_traversal_is_rejected() -> None:
    with pytest.raises(PathOutsideProjectError):
        PATHS.resolve("../outside.csv")


def test_absolute_path_outside_root_is_rejected(tmp_path: Path) -> None:
    outside = tmp_path / "outside.csv"

    with pytest.raises(PathOutsideProjectError):
        PATHS.resolve(outside)


def test_project_paths_rejects_non_directory(tmp_path: Path) -> None:
    missing = tmp_path / "not-a-directory"

    with pytest.raises(ProjectRootError):
        ProjectPaths(missing)


def test_configured_writes_target_the_committed_locations() -> None:
    site = load_config().get_site()

    for path in (
        PATHS.raw_dataset_path(site.site_id),
        PATHS.clean_dataset_path(site.site_id),
        PATHS.features_dataset_path(site.site_id),
    ):
        assert path.parent.is_dir()
        assert paths.PROJECT_ROOT == PROJECT_ROOT
