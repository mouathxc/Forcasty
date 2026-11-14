"""Tests for the M3 exploratory-analysis path helpers.

All filesystem assertions use an injected ``ProjectPaths(tmp_path)`` root, so
the real ``reports/eda`` directory is never touched.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from forcasty.config import load_config
from forcasty.paths import PATHS, PROJECT_ROOT, ProjectPaths


@pytest.fixture
def paths(tmp_path: Path) -> ProjectPaths:
    return ProjectPaths(tmp_path)


def test_eda_paths_use_the_documented_layout(paths: ProjectPaths) -> None:
    root = paths.root / "reports" / "eda" / "tunis"

    assert paths.eda_dir("tunis") == root
    assert paths.eda_summaries_dir("tunis") == root / "summaries"
    assert paths.eda_figures_dir("tunis") == root / "figures"
    assert paths.eda_summary_path("tunis", "daily_profile.json") == (
        root / "summaries" / "daily_profile.json"
    )
    assert paths.eda_figure_path("tunis", "daily_ghi.png") == (
        root / "figures" / "daily_ghi.png"
    )
    assert paths.eda_report_path("tunis") == root / "eda_report.md"
    assert paths.eda_manifest_path("tunis") == root / "manifest.json"


def test_eda_paths_are_inside_the_injected_root(paths: ProjectPaths) -> None:
    for path in (
        paths.eda_dir("tunis"),
        paths.eda_summaries_dir("tunis"),
        paths.eda_figures_dir("tunis"),
        paths.eda_summary_path("tunis", "quality.json"),
        paths.eda_figure_path("tunis", "quality.png"),
        paths.eda_report_path("tunis"),
        paths.eda_manifest_path("tunis"),
    ):
        assert paths.root in path.parents


def test_resolving_eda_paths_does_not_create_directories(
    paths: ProjectPaths,
) -> None:
    paths.eda_dir("tunis")
    paths.eda_summaries_dir("tunis")
    paths.eda_figures_dir("tunis")
    paths.eda_summary_path("tunis", "temporal.json")
    paths.eda_figure_path("tunis", "temporal.png")
    paths.eda_report_path("tunis")
    paths.eda_manifest_path("tunis")

    assert not (paths.root / "reports").exists()


def test_configured_paths_match_the_committed_layout() -> None:
    site = load_config().get_site()
    root = PROJECT_ROOT / "reports" / "eda" / site.site_id

    assert PATHS.eda_dir(site.site_id) == root
    assert PATHS.eda_summaries_dir(site.site_id) == root / "summaries"
    assert PATHS.eda_figures_dir(site.site_id) == root / "figures"
    assert PATHS.eda_summary_path(site.site_id, "manifest.json") == (
        root / "summaries" / "manifest.json"
    )
    assert PATHS.eda_figure_path(site.site_id, "profile.png") == (
        root / "figures" / "profile.png"
    )
    assert PATHS.eda_report_path(site.site_id) == root / "eda_report.md"
    assert PATHS.eda_manifest_path(site.site_id) == root / "manifest.json"


def test_existing_helpers_keep_their_layout(paths: ProjectPaths) -> None:
    assert paths.raw_dataset_path("tunis") == (
        paths.root / "data" / "raw" / "nasa_power_tunis.csv"
    )
    assert paths.hourly_clean_dataset_path("tunis") == (
        paths.root / "data" / "processed" / "forcasty_tunis_hourly_clean.csv"
    )
    assert paths.processing_report_path("tunis", "daily") == (
        paths.root
        / "data"
        / "metadata"
        / "forcasty_tunis_daily_processing.json"
    )


@pytest.mark.parametrize(
    "site_id",
    [
        "",
        " ",
        "..",
        ".",
        ".hidden",
        "../tunis",
        "tunis/../sfax",
        "tunis/../../etc",
        "nested/tunis",
        "nested\\tunis",
        "tunis\\..\\sfax",
        "/absolute/tunis",
        "C:\\absolute\\tunis",
        "Tunis",
        "tunis eve",
        "tunis.",
        "tunis;rm",
    ],
)
def test_invalid_site_ids_are_rejected(
    paths: ProjectPaths, site_id: str
) -> None:
    with pytest.raises(ValueError):
        paths.eda_dir(site_id)


@pytest.mark.parametrize(
    "site_id",
    ["../tunis", "nested/tunis", "Tunis","tunis\n",],
)
@pytest.mark.parametrize(
    "resolve_path",
    [
        lambda p, site: p.eda_summaries_dir(site),
        lambda p, site: p.eda_figures_dir(site),
        lambda p, site: p.eda_summary_path(site, "quality.json"),
        lambda p, site: p.eda_figure_path(site, "quality.png"),
        lambda p, site: p.eda_report_path(site),
        lambda p, site: p.eda_manifest_path(site),
    ],
)
def test_site_validation_applies_to_every_eda_helper(
    paths: ProjectPaths,
    site_id: str,
    resolve_path: Callable[[ProjectPaths, str], Path],
) -> None:
    with pytest.raises(ValueError):
        resolve_path(paths, site_id)


@pytest.mark.parametrize(
    "filename",
    [
        "",
        "   ",
        " report.json",
        "report.json ",
        ".",
        "..",
        "../summary.json",
        "..\\summary.json",
        "nested/summary.json",
        "nested\\summary.json",
        "/absolute/summary.json",
        "C:\\absolute\\summary.json",
        ".hidden.json",
        ".json",
        "summary",
        "summary.txt",
        "summary.json.bak",
        "summary.JSON",
    ],
)
def test_invalid_summary_filenames_are_rejected(
    paths: ProjectPaths, filename: str
) -> None:
    with pytest.raises(ValueError):
        paths.eda_summary_path("tunis", filename)


@pytest.mark.parametrize(
    "filename",
    [
        "",
        "   ",
        "figure.png ",
        "../figure.png",
        "nested/figure.png",
        "nested\\figure.png",
        "/absolute/figure.png",
        "C:\\absolute\\figure.png",
        ".hidden.png",
        ".png",
        "figure",
        "figure.json",
        "figure.png.tmp",
        "figure.PNG",
    ],
)
def test_invalid_figure_filenames_are_rejected(
    paths: ProjectPaths, filename: str
) -> None:
    with pytest.raises(ValueError):
        paths.eda_figure_path("tunis", filename)


def test_absolute_filenames_from_tmp_path_are_rejected(
    paths: ProjectPaths, tmp_path: Path
) -> None:
    with pytest.raises(ValueError):
        paths.eda_summary_path("tunis", str(tmp_path / "summary.json"))
    with pytest.raises(ValueError):
        paths.eda_figure_path("tunis", str(tmp_path / "figure.png"))


def test_rejection_does_not_create_directories(
    paths: ProjectPaths, tmp_path: Path
) -> None:
    with pytest.raises(ValueError):
        paths.eda_summary_path("tunis", "../escape.json")

    assert not (paths.root / "reports").exists()
    assert not (tmp_path / "escape.json").exists()


def test_invalid_filename_does_not_depend_on_site_validity(
    paths: ProjectPaths,
) -> None:
    with pytest.raises(ValueError):
        paths.eda_summary_path("tunis", "not-a-summary.txt")
