"""Tests for project-root discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from forcasty import paths
from forcasty.paths import (
    PROJECT_MARKER,
    PROJECT_ROOT,
    ProjectRootError,
    find_project_root,
)


def test_default_root_is_the_repository() -> None:
    assert (PROJECT_ROOT / PROJECT_MARKER).is_file()
    assert PROJECT_ROOT == Path(__file__).resolve().parents[1]


def test_root_is_found_from_a_file_and_a_directory() -> None:
    assert find_project_root(Path(__file__)) == PROJECT_ROOT
    assert find_project_root(PROJECT_ROOT) == PROJECT_ROOT


def test_environment_override_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORCASTY_ROOT", str(tmp_path))
    assert find_project_root() == tmp_path.resolve()


def test_missing_environment_directory_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "does-not-exist"
    monkeypatch.setenv("FORCASTY_ROOT", str(missing))
    with pytest.raises(ProjectRootError, match="does not exist"):
        find_project_root()


def test_missing_marker_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FORCASTY_ROOT", raising=False)
    monkeypatch.setattr(paths, "PROJECT_MARKER", "forcasty-test-missing-marker.toml")
    with pytest.raises(ProjectRootError, match="Could not locate"):
        find_project_root(tmp_path)
