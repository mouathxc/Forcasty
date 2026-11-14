"""Filesystem conventions for the Forcasty project.

The project root is never hardcoded.  It is resolved once, in this order:

1. the ``FORCASTY_ROOT`` environment variable, when set;
2. the nearest directory (starting at this file) that contains
   ``pyproject.toml``.

Every path that Forcasty writes is constrained to the repository root, so a
misconfigured run fails loudly instead of writing somewhere else on disk.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_MARKER = "pyproject.toml"
ROOT_ENV_VAR = "FORCASTY_ROOT"


class ProjectRootError(RuntimeError):
    """Raised when the project root cannot be determined."""


class PathOutsideProjectError(ValueError):
    """Raised when a path would resolve outside the project root."""


def find_project_root(start: str | Path | None = None) -> Path:
    """Return the absolute path of the Forcasty repository.

    Parameters
    ----------
    start:
        File or directory to start the search from.  Defaults to this module's
        location, which makes the result independent of the working directory.
    """

    env_root = os.environ.get(ROOT_ENV_VAR)
    if env_root:
        candidate = Path(env_root).expanduser()
        if not candidate.is_dir():
            raise ProjectRootError(
                f"{ROOT_ENV_VAR} points to a directory that does not exist: "
                f"{candidate}"
            )
        return candidate.resolve()

    origin = Path(start) if start is not None else Path(__file__)
    origin = origin.resolve()
    if origin.is_file():
        origin = origin.parent

    for candidate in (origin, *origin.parents):
        if (candidate / PROJECT_MARKER).is_file():
            return candidate

    raise ProjectRootError(
        f"Could not locate {PROJECT_MARKER} above {origin}. "
        f"Set {ROOT_ENV_VAR} to the repository root."
    )


@dataclass(frozen=True)
class ProjectPaths:
    """Directory layout for a Forcasty checkout."""

    root: Path

    def __post_init__(self) -> None:
        root = Path(self.root).expanduser().resolve()
        if not root.is_dir():
            raise ProjectRootError(f"Project root is not a directory: {root}")
        object.__setattr__(self, "root", root)

    @property
    def config_dir(self) -> Path:
        return self.root / "config"

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def metadata_dir(self) -> Path:
        return self.data_dir / "metadata"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def config_file(self) -> Path:
        return self.config_dir / "sites.yaml"

    def resolve(self, path: str | Path) -> Path:
        """Resolve ``path`` against the project root and keep it inside it."""

        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        resolved = candidate.resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise PathOutsideProjectError(
                f"Refusing to use a path outside the project root: {resolved}"
            )
        return resolved

    def raw_dataset_path(self, site_id: str, source: str = "nasa_power") -> Path:
        return self.resolve(self.raw_dir / f"{source}_{site_id}.csv")

    def hourly_dataset_path(
        self, site_id: str, source: str = "nasa_power"
    ) -> Path:
        return self.resolve(self.raw_dir / f"{source}_{site_id}_hourly.csv")

    def metadata_path(self, stem: str) -> Path:
        return self.resolve(self.metadata_dir / f"{stem}.json")

    def clean_dataset_path(self, site_id: str) -> Path:
        return self.resolve(
            self.processed_dir / f"forcasty_{site_id}_clean.csv"
        )

    def features_dataset_path(self, site_id: str) -> Path:
        return self.resolve(
            self.processed_dir / f"forcasty_{site_id}_features.csv"
        )


PROJECT_ROOT = find_project_root()
PATHS = ProjectPaths(PROJECT_ROOT)

CONFIG_DIR = PATHS.config_dir
DATA_DIR = PATHS.data_dir
RAW_DATA_DIR = PATHS.raw_dir
PROCESSED_DATA_DIR = PATHS.processed_dir
METADATA_DIR = PATHS.metadata_dir
REPORTS_DIR = PATHS.reports_dir
