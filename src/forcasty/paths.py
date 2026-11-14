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
import re
from dataclasses import dataclass
from pathlib import Path

PROJECT_MARKER = "pyproject.toml"
ROOT_ENV_VAR = "FORCASTY_ROOT"

# Deliberately mirrors ``forcasty.config.SITE_ID_PATTERN``.  ``config`` imports
# this module, so the pattern cannot be imported back without a cycle.
_SITE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

_SUMMARY_SUFFIXES = frozenset({".json"})
_FIGURE_SUFFIXES = frozenset({".png"})


class ProjectRootError(RuntimeError):
    """Raised when the project root cannot be determined."""


class PathOutsideProjectError(ValueError):
    """Raised when a path would resolve outside the project root."""


def _validate_site_id(site_id: str) -> str:
    """Return ``site_id`` unchanged when it is a safe path component."""

    if not isinstance(site_id, str) or not _SITE_ID_PATTERN.fullmatch(site_id):
        raise ValueError(
            "site_id must match "
            f"{_SITE_ID_PATTERN.pattern!r}; got {site_id!r}."
        )
    return site_id

def _validate_filename(
    filename: str, *, suffixes: frozenset[str], kind: str
) -> str:
    """Return ``filename`` unchanged when it is a safe file name.

    A valid name is a single path component: it cannot be absolute, contain a
    directory separator, be ``.``/``..``, be hidden, or use an unexpected
    suffix.  Containment is still re-checked by :meth:`ProjectPaths.resolve`.
    """

    if not isinstance(filename, str) or not filename.strip():
        raise ValueError(f"{kind} filename must be a non-empty string.")
    if filename != filename.strip():
        raise ValueError(
            f"{kind} filename must not have leading or trailing whitespace: "
            f"{filename!r}."
        )
    if filename in {".", ".."} or filename.startswith("."):
        raise ValueError(
            f"{kind} filename must not be hidden or relative: {filename!r}."
        )
    if ":" in filename:
        raise ValueError(
            f"{kind} filename must not contain ':': {filename!r}."
        )

    candidate = Path(filename)
    if (
        candidate.is_absolute()
        or candidate.drive
        or candidate.name != filename
        or "/" in filename
        or "\\" in filename
    ):
        raise ValueError(
            f"{kind} filename must be a single path component: {filename!r}."
        )
    if candidate.suffix not in suffixes:
        allowed = ", ".join(sorted(suffixes))
        raise ValueError(
            f"{kind} filename must end with one of {allowed}: {filename!r}."
        )
    if not candidate.stem:
        raise ValueError(
            f"{kind} filename needs a stem before the suffix: {filename!r}."
        )
    return filename


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

    def hourly_clean_dataset_path(self, site_id: str) -> Path:
        return self.resolve(
            self.processed_dir / f"forcasty_{site_id}_hourly_clean.csv"
        )

    def processing_report_path(self, site_id: str, frequency: str) -> Path:
        if frequency not in {"daily", "hourly"}:
            raise ValueError(
                f"frequency must be 'daily' or 'hourly', got {frequency!r}."
            )
        return self.resolve(
            self.metadata_dir
            / f"forcasty_{site_id}_{frequency}_processing.json"
        )

    def features_dataset_path(self, site_id: str) -> Path:
        return self.resolve(
            self.processed_dir / f"forcasty_{site_id}_features.csv"
        )

    # ------------------------------------------------------------------
    # Exploratory analysis (M3) outputs
    #
    # These helpers only compute paths.  They never create directories, so
    # resolving an EDA artefact path is side-effect free.
    # ------------------------------------------------------------------

    def eda_dir(self, site_id: str) -> Path:
        """Return ``reports/eda/{site_id}``."""

        return self.resolve(
            self.reports_dir / "eda" / _validate_site_id(site_id)
        )

    def eda_summaries_dir(self, site_id: str) -> Path:
        """Return the machine-readable summary directory for ``site_id``."""

        return self.resolve(self.eda_dir(site_id) / "summaries")

    def eda_figures_dir(self, site_id: str) -> Path:
        """Return the figure directory for ``site_id``."""

        return self.resolve(self.eda_dir(site_id) / "figures")

    def eda_summary_path(self, site_id: str, filename: str) -> Path:
        """Return a ``.json`` summary path inside the site's summaries dir."""

        name = _validate_filename(
            filename, suffixes=_SUMMARY_SUFFIXES, kind="summary"
        )
        return self.resolve(self.eda_summaries_dir(site_id) / name)

    def eda_figure_path(self, site_id: str, filename: str) -> Path:
        """Return a ``.png`` figure path inside the site's figures dir."""

        name = _validate_filename(
            filename, suffixes=_FIGURE_SUFFIXES, kind="figure"
        )
        return self.resolve(self.eda_figures_dir(site_id) / name)

    def eda_report_path(self, site_id: str) -> Path:
        """Return ``reports/eda/{site_id}/eda_report.md``."""

        return self.resolve(self.eda_dir(site_id) / "eda_report.md")

    def eda_manifest_path(self, site_id: str) -> Path:
        """Return ``reports/eda/{site_id}/manifest.json``."""

        return self.resolve(self.eda_dir(site_id) / "manifest.json")


PROJECT_ROOT = find_project_root()
PATHS = ProjectPaths(PROJECT_ROOT)

CONFIG_DIR = PATHS.config_dir
DATA_DIR = PATHS.data_dir
RAW_DATA_DIR = PATHS.raw_dir
PROCESSED_DATA_DIR = PATHS.processed_dir
METADATA_DIR = PATHS.metadata_dir
REPORTS_DIR = PATHS.reports_dir
