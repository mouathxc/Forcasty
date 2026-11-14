"""Exploratory data analysis for Forcasty's cleaned datasets.

This package holds the M3 analysis code.  Phase 0 establishes only the
package namespace and the output-path conventions; profiling, temporal
analysis, missingness diagnostics, correlations, figures and reports are
added in later phases.

Daily and hourly data keep their distinct time conventions and units
throughout the analysis package:

* daily  -- LST, timezone-naive calendar dates, GHI in ``kW-hr/m^2/day``;
* hourly -- UTC, timezone-aware timestamps, GHI in ``Wh/m^2``.
"""

from __future__ import annotations

__all__: list[str] = []
