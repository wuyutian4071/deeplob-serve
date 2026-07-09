"""Loader for the FI-2010 limit order book benchmark dataset.

FI-2010 (Ntakaris et al., "Benchmark Dataset for Mid-Price Forecasting of Limit Order Book
Data with Machine Learning Methods", 2018) is a public, widely-cited dataset -- but requires
a manual download; there is no reliable, license-compliant way to auto-fetch it in CI, so
this loader is validated in this project's own test suite against a small SYNTHETIC-FORMAT
fixture (tests/data/test_fi2010.py), not the real file. **Anyone running this loader against
the real dataset for the first time should verify the row-count and row-order assumptions
below against their actual downloaded file before trusting the output** -- they reflect this
project's best understanding of the commonly-distributed format, not something independently
re-verified against a live copy during development.

Acquisition (manual, not scripted here): search for "FI-2010 limit order book dataset
Ntakaris" -- the dataset is hosted by the original authors and mirrored in several places;
the "no auction, z-score normalized" variant is the one most commonly used in DeepLOB-style
papers and the one this loader expects.

Expected format: a whitespace/CSV-delimited text file where **rows are features and columns
are time steps** (transposed relative to this project's own [time, features] convention --
this is a well-known characteristic of how FI-2010 is commonly distributed, not a choice
made here). Only the first 40 rows (the raw LOB features: level 1..10, each [ask_price,
ask_volume, bid_price, bid_volume], matching deeplob.data.lob's own column convention) are
read -- FI-2010's own pre-computed label rows (typically the last 5 rows, one per prediction
horizon) are intentionally NOT read here: this project always computes labels itself via
deeplob.data.labeling, the same pathway used for synthetic data, so there is exactly one
labeling implementation to trust rather than two that might disagree on threshold/window
conventions.
"""

from pathlib import Path

import numpy as np

from deeplob.data.lob import NUM_FEATURES


def load_fi2010(path: str | Path) -> np.ndarray:
    """Loads raw LOB features from an FI-2010-format file. Returns [N, NUM_FEATURES].

    Raises FileNotFoundError if `path` doesn't exist, and ValueError if the file has fewer
    than NUM_FEATURES rows (i.e. isn't in the expected transposed format).
    """
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"{resolved} not found. FI-2010 requires a manual download -- see this module's "
            "own docstring for the acquisition process; it is never fetched automatically."
        )

    raw = np.loadtxt(resolved)
    if raw.ndim != 2 or raw.shape[0] < NUM_FEATURES:
        shape = raw.shape if raw.ndim == 2 else (raw.shape, "(not 2D)")
        raise ValueError(
            f"expected a >= {NUM_FEATURES}-row (features x time) matrix, got shape {shape} "
            "-- see this module's docstring: FI-2010 is typically distributed transposed "
            "relative to this project's [time, features] convention."
        )

    feature_rows = raw[:NUM_FEATURES, :]  # [40, N]
    transposed: np.ndarray = feature_rows.T  # [N, 40]
    return transposed
