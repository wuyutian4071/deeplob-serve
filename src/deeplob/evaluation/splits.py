"""Temporal train/val/test splitting -- the anti-leakage guarantee every model in this
project is built on: LOB windows are never shuffled before splitting. A random split would
let a validation/test window sit chronologically *before* a training window whose own
horizon-labeling already peeked past it, silently leaking future information into training.
This is exactly the leakage risk this module exists to prevent, not an incidental detail.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TemporalSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def temporal_train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> TemporalSplit:
    """Splits `X`/`y` chronologically into train/val/test, in that order along axis 0 --
    never shuffled. `X`/`y` are assumed already in time order, which is what
    `deeplob.data.windowing.make_windows` produces by construction.
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same length")
    if not (0.0 < train_frac < 1.0):
        raise ValueError("train_frac must be in (0, 1)")
    if not (0.0 < val_frac < 1.0):
        raise ValueError("val_frac must be in (0, 1)")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1 (some data must remain for test)")

    n = X.shape[0]
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    if train_end == 0 or val_end in (train_end, n):
        raise ValueError(
            f"n={n} is too small for train_frac={train_frac}, val_frac={val_frac} -- "
            "one or more splits would be empty"
        )

    return TemporalSplit(
        X_train=X[:train_end],
        y_train=y[:train_end],
        X_val=X[train_end:val_end],
        y_val=y[train_end:val_end],
        X_test=X[val_end:],
        y_test=y[val_end:],
    )
