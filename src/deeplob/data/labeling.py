"""The standard FI-2010 / DeepLOB smoothed mid-price-movement labeling scheme.

For each time t, compares the mean mid-price over the PAST `horizon` snapshots (inclusive of
t) to the mean mid-price over the NEXT `horizon` snapshots (exclusive of t) -- not a naive
"raw price at t+horizon vs t" comparison, which is noisier and non-standard in this
literature. Positions too close to either edge of the series to have a full horizon-length
window on both sides get INVALID_LABEL and must be excluded downstream (windowing.py does
this by construction, not left to the caller to remember).
"""

from enum import IntEnum

import numpy as np

INVALID_LABEL = -1


class Label(IntEnum):
    DOWN = 0
    STATIONARY = 1
    UP = 2


def compute_labels(mid_prices: np.ndarray, horizon: int, alpha: float) -> np.ndarray:
    """`mid_prices` is [N]. Returns an [N] int array of `Label` values, `INVALID_LABEL` at
    positions without a full `horizon`-length window on both sides (including when the
    series is shorter than `2 * horizon`, in which case every position is invalid).
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    n = mid_prices.shape[0]
    labels = np.full(n, INVALID_LABEL, dtype=np.int64)

    first_valid = horizon - 1
    last_valid = n - horizon - 1
    if first_valid > last_valid:
        return labels

    # padded_cumsum[i] = sum(mid_prices[0:i]); padded_cumsum[0] = 0. Lets both the backward
    # and forward windowed means be computed as O(1) slices instead of re-summing per t.
    padded_cumsum = np.concatenate(([0.0], np.cumsum(mid_prices)))
    valid_t = np.arange(first_valid, last_valid + 1)

    backward_sum = padded_cumsum[valid_t + 1] - padded_cumsum[valid_t + 1 - horizon]
    forward_sum = padded_cumsum[valid_t + 1 + horizon] - padded_cumsum[valid_t + 1]
    backward_mean = backward_sum / horizon
    forward_mean = forward_sum / horizon
    pct_change = (forward_mean - backward_mean) / backward_mean

    valid_labels = np.where(
        pct_change > alpha,
        Label.UP,
        np.where(pct_change < -alpha, Label.DOWN, Label.STATIONARY),
    )
    labels[valid_t] = valid_labels

    return labels
