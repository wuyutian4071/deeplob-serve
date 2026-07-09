"""Slides a fixed-length window over LOB feature snapshots, pairing each window with the
label at its LAST (most recent) position -- the standard DeepLOB framing: predict the label
for "now" from the `window_size` most recent snapshots leading up to and including now.
"""

import numpy as np

from deeplob.data.labeling import INVALID_LABEL


def make_windows(
    features: np.ndarray, labels: np.ndarray, window_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """`features` is [N, F], `labels` is [N]. Returns (X, y): X is [M, window_size, F], y is
    [M], where M <= N - window_size + 1 -- windows ending on an INVALID_LABEL position are
    dropped entirely, never included with a placeholder label.

    Builds each window via a direct slice copy (`features[i : i + window_size]`) rather than
    a stride-tricks view -- a straightforward loop that's easy to verify correct by
    inspection, deliberately chosen over a cleverer vectorized approach for something this
    consequential: a silently-transposed axis here would scramble every window feeding every
    later milestone's model training, and be very hard to notice after the fact.
    """
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    n = features.shape[0]
    if labels.shape[0] != n:
        raise ValueError("features and labels must have the same length")

    num_features = features.shape[1]
    if window_size > n:
        return (
            np.empty((0, window_size, num_features), dtype=features.dtype),
            np.empty((0,), dtype=labels.dtype),
        )

    num_candidate_windows = n - window_size + 1
    end_indices = np.arange(window_size - 1, n)  # the label position for each window
    window_labels = labels[end_indices]
    valid_mask = window_labels != INVALID_LABEL

    windows = np.empty((num_candidate_windows, window_size, num_features), dtype=features.dtype)
    for i in range(num_candidate_windows):
        windows[i] = features[i : i + window_size]

    return windows[valid_mask], window_labels[valid_mask]
