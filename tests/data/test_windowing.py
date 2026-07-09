import numpy as np
import pytest

from deeplob.data.labeling import INVALID_LABEL
from deeplob.data.windowing import make_windows


def test_hand_computed_windows_and_labels_match_exactly() -> None:
    # 5 snapshots of 4 features each, distinct values so any transposition/misalignment bug
    # would be immediately visible.
    features = np.arange(20, dtype=np.float64).reshape(5, 4)
    labels = np.array([INVALID_LABEL, 0, 1, 2, INVALID_LABEL])

    X, y = make_windows(features, labels, window_size=2)

    # Candidate windows (end index -> label): (1 -> 0), (2 -> 1), (3 -> 2), (4 -> invalid,
    # dropped). 3 windows survive.
    assert X.shape == (3, 2, 4)
    np.testing.assert_array_equal(X[0], features[0:2])
    np.testing.assert_array_equal(X[1], features[1:3])
    np.testing.assert_array_equal(X[2], features[2:4])
    np.testing.assert_array_equal(y, [0, 1, 2])


def test_window_size_larger_than_the_series_returns_empty_arrays() -> None:
    features = np.zeros((3, 4))
    labels = np.array([0, 1, 2])
    X, y = make_windows(features, labels, window_size=10)
    assert X.shape == (0, 10, 4)
    assert y.shape == (0,)


def test_all_windows_dropped_when_every_label_is_invalid() -> None:
    features = np.zeros((5, 4))
    labels = np.full(5, INVALID_LABEL)
    X, y = make_windows(features, labels, window_size=2)
    assert X.shape == (0, 2, 4)
    assert y.shape == (0,)


def test_rejects_non_positive_window_size() -> None:
    with pytest.raises(ValueError, match="window_size must be positive"):
        make_windows(np.zeros((5, 4)), np.zeros(5), window_size=0)


def test_rejects_mismatched_features_and_labels_length() -> None:
    with pytest.raises(ValueError, match="same length"):
        make_windows(np.zeros((5, 4)), np.zeros(4), window_size=2)
