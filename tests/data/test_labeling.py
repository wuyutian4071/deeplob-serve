import numpy as np
import pytest

from deeplob.data.labeling import INVALID_LABEL, Label, compute_labels


def test_hand_computed_labels_match_the_smoothed_backward_forward_scheme() -> None:
    # mid_prices: a flat run of 10s, then a flat run of 20s, horizon=2.
    #   t=0: invalid (no full backward window)
    #   t=1: backward mean([10,10])=10, forward mean([10,10])=10   -> pct=0.0    -> STATIONARY
    #   t=2: backward mean([10,10])=10, forward mean([10,20])=15   -> pct=0.5    -> UP
    #   t=3: backward mean([10,10])=10, forward mean([20,20])=20   -> pct=1.0    -> UP
    #   t=4: backward mean([10,20])=15, forward mean([20,20])=20   -> pct=0.333  -> UP
    #   t=5: backward mean([20,20])=20, forward mean([20,20])=20   -> pct=0.0    -> STATIONARY
    #   t=6,7: invalid (no full forward window)
    mid_prices = np.array([10, 10, 10, 10, 20, 20, 20, 20], dtype=np.float64)

    labels = compute_labels(mid_prices, horizon=2, alpha=0.1)

    expected = np.array(
        [
            INVALID_LABEL,
            Label.STATIONARY,
            Label.UP,
            Label.UP,
            Label.UP,
            Label.STATIONARY,
            INVALID_LABEL,
            INVALID_LABEL,
        ]
    )
    np.testing.assert_array_equal(labels, expected)


def test_downward_movement_is_labeled_down() -> None:
    mid_prices = np.array([20, 20, 20, 20, 10, 10, 10, 10], dtype=np.float64)
    labels = compute_labels(mid_prices, horizon=2, alpha=0.1)
    # t=2: backward mean([20,20])=20, forward mean([20,10])=15 -> pct=-0.25 -> DOWN
    assert labels[2] == Label.DOWN


def test_boundary_positions_are_invalid_for_a_short_series() -> None:
    # n=3, horizon=2: first_valid=1, last_valid=3-2-1=0 -- first_valid > last_valid, so
    # every position must be invalid, not an error.
    mid_prices = np.array([10.0, 10.0, 10.0])
    labels = compute_labels(mid_prices, horizon=2, alpha=0.1)
    assert np.all(labels == INVALID_LABEL)


def test_pct_change_exactly_at_the_threshold_is_stationary_not_up() -> None:
    # backward mean=10, forward mean=11 -> pct=0.1, exactly alpha -- the boundary must count
    # as STATIONARY (strict > / <), not UP, since a non-strict >= would make "exactly at the
    # threshold" arbitrarily flip between UP/DOWN and STATIONARY depending on floating-point
    # noise in real (non-hand-picked) data.
    mid_prices = np.array([10, 10, 11, 11], dtype=np.float64)
    labels = compute_labels(mid_prices, horizon=1, alpha=0.1)
    assert labels[1] == Label.STATIONARY


@pytest.mark.parametrize("horizon", [0, -1])
def test_rejects_non_positive_horizon(horizon: int) -> None:
    with pytest.raises(ValueError, match="horizon must be positive"):
        compute_labels(np.array([1.0, 2.0, 3.0]), horizon=horizon, alpha=0.1)


@pytest.mark.parametrize("alpha", [0.0, -0.1])
def test_rejects_non_positive_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha must be positive"):
        compute_labels(np.array([1.0, 2.0, 3.0]), horizon=1, alpha=alpha)
