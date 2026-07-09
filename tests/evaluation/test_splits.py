import numpy as np
import pytest

from deeplob.evaluation.splits import temporal_train_val_test_split


def test_hand_computed_split_boundaries() -> None:
    # n=100, train_frac=0.7, val_frac=0.2 -> train_end=70, val_end=70+20=90.
    X = np.arange(100).reshape(100, 1)
    y = np.arange(100)

    split = temporal_train_val_test_split(X, y, train_frac=0.7, val_frac=0.2)

    assert split.X_train.shape[0] == 70
    assert split.X_val.shape[0] == 20
    assert split.X_test.shape[0] == 10
    np.testing.assert_array_equal(split.y_train, np.arange(0, 70))
    np.testing.assert_array_equal(split.y_val, np.arange(70, 90))
    np.testing.assert_array_equal(split.y_test, np.arange(90, 100))


def test_splits_preserve_chronological_order_never_shuffled() -> None:
    X = np.arange(50).reshape(50, 1)
    y = np.arange(50)
    split = temporal_train_val_test_split(X, y, train_frac=0.6, val_frac=0.2)

    # Every train index < every val index < every test index -- the anti-leakage guarantee.
    assert split.y_train.max() < split.y_val.min()
    assert split.y_val.max() < split.y_test.min()


def test_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        temporal_train_val_test_split(np.zeros((10, 1)), np.zeros(9))


@pytest.mark.parametrize("train_frac", [0.0, 1.0, -0.1, 1.5])
def test_rejects_train_frac_outside_open_unit_interval(train_frac: float) -> None:
    with pytest.raises(ValueError, match="train_frac must be in"):
        temporal_train_val_test_split(np.zeros((10, 1)), np.zeros(10), train_frac=train_frac)


def test_rejects_fractions_that_leave_no_room_for_test() -> None:
    with pytest.raises(ValueError, match="must be < 1"):
        temporal_train_val_test_split(np.zeros((10, 1)), np.zeros(10), train_frac=0.6, val_frac=0.4)


def test_rejects_too_small_a_dataset_for_the_requested_fractions() -> None:
    # n=2 with default fractions (0.7/0.15/0.15) rounds train_end=1, val_end=1 -- an empty
    # val split, which must be rejected rather than silently producing an unusable split.
    with pytest.raises(ValueError, match="too small"):
        temporal_train_val_test_split(np.zeros((2, 1)), np.zeros(2))
