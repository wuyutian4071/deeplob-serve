import numpy as np
import pytest

from deeplob.data.lob import (
    NUM_FEATURES,
    NUM_LEVELS,
    ask_price,
    ask_volume,
    bid_price,
    bid_volume,
    mid_price,
)


def _make_snapshot() -> np.ndarray:
    # Level i (1-indexed) gets ask_price=100+i, ask_volume=200+i, bid_price=10+i,
    # bid_volume=20+i -- distinct, easy-to-verify-by-hand values per level and per field.
    row = np.empty(NUM_FEATURES, dtype=np.float64)
    for level in range(1, NUM_LEVELS + 1):
        offset = (level - 1) * 4
        row[offset] = 100 + level
        row[offset + 1] = 200 + level
        row[offset + 2] = 10 + level
        row[offset + 3] = 20 + level
    return row


def test_accessors_read_the_correct_columns_for_a_single_snapshot() -> None:
    snapshot = _make_snapshot()
    for level in range(1, NUM_LEVELS + 1):
        assert ask_price(snapshot, level) == 100 + level
        assert ask_volume(snapshot, level) == 200 + level
        assert bid_price(snapshot, level) == 10 + level
        assert bid_volume(snapshot, level) == 20 + level


def test_accessors_work_on_a_batch_via_ellipsis_indexing() -> None:
    batch = np.stack([_make_snapshot(), _make_snapshot() * 2])
    assert ask_price(batch, 1).tolist() == [101, 202]
    assert bid_volume(batch, 10).tolist() == [30, 60]


def test_mid_price_averages_best_ask_and_best_bid() -> None:
    snapshot = _make_snapshot()
    # Level 1: ask_price=101, bid_price=11 -> mid = 56.0
    assert mid_price(snapshot) == pytest.approx(56.0)


@pytest.mark.parametrize("level", [0, -1, 11, 100])
def test_accessors_reject_out_of_range_levels(level: int) -> None:
    snapshot = _make_snapshot()
    with pytest.raises(ValueError, match="level must be in"):
        ask_price(snapshot, level)
