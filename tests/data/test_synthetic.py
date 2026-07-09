import numpy as np
import pytest

from deeplob.data.lob import NUM_FEATURES, NUM_LEVELS, ask_price, ask_volume, bid_price, bid_volume
from deeplob.data.synthetic import generate_synthetic_lob


def test_output_shape_matches_requested_snapshot_count() -> None:
    features = generate_synthetic_lob(num_snapshots=50, seed=1)
    assert features.shape == (50, NUM_FEATURES)


def test_deterministic_for_the_same_seed() -> None:
    a = generate_synthetic_lob(num_snapshots=200, seed=42)
    b = generate_synthetic_lob(num_snapshots=200, seed=42)
    np.testing.assert_array_equal(a, b)


def test_different_seeds_produce_different_data() -> None:
    a = generate_synthetic_lob(num_snapshots=200, seed=1)
    b = generate_synthetic_lob(num_snapshots=200, seed=2)
    assert not np.array_equal(a, b)


def test_rejects_non_positive_snapshot_count() -> None:
    with pytest.raises(ValueError, match="num_snapshots must be positive"):
        generate_synthetic_lob(num_snapshots=0, seed=1)


def test_every_snapshot_has_a_positive_spread_at_the_top_of_book() -> None:
    features = generate_synthetic_lob(num_snapshots=2000, seed=7)
    assert np.all(ask_price(features, 1) > bid_price(features, 1))


def test_ask_prices_strictly_increase_with_depth_in_every_snapshot() -> None:
    features = generate_synthetic_lob(num_snapshots=2000, seed=7)
    for level in range(1, NUM_LEVELS):
        assert np.all(ask_price(features, level) < ask_price(features, level + 1))


def test_bid_prices_strictly_decrease_with_depth_in_every_snapshot() -> None:
    features = generate_synthetic_lob(num_snapshots=2000, seed=7)
    for level in range(1, NUM_LEVELS):
        assert np.all(bid_price(features, level) > bid_price(features, level + 1))


def test_all_volumes_are_positive() -> None:
    features = generate_synthetic_lob(num_snapshots=2000, seed=7)
    for level in range(1, NUM_LEVELS + 1):
        assert np.all(ask_volume(features, level) > 0)
        assert np.all(bid_volume(features, level) > 0)


def test_deepest_bid_level_never_goes_non_positive_across_a_long_volatile_walk() -> None:
    # A long run with high volatility is the scenario most likely to expose the price-floor
    # clamp being wrong -- worth testing explicitly, not just trusting the short default runs
    # above to happen to avoid the edge case.
    features = generate_synthetic_lob(
        num_snapshots=50_000, seed=99, mid_price_volatility=5.0, base_price=1.0
    )
    assert np.all(bid_price(features, NUM_LEVELS) > 0)
