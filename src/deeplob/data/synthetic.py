"""Deterministic, seeded synthetic LOB generator for tests and pipeline development without
needing the real FI-2010 dataset (which requires a manual download -- see fi2010.py). Mirrors
liquibook-x's itch/synth.hpp philosophy: no real-market realism claimed, just structurally
valid data (positive spread, monotonically widening levels, positive volumes) varied enough
to exercise the full pipeline (labeling, windowing, later model training) meaningfully.
"""

import numpy as np

from deeplob.data.lob import NUM_FEATURES, NUM_LEVELS


def generate_synthetic_lob(
    num_snapshots: int,
    seed: int = 0,
    base_price: float = 100.0,
    tick_size: float = 0.01,
    mid_price_volatility: float = 0.05,
) -> np.ndarray:
    """Generates `num_snapshots` LOB feature rows, shape [num_snapshots, NUM_FEATURES].

    The mid-price follows a seeded Gaussian random walk. At each snapshot, `NUM_LEVELS` ask
    levels are placed at increasing prices above a randomly-widened best ask, and
    `NUM_LEVELS` bid levels at decreasing prices below a randomly-widened best bid, each with
    a random positive volume. This is NOT a realistic order-flow model -- it exists to
    exercise downstream code with valid, varied inputs, the same role itch/synth.hpp's
    generator plays for ITCH message decoding in the sibling liquibook-x project.
    """
    if num_snapshots <= 0:
        raise ValueError("num_snapshots must be positive")
    if tick_size <= 0:
        raise ValueError("tick_size must be positive")

    rng = np.random.default_rng(seed)
    features = np.empty((num_snapshots, NUM_FEATURES), dtype=np.float64)

    # Keeps the random walk comfortably clear of zero so a full NUM_LEVELS-deep book (bid
    # levels descending from best_bid) never goes non-positive -- a "can't happen by
    # construction" invariant, not something checked defensively at every level.
    price_floor = tick_size * NUM_LEVELS * 2
    mid = max(base_price, price_floor)

    for i in range(num_snapshots):
        mid = max(mid + rng.normal(loc=0.0, scale=mid_price_volatility), price_floor)

        half_spread = tick_size * (1 + rng.integers(0, 3))
        best_ask = _round_to_tick(mid + half_spread, tick_size)
        best_bid = _round_to_tick(mid - half_spread, tick_size)

        for level in range(NUM_LEVELS):
            offset = level * 4
            features[i, offset] = best_ask + level * tick_size
            features[i, offset + 1] = rng.integers(1, 500)
            features[i, offset + 2] = best_bid - level * tick_size
            features[i, offset + 3] = rng.integers(1, 500)

    return features


def _round_to_tick(price: float, tick_size: float) -> float:
    return round(price / tick_size) * tick_size
