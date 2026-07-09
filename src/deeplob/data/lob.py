"""Limit order book snapshot representation.

Column layout is the standard FI-2010 / DeepLOB convention: for L price levels (this project
uses L=10, matching FI-2010), a feature row has 4*L columns, ordered level 1..L (level 1 =
best), each level contributing four columns: [ask_price, ask_volume, bid_price, bid_volume].
This is the raw-40-feature representation DeepLOB-style CNN models consume directly -- not
FI-2010's full 149-column format, which also includes handcrafted features this project
doesn't use.
"""

import numpy as np

NUM_LEVELS = 10
NUM_FEATURES = 4 * NUM_LEVELS  # 40


def _check_level(level: int) -> None:
    if not (1 <= level <= NUM_LEVELS):
        raise ValueError(f"level must be in [1, {NUM_LEVELS}], got {level}")


def ask_price(features: np.ndarray, level: int) -> np.ndarray:
    """1-indexed level (1 = best ask). `features` is [..., NUM_FEATURES]."""
    _check_level(level)
    return features[..., 4 * (level - 1)]


def ask_volume(features: np.ndarray, level: int) -> np.ndarray:
    _check_level(level)
    return features[..., 4 * (level - 1) + 1]


def bid_price(features: np.ndarray, level: int) -> np.ndarray:
    _check_level(level)
    return features[..., 4 * (level - 1) + 2]


def bid_volume(features: np.ndarray, level: int) -> np.ndarray:
    _check_level(level)
    return features[..., 4 * (level - 1) + 3]


def mid_price(features: np.ndarray) -> np.ndarray:
    """Best ask + best bid, halved. `features` is [..., NUM_FEATURES]."""
    result: np.ndarray = (ask_price(features, 1) + bid_price(features, 1)) / 2.0
    return result
