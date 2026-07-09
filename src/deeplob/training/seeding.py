"""Reproducible seeding across every RNG this project's training pipeline touches: Python's
`random`, numpy's *global* random state, and torch (CPU + accelerator).
"""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seeds Python's `random`, numpy's global RNG, and torch (CPU + CUDA if available).

    Deliberately uses `np.random.seed` (the legacy *global* RNG), not the `Generator` API
    `deeplob.data.synthetic` uses elsewhere in this project -- that module wants a local,
    explicitly-threaded generator for a pure data-generation function; this one wants to seed
    *global* state, since other libraries in the training stack (scikit-learn internals,
    anything calling `np.random.rand()` directly) read from that global state, not a
    Generator instance this function has no way to hand them.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 -- global state is the deliberate point here, see above
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
