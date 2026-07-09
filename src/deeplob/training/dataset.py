"""Wraps windowed LOB features/labels (as produced by `deeplob.data.windowing.make_windows`)
in a `torch.utils.data.Dataset` for use with PyTorch/Lightning DataLoaders.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class LOBWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """`features` is `[N, window_size, NUM_FEATURES]` (float), `labels` is `[N]` (int class
    labels, no `INVALID_LABEL` entries -- `make_windows` already drops those). Converts to
    tensors once at construction, not per-`__getitem__` call, since the whole windowed
    dataset comfortably fits in memory for this project's scale (unlike, say, streaming a
    dataset too large to hold as tensors up front).
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        if features.shape[0] != labels.shape[0]:
            raise ValueError("features and labels must have the same length")
        self._features = torch.as_tensor(features, dtype=torch.float32)
        self._labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return self._features.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._features[index], self._labels[index]
