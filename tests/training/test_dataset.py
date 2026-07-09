import numpy as np
import pytest
import torch

from deeplob.training.dataset import LOBWindowDataset


def test_length_and_item_access_match_the_source_arrays() -> None:
    features = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    labels = np.array([1, 2])

    dataset = LOBWindowDataset(features, labels)

    assert len(dataset) == 2
    x0, y0 = dataset[0]
    np.testing.assert_allclose(x0.numpy(), features[0])
    assert y0.item() == 1
    x1, y1 = dataset[1]
    np.testing.assert_allclose(x1.numpy(), features[1])
    assert y1.item() == 2


def test_tensors_have_the_expected_dtypes() -> None:
    features = np.zeros((1, 3, 4), dtype=np.float64)
    labels = np.array([0])
    dataset = LOBWindowDataset(features, labels)

    x, y = dataset[0]
    assert x.dtype == torch.float32
    assert y.dtype == torch.long


def test_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        LOBWindowDataset(np.zeros((3, 2, 4)), np.zeros(2))
