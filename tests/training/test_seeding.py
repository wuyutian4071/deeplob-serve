import random

import numpy as np
import torch

from deeplob.training.seeding import set_seed


def test_python_random_is_reproducible_after_seeding() -> None:
    set_seed(42)
    a = [random.random() for _ in range(5)]
    set_seed(42)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_numpy_global_random_is_reproducible_after_seeding() -> None:
    set_seed(7)
    a = np.random.rand(5)  # noqa: NPY002 -- this test specifically verifies *global* state
    set_seed(7)
    b = np.random.rand(5)  # noqa: NPY002
    np.testing.assert_array_equal(a, b)


def test_torch_is_reproducible_after_seeding() -> None:
    set_seed(123)
    a = torch.rand(5)
    set_seed(123)
    b = torch.rand(5)
    torch.testing.assert_close(a, b)


def test_different_seeds_produce_different_torch_output() -> None:
    set_seed(1)
    a = torch.rand(5)
    set_seed(2)
    b = torch.rand(5)
    assert not torch.equal(a, b)
