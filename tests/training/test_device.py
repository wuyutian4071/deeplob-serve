import torch

from deeplob.training.device import resolve_accelerator, resolve_precision


def test_resolve_accelerator_matches_this_machines_actual_backend() -> None:
    # Not hardcoded to "mps" -- this test (and the module it tests) is meant to run
    # correctly on whatever machine actually runs it, CI's Linux runners included, where MPS
    # is never available.
    accelerator = resolve_accelerator()
    if torch.backends.mps.is_available():
        assert accelerator == "mps"
    elif torch.cuda.is_available():
        assert accelerator == "gpu"
    else:
        assert accelerator == "cpu"


def test_resolve_precision_uses_mixed_precision_for_mps_and_gpu() -> None:
    assert resolve_precision("mps") == "16-mixed"
    assert resolve_precision("gpu") == "16-mixed"


def test_resolve_precision_uses_full_precision_for_cpu() -> None:
    assert resolve_precision("cpu") == "32-true"
