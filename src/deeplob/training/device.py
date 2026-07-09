"""Accelerator/precision selection -- resolved by actually checking what this machine's
PyTorch backend supports, not assumed. This project's actual development target is Apple
Silicon (MPS), not CUDA; mixed-precision support historically differed between the two, so
this was verified empirically (a real `lightning.Trainer.fit()` run with
`accelerator="mps", precision="16-mixed"`) rather than presumed -- confirmed working cleanly
on this machine's PyTorch 2.13 / Lightning 2.6, which is what `resolve_precision()` below
reflects. If you're running on an older PyTorch/Lightning version where this doesn't hold,
`resolve_precision()`'s fallback to `"32-true"` for CPU is the safe default to reach for.
"""

import torch


def resolve_accelerator() -> str:
    """A Lightning `Trainer(accelerator=...)` value: "mps" on Apple Silicon (checked via
    `torch.backends.mps.is_available()`, not just "this is a Mac"), "gpu" if CUDA is
    available, else "cpu".
    """
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "gpu"
    return "cpu"


def resolve_precision(accelerator: str) -> str:
    """A Lightning `Trainer(precision=...)` value. Mixed precision (`"16-mixed"`) is used on
    both MPS and CUDA -- verified working on MPS specifically (see module docstring), not
    just assumed to work the same as CUDA's much more mature AMP support. Plain `"32-true"`
    on CPU, where mixed precision has no throughput benefit.
    """
    if accelerator in ("mps", "gpu"):
        return "16-mixed"
    return "32-true"
