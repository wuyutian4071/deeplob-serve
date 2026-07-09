"""Differential numerical parity checking across the PyTorch -> ONNX -> quantized-ONNX
export pipeline -- the same "verify the optimized path against a trusted reference"
discipline already established in the sibling liquibook-x project (hash map vs.
`std::unordered_map`, `OrderBook` vs. `ReferenceOrderBook`), applied here to a model export
rather than a C++ data structure.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from deeplob.data.lob import NUM_FEATURES
from deeplob.models.cnn_lstm import DeepLOBCNNLSTM


@dataclass(frozen=True)
class ParityReport:
    max_abs_diff_onnx: float
    max_abs_diff_quantized: float


def max_abs_diff(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Max absolute per-element difference between `reference` and `candidate` -- the
    simplest, most conservative parity metric: a single large discrepancy anywhere fails it,
    unlike a mean-based metric that a few big errors could hide inside many small ones.
    """
    if reference.shape != candidate.shape:
        raise ValueError(
            f"shape mismatch: reference {reference.shape} vs candidate {candidate.shape}"
        )
    return float(np.abs(reference - candidate).max())


def check_export_parity(
    model: DeepLOBCNNLSTM,
    onnx_path: Path,
    quantized_path: Path,
    window_size: int,
    batch_size: int = 1,
    seed: int = 0,
) -> ParityReport:
    """Runs the same random `[batch_size, window_size, NUM_FEATURES]` input through the
    original PyTorch `model`, the exported ONNX model, and the quantized ONNX model, and
    reports how far each optimized path's logits diverge from the PyTorch reference.

    `batch_size` MUST match the `batch_size` `onnx_path`/`quantized_path` were exported with
    (see `export_to_onnx`'s docstring for why the exported graph's batch axis is fixed, not
    dynamic) -- a mismatch fails with onnxruntime's own shape-mismatch error, not a silent
    wrong answer.
    """
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch_size, window_size, NUM_FEATURES, generator=generator)

    with torch.no_grad():
        torch_logits = model(x).numpy()

    onnx_session = ort.InferenceSession(str(onnx_path))
    onnx_logits = onnx_session.run(None, {"lob_window": x.numpy()})[0]

    quantized_session = ort.InferenceSession(str(quantized_path))
    quantized_logits = quantized_session.run(None, {"lob_window": x.numpy()})[0]

    return ParityReport(
        max_abs_diff_onnx=max_abs_diff(torch_logits, onnx_logits),
        max_abs_diff_quantized=max_abs_diff(torch_logits, quantized_logits),
    )
