"""ONNX export + dynamic quantization for the CNN-LSTM -- selected in M6's comparative
evaluation on architecture/robustness grounds, not score (see the README's M6 section).
"""

import tempfile
from pathlib import Path

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process

from deeplob.data.lob import NUM_FEATURES
from deeplob.models.cnn_lstm import DeepLOBCNNLSTM


def export_to_onnx(
    model: DeepLOBCNNLSTM, window_size: int, onnx_path: Path, batch_size: int = 1
) -> None:
    """Exports `model` to ONNX at `onnx_path` with a FIXED `batch_size` (default 1).

    A dynamic batch axis was attempted first and abandoned after actually testing it, not
    assumed to work: `torch.onnx.export`'s `dynamic_shapes` argument left the batch axis
    correctly symbolic for a minimal isolated `nn.LSTM` wrapper, but specialized it back down
    to the concrete traced value for this model regardless -- confirmed by inspecting the
    exported graph's declared input shape directly, not just by a downstream failure. This
    looks like a genuine limitation of the current dynamo-based exporter's handling of
    `nn.LSTM`'s internal batch-dependent state, not a bug in this project's usage of the API.
    A fixed batch size is also the right fit for this project's actual near-term use (M8's
    C++ inference engine, which specifically benchmarks batch-size-1 latency); a dynamic-
    batch or larger-fixed-batch export for throughput-oriented serving is future work, not
    something this milestone claims to already support.

    `window_size` and `NUM_FEATURES` are likewise fixed for this exported graph -- the conv
    stack's kernel sizes (e.g. `conv3`'s `(1, 10)` kernel, reached only by feeding exactly
    `window_size` timesteps through blocks 1-2's fixed `/2/2` feature-axis reduction) are
    baked into `model`'s own construction parameters, not generically dynamic either.
    """
    model.eval()
    dummy_input = torch.randn(batch_size, window_size, NUM_FEATURES)
    torch.onnx.export(
        model,
        (dummy_input,),
        str(onnx_path),
        input_names=["lob_window"],
        output_names=["logits"],
        opset_version=18,
        # The dynamo exporter defaults external_data=True, splitting weights into a
        # companion "<name>.onnx.data" file -- a real, if unsurprising, discovery from
        # actually running the export and checking what files it produced. That's meant for
        # models too large for a single protobuf message; every model this project exports
        # is a few hundred KB, so a single self-contained .onnx file is simpler to move
        # around (e.g. as a C++ test fixture) with nothing to keep in sync.
        external_data=False,
    )


def quantize_onnx_model(onnx_path: Path, quantized_path: Path) -> None:
    """Dynamic quantization (weights to int8, activations quantized at inference time) --
    the "no calibration dataset needed" quantization mode, appropriate here since this
    project has no representative real dataset to calibrate against yet (FI-2010 is a
    manual, undone step; static/QAT quantization would need real data to calibrate on).

    Runs onnxruntime's own `quant_pre_process` first -- confirmed necessary by actually
    running `quantize_dynamic` directly on this dynamo-exported LSTM graph and hitting a
    `ShapeInferenceError` ("Inferred shape and existing shape differ in dimension 0: (64) vs
    (3)"): the dynamo exporter's dynamic-batch LSTM graph isn't directly consumable by
    `quantize_dynamic`'s own shape-inference pass without this pre-processing step first --
    exactly what onnxruntime's own warning message on the unprocessed path points at.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        preprocessed_path = Path(tmpdir) / "preprocessed.onnx"
        quant_pre_process(str(onnx_path), str(preprocessed_path))
        quantize_dynamic(str(preprocessed_path), str(quantized_path), weight_type=QuantType.QInt8)
