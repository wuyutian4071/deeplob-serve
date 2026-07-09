from pathlib import Path

import numpy as np
import pytest

from deeplob.export.onnx_export import export_to_onnx, quantize_onnx_model
from deeplob.export.parity import check_export_parity, max_abs_diff
from deeplob.models.cnn_lstm import DeepLOBCNNLSTM

_WINDOW_SIZE = 30

# Empirically observed on this model architecture (untrained, random init) at two different
# window sizes: ONNX export vs PyTorch differs by ~1e-8-2e-8 (floating-point re-implementation
# noise), quantized ONNX differs by ~2e-4-3e-4 (int8 quantization noise). These thresholds
# give roughly 100-500x margin over what was actually observed, not picked arbitrarily first
# and hoped to hold -- tight enough to catch a genuinely broken export (which would produce
# differences many orders of magnitude larger, not a marginal increase).
_ONNX_TOLERANCE = 1e-4
_QUANTIZED_TOLERANCE = 1e-2


def test_max_abs_diff_hand_computed() -> None:
    a = np.array([[1.0, 2.0, 3.0]])
    b = np.array([[1.0, 2.5, 2.0]])
    # |1-1|=0, |2-2.5|=0.5, |3-2|=1.0 -> max is 1.0.
    assert max_abs_diff(a, b) == pytest.approx(1.0)


def test_max_abs_diff_is_zero_for_identical_arrays() -> None:
    a = np.array([[0.1, 0.2], [0.3, 0.4]])
    assert max_abs_diff(a, a.copy()) == pytest.approx(0.0)


def test_max_abs_diff_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        max_abs_diff(np.zeros((2, 3)), np.zeros((2, 4)))


def test_max_abs_diff_catches_a_deliberately_broken_case() -> None:
    # Verifies the metric has discriminating power -- comparing a real reference against
    # garbage must NOT report a small/passing difference.
    reference = np.array([[1.0, 0.0, 0.0]])
    garbage = np.array([[-5.0, 10.0, -3.0]])
    diff = max_abs_diff(reference, garbage)
    assert diff > 1.0  # nowhere close to _ONNX_TOLERANCE or _QUANTIZED_TOLERANCE


def test_check_export_parity_passes_for_a_real_matched_export(tmp_path: Path) -> None:
    model = DeepLOBCNNLSTM(window_size=_WINDOW_SIZE)
    onnx_path = tmp_path / "model.onnx"
    quantized_path = tmp_path / "model_quantized.onnx"
    export_to_onnx(model, window_size=_WINDOW_SIZE, onnx_path=onnx_path)
    quantize_onnx_model(onnx_path, quantized_path)

    report = check_export_parity(model, onnx_path, quantized_path, window_size=_WINDOW_SIZE)

    assert report.max_abs_diff_onnx < _ONNX_TOLERANCE
    assert report.max_abs_diff_quantized < _QUANTIZED_TOLERANCE


def test_check_export_parity_is_deterministic_across_repeated_calls(tmp_path: Path) -> None:
    # Same seed, same model, same export -- must reproduce identical diffs, not just "small"
    # ones that happen to vary run to run.
    model = DeepLOBCNNLSTM(window_size=_WINDOW_SIZE)
    onnx_path = tmp_path / "model.onnx"
    quantized_path = tmp_path / "model_quantized.onnx"
    export_to_onnx(model, window_size=_WINDOW_SIZE, onnx_path=onnx_path)
    quantize_onnx_model(onnx_path, quantized_path)

    report_a = check_export_parity(
        model, onnx_path, quantized_path, window_size=_WINDOW_SIZE, seed=7
    )
    report_b = check_export_parity(
        model, onnx_path, quantized_path, window_size=_WINDOW_SIZE, seed=7
    )

    assert report_a == report_b
