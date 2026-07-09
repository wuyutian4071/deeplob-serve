from pathlib import Path

import onnx
import pytest

from deeplob.data.lob import NUM_FEATURES
from deeplob.export.onnx_export import export_to_onnx, quantize_onnx_model
from deeplob.models.cnn_lstm import DeepLOBCNNLSTM

_WINDOW_SIZE = 30


def test_export_produces_a_valid_onnx_file(tmp_path: Path) -> None:
    model = DeepLOBCNNLSTM(window_size=_WINDOW_SIZE)
    onnx_path = tmp_path / "model.onnx"

    export_to_onnx(model, window_size=_WINDOW_SIZE, onnx_path=onnx_path)

    assert onnx_path.exists()
    loaded = onnx.load(str(onnx_path))
    onnx.checker.check_model(loaded)


def test_export_declares_the_expected_fixed_input_shape(tmp_path: Path) -> None:
    model = DeepLOBCNNLSTM(window_size=_WINDOW_SIZE)
    onnx_path = tmp_path / "model.onnx"

    export_to_onnx(model, window_size=_WINDOW_SIZE, onnx_path=onnx_path, batch_size=1)

    loaded = onnx.load(str(onnx_path))
    (input_tensor,) = loaded.graph.input
    dims = [d.dim_value for d in input_tensor.type.tensor_type.shape.dim]
    assert dims == [1, _WINDOW_SIZE, NUM_FEATURES]


def test_quantize_produces_a_valid_onnx_file(tmp_path: Path) -> None:
    model = DeepLOBCNNLSTM(window_size=_WINDOW_SIZE)
    onnx_path = tmp_path / "model.onnx"
    quantized_path = tmp_path / "model_quantized.onnx"
    export_to_onnx(model, window_size=_WINDOW_SIZE, onnx_path=onnx_path)

    quantize_onnx_model(onnx_path, quantized_path)

    assert quantized_path.exists()
    onnx.checker.check_model(onnx.load(str(quantized_path)))


def test_quantized_model_is_valid_onnx_of_a_different_size_than_the_original(
    tmp_path: Path,
) -> None:
    # Not asserting smaller specifically -- int8 quantization changes weight storage but adds
    # some quantization metadata/nodes too, so "quantized is always smaller" isn't guaranteed
    # for a model this size; asserting it's a genuinely different (successfully transformed)
    # file is the honest, verifiable claim.
    model = DeepLOBCNNLSTM(window_size=_WINDOW_SIZE)
    onnx_path = tmp_path / "model.onnx"
    quantized_path = tmp_path / "model_quantized.onnx"
    export_to_onnx(model, window_size=_WINDOW_SIZE, onnx_path=onnx_path)

    quantize_onnx_model(onnx_path, quantized_path)

    assert quantized_path.stat().st_size != onnx_path.stat().st_size


@pytest.mark.parametrize("batch_size", [1, 4])
def test_export_supports_different_fixed_batch_sizes(tmp_path: Path, batch_size: int) -> None:
    model = DeepLOBCNNLSTM(window_size=_WINDOW_SIZE)
    onnx_path = tmp_path / f"model_b{batch_size}.onnx"

    export_to_onnx(model, window_size=_WINDOW_SIZE, onnx_path=onnx_path, batch_size=batch_size)

    loaded = onnx.load(str(onnx_path))
    (input_tensor,) = loaded.graph.input
    assert input_tensor.type.tensor_type.shape.dim[0].dim_value == batch_size
