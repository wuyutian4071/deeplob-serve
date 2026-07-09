import numpy as np

from deeplob.data.labeling import compute_labels
from deeplob.data.lob import mid_price
from deeplob.data.synthetic import generate_synthetic_lob
from deeplob.data.windowing import make_windows
from deeplob.evaluation.compare import (
    ModelComparison,
    build_split,
    evaluate_baseline,
    evaluate_neural,
    format_comparison_table,
)
from deeplob.evaluation.metrics import calibrate, evaluate
from deeplob.evaluation.splits import TemporalSplit, temporal_train_val_test_split
from deeplob.models.baselines import LogisticRegressionBaseline
from deeplob.models.cnn_lstm import DeepLOBCNNLSTM


def _build_tiny_split() -> TemporalSplit:
    features = generate_synthetic_lob(num_snapshots=1200, seed=3)
    labels = compute_labels(mid_price(features), horizon=5, alpha=0.0005)
    X, y = make_windows(features, labels, window_size=20)
    return temporal_train_val_test_split(X, y)


def test_format_comparison_table_hand_computed() -> None:
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 2])
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    result = ModelComparison(
        name="perfect_model",
        evaluation=evaluate(y_true, y_pred),
        calibration=calibrate(y_true, probs),
    )

    table = format_comparison_table([result])

    assert "perfect_model" in table
    assert "1.0000" in table  # accuracy, macro F1, and brier=0 all show as 0.0000/1.0000
    assert "0.0000" in table
    header, separator, _row = table.splitlines()
    assert "accuracy" in header
    assert "macro F1" in header
    assert "brier" in header
    assert "ECE" in header
    assert set(separator) == {"-"}


def test_format_comparison_table_handles_multiple_models() -> None:
    y_true = np.array([0, 1, 2])
    probs = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.3, 0.2, 0.5]])
    y_pred = probs.argmax(axis=1)
    result = ModelComparison(
        name="model_a", evaluation=evaluate(y_true, y_pred), calibration=calibrate(y_true, probs)
    )
    table = format_comparison_table([result, result])
    # header + separator + one row per model
    assert len(table.splitlines()) == 4


def test_evaluate_baseline_runs_end_to_end_on_a_tiny_split() -> None:
    split = _build_tiny_split()
    result = evaluate_baseline("logistic_regression", LogisticRegressionBaseline(seed=0), split)

    assert result.name == "logistic_regression"
    assert 0.0 <= result.evaluation.accuracy <= 1.0
    assert result.calibration.brier_score >= 0.0
    assert result.calibration.expected_calibration_error >= 0.0


def test_evaluate_neural_runs_end_to_end_on_a_tiny_split() -> None:
    split = _build_tiny_split()
    result = evaluate_neural("cnn_lstm", DeepLOBCNNLSTM(window_size=20), split)

    assert result.name == "cnn_lstm"
    assert 0.0 <= result.evaluation.accuracy <= 1.0
    assert result.calibration.brier_score >= 0.0
    assert result.calibration.expected_calibration_error >= 0.0


def test_build_split_produces_temporally_ordered_nonoverlapping_splits() -> None:
    split = build_split()
    assert split.X_train.shape[0] > 0
    assert split.X_val.shape[0] > 0
    assert split.X_test.shape[0] > 0
    assert split.X_train.shape[1:] == split.X_val.shape[1:] == split.X_test.shape[1:]
