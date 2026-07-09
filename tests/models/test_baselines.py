import numpy as np

from deeplob.data.labeling import compute_labels
from deeplob.data.lob import mid_price
from deeplob.data.synthetic import generate_synthetic_lob
from deeplob.data.windowing import make_windows
from deeplob.evaluation.metrics import evaluate
from deeplob.evaluation.splits import temporal_train_val_test_split
from deeplob.models.baselines import (
    GradientBoostingBaseline,
    LogisticRegressionBaseline,
    flatten_windows,
)


def test_flatten_windows_hand_computed() -> None:
    X = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    flat = flatten_windows(X)
    assert flat.shape == (2, 12)
    np.testing.assert_array_equal(flat[0], np.arange(12))
    np.testing.assert_array_equal(flat[1], np.arange(12, 24))


def _build_windowed_dataset() -> tuple[np.ndarray, np.ndarray]:
    features = generate_synthetic_lob(num_snapshots=1500, seed=11)
    labels = compute_labels(mid_price(features), horizon=10, alpha=0.0005)
    X, y = make_windows(features, labels, window_size=20)
    return X, y


def test_logistic_regression_baseline_fits_and_predicts_valid_labels_end_to_end() -> None:
    X, y = _build_windowed_dataset()
    split = temporal_train_val_test_split(X, y)

    model = LogisticRegressionBaseline(seed=0).fit(split.X_train, split.y_train)
    predictions = model.predict(split.X_test)

    assert predictions.shape == split.y_test.shape
    assert set(predictions.tolist()) <= {0, 1, 2}

    report = evaluate(split.y_test, predictions)
    assert 0.0 <= report.accuracy <= 1.0


def test_gradient_boosting_baseline_fits_and_predicts_valid_labels_end_to_end() -> None:
    X, y = _build_windowed_dataset()
    split = temporal_train_val_test_split(X, y)

    model = GradientBoostingBaseline(n_estimators=20, max_depth=2, seed=0).fit(
        split.X_train, split.y_train
    )
    predictions = model.predict(split.X_test)

    assert predictions.shape == split.y_test.shape
    assert set(predictions.tolist()) <= {0, 1, 2}

    report = evaluate(split.y_test, predictions)
    assert 0.0 <= report.accuracy <= 1.0
