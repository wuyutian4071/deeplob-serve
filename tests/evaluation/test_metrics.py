import numpy as np
import pytest

from deeplob.data.labeling import Label
from deeplob.evaluation.metrics import evaluate


def test_perfect_predictions_score_maximally_on_every_metric() -> None:
    y_true = np.array([Label.DOWN, Label.STATIONARY, Label.UP, Label.DOWN, Label.UP])
    report = evaluate(y_true, y_true.copy())

    assert report.accuracy == pytest.approx(1.0)
    assert report.macro_f1 == pytest.approx(1.0)
    for name in ("DOWN", "STATIONARY", "UP"):
        assert report.f1_per_class[name] == pytest.approx(1.0)
    # y_true has 2 DOWN, 1 STATIONARY, 2 UP -- that's the expected diagonal.
    np.testing.assert_array_equal(report.confusion, np.diag([2, 1, 2]))


def test_hand_computed_confusion_matrix_and_accuracy() -> None:
    # 4 true DOWN, 2 predicted correctly, 2 predicted as STATIONARY.
    y_true = np.array([Label.DOWN, Label.DOWN, Label.DOWN, Label.DOWN])
    y_pred = np.array([Label.DOWN, Label.DOWN, Label.STATIONARY, Label.STATIONARY])

    report = evaluate(y_true, y_pred)

    assert report.accuracy == pytest.approx(0.5)
    # Confusion rows/cols ordered [DOWN, STATIONARY, UP]: 2 true-DOWN predicted DOWN,
    # 2 true-DOWN predicted STATIONARY, everything else zero.
    expected = np.array([[2, 2, 0], [0, 0, 0], [0, 0, 0]])
    np.testing.assert_array_equal(report.confusion, expected)


def test_a_class_with_zero_true_instances_gets_zero_f1_not_an_error() -> None:
    # No STATIONARY or UP in y_true at all -- f1_score would normally warn/error on
    # undefined precision for classes with zero predicted+true instances; zero_division=0
    # must make this a clean 0.0, not a crash.
    y_true = np.array([Label.DOWN, Label.DOWN, Label.DOWN])
    y_pred = np.array([Label.DOWN, Label.DOWN, Label.DOWN])

    report = evaluate(y_true, y_pred)

    assert report.f1_per_class["DOWN"] == pytest.approx(1.0)
    assert report.f1_per_class["STATIONARY"] == pytest.approx(0.0)
    assert report.f1_per_class["UP"] == pytest.approx(0.0)


def test_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        evaluate(np.array([Label.DOWN, Label.UP]), np.array([Label.DOWN]))


def test_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        evaluate(np.array([]), np.array([]))


def test_summary_is_a_non_empty_readable_string() -> None:
    y_true = np.array([Label.DOWN, Label.UP])
    report = evaluate(y_true, y_true.copy())
    summary = report.summary()
    assert "accuracy" in summary
    assert "macro F1" in summary
