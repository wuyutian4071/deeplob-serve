import numpy as np
import pytest

from deeplob.data.labeling import Label
from deeplob.evaluation.metrics import brier_score, calibrate, evaluate


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


def test_brier_score_is_zero_for_a_perfect_confident_prediction() -> None:
    y_true = np.array([Label.DOWN])
    probs = np.array([[1.0, 0.0, 0.0]])
    assert brier_score(y_true, probs) == pytest.approx(0.0)


def test_brier_score_for_a_uniform_prediction_is_two_thirds() -> None:
    # (1/3 - 1)^2 + (1/3)^2 + (1/3)^2 = 4/9 + 1/9 + 1/9 = 6/9 = 2/3, hand-computed.
    y_true = np.array([Label.DOWN])
    probs = np.array([[1 / 3, 1 / 3, 1 / 3]])
    assert brier_score(y_true, probs) == pytest.approx(2 / 3)


def test_brier_score_for_a_confidently_wrong_prediction_is_two() -> None:
    # y_true=DOWN, predicted UP with full confidence: (0-1)^2 + 0^2 + (1-0)^2 = 2.
    y_true = np.array([Label.DOWN])
    probs = np.array([[0.0, 0.0, 1.0]])
    assert brier_score(y_true, probs) == pytest.approx(2.0)


def test_brier_score_averages_across_samples() -> None:
    # One perfect (brier=0) and one confidently-wrong (brier=2) sample -> mean = 1.0.
    y_true = np.array([Label.DOWN, Label.DOWN])
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert brier_score(y_true, probs) == pytest.approx(1.0)


def test_brier_score_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        brier_score(np.array([Label.DOWN, Label.UP]), np.array([[1.0, 0.0, 0.0]]))


def test_brier_score_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        brier_score(np.array([]), np.zeros((0, 3)))


def test_calibrate_hand_computed_two_bin_example() -> None:
    # Bins: [0, 0.5), [0.5, 1.0]. Three samples, worked by hand (see module docstring's
    # reasoning replicated in the test module for the calling code, not restated here):
    #   A: probs=[0.9, 0.05, 0.05], true=DOWN  -> confidence=0.9, predicted DOWN,  correct
    #   B: probs=[0.05, 0.9, 0.05], true=DOWN  -> confidence=0.9, predicted STATIONARY, wrong
    #   C: probs=[0.4, 0.35, 0.25], true=DOWN  -> confidence=0.4, predicted DOWN,  correct
    y_true = np.array([Label.DOWN, Label.DOWN, Label.DOWN])
    probs = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.4, 0.35, 0.25]])

    report = calibrate(y_true, probs, num_bins=2)

    assert len(report.bins) == 2
    low_bin, high_bin = sorted(report.bins, key=lambda b: b.bin_lower)

    assert low_bin.count == 1
    assert low_bin.mean_confidence == pytest.approx(0.4)
    assert low_bin.observed_accuracy == pytest.approx(1.0)

    assert high_bin.count == 2
    assert high_bin.mean_confidence == pytest.approx(0.9)
    assert high_bin.observed_accuracy == pytest.approx(0.5)

    # ECE = (2/3)*|0.9-0.5| + (1/3)*|0.4-1.0| = (2/3)*0.4 + (1/3)*0.6, hand-computed.
    expected_ece = (2 / 3) * 0.4 + (1 / 3) * 0.6
    assert report.expected_calibration_error == pytest.approx(expected_ece)

    # Brier score for this same set, hand-computed per-sample and averaged:
    #   A: (0.9-1)^2+0.05^2+0.05^2 = 0.015
    #   B: (0.05-1)^2+0.9^2+0.05^2 = 1.715
    #   C: (0.4-1)^2+0.35^2+0.25^2 = 0.545
    #   mean = (0.015+1.715+0.545)/3
    expected_brier = (0.015 + 1.715 + 0.545) / 3
    assert report.brier_score == pytest.approx(expected_brier)


def test_calibrate_a_confidence_of_exactly_one_lands_in_the_last_bin() -> None:
    y_true = np.array([Label.DOWN])
    probs = np.array([[1.0, 0.0, 0.0]])
    report = calibrate(y_true, probs, num_bins=10)
    assert len(report.bins) == 1
    assert report.bins[0].bin_upper == pytest.approx(1.0)
    assert report.bins[0].count == 1


def test_calibrate_omits_empty_bins() -> None:
    # Every sample has confidence >= 0.9 -- with 10 equal-width bins, only the top bin(s)
    # should appear, not 10 bins with 9 reported as zero/NaN.
    y_true = np.array([Label.DOWN, Label.DOWN])
    probs = np.array([[0.95, 0.03, 0.02], [0.92, 0.05, 0.03]])
    report = calibrate(y_true, probs, num_bins=10)
    assert len(report.bins) == 1
    assert report.bins[0].count == 2


def test_calibrate_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        calibrate(np.array([Label.DOWN, Label.UP]), np.array([[1.0, 0.0, 0.0]]))


def test_calibrate_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        calibrate(np.array([]), np.zeros((0, 3)))
