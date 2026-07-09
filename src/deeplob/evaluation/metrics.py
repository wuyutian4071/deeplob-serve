"""Evaluation metrics shared by every model in this project (baselines through the
Transformer) -- built once here, reused everywhere, so every model's numbers are directly
comparable. Uses scikit-learn's own metric implementations (well-tested, standard) rather
than reimplementing F1/confusion-matrix math from scratch.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from deeplob.data.labeling import Label

_CLASS_NAMES = ["DOWN", "STATIONARY", "UP"]
_CLASS_LABELS = [Label.DOWN, Label.STATIONARY, Label.UP]


@dataclass(frozen=True)
class EvaluationReport:
    accuracy: float
    f1_per_class: dict[str, float]
    macro_f1: float
    confusion: np.ndarray  # [3, 3], rows=true class, columns=predicted class

    def summary(self) -> str:
        lines = [
            f"accuracy:  {self.accuracy:.4f}",
            f"macro F1:  {self.macro_f1:.4f}",
            "F1 per class:",
        ]
        lines += [f"  {name:<12} {self.f1_per_class[name]:.4f}" for name in _CLASS_NAMES]
        lines.append(f"confusion matrix (rows=true, cols=predicted, order={_CLASS_NAMES}):")
        lines += [f"  {row}" for row in self.confusion.tolist()]
        return "\n".join(lines)


@dataclass(frozen=True)
class CalibrationBin:
    """One bin of a reliability diagram: predictions bucketed by the model's own top
    predicted probability ("confidence"), each bin's mean confidence checked against how
    often those predictions were actually correct.
    """

    bin_lower: float
    bin_upper: float
    mean_confidence: float
    observed_accuracy: float
    count: int


@dataclass(frozen=True)
class CalibrationReport:
    brier_score: float
    bins: list[CalibrationBin]
    # Weighted average |confidence - accuracy| across non-empty bins -- 0 is perfectly
    # calibrated, higher means the model's confidence systematically over- or
    # under-states how often it's actually right.
    expected_calibration_error: float


def brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Multi-class Brier score: mean over samples of the sum-of-squared-differences between
    the one-hot true label and the predicted probability vector. 0 is a perfect prediction;
    for a uniform (uninformative) 3-class prediction, this works out to 2/3 ~ 0.667 --
    confirmed by hand and covered by a test, not just asserted here.
    """
    if y_true.shape[0] != probs.shape[0]:
        raise ValueError("y_true and probs must have the same length")
    if y_true.shape[0] == 0:
        raise ValueError("cannot score an empty set of predictions")

    n, num_classes = probs.shape
    one_hot = np.zeros((n, num_classes))
    one_hot[np.arange(n), y_true] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def calibrate(y_true: np.ndarray, probs: np.ndarray, num_bins: int = 10) -> CalibrationReport:
    """Reliability diagram + Brier score from `probs` ([N, 3] predicted class
    probabilities). Bins by top-label confidence (`probs.max(axis=1)`) into `num_bins`
    equal-width bins over [0, 1] -- empty bins are omitted from `.bins`, not reported as
    zero/NaN, since an empty bin has no meaningful confidence or accuracy to show.
    """
    if y_true.shape[0] != probs.shape[0]:
        raise ValueError("y_true and probs must have the same length")
    if y_true.shape[0] == 0:
        raise ValueError("cannot calibrate an empty set of predictions")

    confidence = probs.max(axis=1)
    predicted_class = probs.argmax(axis=1)
    correct = predicted_class == y_true

    edges = np.linspace(0.0, 1.0, num_bins + 1)
    bins = []
    ece = 0.0
    n = y_true.shape[0]
    for i in range(num_bins):
        lower, upper = edges[i], edges[i + 1]
        # The last bin's upper edge is inclusive (a confidence of exactly 1.0 must land
        # somewhere), every other bin's upper edge is exclusive.
        in_bin = (
            (confidence >= lower) & (confidence <= upper)
            if i == num_bins - 1
            else (confidence >= lower) & (confidence < upper)
        )
        count = int(in_bin.sum())
        if count == 0:
            continue
        mean_confidence = float(confidence[in_bin].mean())
        observed_accuracy = float(correct[in_bin].mean())
        bins.append(CalibrationBin(lower, upper, mean_confidence, observed_accuracy, count))
        ece += (count / n) * abs(mean_confidence - observed_accuracy)

    return CalibrationReport(
        brier_score=brier_score(y_true, probs), bins=bins, expected_calibration_error=ece
    )


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> EvaluationReport:
    """`y_true`/`y_pred` are [N] arrays of `Label` values (DOWN=0, STATIONARY=1, UP=2) -- no
    `INVALID_LABEL` entries; callers are expected to have already dropped those (which
    `deeplob.data.windowing.make_windows` does by construction).
    """
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")
    if y_true.shape[0] == 0:
        raise ValueError("cannot evaluate an empty set of predictions")

    per_class_f1 = f1_score(y_true, y_pred, labels=_CLASS_LABELS, average=None, zero_division=0)
    macro_f1 = float(
        f1_score(y_true, y_pred, labels=_CLASS_LABELS, average="macro", zero_division=0)
    )
    accuracy = float(np.mean(y_true == y_pred))
    confusion = confusion_matrix(y_true, y_pred, labels=_CLASS_LABELS)

    return EvaluationReport(
        accuracy=accuracy,
        f1_per_class=dict(zip(_CLASS_NAMES, per_class_f1.tolist(), strict=True)),
        macro_f1=macro_f1,
        confusion=confusion,
    )
