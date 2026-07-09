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
