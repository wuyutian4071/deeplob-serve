"""Logistic regression and gradient boosting baselines -- the lower bar every later model
(CNN-LSTM, Transformer) is expected to beat. Both operate on FLATTENED windowed features
(`window_size * NUM_FEATURES`) rather than the sequential `[window_size, NUM_FEATURES]` shape
the CNN-LSTM consumes directly, since these are non-sequential scikit-learn models -- reuses
the same windowed `(X, y)` data `deeplob.data.windowing.make_windows` produces for every
model, just adapted at this boundary, so the data pipeline and evaluation harness stay fully
shared across every model in this project.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def flatten_windows(X: np.ndarray) -> np.ndarray:
    """`[N, window_size, NUM_FEATURES]` -> `[N, window_size * NUM_FEATURES]`."""
    n = X.shape[0]
    return X.reshape(n, -1)


class LogisticRegressionBaseline:
    """Multinomial logistic regression over flattened LOB windows.

    Raw LOB features mix prices (~100) and volumes (~1-500) on very different scales, which
    left unscaled fails to converge within a reasonable iteration budget (confirmed: this
    baseline's first version hit sklearn's own ConvergenceWarning on synthetic data) --
    standardization is genuinely necessary here, not a defensive habit. A `Pipeline` fits the
    scaler on training data only and just transforms with it at predict time, the standard
    leakage-free way to do this (never re-fit the scaler on validation/test data).
    """

    def __init__(self, max_iter: int = 1000, seed: int = 0) -> None:
        self._model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=max_iter, random_state=seed)),
            ]
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionBaseline":
        self._model.fit(flatten_windows(X), y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        result: np.ndarray = self._model.predict(flatten_windows(X))
        return result

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """`[N, 3]` class probabilities, columns ordered `[DOWN, STATIONARY, UP]` -- holds as
        long as training data contains all three classes (sklearn orders `predict_proba`'s
        columns by `classes_`, the sorted unique labels seen during `fit`, which for
        `Label`'s `IntEnum` values 0/1/2 is exactly `[DOWN, STATIONARY, UP]`).
        """
        result: np.ndarray = self._model.predict_proba(flatten_windows(X))
        return result


class GradientBoostingBaseline:
    """Gradient-boosted trees over flattened LOB windows."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 3, seed: int = 0) -> None:
        self._model = GradientBoostingClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=seed
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingBaseline":
        self._model.fit(flatten_windows(X), y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        result: np.ndarray = self._model.predict(flatten_windows(X))
        return result

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """`[N, 3]` class probabilities, columns ordered `[DOWN, STATIONARY, UP]` -- see
        `LogisticRegressionBaseline.predict_proba`'s docstring for why this column order
        holds.
        """
        result: np.ndarray = self._model.predict_proba(flatten_windows(X))
        return result
