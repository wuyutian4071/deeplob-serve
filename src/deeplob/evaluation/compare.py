"""Comparative evaluation across every model this project has built: the two M3 baselines
(logistic regression, gradient boosting) and the two M4/M5 neural models (CNN-LSTM,
Transformer) -- fit/trained and evaluated against the exact same synthetic dataset and
temporal split, so the comparison is apples-to-apples rather than four separate runs each on
different random data.

Run via: uv run python -m deeplob.evaluation.compare
"""

from dataclasses import dataclass

import lightning as L  # noqa: N812 -- Lightning's own docs and ecosystem-wide convention
from torch.utils.data import DataLoader

from deeplob.data.labeling import compute_labels
from deeplob.data.lob import mid_price
from deeplob.data.synthetic import generate_synthetic_lob
from deeplob.data.windowing import make_windows
from deeplob.evaluation.metrics import CalibrationReport, EvaluationReport, calibrate, evaluate
from deeplob.evaluation.splits import TemporalSplit, temporal_train_val_test_split
from deeplob.models.baselines import GradientBoostingBaseline, LogisticRegressionBaseline
from deeplob.models.cnn_lstm import DeepLOBCNNLSTM
from deeplob.models.transformer import LOBTransformer
from deeplob.training.dataset import LOBWindowDataset
from deeplob.training.device import resolve_accelerator, resolve_precision
from deeplob.training.lightning_module import LOBClassifier, collect_probabilities
from deeplob.training.seeding import set_seed

_SEED = 42
_NUM_SYNTHETIC_SNAPSHOTS = 20000
# Deliberately smaller than train.py's own default (100): at window_size=100, the flattened
# baseline feature count is window_size * NUM_FEATURES = 4000, and sklearn's
# GradientBoostingClassifier (an exact, non-histogram split-search, unlike
# HistGradientBoostingClassifier) scales badly with feature count -- confirmed by actually
# running this comparison twice: 100 estimators took 7+ minutes and was killed, and even 30
# estimators was still running past 10 minutes and was killed again. window_size=30 (still
# comfortably above DeepLOBCNNLSTM's window_size>18 requirement) keeps every baseline's fit
# time practical for a comparison script meant to be re-run easily -- this doesn't change the
# qualitative story, since no window size gives this pure-random-walk synthetic data any real
# signal for tree count or feature count to matter.
_WINDOW_SIZE = 30
_HORIZON = 10
_ALPHA = 0.0005
_TRAIN_FRAC = 0.7
_VAL_FRAC = 0.15
_BATCH_SIZE = 64
_MAX_EPOCHS = 5


@dataclass(frozen=True)
class ModelComparison:
    name: str
    evaluation: EvaluationReport
    calibration: CalibrationReport


def build_split() -> TemporalSplit:
    features = generate_synthetic_lob(num_snapshots=_NUM_SYNTHETIC_SNAPSHOTS, seed=_SEED)
    labels = compute_labels(mid_price(features), horizon=_HORIZON, alpha=_ALPHA)
    X, y = make_windows(features, labels, window_size=_WINDOW_SIZE)
    return temporal_train_val_test_split(X, y, train_frac=_TRAIN_FRAC, val_frac=_VAL_FRAC)


def evaluate_baseline(
    name: str,
    model: LogisticRegressionBaseline | GradientBoostingBaseline,
    split: TemporalSplit,
) -> ModelComparison:
    # Re-seed immediately before each model, not just once at the top of main() -- otherwise
    # a later model's random initialization depends on how much randomness an earlier model
    # happened to consume, which would make each model's own result non-reproducible in
    # isolation (this project's whole point with seeding.set_seed is per-run determinism).
    set_seed(_SEED)
    model.fit(split.X_train, split.y_train)
    probs = model.predict_proba(split.X_test)
    y_pred = probs.argmax(axis=1)
    return ModelComparison(
        name=name,
        evaluation=evaluate(split.y_test, y_pred),
        calibration=calibrate(split.y_test, probs),
    )


def evaluate_neural(
    name: str, model: DeepLOBCNNLSTM | LOBTransformer, split: TemporalSplit
) -> ModelComparison:
    set_seed(_SEED)

    train_loader = DataLoader(
        LOBWindowDataset(split.X_train, split.y_train), batch_size=_BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(LOBWindowDataset(split.X_val, split.y_val), batch_size=_BATCH_SIZE)
    test_loader = DataLoader(LOBWindowDataset(split.X_test, split.y_test), batch_size=_BATCH_SIZE)

    module = LOBClassifier(model, learning_rate=1e-3)
    accelerator = resolve_accelerator()
    precision = resolve_precision(accelerator)
    trainer = L.Trainer(
        accelerator=accelerator,
        precision=precision,
        max_epochs=_MAX_EPOCHS,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(module, train_loader, val_loader)

    y_true, probs = collect_probabilities(module, test_loader)
    y_pred = probs.argmax(axis=1)
    return ModelComparison(
        name=name, evaluation=evaluate(y_true, y_pred), calibration=calibrate(y_true, probs)
    )


def format_comparison_table(results: list[ModelComparison]) -> str:
    header = f"{'model':<22}{'accuracy':>10}{'macro F1':>10}{'brier':>10}{'ECE':>10}"
    lines = [header, "-" * len(header)]
    for r in results:
        lines.append(
            f"{r.name:<22}{r.evaluation.accuracy:>10.4f}{r.evaluation.macro_f1:>10.4f}"
            f"{r.calibration.brier_score:>10.4f}{r.calibration.expected_calibration_error:>10.4f}"
        )
    return "\n".join(lines)


def main() -> None:
    split = build_split()

    results = [
        # max_iter raised well past the class's own default (1000): at this comparison's
        # window_size=100, flattened features are window_size * NUM_FEATURES = 4000 --
        # 5x M3's own test fixtures (window_size=20, 800 features) -- and 1000 iterations
        # wasn't enough at this scale even with M3's scaling fix already applied, confirmed
        # by actually running this and hitting sklearn's own ConvergenceWarning.
        evaluate_baseline(
            "logistic_regression", LogisticRegressionBaseline(max_iter=5000, seed=_SEED), split
        ),
        # n_estimators reduced from the class's default (100): boosting is inherently
        # sequential (unlike a random forest, trees can't be built in parallel), and 100
        # trees over 4000 features x ~14000 training samples took long enough that it was
        # killed rather than waited out -- 30 trees is still substantially more than M3's own
        # quick-test fixtures (n_estimators=20) and the qualitative finding (near chance on
        # signal-free synthetic data) doesn't depend on tree count.
        evaluate_baseline(
            "gradient_boosting", GradientBoostingBaseline(n_estimators=30, seed=_SEED), split
        ),
        evaluate_neural("cnn_lstm", DeepLOBCNNLSTM(window_size=_WINDOW_SIZE), split),
        evaluate_neural("transformer", LOBTransformer(window_size=_WINDOW_SIZE), split),
    ]

    print(format_comparison_table(results))


if __name__ == "__main__":
    main()
