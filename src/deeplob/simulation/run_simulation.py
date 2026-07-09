"""Runs the trading-signal simulation end to end on synthetic data: trains the CNN-LSTM
(selected in M6 on architecture/robustness grounds, see the README's M6 section), predicts on
the held-out test set, translates predictions into a naive position, and reports gross vs.
net PnL after a realistic transaction cost.

Run via: uv run python -m deeplob.simulation.run_simulation
"""

import lightning as L  # noqa: N812 -- Lightning's own docs and ecosystem-wide convention
import numpy as np
from torch.utils.data import DataLoader

from deeplob.data.labeling import compute_labels
from deeplob.data.lob import mid_price
from deeplob.data.synthetic import generate_synthetic_lob
from deeplob.data.windowing import make_windows
from deeplob.evaluation.splits import temporal_train_val_test_split
from deeplob.models.cnn_lstm import DeepLOBCNNLSTM
from deeplob.simulation.pnl import predictions_to_positions, simulate_pnl
from deeplob.training.dataset import LOBWindowDataset
from deeplob.training.device import resolve_accelerator, resolve_precision
from deeplob.training.lightning_module import LOBClassifier, collect_predictions
from deeplob.training.seeding import set_seed

_SEED = 42
_NUM_SYNTHETIC_SNAPSHOTS = 20000
_WINDOW_SIZE = 100
_HORIZON = 10
_ALPHA = 0.0005
_TRAIN_FRAC = 0.7
_VAL_FRAC = 0.15
_BATCH_SIZE = 64
_MAX_EPOCHS = 5
# 10 bps (0.001) round-trip-equivalent per unit of position change -- a realistic, if
# arbitrary, illustrative cost, in the same ballpark as typical equity/futures retail
# commission-plus-spread costs. Not calibrated against any specific real venue's actual fee
# schedule; a real deployment would use that venue's own numbers.
_TRANSACTION_COST_BPS = 10.0


def aligned_test_mid_prices(mid: np.ndarray, labels: np.ndarray, window_size: int) -> np.ndarray:
    """The mid-price at each *test* window's end position, aligned index-for-index with
    `y_test`/predictions on that split. Built by windowing and splitting `mid` itself through
    the exact same `make_windows`/`temporal_train_val_test_split` calls the real features go
    through (not a hand-reimplemented copy of their index math) -- consecutive windows in
    this project's data are built with stride 1, so each test window's own last timestep
    *is* the mid-price at that prediction's time, in order.
    """
    mid_as_feature = mid.reshape(-1, 1)
    mid_windows, mid_window_labels = make_windows(mid_as_feature, labels, window_size)
    price_split = temporal_train_val_test_split(
        mid_windows, mid_window_labels, train_frac=_TRAIN_FRAC, val_frac=_VAL_FRAC
    )
    result: np.ndarray = price_split.X_test[:, -1, 0]
    return result


def main() -> None:
    set_seed(_SEED)

    features = generate_synthetic_lob(num_snapshots=_NUM_SYNTHETIC_SNAPSHOTS, seed=_SEED)
    mid = mid_price(features)
    labels = compute_labels(mid, horizon=_HORIZON, alpha=_ALPHA)
    X, y = make_windows(features, labels, window_size=_WINDOW_SIZE)
    split = temporal_train_val_test_split(X, y, train_frac=_TRAIN_FRAC, val_frac=_VAL_FRAC)

    train_loader = DataLoader(
        LOBWindowDataset(split.X_train, split.y_train), batch_size=_BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(LOBWindowDataset(split.X_val, split.y_val), batch_size=_BATCH_SIZE)
    test_loader = DataLoader(LOBWindowDataset(split.X_test, split.y_test), batch_size=_BATCH_SIZE)

    model = DeepLOBCNNLSTM(window_size=_WINDOW_SIZE)
    module = LOBClassifier(model, learning_rate=1e-3)
    accelerator = resolve_accelerator()
    trainer = L.Trainer(
        accelerator=accelerator,
        precision=resolve_precision(accelerator),
        max_epochs=_MAX_EPOCHS,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(module, train_loader, val_loader)

    _, y_pred = collect_predictions(module, test_loader)
    test_mid_prices = aligned_test_mid_prices(mid, labels, _WINDOW_SIZE)
    assert test_mid_prices.shape[0] == y_pred.shape[0]

    # Every predicted position except the last can be evaluated: it needs the *next* window's
    # end price (test_mid_prices[i+1]) to compute the return it earned, which the final
    # prediction doesn't have yet.
    positions = predictions_to_positions(y_pred[:-1])
    report = simulate_pnl(test_mid_prices, positions, transaction_cost_bps=_TRANSACTION_COST_BPS)

    print(report.summary())


if __name__ == "__main__":
    main()
