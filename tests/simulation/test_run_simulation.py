import lightning as L  # noqa: N812 -- Lightning's own docs and ecosystem-wide convention
import numpy as np
from torch.utils.data import DataLoader

from deeplob.data.labeling import Label, compute_labels
from deeplob.data.lob import mid_price
from deeplob.data.synthetic import generate_synthetic_lob
from deeplob.data.windowing import make_windows
from deeplob.evaluation.splits import temporal_train_val_test_split
from deeplob.models.cnn_lstm import DeepLOBCNNLSTM
from deeplob.simulation.pnl import predictions_to_positions, simulate_pnl
from deeplob.simulation.run_simulation import aligned_test_mid_prices
from deeplob.training.dataset import LOBWindowDataset
from deeplob.training.lightning_module import LOBClassifier, collect_predictions

_WINDOW_SIZE = 20


def test_aligned_test_mid_prices_length_matches_test_split_labels() -> None:
    features = generate_synthetic_lob(num_snapshots=1500, seed=5)
    mid = mid_price(features)
    labels = compute_labels(mid, horizon=10, alpha=0.0005)
    X, y = make_windows(features, labels, window_size=_WINDOW_SIZE)
    split = temporal_train_val_test_split(X, y)

    test_prices = aligned_test_mid_prices(mid, labels, _WINDOW_SIZE)

    assert test_prices.shape == split.y_test.shape


def test_aligned_test_mid_prices_are_a_contiguous_slice_of_the_real_mid_price_series() -> None:
    # Consecutive test windows are built with stride 1, so the aligned test prices must
    # themselves be a plain contiguous run of consecutive mid-price values -- verified
    # directly against the real `mid` array, not just trusted from the windowing logic.
    features = generate_synthetic_lob(num_snapshots=1500, seed=5)
    mid = mid_price(features)
    labels = compute_labels(mid, horizon=10, alpha=0.0005)

    test_prices = aligned_test_mid_prices(mid, labels, _WINDOW_SIZE)

    # Find where this exact contiguous run occurs in the real mid-price series.
    first_price = test_prices[0]
    candidate_starts = np.where(np.isclose(mid, first_price))[0]
    assert len(candidate_starts) > 0
    start = candidate_starts[0]
    np.testing.assert_allclose(mid[start : start + len(test_prices)], test_prices)


def test_pnl_pipeline_runs_end_to_end_on_a_tiny_split() -> None:
    features = generate_synthetic_lob(num_snapshots=1500, seed=5)
    mid = mid_price(features)
    labels = compute_labels(mid, horizon=10, alpha=0.0005)
    X, y = make_windows(features, labels, window_size=_WINDOW_SIZE)
    split = temporal_train_val_test_split(X, y)

    train_loader = DataLoader(
        LOBWindowDataset(split.X_train, split.y_train), batch_size=8, shuffle=True
    )
    test_loader = DataLoader(LOBWindowDataset(split.X_test, split.y_test), batch_size=8)

    module = LOBClassifier(DeepLOBCNNLSTM(window_size=_WINDOW_SIZE), learning_rate=1e-3)
    trainer = L.Trainer(
        accelerator="cpu",
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(module, train_loader)

    _, y_pred = collect_predictions(module, test_loader)
    test_prices = aligned_test_mid_prices(mid, labels, _WINDOW_SIZE)
    assert test_prices.shape[0] == y_pred.shape[0]

    positions = predictions_to_positions(y_pred[:-1])
    assert set(positions.tolist()) <= {-1.0, 0.0, 1.0}

    report = simulate_pnl(test_prices, positions, transaction_cost_bps=10.0)

    assert report.num_steps == positions.shape[0]
    assert report.total_transaction_cost >= 0.0
    assert np.isfinite(report.gross_pnl)
    assert np.isfinite(report.net_pnl)
    # Costs only ever reduce PnL relative to the gross figure -- net can equal gross (zero
    # trades) but never exceed it.
    assert report.net_pnl <= report.gross_pnl + 1e-12


def test_predictions_to_positions_handles_every_label_present_in_a_real_prediction_array() -> None:
    predictions = np.array([Label.DOWN, Label.STATIONARY, Label.UP])
    positions = predictions_to_positions(predictions)
    assert positions.shape == (3,)
    np.testing.assert_array_equal(positions, [-1.0, 0.0, 1.0])
