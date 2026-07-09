import lightning as L  # noqa: N812 -- Lightning's own docs and ecosystem-wide convention
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from deeplob.training.dataset import LOBWindowDataset
from deeplob.training.lightning_module import (
    LOBClassifier,
    collect_predictions,
    collect_probabilities,
)

_WINDOW_SIZE = 5
_NUM_FEATURES = 4


class _TinyModel(nn.Module):
    """A minimal model satisfying LOBClassifier's expected interface -- just enough to
    exercise the wrapper's training loop plumbing, not a real architecture (that's the
    CNN-LSTM's job, task #95).
    """

    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(_WINDOW_SIZE * _NUM_FEATURES, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.linear(self.flatten(x))
        return result


def _make_dataloader(n: int) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    features = torch.randn(n, _WINDOW_SIZE, _NUM_FEATURES).numpy()
    labels = torch.randint(0, 3, (n,)).numpy()
    return DataLoader(LOBWindowDataset(features, labels), batch_size=4)


def test_forward_produces_three_class_logits() -> None:
    module = LOBClassifier(_TinyModel())
    x = torch.randn(2, _WINDOW_SIZE, _NUM_FEATURES)
    logits = module(x)
    assert logits.shape == (2, 3)


def test_training_step_returns_a_scalar_loss() -> None:
    module = LOBClassifier(_TinyModel())
    x = torch.randn(4, _WINDOW_SIZE, _NUM_FEATURES)
    y = torch.randint(0, 3, (4,))
    loss = module.training_step((x, y), batch_idx=0)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_a_short_trainer_fit_run_completes_without_error() -> None:
    module = LOBClassifier(_TinyModel(), learning_rate=1e-3)
    train_loader = _make_dataloader(16)
    val_loader = _make_dataloader(8)

    trainer = L.Trainer(
        accelerator="cpu",
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(module, train_loader, val_loader)  # must not raise


def test_collect_predictions_returns_valid_labels_for_every_sample() -> None:
    module = LOBClassifier(_TinyModel())
    loader = _make_dataloader(10)

    y_true, y_pred = collect_predictions(module, loader)

    assert y_true.shape == (10,)
    assert y_pred.shape == (10,)
    assert set(y_pred.tolist()) <= {0, 1, 2}


def test_collect_probabilities_returns_a_valid_distribution_per_sample() -> None:
    module = LOBClassifier(_TinyModel())
    loader = _make_dataloader(10)

    y_true, probs = collect_probabilities(module, loader)

    assert y_true.shape == (10,)
    assert probs.shape == (10, 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    assert (probs >= 0.0).all()


def test_collect_probabilities_argmax_matches_collect_predictions() -> None:
    # Same underlying model, same forward pass -- the two collection functions must agree on
    # which class each sample is predicted as, not just each independently return something
    # shaped correctly.
    module = LOBClassifier(_TinyModel())
    loader = _make_dataloader(10)

    _, y_pred = collect_predictions(module, loader)
    _, probs = collect_probabilities(module, loader)

    np.testing.assert_array_equal(y_pred, probs.argmax(axis=1))
