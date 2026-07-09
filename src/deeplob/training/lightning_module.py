"""A generic Lightning wrapper around any classifier `nn.Module` producing 3-class logits
from windowed LOB sequences -- the CNN-LSTM (M4) and Transformer (M5) both plug into this
same training loop, optimizer, and logging setup rather than each reimplementing it.
"""

from typing import Any

import lightning as L  # noqa: N812 -- Lightning's own docs and ecosystem-wide convention
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


class LOBClassifier(L.LightningModule):
    """Wraps `model` (any `nn.Module` mapping `[batch, window_size, NUM_FEATURES]` to
    `[batch, 3]` logits) with cross-entropy training/validation/test steps and Adam.
    """

    def __init__(self, model: nn.Module, learning_rate: float = 1e-3) -> None:
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.model(x)
        return result

    def _step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        accuracy = (logits.argmax(dim=-1) == y).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_accuracy", accuracy, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._step(batch, "val")

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "test")

    def configure_optimizers(self) -> Any:  # Lightning's own signature returns Any
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def collect_predictions(
    module: LOBClassifier, dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]]
) -> tuple[np.ndarray, np.ndarray]:
    """Runs `module` in eval mode over every batch in `dataloader`, returning `(y_true,
    y_pred)` as numpy arrays -- for feeding into `deeplob.evaluation.metrics.evaluate()`,
    the same evaluation pathway this project's baselines also feed into, so every model's
    reported numbers stay directly comparable.
    """
    module.eval()
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(module.device)
            logits = module(x)
            preds = logits.argmax(dim=-1)
            all_true.append(y.cpu().numpy())
            all_pred.append(preds.cpu().numpy())
    return np.concatenate(all_true), np.concatenate(all_pred)
