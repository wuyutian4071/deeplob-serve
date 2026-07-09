"""Hydra config-driven training entry point. Run via:

    uv run python -m deeplob.training.train

Ties together every M2-M4 piece: the synthetic LOB generator and labeling/windowing (M2),
the temporal split and evaluation harness (M3), and any model `hydra.utils.instantiate()` can
construct from `configs/config.yaml`'s `model` group (this project's own CNN-LSTM -- task
#95 -- or M5's future Transformer, swappable via config, not a code change) -- through the
generic `LOBClassifier` training loop (M4), with local MLflow tracking.
"""

import hydra
import lightning as L  # noqa: N812 -- Lightning's own docs and ecosystem-wide convention
from hydra.utils import instantiate
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from deeplob.data.labeling import compute_labels
from deeplob.data.lob import mid_price
from deeplob.data.synthetic import generate_synthetic_lob
from deeplob.data.windowing import make_windows
from deeplob.evaluation.metrics import evaluate
from deeplob.evaluation.splits import temporal_train_val_test_split
from deeplob.training.dataset import LOBWindowDataset
from deeplob.training.device import resolve_accelerator, resolve_precision
from deeplob.training.lightning_module import LOBClassifier, collect_predictions
from deeplob.training.seeding import set_seed


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)

    # All CI/tests (and this default config) run against the synthetic generator, never the
    # real FI-2010 dataset -- see deeplob.data.fi2010's own docstring for why. Swapping in
    # real data means replacing this one block with deeplob.data.fi2010.load_fi2010(), not a
    # different pipeline: labeling/windowing/splitting/training are identical either way.
    features = generate_synthetic_lob(num_snapshots=cfg.data.num_synthetic_snapshots, seed=cfg.seed)
    labels = compute_labels(mid_price(features), horizon=cfg.data.horizon, alpha=cfg.data.alpha)
    X, y = make_windows(features, labels, window_size=cfg.data.window_size)

    split = temporal_train_val_test_split(
        X, y, train_frac=cfg.data.train_frac, val_frac=cfg.data.val_frac
    )

    train_loader = DataLoader(
        LOBWindowDataset(split.X_train, split.y_train),
        batch_size=cfg.training.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        LOBWindowDataset(split.X_val, split.y_val), batch_size=cfg.training.batch_size
    )
    test_loader = DataLoader(
        LOBWindowDataset(split.X_test, split.y_test), batch_size=cfg.training.batch_size
    )

    model = instantiate(cfg.model)
    module = LOBClassifier(model, learning_rate=cfg.training.learning_rate)

    accelerator = resolve_accelerator()
    precision = resolve_precision(accelerator)

    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.mlflow.experiment_name, tracking_uri=cfg.mlflow.tracking_uri
    )

    trainer = L.Trainer(
        accelerator=accelerator,
        precision=precision,
        max_epochs=cfg.training.max_epochs,
        logger=mlflow_logger,
        # Lightning's default auto-checkpointing saves to a path derived from the logger --
        # for MLFlowLogger, a bare "<experiment_id>/<run_id>/checkpoints/" directory at the
        # repo root, entirely separate from (and not controlled by) MLflow's own
        # artifact_location config -- discovered only by actually running this script and
        # inspecting what it created. Checkpoint persistence isn't a stated M4 requirement
        # yet; disabled here rather than fighting its exact, awkward-to-gitignore save path
        # for a feature this milestone doesn't need.
        enable_checkpointing=False,
    )
    trainer.fit(module, train_loader, val_loader)

    y_true, y_pred = collect_predictions(module, test_loader)
    report = evaluate(y_true, y_pred)
    print(report.summary())


if __name__ == "__main__":
    main()
