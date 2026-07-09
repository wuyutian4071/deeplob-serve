import lightning as L  # noqa: N812 -- Lightning's own docs and ecosystem-wide convention
import pytest
import torch
from torch.utils.data import DataLoader

from deeplob.data.lob import NUM_FEATURES
from deeplob.models.transformer import LOBTransformer
from deeplob.training.dataset import LOBWindowDataset
from deeplob.training.lightning_module import LOBClassifier


def test_forward_produces_three_class_logits() -> None:
    model = LOBTransformer(window_size=100)
    x = torch.randn(4, 100, NUM_FEATURES)
    logits = model(x)
    assert logits.shape == (4, 3)


def test_forward_works_with_a_batch_size_of_one() -> None:
    model = LOBTransformer(window_size=100)
    x = torch.randn(1, 100, NUM_FEATURES)
    logits = model(x)
    assert logits.shape == (1, 3)


def test_forward_works_with_a_short_window() -> None:
    # Unlike the CNN-LSTM, a Transformer has no fixed minimum window size from an unpadded
    # conv stack -- window_size=1 should work cleanly.
    model = LOBTransformer(window_size=1)
    x = torch.randn(2, 1, NUM_FEATURES)
    logits = model(x)
    assert logits.shape == (2, 3)


def test_rejects_d_model_not_divisible_by_nhead() -> None:
    with pytest.raises(ValueError, match="must be divisible by"):
        LOBTransformer(window_size=100, d_model=10, nhead=4)


def test_forward_rejects_the_wrong_feature_dimension() -> None:
    model = LOBTransformer(window_size=100)
    x = torch.randn(2, 100, NUM_FEATURES + 1)
    with pytest.raises(ValueError, match="expected last dim"):
        model(x)


def test_every_parameter_receives_a_gradient_after_backward() -> None:
    model = LOBTransformer(window_size=100)
    x = torch.randn(4, 100, NUM_FEATURES)
    target = torch.randint(0, 3, (4,))

    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, target)
    # A known gap in torch's own stubs: Tensor.backward() itself is untyped.
    loss.backward()  # type: ignore[no-untyped-call]

    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} has a non-finite gradient"


def test_full_training_run_completes_end_to_end_on_synthetic_data() -> None:
    window_size = 30
    n_samples = 64
    features = torch.randn(n_samples, window_size, NUM_FEATURES).numpy()
    labels = torch.randint(0, 3, (n_samples,)).numpy()

    train_loader = DataLoader(LOBWindowDataset(features, labels), batch_size=8)

    module = LOBClassifier(LOBTransformer(window_size=window_size), learning_rate=1e-3)
    trainer = L.Trainer(
        accelerator="cpu",
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(module, train_loader)  # must not raise
