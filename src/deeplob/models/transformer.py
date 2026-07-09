"""A lightweight Transformer encoder for 3-class mid-price-movement prediction from windowed
LOB sequences -- M5's alternative to the CNN-LSTM (M4), reusing the exact same generic
training loop (`deeplob.training.lightning_module.LOBClassifier`) and evaluation harness
(`deeplob.evaluation.metrics.evaluate`) without either needing to change, which is the whole
point of M4's model-agnostic design.
"""

import math

import torch
from torch import nn

from deeplob.data.lob import NUM_FEATURES

_NUM_CLASSES = 3


class _PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., "Attention Is All You Need",
    2017) -- fixed, not learned, since this project's window sizes are small and fixed per
    run; a learned embedding table would just be extra parameters for no real benefit at this
    scale.
    """

    pe: torch.Tensor

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = x + self.pe[: x.shape[1]]
        return result


class LOBTransformer(nn.Module):
    """Consumes `[batch, window_size, NUM_FEATURES]` LOB sequences, outputs `[batch, 3]`
    logits. A linear projection of the 40 raw per-timestep features into a `d_model`
    embedding, sinusoidal positional encoding, a stack of standard `TransformerEncoder`
    layers, then mean-pooling over time into the classifier head -- mean pooling, not just
    the final time step (unlike the CNN-LSTM's LSTM-final-hidden-state choice), since
    self-attention already lets every time step attend to every other one, so a Transformer
    doesn't have the same "only the final step has accumulated everything" structure an LSTM
    does.
    """

    def __init__(
        self,
        window_size: int = 100,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.input_projection = nn.Linear(NUM_FEATURES, d_model)
        self.positional_encoding = _PositionalEncoding(d_model, max_len=window_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, _NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != NUM_FEATURES:
            raise ValueError(f"expected last dim {NUM_FEATURES}, got {x.shape[-1]}")

        h = self.input_projection(x)  # [batch, window, d_model]
        h = self.positional_encoding(h)
        h = self.encoder(h)  # [batch, window, d_model]
        pooled = h.mean(dim=1)  # [batch, d_model]
        logits: torch.Tensor = self.classifier(pooled)
        return logits
