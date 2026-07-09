"""A DeepLOB-style CNN-LSTM (Zhang, Zohren, Roberts, "DeepLOB: Deep Convolutional Neural
Networks for Limit Order Books", 2018) for 3-class mid-price-movement prediction from
windowed LOB sequences.

This follows the paper's key structural ideas -- convolutional feature extraction treating
the `[window_size, NUM_FEATURES]` input as a 2D (time x feature) "image" rather than raw
features flattened or treated as independent channels; a first layer that specifically pairs
each price level's (price, volume) columns via a `(1,2)`-stride-2 kernel; further conv blocks
that progressively merge price levels; an Inception-style multi-branch block for multi-scale
temporal patterns; then an LSTM over the resulting compressed sequence. It is **"DeepLOB-
style", not a byte-exact reproduction** of the paper's specific channel counts and kernel
sizes -- those weren't independently re-verified against the original paper's own tables
during development, and stating that plainly here is more honest than implying a precision
this implementation doesn't actually have.
"""

import torch
from torch import nn

from deeplob.data.lob import NUM_FEATURES

_NUM_CLASSES = 3
# 6 unpadded (4,1) convolutions in the stack below (2 each in conv1/conv2/conv3), each
# reducing the time axis by 3 (kernel_size=4, stride=1, no padding: output = input - 3) --
# 6 * 3 = 18 total. window_size must comfortably exceed this for the LSTM to have any time
# steps left to process.
_TIME_AXIS_REDUCTION = 18


class DeepLOBCNNLSTM(nn.Module):
    """Consumes `[batch, window_size, NUM_FEATURES]` LOB sequences, outputs `[batch, 3]`
    logits (DOWN/STATIONARY/UP, matching `deeplob.data.labeling.Label`'s ordering).
    """

    def __init__(
        self, window_size: int = 100, lstm_hidden_size: int = 64, conv_channels: int = 32
    ) -> None:
        super().__init__()
        if window_size <= _TIME_AXIS_REDUCTION:
            raise ValueError(
                f"window_size must be > {_TIME_AXIS_REDUCTION} (the conv stack's cumulative "
                f"time-axis reduction), got {window_size}"
            )

        c = conv_channels

        # Block 1: pairs (price, volume) at each of the 10 levels via a (1,2)-stride-2
        # kernel over the feature axis (NUM_FEATURES=40 -> 20), then two (4,1) convs over
        # time.
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(c),
            nn.Conv2d(c, c, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(c),
            nn.Conv2d(c, c, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(c),
        )

        # Block 2: merges adjacent price levels (20 -> 10) via another (1,2)-stride-2 kernel.
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(c),
            nn.Conv2d(c, c, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(c),
            nn.Conv2d(c, c, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(c),
        )

        # Block 3: a (1,10) kernel exactly spans the now-10-wide feature axis, collapsing it
        # to 1 -- this specifically assumes NUM_FEATURES=40 fed through blocks 1-2's /2/2
        # reduction (40 -> 20 -> 10), not a generic "any feature count" design.
        self.conv3 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=(1, 10)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(c),
            nn.Conv2d(c, c, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(c),
            nn.Conv2d(c, c, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(c),
        )

        # Inception-style multi-scale temporal block: three parallel branches with different
        # time-axis receptive fields, each padded to preserve the time dimension so they
        # concatenate cleanly.
        self.inception1 = nn.Sequential(
            nn.Conv2d(c, 2 * c, kernel_size=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(2 * c),
            nn.Conv2d(2 * c, 2 * c, kernel_size=(3, 1), padding=(1, 0)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(2 * c),
        )
        self.inception2 = nn.Sequential(
            nn.Conv2d(c, 2 * c, kernel_size=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(2 * c),
            nn.Conv2d(2 * c, 2 * c, kernel_size=(5, 1), padding=(2, 0)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(2 * c),
        )
        self.inception3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(c, 2 * c, kernel_size=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(2 * c),
        )

        self.lstm = nn.LSTM(input_size=2 * c * 3, hidden_size=lstm_hidden_size, batch_first=True)
        self.classifier = nn.Linear(lstm_hidden_size, _NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != NUM_FEATURES:
            raise ValueError(f"expected last dim {NUM_FEATURES}, got {x.shape[-1]}")

        h = x.unsqueeze(1)  # [batch, 1, window, NUM_FEATURES] -- a single input "channel"
        h = self.conv1(h)
        h = self.conv2(h)
        h = self.conv3(h)  # [batch, conv_channels, window', 1] -- feature axis fully collapsed

        branch1 = self.inception1(h)
        branch2 = self.inception2(h)
        branch3 = self.inception3(h)
        h = torch.cat([branch1, branch2, branch3], dim=1)  # [batch, 6*conv_channels, window', 1]

        h = h.squeeze(-1)  # [batch, channels, window']
        h = h.permute(0, 2, 1)  # [batch, window', channels] -- LSTM wants time-major

        lstm_out, _ = self.lstm(h)
        last_hidden = lstm_out[:, -1, :]  # final time step's hidden state
        logits: torch.Tensor = self.classifier(last_hidden)
        return logits
