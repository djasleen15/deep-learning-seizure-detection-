"""Lightweight CNN-LSTM model used for the progress-report proof of concept."""

from __future__ import annotations

import torch
import torch.nn as nn


class SmallCNNLSTM(nn.Module):
    """CNN feature extractor followed by a single-layer LSTM classifier."""

    def __init__(self, feature_dim: int = 128, lstm_hidden: int = 64):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.cnn_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, feature_dim),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.classifier = nn.Linear(lstm_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for batch x seq_len x 1 x height x width input."""
        batch_size, seq_len, channels, height, width = x.shape
        x = x.view(batch_size * seq_len, channels, height, width)

        features = self.cnn(x)
        features = self.cnn_fc(features)
        features = features.view(batch_size, seq_len, -1)

        lstm_out, _ = self.lstm(features)
        final_out = lstm_out[:, -1, :]
        return self.classifier(final_out).squeeze(1)

