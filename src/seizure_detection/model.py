"""CNN-LSTM model definitions for seizure detection."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


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


class ResNet18LSTM(nn.Module):
    """Pretrained ResNet-18 feature extractor followed by a causal LSTM.

    Expected input shape:
      batch x seq_len x 3 x 224 x 224

    ResNet-18 produces one 512-dimensional feature vector per 4-second
    spectrogram image. The unidirectional LSTM then models the chronological
    3-window sequence and emits one seizure/non-seizure logit.
    """

    def __init__(
        self,
        lstm_hidden: int = 256,
        lstm_layers: int = 1,
        pretrained: bool = True,
        freeze_resnet: bool = False,
    ):
        super().__init__()

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        if freeze_resnet:
            for param in backbone.parameters():
                param.requires_grad = False

        self.resnet = backbone
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.classifier = nn.Linear(lstm_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one logit per sequence."""
        batch_size, seq_len, channels, height, width = x.shape
        x = x.reshape(batch_size * seq_len, channels, height, width)

        features = self.resnet(x)
        features = features.reshape(batch_size, seq_len, -1)

        lstm_out, _ = self.lstm(features)
        final_out = lstm_out[:, -1, :]
        return self.classifier(final_out).squeeze(1)


class FeatureLSTM(nn.Module):
    """Causal LSTM classifier for precomputed ResNet feature sequences.

    Expected input shape:
      batch x seq_len x 512
    """

    def __init__(
        self,
        feature_dim: int = 512,
        lstm_hidden: int = 256,
        lstm_layers: int = 1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.classifier = nn.Linear(lstm_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        final_out = lstm_out[:, -1, :]
        return self.classifier(final_out).squeeze(1)


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Return total and trainable parameter counts."""
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {"total_parameters": total, "trainable_parameters": trainable}
