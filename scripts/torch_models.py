"""PyTorch model architectures for market prediction.

Defined at module level for pickle compatibility.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TabularMLP(nn.Module):
    """MLP with batch norm and dropout for tabular data."""

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResBlock(nn.Module):
    """Residual block with batch norm and dropout."""

    def __init__(self, dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class TabularResNet(nn.Module):
    """ResNet for tabular data — skip connections help gradient flow."""

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            ResBlock(64, 0.3),
            ResBlock(64, 0.2),
            ResBlock(64, 0.1),
        )
        self.head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.embed(x)))


class LargeResNet(nn.Module):
    """Larger ResNet with more blocks and wider layers."""

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ResBlock(256, 0.4),
            ResBlock(256, 0.3),
            ResBlock(256, 0.3),
            ResBlock(256, 0.2),
            ResBlock(256, 0.1),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.embed(x)))


class FeatureTokenizer(nn.Module):
    """Tokenizes each feature into a d-dimensional embedding."""

    def __init__(self, n_features: int, d_token: int = 64) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.randn(n_features, d_token))
        self.biases = nn.Parameter(torch.zeros(n_features, d_token))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features) -> (batch, n_features, d_token)
        return x.unsqueeze(-1) * self.weights.unsqueeze(0) + self.biases.unsqueeze(0)


class FTTransformer(nn.Module):
    """Feature Tokenizer + Transformer for tabular data.

    Each feature is tokenized into a learned embedding, then processed
    by transformer encoder layers. A [CLS] token aggregates information.
    """

    def __init__(
        self,
        n_features: int,
        d_token: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Sequential(
            nn.Linear(d_token, d_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)  # (batch, n_features, d_token)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        out = self.transformer(tokens)
        cls_out = self.norm(out[:, 0])
        return self.head(cls_out)
