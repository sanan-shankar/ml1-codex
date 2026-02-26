from __future__ import annotations

import torch
import torch.nn as nn


class _Square(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x


class _SafeLog(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.clamp(x, min=self.eps))


class ShallowFBCSPNet(nn.Module):
    """CPU-friendly shallow ConvNet inspired by ShallowFBCSPNet."""

    def __init__(
        self,
        n_chans: int,
        n_times: int,
        n_classes: int,
        *,
        n_filters_time: int = 24,
        filter_time_length: int = 25,
        pool_time_length: int = 75,
        pool_time_stride: int = 15,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_chans = int(n_chans)
        self.n_times = int(n_times)
        self.n_classes = int(n_classes)

        self.features = nn.Sequential(
            nn.Conv2d(1, n_filters_time, kernel_size=(1, filter_time_length), bias=True),
            nn.Conv2d(n_filters_time, n_filters_time, kernel_size=(n_chans, 1), bias=False),
            nn.BatchNorm2d(n_filters_time),
            _Square(),
            nn.AvgPool2d(kernel_size=(1, pool_time_length), stride=(1, pool_time_stride)),
            _SafeLog(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(self._feature_dim(), n_classes)

    def _feature_dim(self) -> int:
        with torch.no_grad():
            x = torch.zeros(1, 1, self.n_chans, self.n_times)
            y = self.features(x)
            return int(y.numel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4:
            raise ValueError(f"Expected input [B,C,T] or [B,1,C,T], got {tuple(x.shape)}")
        y = self.features(x)
        y = torch.flatten(y, start_dim=1)
        return self.classifier(y)
