from __future__ import annotations

import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """Compact EEGNet-style model for inputs shaped [B,C,T] or [B,1,C,T]."""

    def __init__(
        self,
        n_chans: int,
        n_times: int,
        n_classes: int,
        *,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        sep_kernel_length: int = 16,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        self.n_chans = int(n_chans)
        self.n_times = int(n_times)
        self.n_classes = int(n_classes)

        pad1 = kernel_length // 2
        pad2 = sep_kernel_length // 2

        self.features = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, kernel_length), padding=(0, pad1), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, kernel_size=(n_chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(dropout),
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, sep_kernel_length), padding=(0, pad2), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
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
