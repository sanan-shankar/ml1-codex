from __future__ import annotations

import torch
import torch.nn as nn


class _Square(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x


class _SafeLog(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.clamp(x, min=self.eps))


class _VarPool1d(nn.Module):
    """Variance pooling over time using sliding windows."""

    def __init__(self, pool_length: int = 32, stride: int = 16, eps: float = 1e-6) -> None:
        super().__init__()
        self.pool_length = int(pool_length)
        self.stride = int(stride)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F, T]
        if x.ndim != 3:
            raise ValueError(f"VarPool expects [B,F,T], got {tuple(x.shape)}")
        T = int(x.shape[-1])
        if self.pool_length <= 1 or self.pool_length > T:
            var = x.var(dim=-1, unbiased=False, keepdim=True)
            return var
        chunks = x.unfold(dimension=-1, size=self.pool_length, step=max(self.stride, 1))  # [B,F,W,L]
        var = chunks.var(dim=-1, unbiased=False)
        return var


class FBCNetLite(nn.Module):
    """Compact FBCNet-style model for filter-bank EEG inputs.

    Expected input shape is [B, C_total, T], where channels are stacked as:
      [band1_chans..., band2_chans..., ...].
    The model reshapes this into [B, n_bands, chans_per_band, T], learns per-band
    spatial filters, and uses variance pooling over time.
    """

    def __init__(
        self,
        n_chans: int,
        n_times: int,
        n_classes: int,
        *,
        n_bands: int,
        spatial_filters_per_band: int = 8,
        var_pool_length: int = 32,
        var_pool_stride: int = 16,
        dropout: float = 0.5,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.n_chans_total = int(n_chans)
        self.n_times = int(n_times)
        self.n_classes = int(n_classes)
        self.n_bands = int(n_bands)
        if self.n_bands <= 0:
            raise ValueError(f"n_bands must be positive, got {n_bands}")
        if self.n_chans_total % self.n_bands != 0:
            raise ValueError(
                f"Total channels ({self.n_chans_total}) not divisible by n_bands ({self.n_bands})"
            )
        self.chans_per_band = self.n_chans_total // self.n_bands
        self.spatial_filters_per_band = int(spatial_filters_per_band)

        out_feats = self.n_bands * self.spatial_filters_per_band
        self.spatial = nn.Conv2d(
            in_channels=self.n_bands,
            out_channels=out_feats,
            kernel_size=(self.chans_per_band, 1),
            groups=self.n_bands,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_feats)
        self.act = nn.ELU(inplace=True)
        self.var_pool = _VarPool1d(pool_length=var_pool_length, stride=var_pool_stride, eps=eps)
        self.log = _SafeLog(eps=eps)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self._feature_dim(), self.n_classes)

    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            if x.shape[1] != 1:
                raise ValueError(f"FBCNet expects [B,C,T] or [B,1,C,T], got {tuple(x.shape)}")
            x = x[:, 0]
        if x.ndim != 3:
            raise ValueError(f"FBCNet expects [B,C,T] or [B,1,C,T], got {tuple(x.shape)}")
        if int(x.shape[1]) != self.n_chans_total:
            raise ValueError(f"Channel mismatch: got {int(x.shape[1])}, expected {self.n_chans_total}")
        if int(x.shape[2]) != self.n_times:
            raise ValueError(f"Time mismatch: got {int(x.shape[2])}, expected {self.n_times}")
        return x.view(x.shape[0], self.n_bands, self.chans_per_band, self.n_times)

    def _feature_dim(self) -> int:
        with torch.no_grad():
            x = torch.zeros(1, self.n_chans_total, self.n_times)
            y = self.forward_features(x)
            return int(y.numel())

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self._reshape_input(x)  # [B, bands, Cb, T]
        y = self.spatial(x)         # [B, bands*Fs, 1, T]
        y = self.bn(y)
        y = self.act(y)
        y = y.squeeze(2)            # [B, bands*Fs, T]
        y = self.var_pool(y)        # [B, bands*Fs, W]
        y = self.log(y)
        y = self.dropout(y)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.forward_features(x)
        y = torch.flatten(y, start_dim=1)
        return self.classifier(y)

