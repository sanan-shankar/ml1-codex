from __future__ import annotations

import torch
import torch.nn as nn


class _CompactEEGBranch(nn.Module):
    """EEGNet-style compact feature extractor returning flattened features."""

    def __init__(
        self,
        n_chans: int,
        n_samples: int,
        *,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        sep_kernel_length: int = 16,
        pool1: int = 4,
        pool2: int = 8,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        self.n_chans = int(n_chans)
        self.n_samples = int(n_samples)
        pad1 = max(0, int(kernel_length) // 2)
        pad2 = max(0, int(sep_kernel_length) // 2)
        self.features = nn.Sequential(
            nn.Conv2d(1, int(F1), kernel_size=(1, int(kernel_length)), padding=(0, pad1), bias=False),
            nn.BatchNorm2d(int(F1)),
            nn.Conv2d(int(F1), int(F1) * int(D), kernel_size=(self.n_chans, 1), groups=int(F1), bias=False),
            nn.BatchNorm2d(int(F1) * int(D)),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, int(pool1)), stride=(1, int(pool1))),
            nn.Dropout(float(dropout)),
            nn.Conv2d(
                int(F1) * int(D),
                int(F1) * int(D),
                kernel_size=(1, int(sep_kernel_length)),
                padding=(0, pad2),
                groups=int(F1) * int(D),
                bias=False,
            ),
            nn.Conv2d(int(F1) * int(D), int(F2), kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(int(F2)),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, int(pool2)), stride=(1, int(pool2))),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4:
            raise ValueError(f"Expected branch input [B,C,T] or [B,1,C,T], got {tuple(x.shape)}")
        y = self.features(x)
        return torch.flatten(y, start_dim=1)

    def output_dim(self) -> int:
        with torch.no_grad():
            x = torch.zeros(1, 1, self.n_chans, self.n_samples)
            y = self.features(x)
            return int(y.numel())


class EEGFusionNetLite(nn.Module):
    """Compact two-branch model: raw-time branch + PSD branch fused late."""

    def __init__(
        self,
        n_chans: int,
        n_times: int,
        n_classes: int,
        *,
        dropout: float = 0.35,
        fusion_hidden: int = 96,
        psd_drop_dc: bool = True,
        psd_n_bins: int | None = 128,
        psd_log: bool = True,
        psd_relative_power: bool = True,
        psd_channel_zscore: bool = True,
        time_feat_init_scale: float = 1.0,
        spec_feat_init_scale: float = 0.15,
        use_branch_logit_residual: bool = True,
        time_logit_residual_init: float = 1.0,
        spec_logit_residual_init: float = 0.05,
        time_branch: dict | None = None,
        spectral_branch: dict | None = None,
    ) -> None:
        super().__init__()
        self.n_chans = int(n_chans)
        self.n_times = int(n_times)
        self.n_classes = int(n_classes)
        self.psd_drop_dc = bool(psd_drop_dc)
        self.psd_log = bool(psd_log)
        self.psd_relative_power = bool(psd_relative_power)
        self.psd_channel_zscore = bool(psd_channel_zscore)
        self.use_branch_logit_residual = bool(use_branch_logit_residual)
        self.eps = 1e-6

        full_bins = (self.n_times // 2) + 1
        start_bin = 1 if self.psd_drop_dc else 0
        available = max(1, full_bins - start_bin)
        keep_bins = available if psd_n_bins is None else max(1, min(int(psd_n_bins), available))
        self.psd_start_bin = start_bin
        self.psd_keep_bins = keep_bins

        tb = dict(time_branch or {})
        sb = dict(spectral_branch or {})
        # Spectral branch operates on PSD bins; shorter kernels/pools are more appropriate.
        sb.setdefault("kernel_length", 15)
        sb.setdefault("sep_kernel_length", 7)
        sb.setdefault("pool1", 2)
        sb.setdefault("pool2", 4)

        self.time_branch = _CompactEEGBranch(
            n_chans=self.n_chans,
            n_samples=self.n_times,
            dropout=float(dropout),
            **tb,
        )
        self.spectral_branch = _CompactEEGBranch(
            n_chans=self.n_chans,
            n_samples=self.psd_keep_bins,
            dropout=float(dropout),
            **sb,
        )

        time_dim = self.time_branch.output_dim()
        spec_dim = self.spectral_branch.output_dim()
        fusion_dim = time_dim + spec_dim
        self.time_feat_scale = nn.Parameter(torch.tensor(float(time_feat_init_scale), dtype=torch.float32))
        self.spec_feat_scale = nn.Parameter(torch.tensor(float(spec_feat_init_scale), dtype=torch.float32))
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        if int(fusion_hidden) > 0:
            self.classifier = nn.Sequential(
                nn.Linear(fusion_dim, int(fusion_hidden)),
                nn.ELU(inplace=True),
                nn.Dropout(float(dropout)),
                nn.Linear(int(fusion_hidden), self.n_classes),
            )
        else:
            self.classifier = nn.Linear(fusion_dim, self.n_classes)
        if self.use_branch_logit_residual:
            self.time_head = nn.Linear(time_dim, self.n_classes)
            self.spec_head = nn.Linear(spec_dim, self.n_classes)
            self.time_logit_scale = nn.Parameter(torch.tensor(float(time_logit_residual_init), dtype=torch.float32))
            self.spec_logit_scale = nn.Parameter(torch.tensor(float(spec_logit_residual_init), dtype=torch.float32))
        else:
            self.time_head = None
            self.spec_head = None
            self.time_logit_scale = None
            self.spec_logit_scale = None

    def _compute_psd(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,T]
        x = x - x.mean(dim=-1, keepdim=True)
        spec = torch.fft.rfft(x, dim=-1)
        power = spec.real.square() + spec.imag.square()
        power = power[..., self.psd_start_bin : self.psd_start_bin + self.psd_keep_bins]
        if self.psd_relative_power:
            power = power / power.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        if self.psd_log:
            power = torch.log(power.clamp_min(self.eps))
        if self.psd_channel_zscore:
            mu = power.mean(dim=-1, keepdim=True)
            sd = power.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-5)
            power = (power - mu) / sd
        return power

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x[:, 0]
        if x.ndim != 3:
            raise ValueError(f"Expected input [B,C,T] or [B,1,C,T], got {tuple(x.shape)}")

        x = x.float()
        time_feat = self.time_branch(x)
        psd = self._compute_psd(x)
        spec_feat = self.spectral_branch(psd)
        fused = torch.cat([time_feat * self.time_feat_scale, spec_feat * self.spec_feat_scale], dim=1)
        fused = self.fusion_norm(fused)
        logits = self.classifier(fused)
        if self.use_branch_logit_residual and self.time_head is not None and self.spec_head is not None:
            logits = (
                logits
                + (self.time_logit_scale * self.time_head(time_feat))
                + (self.spec_logit_scale * self.spec_head(spec_feat))
            )
        return logits
