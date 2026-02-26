from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class _AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim: int, attn_dim: int = 64) -> None:
        super().__init__()
        self.proj = nn.Linear(int(hidden_dim), int(attn_dim))
        self.score = nn.Linear(int(attn_dim), 1, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, L, H]
        e = torch.tanh(self.proj(x))
        logits = self.score(e).squeeze(-1)  # [B, L]
        w = torch.softmax(logits, dim=-1)
        ctx = torch.sum(x * w.unsqueeze(-1), dim=1)
        return ctx, w


class PaperCnnLstmAttention(nn.Module):
    """Paper-inspired CNN-LSTM-Attention over ROI-pair bandpower features.

    Input:
      x: [B, C, T] raw EEG epochs (or [B,1,C,T])

    Processing:
      1) Compute FFT power per channel
      2) Aggregate into canonical bands
      3) Build per-ROI-pair features (e.g., left-right diff and/or mean)
      4) Conv1D over ROI-pair sequence
      5) LSTM
      6) Additive attention
      7) Classifier

    This approximates the paper architecture while staying compatible with the
    existing training engine and strict subject-wise evaluation pipeline.
    """

    def __init__(
        self,
        n_chans: int,
        n_times: int,
        n_classes: int,
        *,
        sfreq: float = 160.0,
        roi_pairs: Sequence[Sequence[int]] | None = None,
        band_defs: Sequence[Sequence[float]] | None = None,
        use_relative_power: bool = True,
        log_power: bool = True,
        channel_zscore: bool = False,
        pair_feature_mode: str = "diff_mean",
        conv_channels: int = 32,
        conv_kernel: int = 3,
        conv_layers: int = 2,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        bidirectional: bool = True,
        attn_dim: int = 64,
        fc_hidden: int = 64,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        self.n_chans = int(n_chans)
        self.n_times = int(n_times)
        self.n_classes = int(n_classes)
        self.sfreq = float(sfreq)
        self.use_relative_power = bool(use_relative_power)
        self.log_power = bool(log_power)
        self.channel_zscore = bool(channel_zscore)
        self.pair_feature_mode = str(pair_feature_mode).lower()
        self.eps = 1e-6

        if band_defs is None:
            # Paper-like PSD bands (theta, alpha, beta, gamma)
            band_defs = [(4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0)]
        self.band_defs = [(float(lo), float(hi)) for lo, hi in band_defs]
        if not self.band_defs:
            raise ValueError("band_defs must not be empty")

        if roi_pairs is None:
            # Default assumes the project 23-channel order; caller should pass
            # explicit roi_pairs for custom channel layouts.
            roi_pairs = [
                [2, 4],    # FC1, FC2
                [1, 5],    # FC3, FC4
                [9, 11],   # C1, C2
                [8, 12],   # C3, C4
                [16, 18],  # CP1, CP2
                [15, 19],  # CP3, CP4
            ]
        self.roi_pairs = [tuple(int(i) for i in p) for p in roi_pairs]
        if any(len(p) != 2 for p in self.roi_pairs):
            raise ValueError("Each roi pair must contain exactly 2 channel indices")
        for a, b in self.roi_pairs:
            if a < 0 or a >= self.n_chans or b < 0 or b >= self.n_chans:
                raise ValueError(f"ROI pair {(a, b)} out of bounds for n_chans={self.n_chans}")

        freqs = torch.fft.rfftfreq(self.n_times, d=1.0 / self.sfreq)
        masks = []
        for lo, hi in self.band_defs:
            m = (freqs >= lo) & (freqs < hi)
            if not torch.any(m):
                raise ValueError(f"Band [{lo}, {hi}) has no FFT bins for n_times={self.n_times}, sfreq={self.sfreq}")
            masks.append(m)
        self.register_buffer("_freqs", freqs, persistent=False)
        self.register_buffer("_band_masks", torch.stack(masks, dim=0), persistent=False)  # [BANDS, F]

        pair_feat_dim = self._pair_feature_dim()
        seq_len = len(self.roi_pairs)
        if seq_len < 1:
            raise ValueError("Need at least one ROI pair")

        conv_blocks: list[nn.Module] = []
        in_ch = pair_feat_dim
        out_ch = int(conv_channels)
        pad = max(0, int(conv_kernel) // 2)
        n_conv = max(1, int(conv_layers))
        for _ in range(n_conv):
            conv_blocks.extend(
                [
                    nn.Conv1d(in_ch, out_ch, kernel_size=int(conv_kernel), padding=pad, bias=False),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Dropout(float(dropout)),
                ]
            )
            in_ch = out_ch
        self.conv = nn.Sequential(*conv_blocks)

        self.lstm = nn.LSTM(
            input_size=out_ch,
            hidden_size=int(lstm_hidden),
            num_layers=max(1, int(lstm_layers)),
            batch_first=True,
            bidirectional=bool(bidirectional),
            dropout=float(dropout) if int(lstm_layers) > 1 else 0.0,
        )
        lstm_out_dim = int(lstm_hidden) * (2 if bidirectional else 1)
        self.attn = _AdditiveAttention(lstm_out_dim, attn_dim=int(attn_dim))
        if int(fc_hidden) > 0:
            self.classifier = nn.Sequential(
                nn.Linear(lstm_out_dim, int(fc_hidden)),
                nn.ReLU(inplace=True),
                nn.Dropout(float(dropout)),
                nn.Linear(int(fc_hidden), self.n_classes),
            )
        else:
            self.classifier = nn.Linear(lstm_out_dim, self.n_classes)

    def _pair_feature_dim(self) -> int:
        nb = len(self.band_defs)
        mode = self.pair_feature_mode
        if mode in {"diff", "lr_diff"}:
            return nb
        if mode in {"concat_lr", "lr"}:
            return 2 * nb
        if mode in {"diff_mean", "delta_mean", "diff+mean"}:
            return 2 * nb
        if mode in {"diff_lr_mean", "rich"}:
            return 4 * nb
        raise ValueError(f"Unsupported pair_feature_mode: {self.pair_feature_mode}")

    def _bandpower(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,T] -> [B,C,BANDS]
        x = x.float()
        x = x - x.mean(dim=-1, keepdim=True)
        spec = torch.fft.rfft(x, dim=-1)
        power = spec.real.square() + spec.imag.square()  # [B,C,F]
        if self.use_relative_power:
            power = power / power.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        if self.log_power:
            power = torch.log(power.clamp_min(self.eps))

        band_masks = self._band_masks.to(power.device)  # [BANDS,F]
        band_feats = []
        for i in range(band_masks.shape[0]):
            m = band_masks[i]
            bp = power[..., m].mean(dim=-1)  # [B,C]
            band_feats.append(bp)
        bands = torch.stack(band_feats, dim=-1)  # [B,C,BANDS]
        if self.channel_zscore:
            mu = bands.mean(dim=1, keepdim=True)
            sd = bands.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5)
            bands = (bands - mu) / sd
        return bands

    def _roi_pair_features(self, bandp: torch.Tensor) -> torch.Tensor:
        # bandp: [B,C,BANDS] -> [B, FEAT, PAIRS]
        pair_feats = []
        for a, b in self.roi_pairs:
            left = bandp[:, a, :]   # [B,BANDS]
            right = bandp[:, b, :]  # [B,BANDS]
            diff = left - right
            mean = 0.5 * (left + right)
            mode = self.pair_feature_mode
            if mode in {"diff", "lr_diff"}:
                feat = diff
            elif mode in {"concat_lr", "lr"}:
                feat = torch.cat([left, right], dim=-1)
            elif mode in {"diff_mean", "delta_mean", "diff+mean"}:
                feat = torch.cat([diff, mean], dim=-1)
            elif mode in {"diff_lr_mean", "rich"}:
                feat = torch.cat([diff, left, right, mean], dim=-1)
            else:
                raise RuntimeError(f"Unsupported pair_feature_mode: {mode}")
            pair_feats.append(feat)
        x_seq = torch.stack(pair_feats, dim=1)   # [B, PAIRS, FEAT]
        return x_seq.transpose(1, 2).contiguous()  # [B, FEAT, PAIRS]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            if x.shape[1] != 1:
                raise ValueError(f"Expected [B,C,T] or [B,1,C,T], got {tuple(x.shape)}")
            x = x[:, 0]
        if x.ndim != 3:
            raise ValueError(f"Expected [B,C,T] or [B,1,C,T], got {tuple(x.shape)}")
        if int(x.shape[1]) != self.n_chans:
            raise ValueError(f"Channel mismatch: got {int(x.shape[1])}, expected {self.n_chans}")

        bandp = self._bandpower(x)              # [B,C,BANDS]
        feats = self._roi_pair_features(bandp)  # [B,FEAT,PAIRS]
        y = self.conv(feats)                    # [B,F',PAIRS]
        y = y.transpose(1, 2).contiguous()      # [B,PAIRS,F']
        y, _ = self.lstm(y)                     # [B,PAIRS,H]
        ctx, _ = self.attn(y)                   # [B,H]
        logits = self.classifier(ctx)           # [B,K]
        return logits

