from __future__ import annotations

import torch
import torch.nn as nn


class _SpatialDropout1d(nn.Module):
    """Keras-like SpatialDropout1D over temporal sequences represented as [B,C,T]."""

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.drop = nn.Dropout2d(p=float(p))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"SpatialDropout1d expects [B,C,T], got {tuple(x.shape)}")
        # Drop whole channels/features across all timesteps.
        y = self.drop(x.unsqueeze(-1))
        return y.squeeze(-1)


class PaperCnnGru1D(nn.Module):
    """Paper-inspired 1D CNN-GRU / CNN-BiGRU model (Sensors 2025, PMID 40096214).

    Adaptation notes:
    - Supports arbitrary channel count (paper uses 2-channel symmetric pair inputs and
      also reports SMA combinations). We directly model the selected channel set.
    - Preserves the paper's shallow Conv1D -> GRU/BiGRU design, dropout, and kernel sizes.
    - Optional attention and residual pooling can be enabled for improvements.
    """

    def __init__(
        self,
        n_chans: int,
        n_times: int,
        n_classes: int,
        *,
        conv_filters: int = 32,
        conv1_kernel: int = 20,
        conv2_kernel: int = 20,
        conv3_kernel: int = 6,
        conv4_kernel: int = 6,
        gru_hidden: int = 128,
        bidirectional: bool = False,
        num_gru_layers: int = 1,
        spatial_dropout: float = 0.5,
        dropout: float = 0.35,
        use_attention: bool = False,
        attn_hidden: int = 64,
        use_global_pool_residual: bool = True,
        fc_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.n_chans = int(n_chans)
        self.n_times = int(n_times)
        self.n_classes = int(n_classes)
        self.use_attention = bool(use_attention)
        self.use_global_pool_residual = bool(use_global_pool_residual)

        F = int(conv_filters)
        k1 = int(conv1_kernel)
        k2 = int(conv2_kernel)
        k3 = int(conv3_kernel)
        k4 = int(conv4_kernel)
        p1 = k1 // 2

        self.conv1 = nn.Conv1d(self.n_chans, F, kernel_size=k1, stride=1, padding=p1, bias=False)
        self.bn1 = nn.BatchNorm1d(F)
        self.conv2 = nn.Conv1d(F, F, kernel_size=k2, stride=1, padding=0, bias=False)  # VALID
        self.bn2 = nn.BatchNorm1d(F)
        self.sdrop1 = _SpatialDropout1d(float(spatial_dropout))
        self.conv3 = nn.Conv1d(F, F, kernel_size=k3, stride=1, padding=0, bias=False)  # VALID
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv1d(F, F, kernel_size=k4, stride=1, padding=0, bias=False)  # VALID
        self.sdrop2 = _SpatialDropout1d(float(spatial_dropout))
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(float(dropout))

        self.gru = nn.GRU(
            input_size=F,
            hidden_size=int(gru_hidden),
            num_layers=max(1, int(num_gru_layers)),
            batch_first=True,
            dropout=float(dropout) if int(num_gru_layers) > 1 else 0.0,
            bidirectional=bool(bidirectional),
        )
        gru_out_dim = int(gru_hidden) * (2 if bidirectional else 1)

        if self.use_attention:
            self.attn_proj = nn.Linear(gru_out_dim, int(attn_hidden))
            self.attn_score = nn.Linear(int(attn_hidden), 1, bias=False)
        else:
            self.attn_proj = None
            self.attn_score = None

        head_in = gru_out_dim
        if self.use_global_pool_residual:
            head_in += F  # residual pooled conv feature

        if int(fc_hidden) > 0:
            self.fc = nn.Sequential(
                nn.Linear(head_in, int(fc_hidden)),
                nn.ReLU(inplace=True),
                nn.Dropout(float(dropout)),
                nn.Linear(int(fc_hidden), self.n_classes),
            )
        else:
            self.fc = nn.Linear(head_in, self.n_classes)

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B,C,T] or [B,1,C,T]
        if x.ndim == 4:
            if x.shape[1] != 1:
                raise ValueError(f"Expected [B,C,T] or [B,1,C,T], got {tuple(x.shape)}")
            x = x[:, 0]
        if x.ndim != 3:
            raise ValueError(f"Expected [B,C,T] or [B,1,C,T], got {tuple(x.shape)}")
        if int(x.shape[1]) != self.n_chans:
            raise ValueError(f"Channel mismatch: got {int(x.shape[1])}, expected {self.n_chans}")

        y = self.act(self.bn1(self.conv1(x)))
        y = self.act(self.bn2(self.conv2(y)))
        y = self.sdrop1(y)
        y = self.act(self.conv3(y))
        y = self.pool(y)
        y = self.act(self.conv4(y))
        y = self.sdrop2(y)
        y = self.dropout(y)

        # GRU over time
        seq = y.transpose(1, 2).contiguous()  # [B,T,F]
        seq_out, _ = self.gru(seq)            # [B,T,H]

        if self.use_attention:
            e = torch.tanh(self.attn_proj(seq_out))
            a = torch.softmax(self.attn_score(e).squeeze(-1), dim=-1)  # [B,T]
            ctx = torch.sum(seq_out * a.unsqueeze(-1), dim=1)
        else:
            ctx = seq_out[:, -1, :]  # paper-like final-state readout

        residual = y.mean(dim=-1) if self.use_global_pool_residual else torch.empty(0, device=y.device)
        return ctx, residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx, residual = self.forward_features(x)
        if self.use_global_pool_residual:
            feat = torch.cat([ctx, residual], dim=1)
        else:
            feat = ctx
        return self.fc(feat)

