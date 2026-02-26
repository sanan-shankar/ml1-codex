from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def compute_class_weights(targets: np.ndarray, n_classes: int, min_count: int = 1) -> np.ndarray:
    t = np.asarray(targets, dtype=np.int64)
    counts = np.bincount(t, minlength=n_classes).astype(np.float64)
    counts[counts < min_count] = float(min_count)
    inv = 1.0 / counts
    weights = inv * (n_classes / inv.sum())
    return weights.astype(np.float32)


def make_cross_entropy_loss(
    targets: np.ndarray,
    n_classes: int,
    *,
    weighted: bool,
    device: str = "cpu",
) -> tuple[nn.Module, np.ndarray | None]:
    if weighted:
        weights_np = compute_class_weights(targets, n_classes=n_classes)
        weight = torch.tensor(weights_np, dtype=torch.float32, device=device)
        return nn.CrossEntropyLoss(weight=weight), weights_np
    return nn.CrossEntropyLoss(), None
