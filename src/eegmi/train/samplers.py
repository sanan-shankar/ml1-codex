from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


class NoOpSampler:
    pass


def make_balanced_sampler(targets: np.ndarray) -> WeightedRandomSampler:
    t = np.asarray(targets, dtype=np.int64)
    counts = np.bincount(t)
    counts[counts == 0] = 1
    weights = 1.0 / counts[t]
    weights = torch.as_tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(weights=weights, num_samples=len(t), replacement=True)
