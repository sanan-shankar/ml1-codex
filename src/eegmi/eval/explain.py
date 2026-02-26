from __future__ import annotations

from pathlib import Path

from eegmi.utils import set_matplotlib_env

set_matplotlib_env()

import matplotlib.pyplot as plt
import numpy as np
import torch

from eegmi.data.dataset import EEGEpochDataset


def compute_saliency(model: torch.nn.Module, x: torch.Tensor, target_idx: int | None = None) -> np.ndarray:
    model.eval()
    x = x.clone().detach().requires_grad_(True)
    logits = model(x)
    if target_idx is None:
        target_idx = int(torch.argmax(logits, dim=1).item())
    score = logits[0, target_idx]
    model.zero_grad(set_to_none=True)
    score.backward()
    grad = x.grad.detach().cpu().numpy()
    if grad.ndim == 4:
        grad = grad[:, 0]
    return np.abs(grad[0]).astype(np.float32)


def save_saliency_examples(
    model: torch.nn.Module,
    dataset: EEGEpochDataset,
    out_dir: str | Path,
    *,
    n_samples: int = 3,
    title_prefix: str = "saliency",
) -> list[str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved = []
    if len(dataset) == 0 or n_samples <= 0:
        return saved

    idxs = np.linspace(0, len(dataset) - 1, num=min(n_samples, len(dataset)), dtype=int)
    for rank, idx in enumerate(idxs.tolist(), start=1):
        x, y = dataset[idx]
        x_batch = x.unsqueeze(0)
        sal = compute_saliency(model, x_batch)
        x_np = x.detach().cpu().numpy()
        if x_np.ndim == 3:
            x_np = x_np[0]

        fig, axes = plt.subplots(2, 1, figsize=(8, 5), dpi=120, sharex=True)
        axes[0].imshow(x_np, aspect="auto", origin="lower", cmap="RdBu_r")
        axes[0].set_title(f"Input epoch (idx={idx}, target={int(y.item())})")
        axes[0].set_ylabel("Channel")
        axes[1].imshow(sal, aspect="auto", origin="lower", cmap="magma")
        axes[1].set_title("Gradient saliency |dlogit/dx|")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Channel")
        fig.suptitle(f"{title_prefix} #{rank}")
        fig.tight_layout()
        path = out / f"{title_prefix}_{rank:02d}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(path))
    return saved
