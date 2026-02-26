from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from eegmi.data.dataset import EEGEpochDataset


def _materialize_dataset_view(dataset: EEGEpochDataset) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    idx = dataset.indices
    X = dataset.X_full[idx]
    if dataset.channel_indices is not None:
        X = X[:, dataset.channel_slice] if dataset.channel_slice is not None else X[:, dataset.channel_indices]
    if dataset.time_indices is not None:
        X = X[:, :, dataset.time_slice] if dataset.time_slice is not None else X[:, :, dataset.time_indices]
    X = np.ascontiguousarray(X.astype(np.float32, copy=False))
    y = dataset.y.astype(np.int64, copy=False)
    meta = dataset.meta.reset_index(drop=True).copy()
    return X, y, meta


def _target_counts_from_ratio(counts: np.ndarray, ratio: float) -> np.ndarray:
    max_count = int(counts.max()) if counts.size else 0
    target = np.maximum(counts, np.floor(max_count * float(ratio)).astype(np.int64))
    return target.astype(np.int64)


def _sample_neighbor_pairs(class_indices: np.ndarray, n_pairs: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if class_indices.size == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
    if class_indices.size == 1:
        src = np.repeat(class_indices, n_pairs).astype(np.int64, copy=False)
        nbr = src.copy()
        return src, nbr
    src = rng.choice(class_indices, size=int(n_pairs), replace=True)
    nbr = rng.choice(class_indices, size=int(n_pairs), replace=True)
    same = src == nbr
    if np.any(same):
        alt = rng.choice(class_indices, size=int(np.sum(same)), replace=True)
        # Ensure different when possible.
        if class_indices.size > 1:
            bad = alt == src[same]
            while np.any(bad):
                alt[bad] = rng.choice(class_indices, size=int(np.sum(bad)), replace=True)
                bad = alt == src[same]
        nbr[same] = alt
    return src.astype(np.int64, copy=False), nbr.astype(np.int64, copy=False)


def smote_like_augment_dataset(
    dataset: EEGEpochDataset,
    *,
    ratio: float = 1.0,
    max_new_samples: int | None = None,
    lambda_min: float = 0.1,
    lambda_max: float = 0.9,
    seed: int = 42,
    stage_name: str | None = None,
) -> EEGEpochDataset:
    """Train-only SMOTE-like oversampling within each stage target class.

    Notes:
    - Operates on the already stage-sliced view (channels/time) of `dataset`.
    - Uses same-class random interpolation (SMOTE-like) rather than exact kNN-SMOTE,
      which keeps CPU/memory bounded for large EEG epoch sets.
    - Preserves raw labels + metadata from anchor samples for compatibility with
      stage target remapping and sampler diagnostics.
    """
    if len(dataset) == 0:
        return dataset
    if float(ratio) < 1.0:
        return dataset

    Xv, yv, meta = _materialize_dataset_view(dataset)
    targets = dataset.targets.astype(np.int64, copy=False)
    if targets.size == 0:
        return dataset
    n_classes = int(targets.max()) + 1
    counts = np.bincount(targets, minlength=n_classes).astype(np.int64)
    if np.count_nonzero(counts) <= 1:
        return dataset
    target_counts = _target_counts_from_ratio(counts, ratio=float(ratio))
    needed = np.maximum(0, target_counts - counts)
    total_needed = int(needed.sum())
    if total_needed <= 0:
        return dataset
    if max_new_samples is not None:
        max_new = max(0, int(max_new_samples))
        if total_needed > max_new > 0:
            # Scale down per class proportionally.
            frac = float(max_new) / float(total_needed)
            needed = np.floor(needed * frac).astype(np.int64)
            # Fill any leftover slots greedily by largest remaining deficits.
            rem = max_new - int(needed.sum())
            if rem > 0:
                deficits = np.maximum(0, target_counts - counts - needed)
                order = np.argsort(-deficits)
                for c in order.tolist():
                    if rem <= 0 or deficits[c] <= 0:
                        continue
                    add = min(int(deficits[c]), rem)
                    needed[c] += add
                    rem -= add
        total_needed = int(needed.sum())
    if total_needed <= 0:
        return dataset

    rng = np.random.default_rng(int(seed))
    lam_lo = float(lambda_min)
    lam_hi = float(lambda_max)
    if not (0.0 <= lam_lo <= lam_hi <= 1.0):
        raise ValueError(f"Invalid lambda range [{lam_lo}, {lam_hi}]")

    synth_X_parts: list[np.ndarray] = []
    synth_y_parts: list[np.ndarray] = []
    synth_meta_parts: list[pd.DataFrame] = []

    for c in range(n_classes):
        n_new = int(needed[c])
        if n_new <= 0:
            continue
        cls_idx = np.flatnonzero(targets == c).astype(np.int64)
        if cls_idx.size == 0:
            continue
        src_local, nbr_local = _sample_neighbor_pairs(cls_idx, n_new, rng)
        lam = rng.uniform(lam_lo, lam_hi, size=(n_new, 1, 1)).astype(np.float32)
        x_src = Xv[src_local]
        x_nbr = Xv[nbr_local]
        x_syn = x_src + lam * (x_nbr - x_src)
        y_syn = yv[src_local].astype(np.int64, copy=True)
        meta_syn = meta.iloc[src_local].reset_index(drop=True).copy()
        meta_syn["file"] = meta_syn["file"].astype(str) + "::smote"
        synth_X_parts.append(x_syn.astype(np.float32, copy=False))
        synth_y_parts.append(y_syn)
        synth_meta_parts.append(meta_syn)

    if not synth_X_parts:
        return dataset

    X_aug = np.concatenate([Xv] + synth_X_parts, axis=0).astype(np.float32, copy=False)
    y_aug = np.concatenate([yv] + synth_y_parts, axis=0).astype(np.int64, copy=False)
    meta_aug = pd.concat([meta] + synth_meta_parts, axis=0, ignore_index=True)

    out = EEGEpochDataset(
        X_aug,
        y_aug,
        meta_aug,
        stage=dataset.stage,
        add_channel_dim=dataset.add_channel_dim,
        augmenter=dataset.augmenter,
        indices=None,
        channel_indices=None,
        time_indices=None,
    )
    before = counts.tolist()
    after = np.bincount(out.targets.astype(np.int64), minlength=n_classes).astype(int).tolist()
    tag = f"{stage_name}: " if stage_name else ""
    print(f"[smote] {tag}targets before={before} after={after} (+{int(X_aug.shape[0] - Xv.shape[0])})")
    return out


def smote_cfg_for_stage(train_cfg: dict[str, Any], stage_name: str) -> dict[str, Any] | None:
    cfg = train_cfg.get("smote")
    if not isinstance(cfg, dict) or not cfg.get("enabled", False):
        return None
    stages = cfg.get("stages")
    if stages:
        allowed = {str(s) for s in stages}
        if str(stage_name) not in allowed:
            return None
    stage_overrides = cfg.get("overrides", {})
    out = dict(cfg)
    if isinstance(stage_overrides, dict):
        if stage_name in stage_overrides and isinstance(stage_overrides[stage_name], dict):
            out.update(stage_overrides[stage_name])
    return out
