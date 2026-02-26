from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def _as_contiguous_slice(idx: np.ndarray | None, max_size: int) -> slice | None:
    if idx is None:
        return None
    arr = np.asarray(idx, dtype=np.int64)
    if arr.ndim != 1 or arr.size == 0:
        return None
    if int(arr.min()) < 0 or int(arr.max()) >= int(max_size):
        return None
    if arr.size == 1:
        i = int(arr[0])
        return slice(i, i + 1, 1)
    diffs = np.diff(arr)
    if np.all(diffs == 1):
        return slice(int(arr[0]), int(arr[-1]) + 1, 1)
    return None


def mask_by_subjects(meta: pd.DataFrame, subjects: Sequence[str] | None) -> np.ndarray:
    if not subjects:
        return np.ones(len(meta), dtype=bool)
    return meta["subject"].isin(list(subjects)).to_numpy(dtype=bool)


def mask_by_run_kind(meta: pd.DataFrame, run_kind: str | None) -> np.ndarray:
    if run_kind is None or run_kind == "combined":
        return np.ones(len(meta), dtype=bool)
    if run_kind not in {"baseline", "executed", "imagined"}:
        raise ValueError(f"Unsupported run_kind filter: {run_kind}")
    return (meta["run_kind"].astype(str) == run_kind).to_numpy(dtype=bool)


def mask_by_task_types(meta: pd.DataFrame, task_types: Sequence[str] | None) -> np.ndarray:
    if not task_types:
        return np.ones(len(meta), dtype=bool)
    allowed = {str(t).upper() for t in task_types}
    return meta["task_type"].astype(str).str.upper().isin(allowed).to_numpy(dtype=bool)


def stage_targets(y: np.ndarray, stage: str) -> np.ndarray:
    if stage == "stage_a":
        return (y > 0).astype(np.int64)
    if stage == "stage_a_lr":
        if not np.all(np.isin(y, [0, 1, 2])):
            raise ValueError("Stage A LR target conversion expects labels in {0,1,2}")
        return (y > 0).astype(np.int64)
    if stage == "stage_a_ff":
        if not np.all(np.isin(y, [0, 3, 4])):
            raise ValueError("Stage A FF target conversion expects labels in {0,3,4}")
        return (y > 0).astype(np.int64)
    if stage == "stage_b":
        if np.any(y == 0):
            raise ValueError("Stage B target conversion received REST labels. Filter active-only first.")
        return (y - 1).astype(np.int64)
    if stage == "stage_b_lr":
        if not np.all(np.isin(y, [1, 2])):
            raise ValueError("Stage B LR target conversion expects labels in {1,2}")
        return (y - 1).astype(np.int64)
    if stage == "stage_b_ff":
        if not np.all(np.isin(y, [3, 4])):
            raise ValueError("Stage B FF target conversion expects labels in {3,4}")
        return (y - 3).astype(np.int64)
    if stage in {"multiclass", "multiclass5", "none"}:
        return y.astype(np.int64)
    raise ValueError(f"Unsupported stage target mode: {stage}")


def subset_indices(
    y: np.ndarray,
    meta: pd.DataFrame,
    *,
    subjects: Sequence[str] | None = None,
    run_kind: str | None = None,
    active_only: bool = False,
    task_types: Sequence[str] | None = None,
) -> np.ndarray:
    mask = mask_by_subjects(meta, subjects) & mask_by_run_kind(meta, run_kind) & mask_by_task_types(meta, task_types)
    if active_only:
        mask &= (y > 0)
    return np.flatnonzero(mask).astype(np.int64, copy=False)


def subset_arrays(
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    *,
    subjects: Sequence[str] | None = None,
    run_kind: str | None = None,
    active_only: bool = False,
    task_types: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    idx = subset_indices(y, meta, subjects=subjects, run_kind=run_kind, active_only=active_only, task_types=task_types)
    X_sub = X[idx]
    y_sub = y[idx]
    meta_sub = meta.iloc[idx].reset_index(drop=True)
    return X_sub, y_sub, meta_sub


class EEGEpochDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        meta: pd.DataFrame,
        *,
        stage: str = "multiclass5",
        add_channel_dim: bool = False,
        augmenter=None,
        indices: np.ndarray | Sequence[int] | None = None,
        channel_indices: np.ndarray | Sequence[int] | None = None,
        time_indices: np.ndarray | Sequence[int] | None = None,
    ) -> None:
        assert X.ndim == 3, f"X must be [n,C,T], got {X.shape}"
        assert y.ndim == 1, f"y must be [n], got {y.shape}"
        assert len(meta) == len(y) == X.shape[0], "Dataset arrays/meta length mismatch"
        self.X_full = X.astype(np.float32, copy=False)
        self.y_full = y.astype(np.int64, copy=False)
        self.meta_full = meta.reset_index(drop=True)
        if indices is None:
            self.indices = np.arange(self.X_full.shape[0], dtype=np.int64)
        else:
            self.indices = np.asarray(indices, dtype=np.int64)
            if self.indices.ndim != 1:
                raise ValueError(f"indices must be 1D, got shape={self.indices.shape}")
            if self.indices.size > 0:
                if int(self.indices.min()) < 0 or int(self.indices.max()) >= self.X_full.shape[0]:
                    raise IndexError("Dataset indices out of bounds")
        if channel_indices is None:
            self.channel_indices = None
        else:
            self.channel_indices = np.asarray(channel_indices, dtype=np.int64)
            if self.channel_indices.ndim != 1:
                raise ValueError(f"channel_indices must be 1D, got shape={self.channel_indices.shape}")
            if self.channel_indices.size > 0:
                if int(self.channel_indices.min()) < 0 or int(self.channel_indices.max()) >= self.X_full.shape[1]:
                    raise IndexError("Dataset channel_indices out of bounds")
                if len(np.unique(self.channel_indices)) != len(self.channel_indices):
                    raise ValueError("Dataset channel_indices must be unique")
        self.channel_slice = _as_contiguous_slice(self.channel_indices, self.X_full.shape[1])
        if time_indices is None:
            self.time_indices = None
        else:
            self.time_indices = np.asarray(time_indices, dtype=np.int64)
            if self.time_indices.ndim != 1:
                raise ValueError(f"time_indices must be 1D, got shape={self.time_indices.shape}")
            if self.time_indices.size > 0:
                if int(self.time_indices.min()) < 0 or int(self.time_indices.max()) >= self.X_full.shape[2]:
                    raise IndexError("Dataset time_indices out of bounds")
                if len(np.unique(self.time_indices)) != len(self.time_indices):
                    raise ValueError("Dataset time_indices must be unique")
        self.time_slice = _as_contiguous_slice(self.time_indices, self.X_full.shape[2])
        self.y = self.y_full[self.indices]
        self.meta = self.meta_full.iloc[self.indices].reset_index(drop=True)
        self.stage = stage
        self.add_channel_dim = add_channel_dim
        self.augmenter = augmenter

        if stage == "stage_b" and np.any(self.y == 0):
            raise ValueError("Stage B dataset contains REST labels. Filter active_only=True before dataset creation.")
        if stage == "stage_a_lr" and not np.all(np.isin(self.y, [0, 1, 2])):
            raise ValueError("Stage A LR dataset must contain only REST/LEFT/RIGHT labels")
        if stage == "stage_a_ff" and not np.all(np.isin(self.y, [0, 3, 4])):
            raise ValueError("Stage A FF dataset must contain only REST/FISTS/FEET labels")
        if stage == "stage_b_lr" and not np.all(np.isin(self.y, [1, 2])):
            raise ValueError("Stage B LR dataset must contain only LEFT/RIGHT labels")
        if stage == "stage_b_ff" and not np.all(np.isin(self.y, [3, 4])):
            raise ValueError("Stage B FF dataset must contain only FISTS/FEET labels")
        self.targets = stage_targets(self.y, self.stage)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, index: int):
        src_idx = int(self.indices[index])
        x = self.X_full[src_idx]
        if self.channel_indices is not None:
            x = x[self.channel_slice] if self.channel_slice is not None else x[self.channel_indices]
        if self.time_indices is not None:
            x = x[:, self.time_slice] if self.time_slice is not None else x[:, self.time_indices]
        if self.augmenter is not None:
            x = self.augmenter(x)
        if self.add_channel_dim:
            x = x[None, :, :]
        x_t = torch.from_numpy(x.astype(np.float32, copy=False))
        return x_t, int(self.targets[index])


@dataclass
class EEGDataModule:
    train_dataset: EEGEpochDataset
    val_dataset: EEGEpochDataset
    test_dataset: EEGEpochDataset | None = None

    def train_loader(self, **kwargs) -> DataLoader:
        return DataLoader(self.train_dataset, **kwargs)

    def val_loader(self, **kwargs) -> DataLoader:
        return DataLoader(self.val_dataset, **kwargs)

    def test_loader(self, **kwargs) -> DataLoader:
        if self.test_dataset is None:
            raise ValueError("No test_dataset configured")
        return DataLoader(self.test_dataset, **kwargs)
