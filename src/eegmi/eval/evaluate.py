from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from eegmi.constants import LABEL_TO_NAME, STAGE_A_LABELS, STAGE_B_LABELS
from eegmi.data.channel_selection import resolve_stage_channel_indices
from eegmi.data.time_selection import resolve_stage_time_indices
from eegmi.data.dataset import EEGEpochDataset, subset_arrays, subset_indices
from eegmi.data.loader import load_eegmmidb_epochs
from eegmi.eval.explain import save_saliency_examples
from eegmi.eval.plots import (
    save_confusion_matrix_png,
    save_erd_topomap,
    save_learning_curves,
    save_psd_rest_vs_active,
)
from eegmi.models.heads import build_model
from eegmi.train.checkpointing import load_checkpoint
from eegmi.train.metrics import classification_metrics
from eegmi.utils import ensure_dir, to_serializable, write_json


def _predict_logits_dataset(
    model: torch.nn.Module,
    dataset: EEGEpochDataset,
    batch_size: int = 128,
    device: str = "cpu",
    tta_time_shifts: list[int] | tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if len(dataset) == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0, 0), dtype=np.float32)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval().to(device)
    ys = []
    logits_all = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            if tta_time_shifts:
                logits_sum = None
                for shift in tta_time_shifts:
                    x_shift = _shift_batch_time(x, int(shift))
                    logits_s = model(x_shift)
                    logits_sum = logits_s if logits_sum is None else (logits_sum + logits_s)
                logits = logits_sum / float(len(tta_time_shifts))
            else:
                logits = model(x)
            ys.append(y.cpu().numpy())
            logits_all.append(logits.cpu().numpy().astype(np.float32))
    return np.concatenate(ys), np.concatenate(logits_all, axis=0)


def _shift_batch_time(x: torch.Tensor, shift: int) -> torch.Tensor:
    if shift == 0:
        return x
    y = torch.zeros_like(x)
    if shift > 0:
        y[..., shift:] = x[..., :-shift]
    else:
        y[..., :shift] = x[..., -shift:]
    return y


def _predict_dataset(
    model: torch.nn.Module,
    dataset: EEGEpochDataset,
    batch_size: int = 128,
    device: str = "cpu",
    tta_time_shifts: list[int] | tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    y_true, logits = _predict_logits_dataset(model, dataset, batch_size=batch_size, device=device, tta_time_shifts=tta_time_shifts)
    if logits.size == 0:
        return y_true, np.empty((0,), dtype=np.int64)
    y_pred = np.argmax(logits, axis=1).astype(np.int64)
    return y_true, y_pred


def _set_bn_train_only(model: torch.nn.Module) -> None:
    model.eval()
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            m.train()


def _adapt_bn_stats(
    model: torch.nn.Module | None,
    X: np.ndarray,
    y: np.ndarray,
    meta,
    *,
    stage: str,
    device: str,
    batch_size: int,
    indices: np.ndarray | None = None,
    channel_indices: np.ndarray | None = None,
    time_indices: np.ndarray | None = None,
    num_passes: int = 1,
) -> torch.nn.Module | None:
    if model is None:
        return None
    ds = EEGEpochDataset(
        X,
        y,
        meta,
        indices=indices,
        channel_indices=channel_indices,
        time_indices=time_indices,
        stage=stage,
        add_channel_dim=False,
        augmenter=None,
    )
    if len(ds) == 0:
        return model.eval().to(device)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    _set_bn_train_only(model)
    model.to(device)
    with torch.no_grad():
        for _ in range(max(1, int(num_passes))):
            for x, _ in loader:
                _ = model(x.to(device))
    model.eval()
    return model


def _predict_logits_with_imagery_dispatch(
    base_model: torch.nn.Module,
    imagery_model: torch.nn.Module | None,
    X: np.ndarray,
    y: np.ndarray,
    meta,
    *,
    stage: str,
    batch_size: int,
    device: str,
    indices: np.ndarray | None = None,
    tta_time_shifts: list[int] | tuple[int, ...] | None = None,
    channel_indices: np.ndarray | None = None,
    imagery_channel_indices: np.ndarray | None = None,
    time_indices: np.ndarray | None = None,
    imagery_time_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, Any]:
    ds = EEGEpochDataset(
        X,
        y,
        meta,
        indices=indices,
        channel_indices=channel_indices,
        time_indices=time_indices,
        stage=stage,
        add_channel_dim=False,
        augmenter=None,
    )
    y_true, logits = _predict_logits_dataset(base_model, ds, batch_size=batch_size, device=device, tta_time_shifts=tta_time_shifts)
    meta_ds = ds.meta
    if imagery_model is None or len(ds) == 0:
        return y_true, logits, meta_ds

    imag_local_mask = (meta_ds["run_kind"].astype(str) == "imagined").to_numpy(dtype=bool)
    if not imag_local_mask.any():
        return y_true, logits, meta_ds

    if indices is None:
        X_img = X[imag_local_mask]
        y_img = y[imag_local_mask]
        meta_img = meta.iloc[imag_local_mask].reset_index(drop=True)
        ds_img = EEGEpochDataset(
            X_img,
            y_img,
            meta_img,
            channel_indices=imagery_channel_indices if imagery_channel_indices is not None else channel_indices,
            time_indices=imagery_time_indices if imagery_time_indices is not None else time_indices,
            stage=stage,
            add_channel_dim=False,
            augmenter=None,
        )
    else:
        idx_np = np.asarray(indices, dtype=np.int64)
        ds_img = EEGEpochDataset(
            X,
            y,
            meta,
            indices=idx_np[imag_local_mask],
            channel_indices=imagery_channel_indices if imagery_channel_indices is not None else channel_indices,
            time_indices=imagery_time_indices if imagery_time_indices is not None else time_indices,
            stage=stage,
            add_channel_dim=False,
            augmenter=None,
        )

    _, logits_img = _predict_logits_dataset(imagery_model, ds_img, batch_size=batch_size, device=device, tta_time_shifts=tta_time_shifts)
    logits[imag_local_mask] = logits_img
    return y_true, logits, meta_ds


def _predict_stage_a_logits(
    stage_a_model: torch.nn.Module,
    stage_a_imagined_model: torch.nn.Module | None,
    stage_a_family_models: dict[str, torch.nn.Module] | None,
    stage_a_family_imagined_models: dict[str, torch.nn.Module | None] | None,
    X: np.ndarray,
    y: np.ndarray,
    meta,
    *,
    batch_size: int,
    device: str,
    indices: np.ndarray | None = None,
    tta_time_shifts: list[int] | tuple[int, ...] | None = None,
    stage_a_channel_indices: np.ndarray | None = None,
    stage_a_imagined_channel_indices: np.ndarray | None = None,
    stage_a_family_channel_indices: dict[str, np.ndarray] | None = None,
    stage_a_family_imagined_channel_indices: dict[str, np.ndarray] | None = None,
    stage_a_time_indices: np.ndarray | None = None,
    stage_a_imagined_time_indices: np.ndarray | None = None,
    stage_a_family_time_indices: dict[str, np.ndarray] | None = None,
    stage_a_family_imagined_time_indices: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, Any]:
    y_true, logits, meta_ds = _predict_logits_with_imagery_dispatch(
        stage_a_model,
        stage_a_imagined_model,
        X,
        y,
        meta,
        stage="stage_a",
        batch_size=batch_size,
        device=device,
        indices=indices,
        tta_time_shifts=tta_time_shifts,
        channel_indices=stage_a_channel_indices,
        imagery_channel_indices=stage_a_imagined_channel_indices,
        time_indices=stage_a_time_indices,
        imagery_time_indices=stage_a_imagined_time_indices,
    )
    if not stage_a_family_models or len(meta_ds) == 0:
        return y_true, logits, meta_ds

    task_arr = meta_ds["task_type"].astype(str).str.upper().to_numpy()
    idx_np = None if indices is None else np.asarray(indices, dtype=np.int64)
    for family_key, task_type, stage_mode in (("lr", "LR", "stage_a_lr"), ("ff", "FF", "stage_a_ff")):
        family_model = stage_a_family_models.get(family_key)
        if family_model is None:
            continue
        family_imag_model = None
        if stage_a_family_imagined_models is not None:
            family_imag_model = stage_a_family_imagined_models.get(family_key)
        mask = task_arr == task_type
        if not mask.any():
            continue
        if idx_np is None:
            _, logits_family, _ = _predict_logits_with_imagery_dispatch(
                family_model,
                family_imag_model,
                X[mask],
                y[mask],
                meta.iloc[np.flatnonzero(mask)].reset_index(drop=True),
                stage=stage_mode,
                batch_size=batch_size,
                device=device,
                indices=None,
                tta_time_shifts=tta_time_shifts,
                channel_indices=(
                    (stage_a_family_channel_indices or {}).get(family_key)
                    if stage_a_family_channel_indices is not None
                    else stage_a_channel_indices
                ),
                imagery_channel_indices=(
                    (stage_a_family_imagined_channel_indices or {}).get(family_key)
                    if stage_a_family_imagined_channel_indices is not None
                    else stage_a_imagined_channel_indices
                ),
                time_indices=(
                    (stage_a_family_time_indices or {}).get(family_key)
                    if stage_a_family_time_indices is not None
                    else stage_a_time_indices
                ),
                imagery_time_indices=(
                    (stage_a_family_imagined_time_indices or {}).get(family_key)
                    if stage_a_family_imagined_time_indices is not None
                    else stage_a_imagined_time_indices
                ),
            )
        else:
            _, logits_family, _ = _predict_logits_with_imagery_dispatch(
                family_model,
                family_imag_model,
                X,
                y,
                meta,
                stage=stage_mode,
                batch_size=batch_size,
                device=device,
                indices=idx_np[mask],
                tta_time_shifts=tta_time_shifts,
                channel_indices=(
                    (stage_a_family_channel_indices or {}).get(family_key)
                    if stage_a_family_channel_indices is not None
                    else stage_a_channel_indices
                ),
                imagery_channel_indices=(
                    (stage_a_family_imagined_channel_indices or {}).get(family_key)
                    if stage_a_family_imagined_channel_indices is not None
                    else stage_a_imagined_channel_indices
                ),
                time_indices=(
                    (stage_a_family_time_indices or {}).get(family_key)
                    if stage_a_family_time_indices is not None
                    else stage_a_time_indices
                ),
                imagery_time_indices=(
                    (stage_a_family_imagined_time_indices or {}).get(family_key)
                    if stage_a_family_imagined_time_indices is not None
                    else stage_a_imagined_time_indices
                ),
            )
        logits[mask] = logits_family
    return y_true, logits, meta_ds


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    if logits.size == 0:
        return np.empty_like(logits)
    x = logits.astype(np.float32, copy=False)
    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / np.maximum(ex.sum(axis=1, keepdims=True), 1e-12)


def _stage_a_active_probs(logits_a: np.ndarray) -> np.ndarray:
    if logits_a.ndim != 2 or logits_a.shape[1] != 2:
        raise ValueError(f"Stage A logits must be [N,2], got {logits_a.shape}")
    return _softmax_np(logits_a)[:, 1].astype(np.float32, copy=False)


def _threshold_grid_from_probs(probs: np.ndarray) -> np.ndarray:
    if probs.size == 0:
        return np.array([0.5], dtype=np.float32)
    quantiles = np.linspace(0.02, 0.98, 49, dtype=np.float32)
    qvals = np.quantile(probs, quantiles).astype(np.float32)
    base = np.linspace(0.05, 0.95, 37, dtype=np.float32)
    grid = np.unique(np.clip(np.concatenate([base, qvals, np.array([0.5], dtype=np.float32)]), 0.01, 0.99))
    return grid.astype(np.float32)


def _fit_binary_threshold_bal_acc(y_true_bin: np.ndarray, p_active: np.ndarray) -> dict[str, Any]:
    y_true_bin = np.asarray(y_true_bin, dtype=np.int64)
    p_active = np.asarray(p_active, dtype=np.float32)
    if y_true_bin.size == 0:
        return {"threshold": 0.5, "metrics": {}, "n_samples": 0, "active_rate_true": 0.0}
    best = None
    for thr in _threshold_grid_from_probs(p_active):
        y_pred = (p_active >= float(thr)).astype(np.int64)
        metrics = classification_metrics(y_true_bin, y_pred, labels=[0, 1], label_names=["REST", "ACTIVE"]).to_dict()
        score = float(metrics["balanced_accuracy"] + 0.25 * metrics["macro_f1"])
        cand = {
            "threshold": float(thr),
            "score": score,
            "metrics": metrics,
            "n_samples": int(y_true_bin.size),
            "active_rate_true": float(np.mean(y_true_bin.astype(np.float32))),
        }
        if best is None or cand["score"] > best["score"]:
            best = cand
    assert best is not None
    return best


def calibrate_stage_a_thresholds(
    *,
    stage_a_model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    meta,
    subjects: list[str],
    device: str = "cpu",
    batch_size: int = 128,
    stage_a_imagined_model: torch.nn.Module | None = None,
    stage_a_family_models: dict[str, torch.nn.Module] | None = None,
    stage_a_family_imagined_models: dict[str, torch.nn.Module | None] | None = None,
    stage_a_channel_indices: np.ndarray | None = None,
    stage_a_imagined_channel_indices: np.ndarray | None = None,
    stage_a_family_channel_indices: dict[str, np.ndarray] | None = None,
    stage_a_family_imagined_channel_indices: dict[str, np.ndarray] | None = None,
    stage_a_time_indices: np.ndarray | None = None,
    stage_a_imagined_time_indices: np.ndarray | None = None,
    stage_a_family_time_indices: dict[str, np.ndarray] | None = None,
    stage_a_family_imagined_time_indices: dict[str, np.ndarray] | None = None,
    stage_a_tta_time_shifts: list[int] | tuple[int, ...] | None = None,
    granularity: str = "run_kind",
) -> dict[str, Any]:
    """Fit Stage A REST/ACTIVE thresholds on validation subjects only.

    Uses known `run_kind` metadata to optionally apply separate executed vs imagined thresholds.
    """
    idx_val = subset_indices(y, meta, subjects=subjects, run_kind="combined", active_only=False)
    y_true_a, logits_a, meta_val = _predict_stage_a_logits(
        stage_a_model,
        stage_a_imagined_model,
        stage_a_family_models,
        stage_a_family_imagined_models,
        X,
        y,
        meta,
        batch_size=batch_size,
        device=device,
        indices=idx_val,
        tta_time_shifts=stage_a_tta_time_shifts,
        stage_a_channel_indices=stage_a_channel_indices,
        stage_a_imagined_channel_indices=stage_a_imagined_channel_indices,
        stage_a_family_channel_indices=stage_a_family_channel_indices,
        stage_a_family_imagined_channel_indices=stage_a_family_imagined_channel_indices,
        stage_a_time_indices=stage_a_time_indices,
        stage_a_imagined_time_indices=stage_a_imagined_time_indices,
        stage_a_family_time_indices=stage_a_family_time_indices,
        stage_a_family_imagined_time_indices=stage_a_family_imagined_time_indices,
    )
    p_active = _stage_a_active_probs(logits_a) if logits_a.size else np.empty((0,), dtype=np.float32)
    run_kind_arr = meta_val["run_kind"].astype(str).to_numpy() if len(meta_val) else np.array([], dtype=str)

    out: dict[str, Any] = {
        "default": 0.5,
        "by_run_kind": {},
        "by_run_kind_task": {},
        "fit": {},
        "granularity": str(granularity),
        "target_active_rate_default": None,
        "target_active_rate_by_run_kind": {},
        "target_active_rate_by_run_kind_task": {},
    }
    fit_combined = _fit_binary_threshold_bal_acc(y_true_a, p_active)
    out["default"] = float(fit_combined["threshold"])
    out["target_active_rate_default"] = float(fit_combined.get("active_rate_true", 0.5))
    out["fit"]["combined"] = fit_combined

    for rk in ["executed", "imagined"]:
        mask = run_kind_arr == rk
        fit_rk = _fit_binary_threshold_bal_acc(y_true_a[mask], p_active[mask])
        out["by_run_kind"][rk] = float(fit_rk["threshold"])
        out["target_active_rate_by_run_kind"][rk] = float(fit_rk.get("active_rate_true", 0.5))
        out["fit"][rk] = fit_rk
    if str(granularity).lower() in {"run_kind_task", "run_kind+task_type", "task_type"}:
        task_arr = meta_val["task_type"].astype(str).str.upper().to_numpy() if len(meta_val) else np.array([], dtype=str)
        for rk in ["executed", "imagined"]:
            for tt in ["LR", "FF"]:
                mask = (run_kind_arr == rk) & (task_arr == tt)
                key = f"{rk}:{tt}"
                fit_rt = _fit_binary_threshold_bal_acc(y_true_a[mask], p_active[mask])
                out["by_run_kind_task"][key] = float(fit_rt["threshold"])
                out["target_active_rate_by_run_kind_task"][key] = float(fit_rt.get("active_rate_true", 0.5))
                out["fit"][key] = fit_rt
    return out


def _stage_a_predictions_from_logits(
    logits_a: np.ndarray,
    meta_sub,
    *,
    eval_run_kind: str,
    stage_a_thresholds: dict[str, Any] | None,
) -> np.ndarray:
    if logits_a.size == 0:
        return np.empty((0,), dtype=np.int64)
    if not stage_a_thresholds:
        return np.argmax(logits_a, axis=1).astype(np.int64)
    probs = _stage_a_active_probs(logits_a)
    y_pred = np.zeros((len(probs),), dtype=np.int64)
    mode = str(stage_a_thresholds.get("mode", "fixed")).lower()
    default_thr = float(stage_a_thresholds.get("default", 0.5))
    by_run_kind = dict(stage_a_thresholds.get("by_run_kind", {}))
    by_run_kind_task = dict(stage_a_thresholds.get("by_run_kind_task", {}))
    if mode in {"target_rate_quantile", "quantile_target_rate"}:
        target_default = stage_a_thresholds.get("target_active_rate_default", None)
        target_by_rk = dict(stage_a_thresholds.get("target_active_rate_by_run_kind", {}))
        target_by_rt = dict(stage_a_thresholds.get("target_active_rate_by_run_kind_task", {}))

        def _thr_from_target(group_probs: np.ndarray, target_rate: Any, fallback_thr: float) -> float:
            if group_probs.size == 0:
                return float(fallback_thr)
            if target_rate is None:
                return float(fallback_thr)
            tr = float(np.clip(float(target_rate), 0.0, 1.0))
            if tr <= 0.0:
                return float(np.nextafter(1.0, 2.0))  # predict all REST
            if tr >= 1.0:
                return float(np.nextafter(0.0, -1.0))  # predict all ACTIVE
            q = float(np.quantile(group_probs, 1.0 - tr))
            return q

        if eval_run_kind == "combined":
            rk_arr = meta_sub["run_kind"].astype(str).to_numpy()
            tt_arr = meta_sub["task_type"].astype(str).str.upper().to_numpy()
            thr_vec = np.full((len(probs),), _thr_from_target(probs, target_default, default_thr), dtype=np.float32)
            for rk in np.unique(rk_arr):
                m_rk = rk_arr == rk
                thr_rk = _thr_from_target(probs[m_rk], target_by_rk.get(str(rk)), float(by_run_kind.get(str(rk), default_thr)))
                thr_vec[m_rk] = np.float32(thr_rk)
            for key, tr in target_by_rt.items():
                if ":" not in str(key):
                    continue
                rk, tt = str(key).split(":", 1)
                m = (rk_arr == rk) & (tt_arr == tt.upper())
                if m.any():
                    thr_rt = _thr_from_target(probs[m], tr, float(by_run_kind_task.get(str(key), by_run_kind.get(rk, default_thr))))
                    thr_vec[m] = np.float32(thr_rt)
            return (probs >= thr_vec).astype(np.int64)
        tt_arr = meta_sub["task_type"].astype(str).str.upper().to_numpy()
        thr_vec = np.full((len(probs),), _thr_from_target(probs, target_by_rk.get(eval_run_kind, target_default), float(by_run_kind.get(eval_run_kind, default_thr))), dtype=np.float32)
        for key, tr in target_by_rt.items():
            if ":" not in str(key):
                continue
            rk, tt = str(key).split(":", 1)
            if rk != eval_run_kind:
                continue
            m = tt_arr == tt.upper()
            if m.any():
                thr_rt = _thr_from_target(probs[m], tr, float(by_run_kind_task.get(str(key), by_run_kind.get(eval_run_kind, default_thr))))
                thr_vec[m] = np.float32(thr_rt)
        return (probs >= thr_vec).astype(np.int64)
    if eval_run_kind == "combined":
        rk_arr = meta_sub["run_kind"].astype(str).to_numpy()
        tt_arr = meta_sub["task_type"].astype(str).str.upper().to_numpy()
        thr_vec = np.full((len(probs),), default_thr, dtype=np.float32)
        for rk, thr in by_run_kind.items():
            thr_vec[rk_arr == str(rk)] = float(thr)
        for key, thr in by_run_kind_task.items():
            if ":" not in str(key):
                continue
            rk, tt = str(key).split(":", 1)
            thr_vec[(rk_arr == rk) & (tt_arr == tt.upper())] = float(thr)
        y_pred = (probs >= thr_vec).astype(np.int64)
    else:
        thr_vec = np.full((len(probs),), float(by_run_kind.get(eval_run_kind, default_thr)), dtype=np.float32)
        if by_run_kind_task:
            tt_arr = meta_sub["task_type"].astype(str).str.upper().to_numpy()
            for key, thr in by_run_kind_task.items():
                if ":" not in str(key):
                    continue
                rk, tt = str(key).split(":", 1)
                if rk == eval_run_kind:
                    thr_vec[tt_arr == tt.upper()] = float(thr)
        y_pred = (probs >= thr_vec).astype(np.int64)
    return y_pred


def _apply_task_type_mask_to_logits(logits: np.ndarray, task_types: list[str] | np.ndarray) -> np.ndarray:
    """Mask invalid Stage B classes using known run family (LR vs FF).

    Stage B class indices: 0=LEFT, 1=RIGHT, 2=FISTS, 3=FEET.
    """
    masked = np.array(logits, copy=True)
    task_arr = np.asarray(task_types)
    if masked.ndim != 2 or masked.shape[1] != 4:
        raise ValueError(f"Expected Stage B logits [N,4], got {masked.shape}")
    neg_inf = np.finfo(masked.dtype).min
    lr_mask = np.char.upper(task_arr.astype(str)) == "LR"
    ff_mask = np.char.upper(task_arr.astype(str)) == "FF"
    if lr_mask.any():
        masked[lr_mask, 2:] = neg_inf
    if ff_mask.any():
        masked[ff_mask, :2] = neg_inf
    return masked


def _predict_stage_b_family_logits(
    family_models: dict[str, torch.nn.Module],
    family_imagined_models: dict[str, torch.nn.Module | None] | None,
    family_channel_indices: dict[str, np.ndarray] | None,
    family_imagined_channel_indices: dict[str, np.ndarray] | None,
    family_time_indices: dict[str, np.ndarray] | None,
    family_imagined_time_indices: dict[str, np.ndarray] | None,
    X_active: np.ndarray,
    y_active_true: np.ndarray,
    meta_active,
    *,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, Any]:
    if len(y_active_true) == 0:
        return np.empty((0,), dtype=np.int64), meta_active.iloc[:0].copy()
    final_pred = np.zeros((len(y_active_true),), dtype=np.int64)
    meta_local = meta_active.reset_index(drop=True)
    task_arr = meta_local["task_type"].astype(str).str.upper().to_numpy()
    for family_key, task_type, offset in (("lr", "LR", 1), ("ff", "FF", 3)):
        mask = task_arr == task_type
        if not mask.any():
            continue
        base_model = family_models.get(family_key)
        if base_model is None:
            raise ValueError(f"Missing family Stage B model for {family_key}")
        imagery_model = None
        if family_imagined_models is not None:
            imagery_model = family_imagined_models.get(family_key)
        X_sub = X_active[mask]
        y_sub = y_active_true[mask]
        meta_sub = meta_local.loc[mask].reset_index(drop=True)
        if family_key == "lr":
            y_stage = np.where(np.isin(y_sub, [1, 2]), y_sub, 1).astype(np.int64)
        else:
            y_stage = np.where(np.isin(y_sub, [3, 4]), y_sub, 3).astype(np.int64)
        _, logits, _ = _predict_logits_with_imagery_dispatch(
            base_model,
            imagery_model,
            X_sub,
            y_stage,
            meta_sub,
            stage=("stage_b_lr" if family_key == "lr" else "stage_b_ff"),
            batch_size=batch_size,
            device=device,
            indices=None,
            channel_indices=(family_channel_indices or {}).get(family_key) if family_channel_indices is not None else None,
            imagery_channel_indices=(
                (family_imagined_channel_indices or {}).get(family_key)
                if family_imagined_channel_indices is not None
                else None
            ),
            time_indices=(family_time_indices or {}).get(family_key) if family_time_indices is not None else None,
            imagery_time_indices=(
                (family_imagined_time_indices or {}).get(family_key)
                if family_imagined_time_indices is not None
                else None
            ),
        )
        pred_local = np.argmax(logits, axis=1).astype(np.int64)
        final_pred[mask] = pred_local + offset
    return final_pred, meta_local


def evaluate_stage_model(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    meta,
    *,
    stage: str,
    run_kind: str,
    subjects: list[str],
    device: str = "cpu",
    batch_size: int = 128,
) -> dict[str, Any]:
    active_only = stage == "stage_b"
    idx = subset_indices(y, meta, subjects=subjects, run_kind=run_kind, active_only=active_only)
    ds = EEGEpochDataset(X, y, meta, indices=idx, stage=stage, add_channel_dim=False, augmenter=None)
    y_true_stage, y_pred_stage = _predict_dataset(model, ds, batch_size=batch_size, device=device)

    if stage == "stage_a":
        labels = [0, 1]
        names = [STAGE_A_LABELS[i] for i in labels]
    elif stage == "stage_b":
        labels = [0, 1, 2, 3]
        names = [STAGE_B_LABELS[i] for i in labels]
    else:
        raise ValueError(stage)

    metrics = classification_metrics(y_true_stage, y_pred_stage, labels=labels, label_names=names).to_dict()
    return {
        "stage": stage,
        "run_kind": run_kind,
        "n_samples": int(len(ds)),
        "metrics": metrics,
        "y_true": y_true_stage.tolist(),
        "y_pred": y_pred_stage.tolist(),
    }


def evaluate_end_to_end(
    stage_a: torch.nn.Module,
    stage_b: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    meta,
    *,
    run_kind: str,
    subjects: list[str],
    device: str = "cpu",
    batch_size: int = 128,
    stage_b_task_type_masking: bool = False,
    stage_a_thresholds: dict[str, Any] | None = None,
    stage_a_tta_time_shifts: list[int] | tuple[int, ...] | None = None,
    stage_a_imagined_model: torch.nn.Module | None = None,
    stage_a_channel_indices: np.ndarray | None = None,
    stage_a_imagined_channel_indices: np.ndarray | None = None,
    stage_a_family_models: dict[str, torch.nn.Module] | None = None,
    stage_a_family_imagined_models: dict[str, torch.nn.Module | None] | None = None,
    stage_a_family_channel_indices: dict[str, np.ndarray] | None = None,
    stage_a_family_imagined_channel_indices: dict[str, np.ndarray] | None = None,
    stage_a_time_indices: np.ndarray | None = None,
    stage_a_imagined_time_indices: np.ndarray | None = None,
    stage_a_family_time_indices: dict[str, np.ndarray] | None = None,
    stage_a_family_imagined_time_indices: dict[str, np.ndarray] | None = None,
    stage_b_imagined_model: torch.nn.Module | None = None,
    stage_b_channel_indices: np.ndarray | None = None,
    stage_b_imagined_channel_indices: np.ndarray | None = None,
    stage_b_family_models: dict[str, torch.nn.Module] | None = None,
    stage_b_family_imagined_models: dict[str, torch.nn.Module | None] | None = None,
    stage_b_family_channel_indices: dict[str, np.ndarray] | None = None,
    stage_b_family_imagined_channel_indices: dict[str, np.ndarray] | None = None,
    stage_b_time_indices: np.ndarray | None = None,
    stage_b_imagined_time_indices: np.ndarray | None = None,
    stage_b_family_time_indices: dict[str, np.ndarray] | None = None,
    stage_b_family_imagined_time_indices: dict[str, np.ndarray] | None = None,
    stage_a_adabn: dict[str, Any] | None = None,
) -> dict[str, Any]:
    idx_sub = subset_indices(y, meta, subjects=subjects, run_kind=run_kind, active_only=False)
    y_sub = y[idx_sub]
    meta_sub = meta.iloc[idx_sub].reset_index(drop=True)
    stage_a_eval = stage_a
    stage_a_imag_eval = stage_a_imagined_model
    stage_a_family_eval = stage_a_family_models
    stage_a_family_imag_eval = stage_a_family_imagined_models
    if stage_a_adabn and bool(stage_a_adabn.get("enabled", False)) and idx_sub.size > 0:
        adabn_passes = int(stage_a_adabn.get("num_passes", 1))
        adabn_batch = int(stage_a_adabn.get("batch_size", batch_size))
        stage_a_eval = _adapt_bn_stats(
            copy.deepcopy(stage_a),
            X,
            y,
            meta,
            stage="stage_a",
            device=device,
            batch_size=adabn_batch,
            indices=idx_sub,
            channel_indices=stage_a_channel_indices,
            time_indices=stage_a_time_indices,
            num_passes=adabn_passes,
        )
        if stage_a_imagined_model is not None:
            img_mask = (meta_sub["run_kind"].astype(str).to_numpy() == "imagined")
            idx_img = idx_sub[img_mask]
            if idx_img.size > 0:
                stage_a_imag_eval = _adapt_bn_stats(
                    copy.deepcopy(stage_a_imagined_model),
                    X,
                    y,
                    meta,
                    stage="stage_a",
                    device=device,
                    batch_size=adabn_batch,
                    indices=idx_img,
                    channel_indices=(stage_a_imagined_channel_indices if stage_a_imagined_channel_indices is not None else stage_a_channel_indices),
                    time_indices=(stage_a_imagined_time_indices if stage_a_imagined_time_indices is not None else stage_a_time_indices),
                    num_passes=adabn_passes,
                )
        if stage_a_family_models:
            fam_base: dict[str, torch.nn.Module] = {}
            fam_ft: dict[str, torch.nn.Module | None] | None = {} if stage_a_family_imagined_models is not None else None
            task_arr_sub = meta_sub["task_type"].astype(str).str.upper().to_numpy()
            run_arr_sub = meta_sub["run_kind"].astype(str).to_numpy()
            for fam_key, task_type, stage_mode in (("lr", "LR", "stage_a_lr"), ("ff", "FF", "stage_a_ff")):
                base_m = stage_a_family_models.get(fam_key)
                if base_m is None:
                    continue
                idx_fam = idx_sub[task_arr_sub == task_type]
                fam_base[fam_key] = _adapt_bn_stats(
                    copy.deepcopy(base_m),
                    X,
                    y,
                    meta,
                    stage=stage_mode,
                    device=device,
                    batch_size=adabn_batch,
                    indices=idx_fam if idx_fam.size > 0 else None,
                    channel_indices=((stage_a_family_channel_indices or {}).get(fam_key) if stage_a_family_channel_indices is not None else stage_a_channel_indices),
                    time_indices=((stage_a_family_time_indices or {}).get(fam_key) if stage_a_family_time_indices is not None else stage_a_time_indices),
                    num_passes=adabn_passes,
                )
                if fam_ft is not None:
                    ft_m = (stage_a_family_imagined_models or {}).get(fam_key)
                    if ft_m is None:
                        fam_ft[fam_key] = None
                    else:
                        idx_fam_img = idx_sub[(task_arr_sub == task_type) & (run_arr_sub == "imagined")]
                        fam_ft[fam_key] = _adapt_bn_stats(
                            copy.deepcopy(ft_m),
                            X,
                            y,
                            meta,
                            stage=stage_mode,
                            device=device,
                            batch_size=adabn_batch,
                            indices=idx_fam_img if idx_fam_img.size > 0 else None,
                            channel_indices=(
                                ((stage_a_family_imagined_channel_indices or {}).get(fam_key))
                                if (stage_a_family_imagined_channel_indices is not None and (stage_a_family_imagined_channel_indices or {}).get(fam_key) is not None)
                                else (((stage_a_family_channel_indices or {}).get(fam_key)) if stage_a_family_channel_indices is not None else stage_a_channel_indices)
                            ),
                            time_indices=(
                                ((stage_a_family_imagined_time_indices or {}).get(fam_key))
                                if (stage_a_family_imagined_time_indices is not None and (stage_a_family_imagined_time_indices or {}).get(fam_key) is not None)
                                else (((stage_a_family_time_indices or {}).get(fam_key)) if stage_a_family_time_indices is not None else stage_a_time_indices)
                            ),
                            num_passes=adabn_passes,
                        )
            stage_a_family_eval = fam_base
            stage_a_family_imag_eval = fam_ft
    y_true_a, logits_a, _ = _predict_stage_a_logits(
        stage_a_eval,
        stage_a_imag_eval,
        stage_a_family_eval,
        stage_a_family_imag_eval,
        X,
        y,
        meta,
        batch_size=batch_size,
        device=device,
        indices=idx_sub,
        tta_time_shifts=stage_a_tta_time_shifts,
        stage_a_channel_indices=stage_a_channel_indices,
        stage_a_imagined_channel_indices=stage_a_imagined_channel_indices,
        stage_a_family_channel_indices=stage_a_family_channel_indices,
        stage_a_family_imagined_channel_indices=stage_a_family_imagined_channel_indices,
        stage_a_time_indices=stage_a_time_indices,
        stage_a_imagined_time_indices=stage_a_imagined_time_indices,
        stage_a_family_time_indices=stage_a_family_time_indices,
        stage_a_family_imagined_time_indices=stage_a_family_imagined_time_indices,
    )
    y_pred_a = _stage_a_predictions_from_logits(
        logits_a,
        meta_sub,
        eval_run_kind=run_kind,
        stage_a_thresholds=stage_a_thresholds,
    )

    final_pred = np.zeros_like(y_sub, dtype=np.int64)
    pred_active_mask = y_pred_a.astype(bool)
    if pred_active_mask.any():
        idx_pred_active = idx_sub[pred_active_mask]
        X_active_pred = X[idx_pred_active]
        y_active_pred_true = y_sub[pred_active_mask]
        meta_active_pred = meta_sub.loc[pred_active_mask].reset_index(drop=True)
        if stage_b_family_models:
            y_pred_global_b, _ = _predict_stage_b_family_logits(
                stage_b_family_models,
                stage_b_family_imagined_models,
                stage_b_family_channel_indices,
                stage_b_family_imagined_channel_indices,
                stage_b_family_time_indices,
                stage_b_family_imagined_time_indices,
                X_active_pred,
                np.where(y_active_pred_true > 0, y_active_pred_true, 1),
                meta_active_pred,
                batch_size=batch_size,
                device=device,
            )
            final_pred[pred_active_mask] = y_pred_global_b
        else:
            _, logits_b, meta_b_pred = _predict_logits_with_imagery_dispatch(
                stage_b,
                stage_b_imagined_model,
                X_active_pred,
                np.where(y_active_pred_true > 0, y_active_pred_true, 1),
                meta_active_pred,
                stage="stage_b",
                batch_size=batch_size,
                device=device,
                indices=None,
                channel_indices=stage_b_channel_indices,
                imagery_channel_indices=stage_b_imagined_channel_indices,
                time_indices=stage_b_time_indices,
                imagery_time_indices=stage_b_imagined_time_indices,
            )
            if stage_b_task_type_masking:
                logits_b = _apply_task_type_mask_to_logits(logits_b, meta_b_pred["task_type"].tolist())
            y_pred_b = np.argmax(logits_b, axis=1).astype(np.int64)
            final_pred[pred_active_mask] = y_pred_b + 1

    metrics_5 = classification_metrics(
        y_sub,
        final_pred,
        labels=[0, 1, 2, 3, 4],
        label_names=[LABEL_TO_NAME[i] for i in [0, 1, 2, 3, 4]],
    ).to_dict()
    stage_a_metrics = classification_metrics(y_true_a, y_pred_a, labels=[0, 1], label_names=["REST", "ACTIVE"]).to_dict()

    # Stage B metrics on true active subset for interpretability
    true_active_mask = y_sub > 0
    if true_active_mask.any():
        idx_true_active = idx_sub[true_active_mask]
        if stage_b_family_models:
            y_true_b = (y[idx_true_active] - 1).astype(np.int64)
            y_pred_b_global, meta_b_true = _predict_stage_b_family_logits(
                stage_b_family_models,
                stage_b_family_imagined_models,
                stage_b_family_channel_indices,
                stage_b_family_imagined_channel_indices,
                stage_b_family_time_indices,
                stage_b_family_imagined_time_indices,
                X[idx_true_active],
                y[idx_true_active],
                meta.iloc[idx_true_active].reset_index(drop=True),
                batch_size=batch_size,
                device=device,
            )
            y_pred_b_true_active = (y_pred_b_global - 1).astype(np.int64)
        else:
            # Use run-kind-dispatched Stage B on true active subset for interpretable metrics.
            y_true_b, logits_b_true_active, meta_b_true = _predict_logits_with_imagery_dispatch(
                stage_b,
                stage_b_imagined_model,
                X,
                y,
                meta,
                stage="stage_b",
                batch_size=batch_size,
                device=device,
                indices=idx_true_active,
                channel_indices=stage_b_channel_indices,
                imagery_channel_indices=stage_b_imagined_channel_indices,
                time_indices=stage_b_time_indices,
                imagery_time_indices=stage_b_imagined_time_indices,
            )
            if stage_b_task_type_masking:
                logits_b_true_active = _apply_task_type_mask_to_logits(logits_b_true_active, meta_b_true["task_type"].tolist())
            y_pred_b_true_active = np.argmax(logits_b_true_active, axis=1).astype(np.int64)
        stage_b_metrics = classification_metrics(
            y_true_b,
            y_pred_b_true_active,
            labels=[0, 1, 2, 3],
            label_names=["LEFT", "RIGHT", "FISTS", "FEET"],
        ).to_dict()
    else:
        stage_b_metrics = {}

    return {
        "run_kind": run_kind,
        "stage_b_task_type_masking": bool(stage_b_task_type_masking),
        "stage_a_family_heads": bool(stage_a_family_models),
        "stage_b_family_heads": bool(stage_b_family_models),
        "stage_a_thresholds": to_serializable(stage_a_thresholds) if stage_a_thresholds else None,
        "n_samples": int(len(y_sub)),
        "stage_a": {"metrics": stage_a_metrics, "y_true": y_true_a.tolist(), "y_pred": y_pred_a.tolist()},
        "stage_b_true_active": {"metrics": stage_b_metrics},
        "end_to_end": {"metrics": metrics_5, "y_true": y_sub.astype(int).tolist(), "y_pred": final_pred.astype(int).tolist()},
    }


def _load_models_from_combined_checkpoint(ckpt: dict[str, Any], device: str = "cpu") -> dict[str, torch.nn.Module | None]:
    cfg = ckpt["config"]
    data_cfg = cfg["data"]
    model_cfg = ckpt.get("model_config", cfg["model"])
    model_cfgs = ckpt.get("model_configs", {}) or {}
    n_chans = int(ckpt.get("n_chans", len(data_cfg["channels"])))
    n_times = int(ckpt.get("n_times", data_cfg["baseline_window_len"]))

    def _mc(stage_key: str) -> dict[str, Any]:
        cfg_stage = model_cfgs.get(stage_key)
        return cfg_stage if isinstance(cfg_stage, dict) else model_cfg

    def _nch(stage_key: str) -> int:
        try:
            idx, _ = resolve_stage_channel_indices(data_cfg, stage_key)
            return int(len(idx))
        except Exception:
            return int(n_chans)
    def _nt(stage_key: str) -> int:
        try:
            idx, _ = resolve_stage_time_indices(data_cfg, stage_key, n_times=n_times)
            return int(len(idx))
        except Exception:
            return int(n_times)

    def _bundle_item(bundle: Any, key: str) -> dict[str, Any] | None:
        if not isinstance(bundle, dict):
            return None
        item = bundle.get(key)
        return item if isinstance(item, dict) else None

    stage_a = build_model(_mc("stage_a"), n_chans=_nch("stage_a"), n_times=_nt("stage_a"), n_classes=2)
    stage_a.load_state_dict(ckpt["stage_a"]["model_state_dict"])
    stage_a.to(device).eval()

    stage_a_ft = None
    if ckpt.get("stage_a_finetuned") and ckpt["stage_a_finetuned"].get("model_state_dict"):
        stage_a_ft = build_model(_mc("stage_a_finetune_imagery"), n_chans=_nch("stage_a_finetune_imagery"), n_times=_nt("stage_a_finetune_imagery"), n_classes=2)
        stage_a_ft.load_state_dict(ckpt["stage_a_finetuned"]["model_state_dict"])
        stage_a_ft.to(device).eval()
    stage_a_family = None
    if ckpt.get("stage_a_family") and isinstance(ckpt["stage_a_family"], dict):
        stage_a_family = {}
        lr_item = _bundle_item(ckpt["stage_a_family"], "lr")
        if lr_item and lr_item.get("model_state_dict") is not None:
            m_lr = build_model(_mc("stage_a_lr"), n_chans=_nch("stage_a_lr"), n_times=_nt("stage_a_lr"), n_classes=2)
            m_lr.load_state_dict(lr_item["model_state_dict"])
            m_lr.to(device).eval()
            stage_a_family["lr"] = m_lr
        ff_item = _bundle_item(ckpt["stage_a_family"], "ff")
        if ff_item and ff_item.get("model_state_dict") is not None:
            m_ff = build_model(_mc("stage_a_ff"), n_chans=_nch("stage_a_ff"), n_times=_nt("stage_a_ff"), n_classes=2)
            m_ff.load_state_dict(ff_item["model_state_dict"])
            m_ff.to(device).eval()
            stage_a_family["ff"] = m_ff
    stage_a_family_ft = None
    if ckpt.get("stage_a_family_finetuned") and isinstance(ckpt["stage_a_family_finetuned"], dict):
        stage_a_family_ft = {}
        lr_item = _bundle_item(ckpt["stage_a_family_finetuned"], "lr")
        if lr_item and lr_item.get("model_state_dict") is not None:
            m_lr = build_model(_mc("stage_a_lr_finetune_imagery"), n_chans=_nch("stage_a_lr_finetune_imagery"), n_times=_nt("stage_a_lr_finetune_imagery"), n_classes=2)
            m_lr.load_state_dict(lr_item["model_state_dict"])
            m_lr.to(device).eval()
            stage_a_family_ft["lr"] = m_lr
        ff_item = _bundle_item(ckpt["stage_a_family_finetuned"], "ff")
        if ff_item and ff_item.get("model_state_dict") is not None:
            m_ff = build_model(_mc("stage_a_ff_finetune_imagery"), n_chans=_nch("stage_a_ff_finetune_imagery"), n_times=_nt("stage_a_ff_finetune_imagery"), n_classes=2)
            m_ff.load_state_dict(ff_item["model_state_dict"])
            m_ff.to(device).eval()
            stage_a_family_ft["ff"] = m_ff

    stage_b = build_model(_mc("stage_b"), n_chans=_nch("stage_b"), n_times=_nt("stage_b"), n_classes=4)
    stage_b.load_state_dict(ckpt["stage_b"]["model_state_dict"])
    stage_b.to(device).eval()

    stage_b_ft = None
    if ckpt.get("stage_b_finetuned") and ckpt["stage_b_finetuned"].get("model_state_dict"):
        stage_b_ft = build_model(_mc("stage_b_finetune_imagery"), n_chans=_nch("stage_b_finetune_imagery"), n_times=_nt("stage_b_finetune_imagery"), n_classes=4)
        stage_b_ft.load_state_dict(ckpt["stage_b_finetuned"]["model_state_dict"])
        stage_b_ft.to(device).eval()
    stage_b_family = None
    if ckpt.get("stage_b_family") and isinstance(ckpt["stage_b_family"], dict):
        stage_b_family = {}
        lr_item = _bundle_item(ckpt["stage_b_family"], "lr")
        if lr_item and lr_item.get("model_state_dict") is not None:
            m_lr = build_model(_mc("stage_b_lr"), n_chans=_nch("stage_b_lr"), n_times=_nt("stage_b_lr"), n_classes=2)
            m_lr.load_state_dict(lr_item["model_state_dict"])
            m_lr.to(device).eval()
            stage_b_family["lr"] = m_lr
        ff_item = _bundle_item(ckpt["stage_b_family"], "ff")
        if ff_item and ff_item.get("model_state_dict") is not None:
            m_ff = build_model(_mc("stage_b_ff"), n_chans=_nch("stage_b_ff"), n_times=_nt("stage_b_ff"), n_classes=2)
            m_ff.load_state_dict(ff_item["model_state_dict"])
            m_ff.to(device).eval()
            stage_b_family["ff"] = m_ff
    stage_b_family_ft = None
    if ckpt.get("stage_b_family_finetuned") and isinstance(ckpt["stage_b_family_finetuned"], dict):
        stage_b_family_ft = {}
        lr_item = _bundle_item(ckpt["stage_b_family_finetuned"], "lr")
        if lr_item and lr_item.get("model_state_dict") is not None:
            m_lr = build_model(_mc("stage_b_lr_finetune_imagery"), n_chans=_nch("stage_b_lr_finetune_imagery"), n_times=_nt("stage_b_lr_finetune_imagery"), n_classes=2)
            m_lr.load_state_dict(lr_item["model_state_dict"])
            m_lr.to(device).eval()
            stage_b_family_ft["lr"] = m_lr
        ff_item = _bundle_item(ckpt["stage_b_family_finetuned"], "ff")
        if ff_item and ff_item.get("model_state_dict") is not None:
            m_ff = build_model(_mc("stage_b_ff_finetune_imagery"), n_chans=_nch("stage_b_ff_finetune_imagery"), n_times=_nt("stage_b_ff_finetune_imagery"), n_classes=2)
            m_ff.load_state_dict(ff_item["model_state_dict"])
            m_ff.to(device).eval()
            stage_b_family_ft["ff"] = m_ff

    return {
        "stage_a": stage_a,
        "stage_a_finetuned": stage_a_ft,
        "stage_a_family": stage_a_family,
        "stage_a_family_finetuned": stage_a_family_ft,
        "stage_b": stage_b,
        "stage_b_finetuned": stage_b_ft,
        "stage_b_family": stage_b_family,
        "stage_b_family_finetuned": stage_b_family_ft,
    }


def _save_confusions_for_result(result: dict[str, Any], out_dir: Path) -> None:
    # stage_a
    if "stage_a" in result and result["stage_a"].get("metrics"):
        cm = np.asarray(result["stage_a"]["metrics"]["confusion_matrix"], dtype=int)
        save_confusion_matrix_png(cm, ["REST", "ACTIVE"], out_dir / f"cm_stage_a_{result['run_kind']}.png", title=f"Stage A ({result['run_kind']})")
    if result.get("stage_b_true_active", {}).get("metrics"):
        m = result["stage_b_true_active"]["metrics"]
        if m:
            cm = np.asarray(m["confusion_matrix"], dtype=int)
            save_confusion_matrix_png(cm, ["LEFT", "RIGHT", "FISTS", "FEET"], out_dir / f"cm_stage_b_{result['run_kind']}.png", title=f"Stage B ({result['run_kind']})")
    if "end_to_end" in result and result["end_to_end"].get("metrics"):
        cm = np.asarray(result["end_to_end"]["metrics"]["confusion_matrix"], dtype=int)
        save_confusion_matrix_png(cm, ["REST", "LEFT", "RIGHT", "FISTS", "FEET"], out_dir / f"cm_end_to_end_{result['run_kind']}.png", title=f"End-to-end ({result['run_kind']})")


def evaluate_hierarchical_bundle(
    *,
    cfg: dict[str, Any],
    split: dict[str, Any],
    stage_a_model: torch.nn.Module,
    stage_a_finetuned_model: torch.nn.Module | None = None,
    stage_a_family_models: dict[str, torch.nn.Module] | None = None,
    stage_a_family_finetuned_models: dict[str, torch.nn.Module | None] | None = None,
    stage_b_model: torch.nn.Module,
    stage_b_finetuned_model: torch.nn.Module | None = None,
    stage_b_family_models: dict[str, torch.nn.Module] | None = None,
    stage_b_family_finetuned_models: dict[str, torch.nn.Module | None] | None = None,
    stage_a_thresholds: dict[str, Any] | None = None,
    stage_a_thresholds_base: dict[str, Any] | None = None,
    stage_a_thresholds_finetuned: dict[str, Any] | None = None,
    output_dir: str | Path,
    device: str = "cpu",
    X: np.ndarray | None = None,
    y: np.ndarray | None = None,
    meta=None,
) -> dict[str, Any]:
    out_dir = ensure_dir(output_dir)
    data_cfg = cfg["data"]
    def _ch(stage_key: str) -> np.ndarray:
        idx, _ = resolve_stage_channel_indices(data_cfg, stage_key)
        return idx
    def _tm(stage_key: str) -> np.ndarray:
        idx, _ = resolve_stage_time_indices(data_cfg, stage_key, n_times=int(X.shape[2]) if X is not None else None)
        return idx
    stage_a_ch = _ch("stage_a")
    stage_a_ft_ch = _ch("stage_a_finetune_imagery")
    stage_b_ch = _ch("stage_b")
    stage_b_ft_ch = _ch("stage_b_finetune_imagery")
    stage_a_tm = _tm("stage_a")
    stage_a_ft_tm = _tm("stage_a_finetune_imagery")
    stage_b_tm = _tm("stage_b")
    stage_b_ft_tm = _tm("stage_b_finetune_imagery")
    stage_a_family_ch = {
        "lr": _ch("stage_a_lr"),
        "ff": _ch("stage_a_ff"),
    }
    stage_a_family_tm = {
        "lr": _tm("stage_a_lr"),
        "ff": _tm("stage_a_ff"),
    }
    stage_a_family_ft_ch = {
        "lr": _ch("stage_a_lr_finetune_imagery"),
        "ff": _ch("stage_a_ff_finetune_imagery"),
    }
    stage_a_family_ft_tm = {
        "lr": _tm("stage_a_lr_finetune_imagery"),
        "ff": _tm("stage_a_ff_finetune_imagery"),
    }
    stage_b_family_ch = {
        "lr": _ch("stage_b_lr"),
        "ff": _ch("stage_b_ff"),
    }
    stage_b_family_tm = {
        "lr": _tm("stage_b_lr"),
        "ff": _tm("stage_b_ff"),
    }
    stage_b_family_ft_ch = {
        "lr": _ch("stage_b_lr_finetune_imagery"),
        "ff": _ch("stage_b_ff_finetune_imagery"),
    }
    stage_b_family_ft_tm = {
        "lr": _tm("stage_b_lr_finetune_imagery"),
        "ff": _tm("stage_b_ff_finetune_imagery"),
    }
    test_subjects = list(split["test_9"])
    if X is None or y is None or meta is None:
        X, y, meta = load_eegmmidb_epochs(cfg, subjects=sorted(test_subjects))

    run_kinds = list(cfg.get("eval", {}).get("run_kinds", ["combined", "executed", "imagined"]))
    batch_size = int(cfg.get("train", {}).get("batch_size", 128))
    stage_b_task_type_masking = bool(cfg.get("eval", {}).get("stage_b_task_type_masking", True))
    stage_a_tta_time_shifts = [int(v) for v in (cfg.get("eval", {}).get("stage_a_tta_time_shifts", []) or [])]
    stage_a_adabn = cfg.get("eval", {}).get("stage_a_adabn", {}) or {}
    if stage_a_thresholds is None and cfg.get("eval", {}).get("calibrate_stage_a_thresholds", False):
        # Thresholds are normally fit during training and stored in checkpoint.
        stage_a_thresholds = {"default": 0.5, "by_run_kind": {}}
    if stage_a_thresholds_base is None:
        stage_a_thresholds_base = stage_a_thresholds
    if stage_a_thresholds_finetuned is None:
        stage_a_thresholds_finetuned = stage_a_thresholds_base

    results: dict[str, Any] = {"base": {}, "fine_tuned_imagery": None}

    for rk in run_kinds:
        res = evaluate_end_to_end(
            stage_a_model,
            stage_b_model,
            X,
            y,
            meta,
            run_kind=rk,
            subjects=test_subjects,
            device=device,
            batch_size=batch_size,
            stage_b_task_type_masking=stage_b_task_type_masking,
            stage_a_thresholds=stage_a_thresholds_base,
            stage_a_tta_time_shifts=stage_a_tta_time_shifts,
            stage_a_imagined_model=None,
            stage_a_channel_indices=stage_a_ch,
            stage_a_imagined_channel_indices=stage_a_ft_ch,
            stage_a_family_models=stage_a_family_models,
            stage_a_family_imagined_models=None,
            stage_a_family_channel_indices=stage_a_family_ch,
            stage_a_family_imagined_channel_indices=stage_a_family_ft_ch,
            stage_a_time_indices=stage_a_tm,
            stage_a_imagined_time_indices=stage_a_ft_tm,
            stage_a_family_time_indices=stage_a_family_tm,
            stage_a_family_imagined_time_indices=stage_a_family_ft_tm,
            stage_b_imagined_model=None,
            stage_b_channel_indices=stage_b_ch,
            stage_b_imagined_channel_indices=stage_b_ft_ch,
            stage_b_family_models=stage_b_family_models,
            stage_b_family_imagined_models=None,
            stage_b_family_channel_indices=stage_b_family_ch,
            stage_b_family_imagined_channel_indices=stage_b_family_ft_ch,
            stage_b_time_indices=stage_b_tm,
            stage_b_imagined_time_indices=stage_b_ft_tm,
            stage_b_family_time_indices=stage_b_family_tm,
            stage_b_family_imagined_time_indices=stage_b_family_ft_tm,
            stage_a_adabn=stage_a_adabn,
        )
        results["base"][rk] = res
        _save_confusions_for_result(res, out_dir)

    if (
        (stage_a_finetuned_model is not None)
        or (stage_b_finetuned_model is not None)
        or (stage_a_family_finetuned_models is not None)
        or (stage_b_family_finetuned_models is not None)
    ):
        ft_results = {}
        for rk in run_kinds:
            res = evaluate_end_to_end(
                stage_a_model,
                stage_b_model,
                X,
                y,
                meta,
                run_kind=rk,
                subjects=test_subjects,
                device=device,
                batch_size=batch_size,
                stage_b_task_type_masking=stage_b_task_type_masking,
                stage_a_thresholds=stage_a_thresholds_finetuned,
                stage_a_tta_time_shifts=stage_a_tta_time_shifts,
                stage_a_imagined_model=stage_a_finetuned_model,
                stage_a_channel_indices=stage_a_ch,
                stage_a_imagined_channel_indices=stage_a_ft_ch,
                stage_a_family_models=stage_a_family_models,
                stage_a_family_imagined_models=stage_a_family_finetuned_models,
                stage_a_family_channel_indices=stage_a_family_ch,
                stage_a_family_imagined_channel_indices=stage_a_family_ft_ch,
                stage_a_time_indices=stage_a_tm,
                stage_a_imagined_time_indices=stage_a_ft_tm,
                stage_a_family_time_indices=stage_a_family_tm,
                stage_a_family_imagined_time_indices=stage_a_family_ft_tm,
                stage_b_imagined_model=stage_b_finetuned_model,
                stage_b_channel_indices=stage_b_ch,
                stage_b_imagined_channel_indices=stage_b_ft_ch,
                stage_b_family_models=stage_b_family_models,
                stage_b_family_imagined_models=stage_b_family_finetuned_models,
                stage_b_family_channel_indices=stage_b_family_ch,
                stage_b_family_imagined_channel_indices=stage_b_family_ft_ch,
                stage_b_time_indices=stage_b_tm,
                stage_b_imagined_time_indices=stage_b_ft_tm,
                stage_b_family_time_indices=stage_b_family_tm,
                stage_b_family_imagined_time_indices=stage_b_family_ft_tm,
                stage_a_adabn=stage_a_adabn,
            )
            ft_results[rk] = res
        results["fine_tuned_imagery"] = ft_results

    # Sanity plots on combined held-out data
    X_test, y_test, meta_test = subset_arrays(X, y, meta, subjects=test_subjects, run_kind="combined", active_only=False)
    plots_cfg = cfg.get("eval", {}).get("plots", {})
    if plots_cfg.get("psd", True):
        try:
            save_psd_rest_vs_active(X_test, y_test, float(data_cfg["sfreq"]), out_dir / "psd_rest_vs_active.png")
        except Exception as e:
            print(f"[eval] PSD plot skipped due to error: {e}")
    if plots_cfg.get("erd_topomap", True):
        try:
            save_erd_topomap(X_test, y_test, list(data_cfg["channels"]), float(data_cfg["sfreq"]), out_dir / "erd_topomap.png")
        except Exception as e:
            print(f"[eval] ERD topomap skipped due to error: {e}")

    n_sal = int(plots_cfg.get("saliency_samples", 0) or 0)
    if n_sal > 0:
        try:
            ds_sal = EEGEpochDataset(
                X_test[: min(len(X_test), 128)],
                y_test[: min(len(y_test), 128)],
                meta_test.iloc[: min(len(meta_test), 128)].reset_index(drop=True),
                stage="stage_a",
            )
            save_saliency_examples(stage_a_model, ds_sal, out_dir / "saliency_stage_a", n_samples=n_sal, title_prefix="stage_a")
        except Exception as e:
            print(f"[eval] Saliency plot skipped due to error: {e}")

    metrics_path = out_dir / "evaluation_metrics.json"
    write_json(metrics_path, to_serializable(results))
    return results


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    *,
    data_root: str | None = None,
    output_dir: str | Path | None = None,
    device: str = "cpu",
    with_plots: bool = True,
) -> dict[str, Any]:
    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    if "config" not in ckpt or "split" not in ckpt or "stage_a" not in ckpt or "stage_b" not in ckpt:
        raise ValueError("Checkpoint must be a combined hierarchical checkpoint with config/split/stage states")

    cfg = ckpt["config"]
    if data_root is not None:
        cfg = {**cfg, "data": {**cfg["data"], "data_root": data_root}}
    if not with_plots:
        cfg = {**cfg, "eval": {**cfg.get("eval", {}), "plots": {"psd": False, "erd_topomap": False, "saliency_samples": 0}}}

    out_dir = Path(output_dir) if output_dir is not None else Path(checkpoint_path).resolve().parent / "eval"
    ensure_dir(out_dir)

    models = _load_models_from_combined_checkpoint(ckpt, device=device)
    results = evaluate_hierarchical_bundle(
        cfg=cfg,
        split=ckpt["split"],
        stage_a_model=models["stage_a"],
        stage_a_finetuned_model=models.get("stage_a_finetuned"),
        stage_a_family_models=models.get("stage_a_family"),
        stage_a_family_finetuned_models=models.get("stage_a_family_finetuned"),
        stage_b_model=models["stage_b"],
        stage_b_finetuned_model=models["stage_b_finetuned"],
        stage_b_family_models=models.get("stage_b_family"),
        stage_b_family_finetuned_models=models.get("stage_b_family_finetuned"),
        stage_a_thresholds=ckpt.get("stage_a_thresholds"),
        stage_a_thresholds_base=ckpt.get("stage_a_thresholds_base"),
        stage_a_thresholds_finetuned=ckpt.get("stage_a_thresholds_finetuned"),
        output_dir=out_dir,
        device=device,
    )

    # Learning curves from checkpoint histories if available
    if ckpt.get("stage_a", {}).get("history"):
        save_learning_curves(ckpt["stage_a"]["history"], out_dir / "learning_curve_stage_a.png", title="Stage A Learning Curves")
    if (ckpt.get("stage_a_finetuned") or {}).get("history"):
        save_learning_curves(ckpt["stage_a_finetuned"]["history"], out_dir / "learning_curve_stage_a_finetune.png", title="Stage A Fine-tune Learning Curves")
    stage_a_family_ckpt = ckpt.get("stage_a_family") or {}
    if stage_a_family_ckpt.get("lr", {}).get("history"):
        save_learning_curves(stage_a_family_ckpt["lr"]["history"], out_dir / "learning_curve_stage_a_lr.png", title="Stage A LR Learning Curves")
    if stage_a_family_ckpt.get("ff", {}).get("history"):
        save_learning_curves(stage_a_family_ckpt["ff"]["history"], out_dir / "learning_curve_stage_a_ff.png", title="Stage A FF Learning Curves")
    if ckpt.get("stage_b", {}).get("history"):
        save_learning_curves(ckpt["stage_b"]["history"], out_dir / "learning_curve_stage_b.png", title="Stage B Learning Curves")
    if (ckpt.get("stage_b_finetuned") or {}).get("history"):
        save_learning_curves(ckpt["stage_b_finetuned"]["history"], out_dir / "learning_curve_stage_b_finetune.png", title="Stage B Fine-tune Learning Curves")
    stage_b_family_ckpt = ckpt.get("stage_b_family") or {}
    if stage_b_family_ckpt.get("lr", {}).get("history"):
        save_learning_curves(stage_b_family_ckpt["lr"]["history"], out_dir / "learning_curve_stage_b_lr.png", title="Stage B LR Learning Curves")
    if stage_b_family_ckpt.get("ff", {}).get("history"):
        save_learning_curves(stage_b_family_ckpt["ff"]["history"], out_dir / "learning_curve_stage_b_ff.png", title="Stage B FF Learning Curves")
    return results
