from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

from eegmi.data.channel_selection import resolve_stage_channel_indices, resolve_stage_channel_names
from eegmi.data.dataset import subset_indices
from eegmi.data.loader import load_eegmmidb_epochs
from eegmi.data.time_selection import resolve_stage_time_indices
from eegmi.eval.evaluate import (
    _load_models_from_combined_checkpoint,
    _predict_logits_with_imagery_dispatch,
    _predict_stage_a_logits,
    _save_confusions_for_result,
    _stage_a_predictions_from_logits,
    evaluate_checkpoint,
)
from eegmi.eval.plots import save_learning_curves
from eegmi.train.metrics import classification_metrics
from eegmi.train.checkpointing import load_checkpoint, save_checkpoint
from eegmi.utils import ensure_dir, set_matplotlib_env, to_serializable, write_json


IMAGERY_STAGE_KEYS = [
    "stage_a_finetune_imagery",
    "stage_b_finetune_imagery",
    "stage_a_lr_finetune_imagery",
    "stage_a_ff_finetune_imagery",
    "stage_b_lr_finetune_imagery",
    "stage_b_ff_finetune_imagery",
]

EXECA_BASELINEB_STAGE_KEYS_FROM_SECONDARY = [
    "stage_a_finetune_imagery",
    "stage_b",
    "stage_b_finetune_imagery",
    "stage_b_lr",
    "stage_b_ff",
    "stage_b_lr_finetune_imagery",
    "stage_b_ff_finetune_imagery",
]

EXECA_BASELINEB_COMPONENT_KEYS_FROM_SECONDARY = [
    "stage_a_finetuned",
    "stage_a_family_finetuned",
    "stage_b",
    "stage_b_finetuned",
    "stage_b_family",
    "stage_b_family_finetuned",
]

SWAP_FF_STAGE_KEYS_FROM_SECONDARY = [
    "stage_b_ff",
    "stage_b_ff_finetune_imagery",
]

_GLOBAL_PREPROC_KEYS_FOR_SINGLE_X_HYBRID = [
    "channels",
    "sfreq",
    "tmin",
    "tmax",
    "baseline_window_len",
    "bandpass",
    "notch",
    "reref",
    "csd",
    "rest_zscore",
    "euclidean_alignment",
    "euclidean_alignment_time_window",
    "filter_bank_bands",
]


def _materialize_imagery_stage_overrides(base_cfg: dict[str, Any], imagery_cfg: dict[str, Any], imagery_ckpt: dict[str, Any]) -> None:
    base_data = base_cfg.setdefault("data", {})
    img_data = (imagery_cfg or {}).get("data", {}) or {}
    n_times = int(imagery_ckpt.get("n_times", img_data.get("baseline_window_len", 0) or 0))

    stage_time_windows = base_data.get("stage_time_windows")
    if not isinstance(stage_time_windows, dict):
        stage_time_windows = {}
        base_data["stage_time_windows"] = stage_time_windows

    stage_channels = base_data.get("stage_channels")
    if not isinstance(stage_channels, dict):
        stage_channels = {}
        base_data["stage_channels"] = stage_channels

    for stage_key in IMAGERY_STAGE_KEYS:
        try:
            _, tmeta = resolve_stage_time_indices(img_data, stage_key, n_times=n_times if n_times > 0 else None)
            stage_time_windows[stage_key] = [float(tmeta["tmin"]), float(tmeta["tmax"])]
        except Exception:
            pass
        try:
            ch_names = resolve_stage_channel_names(img_data, stage_key)
            if ch_names:
                stage_channels[stage_key] = list(ch_names)
        except Exception:
            pass


def _materialize_stage_overrides(
    dst_cfg: dict[str, Any],
    src_cfg: dict[str, Any],
    src_ckpt: dict[str, Any],
    stage_keys: list[str],
) -> None:
    dst_data = dst_cfg.setdefault("data", {})
    src_data = (src_cfg or {}).get("data", {}) or {}
    n_times = int(src_ckpt.get("n_times", src_data.get("baseline_window_len", 0) or 0))

    stage_time_windows = dst_data.get("stage_time_windows")
    if not isinstance(stage_time_windows, dict):
        stage_time_windows = {}
        dst_data["stage_time_windows"] = stage_time_windows

    stage_channels = dst_data.get("stage_channels")
    if not isinstance(stage_channels, dict):
        stage_channels = {}
        dst_data["stage_channels"] = stage_channels

    for stage_key in stage_keys:
        try:
            _, tmeta = resolve_stage_time_indices(src_data, stage_key, n_times=n_times if n_times > 0 else None)
            stage_time_windows[stage_key] = [float(tmeta["tmin"]), float(tmeta["tmax"])]
        except Exception:
            pass
        try:
            ch_names = resolve_stage_channel_names(src_data, stage_key)
            if ch_names:
                stage_channels[stage_key] = list(ch_names)
        except Exception:
            pass


def _split_hash(ckpt: dict[str, Any]) -> str | None:
    if "split_hash" in ckpt:
        return str(ckpt["split_hash"])
    split = ckpt.get("split") or {}
    return str(split) if split else None


def _merge_stage_dict(dst: dict[str, Any], src: dict[str, Any], key: str) -> None:
    if key in src and src[key] is not None:
        dst[key] = copy.deepcopy(src[key])


def _merge_family_entry(dst: dict[str, Any], src: dict[str, Any], key: str, family_key: str) -> None:
    src_map = src.get(key)
    if not isinstance(src_map, dict) or family_key not in src_map:
        return
    dst_map = dst.get(key)
    if not isinstance(dst_map, dict):
        dst_map = {}
        dst[key] = dst_map
    dst_map[family_key] = copy.deepcopy(src_map[family_key])


def _single_x_preproc_signature(ckpt: dict[str, Any]) -> dict[str, Any]:
    data_cfg = (ckpt.get("config") or {}).get("data", {}) or {}
    return {k: copy.deepcopy(data_cfg.get(k)) for k in _GLOBAL_PREPROC_KEYS_FOR_SINGLE_X_HYBRID}


def _resolve_eval_stage_indices(data_cfg: dict[str, Any], n_times: int) -> dict[str, Any]:
    def _ch(stage_key: str) -> np.ndarray:
        idx, _ = resolve_stage_channel_indices(data_cfg, stage_key)
        return idx
    def _tm(stage_key: str) -> np.ndarray:
        idx, _ = resolve_stage_time_indices(data_cfg, stage_key, n_times=n_times)
        return idx
    return {
        "stage_a_ch": _ch("stage_a"),
        "stage_a_ft_ch": _ch("stage_a_finetune_imagery"),
        "stage_b_lr_ch": _ch("stage_b_lr"),
        "stage_b_lr_ft_ch": _ch("stage_b_lr_finetune_imagery"),
        "stage_b_ff_ch": _ch("stage_b_ff"),
        "stage_b_ff_ft_ch": _ch("stage_b_ff_finetune_imagery"),
        "stage_a_tm": _tm("stage_a"),
        "stage_a_ft_tm": _tm("stage_a_finetune_imagery"),
        "stage_a_lr_tm": _tm("stage_a_lr"),
        "stage_a_lr_ft_tm": _tm("stage_a_lr_finetune_imagery"),
        "stage_a_ff_tm": _tm("stage_a_ff"),
        "stage_a_ff_ft_tm": _tm("stage_a_ff_finetune_imagery"),
        "stage_a_lr_ch": _ch("stage_a_lr"),
        "stage_a_lr_ft_ch": _ch("stage_a_lr_finetune_imagery"),
        "stage_a_ff_ch": _ch("stage_a_ff"),
        "stage_a_ff_ft_ch": _ch("stage_a_ff_finetune_imagery"),
        "stage_b_lr_tm": _tm("stage_b_lr"),
        "stage_b_lr_ft_tm": _tm("stage_b_lr_finetune_imagery"),
        "stage_b_ff_tm": _tm("stage_b_ff"),
        "stage_b_ff_ft_tm": _tm("stage_b_ff_finetune_imagery"),
    }


def _load_test_data_for_ckpt(ckpt: dict[str, Any], data_root: str | None):
    cfg = copy.deepcopy(ckpt["config"])
    if data_root is not None:
        cfg["data"] = {**cfg.get("data", {}), "data_root": data_root}
    split = ckpt["split"]
    test_subjects = sorted(split["test_9"])
    X, y, meta = load_eegmmidb_epochs(cfg, subjects=test_subjects)
    return cfg, split, test_subjects, X, y, meta


def _assert_aligned_rows(
    y_a: np.ndarray,
    meta_a,
    y_b: np.ndarray,
    meta_b,
    *,
    label: str,
) -> None:
    if y_a.shape != y_b.shape or not np.array_equal(y_a, y_b):
        raise ValueError(f"{label}: y arrays are not aligned; cannot multisource-compose safely")
    if len(meta_a) != len(meta_b):
        raise ValueError(f"{label}: meta lengths differ")
    cols = ["subject", "run", "run_kind", "task_type", "label", "event_desc"]
    for c in cols:
        a = meta_a[c].astype(str).to_numpy() if c != "label" else meta_a[c].to_numpy()
        b = meta_b[c].astype(str).to_numpy() if c != "label" else meta_b[c].to_numpy()
        if not np.array_equal(a, b):
            raise ValueError(f"{label}: meta column '{c}' is not aligned")


def _predict_family_binary_global(
    *,
    family_key: str,
    X: np.ndarray,
    y_global: np.ndarray,
    meta_global,
    base_model,
    imagery_model,
    idx_cfg: dict[str, Any],
    batch_size: int,
    device: str,
) -> np.ndarray:
    if family_key == "lr":
        stage = "stage_b_lr"
        labels_valid = (1, 2)
        offset = 1
        ch_key = "stage_b_lr_ch"
        ch_ft_key = "stage_b_lr_ft_ch"
        tm_key = "stage_b_lr_tm"
        tm_ft_key = "stage_b_lr_ft_tm"
        y_stage = np.where(np.isin(y_global, labels_valid), y_global, 1).astype(np.int64)
    elif family_key == "ff":
        stage = "stage_b_ff"
        labels_valid = (3, 4)
        offset = 3
        ch_key = "stage_b_ff_ch"
        ch_ft_key = "stage_b_ff_ft_ch"
        tm_key = "stage_b_ff_tm"
        tm_ft_key = "stage_b_ff_ft_tm"
        y_stage = np.where(np.isin(y_global, labels_valid), y_global, 3).astype(np.int64)
    else:
        raise ValueError(f"Unknown family_key: {family_key}")
    _, logits, _ = _predict_logits_with_imagery_dispatch(
        base_model,
        imagery_model,
        X,
        y_stage,
        meta_global,
        stage=stage,
        batch_size=batch_size,
        device=device,
        indices=None,
        channel_indices=idx_cfg[ch_key],
        imagery_channel_indices=idx_cfg[ch_ft_key],
        time_indices=idx_cfg[tm_key],
        imagery_time_indices=idx_cfg[tm_ft_key],
    )
    pred_local = np.argmax(logits, axis=1).astype(np.int64)
    return pred_local + offset


def _evaluate_end_to_end_swap_ff_multisource(
    *,
    base_models: dict[str, Any],
    ff_models: dict[str, Any],
    base_cfg: dict[str, Any],
    ff_cfg: dict[str, Any],
    stage_a_thresholds: dict[str, Any] | None,
    X_base: np.ndarray,
    y_base: np.ndarray,
    meta_base,
    X_ff: np.ndarray,
    y_ff: np.ndarray,
    meta_ff,
    run_kind: str,
    subjects: list[str],
    device: str,
    use_imagery_ft: bool,
) -> dict[str, Any]:
    from eegmi.constants import LABEL_TO_NAME

    idx_sub = subset_indices(y_base, meta_base, subjects=subjects, run_kind=run_kind, active_only=False)
    y_sub = y_base[idx_sub]
    meta_sub = meta_base.iloc[idx_sub].reset_index(drop=True)
    X_sub_base = X_base[idx_sub]
    X_sub_ff = X_ff[idx_sub]
    y_sub_ff = y_ff[idx_sub]
    meta_sub_ff = meta_ff.iloc[idx_sub].reset_index(drop=True)
    _assert_aligned_rows(y_sub, meta_sub, y_sub_ff, meta_sub_ff, label=f"subset({run_kind})")

    eval_cfg = base_cfg.get("eval", {}) or {}
    batch_size = int((base_cfg.get("train", {}) or {}).get("batch_size", 128))
    tta = [int(v) for v in (eval_cfg.get("stage_a_tta_time_shifts", []) or [])]

    base_idx = _resolve_eval_stage_indices(base_cfg["data"], n_times=int(X_base.shape[2]))
    ff_idx = _resolve_eval_stage_indices(ff_cfg["data"], n_times=int(X_ff.shape[2]))

    stage_a_ft = base_models.get("stage_a_finetuned") if use_imagery_ft else None
    stage_a_family_ft = base_models.get("stage_a_family_finetuned") if use_imagery_ft else None
    y_true_a, logits_a, _ = _predict_stage_a_logits(
        base_models["stage_a"],
        stage_a_ft,
        base_models.get("stage_a_family"),
        stage_a_family_ft,
        X_sub_base,
        y_sub,
        meta_sub,
        batch_size=batch_size,
        device=device,
        indices=None,
        tta_time_shifts=tta,
        stage_a_channel_indices=base_idx["stage_a_ch"],
        stage_a_imagined_channel_indices=base_idx["stage_a_ft_ch"],
        stage_a_family_channel_indices={"lr": base_idx["stage_a_lr_ch"], "ff": base_idx["stage_a_ff_ch"]},
        stage_a_family_imagined_channel_indices={"lr": base_idx["stage_a_lr_ft_ch"], "ff": base_idx["stage_a_ff_ft_ch"]},
        stage_a_time_indices=base_idx["stage_a_tm"],
        stage_a_imagined_time_indices=base_idx["stage_a_ft_tm"],
        stage_a_family_time_indices={"lr": base_idx["stage_a_lr_tm"], "ff": base_idx["stage_a_ff_tm"]},
        stage_a_family_imagined_time_indices={"lr": base_idx["stage_a_lr_ft_tm"], "ff": base_idx["stage_a_ff_ft_tm"]},
    )
    y_pred_a = _stage_a_predictions_from_logits(logits_a, meta_sub, eval_run_kind=run_kind, stage_a_thresholds=stage_a_thresholds)

    final_pred = np.zeros_like(y_sub, dtype=np.int64)
    pred_active_mask = y_pred_a.astype(bool)
    if pred_active_mask.any():
        task_arr = meta_sub["task_type"].astype(str).str.upper().to_numpy()
        run_arr = meta_sub["run_kind"].astype(str).to_numpy()
        idx_act_local = np.flatnonzero(pred_active_mask)
        idx_lr = idx_act_local[task_arr[pred_active_mask] == "LR"]
        idx_ff = idx_act_local[task_arr[pred_active_mask] == "FF"]

        stage_b_family_base = base_models.get("stage_b_family") or {}
        stage_b_family_base_ft = (base_models.get("stage_b_family_finetuned") or {}) if use_imagery_ft else {}
        stage_b_family_ff = ff_models.get("stage_b_family") or {}
        stage_b_family_ff_ft = (ff_models.get("stage_b_family_finetuned") or {}) if use_imagery_ft else {}

        if idx_lr.size > 0:
            pred_lr = _predict_family_binary_global(
                family_key="lr",
                X=X_sub_base[idx_lr],
                y_global=y_sub[idx_lr],
                meta_global=meta_sub.iloc[idx_lr].reset_index(drop=True),
                base_model=stage_b_family_base["lr"],
                imagery_model=stage_b_family_base_ft.get("lr"),
                idx_cfg=base_idx,
                batch_size=batch_size,
                device=device,
            )
            final_pred[idx_lr] = pred_lr
        if idx_ff.size > 0:
            pred_ff = _predict_family_binary_global(
                family_key="ff",
                X=X_sub_ff[idx_ff],
                y_global=y_sub_ff[idx_ff],
                meta_global=meta_sub_ff.iloc[idx_ff].reset_index(drop=True),
                base_model=stage_b_family_ff["ff"],
                imagery_model=stage_b_family_ff_ft.get("ff"),
                idx_cfg=ff_idx,
                batch_size=batch_size,
                device=device,
            )
            final_pred[idx_ff] = pred_ff

    metrics_5 = classification_metrics(
        y_sub,
        final_pred,
        labels=[0, 1, 2, 3, 4],
        label_names=[LABEL_TO_NAME[i] for i in [0, 1, 2, 3, 4]],
    ).to_dict()
    stage_a_metrics = classification_metrics(y_true_a, y_pred_a, labels=[0, 1], label_names=["REST", "ACTIVE"]).to_dict()

    true_active_mask = y_sub > 0
    stage_b_metrics = {}
    if true_active_mask.any():
        idx_true_act = np.flatnonzero(true_active_mask)
        task_arr_true = meta_sub["task_type"].astype(str).str.upper().to_numpy()[true_active_mask]
        idx_lr = idx_true_act[task_arr_true == "LR"]
        idx_ff = idx_true_act[task_arr_true == "FF"]
        y_true_b = (y_sub[true_active_mask] - 1).astype(np.int64)
        y_pred_b_global = np.zeros_like(y_sub[true_active_mask], dtype=np.int64)
        # Map back into compressed active subset coordinates.
        act_local_index = {int(orig_idx): pos for pos, orig_idx in enumerate(idx_true_act.tolist())}
        stage_b_family_base = base_models.get("stage_b_family") or {}
        stage_b_family_base_ft = (base_models.get("stage_b_family_finetuned") or {}) if use_imagery_ft else {}
        stage_b_family_ff = ff_models.get("stage_b_family") or {}
        stage_b_family_ff_ft = (ff_models.get("stage_b_family_finetuned") or {}) if use_imagery_ft else {}
        if idx_lr.size > 0:
            pred_lr = _predict_family_binary_global(
                family_key="lr",
                X=X_sub_base[idx_lr],
                y_global=y_sub[idx_lr],
                meta_global=meta_sub.iloc[idx_lr].reset_index(drop=True),
                base_model=stage_b_family_base["lr"],
                imagery_model=stage_b_family_base_ft.get("lr"),
                idx_cfg=base_idx,
                batch_size=batch_size,
                device=device,
            )
            for i_local, pred in zip(idx_lr.tolist(), pred_lr.tolist()):
                y_pred_b_global[act_local_index[int(i_local)]] = int(pred)
        if idx_ff.size > 0:
            pred_ff = _predict_family_binary_global(
                family_key="ff",
                X=X_sub_ff[idx_ff],
                y_global=y_sub_ff[idx_ff],
                meta_global=meta_sub_ff.iloc[idx_ff].reset_index(drop=True),
                base_model=stage_b_family_ff["ff"],
                imagery_model=stage_b_family_ff_ft.get("ff"),
                idx_cfg=ff_idx,
                batch_size=batch_size,
                device=device,
            )
            for i_local, pred in zip(idx_ff.tolist(), pred_ff.tolist()):
                y_pred_b_global[act_local_index[int(i_local)]] = int(pred)
        y_pred_b_true_active = (y_pred_b_global - 1).astype(np.int64)
        stage_b_metrics = classification_metrics(
            y_true_b,
            y_pred_b_true_active,
            labels=[0, 1, 2, 3],
            label_names=["LEFT", "RIGHT", "FISTS", "FEET"],
        ).to_dict()

    return {
        "run_kind": run_kind,
        "stage_b_task_type_masking": True,
        "stage_a_family_heads": bool(base_models.get("stage_a_family")),
        "stage_b_family_heads": True,
        "stage_a_thresholds": to_serializable(stage_a_thresholds) if stage_a_thresholds else None,
        "n_samples": int(len(y_sub)),
        "stage_a": {"metrics": stage_a_metrics, "y_true": y_true_a.tolist(), "y_pred": y_pred_a.tolist()},
        "stage_b_true_active": {"metrics": stage_b_metrics},
        "end_to_end": {"metrics": metrics_5, "y_true": y_sub.astype(int).tolist(), "y_pred": final_pred.astype(int).tolist()},
    }


def evaluate_swap_ff_branch_multisource(
    *,
    base_ckpt: dict[str, Any],
    ff_ckpt: dict[str, Any],
    data_root: str | None,
    out_dir: str | Path,
    device: str = "cpu",
    with_plots: bool = False,
) -> dict[str, Any]:
    if base_ckpt.get("split") != ff_ckpt.get("split"):
        raise ValueError("Checkpoint splits differ; cannot hybridize safely")
    out_dir = ensure_dir(out_dir)
    base_cfg, split, test_subjects, X_base, y_base, meta_base = _load_test_data_for_ckpt(base_ckpt, data_root)
    ff_cfg, _, _, X_ff, y_ff, meta_ff = _load_test_data_for_ckpt(ff_ckpt, data_root)
    _assert_aligned_rows(y_base, meta_base, y_ff, meta_ff, label="test set")

    base_models = _load_models_from_combined_checkpoint(base_ckpt, device=device)
    ff_models = _load_models_from_combined_checkpoint(ff_ckpt, device=device)
    if not (base_models.get("stage_b_family") and ff_models.get("stage_b_family")):
        raise ValueError("swap_ff_branch_multisource requires family Stage B heads in both checkpoints")

    run_kinds = list((base_cfg.get("eval", {}) or {}).get("run_kinds", ["combined", "executed", "imagined"]))
    stage_a_thresholds = copy.deepcopy(base_ckpt.get("stage_a_thresholds"))
    results: dict[str, Any] = {"base": {}, "fine_tuned_imagery": None}
    for rk in run_kinds:
        res = _evaluate_end_to_end_swap_ff_multisource(
            base_models=base_models,
            ff_models=ff_models,
            base_cfg=base_cfg,
            ff_cfg=ff_cfg,
            stage_a_thresholds=stage_a_thresholds,
            X_base=X_base,
            y_base=y_base,
            meta_base=meta_base,
            X_ff=X_ff,
            y_ff=y_ff,
            meta_ff=meta_ff,
            run_kind=rk,
            subjects=test_subjects,
            device=device,
            use_imagery_ft=False,
        )
        results["base"][rk] = res
        _save_confusions_for_result(res, Path(out_dir))

    has_any_ft = any([
        base_models.get("stage_a_finetuned") is not None,
        base_models.get("stage_a_family_finetuned") is not None,
        base_models.get("stage_b_finetuned") is not None,
        base_models.get("stage_b_family_finetuned") is not None,
        ff_models.get("stage_b_family_finetuned") is not None,
    ])
    if has_any_ft:
        ft_res = {}
        for rk in run_kinds:
            res = _evaluate_end_to_end_swap_ff_multisource(
                base_models=base_models,
                ff_models=ff_models,
                base_cfg=base_cfg,
                ff_cfg=ff_cfg,
                stage_a_thresholds=stage_a_thresholds,
                X_base=X_base,
                y_base=y_base,
                meta_base=meta_base,
                X_ff=X_ff,
                y_ff=y_ff,
                meta_ff=meta_ff,
                run_kind=rk,
                subjects=test_subjects,
                device=device,
                use_imagery_ft=True,
            )
            ft_res[rk] = res
        results["fine_tuned_imagery"] = ft_res

    write_json(Path(out_dir) / "evaluation_metrics.json", to_serializable(results))
    if with_plots:
        # Learning curves are checkpoint-local; save if present for convenience.
        if base_ckpt.get("stage_a", {}).get("history"):
            save_learning_curves(base_ckpt["stage_a"]["history"], Path(out_dir) / "learning_curve_stage_a_base.png", title="Base Stage A")
        if ff_ckpt.get("stage_b_family", {}).get("ff", {}).get("history"):
            save_learning_curves(ff_ckpt["stage_b_family"]["ff"]["history"], Path(out_dir) / "learning_curve_stage_b_ff_source.png", title="FF Source Stage B FF")
    return results


def _predict_stage_b_family_one_source(
    *,
    models: dict[str, Any],
    cfg: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    meta,
    batch_size: int,
    device: str,
    use_imagery_ft: bool,
) -> np.ndarray:
    idx_cfg = _resolve_eval_stage_indices(cfg["data"], n_times=int(X.shape[2]))
    fam_base = models.get("stage_b_family") or {}
    fam_ft = (models.get("stage_b_family_finetuned") or {}) if use_imagery_ft else {}
    task_arr = meta["task_type"].astype(str).str.upper().to_numpy()
    out = np.zeros((len(y),), dtype=np.int64)
    idx_lr = np.flatnonzero(task_arr == "LR")
    idx_ff = np.flatnonzero(task_arr == "FF")
    if idx_lr.size > 0:
        out[idx_lr] = _predict_family_binary_global(
            family_key="lr",
            X=X[idx_lr],
            y_global=y[idx_lr],
            meta_global=meta.iloc[idx_lr].reset_index(drop=True),
            base_model=fam_base["lr"],
            imagery_model=fam_ft.get("lr"),
            idx_cfg=idx_cfg,
            batch_size=batch_size,
            device=device,
        )
    if idx_ff.size > 0:
        out[idx_ff] = _predict_family_binary_global(
            family_key="ff",
            X=X[idx_ff],
            y_global=y[idx_ff],
            meta_global=meta.iloc[idx_ff].reset_index(drop=True),
            base_model=fam_base["ff"],
            imagery_model=fam_ft.get("ff"),
            idx_cfg=idx_cfg,
            batch_size=batch_size,
            device=device,
        )
    return out


def _evaluate_end_to_end_exec_a_baseline_b_multisource(
    *,
    stagea_models: dict[str, Any],
    stagea_cfg: dict[str, Any],
    stagea_thresholds: dict[str, Any] | None,
    X_stagea: np.ndarray,
    y_stagea: np.ndarray,
    meta_stagea,
    base_models: dict[str, Any],
    base_cfg: dict[str, Any],
    X_base: np.ndarray,
    y_base: np.ndarray,
    meta_base,
    run_kind: str,
    subjects: list[str],
    device: str,
    use_imagery_ft: bool,
) -> dict[str, Any]:
    from eegmi.constants import LABEL_TO_NAME

    idx_sub = subset_indices(y_base, meta_base, subjects=subjects, run_kind=run_kind, active_only=False)
    y_sub = y_base[idx_sub]
    meta_sub = meta_base.iloc[idx_sub].reset_index(drop=True)
    X_sub_base = X_base[idx_sub]
    y_sub_stagea = y_stagea[idx_sub]
    meta_sub_stagea = meta_stagea.iloc[idx_sub].reset_index(drop=True)
    X_sub_stagea = X_stagea[idx_sub]
    _assert_aligned_rows(y_sub, meta_sub, y_sub_stagea, meta_sub_stagea, label=f"subset({run_kind})")

    batch_size = int((base_cfg.get("train", {}) or {}).get("batch_size", 128))
    tta = [int(v) for v in ((stagea_cfg.get("eval", {}) or {}).get("stage_a_tta_time_shifts", []) or [])]
    stagea_idx = _resolve_eval_stage_indices(stagea_cfg["data"], n_times=int(X_stagea.shape[2]))

    stage_a_ft = stagea_models.get("stage_a_finetuned") if use_imagery_ft else None
    stage_a_family_ft = stagea_models.get("stage_a_family_finetuned") if use_imagery_ft else None
    y_true_a, logits_a, _ = _predict_stage_a_logits(
        stagea_models["stage_a"],
        stage_a_ft,
        stagea_models.get("stage_a_family"),
        stage_a_family_ft,
        X_sub_stagea,
        y_sub_stagea,
        meta_sub_stagea,
        batch_size=batch_size,
        device=device,
        indices=None,
        tta_time_shifts=tta,
        stage_a_channel_indices=stagea_idx["stage_a_ch"],
        stage_a_imagined_channel_indices=stagea_idx["stage_a_ft_ch"],
        stage_a_family_channel_indices={"lr": stagea_idx["stage_a_lr_ch"], "ff": stagea_idx["stage_a_ff_ch"]},
        stage_a_family_imagined_channel_indices={"lr": stagea_idx["stage_a_lr_ft_ch"], "ff": stagea_idx["stage_a_ff_ft_ch"]},
        stage_a_time_indices=stagea_idx["stage_a_tm"],
        stage_a_imagined_time_indices=stagea_idx["stage_a_ft_tm"],
        stage_a_family_time_indices={"lr": stagea_idx["stage_a_lr_tm"], "ff": stagea_idx["stage_a_ff_tm"]},
        stage_a_family_imagined_time_indices={"lr": stagea_idx["stage_a_lr_ft_tm"], "ff": stagea_idx["stage_a_ff_ft_tm"]},
    )
    y_pred_a = _stage_a_predictions_from_logits(logits_a, meta_sub_stagea, eval_run_kind=run_kind, stage_a_thresholds=stagea_thresholds)

    final_pred = np.zeros_like(y_sub, dtype=np.int64)
    pred_active_mask = y_pred_a.astype(bool)
    if pred_active_mask.any():
        y_pred_b_global = _predict_stage_b_family_one_source(
            models=base_models,
            cfg=base_cfg,
            X=X_sub_base[pred_active_mask],
            y=y_sub[pred_active_mask],
            meta=meta_sub.iloc[np.flatnonzero(pred_active_mask)].reset_index(drop=True),
            batch_size=batch_size,
            device=device,
            use_imagery_ft=use_imagery_ft,
        )
        final_pred[pred_active_mask] = y_pred_b_global

    metrics_5 = classification_metrics(
        y_sub,
        final_pred,
        labels=[0, 1, 2, 3, 4],
        label_names=[LABEL_TO_NAME[i] for i in [0, 1, 2, 3, 4]],
    ).to_dict()
    stage_a_metrics = classification_metrics(y_true_a, y_pred_a, labels=[0, 1], label_names=["REST", "ACTIVE"]).to_dict()

    stage_b_metrics = {}
    true_active_mask = y_sub > 0
    if true_active_mask.any():
        y_true_b = (y_sub[true_active_mask] - 1).astype(np.int64)
        y_pred_b_global = _predict_stage_b_family_one_source(
            models=base_models,
            cfg=base_cfg,
            X=X_sub_base[true_active_mask],
            y=y_sub[true_active_mask],
            meta=meta_sub.iloc[np.flatnonzero(true_active_mask)].reset_index(drop=True),
            batch_size=batch_size,
            device=device,
            use_imagery_ft=use_imagery_ft,
        )
        y_pred_b_true_active = (y_pred_b_global - 1).astype(np.int64)
        stage_b_metrics = classification_metrics(
            y_true_b,
            y_pred_b_true_active,
            labels=[0, 1, 2, 3],
            label_names=["LEFT", "RIGHT", "FISTS", "FEET"],
        ).to_dict()

    return {
        "run_kind": run_kind,
        "stage_b_task_type_masking": True,
        "stage_a_family_heads": bool(stagea_models.get("stage_a_family")),
        "stage_b_family_heads": bool(base_models.get("stage_b_family")),
        "stage_a_thresholds": to_serializable(stagea_thresholds) if stagea_thresholds else None,
        "n_samples": int(len(y_sub)),
        "stage_a": {"metrics": stage_a_metrics, "y_true": y_true_a.tolist(), "y_pred": y_pred_a.tolist()},
        "stage_b_true_active": {"metrics": stage_b_metrics},
        "end_to_end": {"metrics": metrics_5, "y_true": y_sub.astype(int).tolist(), "y_pred": final_pred.astype(int).tolist()},
    }


def evaluate_exec_a_baseline_b_multisource(
    *,
    exec_ckpt: dict[str, Any],
    base_ckpt: dict[str, Any],
    data_root: str | None,
    out_dir: str | Path,
    device: str = "cpu",
    with_plots: bool = False,
) -> dict[str, Any]:
    if exec_ckpt.get("split") != base_ckpt.get("split"):
        raise ValueError("Checkpoint splits differ; cannot hybridize safely")
    out_dir = ensure_dir(out_dir)
    base_cfg, split, test_subjects, X_base, y_base, meta_base = _load_test_data_for_ckpt(base_ckpt, data_root)
    exec_cfg, _, _, X_exec, y_exec, meta_exec = _load_test_data_for_ckpt(exec_ckpt, data_root)
    _assert_aligned_rows(y_base, meta_base, y_exec, meta_exec, label="test set")

    base_models = _load_models_from_combined_checkpoint(base_ckpt, device=device)
    exec_models = _load_models_from_combined_checkpoint(exec_ckpt, device=device)
    run_kinds = list((base_cfg.get("eval", {}) or {}).get("run_kinds", ["combined", "executed", "imagined"]))
    stage_a_thresholds = copy.deepcopy(exec_ckpt.get("stage_a_thresholds"))

    results: dict[str, Any] = {"base": {}, "fine_tuned_imagery": None}
    for rk in run_kinds:
        t0 = time.perf_counter()
        print(f"[hybrid] exec_a_baseline_b_multisource base {rk} start", flush=True)
        res = _evaluate_end_to_end_exec_a_baseline_b_multisource(
            stagea_models=exec_models,
            stagea_cfg=exec_cfg,
            stagea_thresholds=stage_a_thresholds,
            X_stagea=X_exec,
            y_stagea=y_exec,
            meta_stagea=meta_exec,
            base_models=base_models,
            base_cfg=base_cfg,
            X_base=X_base,
            y_base=y_base,
            meta_base=meta_base,
            run_kind=rk,
            subjects=test_subjects,
            device=device,
            use_imagery_ft=False,
        )
        results["base"][rk] = res
        print(f"[hybrid] exec_a_baseline_b_multisource base {rk} done in {time.perf_counter()-t0:.1f}s", flush=True)
        if with_plots:
            _save_confusions_for_result(res, Path(out_dir))

    has_any_ft = any([
        exec_models.get("stage_a_finetuned") is not None,
        exec_models.get("stage_a_family_finetuned") is not None,
        base_models.get("stage_b_finetuned") is not None,
        base_models.get("stage_b_family_finetuned") is not None,
    ])
    if has_any_ft:
        ft_res = {}
        for rk in run_kinds:
            t0 = time.perf_counter()
            print(f"[hybrid] exec_a_baseline_b_multisource ft {rk} start", flush=True)
            res = _evaluate_end_to_end_exec_a_baseline_b_multisource(
                stagea_models=exec_models,
                stagea_cfg=exec_cfg,
                stagea_thresholds=stage_a_thresholds,
                X_stagea=X_exec,
                y_stagea=y_exec,
                meta_stagea=meta_exec,
                base_models=base_models,
                base_cfg=base_cfg,
                X_base=X_base,
                y_base=y_base,
                meta_base=meta_base,
                run_kind=rk,
                subjects=test_subjects,
                device=device,
                use_imagery_ft=True,
            )
            ft_res[rk] = res
            print(f"[hybrid] exec_a_baseline_b_multisource ft {rk} done in {time.perf_counter()-t0:.1f}s", flush=True)
        results["fine_tuned_imagery"] = ft_res

    write_json(Path(out_dir) / "evaluation_metrics.json", to_serializable(results))
    return results


def _merge_stage_a_thresholds_runwise(base_thr: dict[str, Any] | None, img_thr: dict[str, Any] | None) -> dict[str, Any] | None:
    if not base_thr and not img_thr:
        return None
    out = copy.deepcopy(base_thr or {})
    if "default" not in out and img_thr and "default" in img_thr:
        out["default"] = img_thr["default"]
    out.setdefault("by_run_kind", {})
    out.setdefault("by_run_kind_task", {})
    if img_thr:
        if "imagined" in (img_thr.get("by_run_kind") or {}):
            out["by_run_kind"]["imagined"] = copy.deepcopy(img_thr["by_run_kind"]["imagined"])
        for k, v in (img_thr.get("by_run_kind_task") or {}).items():
            if str(k).startswith("imagined:"):
                out["by_run_kind_task"][k] = copy.deepcopy(v)
    return out


def _evaluate_end_to_end_stagea_imagery_multisource(
    *,
    base_models: dict[str, Any],
    base_cfg: dict[str, Any],
    source_models: dict[str, Any],
    source_cfg: dict[str, Any],
    stage_a_thresholds_mixed: dict[str, Any] | None,
    X_base: np.ndarray,
    y_base: np.ndarray,
    meta_base,
    X_src: np.ndarray,
    y_src: np.ndarray,
    meta_src,
    run_kind: str,
    subjects: list[str],
    device: str,
    use_imagery_ft: bool,
) -> dict[str, Any]:
    from eegmi.constants import LABEL_TO_NAME

    idx_sub = subset_indices(y_base, meta_base, subjects=subjects, run_kind=run_kind, active_only=False)
    y_sub = y_base[idx_sub]
    meta_sub = meta_base.iloc[idx_sub].reset_index(drop=True)
    X_sub_base = X_base[idx_sub]
    y_sub_src = y_src[idx_sub]
    meta_sub_src = meta_src.iloc[idx_sub].reset_index(drop=True)
    X_sub_src = X_src[idx_sub]
    _assert_aligned_rows(y_sub, meta_sub, y_sub_src, meta_sub_src, label=f"subset({run_kind})")

    batch_size = int((base_cfg.get("train", {}) or {}).get("batch_size", 128))
    tta_base = [int(v) for v in ((base_cfg.get("eval", {}) or {}).get("stage_a_tta_time_shifts", []) or [])]
    tta_src = [int(v) for v in ((source_cfg.get("eval", {}) or {}).get("stage_a_tta_time_shifts", []) or [])]
    base_idx = _resolve_eval_stage_indices(base_cfg["data"], n_times=int(X_base.shape[2]))
    src_idx = _resolve_eval_stage_indices(source_cfg["data"], n_times=int(X_src.shape[2]))

    # Base Stage A for all samples (no imagery FT overwrite here).
    y_true_a, logits_a, _ = _predict_stage_a_logits(
        base_models["stage_a"],
        None,
        base_models.get("stage_a_family"),
        None,
        X_sub_base,
        y_sub,
        meta_sub,
        batch_size=batch_size,
        device=device,
        indices=None,
        tta_time_shifts=tta_base,
        stage_a_channel_indices=base_idx["stage_a_ch"],
        stage_a_imagined_channel_indices=base_idx["stage_a_ft_ch"],
        stage_a_family_channel_indices={"lr": base_idx["stage_a_lr_ch"], "ff": base_idx["stage_a_ff_ch"]},
        stage_a_family_imagined_channel_indices={"lr": base_idx["stage_a_lr_ft_ch"], "ff": base_idx["stage_a_ff_ft_ch"]},
        stage_a_time_indices=base_idx["stage_a_tm"],
        stage_a_imagined_time_indices=base_idx["stage_a_ft_tm"],
        stage_a_family_time_indices={"lr": base_idx["stage_a_lr_tm"], "ff": base_idx["stage_a_ff_tm"]},
        stage_a_family_imagined_time_indices={"lr": base_idx["stage_a_lr_ft_tm"], "ff": base_idx["stage_a_ff_ft_tm"]},
    )
    # Source Stage A for imagined samples only (includes source imagery FT dispatch if requested).
    imag_mask = (meta_sub["run_kind"].astype(str).to_numpy() == "imagined")
    if imag_mask.any():
        src_stage_a_ft = source_models.get("stage_a_finetuned") if use_imagery_ft else None
        src_stage_a_family_ft = source_models.get("stage_a_family_finetuned") if use_imagery_ft else None
        _, logits_img, _ = _predict_stage_a_logits(
            source_models["stage_a"],
            src_stage_a_ft,
            source_models.get("stage_a_family"),
            src_stage_a_family_ft,
            X_sub_src,
            y_sub_src,
            meta_sub_src,
            batch_size=batch_size,
            device=device,
            indices=np.flatnonzero(imag_mask),
            tta_time_shifts=tta_src,
            stage_a_channel_indices=src_idx["stage_a_ch"],
            stage_a_imagined_channel_indices=src_idx["stage_a_ft_ch"],
            stage_a_family_channel_indices={"lr": src_idx["stage_a_lr_ch"], "ff": src_idx["stage_a_ff_ch"]},
            stage_a_family_imagined_channel_indices={"lr": src_idx["stage_a_lr_ft_ch"], "ff": src_idx["stage_a_ff_ft_ch"]},
            stage_a_time_indices=src_idx["stage_a_tm"],
            stage_a_imagined_time_indices=src_idx["stage_a_ft_tm"],
            stage_a_family_time_indices={"lr": src_idx["stage_a_lr_tm"], "ff": src_idx["stage_a_ff_tm"]},
            stage_a_family_imagined_time_indices={"lr": src_idx["stage_a_lr_ft_tm"], "ff": src_idx["stage_a_ff_ft_tm"]},
        )
        logits_a[imag_mask] = logits_img

    y_pred_a = _stage_a_predictions_from_logits(logits_a, meta_sub, eval_run_kind=run_kind, stage_a_thresholds=stage_a_thresholds_mixed)

    final_pred = np.zeros_like(y_sub, dtype=np.int64)
    pred_active_mask = y_pred_a.astype(bool)
    if pred_active_mask.any():
        y_pred_b_global = _predict_stage_b_family_one_source(
            models=base_models,
            cfg=base_cfg,
            X=X_sub_base[pred_active_mask],
            y=y_sub[pred_active_mask],
            meta=meta_sub.iloc[np.flatnonzero(pred_active_mask)].reset_index(drop=True),
            batch_size=batch_size,
            device=device,
            use_imagery_ft=use_imagery_ft,
        )
        final_pred[pred_active_mask] = y_pred_b_global

    metrics_5 = classification_metrics(
        y_sub,
        final_pred,
        labels=[0, 1, 2, 3, 4],
        label_names=[LABEL_TO_NAME[i] for i in [0, 1, 2, 3, 4]],
    ).to_dict()
    stage_a_metrics = classification_metrics(y_true_a, y_pred_a, labels=[0, 1], label_names=["REST", "ACTIVE"]).to_dict()

    stage_b_metrics = {}
    true_active_mask = y_sub > 0
    if true_active_mask.any():
        y_true_b = (y_sub[true_active_mask] - 1).astype(np.int64)
        y_pred_b_global = _predict_stage_b_family_one_source(
            models=base_models,
            cfg=base_cfg,
            X=X_sub_base[true_active_mask],
            y=y_sub[true_active_mask],
            meta=meta_sub.iloc[np.flatnonzero(true_active_mask)].reset_index(drop=True),
            batch_size=batch_size,
            device=device,
            use_imagery_ft=use_imagery_ft,
        )
        y_pred_b_true_active = (y_pred_b_global - 1).astype(np.int64)
        stage_b_metrics = classification_metrics(
            y_true_b,
            y_pred_b_true_active,
            labels=[0, 1, 2, 3],
            label_names=["LEFT", "RIGHT", "FISTS", "FEET"],
        ).to_dict()

    return {
        "run_kind": run_kind,
        "stage_b_task_type_masking": True,
        "stage_a_family_heads": bool(base_models.get("stage_a_family") or source_models.get("stage_a_family")),
        "stage_b_family_heads": bool(base_models.get("stage_b_family")),
        "stage_a_thresholds": to_serializable(stage_a_thresholds_mixed) if stage_a_thresholds_mixed else None,
        "n_samples": int(len(y_sub)),
        "stage_a": {"metrics": stage_a_metrics, "y_true": y_true_a.tolist(), "y_pred": y_pred_a.tolist()},
        "stage_b_true_active": {"metrics": stage_b_metrics},
        "end_to_end": {"metrics": metrics_5, "y_true": y_sub.astype(int).tolist(), "y_pred": final_pred.astype(int).tolist()},
    }


def evaluate_stagea_imagery_multisource(
    *,
    base_ckpt: dict[str, Any],
    imagery_stagea_ckpt: dict[str, Any],
    data_root: str | None,
    out_dir: str | Path,
    device: str = "cpu",
    with_plots: bool = False,
) -> dict[str, Any]:
    if base_ckpt.get("split") != imagery_stagea_ckpt.get("split"):
        raise ValueError("Checkpoint splits differ; cannot hybridize safely")
    out_dir = ensure_dir(out_dir)
    base_cfg, split, test_subjects, X_base, y_base, meta_base = _load_test_data_for_ckpt(base_ckpt, data_root)
    src_cfg, _, _, X_src, y_src, meta_src = _load_test_data_for_ckpt(imagery_stagea_ckpt, data_root)
    _assert_aligned_rows(y_base, meta_base, y_src, meta_src, label="test set")

    base_models = _load_models_from_combined_checkpoint(base_ckpt, device=device)
    src_models = _load_models_from_combined_checkpoint(imagery_stagea_ckpt, device=device)
    mixed_thr = _merge_stage_a_thresholds_runwise(base_ckpt.get("stage_a_thresholds"), imagery_stagea_ckpt.get("stage_a_thresholds"))
    run_kinds = list((base_cfg.get("eval", {}) or {}).get("run_kinds", ["combined", "executed", "imagined"]))
    results: dict[str, Any] = {"base": {}, "fine_tuned_imagery": None}
    for rk in run_kinds:
        t0 = time.perf_counter()
        print(f"[hybrid] swap_stagea_imagery_multisource base {rk} start", flush=True)
        res = _evaluate_end_to_end_stagea_imagery_multisource(
            base_models=base_models,
            base_cfg=base_cfg,
            source_models=src_models,
            source_cfg=src_cfg,
            stage_a_thresholds_mixed=mixed_thr,
            X_base=X_base,
            y_base=y_base,
            meta_base=meta_base,
            X_src=X_src,
            y_src=y_src,
            meta_src=meta_src,
            run_kind=rk,
            subjects=test_subjects,
            device=device,
            use_imagery_ft=False,
        )
        results["base"][rk] = res
        print(f"[hybrid] swap_stagea_imagery_multisource base {rk} done in {time.perf_counter()-t0:.1f}s", flush=True)
        if with_plots:
            _save_confusions_for_result(res, Path(out_dir))
    if any([
        base_models.get("stage_b_finetuned") is not None,
        base_models.get("stage_b_family_finetuned") is not None,
        src_models.get("stage_a_finetuned") is not None,
        src_models.get("stage_a_family_finetuned") is not None,
    ]):
        ft_res = {}
        for rk in run_kinds:
            t0 = time.perf_counter()
            print(f"[hybrid] swap_stagea_imagery_multisource ft {rk} start", flush=True)
            res = _evaluate_end_to_end_stagea_imagery_multisource(
                base_models=base_models,
                base_cfg=base_cfg,
                source_models=src_models,
                source_cfg=src_cfg,
                stage_a_thresholds_mixed=mixed_thr,
                X_base=X_base,
                y_base=y_base,
                meta_base=meta_base,
                X_src=X_src,
                y_src=y_src,
                meta_src=meta_src,
                run_kind=rk,
                subjects=test_subjects,
                device=device,
                use_imagery_ft=True,
            )
            ft_res[rk] = res
            print(f"[hybrid] swap_stagea_imagery_multisource ft {rk} done in {time.perf_counter()-t0:.1f}s", flush=True)
        results["fine_tuned_imagery"] = ft_res
    write_json(Path(out_dir) / "evaluation_metrics.json", to_serializable(results))
    return results


def _merge_imagery_stage_metadata(base_cfg: dict[str, Any], imagery_cfg: dict[str, Any]) -> None:
    base_data = base_cfg.setdefault("data", {})
    img_data = (imagery_cfg or {}).get("data", {}) or {}

    # Preserve base stage windows/channels, but copy imagery-specific stage overrides if present.
    for dict_key in ("stage_time_windows", "stage_channels", "stage_channel_subsets"):
        base_map = base_data.get(dict_key)
        if not isinstance(base_map, dict):
            base_map = {}
            base_data[dict_key] = base_map
        img_map = img_data.get(dict_key) or {}
        if not isinstance(img_map, dict):
            continue
        for stage_key in IMAGERY_STAGE_KEYS:
            if stage_key in img_map:
                base_map[stage_key] = copy.deepcopy(img_map[stage_key])


def build_hybrid_checkpoint(base_ckpt: dict[str, Any], imagery_ckpt: dict[str, Any]) -> dict[str, Any]:
    kind = str(base_ckpt.get("kind", ""))
    if "hierarchical" not in kind:
        raise ValueError("base checkpoint must be a hierarchical bundle checkpoint")
    kind_i = str(imagery_ckpt.get("kind", ""))
    if "hierarchical" not in kind_i:
        raise ValueError("imagery checkpoint must be a hierarchical bundle checkpoint")

    # Strict split compatibility check (prevents accidental leakage / incomparable hybridization).
    if base_ckpt.get("split") != imagery_ckpt.get("split"):
        raise ValueError("Checkpoint splits differ; cannot hybridize safely")

    merged = copy.deepcopy(base_ckpt)
    merged["kind"] = "hierarchical_bundle_hybrid"
    merged["hybrid_sources"] = {
        "base_checkpoint_path": None,
        "imagery_checkpoint_path": None,
        "base_split_hash": _split_hash(base_ckpt),
        "imagery_split_hash": _split_hash(imagery_ckpt),
    }

    # Copy imagery fine-tuned branches from imagery checkpoint.
    for key in ("stage_a_finetuned", "stage_a_family_finetuned", "stage_b_finetuned", "stage_b_family_finetuned"):
        _merge_stage_dict(merged, imagery_ckpt, key)

    # Preserve base thresholds by default (they calibrate base Stage A). Explicitly drop imagery thresholds if copied.
    if "stage_a_thresholds" in base_ckpt:
        merged["stage_a_thresholds"] = copy.deepcopy(base_ckpt.get("stage_a_thresholds"))

    # Merge stage-specific model configs for imagery branches so evaluator builds the right shapes/backbones.
    model_cfgs_merged = dict(merged.get("model_configs") or {})
    for stage_key in IMAGERY_STAGE_KEYS:
        img_model_cfgs = imagery_ckpt.get("model_configs") or {}
        if isinstance(img_model_cfgs, dict) and stage_key in img_model_cfgs:
            model_cfgs_merged[stage_key] = copy.deepcopy(img_model_cfgs[stage_key])
    merged["model_configs"] = model_cfgs_merged

    # Merge imagery-specific stage timing/channel metadata into the base config.
    merged_cfg = copy.deepcopy(merged.get("config") or {})
    _merge_imagery_stage_metadata(merged_cfg, imagery_ckpt.get("config") or {})
    _materialize_imagery_stage_overrides(merged_cfg, imagery_ckpt.get("config") or {}, imagery_ckpt)
    merged["config"] = merged_cfg

    # Evaluation from source checkpoints is no longer representative; drop stale copies.
    merged["evaluation"] = None
    return merged


def build_execa_baselineb_checkpoint(exec_ckpt: dict[str, Any], baseline_ckpt: dict[str, Any]) -> dict[str, Any]:
    kind = str(exec_ckpt.get("kind", ""))
    if "hierarchical" not in kind:
        raise ValueError("exec checkpoint must be a hierarchical bundle checkpoint")
    kind_b = str(baseline_ckpt.get("kind", ""))
    if "hierarchical" not in kind_b:
        raise ValueError("baseline checkpoint must be a hierarchical bundle checkpoint")
    if exec_ckpt.get("split") != baseline_ckpt.get("split"):
        raise ValueError("Checkpoint splits differ; cannot hybridize safely")

    merged = copy.deepcopy(exec_ckpt)
    merged["kind"] = "hierarchical_bundle_hybrid_execA_baselineB"
    merged["hybrid_sources"] = {
        "exec_checkpoint_path": None,
        "baseline_checkpoint_path": None,
        "exec_split_hash": _split_hash(exec_ckpt),
        "baseline_split_hash": _split_hash(baseline_ckpt),
    }

    for key in EXECA_BASELINEB_COMPONENT_KEYS_FROM_SECONDARY:
        _merge_stage_dict(merged, baseline_ckpt, key)

    # Copy model configs for swapped stages.
    model_cfgs_merged = dict(merged.get("model_configs") or {})
    base_model_cfgs = baseline_ckpt.get("model_configs") or {}
    for stage_key in EXECA_BASELINEB_STAGE_KEYS_FROM_SECONDARY:
        if isinstance(base_model_cfgs, dict) and stage_key in base_model_cfgs:
            model_cfgs_merged[stage_key] = copy.deepcopy(base_model_cfgs[stage_key])
    merged["model_configs"] = model_cfgs_merged

    # Keep exec Stage A metadata; materialize exact baseline metadata for swapped stages.
    merged_cfg = copy.deepcopy(merged.get("config") or {})
    _merge_imagery_stage_metadata(merged_cfg, baseline_ckpt.get("config") or {})
    _materialize_stage_overrides(
        merged_cfg,
        baseline_ckpt.get("config") or {},
        baseline_ckpt,
        EXECA_BASELINEB_STAGE_KEYS_FROM_SECONDARY,
    )
    merged["config"] = merged_cfg
    merged["evaluation"] = None
    return merged


def build_swap_ff_branch_checkpoint(base_ckpt: dict[str, Any], ff_ckpt: dict[str, Any]) -> dict[str, Any]:
    kind = str(base_ckpt.get("kind", ""))
    if "hierarchical" not in kind:
        raise ValueError("base checkpoint must be a hierarchical bundle checkpoint")
    kind_f = str(ff_ckpt.get("kind", ""))
    if "hierarchical" not in kind_f:
        raise ValueError("ff checkpoint must be a hierarchical bundle checkpoint")
    if base_ckpt.get("split") != ff_ckpt.get("split"):
        raise ValueError("Checkpoint splits differ; cannot hybridize safely")
    sig_base = _single_x_preproc_signature(base_ckpt)
    sig_ff = _single_x_preproc_signature(ff_ckpt)
    if sig_base != sig_ff:
        raise ValueError(
            "swap_ff_branch requires matching global preprocessing because all branches share one loaded X. "
            f"Got base={sig_base} vs ff={sig_ff}"
        )

    merged = copy.deepcopy(base_ckpt)
    merged["kind"] = "hierarchical_bundle_hybrid_swapFF"
    merged["hybrid_sources"] = {
        "base_checkpoint_path": None,
        "ff_checkpoint_path": None,
        "base_split_hash": _split_hash(base_ckpt),
        "ff_split_hash": _split_hash(ff_ckpt),
    }

    _merge_family_entry(merged, ff_ckpt, "stage_b_family", "ff")
    _merge_family_entry(merged, ff_ckpt, "stage_b_family_finetuned", "ff")

    model_cfgs_merged = dict(merged.get("model_configs") or {})
    ff_model_cfgs = ff_ckpt.get("model_configs") or {}
    for stage_key in SWAP_FF_STAGE_KEYS_FROM_SECONDARY:
        if isinstance(ff_model_cfgs, dict) and stage_key in ff_model_cfgs:
            model_cfgs_merged[stage_key] = copy.deepcopy(ff_model_cfgs[stage_key])
    merged["model_configs"] = model_cfgs_merged

    merged_cfg = copy.deepcopy(merged.get("config") or {})
    _materialize_stage_overrides(
        merged_cfg,
        ff_ckpt.get("config") or {},
        ff_ckpt,
        SWAP_FF_STAGE_KEYS_FROM_SECONDARY,
    )
    merged["config"] = merged_cfg
    merged["evaluation"] = None
    return merged


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a hybrid checkpoint by combining base/executed branches and imagery fine-tuned branches from different runs.")
    p.add_argument(
        "--mode",
        default="imagery_ft",
        choices=["imagery_ft", "exec_a_baseline_b", "swap_ff_branch", "swap_ff_branch_multisource", "exec_a_baseline_b_multisource", "swap_stagea_imagery_multisource"],
        help=(
            "Hybrid composition mode: "
            "'imagery_ft' = base/executed branches from --base-checkpoint + imagery FT branches from --imagery-checkpoint; "
            "'exec_a_baseline_b' = Stage A base from --base-checkpoint + baseline Stage B + imagery FT branches from --imagery-checkpoint; "
            "'swap_ff_branch' = everything from --base-checkpoint except Stage B FF family branch (and its imagery FT) from --imagery-checkpoint, requiring matching preprocessing; "
            "'swap_ff_branch_multisource' = same FF branch swap but evaluated with two separately preprocessed test datasets (valid across preprocessing mismatches); "
            "'exec_a_baseline_b_multisource' = Stage A branch from --base-checkpoint and Stage B branch from --imagery-checkpoint with separate preprocessing datasets; "
            "'swap_stagea_imagery_multisource' = keep --base-checkpoint except replace imagined Stage A branch (incl. imagery FT/family imagery FT) from --imagery-checkpoint using separate preprocessing datasets."
        ),
    )
    p.add_argument("--base-checkpoint", required=True, help="Checkpoint providing base branches (stage_a/stage_b/family base).")
    p.add_argument("--imagery-checkpoint", required=True, help="Checkpoint providing imagery fine-tuned branches.")
    p.add_argument("--data-root", default=None, help="Override EEGMMIDB data root")
    p.add_argument("--out-dir", default=None, help="Evaluation output directory (default: <base_run>/eval_hybrid_<imagery_run>)")
    p.add_argument("--save-merged-checkpoint", default=None, help="Optional path to save the merged hybrid checkpoint before evaluation.")
    p.add_argument("--no-plots", action="store_true", help="Disable plotting during evaluation")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    set_matplotlib_env(Path.cwd() / "outputs")

    base_path = Path(args.base_checkpoint).resolve()
    img_path = Path(args.imagery_checkpoint).resolve()
    base_ckpt = load_checkpoint(base_path, map_location="cpu")
    img_ckpt = load_checkpoint(img_path, map_location="cpu")
    if args.mode == "imagery_ft":
        merged = build_hybrid_checkpoint(base_ckpt, img_ckpt)
    elif args.mode == "exec_a_baseline_b":
        merged = build_execa_baselineb_checkpoint(base_ckpt, img_ckpt)
    elif args.mode == "swap_ff_branch":
        merged = build_swap_ff_branch_checkpoint(base_ckpt, img_ckpt)
    elif args.mode == "swap_ff_branch_multisource":
        merged = None
    elif args.mode == "exec_a_baseline_b_multisource":
        merged = None
    elif args.mode == "swap_stagea_imagery_multisource":
        merged = None
    else:
        raise ValueError(args.mode)

    base_run_dir = base_path.parent
    img_run_tag = img_path.parent.name
    default_dir_name = f"eval_hybrid_{args.mode}_{img_run_tag}"
    out_dir = Path(args.out_dir) if args.out_dir else (base_run_dir / default_dir_name)
    ensure_dir(out_dir)
    if args.mode == "swap_ff_branch_multisource":
        results = evaluate_swap_ff_branch_multisource(
            base_ckpt=base_ckpt,
            ff_ckpt=img_ckpt,
            data_root=args.data_root,
            out_dir=out_dir,
            device="cpu",
            with_plots=not args.no_plots,
        )
        merged_ckpt_path = None
    elif args.mode == "exec_a_baseline_b_multisource":
        results = evaluate_exec_a_baseline_b_multisource(
            exec_ckpt=base_ckpt,
            base_ckpt=img_ckpt,
            data_root=args.data_root,
            out_dir=out_dir,
            device="cpu",
            with_plots=not args.no_plots,
        )
        merged_ckpt_path = None
    elif args.mode == "swap_stagea_imagery_multisource":
        results = evaluate_stagea_imagery_multisource(
            base_ckpt=base_ckpt,
            imagery_stagea_ckpt=img_ckpt,
            data_root=args.data_root,
            out_dir=out_dir,
            device="cpu",
            with_plots=not args.no_plots,
        )
        merged_ckpt_path = None
    else:
        merged["hybrid_sources"]["base_checkpoint_path"] = str(base_path)
        merged["hybrid_sources"]["imagery_checkpoint_path"] = str(img_path)
        merged_ckpt_path = Path(args.save_merged_checkpoint) if args.save_merged_checkpoint else (out_dir / "hybrid_best.pt")
        save_checkpoint(merged_ckpt_path, merged)
        results = evaluate_checkpoint(
            merged_ckpt_path,
            data_root=args.data_root,
            output_dir=out_dir,
            device="cpu",
            with_plots=not args.no_plots,
        )
    branch = "fine_tuned_imagery" if results.get("fine_tuned_imagery") else "base"
    combined = results[branch]["combined"]["end_to_end"]["metrics"]
    print(f"Hybrid eval complete ({branch}).")
    if merged_ckpt_path is not None:
        print(f"Saved merged checkpoint: {merged_ckpt_path}")
    print(f"Eval output dir: {out_dir}")
    print(
        "Combined held-out end-to-end: "
        f"acc={combined['accuracy']:.4f}, bal_acc={combined['balanced_accuracy']:.4f}, macro_f1={combined['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
