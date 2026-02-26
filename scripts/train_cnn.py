from __future__ import annotations

import argparse
import copy
import gc
import sys
import zlib
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

from eegmi.config import config_hash, load_config
from eegmi.constants import LABEL_TO_NAME, STAGE_A_LABELS, STAGE_B_LABELS
from eegmi.data.augment import build_augmenter
from eegmi.data.channel_selection import resolve_stage_channel_indices
from eegmi.data.time_selection import resolve_stage_time_indices
from eegmi.data.dataset import EEGEpochDataset, stage_targets, subset_indices
from eegmi.data.loader import load_eegmmidb_epochs
from eegmi.data.splits import discover_subjects, load_subject_split, make_subject_split, save_subject_split, split_hash
from eegmi.eval.evaluate import calibrate_stage_a_thresholds, evaluate_hierarchical_bundle
from eegmi.eval.plots import save_learning_curves
from eegmi.models.heads import build_model
from eegmi.repro import seed_everything
from eegmi.train.checkpointing import save_checkpoint
from eegmi.train.engine import train_stage
from eegmi.utils import ensure_dir, now_stamp, set_matplotlib_env, to_serializable, write_json


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train compact CNNs for EEGMMIDB hierarchical decoding")
    p.add_argument("--config", required=True, help="Path to JSON-compatible YAML config")
    return p


def _stage_train_cfg(train_cfg: dict[str, Any], stage_key: str) -> dict[str, Any]:
    merged = dict(train_cfg)
    merged.update(train_cfg.get(stage_key, {}))
    return merged


def _seed_for_stage(cfg: dict[str, Any], stage_name: str) -> int:
    base_seed = int(cfg["experiment"]["seed"])
    offset = int(zlib.crc32(str(stage_name).encode("utf-8")) & 0xFFFFFFFF)
    return (base_seed + offset) % (2**31 - 1)


def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)
    return dst


def _augment_cfg_for_stage(cfg: dict[str, Any], stage_name: str) -> dict[str, Any]:
    base_aug = copy.deepcopy(cfg.get("augmentation", {}) or {})
    overrides = cfg.get("augmentation_overrides", {}) or {}
    candidates = [stage_name]
    if stage_name.startswith("stage_a_") and ("lr" in stage_name or "ff" in stage_name):
        if "finetune_imagery" in stage_name:
            candidates.append("stage_a_family_finetune_imagery")
        candidates.append("stage_a_family")
    if stage_name.startswith("stage_b_") and ("lr" in stage_name or "ff" in stage_name):
        if "finetune_imagery" in stage_name:
            candidates.append("stage_b_family_finetune_imagery")
        candidates.append("stage_b_family")
    if stage_name.startswith("stage_a"):
        candidates.append("stage_a")
    if stage_name.startswith("stage_b"):
        candidates.append("stage_b")
    for key in candidates:
        ov = overrides.get(key)
        if isinstance(ov, dict):
            _deep_update(base_aug, ov)
    return base_aug


def _augmenter_for_stage(cfg: dict[str, Any], stage_name: str):
    aug_cfg = _augment_cfg_for_stage(cfg, stage_name)
    return build_augmenter(aug_cfg, seed=_seed_for_stage(cfg, f"aug::{stage_name}"))


def _model_cfg_for_stage(cfg: dict[str, Any], stage_key: str) -> dict[str, Any]:
    model_cfg = copy.deepcopy(cfg["model"])
    overrides = cfg.get("model_overrides", {}) or {}
    candidates: list[str] = [stage_key]
    if stage_key.startswith("stage_a_") and ("lr" in stage_key or "ff" in stage_key):
        if "finetune_imagery" in stage_key:
            candidates.append("stage_a_family_finetune_imagery")
        candidates.append("stage_a_family")
    if stage_key.startswith("stage_b_") and ("lr" in stage_key or "ff" in stage_key):
        if "finetune_imagery" in stage_key:
            candidates.append("stage_b_family_finetune_imagery")
        candidates.append("stage_b_family")
    if "finetune_imagery" in stage_key and not (("lr" in stage_key or "ff" in stage_key) and stage_key.startswith(("stage_a_", "stage_b_"))):
        candidates.append(stage_key.replace("_finetune_imagery", ""))
    for key in candidates:
        ov = overrides.get(key)
        if isinstance(ov, dict):
            _deep_update(model_cfg, ov)
    return model_cfg


def _build_stage_model(cfg: dict[str, Any], stage_key: str, *, n_chans: int, n_times: int, n_classes: int):
    model_cfg = _model_cfg_for_stage(cfg, stage_key)
    ch_idx, _ = resolve_stage_channel_indices(cfg["data"], stage_key)
    t_idx, _ = resolve_stage_time_indices(cfg["data"], stage_key, n_times=n_times)
    model = build_model(model_cfg, n_chans=int(len(ch_idx)), n_times=int(len(t_idx)), n_classes=n_classes)
    return model, model_cfg


def _make_dataset(
    X,
    y,
    meta,
    *,
    subjects,
    run_kind,
    active_only,
    task_types=None,
    stage,
    augmenter=None,
    channel_indices=None,
    time_indices=None,
):
    idx = subset_indices(y, meta, subjects=subjects, run_kind=run_kind, active_only=active_only, task_types=task_types)
    return EEGEpochDataset(
        X,
        y,
        meta,
        indices=idx,
        channel_indices=channel_indices,
        time_indices=time_indices,
        stage=stage,
        add_channel_dim=False,
        augmenter=augmenter,
    )


def _save_split(run_dir: Path, cfg: dict[str, Any]):
    subjects = discover_subjects(cfg["data"]["data_root"])
    cfg_subjects = cfg.get("data", {}).get("subjects")
    if cfg_subjects:
        cfg_subjects = sorted(list(cfg_subjects))
        missing = [s for s in cfg_subjects if s not in subjects]
        if missing:
            raise ValueError(f"Configured smoke subjects not found under data_root: {missing}")
        subjects = cfg_subjects
    split_cfg = cfg["split"]
    if split_cfg.get("path"):
        split = load_subject_split(split_cfg["path"])
    else:
        split = make_subject_split(
            subjects,
            seed=int(split_cfg["seed"]),
            n_train_pool=int(split_cfg["n_train_pool"]),
            n_test=int(split_cfg["n_test"]),
            inner_val_count=int(split_cfg["inner_val_count"]),
        )
    split_path = run_dir / "subject_split.json"
    save_subject_split(split_path, split)
    return split, split_path


def _dataset_target_counts(dataset: EEGEpochDataset) -> dict[str, Any]:
    targets = stage_targets(dataset.y, dataset.stage)
    counts = np.bincount(targets, minlength=int(targets.max()) + 1 if len(targets) else 0)
    return {str(i): int(v) for i, v in enumerate(counts.tolist())}


def _base_run_kind_for(train_cfg: dict[str, Any], stage_group: str) -> str:
    default = str(train_cfg.get("base_run_kind", "combined"))
    if stage_group == "stage_a":
        return str(train_cfg.get("stage_a_base_run_kind", default))
    if stage_group == "stage_b":
        return str(train_cfg.get("stage_b_base_run_kind", default))
    return default


def _train_stage_b_family_head(
    *,
    cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    X,
    y,
    meta,
    n_chans: int,
    n_times: int,
    device: str,
    weighted_loss: bool,
    use_balanced_sampler: bool,
    split,
    run_dir: Path,
    family_key: str,  # "lr" or "ff"
    task_type: str,   # "LR" or "FF"
    label_names: list[str],
    ft_imagery_cfg: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any]]:
    stage_mode = "stage_b_lr" if family_key == "lr" else "stage_b_ff"
    stage_name = f"stage_b_{family_key}"
    ch_idx, ch_names = resolve_stage_channel_indices(cfg["data"], stage_name)
    t_idx, t_meta = resolve_stage_time_indices(cfg["data"], stage_name, n_times=n_times)
    data_diag: dict[str, Any] = {}
    data_diag["channels"] = ch_names
    data_diag["time_window"] = t_meta
    base_run_kind = _base_run_kind_for(train_cfg, "stage_b")
    data_diag["base_run_kind"] = base_run_kind
    ds_train = _make_dataset(
        X, y, meta,
        subjects=split.train_inner, run_kind=base_run_kind, active_only=True, task_types=[task_type],
        stage=stage_mode, augmenter=_augmenter_for_stage(cfg, stage_name), channel_indices=ch_idx, time_indices=t_idx,
    )
    ds_val = _make_dataset(
        X, y, meta,
        subjects=split.val_inner, run_kind=base_run_kind, active_only=True, task_types=[task_type],
        stage=stage_mode, augmenter=None, channel_indices=ch_idx, time_indices=t_idx,
    )
    data_diag["train_samples"] = int(len(ds_train))
    data_diag["val_samples"] = int(len(ds_val))
    data_diag["train_class_counts"] = _dataset_target_counts(ds_train) if len(ds_train) else {}
    data_diag["val_class_counts"] = _dataset_target_counts(ds_val) if len(ds_val) else {}
    print(
        f"[diag] {stage_name} samples train={len(ds_train)} val={len(ds_val)} "
        f"class_counts(train)={data_diag['train_class_counts']}"
    )
    if len(ds_train) == 0 or len(ds_val) == 0:
        return None, None, data_diag

    model, _ = _build_stage_model(cfg, stage_name, n_chans=n_chans, n_times=n_times, n_classes=2)
    out_dir = ensure_dir(run_dir / stage_name)
    seed_everything(_seed_for_stage(cfg, stage_name))
    result = train_stage(
        model,
        stage_name=stage_name,
        class_names=label_names,
        train_dataset=ds_train,
        val_dataset=ds_val,
        train_cfg=_stage_train_cfg(train_cfg, "stage_b"),
        output_dir=out_dir,
        device=device,
        weighted_loss=weighted_loss,
        use_balanced_sampler=use_balanced_sampler,
    )
    save_learning_curves(result["history"], out_dir / "learning_curves.png", title=stage_name)
    del ds_train, ds_val
    gc.collect()

    result_ft = None
    if ft_imagery_cfg and ft_imagery_cfg.get("enabled", False):
        ch_idx_ft, ch_names_ft = resolve_stage_channel_indices(cfg["data"], f"{stage_name}_finetune_imagery")
        t_idx_ft, t_meta_ft = resolve_stage_time_indices(cfg["data"], f"{stage_name}_finetune_imagery", n_times=n_times)
        ds_ft_train = _make_dataset(
            X, y, meta,
            subjects=split.train_inner, run_kind="imagined", active_only=True, task_types=[task_type],
            stage=stage_mode, augmenter=_augmenter_for_stage(cfg, f"{stage_name}_finetune_imagery"), channel_indices=ch_idx_ft, time_indices=t_idx_ft,
        )
        ds_ft_val = _make_dataset(
            X, y, meta,
            subjects=split.val_inner, run_kind="imagined", active_only=True, task_types=[task_type],
            stage=stage_mode, augmenter=None, channel_indices=ch_idx_ft, time_indices=t_idx_ft,
        )
        data_diag["imagery_finetune"] = {
            "train_samples": int(len(ds_ft_train)),
            "val_samples": int(len(ds_ft_val)),
            "channels": ch_names_ft,
            "time_window": t_meta_ft,
            "train_class_counts": _dataset_target_counts(ds_ft_train) if len(ds_ft_train) else {},
            "val_class_counts": _dataset_target_counts(ds_ft_val) if len(ds_ft_val) else {},
        }
        print(
            f"[diag] {stage_name} FT(imagery) samples train={len(ds_ft_train)} val={len(ds_ft_val)} "
            f"class_counts(train)={data_diag['imagery_finetune']['train_class_counts']}"
        )
        if len(ds_ft_train) > 0 and len(ds_ft_val) > 0:
            ft_model, _ = _build_stage_model(
                cfg,
                f"{stage_name}_finetune_imagery",
                n_chans=n_chans,
                n_times=n_times,
                n_classes=2,
            )
            if bool(ft_imagery_cfg.get("init_from_base", True)):
                ft_model.load_state_dict(copy.deepcopy(result["model"].state_dict()))
            out_ft = ensure_dir(run_dir / f"{stage_name}_finetune_imagery")
            ft_cfg_merged = _stage_train_cfg({**train_cfg, **ft_imagery_cfg}, "fine_tune_imagery")
            seed_everything(_seed_for_stage(cfg, f"{stage_name}_finetune_imagery"))
            result_ft = train_stage(
                ft_model,
                stage_name=f"{stage_name}_finetune_imagery",
                class_names=label_names,
                train_dataset=ds_ft_train,
                val_dataset=ds_ft_val,
                train_cfg=ft_cfg_merged,
                output_dir=out_ft,
                device=device,
                weighted_loss=weighted_loss,
                use_balanced_sampler=use_balanced_sampler,
            )
            save_learning_curves(result_ft["history"], out_ft / "learning_curves.png", title=f"{stage_name} Fine-tune (Imagery)")
        del ds_ft_train, ds_ft_val
        gc.collect()

    return result, result_ft, data_diag


def _train_stage_a_family_head(
    *,
    cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    X,
    y,
    meta,
    n_chans: int,
    n_times: int,
    device: str,
    weighted_loss: bool,
    use_balanced_sampler: bool,
    split,
    run_dir: Path,
    family_key: str,  # "lr" or "ff"
    task_type: str,   # "LR" or "FF"
    ft_imagery_cfg: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any]]:
    stage_mode = "stage_a_lr" if family_key == "lr" else "stage_a_ff"
    stage_name = f"stage_a_{family_key}"
    ch_idx, ch_names = resolve_stage_channel_indices(cfg["data"], stage_name)
    t_idx, t_meta = resolve_stage_time_indices(cfg["data"], stage_name, n_times=n_times)
    data_diag: dict[str, Any] = {}
    data_diag["channels"] = ch_names
    data_diag["time_window"] = t_meta
    base_run_kind = _base_run_kind_for(train_cfg, "stage_a")
    data_diag["base_run_kind"] = base_run_kind
    ds_train = _make_dataset(
        X, y, meta,
        subjects=split.train_inner, run_kind=base_run_kind, active_only=False, task_types=[task_type],
        stage=stage_mode, augmenter=_augmenter_for_stage(cfg, stage_name), channel_indices=ch_idx, time_indices=t_idx,
    )
    ds_val = _make_dataset(
        X, y, meta,
        subjects=split.val_inner, run_kind=base_run_kind, active_only=False, task_types=[task_type],
        stage=stage_mode, augmenter=None, channel_indices=ch_idx, time_indices=t_idx,
    )
    data_diag["train_samples"] = int(len(ds_train))
    data_diag["val_samples"] = int(len(ds_val))
    data_diag["train_class_counts"] = _dataset_target_counts(ds_train) if len(ds_train) else {}
    data_diag["val_class_counts"] = _dataset_target_counts(ds_val) if len(ds_val) else {}
    print(
        f"[diag] {stage_name} samples train={len(ds_train)} val={len(ds_val)} "
        f"class_counts(train)={data_diag['train_class_counts']}"
    )
    if len(ds_train) == 0 or len(ds_val) == 0:
        return None, None, data_diag

    model, _ = _build_stage_model(cfg, stage_name, n_chans=n_chans, n_times=n_times, n_classes=2)
    out_dir = ensure_dir(run_dir / stage_name)
    seed_everything(_seed_for_stage(cfg, stage_name))
    result = train_stage(
        model,
        stage_name=stage_name,
        class_names=[STAGE_A_LABELS[i] for i in [0, 1]],
        train_dataset=ds_train,
        val_dataset=ds_val,
        train_cfg=_stage_train_cfg(train_cfg, "stage_a"),
        output_dir=out_dir,
        device=device,
        weighted_loss=weighted_loss,
        use_balanced_sampler=use_balanced_sampler,
    )
    save_learning_curves(result["history"], out_dir / "learning_curves.png", title=stage_name)
    del ds_train, ds_val
    gc.collect()

    result_ft = None
    if ft_imagery_cfg and ft_imagery_cfg.get("enabled", False):
        ch_idx_ft, ch_names_ft = resolve_stage_channel_indices(cfg["data"], f"{stage_name}_finetune_imagery")
        t_idx_ft, t_meta_ft = resolve_stage_time_indices(cfg["data"], f"{stage_name}_finetune_imagery", n_times=n_times)
        ds_ft_train = _make_dataset(
            X, y, meta,
            subjects=split.train_inner, run_kind="imagined", active_only=False, task_types=[task_type],
            stage=stage_mode, augmenter=_augmenter_for_stage(cfg, f"{stage_name}_finetune_imagery"), channel_indices=ch_idx_ft, time_indices=t_idx_ft,
        )
        ds_ft_val = _make_dataset(
            X, y, meta,
            subjects=split.val_inner, run_kind="imagined", active_only=False, task_types=[task_type],
            stage=stage_mode, augmenter=None, channel_indices=ch_idx_ft, time_indices=t_idx_ft,
        )
        data_diag["imagery_finetune"] = {
            "train_samples": int(len(ds_ft_train)),
            "val_samples": int(len(ds_ft_val)),
            "channels": ch_names_ft,
            "time_window": t_meta_ft,
            "train_class_counts": _dataset_target_counts(ds_ft_train) if len(ds_ft_train) else {},
            "val_class_counts": _dataset_target_counts(ds_ft_val) if len(ds_ft_val) else {},
        }
        print(
            f"[diag] {stage_name} FT(imagery) samples train={len(ds_ft_train)} val={len(ds_ft_val)} "
            f"class_counts(train)={data_diag['imagery_finetune']['train_class_counts']}"
        )
        if len(ds_ft_train) > 0 and len(ds_ft_val) > 0:
            ft_model, _ = _build_stage_model(
                cfg,
                f"{stage_name}_finetune_imagery",
                n_chans=n_chans,
                n_times=n_times,
                n_classes=2,
            )
            if bool(ft_imagery_cfg.get("init_from_base", True)):
                ft_model.load_state_dict(copy.deepcopy(result["model"].state_dict()))
            out_ft = ensure_dir(run_dir / f"{stage_name}_finetune_imagery")
            ft_cfg_merged = _stage_train_cfg({**train_cfg, **ft_imagery_cfg}, "fine_tune_imagery_stage_a")
            seed_everything(_seed_for_stage(cfg, f"{stage_name}_finetune_imagery"))
            result_ft = train_stage(
                ft_model,
                stage_name=f"{stage_name}_finetune_imagery",
                class_names=[STAGE_A_LABELS[i] for i in [0, 1]],
                train_dataset=ds_ft_train,
                val_dataset=ds_ft_val,
                train_cfg=ft_cfg_merged,
                output_dir=out_ft,
                device=device,
                weighted_loss=weighted_loss,
                use_balanced_sampler=use_balanced_sampler,
            )
            save_learning_curves(result_ft["history"], out_ft / "learning_curves.png", title=f"{stage_name} Fine-tune (Imagery)")
        del ds_ft_train, ds_ft_val
        gc.collect()

    return result, result_ft, data_diag


def _train_all_stages(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    seed_everything(int(cfg["experiment"]["seed"]))
    split, split_path = _save_split(run_dir, cfg)

    # Load only training-pool subjects for fitting to reduce memory pressure.
    X_train, y_train, meta_train = load_eegmmidb_epochs(cfg, subjects=sorted(split.train_pool_100))
    n_chans = int(X_train.shape[1])
    n_times = int(X_train.shape[2])

    train_cfg = cfg["train"]
    stage_a_base_run_kind = _base_run_kind_for(train_cfg, "stage_a")
    stage_b_base_run_kind = _base_run_kind_for(train_cfg, "stage_b")
    device = str(cfg["experiment"].get("device", "cpu"))
    use_balanced_sampler = bool(train_cfg.get("use_balanced_sampler", True))
    weighted_loss = bool(train_cfg.get("weighted_loss", True))
    data_diag: dict[str, Any] = {
        "train_pool_subjects": list(split.train_pool_100),
        "train_inner_subjects": list(split.train_inner),
        "val_inner_subjects": list(split.val_inner),
        "test_subjects": list(split.test_9),
        "train_pool_samples": int(len(y_train)),
        "stage_a_base_run_kind": stage_a_base_run_kind,
        "stage_b_base_run_kind": stage_b_base_run_kind,
    }
    model_cfgs_used: dict[str, Any] = {}
    stage_channel_map: dict[str, list[str]] = {}
    stage_time_map: dict[str, dict[str, Any]] = {}

    # Stage A datasets (REST vs ACTIVE), combined executed+imagined+baseline within split subjects
    ch_stage_a, ch_stage_a_names = resolve_stage_channel_indices(cfg["data"], "stage_a")
    t_stage_a, t_stage_a_meta = resolve_stage_time_indices(cfg["data"], "stage_a", n_times=n_times)
    stage_channel_map["stage_a"] = ch_stage_a_names
    stage_time_map["stage_a"] = t_stage_a_meta
    ds_a_train = _make_dataset(
        X_train, y_train, meta_train,
        subjects=split.train_inner, run_kind=stage_a_base_run_kind, active_only=False,
        stage="stage_a", augmenter=_augmenter_for_stage(cfg, "stage_a"), channel_indices=ch_stage_a, time_indices=t_stage_a,
    )
    ds_a_val = _make_dataset(
        X_train, y_train, meta_train,
        subjects=split.val_inner, run_kind=stage_a_base_run_kind, active_only=False,
        stage="stage_a", augmenter=None, channel_indices=ch_stage_a, time_indices=t_stage_a,
    )
    data_diag["stage_a"] = {
        "train_samples": int(len(ds_a_train)),
        "val_samples": int(len(ds_a_val)),
        "channels": ch_stage_a_names,
        "time_window": t_stage_a_meta,
        "base_run_kind": stage_a_base_run_kind,
        "train_class_counts": _dataset_target_counts(ds_a_train),
        "val_class_counts": _dataset_target_counts(ds_a_val),
    }
    print(f"[diag] Stage A samples train={len(ds_a_train)} val={len(ds_a_val)} class_counts(train)={data_diag['stage_a']['train_class_counts']}")

    stage_a_model, model_cfgs_used["stage_a"] = _build_stage_model(cfg, "stage_a", n_chans=n_chans, n_times=n_times, n_classes=2)
    out_stage_a = ensure_dir(run_dir / "stage_a")
    seed_everything(_seed_for_stage(cfg, "stage_a"))
    result_a = train_stage(
        stage_a_model,
        stage_name="stage_a",
        class_names=[STAGE_A_LABELS[i] for i in [0, 1]],
        train_dataset=ds_a_train,
        val_dataset=ds_a_val,
        train_cfg=_stage_train_cfg(train_cfg, "stage_a"),
        output_dir=out_stage_a,
        device=device,
        weighted_loss=weighted_loss,
        use_balanced_sampler=use_balanced_sampler,
    )
    save_learning_curves(result_a["history"], out_stage_a / "learning_curves.png", title="Stage A")
    del ds_a_train, ds_a_val
    gc.collect()

    result_a_ft = None
    ft_a_cfg = train_cfg.get("fine_tune_imagery_stage_a", {})
    if ft_a_cfg.get("enabled", False):
        ch_stage_aft, ch_stage_aft_names = resolve_stage_channel_indices(cfg["data"], "stage_a_finetune_imagery")
        t_stage_aft, t_stage_aft_meta = resolve_stage_time_indices(cfg["data"], "stage_a_finetune_imagery", n_times=n_times)
        stage_channel_map["stage_a_finetune_imagery"] = ch_stage_aft_names
        stage_time_map["stage_a_finetune_imagery"] = t_stage_aft_meta
        # Fine-tune Stage A on imagery runs only (includes T0/T1/T2 inside imagined runs).
        ds_aft_train = _make_dataset(
            X_train, y_train, meta_train,
            subjects=split.train_inner, run_kind="imagined", active_only=False,
            stage="stage_a", augmenter=_augmenter_for_stage(cfg, "stage_a_finetune_imagery"), channel_indices=ch_stage_aft, time_indices=t_stage_aft,
        )
        ds_aft_val = _make_dataset(
            X_train, y_train, meta_train,
            subjects=split.val_inner, run_kind="imagined", active_only=False,
            stage="stage_a", augmenter=None, channel_indices=ch_stage_aft, time_indices=t_stage_aft,
        )
        data_diag["stage_a_finetune_imagery"] = {
            "train_samples": int(len(ds_aft_train)),
            "val_samples": int(len(ds_aft_val)),
            "channels": ch_stage_aft_names,
            "time_window": t_stage_aft_meta,
            "train_class_counts": _dataset_target_counts(ds_aft_train) if len(ds_aft_train) else {},
            "val_class_counts": _dataset_target_counts(ds_aft_val) if len(ds_aft_val) else {},
        }
        print(f"[diag] Stage A FT(imagery) samples train={len(ds_aft_train)} val={len(ds_aft_val)} class_counts(train)={data_diag['stage_a_finetune_imagery']['train_class_counts']}")
        if len(ds_aft_train) > 0 and len(ds_aft_val) > 0:
            stage_a_ft_model, model_cfgs_used["stage_a_finetune_imagery"] = _build_stage_model(
                cfg,
                "stage_a_finetune_imagery",
                n_chans=n_chans,
                n_times=n_times,
                n_classes=2,
            )
            if bool(ft_a_cfg.get("init_from_base", True)):
                stage_a_ft_model.load_state_dict(copy.deepcopy(result_a["model"].state_dict()))
            out_stage_aft = ensure_dir(run_dir / "stage_a_finetune_imagery")
            ft_a_train_cfg = _stage_train_cfg({**train_cfg, **ft_a_cfg}, "fine_tune_imagery_stage_a")
            seed_everything(_seed_for_stage(cfg, "stage_a_finetune_imagery"))
            result_a_ft = train_stage(
                stage_a_ft_model,
                stage_name="stage_a_finetune_imagery",
                class_names=[STAGE_A_LABELS[i] for i in [0, 1]],
                train_dataset=ds_aft_train,
                val_dataset=ds_aft_val,
                train_cfg=ft_a_train_cfg,
                output_dir=out_stage_aft,
                device=device,
                weighted_loss=weighted_loss,
                use_balanced_sampler=use_balanced_sampler,
            )
            save_learning_curves(result_a_ft["history"], out_stage_aft / "learning_curves.png", title="Stage A Fine-tune (Imagery)")
        del ds_aft_train, ds_aft_val
        gc.collect()

    result_a_lr = None
    result_a_lr_ft = None
    result_a_ff = None
    result_a_ff_ft = None
    family_stage_a_cfg = train_cfg.get("stage_a_family_heads", {})
    if family_stage_a_cfg.get("enabled", False):
        families = {str(v).lower() for v in family_stage_a_cfg.get("families", ["lr", "ff"])}
        if "lr" in families:
            result_a_lr, result_a_lr_ft, diag_a_lr = _train_stage_a_family_head(
                cfg=cfg,
                train_cfg=train_cfg,
                X=X_train,
                y=y_train,
                meta=meta_train,
                n_chans=n_chans,
                n_times=n_times,
                device=device,
                weighted_loss=weighted_loss,
                use_balanced_sampler=use_balanced_sampler,
                split=split,
                run_dir=run_dir,
                family_key="lr",
                task_type="LR",
                ft_imagery_cfg=ft_a_cfg,
            )
            data_diag["stage_a_family_lr"] = diag_a_lr
            if isinstance(diag_a_lr.get("channels"), list):
                stage_channel_map["stage_a_lr"] = list(diag_a_lr["channels"])
            if isinstance(diag_a_lr.get("time_window"), dict):
                stage_time_map["stage_a_lr"] = dict(diag_a_lr["time_window"])
            if isinstance(diag_a_lr.get("imagery_finetune", {}).get("channels"), list):
                stage_channel_map["stage_a_lr_finetune_imagery"] = list(diag_a_lr["imagery_finetune"]["channels"])
            if isinstance(diag_a_lr.get("imagery_finetune", {}).get("time_window"), dict):
                stage_time_map["stage_a_lr_finetune_imagery"] = dict(diag_a_lr["imagery_finetune"]["time_window"])
            if result_a_lr is not None:
                model_cfgs_used["stage_a_lr"] = _model_cfg_for_stage(cfg, "stage_a_lr")
            if result_a_lr_ft is not None:
                model_cfgs_used["stage_a_lr_finetune_imagery"] = _model_cfg_for_stage(cfg, "stage_a_lr_finetune_imagery")
        if "ff" in families:
            result_a_ff, result_a_ff_ft, diag_a_ff = _train_stage_a_family_head(
                cfg=cfg,
                train_cfg=train_cfg,
                X=X_train,
                y=y_train,
                meta=meta_train,
                n_chans=n_chans,
                n_times=n_times,
                device=device,
                weighted_loss=weighted_loss,
                use_balanced_sampler=use_balanced_sampler,
                split=split,
                run_dir=run_dir,
                family_key="ff",
                task_type="FF",
                ft_imagery_cfg=ft_a_cfg,
            )
            data_diag["stage_a_family_ff"] = diag_a_ff
            if isinstance(diag_a_ff.get("channels"), list):
                stage_channel_map["stage_a_ff"] = list(diag_a_ff["channels"])
            if isinstance(diag_a_ff.get("time_window"), dict):
                stage_time_map["stage_a_ff"] = dict(diag_a_ff["time_window"])
            if isinstance(diag_a_ff.get("imagery_finetune", {}).get("channels"), list):
                stage_channel_map["stage_a_ff_finetune_imagery"] = list(diag_a_ff["imagery_finetune"]["channels"])
            if isinstance(diag_a_ff.get("imagery_finetune", {}).get("time_window"), dict):
                stage_time_map["stage_a_ff_finetune_imagery"] = dict(diag_a_ff["imagery_finetune"]["time_window"])
            if result_a_ff is not None:
                model_cfgs_used["stage_a_ff"] = _model_cfg_for_stage(cfg, "stage_a_ff")
            if result_a_ff_ft is not None:
                model_cfgs_used["stage_a_ff_finetune_imagery"] = _model_cfg_for_stage(cfg, "stage_a_ff_finetune_imagery")

    # Stage B datasets (ACTIVE only, 4-class), combined executed+imagined
    ch_stage_b, ch_stage_b_names = resolve_stage_channel_indices(cfg["data"], "stage_b")
    t_stage_b, t_stage_b_meta = resolve_stage_time_indices(cfg["data"], "stage_b", n_times=n_times)
    stage_channel_map["stage_b"] = ch_stage_b_names
    stage_time_map["stage_b"] = t_stage_b_meta
    ds_b_train = _make_dataset(
        X_train, y_train, meta_train,
        subjects=split.train_inner, run_kind=stage_b_base_run_kind, active_only=True,
        stage="stage_b", augmenter=_augmenter_for_stage(cfg, "stage_b"), channel_indices=ch_stage_b, time_indices=t_stage_b,
    )
    ds_b_val = _make_dataset(
        X_train, y_train, meta_train,
        subjects=split.val_inner, run_kind=stage_b_base_run_kind, active_only=True,
        stage="stage_b", augmenter=None, channel_indices=ch_stage_b, time_indices=t_stage_b,
    )
    data_diag["stage_b"] = {
        "train_samples": int(len(ds_b_train)),
        "val_samples": int(len(ds_b_val)),
        "channels": ch_stage_b_names,
        "time_window": t_stage_b_meta,
        "base_run_kind": stage_b_base_run_kind,
        "train_class_counts": _dataset_target_counts(ds_b_train),
        "val_class_counts": _dataset_target_counts(ds_b_val),
    }
    print(f"[diag] Stage B samples train={len(ds_b_train)} val={len(ds_b_val)} class_counts(train)={data_diag['stage_b']['train_class_counts']}")

    stage_b_model, model_cfgs_used["stage_b"] = _build_stage_model(cfg, "stage_b", n_chans=n_chans, n_times=n_times, n_classes=4)
    out_stage_b = ensure_dir(run_dir / "stage_b")
    seed_everything(_seed_for_stage(cfg, "stage_b"))
    result_b = train_stage(
        stage_b_model,
        stage_name="stage_b",
        class_names=[STAGE_B_LABELS[i] for i in [0, 1, 2, 3]],
        train_dataset=ds_b_train,
        val_dataset=ds_b_val,
        train_cfg=_stage_train_cfg(train_cfg, "stage_b"),
        output_dir=out_stage_b,
        device=device,
        weighted_loss=weighted_loss,
        use_balanced_sampler=use_balanced_sampler,
    )
    save_learning_curves(result_b["history"], out_stage_b / "learning_curves.png", title="Stage B")
    del ds_b_train, ds_b_val
    gc.collect()

    result_b_ft = None
    ft_cfg = train_cfg.get("fine_tune_imagery", {})
    if ft_cfg.get("enabled", False):
        ch_stage_bft, ch_stage_bft_names = resolve_stage_channel_indices(cfg["data"], "stage_b_finetune_imagery")
        t_stage_bft, t_stage_bft_meta = resolve_stage_time_indices(cfg["data"], "stage_b_finetune_imagery", n_times=n_times)
        stage_channel_map["stage_b_finetune_imagery"] = ch_stage_bft_names
        stage_time_map["stage_b_finetune_imagery"] = t_stage_bft_meta
        # Fine-tune Stage B on imagery-only active epochs using train subjects only (inner train/val from train pool)
        ds_bft_train = _make_dataset(
            X_train, y_train, meta_train,
            subjects=split.train_inner, run_kind="imagined", active_only=True,
            stage="stage_b", augmenter=_augmenter_for_stage(cfg, "stage_b_finetune_imagery"), channel_indices=ch_stage_bft, time_indices=t_stage_bft,
        )
        ds_bft_val = _make_dataset(
            X_train, y_train, meta_train,
            subjects=split.val_inner, run_kind="imagined", active_only=True,
            stage="stage_b", augmenter=None, channel_indices=ch_stage_bft, time_indices=t_stage_bft,
        )
        data_diag["stage_b_finetune_imagery"] = {
            "train_samples": int(len(ds_bft_train)),
            "val_samples": int(len(ds_bft_val)),
            "channels": ch_stage_bft_names,
            "time_window": t_stage_bft_meta,
            "train_class_counts": _dataset_target_counts(ds_bft_train) if len(ds_bft_train) else {},
            "val_class_counts": _dataset_target_counts(ds_bft_val) if len(ds_bft_val) else {},
        }
        print(f"[diag] Stage B FT(imagery) samples train={len(ds_bft_train)} val={len(ds_bft_val)} class_counts(train)={data_diag['stage_b_finetune_imagery']['train_class_counts']}")
        if len(ds_bft_train) > 0 and len(ds_bft_val) > 0:
            stage_b_ft_model, model_cfgs_used["stage_b_finetune_imagery"] = _build_stage_model(
                cfg,
                "stage_b_finetune_imagery",
                n_chans=n_chans,
                n_times=n_times,
                n_classes=4,
            )
            stage_b_ft_model.load_state_dict(copy.deepcopy(result_b["model"].state_dict()))
            out_stage_bft = ensure_dir(run_dir / "stage_b_finetune_imagery")
            ft_train_cfg = _stage_train_cfg({**train_cfg, **ft_cfg}, "fine_tune_imagery")
            seed_everything(_seed_for_stage(cfg, "stage_b_finetune_imagery"))
            result_b_ft = train_stage(
                stage_b_ft_model,
                stage_name="stage_b_finetune_imagery",
                class_names=[STAGE_B_LABELS[i] for i in [0, 1, 2, 3]],
                train_dataset=ds_bft_train,
                val_dataset=ds_bft_val,
                train_cfg=ft_train_cfg,
                output_dir=out_stage_bft,
                device=device,
                weighted_loss=weighted_loss,
                use_balanced_sampler=use_balanced_sampler,
            )
            save_learning_curves(result_b_ft["history"], out_stage_bft / "learning_curves.png", title="Stage B Fine-tune (Imagery)")
        del ds_bft_train, ds_bft_val
        gc.collect()

    result_b_lr = None
    result_b_lr_ft = None
    result_b_ff = None
    result_b_ff_ft = None
    family_heads_cfg = train_cfg.get("stage_b_family_heads", {})
    if family_heads_cfg.get("enabled", False):
        result_b_lr, result_b_lr_ft, diag_lr = _train_stage_b_family_head(
            cfg=cfg,
            train_cfg=train_cfg,
            X=X_train,
            y=y_train,
            meta=meta_train,
            n_chans=n_chans,
            n_times=n_times,
            device=device,
            weighted_loss=weighted_loss,
            use_balanced_sampler=use_balanced_sampler,
            split=split,
            run_dir=run_dir,
            family_key="lr",
            task_type="LR",
            label_names=["LEFT", "RIGHT"],
            ft_imagery_cfg=ft_cfg,
        )
        result_b_ff, result_b_ff_ft, diag_ff = _train_stage_b_family_head(
            cfg=cfg,
            train_cfg=train_cfg,
            X=X_train,
            y=y_train,
            meta=meta_train,
            n_chans=n_chans,
            n_times=n_times,
            device=device,
            weighted_loss=weighted_loss,
            use_balanced_sampler=use_balanced_sampler,
            split=split,
            run_dir=run_dir,
            family_key="ff",
            task_type="FF",
            label_names=["FISTS", "FEET"],
            ft_imagery_cfg=ft_cfg,
        )
        data_diag["stage_b_family_lr"] = diag_lr
        data_diag["stage_b_family_ff"] = diag_ff
        if isinstance(diag_lr.get("channels"), list):
            stage_channel_map["stage_b_lr"] = list(diag_lr["channels"])
        if isinstance(diag_lr.get("time_window"), dict):
            stage_time_map["stage_b_lr"] = dict(diag_lr["time_window"])
        if isinstance(diag_lr.get("imagery_finetune", {}).get("channels"), list):
            stage_channel_map["stage_b_lr_finetune_imagery"] = list(diag_lr["imagery_finetune"]["channels"])
        if isinstance(diag_lr.get("imagery_finetune", {}).get("time_window"), dict):
            stage_time_map["stage_b_lr_finetune_imagery"] = dict(diag_lr["imagery_finetune"]["time_window"])
        if isinstance(diag_ff.get("channels"), list):
            stage_channel_map["stage_b_ff"] = list(diag_ff["channels"])
        if isinstance(diag_ff.get("time_window"), dict):
            stage_time_map["stage_b_ff"] = dict(diag_ff["time_window"])
        if isinstance(diag_ff.get("imagery_finetune", {}).get("channels"), list):
            stage_channel_map["stage_b_ff_finetune_imagery"] = list(diag_ff["imagery_finetune"]["channels"])
        if isinstance(diag_ff.get("imagery_finetune", {}).get("time_window"), dict):
            stage_time_map["stage_b_ff_finetune_imagery"] = dict(diag_ff["imagery_finetune"]["time_window"])
        if result_b_lr is not None:
            model_cfgs_used["stage_b_lr"] = _model_cfg_for_stage(cfg, "stage_b_lr")
        if result_b_lr_ft is not None:
            model_cfgs_used["stage_b_lr_finetune_imagery"] = _model_cfg_for_stage(cfg, "stage_b_lr_finetune_imagery")
        if result_b_ff is not None:
            model_cfgs_used["stage_b_ff"] = _model_cfg_for_stage(cfg, "stage_b_ff")
        if result_b_ff_ft is not None:
            model_cfgs_used["stage_b_ff_finetune_imagery"] = _model_cfg_for_stage(cfg, "stage_b_ff_finetune_imagery")

    stage_a_thresholds = None
    stage_a_thresholds_base = None
    stage_a_thresholds_finetuned = None
    if cfg.get("eval", {}).get("calibrate_stage_a_thresholds", True):
        try:
            a_ch, _ = resolve_stage_channel_indices(cfg["data"], "stage_a")
            a_ft_ch, _ = resolve_stage_channel_indices(cfg["data"], "stage_a_finetune_imagery")
            a_tm, _ = resolve_stage_time_indices(cfg["data"], "stage_a", n_times=n_times)
            a_ft_tm, _ = resolve_stage_time_indices(cfg["data"], "stage_a_finetune_imagery", n_times=n_times)
            a_family_ch = {
                "lr": resolve_stage_channel_indices(cfg["data"], "stage_a_lr")[0],
                "ff": resolve_stage_channel_indices(cfg["data"], "stage_a_ff")[0],
            }
            a_family_tm = {
                "lr": resolve_stage_time_indices(cfg["data"], "stage_a_lr", n_times=n_times)[0],
                "ff": resolve_stage_time_indices(cfg["data"], "stage_a_ff", n_times=n_times)[0],
            }
            a_family_ft_ch = {
                "lr": resolve_stage_channel_indices(cfg["data"], "stage_a_lr_finetune_imagery")[0],
                "ff": resolve_stage_channel_indices(cfg["data"], "stage_a_ff_finetune_imagery")[0],
            }
            a_family_ft_tm = {
                "lr": resolve_stage_time_indices(cfg["data"], "stage_a_lr_finetune_imagery", n_times=n_times)[0],
                "ff": resolve_stage_time_indices(cfg["data"], "stage_a_ff_finetune_imagery", n_times=n_times)[0],
            }
            stage_a_family_models_base = (
                {
                    **({"lr": result_a_lr["model"]} if result_a_lr is not None else {}),
                    **({"ff": result_a_ff["model"]} if result_a_ff is not None else {}),
                } or None
            )
            stage_a_family_models_ft = (
                {
                    **({"lr": (result_a_lr_ft["model"] if result_a_lr_ft is not None else None)} if result_a_lr is not None else {}),
                    **({"ff": (result_a_ff_ft["model"] if result_a_ff_ft is not None else None)} if result_a_ff is not None else {}),
                } or None
            )
            common_thr_kwargs = dict(
                X=X_train,
                y=y_train,
                meta=meta_train,
                subjects=list(split.val_inner),
                device=device,
                batch_size=int(train_cfg.get("batch_size", 128)),
                stage_a_tta_time_shifts=[int(v) for v in (cfg.get("eval", {}).get("stage_a_tta_time_shifts", []) or [])],
                granularity=str(cfg.get("eval", {}).get("stage_a_threshold_granularity", "run_kind")),
                stage_a_channel_indices=a_ch,
                stage_a_imagined_channel_indices=a_ft_ch,
                stage_a_family_channel_indices=a_family_ch,
                stage_a_family_imagined_channel_indices=a_family_ft_ch,
                stage_a_time_indices=a_tm,
                stage_a_imagined_time_indices=a_ft_tm,
                stage_a_family_time_indices=a_family_tm,
                stage_a_family_imagined_time_indices=a_family_ft_tm,
            )
            use_split_thresholds = bool(cfg.get("eval", {}).get("separate_stage_a_thresholds_by_ft", False))
            if use_split_thresholds:
                stage_a_thresholds_base = calibrate_stage_a_thresholds(
                    stage_a_model=result_a["model"],
                    stage_a_imagined_model=None,
                    stage_a_family_models=stage_a_family_models_base,
                    stage_a_family_imagined_models=None,
                    **common_thr_kwargs,
                )
                if (result_a_ft is not None) or (result_a_lr_ft is not None) or (result_a_ff_ft is not None):
                    stage_a_thresholds_finetuned = calibrate_stage_a_thresholds(
                        stage_a_model=result_a["model"],
                        stage_a_imagined_model=(result_a_ft["model"] if result_a_ft is not None else None),
                        stage_a_family_models=stage_a_family_models_base,
                        stage_a_family_imagined_models=stage_a_family_models_ft,
                        **common_thr_kwargs,
                    )
                stage_a_thresholds = stage_a_thresholds_finetuned or stage_a_thresholds_base
            else:
                stage_a_thresholds = calibrate_stage_a_thresholds(
                    stage_a_model=result_a["model"],
                    stage_a_imagined_model=(result_a_ft["model"] if result_a_ft is not None else None),
                    stage_a_family_models=stage_a_family_models_base,
                    stage_a_family_imagined_models=stage_a_family_models_ft,
                    **common_thr_kwargs,
                )
                stage_a_thresholds_base = None
                stage_a_thresholds_finetuned = None
            write_json(run_dir / "stage_a_thresholds.json", to_serializable(stage_a_thresholds))
            if stage_a_thresholds_base is not None:
                write_json(run_dir / "stage_a_thresholds_base.json", to_serializable(stage_a_thresholds_base))
            if stage_a_thresholds_finetuned is not None:
                write_json(run_dir / "stage_a_thresholds_finetuned.json", to_serializable(stage_a_thresholds_finetuned))
            print(
                "[diag] Stage A thresholds "
                f"default={stage_a_thresholds.get('default', 0.5):.3f} "
                f"exec={stage_a_thresholds.get('by_run_kind', {}).get('executed', float('nan')):.3f} "
                f"imag={stage_a_thresholds.get('by_run_kind', {}).get('imagined', float('nan')):.3f}"
            )
        except Exception as e:
            print(f"[warn] Stage A threshold calibration failed; falling back to argmax/0.5. Error: {e}")
            stage_a_thresholds = None
            stage_a_thresholds_base = None
            stage_a_thresholds_finetuned = None

    combined_ckpt = {
        "kind": "hierarchical_bundle",
        "config": cfg,
        "config_hash": config_hash(cfg),
        "split": split.to_dict(),
        "split_hash": split_hash(split),
        "split_path": str(split_path),
        "model_config": cfg["model"],
        "model_configs": to_serializable(model_cfgs_used),
        "stage_channels": to_serializable(stage_channel_map),
        "stage_times": to_serializable(stage_time_map),
        "n_chans": n_chans,
        "n_times": n_times,
        "stage_a": {
            "stage": "A",
            "class_names": [STAGE_A_LABELS[i] for i in [0, 1]],
            "model_state_dict": result_a["model"].state_dict(),
            "best_epoch": result_a["best_epoch"],
            "best_score": result_a["best_score"],
            "best_metrics": result_a["best_metrics"],
            "class_weights": result_a["class_weights"],
            "history": result_a["history"],
        },
        "stage_b": {
            "stage": "B",
            "class_names": [STAGE_B_LABELS[i] for i in [0, 1, 2, 3]],
            "model_state_dict": result_b["model"].state_dict(),
            "best_epoch": result_b["best_epoch"],
            "best_score": result_b["best_score"],
            "best_metrics": result_b["best_metrics"],
            "class_weights": result_b["class_weights"],
            "history": result_b["history"],
        },
        "stage_a_finetuned": None,
        "stage_a_family": None,
        "stage_a_family_finetuned": None,
        "stage_b_finetuned": None,
        "stage_b_family": None,
        "stage_b_family_finetuned": None,
        "stage_a_thresholds": stage_a_thresholds,
        "stage_a_thresholds_base": stage_a_thresholds_base,
        "stage_a_thresholds_finetuned": stage_a_thresholds_finetuned,
        "evaluation": None,
    }
    if result_a_ft is not None:
        combined_ckpt["stage_a_finetuned"] = {
            "stage": "A_finetune_imagery",
            "class_names": [STAGE_A_LABELS[i] for i in [0, 1]],
            "model_state_dict": result_a_ft["model"].state_dict(),
            "best_epoch": result_a_ft["best_epoch"],
            "best_score": result_a_ft["best_score"],
            "best_metrics": result_a_ft["best_metrics"],
            "class_weights": result_a_ft["class_weights"],
            "history": result_a_ft["history"],
        }
    if result_a_lr is not None or result_a_ff is not None:
        combined_ckpt["stage_a_family"] = {}
        if result_a_lr is not None:
            combined_ckpt["stage_a_family"]["lr"] = {
                "stage": "A_LR",
                "class_names": ["REST", "ACTIVE"],
                "model_state_dict": result_a_lr["model"].state_dict(),
                "best_epoch": result_a_lr["best_epoch"],
                "best_score": result_a_lr["best_score"],
                "best_metrics": result_a_lr["best_metrics"],
                "class_weights": result_a_lr["class_weights"],
                "history": result_a_lr["history"],
            }
        if result_a_ff is not None:
            combined_ckpt["stage_a_family"]["ff"] = {
                "stage": "A_FF",
                "class_names": ["REST", "ACTIVE"],
                "model_state_dict": result_a_ff["model"].state_dict(),
                "best_epoch": result_a_ff["best_epoch"],
                "best_score": result_a_ff["best_score"],
                "best_metrics": result_a_ff["best_metrics"],
                "class_weights": result_a_ff["class_weights"],
                "history": result_a_ff["history"],
            }
    if result_a_lr_ft is not None or result_a_ff_ft is not None:
        combined_ckpt["stage_a_family_finetuned"] = {}
        if result_a_lr_ft is not None:
            combined_ckpt["stage_a_family_finetuned"]["lr"] = {
                "stage": "A_LR_finetune_imagery",
                "class_names": ["REST", "ACTIVE"],
                "model_state_dict": result_a_lr_ft["model"].state_dict(),
                "best_epoch": result_a_lr_ft["best_epoch"],
                "best_score": result_a_lr_ft["best_score"],
                "best_metrics": result_a_lr_ft["best_metrics"],
                "class_weights": result_a_lr_ft["class_weights"],
                "history": result_a_lr_ft["history"],
            }
        if result_a_ff_ft is not None:
            combined_ckpt["stage_a_family_finetuned"]["ff"] = {
                "stage": "A_FF_finetune_imagery",
                "class_names": ["REST", "ACTIVE"],
                "model_state_dict": result_a_ff_ft["model"].state_dict(),
                "best_epoch": result_a_ff_ft["best_epoch"],
                "best_score": result_a_ff_ft["best_score"],
                "best_metrics": result_a_ff_ft["best_metrics"],
                "class_weights": result_a_ff_ft["class_weights"],
                "history": result_a_ff_ft["history"],
            }
    if result_b_ft is not None:
        combined_ckpt["stage_b_finetuned"] = {
            "stage": "B_finetune_imagery",
            "class_names": [STAGE_B_LABELS[i] for i in [0, 1, 2, 3]],
            "model_state_dict": result_b_ft["model"].state_dict(),
            "best_epoch": result_b_ft["best_epoch"],
            "best_score": result_b_ft["best_score"],
            "best_metrics": result_b_ft["best_metrics"],
            "class_weights": result_b_ft["class_weights"],
            "history": result_b_ft["history"],
        }
    if result_b_lr is not None and result_b_ff is not None:
        combined_ckpt["stage_b_family"] = {
            "lr": {
                "stage": "B_LR",
                "class_names": ["LEFT", "RIGHT"],
                "model_state_dict": result_b_lr["model"].state_dict(),
                "best_epoch": result_b_lr["best_epoch"],
                "best_score": result_b_lr["best_score"],
                "best_metrics": result_b_lr["best_metrics"],
                "class_weights": result_b_lr["class_weights"],
                "history": result_b_lr["history"],
            },
            "ff": {
                "stage": "B_FF",
                "class_names": ["FISTS", "FEET"],
                "model_state_dict": result_b_ff["model"].state_dict(),
                "best_epoch": result_b_ff["best_epoch"],
                "best_score": result_b_ff["best_score"],
                "best_metrics": result_b_ff["best_metrics"],
                "class_weights": result_b_ff["class_weights"],
                "history": result_b_ff["history"],
            },
        }
    if result_b_lr_ft is not None or result_b_ff_ft is not None:
        combined_ckpt["stage_b_family_finetuned"] = {}
        if result_b_lr_ft is not None:
            combined_ckpt["stage_b_family_finetuned"]["lr"] = {
                "stage": "B_LR_finetune_imagery",
                "class_names": ["LEFT", "RIGHT"],
                "model_state_dict": result_b_lr_ft["model"].state_dict(),
                "best_epoch": result_b_lr_ft["best_epoch"],
                "best_score": result_b_lr_ft["best_score"],
                "best_metrics": result_b_lr_ft["best_metrics"],
                "class_weights": result_b_lr_ft["class_weights"],
                "history": result_b_lr_ft["history"],
            }
        if result_b_ff_ft is not None:
            combined_ckpt["stage_b_family_finetuned"]["ff"] = {
                "stage": "B_FF_finetune_imagery",
                "class_names": ["FISTS", "FEET"],
                "model_state_dict": result_b_ff_ft["model"].state_dict(),
                "best_epoch": result_b_ff_ft["best_epoch"],
                "best_score": result_b_ff_ft["best_score"],
                "best_metrics": result_b_ff_ft["best_metrics"],
                "class_weights": result_b_ff_ft["class_weights"],
                "history": result_b_ff_ft["history"],
            }

    # Save a recoverable combined checkpoint before evaluation so plot failures don't waste training time.
    save_checkpoint(run_dir / "best.pt", combined_ckpt)

    # Held-out evaluation on test subjects
    eval_out = ensure_dir(run_dir / "evaluation")
    # Free training arrays before loading held-out subjects to keep RSS stable on macOS.
    del X_train, y_train, meta_train
    gc.collect()
    X_test, y_test, meta_test = load_eegmmidb_epochs(cfg, subjects=sorted(split.test_9))
    data_diag["test_samples"] = int(len(y_test))
    print(f"[diag] Held-out test samples={len(y_test)} subjects={len(split.test_9)}")
    write_json(run_dir / "data_diagnostics.json", data_diag)
    eval_results = evaluate_hierarchical_bundle(
        cfg=cfg,
        split=split.to_dict(),
        stage_a_model=result_a["model"],
        stage_a_finetuned_model=(result_a_ft["model"] if result_a_ft is not None else None),
        stage_a_family_models=(
            {
                **({"lr": result_a_lr["model"]} if result_a_lr is not None else {}),
                **({"ff": result_a_ff["model"]} if result_a_ff is not None else {}),
            } or None
        ),
        stage_a_family_finetuned_models=(
            {
                **({"lr": (result_a_lr_ft["model"] if result_a_lr_ft is not None else None)} if result_a_lr is not None else {}),
                **({"ff": (result_a_ff_ft["model"] if result_a_ff_ft is not None else None)} if result_a_ff is not None else {}),
            } or None
        ),
        stage_b_model=result_b["model"],
        stage_b_finetuned_model=(result_b_ft["model"] if result_b_ft is not None else None),
        stage_b_family_models=(
            {"lr": result_b_lr["model"], "ff": result_b_ff["model"]}
            if (result_b_lr is not None and result_b_ff is not None)
            else None
        ),
        stage_b_family_finetuned_models=(
            {
                "lr": (result_b_lr_ft["model"] if result_b_lr_ft is not None else None),
                "ff": (result_b_ff_ft["model"] if result_b_ff_ft is not None else None),
            }
            if (result_b_lr is not None and result_b_ff is not None)
            else None
        ),
        stage_a_thresholds=stage_a_thresholds,
        stage_a_thresholds_base=stage_a_thresholds_base,
        stage_a_thresholds_finetuned=stage_a_thresholds_finetuned,
        output_dir=eval_out,
        device=device,
        X=X_test,
        y=y_test,
        meta=meta_test,
    )
    del X_test, y_test, meta_test
    gc.collect()
    combined_ckpt["evaluation"] = eval_results

    save_checkpoint(run_dir / "best.pt", combined_ckpt)
    write_json(run_dir / "run_summary.json", {
        "config_hash": combined_ckpt["config_hash"],
        "split_hash": combined_ckpt["split_hash"],
        "stage_a_thresholds": stage_a_thresholds,
        "stage_a_thresholds_base": stage_a_thresholds_base,
        "stage_a_thresholds_finetuned": stage_a_thresholds_finetuned,
        "stage_a_best": result_a["best_metrics"],
        "stage_a_finetuned_best": result_a_ft["best_metrics"] if result_a_ft else None,
        "stage_a_family_lr_best": result_a_lr["best_metrics"] if result_a_lr else None,
        "stage_a_family_ff_best": result_a_ff["best_metrics"] if result_a_ff else None,
        "stage_b_best": result_b["best_metrics"],
        "stage_b_finetuned_best": result_b_ft["best_metrics"] if result_b_ft else None,
        "stage_b_family_lr_best": result_b_lr["best_metrics"] if result_b_lr else None,
        "stage_b_family_ff_best": result_b_ff["best_metrics"] if result_b_ff else None,
        "test_evaluation": to_serializable(eval_results),
        "labels": LABEL_TO_NAME,
    })

    return {
        "run_dir": str(run_dir),
        "checkpoint": str(run_dir / "best.pt"),
        "split_path": str(split_path),
        "evaluation": eval_results,
    }


def main() -> None:
    args = build_argparser().parse_args()
    cfg = load_config(args.config)
    set_matplotlib_env(Path.cwd() / cfg["experiment"].get("output_root", "outputs"))

    exp_name = str(cfg["experiment"].get("name", "eegmi"))
    run_id = f"{now_stamp()}_{exp_name}_{config_hash(cfg)[:8]}"
    run_dir = ensure_dir(Path(cfg["experiment"].get("output_root", "outputs")) / run_id)
    write_json(run_dir / "config_snapshot.json", cfg)

    result = _train_all_stages(cfg, run_dir)
    print(f"Training complete. Run dir: {result['run_dir']}")
    print(f"Combined checkpoint: {result['checkpoint']}")
    base_combined = result["evaluation"]["base"]["combined"]["end_to_end"]["metrics"]
    print(
        "Held-out combined end-to-end (base): "
        f"acc={base_combined['accuracy']:.4f}, bal_acc={base_combined['balanced_accuracy']:.4f}, macro_f1={base_combined['macro_f1']:.4f}"
    )
    if result["evaluation"].get("fine_tuned_imagery"):
        ft_combined = result["evaluation"]["fine_tuned_imagery"]["combined"]["end_to_end"]["metrics"]
        print(
            "Held-out combined end-to-end (fine_tuned_imagery): "
            f"acc={ft_combined['accuracy']:.4f}, bal_acc={ft_combined['balanced_accuracy']:.4f}, macro_f1={ft_combined['macro_f1']:.4f}"
        )
        base_img = result["evaluation"]["base"]["imagined"]["end_to_end"]["metrics"]
        ft_img = result["evaluation"]["fine_tuned_imagery"]["imagined"]["end_to_end"]["metrics"]
        print(
            "Imagery held-out comparison (end-to-end): "
            f"base bal_acc={base_img['balanced_accuracy']:.4f}, ft bal_acc={ft_img['balanced_accuracy']:.4f}; "
            f"base macro_f1={base_img['macro_f1']:.4f}, ft macro_f1={ft_img['macro_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
