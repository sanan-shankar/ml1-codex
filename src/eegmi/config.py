from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .constants import DEFAULT_MOTOR_CHANNELS
from .utils import sha1_hex_from_obj


def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def default_config() -> dict[str, Any]:
    return {
        "experiment": {"name": "eegmi", "seed": 42, "output_root": "outputs", "device": "cpu"},
        "data": {
            "data_root": "/Users/sanan/Desktop/ML/my-ml1/files",
            "channels": list(DEFAULT_MOTOR_CHANNELS),
            "sfreq": 160,
            "tmin": 0.5,
            "tmax": 4.0,
            "baseline_window_len": 561,
            "baseline_window_stride": 561,
            "bandpass": [8.0, 30.0],
            "notch": None,
            "reref": None,
            "csd": False,
            "stage_channels": {},
            "stage_time_windows": {},
            "ica": {
                "enabled": False,
                "fit_l_freq": 1.0,
                "fit_h_freq": 40.0,
                "method": "fastica",
                "n_components": 0.99,
                "max_iter": 256,
                "decim": 4,
                "eog_proxy_channels": ["FP1", "FP2", "FPZ", "AF7", "AF3", "AFZ", "AF4", "AF8"],
                "eog_threshold": 3.0,
                "max_eog_components": 4,
            },
            "rest_zscore": True,
            "euclidean_alignment": False,
            "euclidean_alignment_time_window": None,
            "cache": {"enabled": True, "dir": "outputs/cache", "version": "v1"},
        },
        "split": {"seed": 42, "n_train_pool": 100, "n_test": 9, "inner_val_count": 10, "path": None},
        "augmentation": {
            "enabled": True,
            "gaussian_noise_std": 0.0,
            "time_shift_max": 0,
            "amplitude_scale_min": 1.0,
            "amplitude_scale_max": 1.0,
        },
        "model": {
            "type": "eegnet",
            "dropout": 0.35,
            "eegnet": {},
            "shallow": {},
            "fbcnet": {},
            "fusion": {},
            "paper_cnn_gru": {},
            "paper_cla": {},
        },
        "model_overrides": {},
        "train": {
            "num_workers": 0,
            "batch_size": 64,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "max_epochs": 40,
            "patience": 8,
            "min_delta": 5e-4,
            "use_balanced_sampler": True,
            "weighted_loss": True,
            "smote": {"enabled": False},
            "stage_a": {},
            "stage_a_family_heads": {"enabled": False},
            "stage_b": {},
            "stage_b_family_heads": {"enabled": False},
            "fine_tune_imagery_stage_a": {"enabled": False, "init_from_base": True},
            "fine_tune_imagery": {"enabled": False, "init_from_base": True},
        },
        "eval": {
            "run_kinds": ["combined", "executed", "imagined"],
            "stage_b_task_type_masking": True,
            "calibrate_stage_a_thresholds": True,
            "stage_a_threshold_granularity": "run_kind",
            "separate_stage_a_thresholds_by_ft": False,
            "stage_a_tta_time_shifts": [],
            "stage_a_adabn": {"enabled": False, "num_passes": 1, "batch_size": 128},
            "plots": {},
        },
    }


def load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    try:
        user_cfg = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse config {p}. This project expects JSON-compatible YAML (.yaml with JSON syntax)."
        ) from e
    cfg = default_config()
    _deep_update(cfg, user_cfg)
    return cfg


def clone_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(cfg)


def config_hash(cfg: dict[str, Any], keys: list[str] | None = None) -> str:
    if keys is None:
        obj = cfg
    else:
        obj = {k: cfg[k] for k in keys if k in cfg}
    return sha1_hex_from_obj(obj)
