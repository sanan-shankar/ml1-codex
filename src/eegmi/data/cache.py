from __future__ import annotations

from pathlib import Path
from typing import Any
import errno
import shutil

import numpy as np
import pandas as pd

from eegmi.constants import META_COLUMNS
from eegmi.utils import ensure_dir, read_json, sha1_hex_from_obj, write_json


def preproc_signature(data_cfg: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "channels",
        "sfreq",
        "tmin",
        "tmax",
        "baseline_window_len",
        "baseline_window_stride",
        "bandpass",
        "notch",
        "filter_bank_bands",
        "reref",
        "csd",
        "ica",
        "rest_zscore",
        "euclidean_alignment",
        "euclidean_alignment_time_window",
    ]
    sig = {k: data_cfg.get(k) for k in keys}
    cache_cfg = data_cfg.get("cache", {})
    sig["cache_version"] = cache_cfg.get("version", "v1")
    sig["schema"] = "X_y_meta_v1"
    return sig


def preproc_hash(data_cfg: dict[str, Any]) -> str:
    return sha1_hex_from_obj(preproc_signature(data_cfg))[:16]


def cache_dir_for_cfg(data_cfg: dict[str, Any]) -> Path:
    cache_root = Path(data_cfg.get("cache", {}).get("dir", "outputs/cache"))
    return ensure_dir(cache_root / preproc_hash(data_cfg))


def subject_cache_paths(data_cfg: dict[str, Any], subject: str) -> dict[str, Path]:
    base = cache_dir_for_cfg(data_cfg)
    stem = f"{subject}"
    return {
        "npz": base / f"{stem}.npz",
        "meta_parquet": base / f"{stem}.meta.parquet",
        "meta_csv": base / f"{stem}.meta.csv",
        "manifest": base / f"{stem}.manifest.json",
    }


def cache_exists(data_cfg: dict[str, Any], subject: str) -> bool:
    paths = subject_cache_paths(data_cfg, subject)
    meta_exists = paths["meta_parquet"].exists() or paths["meta_csv"].exists()
    return paths["npz"].exists() and paths["manifest"].exists() and meta_exists


def save_subject_cache(data_cfg: dict[str, Any], subject: str, X: np.ndarray, y: np.ndarray, meta: pd.DataFrame) -> None:
    paths = subject_cache_paths(data_cfg, subject)
    ensure_dir(paths["npz"].parent)
    free_bytes = shutil.disk_usage(paths["npz"].parent).free
    if free_bytes < 1_000_000_000:
        print(f"[cache] Skipping cache write for {subject}: low free space ({free_bytes} bytes)")
        return
    try:
        np.savez_compressed(paths["npz"], X=X.astype(np.float32), y=y.astype(np.int64))
    except OSError as e:
        if getattr(e, "errno", None) == errno.ENOSPC:
            for key in ("npz", "meta_parquet", "meta_csv", "manifest"):
                try:
                    if paths[key].exists():
                        paths[key].unlink()
                except Exception:
                    pass
            print(f"[cache] Skipping cache write for {subject}: no space left on device")
            return
        raise
    meta_to_save = meta.loc[:, META_COLUMNS].copy()
    saved_meta_format = "csv"
    try:
        meta_to_save.to_parquet(paths["meta_parquet"], index=False)
        saved_meta_format = "parquet"
        if paths["meta_csv"].exists():
            paths["meta_csv"].unlink()
    except Exception:
        meta_to_save.to_csv(paths["meta_csv"], index=False)
        if paths["meta_parquet"].exists():
            try:
                paths["meta_parquet"].unlink()
            except Exception:
                pass
    manifest = {
        "subject": subject,
        "preproc_hash": preproc_hash(data_cfg),
        "preproc_signature": preproc_signature(data_cfg),
        "n_samples": int(X.shape[0]),
        "n_channels": int(X.shape[1]),
        "n_times": int(X.shape[2]),
        "meta_columns": META_COLUMNS,
        "meta_format": saved_meta_format,
        "x_dtype": str(X.dtype),
        "y_dtype": str(y.dtype),
    }
    write_json(paths["manifest"], manifest)


def load_subject_cache(data_cfg: dict[str, Any], subject: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    paths = subject_cache_paths(data_cfg, subject)
    manifest = read_json(paths["manifest"])
    expected_hash = preproc_hash(data_cfg)
    if manifest.get("preproc_hash") != expected_hash:
        raise ValueError(
            f"Cache hash mismatch for {subject}: {manifest.get('preproc_hash')} != {expected_hash}"
        )
    arr = np.load(paths["npz"], allow_pickle=False)
    X = arr["X"].astype(np.float32, copy=False)
    y = arr["y"].astype(np.int64, copy=False)
    if paths["meta_parquet"].exists():
        try:
            meta = pd.read_parquet(paths["meta_parquet"])
        except Exception:
            if not paths["meta_csv"].exists():
                raise
            meta = pd.read_csv(paths["meta_csv"])
    elif paths["meta_csv"].exists():
        meta = pd.read_csv(paths["meta_csv"])
    else:
        raise FileNotFoundError(f"No cached meta file for {subject}")

    meta = meta.loc[:, META_COLUMNS].copy()
    assert X.ndim == 3, f"Cached X must be [n,C,T], got {X.shape}"
    assert y.ndim == 1, f"Cached y must be [n], got {y.shape}"
    assert len(meta) == X.shape[0] == y.shape[0], "Cache sample count mismatch"
    return X, y, meta
