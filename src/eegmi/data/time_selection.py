from __future__ import annotations

from typing import Any

import numpy as np


def _stage_time_candidates(stage_key: str) -> list[str]:
    key = str(stage_key)
    candidates: list[str] = [key]
    if key.startswith("stage_a_") and ("lr" in key or "ff" in key):
        if "finetune_imagery" in key:
            candidates.append("stage_a_family_finetune_imagery")
        candidates.append("stage_a_family")
        candidates.append("stage_a")
    elif key.startswith("stage_b_") and ("lr" in key or "ff" in key):
        if "finetune_imagery" in key:
            candidates.append("stage_b_family_finetune_imagery")
        candidates.append("stage_b_family")
        candidates.append("stage_b")
    elif "finetune_imagery" in key:
        candidates.append(key.replace("_finetune_imagery", ""))
    return candidates


def _parse_window_spec(spec: Any, stage_key: str) -> tuple[float, float]:
    if isinstance(spec, dict):
        if "tmin" not in spec or "tmax" not in spec:
            raise ValueError(f"data.stage_time_windows[{stage_key!r}] must include tmin and tmax")
        tmin = float(spec["tmin"])
        tmax = float(spec["tmax"])
    elif isinstance(spec, (list, tuple)) and len(spec) == 2:
        tmin = float(spec[0])
        tmax = float(spec[1])
    else:
        raise ValueError(
            f"data.stage_time_windows[{stage_key!r}] must be [tmin,tmax] or {{'tmin':..,'tmax':..}}"
        )
    if tmax <= tmin:
        raise ValueError(f"Invalid stage time window for {stage_key}: tmax ({tmax}) must be > tmin ({tmin})")
    return tmin, tmax


def resolve_stage_time_window(data_cfg: dict[str, Any], stage_key: str) -> tuple[float, float]:
    global_tmin = float(data_cfg["tmin"])
    global_tmax = float(data_cfg["tmax"])
    stage_windows_cfg = data_cfg.get("stage_time_windows") or {}
    if not isinstance(stage_windows_cfg, dict):
        raise ValueError("data.stage_time_windows must be a dict when provided")

    chosen_key = None
    chosen_spec = None
    for cand in _stage_time_candidates(stage_key):
        if cand in stage_windows_cfg:
            chosen_key = cand
            chosen_spec = stage_windows_cfg[cand]
            break

    if chosen_spec is None:
        return global_tmin, global_tmax

    tmin, tmax = _parse_window_spec(chosen_spec, chosen_key or stage_key)
    eps = 1e-9
    if tmin < (global_tmin - eps) or tmax > (global_tmax + eps):
        raise ValueError(
            f"Stage time window for {stage_key} ({tmin},{tmax}) must be within global loaded window "
            f"({global_tmin},{global_tmax})"
        )
    return tmin, tmax


def resolve_stage_time_indices(data_cfg: dict[str, Any], stage_key: str, *, n_times: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    sfreq = float(data_cfg["sfreq"])
    global_tmin = float(data_cfg["tmin"])
    global_tmax = float(data_cfg["tmax"])
    stage_tmin, stage_tmax = resolve_stage_time_window(data_cfg, stage_key)

    start_f = (stage_tmin - global_tmin) * sfreq
    stop_f = (stage_tmax - global_tmin) * sfreq
    start = int(round(start_f))
    stop = int(round(stop_f)) + 1  # inclusive endpoint in MNE Epochs

    if abs(start - start_f) > 1e-4 or abs((stop - 1) - stop_f) > 1e-4:
        raise ValueError(
            f"Stage time window for {stage_key} is not aligned to sample grid at sfreq={sfreq}: "
            f"{stage_tmin}-{stage_tmax}"
        )

    if start < 0 or stop <= start:
        raise ValueError(f"Computed invalid time indices for {stage_key}: start={start}, stop={stop}")
    if n_times is not None and stop > int(n_times):
        raise ValueError(
            f"Stage time window for {stage_key} exceeds loaded epoch length: stop={stop}, n_times={n_times}"
        )

    idx = np.arange(start, stop, dtype=np.int64)
    expected_len = int(round((stage_tmax - stage_tmin) * sfreq)) + 1
    if len(idx) != expected_len:
        raise AssertionError(
            f"Stage time index length mismatch for {stage_key}: {len(idx)} != {expected_len}"
        )
    meta = {
        "tmin": float(stage_tmin),
        "tmax": float(stage_tmax),
        "n_times": int(len(idx)),
        "sample_start": int(start),
        "sample_stop_exclusive": int(stop),
        "global_tmin": float(global_tmin),
        "global_tmax": float(global_tmax),
        "sfreq": float(sfreq),
    }
    return idx, meta

