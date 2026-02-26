from __future__ import annotations

from typing import Any

import numpy as np


def clean_channel_name_local(name: str) -> str:
    return str(name).replace(".", "").strip().upper()


def _stage_channel_candidates(stage_key: str) -> list[str]:
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

    # Last resort: no slicing override => all channels.
    return candidates


def resolve_stage_channel_names(data_cfg: dict[str, Any], stage_key: str) -> list[str]:
    base_channels = [clean_channel_name_local(ch) for ch in data_cfg.get("channels", [])]
    if not base_channels:
        raise ValueError("data.channels must be a non-empty list")
    stage_channels_cfg = data_cfg.get("stage_channels") or {}
    if not isinstance(stage_channels_cfg, dict):
        raise ValueError("data.stage_channels must be a dict when provided")

    chosen = None
    for cand in _stage_channel_candidates(stage_key):
        if cand in stage_channels_cfg:
            chosen = stage_channels_cfg[cand]
            break
    if chosen is None:
        return base_channels
    if not isinstance(chosen, (list, tuple)) or len(chosen) == 0:
        raise ValueError(f"data.stage_channels[{cand!r}] must be a non-empty list of channel names")

    chosen_clean = [clean_channel_name_local(ch) for ch in chosen]
    base_set = set(base_channels)
    missing = [ch for ch in chosen_clean if ch not in base_set]
    if missing:
        raise ValueError(
            f"Stage channel subset for {stage_key} includes channels not in data.channels: {missing}"
        )
    if len(set(chosen_clean)) != len(chosen_clean):
        raise ValueError(f"Stage channel subset for {stage_key} contains duplicates")
    return chosen_clean


def resolve_stage_channel_indices(data_cfg: dict[str, Any], stage_key: str) -> tuple[np.ndarray, list[str]]:
    base_channels = [clean_channel_name_local(ch) for ch in data_cfg.get("channels", [])]
    chosen_channels = resolve_stage_channel_names(data_cfg, stage_key)
    index_map = {ch: i for i, ch in enumerate(base_channels)}
    idx_base = np.asarray([index_map[ch] for ch in chosen_channels], dtype=np.int64)

    bands_cfg = data_cfg.get("filter_bank_bands") or []
    n_bands = int(len(bands_cfg)) if bands_cfg else 1
    if n_bands <= 1:
        return idx_base, chosen_channels

    expanded_idx_parts: list[np.ndarray] = []
    expanded_names: list[str] = []
    base_count = int(len(base_channels))
    for b in range(n_bands):
        offset = b * base_count
        expanded_idx_parts.append(idx_base + offset)
        expanded_names.extend([f"B{b+1}:{ch}" for ch in chosen_channels])
    idx = np.concatenate(expanded_idx_parts, axis=0).astype(np.int64, copy=False)
    return idx, expanded_names
