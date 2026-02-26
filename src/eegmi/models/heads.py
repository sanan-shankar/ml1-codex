from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch.nn as nn

from eegmi.models.eegnet import EEGNet
from eegmi.models.fbcnet import FBCNetLite
from eegmi.models.fusion import EEGFusionNetLite
from eegmi.models.shallow_fbcsp import ShallowFBCSPNet


@dataclass
class HierarchicalBundle:
    stage_a: nn.Module
    stage_b: nn.Module
    stage_b_finetuned: nn.Module | None = None


def build_model(model_cfg: dict[str, Any], *, n_chans: int, n_times: int, n_classes: int) -> nn.Module:
    model_type = str(model_cfg.get("type", "eegnet")).lower()
    dropout = float(model_cfg.get("dropout", 0.35))
    if model_type == "eegnet":
        params = dict(model_cfg.get("eegnet", {}))
        return EEGNet(n_chans=n_chans, n_times=n_times, n_classes=n_classes, dropout=dropout, **params)
    if model_type in {"shallow", "shallowfbcsp", "shallow_fbcsp"}:
        params = dict(model_cfg.get("shallow", {}))
        return ShallowFBCSPNet(n_chans=n_chans, n_times=n_times, n_classes=n_classes, dropout=dropout, **params)
    if model_type in {"fbcnet", "fbcnet_lite", "fbcnetlite"}:
        params = dict(model_cfg.get("fbcnet", {}))
        return FBCNetLite(n_chans=n_chans, n_times=n_times, n_classes=n_classes, dropout=dropout, **params)
    if model_type in {"fusion", "eegfusion", "eegnet_fusion", "eegfusionnet"}:
        params = dict(model_cfg.get("fusion", {}))
        return EEGFusionNetLite(n_chans=n_chans, n_times=n_times, n_classes=n_classes, dropout=dropout, **params)
    raise ValueError(f"Unsupported model type: {model_type}")
