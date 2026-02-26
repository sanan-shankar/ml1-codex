from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from eegmi.utils import ensure_dir


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    torch.save(payload, p)
    return p


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location, weights_only=False)
