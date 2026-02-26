from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def json_dumps_canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha1_hex_from_obj(obj: Any) -> str:
    return hashlib.sha1(json_dumps_canonical(obj).encode("utf-8")).hexdigest()


def write_json(path: str | Path, obj: Any, indent: int = 2) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, sort_keys=True, default=str)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def set_matplotlib_env(base_dir: str | Path | None = None) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    if base_dir is None:
        base = Path.cwd() / "outputs" / ".mplconfig"
    else:
        base = Path(base_dir) / ".mplconfig"
    try:
        ensure_dir(base)
        os.environ.setdefault("MPLCONFIGDIR", str(base))
    except Exception:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")


def to_serializable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    return obj
