from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Any

from eegmi.utils import ensure_dir, to_serializable


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(to_serializable(row), sort_keys=True) + "\n")


def save_history(path: str | Path, history: list[dict[str, Any]]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump([to_serializable(h) for h in history], f, indent=2, sort_keys=True)


def save_history_csv(path: str | Path, history: list[dict[str, Any]]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "train_bal_acc",
        "val_bal_acc",
        "train_macro_f1",
        "val_macro_f1",
    ]
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for h in history:
            writer.writerow(
                {
                    "epoch": int(h.get("epoch", 0)),
                    "train_loss": float(h.get("train", {}).get("loss", 0.0)),
                    "val_loss": float(h.get("val", {}).get("loss", 0.0)),
                    "train_bal_acc": float(h.get("train", {}).get("balanced_accuracy", 0.0)),
                    "val_bal_acc": float(h.get("val", {}).get("balanced_accuracy", 0.0)),
                    "train_macro_f1": float(h.get("train", {}).get("macro_f1", 0.0)),
                    "val_macro_f1": float(h.get("val", {}).get("macro_f1", 0.0)),
                }
            )
