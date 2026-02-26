from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass
class ClassificationMetrics:
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    confusion_matrix: list[list[int]]
    support: dict[str, int]
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    per_class_f1: dict[str, float]
    predicted_counts: dict[str, int]
    true_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": float(self.accuracy),
            "balanced_accuracy": float(self.balanced_accuracy),
            "macro_f1": float(self.macro_f1),
            "confusion_matrix": self.confusion_matrix,
            "support": {str(k): int(v) for k, v in self.support.items()},
            "per_class_precision": {str(k): float(v) for k, v in self.per_class_precision.items()},
            "per_class_recall": {str(k): float(v) for k, v in self.per_class_recall.items()},
            "per_class_f1": {str(k): float(v) for k, v in self.per_class_f1.items()},
            "predicted_counts": {str(k): int(v) for k, v in self.predicted_counts.items()},
            "true_counts": {str(k): int(v) for k, v in self.true_counts.items()},
        }


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[int]) -> np.ndarray:
    label_to_idx = {int(lbl): i for i, lbl in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        if int(t) not in label_to_idx or int(p) not in label_to_idx:
            continue
        cm[label_to_idx[int(t)], label_to_idx[int(p)]] += 1
    return cm


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def classification_metrics(
    y_true: np.ndarray | Iterable[int],
    y_pred: np.ndarray | Iterable[int],
    *,
    labels: Sequence[int],
    label_names: Sequence[str] | None = None,
) -> ClassificationMetrics:
    y_true_arr = np.asarray(list(y_true) if not isinstance(y_true, np.ndarray) else y_true, dtype=np.int64)
    y_pred_arr = np.asarray(list(y_pred) if not isinstance(y_pred, np.ndarray) else y_pred, dtype=np.int64)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(f"y_true/y_pred shape mismatch: {y_true_arr.shape} vs {y_pred_arr.shape}")
    if y_true_arr.ndim != 1:
        raise ValueError("classification_metrics expects 1D arrays")
    if label_names is None:
        label_names = [str(int(lbl)) for lbl in labels]
    if len(label_names) != len(labels):
        raise ValueError("label_names length must match labels")

    cm = confusion_matrix_np(y_true_arr, y_pred_arr, labels)
    total = int(cm.sum())
    acc = _safe_div(np.trace(cm), total)

    recalls = []
    f1s = []
    per_prec = {}
    per_rec = {}
    per_f1 = {}
    support = {}
    true_counts = {}
    pred_counts = {}

    for i, name in enumerate(label_names):
        tp = float(cm[i, i])
        fn = float(cm[i, :].sum() - cm[i, i])
        fp = float(cm[:, i].sum() - cm[i, i])
        supp = int(cm[i, :].sum())
        support[name] = supp
        true_counts[name] = supp
        pred_counts[name] = int(cm[:, i].sum())
        prec = _safe_div(tp, tp + fp)
        rec = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * prec * rec, prec + rec) if (prec + rec) > 0 else 0.0
        per_prec[name] = prec
        per_rec[name] = rec
        per_f1[name] = f1
        if supp > 0:
            recalls.append(rec)
            f1s.append(f1)

    bal_acc = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    return ClassificationMetrics(
        accuracy=acc,
        balanced_accuracy=bal_acc,
        macro_f1=macro_f1,
        confusion_matrix=cm.astype(int).tolist(),
        support=support,
        per_class_precision=per_prec,
        per_class_recall=per_rec,
        per_class_f1=per_f1,
        predicted_counts=pred_counts,
        true_counts=true_counts,
    )


def summarize_loss_and_metrics(loss: float, metrics: ClassificationMetrics) -> dict[str, Any]:
    out = {"loss": float(loss)}
    out.update(metrics.to_dict())
    return out
