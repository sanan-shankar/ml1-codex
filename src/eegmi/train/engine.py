from __future__ import annotations

import copy
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from eegmi.data.dataset import EEGEpochDataset, stage_targets
from eegmi.data.smote import smote_cfg_for_stage, smote_like_augment_dataset
from eegmi.train.checkpointing import save_checkpoint
from eegmi.train.logging import append_jsonl, save_history, save_history_csv
from eegmi.train.losses import make_cross_entropy_loss
from eegmi.train.metrics import classification_metrics
from eegmi.train.samplers import make_balanced_sampler
from eegmi.utils import ensure_dir

try:
    import resource
except Exception:  # pragma: no cover
    resource = None


def _dataset_targets(dataset: EEGEpochDataset) -> np.ndarray:
    return stage_targets(dataset.y, dataset.stage)


def _encode_groups(keys: list[tuple[Any, ...]] | np.ndarray) -> np.ndarray:
    if isinstance(keys, np.ndarray) and np.issubdtype(keys.dtype, np.integer):
        return keys.astype(np.int64, copy=False)
    uniq: dict[Any, int] = {}
    out = np.zeros((len(keys),), dtype=np.int64)
    for i, k in enumerate(keys):
        if k not in uniq:
            uniq[k] = len(uniq)
        out[i] = uniq[k]
    return out


def _dataset_sampler_groups(dataset: EEGEpochDataset, strategy: str) -> np.ndarray:
    strategy = str(strategy or "targets").lower()
    if strategy in {"targets", "target", "class", "stage_targets"}:
        return _dataset_targets(dataset)
    if strategy == "stage_a_label":
        if dataset.stage != "stage_a":
            return _dataset_targets(dataset)
        return _encode_groups(dataset.y.astype(np.int64, copy=False))
    if strategy in {"stage_a_runkind_label", "stage_a_run_label"}:
        if dataset.stage != "stage_a":
            return _dataset_targets(dataset)
        rk = dataset.meta["run_kind"].astype(str).to_numpy()
        yy = dataset.y.astype(np.int64, copy=False)
        keys = [(rk_i, int(y_i)) for rk_i, y_i in zip(rk.tolist(), yy.tolist())]
        return _encode_groups(keys)
    if strategy in {"stage_a_runkind_task_label", "stage_a_run_task_label"}:
        if dataset.stage != "stage_a":
            return _dataset_targets(dataset)
        rk = dataset.meta["run_kind"].astype(str).to_numpy()
        tt = dataset.meta["task_type"].astype(str).str.upper().to_numpy()
        yy = dataset.y.astype(np.int64, copy=False)
        keys = [(rk_i, tt_i, int(y_i)) for rk_i, tt_i, y_i in zip(rk.tolist(), tt.tolist(), yy.tolist())]
        return _encode_groups(keys)
    raise ValueError(f"Unsupported sampler_strategy: {strategy}")


def _move_batch(batch, device: str):
    x, y = batch
    return x.to(device), y.to(device)


def _rss_mb() -> float | None:
    if resource is None:
        return None
    try:
        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None
    # macOS reports bytes, Linux usually KiB.
    if sys.platform == "darwin":
        return rss / (1024.0 * 1024.0)
    if rss > 10_000_000:
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def run_inference(model: torch.nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    all_true = []
    all_pred = []
    losses = []
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            x, y = _move_batch(batch, device)
            logits = model(x)
            loss = criterion(logits, y)
            losses.append(float(loss.item()))
            pred = torch.argmax(logits, dim=1)
            all_true.append(y.detach().cpu().numpy())
            all_pred.append(pred.detach().cpu().numpy())
    if not all_true:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64), 0.0
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    return y_true, y_pred, float(np.mean(losses)) if losses else 0.0


def _train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.train()
    all_true = []
    all_pred = []
    losses = []
    for batch in loader:
        x, y = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        pred = torch.argmax(logits.detach(), dim=1)
        all_true.append(y.detach().cpu().numpy())
        all_pred.append(pred.cpu().numpy())
    y_true = np.concatenate(all_true) if all_true else np.empty((0,), dtype=np.int64)
    y_pred = np.concatenate(all_pred) if all_pred else np.empty((0,), dtype=np.int64)
    return y_true, y_pred, float(np.mean(losses)) if losses else 0.0


def _make_loader(
    dataset: EEGEpochDataset,
    *,
    batch_size: int,
    num_workers: int,
    train: bool,
    use_balanced_sampler: bool,
    sampler_strategy: str = "targets",
) -> DataLoader:
    sampler = None
    shuffle = train
    if train and use_balanced_sampler:
        groups = _dataset_sampler_groups(dataset, sampler_strategy)
        sampler = make_balanced_sampler(groups)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=False,
    )


def train_stage(
    model: torch.nn.Module,
    *,
    stage_name: str,
    class_names: list[str],
    train_dataset: EEGEpochDataset,
    val_dataset: EEGEpochDataset,
    train_cfg: dict[str, Any],
    output_dir: str | Path,
    device: str = "cpu",
    weighted_loss: bool = True,
    use_balanced_sampler: bool = True,
) -> dict[str, Any]:
    out_dir = ensure_dir(output_dir)
    n_classes = len(class_names)
    batch_size = int(train_cfg.get("batch_size", 64))
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    max_epochs = int(train_cfg.get("max_epochs", 40))
    patience = int(train_cfg.get("patience", 8))
    min_delta = float(train_cfg.get("min_delta", 0.0))
    num_workers = int(train_cfg.get("num_workers", 0))
    sampler_strategy = str(train_cfg.get("sampler_strategy", "targets"))

    smote_cfg = smote_cfg_for_stage(train_cfg, stage_name)
    if smote_cfg is not None:
        train_dataset = smote_like_augment_dataset(
            train_dataset,
            ratio=float(smote_cfg.get("ratio", 1.0)),
            max_new_samples=smote_cfg.get("max_new_samples"),
            lambda_min=float(smote_cfg.get("lambda_min", 0.1)),
            lambda_max=float(smote_cfg.get("lambda_max", 0.9)),
            seed=int(smote_cfg.get("seed", 42)),
            stage_name=stage_name,
        )

    train_loader = _make_loader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        train=True,
        use_balanced_sampler=use_balanced_sampler,
        sampler_strategy=sampler_strategy,
    )
    val_loader = _make_loader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        train=False,
        use_balanced_sampler=False,
    )

    model = model.to(device)
    train_targets = _dataset_targets(train_dataset)
    criterion, class_weights = make_cross_entropy_loss(
        train_targets,
        n_classes=n_classes,
        weighted=weighted_loss,
        device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: list[dict[str, Any]] = []
    jsonl_path = out_dir / f"{stage_name}_epochs.jsonl"
    best_state = copy.deepcopy(model.state_dict())
    best_metrics = None
    best_epoch = -1
    best_score = -float("inf")
    bad_epochs = 0

    epoch_bar = tqdm(range(1, max_epochs + 1), desc=f"Train {stage_name}", leave=False)
    for epoch in epoch_bar:
        t0 = time.perf_counter()
        y_tr, p_tr, tr_loss = _train_one_epoch(model, train_loader, optimizer, criterion, device)
        y_va, p_va, va_loss = run_inference(model, val_loader, device)
        epoch_sec = float(time.perf_counter() - t0)
        rss_mb = _rss_mb()

        train_metrics = classification_metrics(y_tr, p_tr, labels=list(range(n_classes)), label_names=class_names)
        val_metrics = classification_metrics(y_va, p_va, labels=list(range(n_classes)), label_names=class_names)

        row = {
            "epoch": epoch,
            "train": {"loss": tr_loss, **train_metrics.to_dict()},
            "val": {"loss": va_loss, **val_metrics.to_dict()},
            "class_weights": class_weights.tolist() if class_weights is not None else None,
            "system": {"rss_mb": rss_mb, "epoch_sec": epoch_sec},
        }
        history.append(row)
        append_jsonl(jsonl_path, row)

        score = float(val_metrics.balanced_accuracy + 0.5 * val_metrics.macro_f1)
        improved = score > (best_score + min_delta)
        if improved:
            best_score = score
            best_epoch = epoch
            bad_epochs = 0
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = {"loss": va_loss, **val_metrics.to_dict()}
            save_checkpoint(
                out_dir / f"best_{stage_name}.pt",
                {
                    "stage": stage_name,
                    "epoch": epoch,
                    "model_state_dict": best_state,
                    "best_score": best_score,
                    "best_metrics": best_metrics,
                    "class_names": class_names,
                    "n_classes": n_classes,
                    "history": history,
                },
            )
        else:
            bad_epochs += 1

        epoch_bar.set_postfix(
            val_bal_acc=f"{val_metrics.balanced_accuracy:.3f}",
            val_f1=f"{val_metrics.macro_f1:.3f}",
            val_loss=f"{va_loss:.3f}",
            rss_gb=(f"{rss_mb/1024.0:.2f}" if rss_mb is not None else "na"),
            ep_s=f"{epoch_sec:.1f}",
            bad=bad_epochs,
        )

        if bad_epochs >= patience:
            break

    model.load_state_dict(best_state)
    save_checkpoint(
        out_dir / f"last_{stage_name}.pt",
        {
            "stage": stage_name,
            "epoch": len(history),
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "n_classes": n_classes,
            "history": history,
        },
    )
    save_history(out_dir / f"history_{stage_name}.json", history)
    save_history_csv(out_dir / f"history_{stage_name}.csv", history)

    return {
        "stage": stage_name,
        "model": model,
        "class_names": class_names,
        "n_classes": n_classes,
        "history": history,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_metrics": best_metrics or {},
        "class_weights": class_weights.tolist() if class_weights is not None else None,
    }
