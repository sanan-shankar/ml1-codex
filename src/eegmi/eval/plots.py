from __future__ import annotations

from pathlib import Path
from typing import Any

from eegmi.utils import set_matplotlib_env

set_matplotlib_env()

import matplotlib.pyplot as plt
import numpy as np


def save_confusion_matrix_png(
    cm: np.ndarray,
    class_names: list[str],
    out_path: str | Path,
    *,
    title: str,
    normalize: bool = False,
) -> None:
    cm_arr = np.asarray(cm, dtype=float)
    display = cm_arr.copy()
    if normalize:
        row_sums = display.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        display = display / row_sums
    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    im = ax.imshow(display, cmap="Blues", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    thresh = float(display.max() * 0.6) if display.size else 0.5
    for i in range(display.shape[0]):
        for j in range(display.shape[1]):
            text_val = f"{display[i, j]:.2f}" if normalize else f"{int(cm_arr[i, j])}"
            ax.text(j, i, text_val, ha="center", va="center", color="white" if display[i, j] > thresh else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def save_learning_curves(history: list[dict[str, Any]], out_path: str | Path, *, title: str) -> None:
    if not history:
        return
    epochs = [int(h["epoch"]) for h in history]
    tr_loss = [float(h["train"]["loss"]) for h in history]
    va_loss = [float(h["val"]["loss"]) for h in history]
    tr_bal = [float(h["train"]["balanced_accuracy"]) for h in history]
    va_bal = [float(h["val"]["balanced_accuracy"]) for h in history]
    tr_f1 = [float(h["train"]["macro_f1"]) for h in history]
    va_f1 = [float(h["val"]["macro_f1"]) for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=120)
    axes[0].plot(epochs, tr_loss, label="train")
    axes[0].plot(epochs, va_loss, label="val")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs, tr_bal, label="train")
    axes[1].plot(epochs, va_bal, label="val")
    axes[1].set_title("Balanced Acc")
    axes[1].legend()

    axes[2].plot(epochs, tr_f1, label="train")
    axes[2].plot(epochs, va_f1, label="val")
    axes[2].set_title("Macro-F1")
    axes[2].legend()

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.2)
    fig.suptitle(title)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _mean_psd(x: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    # x: [N,C,T]
    if x.size == 0:
        return np.array([]), np.array([])
    x = np.asarray(x, dtype=np.float32)
    spec = np.fft.rfft(x, axis=-1)
    psd = (np.abs(spec) ** 2).mean(axis=(0, 1))
    freqs = np.fft.rfftfreq(x.shape[-1], d=1.0 / sfreq)
    return freqs, psd


def save_psd_rest_vs_active(X: np.ndarray, y: np.ndarray, sfreq: float, out_path: str | Path) -> None:
    X = np.asarray(X)
    y = np.asarray(y)
    X_rest = X[y == 0]
    X_active = X[y > 0]
    f_r, p_r = _mean_psd(X_rest, sfreq)
    f_a, p_a = _mean_psd(X_active, sfreq)
    if f_r.size == 0 or f_a.size == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
    ax.plot(f_r, 10 * np.log10(np.maximum(p_r, 1e-12)), label="REST")
    ax.plot(f_a, 10 * np.log10(np.maximum(p_a, 1e-12)), label="ACTIVE")
    ax.set_xlim(0, min(45, f_r.max()))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.set_title("PSD: REST vs ACTIVE")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _band_power_per_channel(X: np.ndarray, sfreq: float, band: tuple[float, float]) -> np.ndarray:
    if X.size == 0:
        return np.zeros((X.shape[1],), dtype=np.float32)
    spec = np.fft.rfft(X, axis=-1)
    psd = (np.abs(spec) ** 2)
    freqs = np.fft.rfftfreq(X.shape[-1], d=1.0 / sfreq)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask):
        return np.zeros((X.shape[1],), dtype=np.float32)
    # psd[..., mask] -> [N, C, F]; average over frequency then over epochs
    bp = psd[..., mask].mean(axis=-1)  # [N, C]
    return bp.mean(axis=0).astype(np.float32)


def save_erd_topomap(
    X: np.ndarray,
    y: np.ndarray,
    ch_names: list[str],
    sfreq: float,
    out_path: str | Path,
) -> None:
    try:
        import mne
    except Exception:
        return
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 3 or X.shape[1] != len(ch_names):
        # Filter-bank / channel-stacked inputs are not compatible with a single-channel topomap.
        return
    if any(str(ch).upper().endswith("_LAP") for ch in ch_names):
        # Synthetic channels do not have physical scalp coordinates for topomaps.
        return
    rest = X[y == 0]
    active = X[y > 0]
    if rest.size == 0 or active.size == 0:
        return

    alpha = (8.0, 13.0)
    beta = (13.0, 30.0)
    alpha_rest = _band_power_per_channel(rest, sfreq, alpha)
    alpha_active = _band_power_per_channel(active, sfreq, alpha)
    beta_rest = _band_power_per_channel(rest, sfreq, beta)
    beta_active = _band_power_per_channel(active, sfreq, beta)

    erd_alpha = (alpha_active - alpha_rest) / np.maximum(alpha_rest, 1e-12)
    erd_beta = (beta_active - beta_rest) / np.maximum(beta_rest, 1e-12)

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    try:
        info.set_montage(montage, match_case=False, on_missing="ignore")
    except TypeError:
        info.set_montage(montage, on_missing="ignore")

    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=120)
        mne.viz.plot_topomap(erd_alpha, info, axes=axes[0], show=False, contours=0, cmap="RdBu_r")
        axes[0].set_title("ERD proxy: 8-13 Hz")
        mne.viz.plot_topomap(erd_beta, info, axes=axes[1], show=False, contours=0, cmap="RdBu_r")
        axes[1].set_title("ERD proxy: 13-30 Hz")
        fig.suptitle("Active vs Rest Relative Band Power")
        fig.tight_layout()
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        try:
            plt.close("all")
        except Exception:
            pass
        print(f"[plot] ERD topomap skipped due to error: {e}")
