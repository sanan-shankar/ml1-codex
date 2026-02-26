from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EEGAugment:
    gaussian_noise_std: float = 0.0
    time_shift_max: int = 0
    amplitude_scale_min: float = 1.0
    amplitude_scale_max: float = 1.0
    seed: int = 42

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def _time_shift(self, x: np.ndarray, shift: int) -> np.ndarray:
        if shift == 0:
            return x
        out = np.zeros_like(x)
        if shift > 0:
            out[:, shift:] = x[:, :-shift]
        else:
            out[:, :shift] = x[:, -shift:]
        return out

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y = x.astype(np.float32, copy=True)
        if self.amplitude_scale_max > self.amplitude_scale_min:
            scale = self.rng.uniform(self.amplitude_scale_min, self.amplitude_scale_max)
            y *= np.float32(scale)
        if self.time_shift_max > 0:
            shift = int(self.rng.integers(-self.time_shift_max, self.time_shift_max + 1))
            y = self._time_shift(y, shift)
        if self.gaussian_noise_std > 0:
            noise = self.rng.normal(0.0, self.gaussian_noise_std, size=y.shape).astype(np.float32)
            y += noise
        return y


def build_augmenter(cfg: dict | None, seed: int = 42) -> EEGAugment | None:
    if not cfg or not cfg.get("enabled", False):
        return None
    return EEGAugment(
        gaussian_noise_std=float(cfg.get("gaussian_noise_std", 0.0)),
        time_shift_max=int(cfg.get("time_shift_max", 0)),
        amplitude_scale_min=float(cfg.get("amplitude_scale_min", 1.0)),
        amplitude_scale_max=float(cfg.get("amplitude_scale_max", 1.0)),
        seed=int(seed),
    )
