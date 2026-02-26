from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

from eegmi.utils import set_matplotlib_env

set_matplotlib_env()

import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from tqdm.auto import tqdm

from eegmi.constants import ALL_RUNS, META_COLUMNS, describe_run, label_name, map_event_to_label
from eegmi.data.cache import cache_exists, load_subject_cache, save_subject_cache


EVENT_ID = {"T0": 1, "T1": 2, "T2": 3}
_STANDARD_1020_MONTAGE = mne.channels.make_standard_montage("standard_1020")
_MONTAGE_NAME_BY_UPPER = {str(name).upper(): str(name) for name in _STANDARD_1020_MONTAGE.ch_names}
_VIRTUAL_CHANNEL_RECIPES: dict[str, dict[str, Any]] = {
    "CZ_LAP": {"center": "CZ", "neighbors": ["C1", "C2", "FCZ", "CPZ"]},
    "CPZ_LAP": {"center": "CPZ", "neighbors": ["CP1", "CP2", "CZ", "PZ"]},
    "FCZ_LAP": {"center": "FCZ", "neighbors": ["FC1", "FC2", "FZ", "CZ"]},
}


def expected_n_times(sfreq: float, tmin: float, tmax: float) -> int:
    return int(round((tmax - tmin) * sfreq)) + 1


def parse_filter_bank_bands(data_cfg: dict[str, Any]) -> list[tuple[float, float]]:
    bands_cfg = data_cfg.get("filter_bank_bands")
    if not bands_cfg:
        return []
    bands: list[tuple[float, float]] = []
    for band in bands_cfg:
        if not isinstance(band, (list, tuple)) or len(band) != 2:
            raise ValueError(f"Each filter_bank_bands entry must be [low, high], got {band}")
        lo, hi = float(band[0]), float(band[1])
        if not (0.0 < lo < hi):
            raise ValueError(f"Invalid filter-bank band: {band}")
        bands.append((lo, hi))
    return bands


def effective_n_channels(data_cfg: dict[str, Any]) -> int:
    base = len(data_cfg["channels"])
    bands = parse_filter_bank_bands(data_cfg)
    return int(base * max(1, len(bands)))


def is_virtual_channel_name(name: str) -> bool:
    return clean_channel_name(name) in _VIRTUAL_CHANNEL_RECIPES


def split_real_and_virtual_requested_channels(channels: list[str]) -> tuple[list[str], list[str]]:
    cleaned = [clean_channel_name(ch) for ch in channels]
    real = [ch for ch in cleaned if ch not in _VIRTUAL_CHANNEL_RECIPES]
    virt = [ch for ch in cleaned if ch in _VIRTUAL_CHANNEL_RECIPES]
    return real, virt


def clean_channel_name(name: str) -> str:
    return str(name).replace(".", "").strip().upper()


def clean_raw_channel_names(raw: mne.io.BaseRaw) -> None:
    mapping = {}
    seen = set()
    for ch in raw.ch_names:
        cleaned = clean_channel_name(ch)
        if cleaned in seen and cleaned != ch:
            raise ValueError(f"Duplicate channel after cleaning: {ch} -> {cleaned}")
        seen.add(cleaned)
        if ch != cleaned:
            mapping[ch] = cleaned
    if mapping:
        raw.rename_channels(mapping)


def canonical_montage_name(name: str) -> str:
    """Map cleaned uppercase names to canonical standard_1020 names when available."""
    cleaned = clean_channel_name(name)
    return _MONTAGE_NAME_BY_UPPER.get(cleaned, cleaned)


def set_montage_for_plots(raw: mne.io.BaseRaw) -> None:
    montage = _STANDARD_1020_MONTAGE
    try:
        raw.set_montage(montage, match_case=False, on_missing="ignore", verbose="ERROR")
    except TypeError:
        raw.set_montage(montage, on_missing="ignore", verbose="ERROR")


def validate_and_pick_channels(raw: mne.io.BaseRaw, channels: list[str]) -> None:
    missing = [ch for ch in channels if ch not in raw.ch_names]
    if missing:
        raise ValueError(
            f"Missing requested channels after cleaning. Missing={missing}. Available sample={raw.ch_names[:10]}"
        )
    raw.pick(channels, verbose="ERROR")
    if raw.ch_names != channels:
        raise AssertionError("Channel order mismatch after pick; expected stable requested order")


def apply_reference(raw: mne.io.BaseRaw, reref: Any) -> None:
    if reref is None or str(reref).lower() in {"none", "null", ""}:
        return
    mode = str(reref).lower()
    if mode in {"average", "avg", "car", "common_average"}:
        raw.set_eeg_reference(ref_channels="average", projection=False, verbose="ERROR")
        return
    raise ValueError(f"Unsupported reref mode: {reref}")


def apply_csd(raw: mne.io.BaseRaw, enabled: Any) -> mne.io.BaseRaw:
    if not enabled:
        return raw
    if not hasattr(mne.preprocessing, "compute_current_source_density"):
        raise RuntimeError("MNE current source density (CSD) is not available in this mne version")
    try:
        with mne.utils.use_log_level("ERROR"):
            mne.preprocessing.compute_current_source_density(raw, copy=False)
        return raw
    except TypeError:
        # Older MNE versions return a new object and may not support copy=False.
        with mne.utils.use_log_level("ERROR"):
            return mne.preprocessing.compute_current_source_density(raw)


def _resolve_ica_n_components(n_components: Any, n_chans: int) -> Any:
    if n_components is None:
        return None
    if isinstance(n_components, int):
        return max(1, min(int(n_components), max(1, n_chans - 1)))
    try:
        val = float(n_components)
    except Exception:
        return n_components
    if val > 1.0:
        return max(1, min(int(round(val)), max(1, n_chans - 1)))
    return val


def apply_ica_artifact_cleanup(raw: mne.io.BaseRaw, data_cfg: dict[str, Any]) -> mne.io.BaseRaw:
    ica_cfg = data_cfg.get("ica")
    if not isinstance(ica_cfg, dict) or not ica_cfg.get("enabled", False):
        return raw
    if not hasattr(mne.preprocessing, "ICA"):
        raise RuntimeError("MNE ICA is not available in this mne version")

    # Use a high-pass filtered copy for ICA fitting (MNE recommendation), apply to the original raw.
    fit_l = float(ica_cfg.get("fit_l_freq", 1.0))
    fit_h = ica_cfg.get("fit_h_freq")
    fit_h = None if fit_h in {None, "none", "null"} else float(fit_h)
    decim = max(1, int(ica_cfg.get("decim", 4)))
    method = str(ica_cfg.get("method", "fastica"))
    max_iter = ica_cfg.get("max_iter", 256)
    eog_threshold = float(ica_cfg.get("eog_threshold", 3.0))
    max_eog_components = max(0, int(ica_cfg.get("max_eog_components", 4)))
    proxy_channels = [clean_channel_name(ch) for ch in ica_cfg.get("eog_proxy_channels", [])]

    # Fit on EEG channels before task-channel subsetting so frontal proxies are available.
    eeg_ch_names = []
    for idx, ch_name in enumerate(raw.ch_names):
        if mne.channel_type(raw.info, idx) == "eeg":
            eeg_ch_names.append(ch_name)
    if len(eeg_ch_names) < 8:
        return raw

    fit_raw = raw.copy().pick(eeg_ch_names, verbose="ERROR")
    fit_raw.filter(fit_l, fit_h, verbose="ERROR")
    apply_reference(fit_raw, data_cfg.get("reref"))
    n_components = _resolve_ica_n_components(ica_cfg.get("n_components", 0.99), len(fit_raw.ch_names))

    with mne.utils.use_log_level("ERROR"):
        ica = mne.preprocessing.ICA(
            n_components=n_components,
            method=method,
            max_iter=max_iter,
            random_state=42,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="FastICA did not converge.*")
            ica.fit(fit_raw, decim=decim)

    exclude: set[int] = set()
    available_proxy = [ch for ch in proxy_channels if ch in fit_raw.ch_names]
    for ch in available_proxy:
        with mne.utils.use_log_level("ERROR"):
            bads, _scores = ica.find_bads_eog(fit_raw, ch_name=ch, threshold=eog_threshold)
        for comp in bads[:max_eog_components]:
            exclude.add(int(comp))

    if exclude:
        ica.exclude = sorted(exclude)
        with mne.utils.use_log_level("ERROR"):
            ica.apply(raw)
        print(
            f"[loader][ica] Removed {len(ica.exclude)} comps "
            f"(proxies={available_proxy[:4]}{'...' if len(available_proxy) > 4 else ''})"
        )
    else:
        print("[loader][ica] No EOG-like components detected; skipping ICA apply")
    return raw


def _rest_sample_mask(raw: mne.io.BaseRaw) -> np.ndarray:
    mask = np.zeros(raw.n_times, dtype=bool)
    ann = raw.annotations
    for onset, duration, desc in zip(ann.onset, ann.duration, ann.description):
        if str(desc).upper() != "T0":
            continue
        start = int(np.floor(onset * raw.info["sfreq"]))
        stop = int(np.ceil((onset + duration) * raw.info["sfreq"]))
        start = max(0, start)
        stop = min(raw.n_times, stop)
        if stop > start:
            mask[start:stop] = True
    return mask


def apply_rest_zscore(raw: mne.io.BaseRaw) -> None:
    if not getattr(raw, "preload", False):
        raise ValueError("REST z-score requires preloaded raw data")
    data = raw._data  # MNE Raw stores preloaded samples here; mutate in place.
    mask = _rest_sample_mask(raw)
    if not mask.any():
        raise ValueError("REST z-score requested but no T0 samples were found in annotations")
    rest = data[:, mask]
    mu = rest.mean(axis=1, keepdims=True)
    sigma = rest.std(axis=1, keepdims=True)
    sigma[sigma < 1e-6] = 1.0
    data[:] = (data - mu) / sigma


def _inv_sqrtm_spd(mat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    w, v = np.linalg.eigh(mat.astype(np.float64, copy=False))
    w = np.maximum(w, eps)
    inv_sqrt = (v * (1.0 / np.sqrt(w))[None, :]) @ v.T
    return inv_sqrt.astype(np.float32)


def _resolve_window_indices_from_spec(
    *,
    spec: Any,
    data_cfg: dict[str, Any],
    n_times: int,
) -> np.ndarray:
    if spec is None or spec is False:
        return np.arange(int(n_times), dtype=np.int64)
    if isinstance(spec, dict):
        if "tmin" not in spec or "tmax" not in spec:
            raise ValueError("EA window spec dict must include tmin and tmax")
        tmin = float(spec["tmin"])
        tmax = float(spec["tmax"])
    elif isinstance(spec, (list, tuple)) and len(spec) == 2:
        tmin = float(spec[0])
        tmax = float(spec[1])
    else:
        raise ValueError("EA window spec must be [tmin,tmax] or {'tmin':..,'tmax':..}")
    if tmax <= tmin:
        raise ValueError(f"Invalid EA time window: tmin={tmin}, tmax={tmax}")
    global_tmin = float(data_cfg["tmin"])
    global_tmax = float(data_cfg["tmax"])
    sfreq = float(data_cfg["sfreq"])
    if tmin < global_tmin - 1e-9 or tmax > global_tmax + 1e-9:
        raise ValueError(
            f"EA time window ({tmin},{tmax}) must be within loaded epoch window ({global_tmin},{global_tmax})"
        )
    start_f = (tmin - global_tmin) * sfreq
    stop_f = (tmax - global_tmin) * sfreq
    start = int(round(start_f))
    stop = int(round(stop_f)) + 1
    if abs(start - start_f) > 1e-4 or abs((stop - 1) - stop_f) > 1e-4:
        raise ValueError(f"EA time window is not aligned to sample grid: {spec}")
    if start < 0 or stop > int(n_times) or stop <= start:
        raise ValueError(f"EA time indices out of bounds: start={start}, stop={stop}, n_times={n_times}")
    return np.arange(start, stop, dtype=np.int64)


def apply_euclidean_alignment_epochs(
    X: np.ndarray,
    eps: float = 1e-6,
    ref_time_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Subject-wise Euclidean alignment (EA): R^{-1/2} X for each epoch.

    X shape [N, C, T]. Uses unlabeled subject epochs only (safe for test subjects).
    """
    if X.ndim != 3:
        raise ValueError(f"EA expects X [N,C,T], got {X.shape}")
    if X.shape[0] == 0:
        return X
    X_ref = X
    if ref_time_indices is not None:
        idx = np.asarray(ref_time_indices, dtype=np.int64)
        if idx.ndim != 1:
            raise ValueError(f"EA ref_time_indices must be 1D, got shape={idx.shape}")
        if idx.size == 0:
            raise ValueError("EA ref_time_indices cannot be empty")
        if int(idx.min()) < 0 or int(idx.max()) >= X.shape[2]:
            raise IndexError("EA ref_time_indices out of bounds")
        X_ref = X[:, :, idx]
    cov_sum = np.zeros((X.shape[1], X.shape[1]), dtype=np.float64)
    for i in range(X_ref.shape[0]):
        xi = X_ref[i].astype(np.float64, copy=False)
        cov_sum += (xi @ xi.T) / max(int(xi.shape[1]), 1)
    ref = cov_sum / float(X_ref.shape[0])
    W = _inv_sqrtm_spd(ref, eps=eps)  # [C,C]
    X_aligned = np.einsum("dc,nct->ndt", W, X.astype(np.float32, copy=False), optimize=True).astype(np.float32, copy=False)
    return X_aligned


def apply_preprocessing(raw: mne.io.BaseRaw, data_cfg: dict[str, Any]) -> mne.io.BaseRaw:
    clean_raw_channel_names(raw)
    set_montage_for_plots(raw)
    raw = apply_ica_artifact_cleanup(raw, data_cfg)
    channels_requested = [clean_channel_name(ch) for ch in data_cfg["channels"]]
    real_channels, _virtual_channels = split_real_and_virtual_requested_channels(channels_requested)
    validate_and_pick_channels(raw, real_channels)
    band = data_cfg.get("bandpass")
    if band:
        raw.filter(float(band[0]), float(band[1]), verbose="ERROR")
    notch = data_cfg.get("notch")
    if notch:
        freqs = [float(notch)] if np.isscalar(notch) else [float(v) for v in notch]
        raw.notch_filter(freqs=freqs, verbose="ERROR")
    apply_reference(raw, data_cfg.get("reref"))
    raw = apply_csd(raw, data_cfg.get("csd"))
    if data_cfg.get("rest_zscore", False):
        apply_rest_zscore(raw)
    return raw


def append_virtual_channels_epochs(
    X: np.ndarray,
    *,
    real_channel_names: list[str],
    requested_channel_names: list[str],
) -> np.ndarray:
    """Append built-in virtual channels (e.g., local Laplacian proxies) to match requested order.

    X shape [N, C_real, T]; output shape [N, C_requested, T].
    """
    if X.ndim != 3:
        raise ValueError(f"append_virtual_channels_epochs expects X [N,C,T], got {X.shape}")
    real_names = [clean_channel_name(ch) for ch in real_channel_names]
    req_names = [clean_channel_name(ch) for ch in requested_channel_names]
    if len(real_names) != X.shape[1]:
        raise ValueError(f"real_channel_names length {len(real_names)} does not match X channels {X.shape[1]}")
    if not req_names:
        raise ValueError("requested_channel_names must be non-empty")
    if all(ch not in _VIRTUAL_CHANNEL_RECIPES for ch in req_names):
        # Reorder/pass-through to requested real channel order.
        idx_map = {ch: i for i, ch in enumerate(real_names)}
        missing = [ch for ch in req_names if ch not in idx_map]
        if missing:
            raise ValueError(f"Requested channels missing from real data: {missing}")
        idx = np.asarray([idx_map[ch] for ch in req_names], dtype=np.int64)
        return X[:, idx, :].astype(np.float32, copy=False)

    real_idx_map = {ch: i for i, ch in enumerate(real_names)}
    cache: dict[str, np.ndarray] = {}
    out_parts: list[np.ndarray] = []
    for ch in req_names:
        if ch in cache:
            out_parts.append(cache[ch])
            continue
        if ch in real_idx_map:
            arr = X[:, real_idx_map[ch] : real_idx_map[ch] + 1, :].astype(np.float32, copy=False)
            cache[ch] = arr
            out_parts.append(arr)
            continue
        recipe = _VIRTUAL_CHANNEL_RECIPES.get(ch)
        if recipe is None:
            raise ValueError(f"Unsupported virtual channel: {ch}")
        center = clean_channel_name(recipe["center"])
        neighbors = [clean_channel_name(v) for v in recipe["neighbors"]]
        needed = [center] + neighbors
        missing = [v for v in needed if v not in real_idx_map]
        if missing:
            raise ValueError(f"Virtual channel {ch} requires unavailable channels: {missing}")
        center_x = X[:, real_idx_map[center], :]
        neigh_x = np.stack([X[:, real_idx_map[n], :] for n in neighbors], axis=0).mean(axis=0)
        arr = (center_x - neigh_x).astype(np.float32, copy=False)[:, None, :]
        cache[ch] = arr
        out_parts.append(arr)
    out = np.concatenate(out_parts, axis=1).astype(np.float32, copy=False)
    if out.shape[1] != len(req_names):
        raise AssertionError("Virtual channel assembly produced unexpected channel count")
    return out


def apply_filter_bank_epochs(X: np.ndarray, sfreq: float, bands: list[tuple[float, float]], order: int = 4) -> np.ndarray:
    """Apply sub-band filtering to epoched data and stack bands on channel axis.

    Input/output shapes:
      X: [N, C, T]
      out: [N, C * n_bands, T]
    """
    if not bands:
        return X.astype(np.float32, copy=False)
    if X.ndim != 3:
        raise ValueError(f"Filter-bank expects X [N,C,T], got {X.shape}")
    nyq = 0.5 * float(sfreq)
    if nyq <= 0:
        raise ValueError(f"Invalid sfreq for filter-bank: {sfreq}")
    x64 = X.astype(np.float64, copy=False)
    out_parts: list[np.ndarray] = []
    for lo, hi in bands:
        if hi >= nyq:
            raise ValueError(f"Filter-bank upper edge {hi} must be < Nyquist {nyq}")
        sos = butter(order, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
        xb = sosfiltfilt(sos, x64, axis=-1).astype(np.float32, copy=False)
        out_parts.append(xb)
    return np.concatenate(out_parts, axis=1).astype(np.float32, copy=False)


def segment_baseline_windows(data: np.ndarray, window_len: int, stride: int) -> np.ndarray:
    if data.ndim != 2:
        raise ValueError(f"Baseline data must be [C,T], got {data.shape}")
    if window_len <= 0 or stride <= 0:
        raise ValueError("window_len and stride must be positive")
    starts = list(range(0, data.shape[1] - window_len + 1, stride))
    if not starts:
        raise ValueError(
            f"Recording too short for baseline windowing: n_times={data.shape[1]}, window_len={window_len}"
        )
    X = np.stack([data[:, s : s + window_len] for s in starts], axis=0).astype(np.float32)
    return X


def _read_raw_edf(edf_path: Path) -> mne.io.BaseRaw:
    return mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")


def _build_meta_rows(subject: str, run: str, run_kind: str, task_type: str, labels: np.ndarray, event_descs: list[str], file: str) -> pd.DataFrame:
    rows = []
    for label, event_desc in zip(labels.tolist(), event_descs):
        rows.append(
            {
                "subject": subject,
                "run": run,
                "run_kind": run_kind,
                "task_type": task_type,
                "label": int(label),
                "label_name": label_name(int(label)),
                "event_desc": str(event_desc),
                "file": file,
            }
        )
    meta = pd.DataFrame(rows, columns=META_COLUMNS)
    return meta


def load_run_epochs(edf_path: str | Path, data_cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    path = Path(edf_path)
    subject = path.parent.name
    run_token = path.stem[-3:]
    rd = describe_run(run_token)
    raw = _read_raw_edf(path)

    sfreq_expected = float(data_cfg.get("sfreq", 160.0))
    sfreq_actual = float(raw.info["sfreq"])
    if abs(sfreq_actual - sfreq_expected) > 1e-6:
        print(f"[loader] Resampling sfreq mismatch in {path.name}: {sfreq_actual} -> {sfreq_expected}")
        raw.resample(sfreq_expected, verbose="ERROR")

    raw = apply_preprocessing(raw, data_cfg)
    requested_channels = [clean_channel_name(ch) for ch in data_cfg["channels"]]
    real_channels_after_pick = [clean_channel_name(ch) for ch in raw.ch_names]

    n_times_expected = int(data_cfg.get("baseline_window_len") or expected_n_times(raw.info["sfreq"], data_cfg["tmin"], data_cfg["tmax"]))
    if rd.run_kind == "baseline":
        data = raw.get_data().astype(np.float32, copy=False)
        X = segment_baseline_windows(
            data,
            window_len=int(data_cfg["baseline_window_len"]),
            stride=int(data_cfg["baseline_window_stride"]),
        )
        X = append_virtual_channels_epochs(
            X,
            real_channel_names=real_channels_after_pick,
            requested_channel_names=requested_channels,
        )
        y = np.zeros(X.shape[0], dtype=np.int64)
        meta = _build_meta_rows(subject, rd.run, rd.run_kind, rd.task_type, y, ["T0"] * len(y), str(path))
    else:
        events, event_id = mne.events_from_annotations(raw, event_id=EVENT_ID, verbose="ERROR")
        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=float(data_cfg["tmin"]),
            tmax=float(data_cfg["tmax"]),
            baseline=None,
            preload=True,
            reject_by_annotation=False,
            verbose="ERROR",
        )
        X = epochs.get_data(copy=True).astype(np.float32)
        X = append_virtual_channels_epochs(
            X,
            real_channel_names=real_channels_after_pick,
            requested_channel_names=requested_channels,
        )
        inv_event = {v: k for k, v in event_id.items()}
        event_descs = [str(inv_event[int(code)]) for code in epochs.events[:, 2]]
        y = np.asarray([map_event_to_label(rd.task_type, desc) for desc in event_descs], dtype=np.int64)
        meta = _build_meta_rows(subject, rd.run, rd.run_kind, rd.task_type, y, event_descs, str(path))

    fb_bands = parse_filter_bank_bands(data_cfg)
    if fb_bands:
        X = apply_filter_bank_epochs(X, float(raw.info["sfreq"]), fb_bands)

    if X.ndim != 3:
        raise AssertionError(f"X must be 3D [n,C,T], got {X.shape} from {path}")
    n_ch_expected = effective_n_channels(data_cfg)
    if X.shape[1] != n_ch_expected:
        raise AssertionError(f"Channel count mismatch for {path}: {X.shape[1]} != {n_ch_expected}")
    if X.shape[2] != n_times_expected:
        raise AssertionError(f"Time length mismatch for {path}: {X.shape[2]} != {n_times_expected}")
    if len(meta) != X.shape[0] or len(y) != X.shape[0]:
        raise AssertionError(f"Sample length mismatch in {path}")
    return X, y, meta


def subject_edf_paths(subject_dir: str | Path) -> list[Path]:
    p = Path(subject_dir)
    paths = [p / f"{p.name}R{run:02d}.edf" for run in sorted(ALL_RUNS)]
    missing = [str(x) for x in paths if not x.exists()]
    if missing:
        raise FileNotFoundError(f"Missing EDF files for {p.name}: {missing[:5]}")
    return paths


def load_subject_epochs(subject: str, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    data_cfg = cfg.get("data", cfg)
    if data_cfg.get("cache", {}).get("enabled", False) and cache_exists(data_cfg, subject):
        X, y, meta = load_subject_cache(data_cfg, subject)
        return X, y, meta

    data_root = Path(data_cfg["data_root"])
    subject_dir = data_root / subject
    if not subject_dir.exists():
        raise FileNotFoundError(f"Subject directory not found: {subject_dir}")

    all_X = []
    all_y = []
    all_meta = []
    for edf_path in subject_edf_paths(subject_dir):
        X_run, y_run, meta_run = load_run_epochs(edf_path, data_cfg)
        all_X.append(X_run)
        all_y.append(y_run)
        all_meta.append(meta_run)

    X = np.concatenate(all_X, axis=0).astype(np.float32, copy=False)
    y = np.concatenate(all_y, axis=0).astype(np.int64, copy=False)
    meta = pd.concat(all_meta, axis=0, ignore_index=True)
    meta = meta.loc[:, META_COLUMNS].copy()

    if data_cfg.get("euclidean_alignment", False):
        ea_idx = _resolve_window_indices_from_spec(
            spec=data_cfg.get("euclidean_alignment_time_window"),
            data_cfg=data_cfg,
            n_times=int(X.shape[2]),
        )
        X = apply_euclidean_alignment_epochs(X, ref_time_indices=ea_idx)

    if data_cfg.get("cache", {}).get("enabled", False):
        save_subject_cache(data_cfg, subject, X, y, meta)

    return X, y, meta


def load_eegmmidb_epochs(cfg: dict[str, Any], subjects: list[str] | None = None) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    data_cfg = cfg.get("data", cfg)
    data_root = Path(data_cfg["data_root"])
    if subjects is None:
        subjects = sorted(p.name for p in data_root.iterdir() if p.is_dir() and p.name.startswith("S") and len(p.name) == 4)
    subjects = sorted(subjects)
    if not subjects:
        raise ValueError("No subjects provided/found")

    chunks_X: list[np.ndarray] = []
    chunks_y: list[np.ndarray] = []
    chunks_meta: list[pd.DataFrame] = []
    iterator = tqdm(subjects, desc="Loading subjects", leave=False)
    for subject in iterator:
        X_sub, y_sub, meta_sub = load_subject_epochs(subject, data_cfg)
        chunks_X.append(X_sub)
        chunks_y.append(y_sub)
        chunks_meta.append(meta_sub)

    X = np.concatenate(chunks_X, axis=0).astype(np.float32, copy=False)
    y = np.concatenate(chunks_y, axis=0).astype(np.int64, copy=False)
    meta = pd.concat(chunks_meta, axis=0, ignore_index=True)
    meta = meta.loc[:, META_COLUMNS].copy()

    n_times_expected = int(data_cfg.get("baseline_window_len") or expected_n_times(data_cfg["sfreq"], data_cfg["tmin"], data_cfg["tmax"]))
    assert X.ndim == 3, f"X must be [n,C,T], got {X.shape}"
    assert X.shape[1] == effective_n_channels(data_cfg), f"X channel mismatch: {X.shape[1]}"
    assert X.shape[2] == n_times_expected, f"X time mismatch: {X.shape[2]} vs {n_times_expected}"
    assert X.dtype == np.float32, f"X dtype must be float32, got {X.dtype}"
    assert np.issubdtype(y.dtype, np.integer), f"y dtype must be integer, got {y.dtype}"
    assert list(meta.columns) == META_COLUMNS, f"meta columns mismatch: {list(meta.columns)}"
    assert len(meta) == len(y) == X.shape[0], "X/y/meta sample mismatch"
    return X, y, meta
