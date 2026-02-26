from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eegmi.constants import ALL_RUNS, LABEL_TO_NAME, META_COLUMNS, PAIN_POINTS, describe_run
from eegmi.data.loader import clean_channel_name, expected_n_times, load_eegmmidb_epochs


def audit_subject_files(data_root: str | Path, subjects: list[str]) -> dict[str, Any]:
    root = Path(data_root)
    per_subject = {}
    for subject in subjects:
        sdir = root / subject
        expected = [sdir / f"{subject}R{run:02d}.edf" for run in sorted(ALL_RUNS)]
        missing = [str(p) for p in expected if not p.exists()]
        per_subject[subject] = {
            "exists": sdir.exists(),
            "missing_edf": missing,
            "n_found_edf": int(sum(p.exists() for p in expected)),
        }
    return per_subject


def summarize_loaded_data(X: np.ndarray, y: np.ndarray, meta: pd.DataFrame, data_cfg: dict[str, Any]) -> dict[str, Any]:
    label_counts = Counter(int(v) for v in y.tolist())
    label_name_counts = {LABEL_TO_NAME[k]: int(v) for k, v in sorted(label_counts.items())}
    run_kind_counts = meta["run_kind"].value_counts().sort_index().to_dict()
    task_type_counts = meta["task_type"].value_counts().sort_index().to_dict()
    subject_counts = meta["subject"].value_counts().sort_index().to_dict()
    event_desc_counts = meta["event_desc"].value_counts().sort_index().to_dict()
    shape_checks = {
        "X_shape": list(map(int, X.shape)),
        "X_dtype": str(X.dtype),
        "y_dtype": str(y.dtype),
        "meta_columns": list(meta.columns),
        "expected_n_times": expected_n_times(float(data_cfg["sfreq"]), float(data_cfg["tmin"]), float(data_cfg["tmax"])),
        "baseline_window_len": int(data_cfg["baseline_window_len"]),
    }
    return {
        "shape_checks": shape_checks,
        "label_counts": label_name_counts,
        "run_kind_counts": {str(k): int(v) for k, v in run_kind_counts.items()},
        "task_type_counts": {str(k): int(v) for k, v in task_type_counts.items()},
        "event_desc_counts": {str(k): int(v) for k, v in event_desc_counts.items()},
        "subject_counts": {str(k): int(v) for k, v in subject_counts.items()},
    }


def label_mapping_examples() -> dict[str, dict[str, int]]:
    examples = {}
    for run in ["R01", "R03", "R04", "R05", "R06"]:
        rd = describe_run(run)
        examples[run] = {"task_type": rd.task_type, "run_kind": rd.run_kind}
    return examples


def generate_audit_report(cfg: dict[str, Any], subjects: list[str]) -> dict[str, Any]:
    data_cfg = cfg.get("data", cfg)
    file_audit = audit_subject_files(data_cfg["data_root"], subjects)
    missing_any = [s for s, info in file_audit.items() if info["missing_edf"]]
    if missing_any:
        raise FileNotFoundError(f"Audit failed: missing EDF files for subjects {missing_any}")

    X, y, meta = load_eegmmidb_epochs(cfg, subjects=subjects)
    summary = summarize_loaded_data(X, y, meta, data_cfg)
    issues = []
    if X.shape[2] != int(data_cfg["baseline_window_len"]):
        issues.append("Baseline/task time dimension mismatch")
    if list(meta.columns) != META_COLUMNS:
        issues.append("Meta schema mismatch")
    if not np.all(np.isin(y, [0, 1, 2, 3, 4])):
        issues.append("Unexpected labels outside 0..4")

    report = {
        "subjects": subjects,
        "data_root": str(data_cfg["data_root"]),
        "channel_cleaning_example": [clean_channel_name(v) for v in ["Fc5.", "C5..", "Poz."]],
        "file_audit": file_audit,
        "summary": summary,
        "label_mapping_examples": label_mapping_examples(),
        "pain_points": PAIN_POINTS,
        "issues": issues,
        "status": "ok" if not issues else "warning",
    }
    return report
