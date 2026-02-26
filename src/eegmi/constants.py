from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MOTOR_CHANNELS = [
    "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "CZ", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
    "FZ", "PZ",
]

LABEL_TO_NAME = {
    0: "REST",
    1: "LEFT",
    2: "RIGHT",
    3: "FISTS",
    4: "FEET",
}
NAME_TO_LABEL = {v: k for k, v in LABEL_TO_NAME.items()}
ACTIVE_LABELS = [1, 2, 3, 4]
ACTIVE_LABEL_NAMES = [LABEL_TO_NAME[i] for i in ACTIVE_LABELS]

STAGE_A_LABELS = {0: "REST", 1: "ACTIVE"}
STAGE_B_LABELS = {0: "LEFT", 1: "RIGHT", 2: "FISTS", 3: "FEET"}

BASELINE_RUNS = {1, 2}
EXECUTED_RUNS = {3, 5, 7, 9, 11, 13}
IMAGINED_RUNS = {4, 6, 8, 10, 12, 14}
LR_RUNS = {3, 4, 7, 8, 11, 12}
FF_RUNS = {5, 6, 9, 10, 13, 14}
ALL_RUNS = set(range(1, 15))

META_COLUMNS = [
    "subject",
    "run",
    "run_kind",
    "task_type",
    "label",
    "label_name",
    "event_desc",
    "file",
]

PAIN_POINTS = [
    "channel cleaning with strict validation and stable reordering",
    "baseline/task window shape mismatch handled by fixed-length baseline segmentation",
    "subject leakage prevention using deterministic saved subject splits",
    "REST dominance mitigation via weighted CE, optional balanced sampler, hierarchical decoding",
    "LR vs FF label mapping correctness tests and audit summaries",
    "CPU efficiency with float32, caching, compact models, and modest batch sizes",
    "overfitting controls using inner validation subjects and early stopping",
    "reproducibility via seeds, saved configs/splits, and checkpoint metadata hashes",
    "model collapse to REST diagnostics via confusion matrices and class histograms",
    "clear errors and explicit audit output for missing files/channels/shape mismatches",
]


@dataclass(frozen=True)
class RunDescriptor:
    run: str
    run_num: int
    run_kind: str
    task_type: str


def parse_run_num(run: str | int) -> int:
    if isinstance(run, int):
        return run
    r = str(run).upper()
    if r.startswith("R"):
        r = r[1:]
    return int(r)


def format_run(run_num: int) -> str:
    return f"R{run_num:02d}"


def describe_run(run: str | int) -> RunDescriptor:
    run_num = parse_run_num(run)
    if run_num in BASELINE_RUNS:
        return RunDescriptor(format_run(run_num), run_num, "baseline", "BL")
    if run_num in EXECUTED_RUNS:
        task_type = "LR" if run_num in LR_RUNS else "FF"
        return RunDescriptor(format_run(run_num), run_num, "executed", task_type)
    if run_num in IMAGINED_RUNS:
        task_type = "LR" if run_num in LR_RUNS else "FF"
        return RunDescriptor(format_run(run_num), run_num, "imagined", task_type)
    raise ValueError(f"Unsupported run number: {run_num}")


def map_event_to_label(task_type: str, event_desc: str) -> int:
    event = str(event_desc).upper()
    task = str(task_type).upper()
    if event == "T0":
        return 0
    if task == "LR":
        if event == "T1":
            return 1
        if event == "T2":
            return 2
    if task == "FF":
        if event == "T1":
            return 3
        if event == "T2":
            return 4
    if task == "BL" and event == "T0":
        return 0
    raise ValueError(f"Invalid task/event mapping: task_type={task_type}, event_desc={event_desc}")


def label_name(label: int) -> str:
    return LABEL_TO_NAME[int(label)]
