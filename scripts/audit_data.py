from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

from eegmi.config import default_config
from eegmi.data.audit import generate_audit_report
from eegmi.utils import ensure_dir, now_stamp, set_matplotlib_env, write_json


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit EEGMMIDB local files and preprocessing/epoch outputs")
    p.add_argument("--subjects", nargs="+", required=True, help="Subject IDs, e.g. S001 S002 S003")
    p.add_argument("--data-root", required=True, help="Path to EEGMMIDB local root")
    p.add_argument("--out", default=None, help="Optional output JSON path")
    return p


def main() -> None:
    set_matplotlib_env(Path.cwd() / "outputs")
    args = build_argparser().parse_args()
    cfg = default_config()
    cfg["data"]["data_root"] = args.data_root
    cfg["data"]["cache"]["enabled"] = True

    report = generate_audit_report(cfg, subjects=args.subjects)
    out_path = Path(args.out) if args.out else ensure_dir("outputs/audits") / f"audit_{now_stamp()}.json"
    write_json(out_path, report)

    summary = report["summary"]
    print(f"Audit status: {report['status']}")
    print(f"Saved report: {out_path}")
    print(f"X shape: {summary['shape_checks']['X_shape']}, dtype={summary['shape_checks']['X_dtype']}")
    print(f"Epoch n_times: {summary['shape_checks']['X_shape'][2]}")
    print(f"Event desc counts: {summary.get('event_desc_counts', {})}")
    print(f"Label counts: {summary['label_counts']}")
    print(f"Run-kind counts: {summary['run_kind_counts']}")
    if report["issues"]:
        print(f"Issues: {report['issues']}")


if __name__ == "__main__":
    main()
