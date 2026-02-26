from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

from eegmi.config import config_hash, load_config
from eegmi.eval.evaluate import evaluate_checkpoint
from eegmi.train.checkpointing import load_checkpoint, save_checkpoint
from eegmi.utils import ensure_dir, now_stamp, set_matplotlib_env, write_json

from scripts.evaluate_hybrid import build_execa_baselineb_checkpoint, build_hybrid_checkpoint
from scripts.train_cnn import _train_all_stages


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Train (or reuse) an execbranch checkpoint and compose it with a baseline checkpoint, "
            "then run final evaluation."
        )
    )
    p.add_argument(
        "--mode",
        default="exec_a_baseline_b",
        choices=["exec_a_baseline_b", "imagery_ft"],
        help=(
            "Composition mode. "
            "'exec_a_baseline_b' = Stage A base from execbranch checkpoint + baseline Stage B + baseline imagery FT; "
            "'imagery_ft' = base branches from execbranch checkpoint + imagery FT branches from baseline checkpoint."
        ),
    )
    p.add_argument("--baseline-checkpoint", required=True, help="Path to baseline combined checkpoint (e.g., the current best full run).")
    p.add_argument("--exec-checkpoint", default=None, help="Path to an existing execbranch combined checkpoint.")
    p.add_argument("--exec-config", default=None, help="Path to execbranch config to train first (if --exec-checkpoint is not provided).")
    p.add_argument("--data-root", default=None, help="Override data root for training/evaluation.")
    p.add_argument("--out-dir", default=None, help="Output directory for composed checkpoint + evaluation.")
    p.add_argument("--save-merged-checkpoint", default=None, help="Optional explicit path to save the composed checkpoint.")
    p.add_argument("--no-plots", action="store_true", help="Disable evaluation plots.")
    return p


def _train_exec_checkpoint_from_config(config_path: str, data_root: str | None = None) -> str:
    cfg = load_config(config_path)
    if data_root is not None:
        cfg["data"]["data_root"] = data_root

    set_matplotlib_env(Path.cwd() / cfg["experiment"].get("output_root", "outputs"))
    exp_name = str(cfg["experiment"].get("name", "eegmi"))
    run_id = f"{now_stamp()}_{exp_name}_{config_hash(cfg)[:8]}"
    run_dir = ensure_dir(Path(cfg["experiment"].get("output_root", "outputs")) / run_id)
    write_json(run_dir / "config_snapshot.json", cfg)

    result = _train_all_stages(cfg, run_dir)
    return str(result["checkpoint"])


def main() -> None:
    args = build_argparser().parse_args()
    set_matplotlib_env(Path.cwd() / "outputs")

    if not args.exec_checkpoint and not args.exec_config:
        raise SystemExit("Provide either --exec-checkpoint or --exec-config")

    if args.exec_checkpoint:
        exec_ckpt_path = Path(args.exec_checkpoint).resolve()
    else:
        exec_ckpt_path = Path(_train_exec_checkpoint_from_config(args.exec_config, data_root=args.data_root)).resolve()

    baseline_ckpt_path = Path(args.baseline_checkpoint).resolve()
    exec_ckpt = load_checkpoint(exec_ckpt_path, map_location="cpu")
    baseline_ckpt = load_checkpoint(baseline_ckpt_path, map_location="cpu")

    if args.mode == "exec_a_baseline_b":
        merged = build_execa_baselineb_checkpoint(exec_ckpt, baseline_ckpt)
    elif args.mode == "imagery_ft":
        merged = build_hybrid_checkpoint(exec_ckpt, baseline_ckpt)
    else:
        raise ValueError(args.mode)

    merged["hybrid_sources"]["base_checkpoint_path"] = str(exec_ckpt_path)
    merged["hybrid_sources"]["imagery_checkpoint_path"] = str(baseline_ckpt_path)

    exec_run_dir = exec_ckpt_path.parent
    base_tag = baseline_ckpt_path.parent.name
    out_dir = Path(args.out_dir) if args.out_dir else (exec_run_dir / f"composed_{args.mode}_{base_tag}")
    ensure_dir(out_dir)

    merged_ckpt_path = Path(args.save_merged_checkpoint) if args.save_merged_checkpoint else (out_dir / "best.pt")
    save_checkpoint(merged_ckpt_path, merged)

    results = evaluate_checkpoint(
        merged_ckpt_path,
        data_root=args.data_root,
        output_dir=out_dir,
        device="cpu",
        with_plots=not args.no_plots,
    )
    branch = "fine_tuned_imagery" if results.get("fine_tuned_imagery") else "base"
    combined = results[branch]["combined"]["end_to_end"]["metrics"]
    summary = {
        "mode": args.mode,
        "exec_checkpoint": str(exec_ckpt_path),
        "baseline_checkpoint": str(baseline_ckpt_path),
        "merged_checkpoint": str(merged_ckpt_path),
        "eval_output_dir": str(out_dir),
        "selected_branch": branch,
        "combined_metrics": combined,
    }
    write_json(out_dir / "composed_summary.json", summary)

    print(f"Exec checkpoint: {exec_ckpt_path}")
    print(f"Baseline checkpoint: {baseline_ckpt_path}")
    print(f"Composed checkpoint: {merged_ckpt_path}")
    print(f"Eval output dir: {out_dir}")
    print(
        f"Composed eval ({branch}) combined held-out: "
        f"acc={combined['accuracy']:.4f}, bal_acc={combined['balanced_accuracy']:.4f}, macro_f1={combined['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
