from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

from eegmi.eval.evaluate import evaluate_checkpoint
from eegmi.utils import set_matplotlib_env


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate hierarchical EEGMMIDB checkpoint")
    p.add_argument("--checkpoint", required=True, help="Path to combined checkpoint (best.pt)")
    p.add_argument("--data-root", default=None, help="Override data root")
    p.add_argument("--out-dir", default=None, help="Output directory (defaults to <ckpt_dir>/eval)")
    p.add_argument("--no-plots", action="store_true", help="Disable plot generation")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    set_matplotlib_env(Path.cwd() / "outputs")
    results = evaluate_checkpoint(
        args.checkpoint,
        data_root=args.data_root,
        output_dir=args.out_dir,
        with_plots=not args.no_plots,
        device="cpu",
    )
    branch = "fine_tuned_imagery" if results.get("fine_tuned_imagery") else "base"
    combined = results[branch]["combined"]["end_to_end"]["metrics"]
    print(
        f"Eval complete ({branch}). Combined held-out end-to-end: acc={combined['accuracy']:.4f}, "
        f"bal_acc={combined['balanced_accuracy']:.4f}, macro_f1={combined['macro_f1']:.4f}"
    )
    if branch == "fine_tuned_imagery":
        base_combined = results["base"]["combined"]["end_to_end"]["metrics"]
        print(
            "Base branch comparison: "
            f"acc={base_combined['accuracy']:.4f}, "
            f"bal_acc={base_combined['balanced_accuracy']:.4f}, macro_f1={base_combined['macro_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
