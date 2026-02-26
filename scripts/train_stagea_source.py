from __future__ import annotations

import argparse
import copy
import gc
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

import scripts.train_cnn as tc

from eegmi.config import config_hash, load_config
from eegmi.data.augment import build_augmenter
from eegmi.data.channel_selection import resolve_stage_channel_indices
from eegmi.data.loader import load_eegmmidb_epochs
from eegmi.data.splits import SubjectSplit
from eegmi.data.time_selection import resolve_stage_time_indices
from eegmi.eval.evaluate import calibrate_stage_a_thresholds
from eegmi.eval.plots import save_learning_curves
from eegmi.repro import seed_everything
from eegmi.train.checkpointing import load_checkpoint, save_checkpoint
from eegmi.utils import ensure_dir, now_stamp, set_matplotlib_env, to_serializable, write_json


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Stage A source heads only and merge into a template hierarchical checkpoint.")
    p.add_argument("--config", required=True, help="Source preprocessing/model config (JSON-compatible YAML)")
    p.add_argument("--template-checkpoint", required=True, help="Existing full hierarchical checkpoint used for split and Stage B weights")
    p.add_argument("--out-dir", default=None, help="Output directory (default under outputs/stagea_sources/...)")
    return p


def _stage_result_entry(stage_name: str, result: dict[str, Any], class_names: list[str]) -> dict[str, Any]:
    return {
        "stage": stage_name,
        "class_names": class_names,
        "model_state_dict": result["model"].state_dict(),
        "best_epoch": result["best_epoch"],
        "best_score": result["best_score"],
        "best_metrics": result["best_metrics"],
        "class_weights": result["class_weights"],
        "history": result["history"],
    }


def main() -> None:
    args = build_argparser().parse_args()
    cfg = load_config(args.config)
    set_matplotlib_env(Path.cwd() / cfg["experiment"].get("output_root", "outputs"))

    template_path = Path(args.template_checkpoint).resolve()
    template_ckpt = load_checkpoint(template_path, map_location="cpu")
    if "split" not in template_ckpt or "stage_b" not in template_ckpt:
        raise ValueError("Template checkpoint must be a full hierarchical bundle with split + Stage B")
    split = SubjectSplit.from_dict(template_ckpt["split"])

    exp_name = str(cfg["experiment"].get("name", "stagea_source"))
    out_root = Path(args.out_dir) if args.out_dir else Path("outputs/stagea_sources") / f"{now_stamp()}_{exp_name}_{config_hash(cfg)[:8]}"
    run_dir = ensure_dir(out_root)
    write_json(run_dir / "config_snapshot.json", cfg)
    write_json(run_dir / "template_checkpoint_meta.json", {"template_checkpoint": str(template_path), "template_split_hash": template_ckpt.get("split_hash")})
    write_json(run_dir / "subject_split.json", template_ckpt["split"])

    seed_everything(int(cfg["experiment"]["seed"]))
    train_subjects = sorted(split.train_pool_100)
    X_train, y_train, meta_train = load_eegmmidb_epochs(cfg, subjects=train_subjects)
    n_chans = int(X_train.shape[1])
    n_times = int(X_train.shape[2])

    aug_train = build_augmenter(cfg.get("augmentation", {}), seed=int(cfg["experiment"]["seed"]))
    train_cfg = cfg["train"]
    device = str(cfg["experiment"].get("device", "cpu"))
    weighted_loss = bool(train_cfg.get("weighted_loss", True))
    use_balanced_sampler = bool(train_cfg.get("use_balanced_sampler", True))

    data_diag: dict[str, Any] = {
        "train_pool_subjects": list(split.train_pool_100),
        "train_inner_subjects": list(split.train_inner),
        "val_inner_subjects": list(split.val_inner),
        "train_pool_samples": int(len(y_train)),
    }
    stage_channel_map: dict[str, Any] = {}
    stage_time_map: dict[str, Any] = {}
    model_cfgs_used: dict[str, Any] = {}

    # Stage A base
    ch_stage_a, ch_stage_a_names = resolve_stage_channel_indices(cfg["data"], "stage_a")
    t_stage_a, t_stage_a_meta = resolve_stage_time_indices(cfg["data"], "stage_a", n_times=n_times)
    stage_channel_map["stage_a"] = ch_stage_a_names
    stage_time_map["stage_a"] = t_stage_a_meta
    ds_a_train = tc._make_dataset(
        X_train, y_train, meta_train,
        subjects=split.train_inner, run_kind="combined", active_only=False,
        stage="stage_a", augmenter=aug_train, channel_indices=ch_stage_a, time_indices=t_stage_a,
    )
    ds_a_val = tc._make_dataset(
        X_train, y_train, meta_train,
        subjects=split.val_inner, run_kind="combined", active_only=False,
        stage="stage_a", augmenter=None, channel_indices=ch_stage_a, time_indices=t_stage_a,
    )
    data_diag["stage_a"] = {
        "train_samples": int(len(ds_a_train)),
        "val_samples": int(len(ds_a_val)),
        "channels": ch_stage_a_names,
        "time_window": t_stage_a_meta,
        "train_class_counts": tc._dataset_target_counts(ds_a_train),
        "val_class_counts": tc._dataset_target_counts(ds_a_val),
    }
    print(f"[diag] Stage A samples train={len(ds_a_train)} val={len(ds_a_val)}")

    stage_a_model, model_cfgs_used["stage_a"] = tc._build_stage_model(cfg, "stage_a", n_chans=n_chans, n_times=n_times, n_classes=2)
    out_stage_a = ensure_dir(run_dir / "stage_a")
    seed_everything(tc._seed_for_stage(cfg, "stage_a"))
    result_a = tc.train_stage(
        stage_a_model,
        stage_name="stage_a",
        class_names=[tc.STAGE_A_LABELS[i] for i in [0, 1]],
        train_dataset=ds_a_train,
        val_dataset=ds_a_val,
        train_cfg=tc._stage_train_cfg(train_cfg, "stage_a"),
        output_dir=out_stage_a,
        device=device,
        weighted_loss=weighted_loss,
        use_balanced_sampler=use_balanced_sampler,
    )
    save_learning_curves(result_a["history"], out_stage_a / "learning_curves.png", title="Stage A")
    del ds_a_train, ds_a_val
    gc.collect()

    # Stage A imagery fine-tune
    result_a_ft = None
    ft_a_cfg = train_cfg.get("fine_tune_imagery_stage_a", {})
    if ft_a_cfg.get("enabled", False):
        ch_stage_aft, ch_stage_aft_names = resolve_stage_channel_indices(cfg["data"], "stage_a_finetune_imagery")
        t_stage_aft, t_stage_aft_meta = resolve_stage_time_indices(cfg["data"], "stage_a_finetune_imagery", n_times=n_times)
        stage_channel_map["stage_a_finetune_imagery"] = ch_stage_aft_names
        stage_time_map["stage_a_finetune_imagery"] = t_stage_aft_meta
        ds_aft_train = tc._make_dataset(
            X_train, y_train, meta_train,
            subjects=split.train_inner, run_kind="imagined", active_only=False,
            stage="stage_a", augmenter=aug_train, channel_indices=ch_stage_aft, time_indices=t_stage_aft,
        )
        ds_aft_val = tc._make_dataset(
            X_train, y_train, meta_train,
            subjects=split.val_inner, run_kind="imagined", active_only=False,
            stage="stage_a", augmenter=None, channel_indices=ch_stage_aft, time_indices=t_stage_aft,
        )
        data_diag["stage_a_finetune_imagery"] = {
            "train_samples": int(len(ds_aft_train)),
            "val_samples": int(len(ds_aft_val)),
            "channels": ch_stage_aft_names,
            "time_window": t_stage_aft_meta,
            "train_class_counts": tc._dataset_target_counts(ds_aft_train) if len(ds_aft_train) else {},
            "val_class_counts": tc._dataset_target_counts(ds_aft_val) if len(ds_aft_val) else {},
        }
        print(f"[diag] Stage A FT(imagery) samples train={len(ds_aft_train)} val={len(ds_aft_val)}")
        if len(ds_aft_train) > 0 and len(ds_aft_val) > 0:
            stage_a_ft_model, model_cfgs_used["stage_a_finetune_imagery"] = tc._build_stage_model(
                cfg, "stage_a_finetune_imagery", n_chans=n_chans, n_times=n_times, n_classes=2
            )
            if bool(ft_a_cfg.get("init_from_base", True)):
                try:
                    stage_a_ft_model.load_state_dict(copy.deepcopy(result_a["model"].state_dict()))
                except RuntimeError as e:
                    print(f"[warn] Stage A FT init_from_base skipped due to shape mismatch; training from scratch. Error: {e}")
            out_stage_aft = ensure_dir(run_dir / "stage_a_finetune_imagery")
            ft_a_train_cfg = tc._stage_train_cfg({**train_cfg, **ft_a_cfg}, "fine_tune_imagery_stage_a")
            seed_everything(tc._seed_for_stage(cfg, "stage_a_finetune_imagery"))
            result_a_ft = tc.train_stage(
                stage_a_ft_model,
                stage_name="stage_a_finetune_imagery",
                class_names=[tc.STAGE_A_LABELS[i] for i in [0, 1]],
                train_dataset=ds_aft_train,
                val_dataset=ds_aft_val,
                train_cfg=ft_a_train_cfg,
                output_dir=out_stage_aft,
                device=device,
                weighted_loss=weighted_loss,
                use_balanced_sampler=use_balanced_sampler,
            )
            save_learning_curves(result_a_ft["history"], out_stage_aft / "learning_curves.png", title="Stage A Fine-tune (Imagery)")
        del ds_aft_train, ds_aft_val
        gc.collect()

    # Optional Stage A family heads
    result_a_lr = result_a_lr_ft = None
    result_a_ff = result_a_ff_ft = None
    family_stage_a_cfg = train_cfg.get("stage_a_family_heads", {})
    if family_stage_a_cfg.get("enabled", False):
        families = {str(v).lower() for v in family_stage_a_cfg.get("families", ["lr", "ff"])}
        if "lr" in families:
            result_a_lr, result_a_lr_ft, diag_lr = tc._train_stage_a_family_head(
                cfg=cfg,
                train_cfg=train_cfg,
                X=X_train,
                y=y_train,
                meta=meta_train,
                n_chans=n_chans,
                n_times=n_times,
                device=device,
                weighted_loss=weighted_loss,
                use_balanced_sampler=use_balanced_sampler,
                split=split,
                run_dir=run_dir,
                aug_train=aug_train,
                family_key="lr",
                task_type="LR",
                ft_imagery_cfg=ft_a_cfg,
            )
            data_diag["stage_a_family_lr"] = diag_lr
            if isinstance(diag_lr.get("channels"), list):
                stage_channel_map["stage_a_lr"] = list(diag_lr["channels"])
            if isinstance(diag_lr.get("time_window"), dict):
                stage_time_map["stage_a_lr"] = dict(diag_lr["time_window"])
            if isinstance(diag_lr.get("imagery_finetune", {}).get("channels"), list):
                stage_channel_map["stage_a_lr_finetune_imagery"] = list(diag_lr["imagery_finetune"]["channels"])
            if isinstance(diag_lr.get("imagery_finetune", {}).get("time_window"), dict):
                stage_time_map["stage_a_lr_finetune_imagery"] = dict(diag_lr["imagery_finetune"]["time_window"])
            model_cfgs_used["stage_a_lr"] = tc._model_cfg_for_stage(cfg, "stage_a_lr")
            if result_a_lr_ft is not None:
                model_cfgs_used["stage_a_lr_finetune_imagery"] = tc._model_cfg_for_stage(cfg, "stage_a_lr_finetune_imagery")
        if "ff" in families:
            result_a_ff, result_a_ff_ft, diag_ff = tc._train_stage_a_family_head(
                cfg=cfg,
                train_cfg=train_cfg,
                X=X_train,
                y=y_train,
                meta=meta_train,
                n_chans=n_chans,
                n_times=n_times,
                device=device,
                weighted_loss=weighted_loss,
                use_balanced_sampler=use_balanced_sampler,
                split=split,
                run_dir=run_dir,
                aug_train=aug_train,
                family_key="ff",
                task_type="FF",
                ft_imagery_cfg=ft_a_cfg,
            )
            data_diag["stage_a_family_ff"] = diag_ff
            if isinstance(diag_ff.get("channels"), list):
                stage_channel_map["stage_a_ff"] = list(diag_ff["channels"])
            if isinstance(diag_ff.get("time_window"), dict):
                stage_time_map["stage_a_ff"] = dict(diag_ff["time_window"])
            if isinstance(diag_ff.get("imagery_finetune", {}).get("channels"), list):
                stage_channel_map["stage_a_ff_finetune_imagery"] = list(diag_ff["imagery_finetune"]["channels"])
            if isinstance(diag_ff.get("imagery_finetune", {}).get("time_window"), dict):
                stage_time_map["stage_a_ff_finetune_imagery"] = dict(diag_ff["imagery_finetune"]["time_window"])
            model_cfgs_used["stage_a_ff"] = tc._model_cfg_for_stage(cfg, "stage_a_ff")
            if result_a_ff_ft is not None:
                model_cfgs_used["stage_a_ff_finetune_imagery"] = tc._model_cfg_for_stage(cfg, "stage_a_ff_finetune_imagery")

    # Stage A thresholds (validation subjects only)
    stage_a_thresholds = None
    if cfg.get("eval", {}).get("calibrate_stage_a_thresholds", False):
        try:
            a_ch, _ = resolve_stage_channel_indices(cfg["data"], "stage_a")
            a_ft_ch, _ = resolve_stage_channel_indices(cfg["data"], "stage_a_finetune_imagery")
            a_tm, _ = resolve_stage_time_indices(cfg["data"], "stage_a", n_times=n_times)
            a_ft_tm, _ = resolve_stage_time_indices(cfg["data"], "stage_a_finetune_imagery", n_times=n_times)
            a_family_ch = {"lr": resolve_stage_channel_indices(cfg["data"], "stage_a_lr")[0], "ff": resolve_stage_channel_indices(cfg["data"], "stage_a_ff")[0]}
            a_family_ft_ch = {
                "lr": resolve_stage_channel_indices(cfg["data"], "stage_a_lr_finetune_imagery")[0],
                "ff": resolve_stage_channel_indices(cfg["data"], "stage_a_ff_finetune_imagery")[0],
            }
            a_family_tm = {"lr": resolve_stage_time_indices(cfg["data"], "stage_a_lr", n_times=n_times)[0], "ff": resolve_stage_time_indices(cfg["data"], "stage_a_ff", n_times=n_times)[0]}
            a_family_ft_tm = {
                "lr": resolve_stage_time_indices(cfg["data"], "stage_a_lr_finetune_imagery", n_times=n_times)[0],
                "ff": resolve_stage_time_indices(cfg["data"], "stage_a_ff_finetune_imagery", n_times=n_times)[0],
            }
            stage_a_thresholds = calibrate_stage_a_thresholds(
                stage_a_model=result_a["model"],
                stage_a_imagined_model=(result_a_ft["model"] if result_a_ft is not None else None),
                X=X_train,
                y=y_train,
                meta=meta_train,
                subjects=list(split.val_inner),
                device=device,
                batch_size=int(train_cfg.get("batch_size", 128)),
                stage_a_tta_time_shifts=[int(v) for v in (cfg.get("eval", {}).get("stage_a_tta_time_shifts", []) or [])],
                granularity=str(cfg.get("eval", {}).get("stage_a_threshold_granularity", "run_kind")),
                stage_a_family_models=(
                    {
                        **({"lr": result_a_lr["model"]} if result_a_lr is not None else {}),
                        **({"ff": result_a_ff["model"]} if result_a_ff is not None else {}),
                    } or None
                ),
                stage_a_family_imagined_models=(
                    {
                        **({"lr": (result_a_lr_ft["model"] if result_a_lr_ft is not None else None)} if result_a_lr is not None else {}),
                        **({"ff": (result_a_ff_ft["model"] if result_a_ff_ft is not None else None)} if result_a_ff is not None else {}),
                    } or None
                ),
                stage_a_channel_indices=a_ch,
                stage_a_imagined_channel_indices=a_ft_ch,
                stage_a_family_channel_indices=a_family_ch,
                stage_a_family_imagined_channel_indices=a_family_ft_ch,
                stage_a_time_indices=a_tm,
                stage_a_imagined_time_indices=a_ft_tm,
                stage_a_family_time_indices=a_family_tm,
                stage_a_family_imagined_time_indices=a_family_ft_tm,
            )
            write_json(run_dir / "stage_a_thresholds.json", to_serializable(stage_a_thresholds))
        except Exception as e:
            print(f"[warn] Stage A threshold calibration failed: {e}")
            stage_a_thresholds = None

    # Merge into template checkpoint.
    out_ckpt = copy.deepcopy(template_ckpt)
    out_ckpt["kind"] = "hierarchical_bundle_stagea_source_merged"
    out_ckpt["config"] = cfg
    out_ckpt["config_hash"] = config_hash(cfg)
    out_ckpt["model_config"] = cfg["model"]
    model_cfgs_all = dict(out_ckpt.get("model_configs") or {})
    for k, v in model_cfgs_used.items():
        model_cfgs_all[k] = to_serializable(v)
    out_ckpt["model_configs"] = model_cfgs_all
    stage_channels_all = dict(out_ckpt.get("stage_channels") or {})
    stage_channels_all.update(to_serializable(stage_channel_map))
    out_ckpt["stage_channels"] = stage_channels_all
    stage_times_all = dict(out_ckpt.get("stage_times") or {})
    stage_times_all.update(to_serializable(stage_time_map))
    out_ckpt["stage_times"] = stage_times_all
    out_ckpt["n_chans"] = n_chans
    out_ckpt["n_times"] = n_times
    out_ckpt["stage_a"] = _stage_result_entry("A", result_a, [tc.STAGE_A_LABELS[i] for i in [0, 1]])
    out_ckpt["stage_a_finetuned"] = None if result_a_ft is None else _stage_result_entry("A_finetune_imagery", result_a_ft, [tc.STAGE_A_LABELS[i] for i in [0, 1]])
    out_ckpt["stage_a_family"] = None
    out_ckpt["stage_a_family_finetuned"] = None
    if result_a_lr is not None or result_a_ff is not None:
        out_ckpt["stage_a_family"] = {}
        if result_a_lr is not None:
            out_ckpt["stage_a_family"]["lr"] = _stage_result_entry("A_LR", result_a_lr, ["REST", "ACTIVE"])
        if result_a_ff is not None:
            out_ckpt["stage_a_family"]["ff"] = _stage_result_entry("A_FF", result_a_ff, ["REST", "ACTIVE"])
    if result_a_lr_ft is not None or result_a_ff_ft is not None:
        out_ckpt["stage_a_family_finetuned"] = {}
        if result_a_lr_ft is not None:
            out_ckpt["stage_a_family_finetuned"]["lr"] = _stage_result_entry("A_LR_finetune_imagery", result_a_lr_ft, ["REST", "ACTIVE"])
        if result_a_ff_ft is not None:
            out_ckpt["stage_a_family_finetuned"]["ff"] = _stage_result_entry("A_FF_finetune_imagery", result_a_ff_ft, ["REST", "ACTIVE"])
    out_ckpt["stage_a_thresholds"] = stage_a_thresholds
    out_ckpt["evaluation"] = None
    out_ckpt["stagea_source_meta"] = {
        "template_checkpoint": str(template_path),
        "run_dir": str(run_dir),
        "data_diag": to_serializable(data_diag),
    }

    ckpt_path = run_dir / "best.pt"
    save_checkpoint(ckpt_path, out_ckpt)
    write_json(run_dir / "run_summary.json", {
        "template_checkpoint": str(template_path),
        "source_config_hash": out_ckpt["config_hash"],
        "stage_a_best": result_a["best_metrics"],
        "stage_a_finetuned_best": result_a_ft["best_metrics"] if result_a_ft else None,
        "stage_a_family_lr_best": result_a_lr["best_metrics"] if result_a_lr else None,
        "stage_a_family_ff_best": result_a_ff["best_metrics"] if result_a_ff else None,
        "stage_a_thresholds": stage_a_thresholds,
    })
    print(f"Stage A source training complete. Run dir: {run_dir}")
    print(f"Merged checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
