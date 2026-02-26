# EEGMMIDB Motor Imagery CNN (CPU/macOS)

Compact PyTorch CNN pipelines for subject-independent EEG motor imagery decoding on PhysioNet EEGMMIDB with strict leakage prevention.

## Features
- Single MNE-based loader (`X, y, meta`) with channel cleaning, filtering, optional REST-based z-score normalization, and subject-wise caching.
- Hierarchical decoding:
  - Stage A: `REST` vs `ACTIVE`
  - Stage B: 4-class active (`LEFT`, `RIGHT`, `FISTS`, `FEET`)
- Two compact CPU-friendly models: EEGNet-style and ShallowFBCSPNet-style.
- Subject-independent splits (`train_pool_100`, `train_inner`, `val_inner`, `test_9`) with persisted split JSON.
- Weighted loss + optional balanced sampler to handle REST dominance without deleting baseline data.
- Evaluation reports for held-out subjects: executed-only, imagery-only, combined.
- Confusion matrices, learning curves, PSD and ERD-style topomap sanity plots, optional saliency.

## Commands
```bash
source /Users/sanan/Desktop/ML/my-ml1/sanenv/bin/activate
python -m scripts.audit_data --subjects S001 S002 S003 --data-root /Users/sanan/Desktop/ML/my-ml1/files
python -m scripts.train_cnn --config configs/eegnet.yaml
python -m scripts.train_cnn --config configs/shallow.yaml
python -m scripts.evaluate --checkpoint outputs/<run_id>/best.pt
```

## Leakage-Prevention Checklist
- Subject-wise splits only. No window-level random splitting across subjects.
- Test subjects are excluded from training, sampler fitting, and fine-tuning.
- Inner validation subjects come only from the 100-subject training pool.
- Stage A/B are trained separately; Stage B sees active epochs only.
- End-to-end metrics are computed post hoc on held-out subjects only.
- Per-recording z-score uses only T0 (REST) samples from that recording (baseline runs are fully T0).
- Cache keys are preprocessing-only; caches are not used to tune hyperparameters on test labels.

## Pain Points Addressed
- Channel naming cleanup (`Fc5.` -> `FC5`) with strict validation.
- Baseline/task window shape mismatch solved via fixed-length baseline segmentation (`n_times=561`).
- Subject leakage prevention with deterministic saved splits.
- REST dominance handled via hierarchical decoding, weighted CE, and optional balanced sampling.
- LR vs FF run/event label mapping validated in tests and audits.
- CPU efficiency through float32, compact models, subject-wise cache, and moderate batch sizes.
- Overfitting controls (early stopping, dropout, weight decay, validation subjects).
- Reproducibility via seeded RNGs, saved configs/splits, and checkpoint metadata hashes.
- Model-collapse diagnostics (class histograms, confusion matrices, active-rate checks).
- Clear failure messages and strict shape assertions.
