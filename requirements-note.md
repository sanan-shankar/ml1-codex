# Requirements Notes (No Extra Dependencies Path)

This project is designed to run in the existing venv:
`/Users/sanan/Desktop/ML/my-ml1/sanenv`

Detected core packages (used):
- `mne`
- `numpy`
- `pandas`
- `torch`
- `matplotlib`
- `tqdm`

Optional packages NOT required:
- `scikit-learn` (metrics implemented locally)
- `PyYAML` (configs are JSON-compatible `.yaml` and parsed with `json` fallback)
- `pyarrow`/`fastparquet` (cache metadata falls back to CSV if parquet is unavailable)

macOS note:
- Scripts force `MPLBACKEND=Agg` and set a writable `MPLCONFIGDIR` under `outputs/` or `/tmp` to avoid matplotlib cache permission issues.
