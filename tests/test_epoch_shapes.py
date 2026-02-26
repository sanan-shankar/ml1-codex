from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eegmi.config import default_config
from eegmi.constants import META_COLUMNS
from eegmi.data.loader import (
    append_virtual_channels_epochs,
    expected_n_times,
    load_run_epochs,
    segment_baseline_windows,
)
from eegmi.data.time_selection import resolve_stage_time_indices


class TestEpochShapes(unittest.TestCase):
    def test_expected_n_times(self):
        self.assertEqual(expected_n_times(160, 0.5, 4.0), 561)

    def test_segment_baseline_windows_exact_shape(self):
        x = np.random.randn(23, 561 * 3 + 20).astype(np.float32)
        seg = segment_baseline_windows(x, window_len=561, stride=561)
        self.assertEqual(seg.shape, (3, 23, 561))
        self.assertEqual(seg.dtype, np.float32)

    def test_append_virtual_midline_laplacian_channel(self):
        # Real channels arranged so CZ is 10 and neighbors are 2,4,6,8 => CZ_LAP = 5
        real_names = ["FCZ", "C1", "CZ", "C2", "CPZ", "FZ"]
        X = np.zeros((2, len(real_names), 5), dtype=np.float32)
        X[:, real_names.index("CZ"), :] = 10.0
        X[:, real_names.index("C1"), :] = 2.0
        X[:, real_names.index("C2"), :] = 4.0
        X[:, real_names.index("FCZ"), :] = 6.0
        X[:, real_names.index("CPZ"), :] = 8.0
        out = append_virtual_channels_epochs(
            X,
            real_channel_names=real_names,
            requested_channel_names=["CZ", "CZ_LAP", "CPZ"],
        )
        self.assertEqual(out.shape, (2, 3, 5))
        np.testing.assert_allclose(out[:, 0, :], 10.0)
        np.testing.assert_allclose(out[:, 1, :], 5.0)
        np.testing.assert_allclose(out[:, 2, :], 8.0)

    def test_load_single_run_shape_and_meta_schema_if_data_present(self):
        cfg = default_config()
        path = Path(cfg["data"]["data_root"]) / "S001" / "S001R03.edf"
        if not path.exists():
            self.skipTest(f"Local EEGMMIDB file missing: {path}")
        X, y, meta = load_run_epochs(path, cfg["data"])
        self.assertEqual(X.shape[1], 23)
        self.assertEqual(X.shape[2], 561)
        self.assertEqual(str(X.dtype), "float32")
        self.assertTrue(np.issubdtype(y.dtype, np.integer))
        self.assertEqual(list(meta.columns), META_COLUMNS)
        self.assertEqual(len(meta), X.shape[0])

    def test_stage_time_indices_off_by_one(self):
        cfg = default_config()
        cfg["data"]["tmin"] = 0.5
        cfg["data"]["tmax"] = 4.5
        cfg["data"]["stage_time_windows"] = {
            "stage_a": [1.0, 4.5],
            "stage_b": [0.5, 4.0],
        }
        idx_a, meta_a = resolve_stage_time_indices(cfg["data"], "stage_a", n_times=641)
        idx_b, meta_b = resolve_stage_time_indices(cfg["data"], "stage_b", n_times=641)
        self.assertEqual(len(idx_a), 561)
        self.assertEqual(len(idx_b), 561)
        self.assertEqual(int(idx_a[0]), 80)   # 0.5s shift at 160 Hz
        self.assertEqual(int(idx_b[0]), 0)
        self.assertEqual(meta_a["tmin"], 1.0)
        self.assertEqual(meta_b["tmax"], 4.0)


if __name__ == "__main__":
    unittest.main()
