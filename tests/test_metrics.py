from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eegmi.train.metrics import classification_metrics


class TestMetrics(unittest.TestCase):
    def test_balanced_acc_and_macro_f1(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0])
        m = classification_metrics(y_true, y_pred, labels=[0, 1, 2], label_names=["A", "B", "C"]).to_dict()
        # Recalls: A=0.5, B=1.0, C=0.5 => balanced_acc=2/3
        self.assertAlmostEqual(m["balanced_accuracy"], 2 / 3, places=6)
        # F1s: A=0.5, B=0.8, C=2/3 => macro ~0.655555...
        self.assertAlmostEqual(m["macro_f1"], (0.5 + 0.8 + (2 / 3)) / 3, places=6)
        self.assertEqual(m["confusion_matrix"], [[1, 1, 0], [0, 2, 0], [1, 0, 1]])


if __name__ == "__main__":
    unittest.main()
