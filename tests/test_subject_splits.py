from __future__ import annotations

import sys
from pathlib import Path
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eegmi.data.splits import make_subject_split


class TestSubjectSplits(unittest.TestCase):
    def test_subject_split_sizes_and_disjointness(self):
        subjects = [f"S{i:03d}" for i in range(1, 110)]
        split = make_subject_split(subjects, seed=42, n_train_pool=100, n_test=9, inner_val_count=10)
        self.assertEqual(len(split.train_pool_100), 100)
        self.assertEqual(len(split.test_9), 9)
        self.assertEqual(len(split.val_inner), 10)
        self.assertEqual(len(split.train_inner), 90)
        self.assertTrue(set(split.train_pool_100).isdisjoint(set(split.test_9)))
        self.assertTrue(set(split.train_inner).isdisjoint(set(split.val_inner)))
        self.assertEqual(set(split.train_pool_100), set(split.train_inner) | set(split.val_inner))


if __name__ == "__main__":
    unittest.main()
