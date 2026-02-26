from __future__ import annotations

import sys
from pathlib import Path
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eegmi.constants import describe_run, map_event_to_label


class TestLabelMapping(unittest.TestCase):
    def test_baseline_mapping(self):
        rd = describe_run("R01")
        self.assertEqual(rd.run_kind, "baseline")
        self.assertEqual(rd.task_type, "BL")
        self.assertEqual(map_event_to_label(rd.task_type, "T0"), 0)

    def test_lr_mapping(self):
        for run in ["R03", "R04", "R07", "R08", "R11", "R12"]:
            rd = describe_run(run)
            self.assertEqual(rd.task_type, "LR")
            self.assertEqual(map_event_to_label(rd.task_type, "T1"), 1)
            self.assertEqual(map_event_to_label(rd.task_type, "T2"), 2)

    def test_ff_mapping(self):
        for run in ["R05", "R06", "R09", "R10", "R13", "R14"]:
            rd = describe_run(run)
            self.assertEqual(rd.task_type, "FF")
            self.assertEqual(map_event_to_label(rd.task_type, "T1"), 3)
            self.assertEqual(map_event_to_label(rd.task_type, "T2"), 4)

    def test_invalid_mapping_raises(self):
        with self.assertRaises(ValueError):
            map_event_to_label("LR", "BAD")


if __name__ == "__main__":
    unittest.main()
