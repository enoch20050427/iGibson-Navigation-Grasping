import ast
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Optional
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "integration"))
from sugarbox_arm_feedback import read_angles, raw_to_degrees


class ArmFeedbackTests(unittest.TestCase):
    def test_conversion_matches_existing_v23(self):
        repo = next(parent for parent in ROOT.parents
                    if (parent / "grasp/v23/x3plus_real_grasp.py").is_file())
        source = repo / "grasp/v23/x3plus_real_grasp.py"
        tree = ast.parse(source.read_text(encoding="utf-8-sig"))
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_rosmaster_raw_to_deg"]
        ns = dict(math=math, Optional=Optional)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), ns)
        for servo_id in range(1, 7):
            for raw in (0, 380, 899, 900, 1267, 2000, 3100, 3700, 4000, float("nan")):
                self.assertEqual(raw_to_degrees(servo_id, raw), ns["_rosmaster_raw_to_deg"](servo_id, raw))

    def test_partial_array_only_reads_missing_joint(self):
        raw = Mock(return_value=(6, 1267))
        device = SimpleNamespace(get_uart_servo_angle_array=lambda: [90, 140, 0, 0, 90, -1],
                                 get_uart_servo_value=raw)
        self.assertEqual(read_angles(device), [90, 140, 0, 0, 90, 30])
        raw.assert_called_once_with(6)

    def test_shifted_response_is_credited_to_actual_joint(self):
        replies = [(2, 1389), (1, 2000), (3, 3100), (4, 3100), (5, 1487), (6, 1267)]
        device = SimpleNamespace(get_uart_servo_value=Mock(side_effect=replies))
        with patch("sugarbox_arm_feedback.time.sleep"):
            self.assertEqual(read_angles(device), [90, 140, 0, 0, 90, 30])

    def test_unrecoverable_reads_are_not_fabricated(self):
        device = SimpleNamespace(get_uart_servo_angle=lambda _: -1)
        with patch("sugarbox_arm_feedback.time.sleep"), self.assertRaisesRegex(ValueError, "S1,S2,S3,S4,S5,S6"):
            read_angles(device)


if __name__ == "__main__":
    unittest.main()
