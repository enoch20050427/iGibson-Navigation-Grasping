import ast
from dataclasses import dataclass
import math
from pathlib import Path
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "integration" / "sugarbox_rl_approach_final2.py"
tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
names = {"TARGET_VISUAL_RECENT_S", "TARGET_ACQUIRE_CONFIRM_FRAMES",
         "TARGET_REACQUIRE_CONFIRM_FRAMES", "TARGET_CONFIRM_MAX_GAP_S",
         "TARGET_ACQUIRE_MAX_ERROR_M", "TARGET_REACQUIRE_BASE_ERROR_M",
         "TARGET_REACQUIRE_RANGE_GAIN", "TARGET_REACQUIRE_MAX_ERROR_M", "TARGET_EMA_ALPHA"}
nodes = [node for node in tree.body if
         (isinstance(node, ast.ClassDef) and node.name == "TargetState") or
         (isinstance(node, ast.FunctionDef) and node.name == "get_age") or
         (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names
                                               for t in node.targets))]
namespace = {"dataclass": dataclass, "math": math}
# Compile only the pure memory logic, without importing camera/ROS/motor code.
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
TargetState = namespace["TargetState"]


def measurement(x=1.0, y=0.2):
    return {"x": x, "y": y, "det": {"conf": 0.9, "class_name": "sugarbox"},
            "u": 320.0, "v": 360.0, "on_floor": True}


class InitialOnlyTargetTests(unittest.TestCase):
    def test_first_valid_measurement_locks_without_three_frame_average(self):
        target = TargetState(freeze_visual_updates=True)
        accepted, status, _, _ = target.consider_measurement(measurement(), 100.0)
        self.assertTrue(accepted)
        self.assertEqual(status, "TARGET_LOCKED")
        self.assertTrue(target.has_memory())
        self.assertEqual((target.x, target.y), (1.0, 0.2))
        self.assertEqual(target.candidate_count, 0)

    def test_all_later_detections_leave_memory_and_freshness_unchanged(self):
        target = TargetState(freeze_visual_updates=True)
        target.consider_measurement(measurement(), 100.0)
        for x, y, stamp in ((1.01, 0.21, 100.2), (4.0, -2.0, 101.0), (1.0, 0.2, 120.0)):
            before = vars(target).copy()
            accepted, status, _, _ = target.consider_measurement(measurement(x, y), stamp)
            self.assertFalse(accepted)
            self.assertEqual(status, "TARGET_INITIAL_ONLY_ODOM")
            self.assertEqual(vars(target), before)

    def test_odometry_still_updates_range_and_bearing_after_vision_lock(self):
        target = TargetState(freeze_visual_updates=True)
        target.consider_measurement(measurement(), 100.0)
        target.propagate_with_odom(0.1, 0.2, 100.2)
        self.assertAlmostEqual(target.x, math.cos(0.04) * 0.98 + math.sin(0.04) * 0.2)
        self.assertAlmostEqual(target.y, -math.sin(0.04) * 0.98 + math.cos(0.04) * 0.2)
        self.assertAlmostEqual(target.dist, math.hypot(target.x, target.y))
        self.assertAlmostEqual(target.bearing, math.atan2(target.y, target.x))
        self.assertEqual(target.last_geometry_stamp, 100.0)
        self.assertTrue(target.valid(120.0))
        before = vars(target).copy()
        target.consider_measurement(measurement(2.0, 0.0), 100.3, geometry_locked=True)
        self.assertEqual(vars(target), before)

    def test_only_explicit_clear_allows_new_initial_target(self):
        target = TargetState(freeze_visual_updates=True)
        target.consider_measurement(measurement(), 100.0)
        target.clear()
        self.assertTrue(target.freeze_visual_updates)
        self.assertFalse(target.has_memory())
        self.assertTrue(target.consider_measurement(measurement(2.0, -0.1), 110.0)[0])
        self.assertEqual((target.x, target.y), (2.0, -0.1))

    def test_default_memory_still_requires_three_consistent_detections(self):
        target = TargetState()
        self.assertFalse(target.consider_measurement(measurement(), 100.0)[0])
        self.assertFalse(target.consider_measurement(measurement(), 100.2)[0])
        self.assertTrue(target.consider_measurement(measurement(), 100.4)[0])
        target.consider_measurement(measurement(1.1, 0.2), 100.5)
        self.assertGreater(target.x, 1.0)

    def test_main_wires_opt_in_flag_to_memory(self):
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        constructor = next(n for n in ast.walk(main) if isinstance(n, ast.Call) and
                           isinstance(n.func, ast.Name) and n.func.id == "TargetState")
        flag = next(k.value for k in constructor.keywords if k.arg == "freeze_visual_updates")
        self.assertIsInstance(flag, ast.Name)
        self.assertEqual(flag.id, "TARGET_INITIAL_ONLY")


if __name__ == "__main__":
    unittest.main()
