"""Characterize current deployment behavior, not successful navigation."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "integration" / "sugarbox_rl_approach_final2.py"
tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
functions = {"wrap_pi", "clamp", "angle_to_scan_index", "sanitize_policy_range",
             "scan_to_policy_rays", "action_to_velocity", "build_raw_obs"}
constants = {"NUM_LIDAR_RAYS", "LIDAR_MAX_M", "OBS_DIM", "POLICY_MAX_FWD",
             "POLICY_MAX_REV", "POLICY_MAX_WZ", "TEST_MAX_FWD", "TEST_MAX_REV",
             "TEST_MAX_WZ", "LIDAR_REFLECT_FRONT_BACK"}
nodes = [n for n in tree.body if
         (isinstance(n, ast.FunctionDef) and n.name in functions) or
         (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and
          t.id in constants for t in n.targets))]
ns = {"math": math, "np": np, "POLICY_ANGLES_DEG": np.linspace(-90, 90, 48),
      "PPO_LIDAR_DISTANCE_SCALE": 1.0}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), ns)

motor_source = ROOT / "integration" / "sugarbox_rl_motor_server_no_amplification.py"
motor_tree = ast.parse(motor_source.read_text(encoding="utf-8-sig"))
motor_nodes = [n for n in motor_tree.body if isinstance(n, ast.FunctionDef)
               and n.name in {"clamp", "velocity_to_motor_values"}]
motor_constants = {}
for node in motor_tree.body:
    if isinstance(node, ast.Assign):
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                motor_constants[target.id] = value
motor_ns = {**motor_constants, "Tuple": Tuple, "ANGULAR_MOTOR_PER_RAD_S": 25.0}
exec(compile(ast.Module(body=motor_nodes, type_ignores=[]), str(motor_source), "exec"), motor_ns)


class PolicyContractAuditTests(unittest.TestCase):
    def test_positive_throttles_collapse_at_deployment_limit(self):
        for throttle in (0.3, 0.5, 1.0):
            self.assertAlmostEqual(ns["action_to_velocity"]([throttle, 0])[0], 0.15)

    def test_small_throttle_is_preserved_before_motor_mapping(self):
        self.assertAlmostEqual(ns["action_to_velocity"]([0.1, 0])[0], 0.05)

    def test_narrow_obstacle_can_fall_between_policy_rays(self):
        angles = np.linspace(-math.pi, math.pi, 2031)
        body = np.arctan2(np.sin(math.pi - angles), np.cos(math.pi - angles))
        ranges = np.full(angles.shape, 4.0)
        ranges[np.abs(body) < math.radians(1)] = 0.3
        scan = SimpleNamespace(ranges=ranges, angle_min=float(angles[0]),
                               angle_increment=float(angles[1] - angles[0]))
        self.assertAlmostEqual(float(ranges.min()), 0.3)
        self.assertAlmostEqual(float(ns["scan_to_policy_rays"](scan).min()), 4.0)

    def test_adapter_can_see_obstacle_aligned_with_a_sample(self):
        angles = np.linspace(-math.pi, math.pi, 2031)
        body = np.arctan2(np.sin(math.pi - angles), np.cos(math.pi - angles))
        ranges = np.full(angles.shape, 4.0)
        ranges[np.abs(body - math.radians(90 / 47)) < math.radians(1)] = 0.3
        scan = SimpleNamespace(ranges=ranges, angle_min=float(angles[0]),
                               angle_increment=float(angles[1] - angles[0]))
        self.assertAlmostEqual(float(ns["scan_to_policy_rays"](scan).min()), 0.3)

    def test_observation_preserves_raw_meters_and_previous_raw_action(self):
        obs = ns["build_raw_obs"](1.2, 0.3, 0.1, -0.2, [0.8, -0.6], np.full(48, 0.4))
        self.assertEqual(obs.shape, (55,))
        np.testing.assert_allclose(obs[:7], [1.2, math.sin(0.3), math.cos(0.3), 0.1, -0.2, 0.8, -0.6])
        np.testing.assert_allclose(obs[7:], 0.4)

    def test_motor_peak_limit_scales_both_wheel_groups(self):
        self.assertEqual(motor_ns["velocity_to_motor_values"](0.15, 0.78), (6, 6, 30, 30))
        self.assertEqual(motor_ns["velocity_to_motor_values"](0.06, 0), (12, 12, 12, 12))


if __name__ == "__main__":
    unittest.main()
