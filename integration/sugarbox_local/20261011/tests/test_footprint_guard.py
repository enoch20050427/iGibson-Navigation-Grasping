import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace
import time
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "integration" / "sugarbox_rl_approach_final2.py"
tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
names = {"SCAN_STALE_TIMEOUT", "LIDAR_REFLECT_FRONT_BACK", "MOTOR_SEND_INTERVAL",
         "SIDE_SAFETY_MAX_ANGLE_DEG"}
nodes = [node for node in tree.body if (
    isinstance(node, ast.ClassDef) and node.name in {"FootprintSafetyGuard", "MotorClient"}
) or (
    isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id in names for target in node.targets
    )
)]
namespace = {"math": math, "time": time, "json": json,
             "FOOTPRINT_GUARD_ENABLED": True}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
Guard = namespace["FootprintSafetyGuard"]
MotorClient = namespace["MotorClient"]


def make_scan(points=(), stamp=10.0):
    ranges = [4.0] * 720
    increment = 2.0 * math.pi / len(ranges)
    for x, y in points:
        raw_angle = math.pi - math.atan2(y, x)
        index = round((raw_angle + math.pi) / increment) % len(ranges)
        ranges[index] = math.hypot(x, y)
    return SimpleNamespace(ranges=ranges, angle_min=-math.pi,
                           angle_increment=increment, range_min=0.02,
                           range_max=8.0, stamp=stamp)


class FootprintTests(unittest.TestCase):
    def test_disabled_experimental_guard_preserves_ppo_escape(self):
        motor = MotorClient("unused", 0)
        sent = []
        motor._send_json = sent.append
        namespace["FOOTPRINT_GUARD_ENABLED"] = False
        try:
            for vx, wz in ((0.1, -0.4), (-0.08, -0.3), (0.0, 0.0)):
                motor.send_velocity(vx, wz, force=True)
                self.assertEqual(motor.last_sent_velocity, (vx, wz))
                self.assertEqual(motor.last_guard_reason, "")
                self.assertEqual(sent[-1], {"action": "velocity", "vx": vx, "wz": wz})
        finally:
            namespace["FOOTPRINT_GUARD_ENABLED"] = True

    def check(self, points, vx, wz):
        return Guard().filter(make_scan(points), vx, wz, 10.0)

    def test_clear_path_preserves_commands(self):
        for vx, wz in ((0.15, 0.0), (0.0, 0.5), (0.1, -0.4)):
            self.assertEqual(self.check([], vx, wz), (vx, wz, ""))

    def test_near_side_blocks_forward_and_turn(self):
        for vx, wz in ((0.06, 0.0), (0.06, -0.65), (0.0, 0.5)):
            result = self.check([(0.0, 0.15)], vx, wz)
            self.assertEqual(result[:2], (0.0, 0.0))
            self.assertIn("FOOTPRINT_BLOCK", result[2])

    def test_forward_swept_path_and_small_motor_amplification(self):
        self.assertEqual(self.check([(0.14, 0.0)], 0.06, 0.0)[:2], (0.0, 0.0))

    def test_reverse_is_blocked_without_trusted_rear_coverage(self):
        for points in ([], [(-0.45, 0.0)]):
            self.assertEqual(self.check(points, -0.08, 0.0),
                             (0.0, 0.0, "FOOTPRINT_REVERSE_UNOBSERVED"))

    def test_bounded_straight_recovery_can_reverse_without_rear_scan(self):
        scan = make_scan()
        result = Guard().filter(scan, -0.08, 0.0, 10.0, allow_blind_recovery=True)
        self.assertEqual(result, (-0.08, 0.0, "BLIND_RECOVERY_NO_REAR_COVERAGE"))
        self.assertEqual(Guard().filter(scan, -0.2, 0.0, 10.0, True)[0], -0.08)
        self.assertEqual(Guard().filter(scan, -0.08, 0.4, 10.0, True)[:2], (0.0, 0.0))
        self.assertEqual(Guard().filter(None, -0.08, 0.0, 10.0, True)[:2], (0.0, 0.0))

    def test_rotation_checks_corner_in_trusted_sector(self):
        self.assertEqual(self.check([(0.0, 0.20)], 0.0, 0.4)[:2], (0.0, 0.0))

    def test_rear_chassis_echoes_do_not_block_forward_motion(self):
        self.assertEqual(self.check([(-0.11, 0.0), (-0.26, -0.14)], 0.1, 0.0),
                         (0.1, 0.0, ""))

    def test_wrapped_angles_remain_in_trusted_sector(self):
        for sign in (-1, 1):
            result = self.check([(0.05, sign * 0.15)], 0.1, 0.0)
            self.assertEqual(result[:2], (0.0, 0.0))

    def test_points_outside_ninety_degrees_are_excluded(self):
        self.assertEqual(self.check([(-0.05, 0.15)], 0.1, 0.0), (0.1, 0.0, ""))

    def test_obstacle_outside_swept_path_passes(self):
        self.assertEqual(self.check([(-0.1, 0.5)], 0.1, 0.0), (0.1, 0.0, ""))

    def test_missing_stale_invalid_scan_blocks_motion(self):
        scans = [None, make_scan(stamp=8.0), make_scan()]
        scans[-1].ranges = [float("nan")] * 720
        for scan in scans:
            self.assertEqual(Guard().filter(scan, 0.1, 0.0, 10.0)[:2], (0.0, 0.0))

    def test_stop_always_passes_without_scan(self):
        self.assertEqual(Guard().filter(None, 0.0, 0.0, 10.0), (0.0, 0.0, ""))

    def test_motor_boundary_catches_every_motion_mode(self):
        motor = MotorClient("unused", 0)
        sent = []
        motor._send_json = sent.append
        motor.guard_scan = make_scan([(0.0, 0.15)], stamp=time.time())
        for vx, wz in ((0.15, -0.5), (0.0, 0.5), (0.08, 0.0), (-0.08, 0.0)):
            motor.send_velocity(vx, wz, force=True)
            self.assertEqual(sent[-1], {"action": "velocity", "vx": 0.0, "wz": 0.0})
            self.assertEqual(motor.last_sent_velocity, (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
