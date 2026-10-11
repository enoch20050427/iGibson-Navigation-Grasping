import ast
import math
from pathlib import Path
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "integration" / "sugarbox_rl_approach_final2.py"
tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
prefixes = ("AVOID_TURN_", "HARD_RECOVERY_", "NO_PROGRESS_", "FRONT_", "SIDE_", "TEST_MAX_")
nodes = [node for node in tree.body if (
    isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id.startswith(prefixes)
        for target in node.targets
    )
) or (isinstance(node, ast.ClassDef) and node.name == "AntiJitterSafetyFilter")]
namespace = {"math": math, "clamp": lambda v, lo, hi: max(lo, min(hi, v))}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
Filter = namespace["AntiJitterSafetyFilter"]


class ProgressRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.filter = Filter()

    def command(self, now, distance=1.0, odom_vx=0.0, side=0.3):
        return self.filter.apply(
            0.1, -0.3, 1.0, 0.95, side, 1.0, odom_vx, now,
            target_distance=distance
        )

    def test_side_obstacle_stall_starts_bounded_backup(self):
        self.command(100.0)
        self.assertNotIn("RECOVERY", self.command(105.9)[2])
        vx, wz, reason, _ = self.command(106.0)
        self.assertEqual((vx, wz), (-0.08, 0.0))
        self.assertIn("cause=NO_PROGRESS", reason)

    def test_progress_and_clearance_do_not_start_backup(self):
        for index in range(20):
            self.assertNotIn("RECOVERY", self.command(100 + index, 1.0 - index * 0.04)[2])
        self.filter.reset_all()
        for index in range(20):
            self.assertNotIn("RECOVERY", self.command(100 + index, side=1.0)[2])

    def test_backup_timeout_returns_control_and_rearms_timer(self):
        self.command(100.0)
        self.command(106.0)
        self.assertIn("RECOVERY_BACKUP", self.command(106.5)[2])
        self.assertIn("RECOVERY_DONE", self.command(107.5)[2])
        self.assertIn("RECOVERY_SETTLE", self.command(107.6)[2])
        self.assertNotIn("RECOVERY", self.command(107.8)[2])
        self.assertNotIn("RECOVERY", self.command(108.0)[2])

    def test_backup_ends_at_measured_distance(self):
        self.command(100.0)
        self.command(106.0)
        for step in range(1, 4):
            result = self.command(106.0 + step * 0.25, odom_vx=-0.1)
        self.assertIn("RECOVERY_DONE", result[2])

    def test_pause_discards_previous_stall_time(self):
        self.command(100.0)
        self.filter.reset_progress()
        self.assertNotIn("RECOVERY", self.command(120.0)[2])

    def test_side_rejected_turn_cannot_become_forward_push(self):
        for right, left, turn in ((1.0, 0.1, 0.7), (0.1, 1.0, -0.7)):
            self.filter.reset_all()
            vx, wz, reason, _ = self.filter.apply(
                0.15, turn, 1.0, 0.95, right, left, 0.0, 100.0
            )
            self.assertEqual((vx, wz), (0.0, 0.0))
            self.assertIn("SIDE_GUARD", reason)

    def test_away_turn_and_reverse_remain_possible(self):
        for right, left, away in ((1.0, 0.1, -0.7), (0.1, 1.0, 0.7)):
            self.filter.reset_all()
            vx, wz, _, _ = self.filter.apply(
                0.15, away, 1.0, 0.95, right, left, 0.0, 100.0
            )
            self.assertEqual(wz, away)
            self.assertGreater(vx, 0.0)
            vx, wz, _, _ = self.filter.apply(
                -0.08, -away, 1.0, 0.95, right, left, 0.0, 100.2
            )
            self.assertEqual(vx, -0.08)

    def test_side_close_straight_command_is_not_allowed_to_push(self):
        vx, wz, reason, _ = self.filter.apply(
            0.15, 0.0, 1.0, 0.95, 0.1, 1.0, 0.0, 100.0
        )
        self.assertEqual((vx, wz), (0.0, 0.0))
        self.assertIn("NO_FORWARD_WITHOUT_ESCAPE_STEERING", reason)


if __name__ == "__main__":
    unittest.main()
