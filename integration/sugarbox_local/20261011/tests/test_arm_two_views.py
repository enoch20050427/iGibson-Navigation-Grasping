import sys
from pathlib import Path
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "diagnostics"))
from check_arm_two_views import check_views, ArmMotionCancelled


class TwoViewCheckTests(unittest.TestCase):
    def motor(self, pose=(90, 180, 0, 0, 90, 30)):
        motor = Mock()
        motor.request.return_value = dict(servo_deg=list(pose) if pose is not None else None)
        motor.view.side_effect = [dict(servo_deg=[90, 140, 0, 0, 90, 30]),
                                  dict(servo_deg=[90, 120, 0, 0, 90, 30])]
        return motor

    def test_only_authorized_views_in_order(self):
        motor, capture = self.motor(), Mock()
        check_views(motor, lambda: False, capture)
        self.assertEqual([call.args[0] for call in motor.view.call_args_list], ["travel", "down20"])
        self.assertEqual([call.args[0] for call in capture.call_args_list], ["travel", "down20"])
        self.assertTrue(all(call.kwargs["oem_path_checked"] for call in motor.view.call_args_list))
        motor.initialize.assert_not_called()
        motor.velocity.assert_not_called()

    def test_cancel_before_any_command(self):
        motor = self.motor()
        with self.assertRaises(ArmMotionCancelled):
            check_views(motor, lambda: True, Mock())
        motor.request.assert_not_called()
        motor.view.assert_not_called()

    def test_wrong_initial_pose_refused(self):
        for pose in (None, [90, 140, 0, 0, 90, 30], [90, 180, 20, 0, 90, 30]):
            with self.subTest(pose=pose):
                motor = self.motor(pose)
                with self.assertRaises(RuntimeError):
                    check_views(motor, lambda: False, Mock())
                motor.view.assert_not_called()

    def test_wrong_readback_stops_before_second_view(self):
        motor, capture = self.motor(), Mock()
        motor.view.side_effect = [dict(servo_deg=[90, 180, 0, 0, 90, 30])]
        with self.assertRaises(RuntimeError):
            check_views(motor, lambda: False, capture)
        self.assertEqual(motor.view.call_count, 1)
        capture.assert_not_called()

    def test_capture_failure_stops_before_second_view(self):
        motor, capture = self.motor(), Mock(side_effect=RuntimeError("camera missing"))
        with self.assertRaises(RuntimeError):
            check_views(motor, lambda: False, capture)
        self.assertEqual(motor.view.call_count, 1)


if __name__ == "__main__":
    unittest.main()
