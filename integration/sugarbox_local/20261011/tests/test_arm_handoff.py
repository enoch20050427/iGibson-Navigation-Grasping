"""Arm/pixel handoff checks with fake devices; never contact the robot."""
import ast
import math
from pathlib import Path
import socket
import sys
import threading
from types import SimpleNamespace
from typing import Optional, Tuple
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "integration"))
import sugarbox_arm_approach as approach
from sugarbox_arm_feedback import read_angles, raw_to_degrees
from sugarbox_arm_pixels import NAV_POSE

SOURCE = ROOT / "integration" / "sugarbox_rl_motor_server_no_amplification.py"
tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SugarboxRLMotorServer"]
ns = dict(math=math, socket=socket, threading=threading, Optional=Optional, Tuple=Tuple,
          HOST="0.0.0.0", PORT=7000, time=SimpleNamespace(sleep=lambda _: None), read_angles=read_angles)
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), ns)
Server = ns["SugarboxRLMotorServer"]


class FakeDevice:
    def __init__(self):
        self.angles = list(NAV_POSE)
        self.motion = (0, 0, 0)
        self.writes = []

    def get_uart_servo_angle(self, servo):
        return self.angles[servo-1]

    def get_motion_data(self):
        return self.motion

    def set_uart_servo_angle(self, servo, angle, run_time):
        self.writes.append((servo, angle, run_time))
        self.angles[servo-1] = angle

    def set_uart_servo_angle_array(self, angle_s, run_time):
        self.writes.append(("array", list(angle_s), run_time))
        self.angles = list(angle_s)


def server(enabled=False):
    value = Server.__new__(Server)
    value.device = FakeDevice()
    value.allow_arm_observation = enabled
    value.allow_arm_initialization = False
    value.serial_lock = threading.Lock()
    value.stop = Mock()
    return value


def observe(value, **overrides):
    command = dict(action="arm_observe", expected_deg=list(NAV_POSE), s2_deg=137, confirmed=True)
    command.update(overrides)
    return value.arm_request(command)


class ArmRPCtests(unittest.TestCase):
    def test_initialization_disabled_by_default(self):
        import inspect
        self.assertIs(inspect.signature(Server.__init__).parameters["allow_arm_initialization"].default, False)

    def test_initialization_requires_empty_arm_operator_and_clear_workspace(self):
        fields = dict(action="arm_initialize", confirmed=True, empty_gripper=True, workspace_clear=True)
        for missing in ("server_opt_in", "confirmed", "empty_gripper", "workspace_clear"):
            value = server(True)
            value.allow_arm_initialization = missing != "server_opt_in"
            command = dict(fields)
            if missing != "server_opt_in":
                command[missing] = False
            self.assertFalse(value.arm_request(command)["ok"])
            self.assertEqual(value.device.writes, [])

    def test_supervised_bootstrap_recovers_sag_and_verifies_named_pose(self):
        value = server(True)
        value.allow_arm_initialization = True
        value.device.angles[2] = -1  # Unreadable sag, never reused as reached feedback.
        result = value.arm_request(dict(action="arm_initialize", confirmed=True,
                                        empty_gripper=True, workspace_clear=True))
        self.assertTrue(result["ok"])
        self.assertTrue(result["initialized"])
        self.assertEqual(value.device.writes, [("array", list(NAV_POSE), 2000)])

    def test_failed_initialization_readback_or_moving_chassis_is_not_success(self):
        for moving in (False, True):
            value = server(True)
            value.allow_arm_initialization = True
            if moving:
                value.device.motion = (.1, 0, 0)
            else:
                value.device.set_uart_servo_angle_array = Mock()
                value.device.angles[2] = -1
            result = value.arm_request(dict(action="arm_initialize", confirmed=True,
                                            empty_gripper=True, workspace_clear=True))
            self.assertFalse(result["ok"])
            value.stop.assert_called()

    def test_constructor_disables_movement_by_default(self):
        import inspect
        self.assertIs(inspect.signature(Server.__init__).parameters["allow_arm_observation"].default, False)

    def test_disabled_bridge_or_unconfirmed_request_never_writes(self):
        for enabled, confirmed in ((False, True), (True, False)):
            value = server(enabled)
            self.assertFalse(observe(value, confirmed=confirmed)["ok"])
            self.assertEqual(value.device.writes, [])
            value.stop.assert_called()

    def test_reads_status_without_joint_commands(self):
        value = server()
        reply = value.process_command(dict(action=" ARM_STATUS ", request_id="test"))
        self.assertTrue(reply["ok"])
        self.assertFalse(reply["allow_arm_observation"])
        self.assertEqual(reply["request_id"], "test")
        self.assertEqual(value.device.writes, [])

    def test_only_s2_changes_and_can_return_to_travel(self):
        value = server(True)
        self.assertTrue(observe(value)["ok"])
        self.assertTrue(observe(value, expected_deg=list(value.device.angles), s2_deg=140)["ok"])
        self.assertEqual(value.device.writes, [(2, 137, 1000), (2, 140, 1000)])
        self.assertEqual(value.device.angles, list(NAV_POSE))

    def test_invalid_target_or_changed_pose_never_writes(self):
        for changes in (dict(s2_deg=119), dict(s2_deg=float("nan")),
                        dict(expected_deg=[0]*6), dict(expected_deg=None)):
            value = server(True)
            self.assertFalse(observe(value, **changes)["ok"])
            self.assertEqual(value.device.writes, [])

    def test_non_travel_closed_jaw_or_moving_chassis_never_writes(self):
        for index, angle in ((1, 74.2), (5, 110), (0, 100)):
            value = server(True)
            value.device.angles[index] = angle
            self.assertFalse(observe(value, expected_deg=list(value.device.angles))["ok"])
            self.assertEqual(value.device.writes, [])
        value = server(True)
        value.device.motion = (.02, 0, 0)
        self.assertFalse(observe(value)["ok"])
        self.assertEqual(value.device.writes, [])

    def test_servo_feedback_failure_blocks_motion(self):
        value = server(True)
        value.device.angles[2] = -1
        self.assertFalse(observe(value)["ok"])
        self.assertEqual(value.device.writes, [])

    def test_no_step_between_extremes(self):
        value = server(True)
        value.device.angles[1] = 137
        self.assertFalse(observe(value, expected_deg=list(value.device.angles), s2_deg=143)["ok"])
        self.assertEqual(value.device.writes, [])


class HandoffTests(unittest.TestCase):
    def test_cancel_before_step_never_moves_or_continues_to_grasp(self):
        motor = approach.MotorConnection.__new__(approach.MotorConnection)
        motor.request = Mock(return_value=dict(protocol="sugarbox-arm-observe-v1",
                                             allow_arm_observation=True, servo_deg=list(NAV_POSE)))
        motor.stop = Mock()
        with self.assertRaises(approach.ArmMotionCancelled):
            motor.observe(130, cancel=Mock(side_effect=[False, True]))
        self.assertEqual(motor.request.call_args.args[0], "arm_status")
        self.assertEqual(motor.request.call_count, 1)
        motor.stop.assert_called_once()

    def test_cancel_after_step_prevents_remaining_pose_commands(self):
        motor = approach.MotorConnection.__new__(approach.MotorConnection)
        value = server(True)
        motor.stop = Mock()
        def request(action, **fields):
            return value.arm_request(dict(action=action, **fields))
        motor.request = request
        with self.assertRaises(approach.ArmMotionCancelled):
            motor.observe(130, cancel=Mock(side_effect=[False, False, True]))
        self.assertEqual(value.device.writes, [(2, 138, 1000)])
        motor.stop.assert_called_once()

    def test_lower_observation_and_return_use_verified_three_degree_steps(self):
        motor = approach.MotorConnection.__new__(approach.MotorConnection)
        value = server(True)
        calls = []
        def request(action, **fields):
            calls.append((action, fields))
            reply = value.arm_request(dict(action=action, **fields))
            if not reply["ok"]:
                raise RuntimeError(str(reply))
            return reply
        motor.request = request
        self.assertEqual(motor.observe(130)["servo_deg"][1], 130)
        self.assertEqual(motor.observe(140)["servo_deg"][1], 140)
        for action, fields in calls:
            if action == "arm_observe":
                self.assertLessEqual(abs(fields["s2_deg"] - fields["expected_deg"][1]), 3)
        self.assertEqual([v[1] for v in value.device.writes], [138, 136, 134, 132, 130, 132, 134, 136, 138, 140])

    def test_unchecked_observation_target_is_rejected_before_rpc(self):
        motor = approach.MotorConnection.__new__(approach.MotorConnection)
        motor.request = Mock()
        for target in (119, 143, float("nan")):
            with self.assertRaises(ValueError):
                motor.observe(target)
        motor.request.assert_not_called()

    def test_integer_readback_settling_does_not_exceed_physical_step_limit(self):
        motor = approach.MotorConnection.__new__(approach.MotorConnection)
        value = server(True)
        value.device.angles[1] = 137
        read = value.read_arm_angles
        count = [0]
        def settling_read():
            count[0] += 1
            if count[0] in (3, 6):
                value.device.angles[1] -= 1
            return read()
        value.read_arm_angles = settling_read
        def request(action, **fields):
            reply = value.arm_request(dict(action=action, **fields))
            if not reply["ok"]:
                raise RuntimeError(str(reply))
            return reply
        motor.request = request
        self.assertEqual(motor.observe(140)["servo_deg"][1], 140)
        self.assertEqual([v[1] for v in value.device.writes], [139, 140])

    def test_yolo_selector_receives_class_names(self):
        detector = approach.PixelDetector.__new__(approach.PixelDetector)
        detector.running = True
        detector.lock = threading.Lock()
        detector.model = Mock(names={0: "sugarbox"})
        result = object()
        detector.model.predict.return_value = [result]
        detector.select_box = Mock(return_value=dict(bbox=(1, 2, 3, 4), conf=.9))
        def read():
            detector.running = False
            return True, object(), 100
        detector.reader = SimpleNamespace(read=read)
        detector.loop()
        detector.select_box.assert_called_once_with(result, detector.model.names)
        self.assertEqual(detector.snapshot(), ((1, 2, 3, 4), 100, .9))

    def test_camera_preflight_releases_on_valid_or_invalid_size(self):
        for shape, valid in (((480, 640, 3), True), ((720, 1280, 3), False)):
            reader = Mock()
            reader.read.return_value = (True, SimpleNamespace(shape=shape), 100)
            app = SimpleNamespace(LatestFrameReader=Mock(return_value=reader))
            with patch.object(approach.time, "time", return_value=100):
                if valid:
                    approach.check_arm_camera(app)
                else:
                    with self.assertRaises(RuntimeError):
                        approach.check_arm_camera(app)
            reader.release.assert_called_once()

    def test_abort_before_handoff_starts_no_remote_work(self):
        task = approach.GraspTask("127.0.0.1")
        task.cancel()
        with patch.object(approach, "ssh") as ssh:
            task.run()
        ssh.assert_not_called()
        self.assertTrue(task.done.is_set())
        self.assertIsNotNone(task.error)

    def test_grasp_never_starts_if_rl_owner_did_not_stop(self):
        task = approach.GraspTask("127.0.0.1")
        with patch.object(approach, "ssh", side_effect=["", "active\n"]) as ssh:
            task.run()
        self.assertEqual(ssh.call_count, 2)
        self.assertFalse(task.owns_grasp_phase)
        self.assertIn("did not stop", task.error)

    def test_missing_transient_unit_can_handoff_only_after_device_release(self):
        task = approach.GraspTask("127.0.0.1")
        replies = ["", "", '{"free":true}', "", ""]
        statuses = [dict(ok=True, servo_deg=list(NAV_POSE)), dict(ok=True),
                    dict(ok=True, servo_deg=list(approach.E1_POSE)), dict(confirmed=True)]
        with patch.object(approach, "ssh", side_effect=replies) as ssh, \
                patch.object(approach, "graspctl", side_effect=statuses):
            task.run()
        self.assertIsNone(task.error)
        self.assertTrue(task.result["confirmed"])
        self.assertIn("fuser", ssh.call_args_list[2].args[1])

    def test_missing_unit_with_busy_serial_never_starts_grasp(self):
        task = approach.GraspTask("127.0.0.1")
        with patch.object(approach, "ssh", side_effect=["", "", '{"free":false}']) as ssh:
            task.run()
        self.assertEqual(ssh.call_count, 3)
        self.assertFalse(task.owns_grasp_phase)
        self.assertIn("serial port", task.error)

    def test_manual_navigation_exit_never_grasps_or_tilts_arm(self):
        profile = SimpleNamespace(rear_handoff_v=360)
        motor = Mock()
        motor.request.return_value = dict(protocol="sugarbox-arm-observe-v1",
                                         allow_arm_observation=True, servo_deg=list(NAV_POSE))
        motor.initialize.return_value = dict(initialized=True, servo_deg=list(NAV_POSE))
        with patch.object(approach, "load_profile", return_value=profile), \
                patch.object(approach, "MotorConnection", return_value=motor), \
                patch.object(approach, "check_arm_camera"), \
                patch.object(approach.subprocess, "run", return_value=SimpleNamespace(returncode=0)), \
                patch.object(approach, "run_pixel_stage") as pixel, \
                patch.dict(approach.os.environ, {}, clear=True), \
                patch.dict(sys.modules, {"cv2": Mock(), "sugarbox_rl_approach_final2": Mock()}):
            self.assertEqual(approach.main(["--real", "--i-am-beside-the-robot", "--arm-empty-and-clear",
                                           "--profile", "fake.json"]), 0)
        pixel.assert_not_called()
        motor.observe.assert_not_called()
        motor.initialize.assert_called_once()
        motor.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
