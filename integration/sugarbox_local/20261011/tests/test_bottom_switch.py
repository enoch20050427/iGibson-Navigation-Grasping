import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "integration"))
from sugarbox_arm_pixels import (ApproachProfile, BottomEdgeLatch, PixelProfile, StagedPixelServo,
                                NAV_POSE, FORWARD_VIEW_POSE)
import sugarbox_arm_approach as approach
from test_arm_handoff import server


def profile(**changes):
    data = dict(schema="sugarbox-arm-bottom-switch-v3", rear_handoff_v=470, arm_switch_v=432,
                travel_view_confirmed=True, forward_view_confirmed=True,
                final=dict(schema="sugarbox-arm-pixels-v1", target_u=320, target_v=350,
                           yaw_sign=1, forward_pixel_sign=1, observation_s2_deg=120, confirmed=True))
    data.update(changes)
    return ApproachProfile.from_dict(data)


class BottomSwitchTests(unittest.TestCase):
    def test_twenty_degree_mode_is_separate_from_legacy_two_degree_stages(self):
        p = profile()
        self.assertEqual([s.observation_s2_deg for s in p.stages], [140, 120])
        self.assertEqual(p.stages[0].target_u, 320)
        self.assertGreater(p.stages[0].target_v, p.bottom_switch_v)
        with self.assertRaises(ValueError):
            ApproachProfile(p.stages, 470)

    def test_edge_switch_does_not_wait_for_centering_or_grasp_ready(self):
        s = StagedPixelServo(profile())
        for i in range(3):
            t = 100+i*.1
            out = s.step([390, 300, 470, 440], t, t, .9)
        self.assertEqual(out, (0, 0, "ARM_POSE_SWITCH"))
        self.assertEqual(s.pending_pose, 120)
        self.assertFalse(s.ready)
        s.confirm_pose(120, 101)
        self.assertEqual(s.profile.target_v, 350)
        self.assertFalse(s.ready)
        self.assertEqual(s.step([280, 250, 360, 350], 100.9, 101.4, .9)[2], "ARM_POSE_SETTLING")

    def test_bottom_clipping_can_switch_view_but_never_authorize_grasp(self):
        s = StagedPixelServo(profile())
        for i in range(3):
            t = 100+i*.1
            out = s.step([280, 330, 360, 480], t, t, .9)
        self.assertEqual(out[2], "ARM_POSE_SWITCH")
        self.assertFalse(s.ready)

    def test_duplicates_missing_low_confidence_jump_and_stale_do_not_switch(self):
        latch = BottomEdgeLatch(432)
        box = [280, 300, 360, 440]
        self.assertFalse(latch.step(box, 100, 100, .9))
        for _ in range(10):
            self.assertFalse(latch.step(box, 100, 100.1, .9))
        self.assertEqual(latch.count, 1)
        for bad in ((None, 100.1, 100.1, .9), (box, 100.1, 100.1, .2),
                    (box, 100, 101, .9), ([0, 300, 70, 440], 100.1, 100.1, .9)):
            self.assertFalse(latch.step(*bad))
            self.assertEqual(latch.count, 0)
        latch.step(box, 102, 102, .9)
        latch.step(box, 102.1, 102.1, .9)
        self.assertFalse(latch.step([450, 300, 530, 440], 102.2, 102.2, .9))
        self.assertEqual(latch.count, 1)

    def test_sensor_hold_resets_pending_edge_evidence(self):
        s = StagedPixelServo(profile())
        for t in (100, 100.1):
            s.step([280, 300, 360, 440], t, t, .9)
        s.hold("ARM_SENSOR_STALE")
        self.assertNotEqual(s.step([280, 300, 360, 440], 101, 101, .9)[2], "ARM_POSE_SWITCH")

    def test_invalid_mode_edges_or_final_pose_refused(self):
        for edge in (True, 350, 451, float("nan")):
            with self.assertRaises(ValueError):
                profile(arm_switch_v=edge)
        p = profile()
        with self.assertRaises(ValueError):
            ApproachProfile((p.stages[0], PixelProfile(320, 350, 1, 1, 130, True)), 470, 432)

    def test_unconfirmed_template_fails_before_hardware(self):
        path = ROOT / "diagnostics/arm_bottom_switch.template.json"
        with patch.object(approach, "MotorConnection") as motor:
            with self.assertRaises(ValueError):
                approach.main(["--real", "--i-am-beside-the-robot", "--arm-empty-and-clear",
                               "--profile", str(path)])
        motor.assert_not_called()


class NamedViewTests(unittest.TestCase):
    def test_supervised_steps_require_opt_in_empty_arm_and_small_request(self):
        for fault in ("disabled", "large", "grip", "motion"):
            value = server(True)
            value.allow_oem_validation = fault != "disabled"
            if fault == "grip":
                value.device.angles[5] = 110
            if fault == "motion":
                value.device.motion = (.05, 0, 0)
            command = dict(action="arm_oem_step", expected_deg=list(value.device.angles),
                           s2_deg=180 if fault == "large" else 142, supervised=True)
            self.assertFalse(value.arm_request(command)["ok"])
            self.assertEqual(value.device.writes, [])

    def test_supervised_client_uses_small_steps_but_does_not_claim_path_safe(self):
        value = server(True)
        value.allow_oem_validation = True
        motor = approach.MotorConnection.__new__(approach.MotorConnection)
        def request(action, **fields):
            reply = value.arm_request(dict(action=action, **fields))
            if not reply["ok"]:
                raise RuntimeError(str(reply))
            if action == "arm_oem_step":
                self.assertFalse(reply["path_validated"])
            return reply
        motor.request = request
        readings = motor.validate_oem_path()
        self.assertEqual(readings[-1], list(FORWARD_VIEW_POSE))
        self.assertEqual(len(value.device.writes), 20)
        self.assertTrue(all(joint == 2 and runtime == 500 for joint, _, runtime in value.device.writes))
        self.assertTrue(all(b[1]-a[1] <= 2 for a, b in zip(readings, readings[1:])))

    def test_supervised_validation_cancel_prevents_remaining_steps(self):
        value = server(True)
        value.allow_oem_validation = True
        motor = approach.MotorConnection.__new__(approach.MotorConnection)
        motor.stop = Mock()
        motor.request = lambda action, **fields: value.arm_request(dict(action=action, **fields))
        with self.assertRaises(approach.ArmMotionCancelled):
            motor.validate_oem_path(cancel=Mock(side_effect=[False, False, True]))
        self.assertEqual(value.device.writes, [(2, 142, 500)])

    def test_oem_forward_pose_changes_only_lower_s2_joint(self):
        self.assertEqual([i for i, (a, b) in enumerate(zip(NAV_POSE, FORWARD_VIEW_POSE)) if a != b], [1])
        self.assertEqual(FORWARD_VIEW_POSE[1], 180)

    def test_unvalidated_forward_client_never_opens_an_arm_request(self):
        m = approach.MotorConnection.__new__(approach.MotorConnection)
        m.request = Mock()
        with self.assertRaises(ValueError):
            m.view("forward")
        m.request.assert_not_called()

    def command(self, value, view="down20", **changes):
        command = dict(action="arm_view", view=view, expected_deg=list(value.device.angles), confirmed=True)
        command.update(changes)
        return value.arm_request(command)

    def test_down_twenty_is_one_named_motion_not_ten_calibration_steps(self):
        value = server(True)
        reply = self.command(value)
        self.assertTrue(reply["ok"])
        self.assertEqual(value.device.writes, [(2, 120, 2000)])
        self.assertTrue(self.command(value, "travel")["ok"])
        self.assertEqual(value.device.angles, list(NAV_POSE))

    def test_forward_requires_separate_opt_in_and_actual_empty_pose(self):
        value = server(True)
        self.assertFalse(self.command(value, "forward")["ok"])
        self.assertEqual(value.device.writes, [])
        value.allow_arm_forward_view = True
        self.assertFalse(self.command(value, "forward")["ok"])
        self.assertEqual(value.device.writes, [])
        self.assertTrue(self.command(value, "forward", oem_path_checked=True)["ok"])
        self.assertEqual(value.device.angles, [90, 180, 0, 0, 90, 30])
        self.assertEqual(value.device.writes, [(2, 180, 2000)])
        self.assertTrue(self.command(value, "travel", oem_path_checked=True)["ok"])

    def test_moving_chassis_closed_gripper_and_unknown_pose_never_write(self):
        for fault in ("motion", "grip", "unknown", "expected", "confirmed"):
            value = server(True)
            changes = {}
            if fault == "motion":
                value.device.motion = (.05, 0, 0)
            elif fault == "grip":
                value.device.angles[5] = 110
            elif fault == "unknown":
                value.device.angles[2] = 30
            elif fault == "expected":
                changes["expected_deg"] = [0]*6
            else:
                changes["confirmed"] = False
            self.assertFalse(self.command(value, **changes)["ok"])
            self.assertEqual(value.device.writes, [])

    def test_failed_readback_not_success(self):
        value = server(True)
        value.device.set_uart_servo_angle = Mock()
        self.assertFalse(self.command(value)["ok"])

    def test_client_validates_capability_actual_view_and_cancel(self):
        m = approach.MotorConnection.__new__(approach.MotorConnection)
        m.stop = Mock()
        m.request = Mock(return_value=dict(named_views=[]))
        with self.assertRaises(RuntimeError):
            m.view("forward", oem_path_checked=True)
        self.assertEqual(m.request.call_count, 1)
        m.request.reset_mock()
        with self.assertRaises(approach.ArmMotionCancelled):
            m.view("forward", cancel=lambda: True, oem_path_checked=True)
        m.request.assert_not_called()
        m.request.side_effect = [dict(named_views=["down20"], servo_deg=list(NAV_POSE)),
                                 dict(view="down20", servo_deg=list(NAV_POSE))]
        with self.assertRaises(RuntimeError):
            m.view("down20")

    def test_forward_profile_requires_physical_check_before_hardware(self):
        data = dict(schema="sugarbox-arm-bottom-switch-v3", rear_handoff_v=470, arm_switch_v=432,
                    travel_view_confirmed=True, forward_view_confirmed=False,
                    final=dict(schema="sugarbox-arm-pixels-v1", target_u=320, target_v=350,
                               yaw_sign=1, forward_pixel_sign=1, observation_s2_deg=120, confirmed=True))
        with patch.object(Path, "read_text", return_value=json.dumps(data)), \
                patch.object(approach, "MotorConnection") as motor:
            with self.assertRaises(ValueError):
                approach.main(["--real", "--i-am-beside-the-robot", "--arm-empty-and-clear",
                               "--profile", "unused.json"])
        motor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
