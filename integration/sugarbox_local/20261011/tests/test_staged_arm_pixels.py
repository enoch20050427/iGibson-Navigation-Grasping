import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integration"))
import sugarbox_arm_approach as approach
from sugarbox_arm_pixels import (ApproachProfile, PixelProfile, StagedPixelServo,
                                rear_at_handoff_edge, should_handoff)


def pixel(s2, **changes):
    fields = dict(target_u=320, target_v=350, yaw_sign=1, forward_pixel_sign=1,
                  observation_s2_deg=s2, confirmed=True)
    fields.update(changes)
    return PixelProfile(**fields)


def staged():
    return ApproachProfile((pixel(130), pixel(128, target_u=340, target_v=390)), rear_handoff_v=360)


def box(profile):
    return [profile.target_u-40, profile.target_v-100, profile.target_u+40, profile.target_v]


def reach(servo, start):
    for i in range(5):
        stamp = start + i*.1
        output = servo.step(box(servo.profile), stamp, stamp, .9)
    return output


class StagedArmTests(unittest.TestCase):
    def test_template_refuses_real_run_before_hardware_or_navigation(self):
        template = Path(__file__).resolve().parents[1] / "diagnostics/arm_staged_profile.template.json"
        with patch.object(approach, "MotorConnection") as motor, \
                patch.object(approach.subprocess, "run") as subprocess:
            with self.assertRaises(ValueError):
                approach.main(["--real", "--i-am-beside-the-robot", "--arm-empty-and-clear",
                               "--profile", str(template)])
        motor.assert_not_called()
        subprocess.assert_not_called()

    def test_boolean_calibration_values_cannot_authorize_motion(self):
        for changes in (dict(target_u=True), dict(yaw_sign=True), dict(forward_pixel_sign=True)):
            with self.assertRaises(ValueError):
                pixel(130, **changes)

    def test_unconfirmed_any_stage_refuses_entire_pipeline(self):
        profile = ApproachProfile((pixel(130), pixel(128, confirmed=False)), 360)
        with self.assertRaises(ValueError):
            StagedPixelServo(profile)

    def test_wrong_order_large_steps_missing_edge_or_unchecked_angle_is_rejected(self):
        for poses, edge in (((130, 134), 360), ((130, 126), 360), ((132, 130), 360),
                            ((130, 128), None), ((130, 128), math.nan), ((130, 128), 100)):
            with self.assertRaises(ValueError):
                ApproachProfile(tuple(pixel(s2) for s2 in poses), edge)
        with self.assertRaises(ValueError):
            pixel(119)

    def test_intermediate_ready_only_requests_pose_switch(self):
        servo = StagedPixelServo(staged())
        self.assertEqual(reach(servo, 100), (0, 0, "ARM_POSE_SWITCH"))
        self.assertFalse(servo.ready)
        self.assertEqual(servo.pending_pose, 128)
        self.assertEqual(servo.step(box(servo.profile), 100.5, 100.5, .9)[:2], (0, 0))

    def test_wrong_pose_confirmation_preserves_stage_and_never_grasps(self):
        servo = StagedPixelServo(staged())
        reach(servo, 100)
        with self.assertRaises(ValueError):
            servo.confirm_pose(131, 101)
        self.assertEqual(servo.index, 0)
        self.assertEqual(servo.pending_pose, 128)
        self.assertFalse(servo.ready)

    def test_pre_motion_and_settling_frames_cannot_resume_or_trigger_grasp(self):
        servo = StagedPixelServo(staged())
        reach(servo, 100)
        servo.confirm_pose(128, 101)
        self.assertEqual(servo.profile.target_v, 390)
        for stamp in (100.4, 101.1, 101.3):
            self.assertEqual(servo.step(box(servo.profile), stamp, 101.3, .9)[2], "ARM_POSE_SETTLING")
        self.assertFalse(servo.ready)
        for stamp in (101.4, 101.5):
            self.assertEqual(servo.step(box(servo.profile), stamp, stamp, .9)[:2], (0, 0))
            self.assertFalse(servo.ready)
        self.assertEqual(servo.step(box(servo.profile), 101.6, 101.6, .9)[2], "ARM_VERIFY_READY")
        self.assertFalse(servo.ready)

    def test_only_final_pose_five_fresh_frames_authorize_grasp(self):
        servo = StagedPixelServo(staged())
        reach(servo, 100)
        servo.confirm_pose(128, 101)
        self.assertEqual(reach(servo, 101.4)[2], "ARM_READY")
        self.assertTrue(servo.ready)
        self.assertIsNone(servo.pending_pose)

    def test_previous_pose_pixel_goal_is_not_final_goal(self):
        servo = StagedPixelServo(staged())
        previous_box = box(servo.profile)
        reach(servo, 100)
        servo.confirm_pose(128, 101)
        for stamp in (101.4, 101.5, 101.6, 101.7, 101.8):
            output = servo.step(previous_box, stamp, stamp, .9)
        self.assertFalse(servo.ready)
        self.assertGreater(output[0], 0)

    def test_loss_after_pose_switch_never_blind_moves(self):
        servo = StagedPixelServo(staged())
        reach(servo, 100)
        servo.confirm_pose(128, 101)
        self.assertEqual(servo.step(None, 101.4, 101.4, .9)[:2], (0, 0))
        self.assertFalse(servo.ready)

    def test_single_pose_legacy_profile_still_works(self):
        servo = StagedPixelServo(ApproachProfile((pixel(130),)))
        self.assertEqual(reach(servo, 100)[2], "ARM_READY")
        self.assertTrue(servo.ready)

    def test_early_handoff_does_not_wait_for_loss_but_retains_clearance_gate(self):
        result = dict(det=dict(bbox=[280, 200, 360, 370], conf=.9))
        edge = rear_at_handoff_edge(result, 100, 100.1, 360)
        self.assertTrue(edge)
        self.assertTrue(should_handoff(100.1, 100, .7, True, True, edge))
        self.assertFalse(should_handoff(100.1, 100, .7, False, True, edge))
        self.assertFalse(should_handoff(100.1, 100, 1.2, True, True, edge))

    def test_stale_wrong_or_unconfigured_rear_edge_is_ignored(self):
        for result, stamp, now, edge in ((dict(det=dict(bbox=[280, 200, 360, 370], conf=.9)), 100, 101, 360),
                                       (dict(det=dict(bbox=[280, 200, 360, 370], conf=.2)), 100, 100, 360),
                                       (dict(det=dict(bbox=[280, 200, 360, 370], conf=.9)), 100, 100, None),
                                       (None, 100, 100, 360)):
            self.assertFalse(rear_at_handoff_edge(result, stamp, now, edge))


class Clock:
    def __init__(self):
        self.now = 100
        self.ticks = 0

    def time(self):
        self.now += .1
        return self.now

    def monotonic(self):
        self.ticks += 1
        return self.ticks


class StagedRuntimeTests(unittest.TestCase):
    def run_stage(self, cv_key=0, wrong_pose=False, bottom=False):
        profile = (ApproachProfile((pixel(140, target_v=456), pixel(120, target_v=350)), 470, 432, True)
                   if bottom else staged())
        clock = Clock()
        motor, cv2, reader, detector, ros, task = (Mock() for _ in range(6))
        current_pose = [profile.stages[0].observation_s2_deg]
        def observe(target, **kwargs):
            current_pose[0] = target
            angle = target + 2 if wrong_pose and target == profile.stages[1].observation_s2_deg else target
            return dict(servo_deg=[90, angle, 0, 0, 90, 30])
        motor.observe.side_effect = observe
        motor.view.side_effect = lambda name, **kwargs: observe({"travel": 140, "down20": 120}[name], **kwargs)
        reader.read.side_effect = lambda: (True, Mock(), clock.now)
        detector.snapshot.side_effect = lambda: (
            box(profile.stages[0] if current_pose[0] == profile.stages[0].observation_s2_deg else profile.stages[1]), clock.now, .9)
        ros.snapshot.side_effect = lambda: (SimpleNamespace(stamp=clock.now), SimpleNamespace(stamp=clock.now))
        cv2.waitKey.return_value = cv_key
        task.done.is_set.return_value = True
        task.error = None
        task.result = dict(confirmed=True)
        app = SimpleNamespace(LatestFrameReader=Mock(return_value=reader), YOLO=Mock(), YOLO_MODEL_PATH="unused",
                              select_best_box=Mock(), RosSensorBridge=Mock(return_value=ros), ROSBRIDGE_PORT=9090,
                              SCAN_STALE_TIMEOUT=.3, ODOM_STALE_TIMEOUT=.3, FRONT_ROBUST_MIN_POINTS=4,
                              raw_front_metrics=lambda _: (1, 1, 10))
        with patch.object(approach, "MotorConnection", return_value=motor), \
                patch.object(approach, "PixelDetector", return_value=detector), \
                patch.object(approach, "GraspTask", return_value=task) as grasp, \
                patch.object(approach.time, "time", side_effect=clock.time), \
                patch.object(approach.time, "monotonic", side_effect=clock.monotonic):
            if wrong_pose:
                with self.assertRaises(ValueError):
                    approach.run_pixel_stage("127.0.0.1", profile, cv2, app)
                result = False
            else:
                result = approach.run_pixel_stage("127.0.0.1", profile, cv2, app)
            called = grasp.call_count
        motor.close.assert_called_once()
        return result, motor, called

    def test_runtime_only_grasps_after_last_stage_and_travel_restore(self):
        result, motor, called = self.run_stage()
        self.assertTrue(result)
        self.assertEqual([c.args[0] for c in motor.observe.call_args_list], [130, 128, 140])
        self.assertEqual(called, 1)

    def test_escape_never_changes_to_next_pose_or_grasps(self):
        result, motor, called = self.run_stage(cv_key=27)
        self.assertFalse(result)
        self.assertEqual(motor.observe.call_count, 1)
        self.assertEqual(called, 0)

    def test_wrong_pose_feedback_never_grasps(self):
        result, motor, called = self.run_stage(wrong_pose=True)
        self.assertFalse(result)
        self.assertEqual(called, 0)

    def test_bottom_mode_uses_named_views_and_grasps_only_after_final_goal(self):
        result, motor, called = self.run_stage(bottom=True)
        self.assertTrue(result)
        self.assertEqual([c.args[0] for c in motor.view.call_args_list], ["travel", "down20", "travel"])
        motor.observe.assert_not_called()
        self.assertEqual(called, 1)

    def test_bottom_mode_escape_never_tilts_twenty_or_grasps(self):
        result, motor, called = self.run_stage(bottom=True, cv_key=27)
        self.assertFalse(result)
        self.assertEqual([c.args[0] for c in motor.view.call_args_list], ["travel"])
        self.assertEqual(called, 0)

    def test_bottom_mode_wrong_readback_never_grasps(self):
        result, motor, called = self.run_stage(bottom=True, wrong_pose=True)
        self.assertFalse(result)
        self.assertEqual(called, 0)


if __name__ == "__main__":
    unittest.main()
