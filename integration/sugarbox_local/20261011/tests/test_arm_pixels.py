import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integration"))
from sugarbox_arm_pixels import PixelProfile, PixelServo, should_handoff


def profile(**overrides):
    fields = dict(target_u=320, target_v=350, yaw_sign=1,
                  forward_pixel_sign=1, observation_s2_deg=140, confirmed=True)
    fields.update(overrides)
    return PixelProfile(**fields)


class ArmPixelsTests(unittest.TestCase):
    def test_unconfirmed_profile_cannot_move(self):
        with self.assertRaises(ValueError):
            PixelServo(profile(confirmed=False))

    def test_unknown_or_out_of_bounds_calibration_is_rejected(self):
        for fields in (dict(target_u=float("nan")), dict(yaw_sign=0),
                       dict(forward_pixel_sign=0), dict(target_v=500),
                       dict(observation_s2_deg=119), dict(observation_s2_deg=143),
                       dict(max_vx=0.1), dict(front_stop_m=.05)):
            with self.assertRaises(ValueError):
                profile(**fields)

    def test_three_distinct_frames_required_before_motion(self):
        servo = PixelServo(profile())
        box = [280, 100, 360, 200]
        self.assertEqual(servo.step(box, 100, 100, .9)[:2], (0, 0))
        self.assertEqual(servo.step(box, 100, 100.1, .9)[:2], (0, 0))
        self.assertEqual(servo.step(box, 100.1, 100.1, .9)[:2], (0, 0))
        self.assertGreater(servo.step(box, 100.2, 100.2, .9)[0], 0)

    def test_lost_target_immediately_stops_and_requires_reconfirmation(self):
        servo = PixelServo(profile())
        for stamp in (100, 100.1, 100.2):
            servo.step([280, 100, 360, 200], stamp, stamp, .9)
        self.assertEqual(servo.step(None, 100.3, 100.3, .9)[:2], (0, 0))
        self.assertEqual(servo.step([280, 100, 360, 200], 100.4, 100.4, .9)[:2], (0, 0))

    def test_stale_future_low_confidence_or_clipped_detection_stops(self):
        for box, stamp, now, conf in (([280, 100, 360, 200], 100, 101, .9),
                                     ([280, 100, 360, 200], 101, 100, .9),
                                     ([280, 100, 360, 200], 100, 100, .2),
                                     ([280, 100, 360, 200], 100, 100, math.nan),
                                     ([0, 100, 360, 200], 100, 100, .9)):
            self.assertEqual(PixelServo(profile()).step(box, stamp, now, conf)[:2], (0, 0))

    def test_ready_requires_fresh_frames_and_stays_latched(self):
        servo = PixelServo(profile())
        for i in range(4):
            servo.step([280, 250, 360, 350], 100 + .1*i, 100 + .1*i, .9)
            self.assertFalse(servo.ready)
        servo.step([280, 250, 360, 350], 100.4, 100.4, .9)
        self.assertTrue(servo.ready)
        self.assertEqual(servo.step(None, 100.5, 100.5)[:2], (0, 0))
        servo.reset()
        self.assertFalse(servo.ready)

    def test_off_center_at_target_range_uses_arc_not_stationary_rotation(self):
        servo = PixelServo(profile())
        for stamp in (100, 100.1, 100.2):
            result = servo.step([400, 250, 480, 350], stamp, stamp, .9)
        self.assertNotEqual(result[0], 0)
        self.assertNotEqual(result[1], 0)
        self.assertLessEqual(abs(result[0]), .04)
        self.assertLessEqual(abs(result[1]), .25)

    def test_measured_pixel_signs_are_applied(self):
        servo = PixelServo(profile(yaw_sign=-1, forward_pixel_sign=-1))
        for stamp in (100, 100.1, 100.2):
            result = servo.step([400, 100, 480, 200], stamp, stamp, .9)
        self.assertLess(result[0], 0)
        self.assertGreater(result[1], 0)

    def test_jump_stops_without_blending_wrong_target(self):
        servo = PixelServo(profile())
        for stamp in (100, 100.1, 100.2):
            servo.step([280, 100, 360, 200], stamp, stamp, .9)
        self.assertEqual(servo.step([480, 100, 560, 200], 100.3, 100.3, .9)[2], "ARM_DETECTION_JUMP")

    def test_handoff_requires_recent_rear_loss_near_goal_and_clearance(self):
        self.assertTrue(should_handoff(100.7, 100, .8, True, True))
        for args in ((100.1, 100, .8, True, True), (100.7, 100, 1.2, True, True),
                     (100.7, 100, .8, False, True), (100.7, 0, .8, True, True),
                     (100.7, 100, .8, True, False)):
            self.assertFalse(should_handoff(*args))
