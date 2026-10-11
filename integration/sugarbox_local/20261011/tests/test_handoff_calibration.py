import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "diagnostics"))
from handoff_calibration import HandoffDraft


class HandoffCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.template = json.loads((ROOT / "diagnostics/arm_staged_profile.template.json").read_text())
        self.draft = HandoffDraft(self.template)
        self.pose = [90, 130, 0, 0, 90, 30]

    def mark(self, camera="rear", **changes):
        fields = dict(camera=camera, x=320, y=360, stamp=100, now=100.1,
                      shape=(480, 640, 3), angles=self.pose)
        fields.update(changes)
        self.draft.mark(**fields)

    def test_export_requires_both_views_and_actual_pose(self):
        self.mark()
        with self.assertRaises(ValueError):
            self.draft.export(self.pose)
        self.mark("arm")
        with self.assertRaises(ValueError):
            self.draft.export([90, 140, 0, 0, 90, 30])

    def test_reference_never_becomes_motion_or_grasp_goal(self):
        self.mark()
        self.mark("arm", y=100)
        profile, evidence = self.draft.export(self.pose)
        self.assertEqual(profile["rear_handoff_v"], 360)
        self.assertEqual(profile["stages"], self.template["stages"])
        self.assertTrue(all(stage["confirmed"] is False for stage in profile["stages"]))
        self.assertEqual(evidence["marks"]["arm"]["v"], 100)
        self.assertIn("not_motion_goal", evidence["arm_mark_role"])
        self.assertIsNone(self.template["rear_handoff_v"])

    def test_stale_future_wrong_size_and_outside_clicks_refused(self):
        for changes in (dict(now=101), dict(stamp=101), dict(shape=(720, 1280, 3)),
                        dict(x=640), dict(y=239), dict(y=471)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.mark(**changes)

    def test_missing_nonfinite_or_wrong_feedback_refused(self):
        for pose in ([], [90, 129, 0, 0, 90, 60], [90, math.nan, 0, 0, 90, 30],
                     [90, True, 0, 0, 90, 30], [90, 128, 0, 0, 90, 30]):
            with self.subTest(pose=pose), self.assertRaises(ValueError):
                self.mark(angles=pose)

    def test_confirmed_profile_cannot_be_reused_as_draft(self):
        self.template["stages"][0]["confirmed"] = True
        with self.assertRaises(ValueError):
            HandoffDraft(self.template)


if __name__ == "__main__":
    unittest.main()
