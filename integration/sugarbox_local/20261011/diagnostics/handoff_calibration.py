"""Draft-only handoff marks; an arm reference is not a driving/grasp goal."""
import copy
import math


class HandoffDraft:
    def __init__(self, template):
        self.profile = copy.deepcopy(template)
        if self.profile.get("schema") != "sugarbox-arm-staged-pixels-v2":
            raise ValueError("Expected a staged pixel template")
        stages = self.profile.get("stages")
        if not isinstance(stages, list) or not stages or stages[0].get("observation_s2_deg") != 130:
            raise ValueError("Handoff view must be the checked S2=130 pose")
        if any(stage.get("confirmed") is not False for stage in stages):
            raise ValueError("Use an unconfirmed draft, not a motion-authorized profile")
        self.marks = {}

    @staticmethod
    def check_pose(angles):
        expected = (90, 130, 0, 0, 90, 30)
        if not isinstance(angles, list) or len(angles) != 6:
            raise ValueError("Missing six actual servo readings")
        for index, (actual, target) in enumerate(zip(angles, expected)):
            tolerance = 1 if index == 1 else 3
            if (isinstance(actual, bool) or not isinstance(actual, (int, float))
                    or not math.isfinite(actual) or abs(actual-target) > tolerance):
                raise ValueError("Actual arm pose does not match the handoff view")

    def mark(self, camera, x, y, stamp, now, shape, angles):
        if camera not in ("rear", "arm"):
            raise ValueError("Unknown camera")
        self.check_pose(angles)
        if tuple(shape[:2]) != (480, 640) or not 0 <= now-stamp <= .35:
            raise ValueError("Click requires a fresh 640x480 frame")
        if not (0 <= x < 640 and 0 <= y < 480):
            raise ValueError("Click is outside the frame")
        if camera == "rear" and not 240 <= y <= 470:
            raise ValueError("Rear handoff edge must be in the lower frame")
        self.marks[camera] = dict(u=x, v=y, frame_stamp=stamp, servo_deg=list(angles))

    def export(self, angles):
        self.check_pose(angles)
        if set(self.marks) != {"rear", "arm"}:
            raise ValueError("Both camera marks are required")
        profile = copy.deepcopy(self.profile)
        profile["rear_handoff_v"] = self.marks["rear"]["v"]
        # The arm mark describes overlap only. Stage-switch and final grasp
        # goals require separate physical positions and must remain untouched.
        evidence = dict(schema="sugarbox-handoff-evidence-v1", marks=copy.deepcopy(self.marks),
                        saved_servo_deg=list(angles), arm_mark_role="overlap_reference_not_motion_goal")
        return profile, evidence
