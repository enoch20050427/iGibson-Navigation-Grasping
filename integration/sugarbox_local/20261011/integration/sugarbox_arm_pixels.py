"""Pure, pose-specific pixel servoing; no ROS, serial or camera side effects."""
from dataclasses import dataclass, replace
import math


NAV_POSE = (90.0, 140.0, 0.0, 0.0, 90.0, 30.0)
# Jetson Rosmaster/rosmaster/rosmaster_main.py cmd 43, pose 1 (OEM anti-collision).
# Identification is not physical path validation for this robot's attachments.
FORWARD_VIEW_POSE = (90.0, 180.0, 0.0, 0.0, 90.0, 30.0)
DOWN20_POSE = (90.0, 120.0, 0.0, 0.0, 90.0, 30.0)
E1_POSE = (90.0, 74.2, 8.6, 8.6, 90.0, 30.0)
HANDOFF_EXIT_CODE = 42


def should_handoff(now, last_rear_usable, distance, clearance_ok, has_memory,
                   rear_near_edge=False):
    return (has_memory and clearance_ok and last_rear_usable > 0
            and (rear_near_edge or now - last_rear_usable >= 0.6)
            and 0 < distance <= 0.85)


def rear_at_handoff_edge(result, stamp, now, boundary):
    if (boundary is None or result is None or not math.isfinite(stamp)
            or not 0 <= now-stamp <= .35):
        return False
    detection = result.get("det") or {}
    box = detection.get("bbox")
    confidence = detection.get("conf", 0)
    if (box is None or len(box) != 4 or not all(math.isfinite(v) for v in box)
            or not math.isfinite(confidence) or confidence < .45):
        return False
    x1, y1, x2, y2 = box
    return 0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 480 and y2 >= boundary


@dataclass(frozen=True)
class PixelProfile:
    target_u: float
    target_v: float
    yaw_sign: int
    forward_pixel_sign: int
    observation_s2_deg: float
    confirmed: bool = False
    width: int = 640
    height: int = 480
    tolerance_u: float = 18.0
    tolerance_v: float = 8.0
    max_vx: float = 0.04
    max_wz: float = 0.25
    front_stop_m: float = 0.18

    def __post_init__(self):
        values = (self.target_u, self.target_v, self.observation_s2_deg,
                  self.tolerance_u, self.tolerance_v, self.max_vx, self.max_wz, self.front_stop_m)
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values):
            raise ValueError("Pixel profile requires finite calibration values")
        if (type(self.yaw_sign) is not int or type(self.forward_pixel_sign) is not int
                or self.yaw_sign not in (-1, 1) or self.forward_pixel_sign not in (-1, 1)):
            raise ValueError("Image-control signs must be measured as -1 or +1")
        if (self.width, self.height) != (640, 480):
            raise ValueError("Pixel profile must match the 640x480 stream")
        if not (0 < self.target_u < self.width and 0 < self.target_v < self.height):
            raise ValueError("Target point must be inside the frame")
        if not 120 <= self.observation_s2_deg <= 140:
            raise ValueError("Observation S2 must stay in the checked downward-only 120..140 range")
        if not (0 < self.tolerance_u <= 30 and 0 < self.tolerance_v <= 20
                and 0.02 <= self.max_vx <= 0.04 and 0 < self.max_wz <= 0.25
                and 0.10 <= self.front_stop_m <= 0.18):
            raise ValueError("Invalid visual approach tolerances/speed limits")

    @classmethod
    def from_dict(cls, data):
        if data.get("schema") != "sugarbox-arm-pixels-v1":
            raise ValueError("Unsupported pixel profile schema")
        fields = dict(data)
        fields.pop("schema")
        return cls(**fields)


@dataclass(frozen=True)
class ApproachProfile:
    stages: tuple
    rear_handoff_v: float = None
    bottom_switch_v: float = None
    forward_view_confirmed: bool = False

    def __post_init__(self):
        if (not isinstance(self.stages, tuple) or not 1 <= len(self.stages) <= 6
                or not all(isinstance(p, PixelProfile) for p in self.stages)):
            raise ValueError("Approach requires one to six pose-specific pixel profiles")
        if self.rear_handoff_v is not None and (
                not isinstance(self.rear_handoff_v, (int, float))
                or not math.isfinite(self.rear_handoff_v) or not 240 <= self.rear_handoff_v <= 470):
            raise ValueError("Rear handoff edge must be calibrated in the lower half of the frame")
        if self.bottom_switch_v is not None:
            if (isinstance(self.bottom_switch_v, bool)
                    or not isinstance(self.bottom_switch_v, (int, float))
                    or not math.isfinite(self.bottom_switch_v) or not 360 <= self.bottom_switch_v <= 450
                    or tuple(p.observation_s2_deg for p in self.stages) != (140, 120)
                    or self.rear_handoff_v is None or type(self.forward_view_confirmed) is not bool):
                raise ValueError("Bottom switch requires travel 140 -> down20 120 and a lower-frame edge")
        elif len(self.stages) > 1:
            if self.stages[0].observation_s2_deg != 130 or self.rear_handoff_v is None:
                raise ValueError("Staged approach starts at S2=130 with a calibrated rear handoff edge")
            for before, after in zip(self.stages, self.stages[1:]):
                delta = before.observation_s2_deg - after.observation_s2_deg
                if not 0 < delta <= 2:
                    raise ValueError("Approach stages must look downward in at most two-degree increments")

    @classmethod
    def from_dict(cls, data):
        if data.get("schema") == "sugarbox-arm-pixels-v1":
            return cls((PixelProfile.from_dict(data),))
        if data.get("schema") == "sugarbox-arm-bottom-switch-v3":
            keys = {"schema", "rear_handoff_v", "arm_switch_v", "travel_view_confirmed",
                    "forward_view_confirmed", "final"}
            if set(data) != keys or type(data["travel_view_confirmed"]) is not bool:
                raise ValueError("Invalid two-view profile fields")
            final = PixelProfile.from_dict(data["final"])
            edge = data["arm_switch_v"]
            if isinstance(edge, bool) or not isinstance(edge, (int, float)) or not math.isfinite(edge):
                raise ValueError("Invalid bottom switch edge")
            # This is a driving setpoint beyond the switch line, not a grasp goal.
            travel = replace(final, observation_s2_deg=140, target_u=320, target_v=edge+24,
                             confirmed=data["travel_view_confirmed"])
            return cls((travel, final), data["rear_handoff_v"], edge, data["forward_view_confirmed"])
        if data.get("schema") != "sugarbox-arm-staged-pixels-v2":
            raise ValueError("Unsupported approach profile schema")
        if set(data) != {"schema", "rear_handoff_v", "stages"}:
            raise ValueError("Unexpected or missing staged-profile fields")
        if not isinstance(data["stages"], list):
            raise ValueError("Staged profiles must be a JSON array")
        return cls(tuple(PixelProfile.from_dict(stage) for stage in data["stages"]), data["rear_handoff_v"])


class BottomEdgeLatch:
    """Three distinct, stable edge detections; bottom clipping may trigger tilt."""
    def __init__(self, boundary):
        self.boundary = boundary
        self.reset()

    def reset(self):
        self.previous = None
        self.stamp = 0.0
        self.count = 0

    def step(self, bbox, stamp, now, confidence):
        if (bbox is None or len(bbox) != 4 or not all(math.isfinite(v) for v in bbox)
                or not math.isfinite(stamp) or stamp <= 0 or not 0 <= now-stamp <= .35
                or not math.isfinite(confidence) or confidence < .45):
            self.reset()
            return False
        x1, y1, x2, y2 = bbox
        if not (2 < x1 < x2 < 638 and 0 <= y1 < y2 <= 480):
            self.reset()
            return False
        if stamp <= self.stamp:
            return False
        point = ((x1+x2)/2, y2, x2-x1, y2-y1)
        if self.previous is not None:
            old = self.previous
            if (math.hypot(point[0]-old[0], point[1]-old[1]) > 75
                    or not .5 <= point[2]/old[2] <= 2 or not .5 <= point[3]/old[3] <= 2):
                self.count = 0
        self.previous, self.stamp = point, stamp
        self.count = self.count+1 if y2 >= self.boundary else 0
        return self.count >= 3


class StagedPixelServo:
    """Only the final pose can authorize grasp; each pose owns its own pixel goal."""
    def __init__(self, profile):
        if not all(p.confirmed is True for p in profile.stages):
            raise ValueError("Every pose-specific pixel calibration must be physically confirmed")
        self.profiles = profile.stages
        self.index = 0
        self.servo = PixelServo(self.profiles[0])
        self.pending_pose = None
        self.not_before = 0.0
        self.ready = False
        self.bottom_edge = BottomEdgeLatch(profile.bottom_switch_v) if profile.bottom_switch_v is not None else None

    @property
    def profile(self):
        return self.profiles[self.index]

    def hold(self, reason):
        if self.bottom_edge is not None:
            self.bottom_edge.reset()
        return self.servo.hold(reason)

    def step(self, bbox, stamp, now, confidence=0):
        if self.pending_pose is not None:
            return 0.0, 0.0, "ARM_POSE_SWITCH"
        if stamp <= self.not_before:
            return self.hold("ARM_POSE_SETTLING")
        if self.bottom_edge is not None and self.index == 0:
            if self.bottom_edge.step(bbox, stamp, now, confidence):
                self.pending_pose = self.profiles[1].observation_s2_deg
                return 0.0, 0.0, "ARM_POSE_SWITCH"
            return self.servo.step(bbox, stamp, now, confidence)
        output = self.servo.step(bbox, stamp, now, confidence)
        if self.servo.ready:
            if self.index + 1 < len(self.profiles):
                self.pending_pose = self.profiles[self.index+1].observation_s2_deg
                return 0.0, 0.0, "ARM_POSE_SWITCH"
            self.ready = True
        return output

    def confirm_pose(self, actual_s2, finished_at):
        if (self.pending_pose is None or not math.isfinite(actual_s2)
                or abs(actual_s2-self.pending_pose) > 1 or not math.isfinite(finished_at)):
            raise ValueError("Unexpected/unreached observation pose; no pixel motion or grasp")
        self.index += 1
        self.servo = PixelServo(self.profiles[self.index])
        self.pending_pose = None
        self.ready = False
        # Ignore in-flight frames after the physical servo/readback settle period.
        self.not_before = finished_at + .35


class PixelServo:
    """Forward/reverse arcs only, never rotation in place or blind translation.

    Each result is tied to a capture timestamp. A target must be consistent on
    three distinct frames before motion; ready is latched until explicit reset.
    """

    def __init__(self, profile):
        if profile.confirmed is not True:
            raise ValueError("Physical pose, pixel target and signs are not confirmed")
        self.profile = profile
        self.reset()

    def reset(self):
        self.previous = None
        self.last_stamp = 0.0
        self.consistent = 0
        self.ready_count = 0
        self.ready = False
        self.last_output = (0.0, 0.0, "ARM_WAIT")

    def hold(self, reason):
        self.previous = None
        self.consistent = 0
        self.ready_count = 0
        self.last_output = (0.0, 0.0, reason)
        return self.last_output

    def step(self, bbox, stamp, now, confidence=0.0):
        if self.ready:
            return 0.0, 0.0, "ARM_READY_LATCHED"
        if (bbox is None or not math.isfinite(stamp) or stamp <= 0
                or not 0 <= now - stamp <= 0.35
                or not math.isfinite(confidence) or confidence < 0.45):
            return self.hold("ARM_TARGET_MISSING_OR_STALE")
        if len(bbox) != 4 or not all(math.isfinite(v) for v in bbox):
            return self.hold("ARM_INVALID_BOX")
        x1, y1, x2, y2 = bbox
        p = self.profile
        if not (2 < x1 < x2 < p.width - 2 and 0 <= y1 < y2 < p.height - 2):
            return self.hold("ARM_CLIPPED_BOX")
        if stamp <= self.last_stamp:
            return self.last_output
        self.last_stamp = stamp
        point = ((x1 + x2) / 2, y2, x2 - x1, y2 - y1)
        if self.previous is not None:
            jump = math.hypot(point[0] - self.previous[0], point[1] - self.previous[1])
            if (jump > 75 or not 0.5 <= point[2] / self.previous[2] <= 2
                    or not 0.5 <= point[3] / self.previous[3] <= 2):
                self.last_output = self.hold("ARM_DETECTION_JUMP")
                self.previous = point
                return self.last_output
        self.previous = point
        self.consistent += 1
        if self.consistent < 3:
            self.last_output = (0.0, 0.0, "ARM_CONFIRMING")
            return self.last_output
        error_u = p.target_u - point[0]
        error_v = p.target_v - point[1]
        centered = abs(error_u) <= p.tolerance_u
        in_range = abs(error_v) <= p.tolerance_v
        if centered and in_range:
            self.ready_count += 1
            self.ready = self.ready_count >= 3
            self.last_output = (0.0, 0.0, "ARM_READY" if self.ready else "ARM_VERIFY_READY")
            return self.last_output
        self.ready_count = 0
        vx = max(-p.max_vx, min(p.max_vx, p.forward_pixel_sign * error_v * 0.001))
        wz = 0.0 if centered else max(-p.max_wz, min(p.max_wz, p.yaw_sign * error_u * 0.003))
        if in_range and not centered:
            # Make space for a visible corrective arc rather than pivoting in place.
            vx = -0.02
        elif abs(vx) < 0.02:
            vx = math.copysign(0.02, vx)
        self.last_output = (vx, wz, "ARM_PIXEL_APPROACH")
        return self.last_output
