#!/usr/bin/env python3
"""TCP motor bridge for ``sugarbox_rl_approach_final2.py``.

This process is the sole owner of ``/dev/myserial`` while the standalone
Sugarbox approach controller is running.  It accepts newline-delimited JSON
velocity commands on TCP port 7000 and publishes the repository's calibrated
wheel-feedback odometry on ``/odom_setmotor``.

The old ``~/ai_motor_server_RL.py`` duplicated odometry calibration constants
and inverted yaw feedback.  This replacement deliberately delegates every
feedback conversion and integration step to ``feedback_odom.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

try:
    from .sugarbox_arm_feedback import read_angles
except ImportError:
    from sugarbox_arm_feedback import read_angles

try:
    from .feedback_odom import FeedbackOdomReader
except ImportError:
    from feedback_odom import FeedbackOdomReader


HOST = "0.0.0.0"
PORT = 7000
WATCHDOG_TIMEOUT_S = 0.50
ODOM_RATE_HZ = 20.0

ABSOLUTE_MAX_MOTOR = 60
RL_MOTOR_LIMIT = 30
MIN_ACTIVE_MOTOR = 20.0
CMD_MAX_FORWARD_VX = 0.15
CMD_MAX_REVERSE_VX = 0.08
CMD_MAX_ABS_WZ = 1.00
CMD_VX_DEADBAND = 0.015
CMD_WZ_DEADBAND = 0.03
LINEAR_MOTOR_PER_MPS = 200.0
ANGULAR_MOTOR_PER_RAD_S = float(os.environ.get("G2_ANGULAR_MOTOR_PER_RAD_S", "25.0"))


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def velocity_to_motor_values(vx: float, wz: float) -> Tuple[int, int, int, int]:
    """Preserve the wheel mapping verified in the previous RL server."""
    vx = float(clamp(vx, -CMD_MAX_REVERSE_VX, CMD_MAX_FORWARD_VX))
    wz = float(clamp(wz, -CMD_MAX_ABS_WZ, CMD_MAX_ABS_WZ))
    if abs(vx) < CMD_VX_DEADBAND:
        vx = 0.0
    if abs(wz) < CMD_WZ_DEADBAND:
        wz = 0.0

    left = vx * LINEAR_MOTOR_PER_MPS - wz * ANGULAR_MOTOR_PER_RAD_S
    right = vx * LINEAR_MOTOR_PER_MPS + wz * ANGULAR_MOTOR_PER_RAD_S
    peak = max(abs(left), abs(right))
    # Keep small commands small; do not override the client's clearance limit.
    if peak > RL_MOTOR_LIMIT:
        scale = RL_MOTOR_LIMIT / peak
        left *= scale
        right *= scale
    return tuple(int(round(v)) for v in (left, left, right, right))


class SugarboxRLMotorServer:
    def __init__(self, device, rospy, odom_type, transform_type, tf_type,
                 *, host: str = HOST, port: int = PORT, allow_arm_observation=False,
                 allow_arm_initialization=False, allow_arm_forward_view=False, allow_oem_validation=False):
        self.device = device
        self.rospy = rospy
        self.Odometry = odom_type
        self.TransformStamped = transform_type
        self.TFMessage = tf_type
        self.host = host
        self.port = int(port)
        self.allow_arm_observation = bool(allow_arm_observation)
        self.allow_arm_initialization = bool(allow_arm_initialization)
        self.allow_arm_forward_view = bool(allow_arm_forward_view)
        self.allow_oem_validation = bool(allow_oem_validation)
        self.serial_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.reader = FeedbackOdomReader(device, serial_lock=self.serial_lock)
        self.last_command = 0.0
        self.last_motor_values: Optional[Tuple[int, int, int, int]] = None
        self.shutting_down = threading.Event()
        self.server_socket: Optional[socket.socket] = None
        self.odom_pub = rospy.Publisher("/odom_setmotor", odom_type, queue_size=20)
        self.tf_pub = rospy.Publisher("/tf", tf_type, queue_size=20)

    def set_motors(self, values: Tuple[int, int, int, int]) -> None:
        values = tuple(int(clamp(v, -ABSOLUTE_MAX_MOTOR, ABSOLUTE_MAX_MOTOR))
                       for v in values)
        if values == self.last_motor_values:
            return
        with self.serial_lock:
            self.device.set_motor(*values)
        self.last_motor_values = values
        print("[MOTOR] m1={} m2={} m3={} m4={}".format(*values))

    def stop(self) -> None:
        self.set_motors((0, 0, 0, 0))

    def mark_command(self) -> None:
        with self.state_lock:
            self.last_command = time.monotonic()

    def command_age(self) -> float:
        with self.state_lock:
            stamp = self.last_command
        return float("inf") if stamp <= 0.0 else time.monotonic() - stamp

    def process_command(self, command: object) -> Optional[dict]:
        if not isinstance(command, dict):
            raise ValueError("command must be a JSON object")
        action = str(command.get("action", "stop")).strip().lower()
        if action in ("arm_status", "arm_observe", "arm_initialize", "arm_view", "arm_oem_step"):
            self.stop()
            reply = self.arm_request(dict(command, action=action))
            reply["request_id"] = command.get("request_id")
            return reply
        if action == "stop":
            self.mark_command()
            self.stop()
            return
        if action != "velocity":
            raise ValueError("unsupported action: {}".format(action))
        vx = float(command.get("vx", 0.0))
        wz = float(command.get("wz", 0.0))
        if not math.isfinite(vx) or not math.isfinite(wz):
            raise ValueError("vx and wz must be finite")
        self.mark_command()
        values = velocity_to_motor_values(vx, wz)
        self.set_motors(values)

    def read_arm_angles(self):
        with self.serial_lock:
            return read_angles(self.device)

    def arm_request(self, command):
        """Small supervised travel-S2 adjustment, through this sole serial owner.

        No jaw-close command, arbitrary pose or grasp policy is accepted here. Production
        grasp still belongs to the existing guarded v23 service after ownership
        transfer. The movement endpoint is disabled unless explicitly opted in.
        """
        try:
            if command["action"] == "arm_initialize":
                return self.initialize_arm(command)
            if command["action"] == "arm_view":
                return self.named_arm_view(command)
            if command["action"] == "arm_oem_step":
                return self.oem_validation_step(command)
            angles = self.read_arm_angles()
            if command["action"] == "arm_status":
                return {"ok": True, "protocol": "sugarbox-arm-observe-v1",
                        "servo_deg": angles, "allow_arm_observation": self.allow_arm_observation,
                        "oem_validation_enabled": getattr(self, "allow_oem_validation", False),
                        "named_views": (["travel", "down20", "forward"] if getattr(self, "allow_arm_forward_view", False)
                                        else ["travel", "down20"]) if self.allow_arm_observation else [],
                        "allow_arm_initialization": getattr(self, "allow_arm_initialization", False)}
            if not self.allow_arm_observation or command.get("confirmed") is not True:
                raise ValueError("arm_observation_not_authorized")
            target = command.get("s2_deg")
            if not isinstance(target, (int, float)) or not math.isfinite(target) or not 120 <= target <= 140:
                raise ValueError("S2 observation target must be in the model-checked downward-only 120..140 range")
            expected = command.get("expected_deg")
            if (not isinstance(expected, list) or len(expected) != 6
                    or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in expected)
                    or any(abs(a - b) > 2 for a, b in zip(angles, expected))):
                raise ValueError("arm_pose_changed")
            travel = (90, 140, 0, 0, 90, 30)
            if (any(abs(angles[i] - travel[i]) > 3 for i in (0, 2, 3, 4, 5))
                    or not 120 <= angles[1] <= 140):
                raise ValueError("arm_not_in_empty_travel_pose")
            time.sleep(0.6)
            with self.serial_lock:
                motion = self.device.get_motion_data()
            if (len(motion) != 3 or not all(math.isfinite(v) and abs(v) <= 0.01 for v in motion)):
                raise ValueError("chassis_not_stationary")
            again = self.read_arm_angles()
            if any(abs(a - b) > 1 for a, b in zip(angles, again)):
                raise ValueError("arm_not_stable")
            if abs(target - again[1]) > 3:
                raise ValueError("observation_step_exceeds_three_degrees")
            if abs(target - again[1]) > 0.5:
                with self.serial_lock:
                    self.device.set_uart_servo_angle(2, target, run_time=1000)
                time.sleep(1.5)
            reached = self.read_arm_angles()
            if abs(reached[1] - target) > 1 or any(abs(reached[i] - again[i]) > 1 for i in (0, 2, 3, 4, 5)):
                raise ValueError("observation_pose_not_reached")
            return {"ok": True, "servo_deg": reached}
        except Exception as exc:
            self.stop()
            return {"ok": False, "reason": str(exc)}

    def named_arm_view(self, command):
        if not self.allow_arm_observation or command.get("confirmed") is not True:
            raise ValueError("arm_observation_not_authorized")
        poses = {"travel": [90, 140, 0, 0, 90, 30], "down20": [90, 120, 0, 0, 90, 30]}
        if getattr(self, "allow_arm_forward_view", False):
            poses["forward"] = [90, 180, 0, 0, 90, 30]
        view = command.get("view")
        if view not in poses:
            raise ValueError("named_view_not_enabled")
        angles = self.read_arm_angles()
        expected = command.get("expected_deg")
        if (not isinstance(expected, list) or len(expected) != 6
                or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in expected)
                or any(abs(a-b) > 2 for a, b in zip(angles, expected))):
            raise ValueError("arm_pose_changed")
        if not any(all(abs(a-b) <= 3 for a, b in zip(angles, pose)) for pose in poses.values()):
            raise ValueError("arm_not_in_known_empty_view_pose")
        if (view == "forward" or abs(angles[1]-180) <= 3) and command.get("oem_path_checked") is not True:
            raise ValueError("OEM_forward_path_requires_physical_validation")
        time.sleep(.6)
        with self.serial_lock:
            motion = self.device.get_motion_data()
        if len(motion) != 3 or not all(math.isfinite(v) and abs(v) <= .01 for v in motion):
            raise ValueError("chassis_not_stationary")
        again = self.read_arm_angles()
        if any(abs(a-b) > 1 for a, b in zip(angles, again)):
            raise ValueError("arm_not_stable")
        target = poses[view]
        if any(abs(a-b) > .5 for a, b in zip(again, target)):
            # All named views change S2 only; never rewrite the other five joints.
            # The OEM forward route additionally requires physical validation.
            with self.serial_lock:
                self.device.set_uart_servo_angle(2, target[1], run_time=2000)
            time.sleep(2.5)
        reached = self.read_arm_angles()
        if any(abs(a-b) > 1 for a, b in zip(reached, target)):
            raise ValueError("named_view_not_reached")
        return {"ok": True, "servo_deg": reached, "view": view}

    def oem_validation_step(self, command):
        """Explicit supervised path exploration, not an automatically approved view."""
        if not getattr(self, "allow_oem_validation", False) or command.get("supervised") is not True:
            raise ValueError("OEM_validation_not_enabled")
        target = command.get("s2_deg")
        if (isinstance(target, bool) or not isinstance(target, (int, float))
                or not math.isfinite(target) or not 140 <= target <= 180):
            raise ValueError("invalid_OEM_validation_target")
        expected = command.get("expected_deg")
        angles = self.read_arm_angles()
        if (not isinstance(expected, list) or len(expected) != 6
                or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in expected)
                or any(abs(a-b) > 1 for a, b in zip(angles, expected))):
            raise ValueError("arm_pose_changed")
        travel = (90, 140, 0, 0, 90, 30)
        if (any(abs(angles[i]-travel[i]) > 1 for i in (0, 2, 3, 4, 5))
                or not 140 <= angles[1] <= 180):
            raise ValueError("not_empty_S2_only_validation_pose")
        time.sleep(.2)
        with self.serial_lock:
            motion = self.device.get_motion_data()
        if len(motion) != 3 or not all(math.isfinite(v) and abs(v) <= .01 for v in motion):
            raise ValueError("chassis_not_stationary")
        again = self.read_arm_angles()
        if any(abs(a-b) > 1 for a, b in zip(angles, again)) or abs(target-again[1]) > 3:
            raise ValueError("OEM_step_unstable_or_too_large")
        if abs(target-again[1]) > .5:
            with self.serial_lock:
                self.device.set_uart_servo_angle(2, target, run_time=500)
            time.sleep(.8)
        reached = self.read_arm_angles()
        if abs(reached[1]-target) > 1 or any(abs(reached[i]-again[i]) > 1 for i in (0, 2, 3, 4, 5)):
            raise ValueError("OEM_validation_step_not_reached")
        return {"ok": True, "servo_deg": reached, "path_validated": False}

    def initialize_arm(self, command):
        """Supervised named-pose bootstrap, not a verified path from arbitrary sag.

        A powered-off arm may sit outside the normal command-angle range. Only
        this separate, explicit operator-approved bootstrap can command travel
        before valid joint feedback is available. No chassis motion is accepted
        during the serialized request, and failed readback never permits PPO.
        """
        if (not getattr(self, "allow_arm_initialization", False)
                or any(command.get(key) is not True for key in
                       ("confirmed", "empty_gripper", "workspace_clear"))):
            raise ValueError("supervised_empty_arm_initialization_not_authorized")
        time.sleep(.6)
        with self.serial_lock:
            motion = self.device.get_motion_data()
        if len(motion) != 3 or not all(math.isfinite(v) and abs(v) <= .01 for v in motion):
            raise ValueError("chassis_not_stationary")
        travel = [90, 140, 0, 0, 90, 30]
        # Same named nav initialization and 2-second duration as grasp/move_arm.py.
        with self.serial_lock:
            self.device.set_uart_servo_angle_array(angle_s=travel, run_time=2000)
        time.sleep(2.5)
        reached = self.read_arm_angles()
        if any(abs(a-b) > 3 for a, b in zip(reached, travel)):
            raise ValueError("travel_initialization_not_reached: " + repr(reached))
        return {"ok": True, "servo_deg": reached, "initialized": True}

    def publish_odom(self, state) -> None:
        stamp = self.rospy.Time.now()
        half_yaw = state.yaw * 0.5
        qz, qw = math.sin(half_yaw), math.cos(half_yaw)

        msg = self.Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = "odom"
        msg.child_frame_id = "base_footprint"
        msg.pose.pose.position.x = state.x
        msg.pose.pose.position.y = state.y
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        msg.pose.covariance = self.reader.odom.covariance()
        msg.twist.twist.linear.x = state.vx
        msg.twist.twist.linear.y = state.vy
        msg.twist.twist.angular.z = state.wz
        msg.twist.covariance = self.reader.odom.covariance()
        self.odom_pub.publish(msg)

        transform = self.TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = "odom"
        transform.child_frame_id = "base_footprint"
        transform.transform.translation.x = state.x
        transform.transform.translation.y = state.y
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw
        tf_msg = self.TFMessage()
        tf_msg.transforms = [transform]
        self.tf_pub.publish(tf_msg)

    def odom_loop(self) -> None:
        rate = self.rospy.Rate(ODOM_RATE_HZ)
        while not self.rospy.is_shutdown() and not self.shutting_down.is_set():
            state = self.reader.poll()
            if state.valid:
                self.publish_odom(state)
            elif self.reader.odom.failure_count % 40 == 1:
                self.rospy.logwarn("feedback odom unavailable: {}".format(state.reason))
            rate.sleep()

    def watchdog_loop(self) -> None:
        while not self.rospy.is_shutdown() and not self.shutting_down.is_set():
            if self.command_age() > WATCHDOG_TIMEOUT_S:
                self.stop()
            time.sleep(0.05)

    def handle_client(self, connection: socket.socket, address) -> None:
        print("[INFO] client connected: {}".format(address))
        connection.settimeout(0.20)
        buffer = ""
        try:
            while not self.rospy.is_shutdown() and not self.shutting_down.is_set():
                try:
                    chunk = connection.recv(4096)
                except socket.timeout:
                    continue
                except (ConnectionError, OSError) as exc:
                    print("[WARN] client receive failed: {}".format(exc))
                    break
                if not chunk:
                    break
                buffer += chunk.decode("utf-8", errors="replace")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if not line.strip():
                        continue
                    try:
                        reply = self.process_command(json.loads(line))
                        if reply is not None:
                            connection.sendall((json.dumps(reply) + "\n").encode("utf-8"))
                    except Exception as exc:
                        print("[WARN] command error: {}".format(exc))
                        self.stop()
        finally:
            self.stop()
            connection.close()
            print("[INFO] client disconnected: {}".format(address))

    def serve(self) -> None:
        threading.Thread(target=self.odom_loop, daemon=True).start()
        threading.Thread(target=self.watchdog_loop, daemon=True).start()
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket = server
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(1)
        server.settimeout(0.5)
        print("[INFO] Sugarbox RL motor server listening on {}:{}".format(
            self.host, self.port))
        try:
            while not self.rospy.is_shutdown() and not self.shutting_down.is_set():
                try:
                    connection, address = server.accept()
                except socket.timeout:
                    continue
                connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.handle_client(connection, address)
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        if self.shutting_down.is_set():
            return
        self.shutting_down.set()
        try:
            self.stop()
        finally:
            if self.server_socket is not None:
                self.server_socket.close()
            try:
                self.device.set_auto_report_state(False, False)
            except Exception:
                pass
            serial = getattr(self.device, "ser", None)
            if serial is not None:
                try:
                    serial.close()
                except Exception:
                    pass


def load_rosmaster(serial_port: str):
    root = Path(__file__).resolve().parents[1]
    grasp_dir = root / "grasp"
    if str(grasp_dir) not in sys.path:
        sys.path.insert(0, str(grasp_dir))
    from Rosmaster_Lib import Rosmaster
    try:
        return Rosmaster(com=serial_port)
    except TypeError:
        return Rosmaster()


def run_selftest() -> None:
    assert velocity_to_motor_values(0.0, 0.0) == (0, 0, 0, 0)
    assert velocity_to_motor_values(0.15, 0.0) == (30, 30, 30, 30)
    assert velocity_to_motor_values(-0.08, 0.0) == (-16, -16, -16, -16)
    left = velocity_to_motor_values(0.0, 0.4)
    right = velocity_to_motor_values(0.0, -0.4)
    assert left[0] < 0 < left[2]
    assert right[0] > 0 > right[2]
    assert max(map(abs, velocity_to_motor_values(999.0, 999.0))) <= RL_MOTOR_LIMIT
    assert velocity_to_motor_values(0.06, 0.0) == (12, 12, 12, 12)
    assert velocity_to_motor_values(0.0, 0.0) == (0, 0, 0, 0)
    print("[selftest] Sugarbox RL motor mapping OK (no low-speed amplification)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--serial-port", default="/dev/myserial")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--allow-arm-observation", action="store_true",
                        help="allow supervised downward-only S2 poses in 120..140 degrees, max 3 degrees per step")
    parser.add_argument("--allow-arm-initialization", action="store_true",
                        help="allow supervised empty-arm named travel initialization with readback")
    parser.add_argument("--allow-arm-forward-view", action="store_true",
                        help="allow supervised named forward view; chassis must be stopped and gripper empty")
    parser.add_argument("--allow-oem-validation", action="store_true",
                        help="supervised empty-arm S2 path validation only, max 3 degrees per request")
    args = parser.parse_args()
    if args.selftest:
        run_selftest()
        return

    import rospy
    from geometry_msgs.msg import TransformStamped
    from nav_msgs.msg import Odometry
    from tf2_msgs.msg import TFMessage

    rospy.init_node("sugarbox_rl_motor_server", anonymous=False)
    device = load_rosmaster(args.serial_port)
    device.create_receive_threading()
    try:
        device.set_auto_report_state(True, False)
    except TypeError:
        device.set_auto_report_state(enable=True, forever=False)
    server = SugarboxRLMotorServer(
        device, rospy, Odometry, TransformStamped, TFMessage,
        host=args.host, port=args.port, allow_arm_observation=args.allow_arm_observation,
        allow_arm_initialization=args.allow_arm_initialization,
        allow_arm_forward_view=args.allow_arm_forward_view,
        allow_oem_validation=args.allow_oem_validation)
    rospy.on_shutdown(server.shutdown)
    server.stop()
    server.serve()


if __name__ == "__main__":
    main()
