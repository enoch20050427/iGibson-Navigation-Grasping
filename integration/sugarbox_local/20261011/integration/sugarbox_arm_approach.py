"""Independent PPO -> arm pixel approach -> existing v23 grasp orchestration.

Default starts no hardware. Real motion requires a physically confirmed
pixel/pose profile and --real --i-am-beside-the-robot. No teammate code is changed.
"""
import argparse
import ipaddress
import json
import math
import os
from pathlib import Path
import socket
import shlex
import subprocess
import sys
import threading
import time

from sugarbox_arm_pixels import (E1_POSE, HANDOFF_EXIT_CODE, NAV_POSE,
                                FORWARD_VIEW_POSE, DOWN20_POSE, ApproachProfile, StagedPixelServo)

HERE = Path(__file__).resolve().parent
REMOTE_CTL = "/home/jetson/Documents/deploy_jetson2/grasp/v23/graspctl.py"


class ArmMotionCancelled(RuntimeError):
    pass


class MotorConnection:
    def __init__(self, host, port=7000):
        self.sock = socket.create_connection((host, port), timeout=12)
        self.pending = b""
        self.counter = 0

    def send(self, data):
        self.sock.sendall((json.dumps(data) + "\n").encode())

    def velocity(self, vx, wz):
        self.send({"action": "velocity", "vx": vx, "wz": wz})

    def stop(self):
        self.send({"action": "stop"})

    def request(self, action, **fields):
        self.stop()
        self.counter += 1
        request_id = "arm-" + str(self.counter)
        self.send(dict(fields, action=action, request_id=request_id))
        while b"\n" not in self.pending:
            part = self.sock.recv(4096)
            if not part:
                raise RuntimeError("Motor bridge disconnected during arm request")
            self.pending += part
            if len(self.pending) > 8192:
                raise RuntimeError("Oversized motor bridge response")
        line, self.pending = self.pending.split(b"\n", 1)
        value = json.loads(line)
        if value.get("request_id") != request_id or value.get("ok") is not True:
            raise RuntimeError("Arm request refused: " + str(value))
        return value

    def observe(self, target, cancel=None):
        def check_cancel():
            if cancel is not None and cancel():
                self.stop()
                raise ArmMotionCancelled("Observation movement cancelled; no grasp")
        if not isinstance(target, (int, float)) or not math.isfinite(target) or not 120 <= target <= 140:
            raise ValueError("Observation target must stay within 120..140 degrees")
        check_cancel()
        status = self.request("arm_status")
        if (status.get("protocol") != "sugarbox-arm-observe-v1"
                or status.get("allow_arm_observation") is not True):
            raise RuntimeError("Observation-capable motor bridge is not explicitly enabled")
        # Leave one degree of margin for integer readback settling before the
        # server's second pose read; its physical step limit remains 3 degrees.
        for _ in range(32):
            check_cancel()
            current = status["servo_deg"][1]
            if abs(target-current) <= 2:
                reached = self.request("arm_observe", expected_deg=status["servo_deg"],
                                       s2_deg=target, confirmed=True)
                check_cancel()
                return reached
            step = current + math.copysign(2, target-current)
            status = self.request("arm_observe", expected_deg=status["servo_deg"],
                                  s2_deg=step, confirmed=True)
            check_cancel()
        raise RuntimeError("Observation pose did not converge within the bounded steps")

    def initialize(self):
        return self.request("arm_initialize", confirmed=True, empty_gripper=True, workspace_clear=True)

    def view(self, name, cancel=None, oem_path_checked=False):
        poses = {"travel": NAV_POSE, "forward": FORWARD_VIEW_POSE, "down20": DOWN20_POSE}
        if name not in poses:
            raise ValueError("Unknown named arm view")
        if name == "forward" and oem_path_checked is not True:
            raise ValueError("OEM forward route must be physically validated before automatic use")
        def check_cancel():
            if cancel is not None and cancel():
                self.stop()
                raise ArmMotionCancelled("Named view cancelled; no grasp")
        check_cancel()
        status = self.request("arm_status")
        if name not in status.get("named_views", []):
            raise RuntimeError("Named-view bridge not enabled; no arm movement")
        check_cancel()
        reached = self.request("arm_view", view=name, expected_deg=status["servo_deg"], confirmed=True,
                               oem_path_checked=oem_path_checked)
        check_cancel()
        if reached.get("view") != name or not near_pose(reached.get("servo_deg"), poses[name], 1):
            raise RuntimeError("Named view feedback does not match its target")
        return reached

    def validate_oem_path(self, cancel=None):
        def check_cancel():
            if cancel is not None and cancel():
                self.stop()
                raise ArmMotionCancelled("Supervised OEM validation cancelled; no further motion")
        check_cancel()
        status = self.request("arm_status")
        if status.get("oem_validation_enabled") is not True or not near_pose(status.get("servo_deg"), NAV_POSE, 1):
            raise RuntimeError("Supervised OEM path validation is not enabled at actual travel pose")
        readings = [status["servo_deg"]]
        for _ in range(32):
            check_cancel()
            current = readings[-1][1]
            target = min(180, current+2)
            reached = self.request("arm_oem_step", s2_deg=target, expected_deg=readings[-1], supervised=True)
            angles = reached.get("servo_deg")
            expected = list(NAV_POSE)
            expected[1] = target
            if not near_pose(angles, expected, 1):
                raise RuntimeError("Incorrect supervised OEM step feedback")
            readings.append(angles)
            check_cancel()
            if target == 180:
                return readings
        raise RuntimeError("Supervised OEM path did not converge")

    def close(self):
        try:
            self.stop()
        finally:
            self.sock.close()


class PixelDetector:
    def __init__(self, reader, model, select_box):
        self.reader, self.model, self.select_box = reader, model, select_box
        self.running = True
        self.lock = threading.Lock()
        self.latest = (None, 0.0, 0.0)
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def loop(self):
        previous = 0.0
        while self.running:
            ok, frame, stamp = self.reader.read()
            if not ok or stamp <= previous:
                time.sleep(.01)
                continue
            previous = stamp
            try:
                result = self.model.predict(frame, conf=.45, imgsz=640, verbose=False)[0]
                detection = self.select_box(result, self.model.names)
                value = (detection["bbox"], stamp, detection["conf"]) if detection else (None, stamp, 0.0)
            except Exception as exc:
                print("[ARM_VISION_ERROR] " + str(exc), flush=True)
                value = (None, stamp, 0.0)
            with self.lock:
                self.latest = value

    def snapshot(self):
        with self.lock:
            return self.latest

    def close(self):
        self.running = False
        self.thread.join(timeout=5)


def ssh(host, command, timeout=30):
    ipaddress.IPv4Address(host)
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                             "jetson@" + host, command], capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError("Jetson command failed: " + result.stdout + result.stderr)
    return result.stdout


def graspctl(host, command):
    if command not in ("status", "home", "grasp"):
        raise ValueError("Unsupported grasp command")
    value = json.loads(ssh(host, "python3 " + REMOTE_CTL + " " + command,
                           timeout=180 if command == "grasp" else 75))
    if value.get("ok") is not True:
        raise RuntimeError("v23 refused command: " + str(value))
    return value


def ensure_serial_free(host):
    code = ("import json,os,subprocess; "
            "r=subprocess.run(['sudo','-n','fuser','/dev/myserial'],stdout=subprocess.PIPE,stderr=subprocess.PIPE); "
            "print(json.dumps({'free':os.path.exists('/dev/myserial') and r.returncode==1 "
            "and not r.stdout and not r.stderr}))")
    value = json.loads(ssh(host, "python3 -c " + shlex.quote(code)))
    if value.get("free") is not True:
        raise RuntimeError("Robot serial port is still owned or its release cannot be verified")


def near_pose(angles, pose, tolerance=2):
    return (isinstance(angles, list) and len(angles) == 6
            and all(isinstance(a, (int, float)) and math.isfinite(a) and abs(a-b) <= tolerance
                    for a, b in zip(angles, pose)))


class GraspTask:
    """Bounded remote work off the GUI thread; cancellation never restarts RL."""
    def __init__(self, host):
        self.host = host
        self.cancelled = threading.Event()
        self.done = threading.Event()
        self.result = None
        self.error = None
        self.owns_grasp_phase = False
        self.thread = threading.Thread(target=self.run, daemon=True)

    def check_cancel(self):
        if self.cancelled.is_set():
            raise RuntimeError("Grasp handoff cancelled")

    def cancel(self):
        self.cancelled.set()
        # SIGTERM is the existing service's KeyboardInterrupt path, not a pose command.
        if self.owns_grasp_phase:
            threading.Thread(target=self.abort_service, daemon=True).start()

    def abort_service(self):
        try:
            ssh(self.host, "sudo -n systemctl stop grasp-service.service grasp-vision.service", 30)
        except Exception as exc:
            print("[GRASP_ABORT_ERROR] " + str(exc), flush=True)

    def run(self):
        try:
            self.check_cancel()
            ssh(self.host, "sudo -n systemctl stop codex-sugarbox-motor.service codex-arm-stream.service")
            self.check_cancel()
            # The old serial owner must be fully stopped before starting the new one.
            states = ssh(self.host, "systemctl show codex-sugarbox-motor.service --property=ActiveState --value").strip()
            # Transient systemd units can disappear after stop; an empty result
            # is acceptable only when the device itself has no owner.
            if states not in ("", "inactive"):
                raise RuntimeError("RL serial owner did not stop")
            ensure_serial_free(self.host)
            self.owns_grasp_phase = True
            ssh(self.host, "sudo -n systemctl start grasp-service.service")
            self.check_cancel()
            status = None
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                self.check_cancel()
                try:
                    status = graspctl(self.host, "status")
                    break
                except Exception:
                    time.sleep(.5)
            if status is None or not near_pose(status.get("servo_deg"), NAV_POSE, tolerance=4):
                raise RuntimeError("Travel pose/servo feedback not confirmed; no E1 motion")
            self.check_cancel()
            graspctl(self.host, "home")
            self.check_cancel()
            status = graspctl(self.host, "status")
            if not near_pose(status.get("servo_deg"), E1_POSE, tolerance=3):
                raise RuntimeError("E1 not reached; no grasp")
            ssh(self.host, "sudo -n systemctl start grasp-vision.service")
            self.check_cancel()
            # v23 itself verifies fresh post-motion E1 vision, envelope and floor clearance.
            self.result = graspctl(self.host, "grasp")
            if self.result.get("confirmed") is not True:
                raise RuntimeError("v23 did not confirm a held object")
        except Exception as exc:
            self.error = str(exc)
        finally:
            if (self.cancelled.is_set() or self.error) and self.owns_grasp_phase:
                self.abort_service()
            self.done.set()


def load_profile(path):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    profile = ApproachProfile.from_dict(data)
    StagedPixelServo(profile)  # Reject every unconfirmed stage before any hardware startup.
    if profile.bottom_switch_v is not None and not profile.forward_view_confirmed:
        raise ValueError("Forward view must be physically checked for rear-camera occlusion")
    return profile


def check_arm_camera(app, timeout=8):
    reader = app.LatestFrameReader(udp_port=5602)
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            ok, frame, stamp = reader.read()
            if ok and 0 <= time.time() - stamp <= .35:
                if frame.shape[:2] != (480, 640):
                    raise RuntimeError("Arm camera dimensions do not match the pixel profile")
                return
            time.sleep(.02)
        raise RuntimeError("No fresh bound arm-camera stream; PPO has not started")
    finally:
        reader.release()


def run_pixel_stage(host, profile, cv2, app):
    motor = MotorConnection(host)
    reader = detector = ros = None
    ready = False
    last_frame = None
    def cancel_pose():
        if reader:
            ok, frame, _ = reader.read()
            if ok:
                panel = frame.copy()
                cv2.putText(panel, "ARM_POSE_SWITCH - CHASSIS STOPPED", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1)
                cv2.imshow("Sugarbox Arm Pixels", panel)
        return cv2.waitKey(1) & 255 in (27, 32)
    try:
        motor.stop()
        reader = app.LatestFrameReader(udp_port=5602)
        if profile.bottom_switch_v is not None:
            motor.view("travel", cancel=cancel_pose, oem_path_checked=profile.forward_view_confirmed)
        else:
            motor.observe(profile.stages[0].observation_s2_deg, cancel=cancel_pose)
        detector = PixelDetector(reader, app.YOLO(app.YOLO_MODEL_PATH), app.select_best_box)
        ros = app.RosSensorBridge(host, app.ROSBRIDGE_PORT)
        ros.start()
        servo = StagedPixelServo(profile)
        last_control = last_print = 0.0
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            now = time.time()
            ok, frame, stamp = reader.read()
            box, box_stamp, confidence = detector.snapshot()
            scan, odom = ros.snapshot()
            if (not ok or now-stamp > .35 or now-scan.stamp > app.SCAN_STALE_TIMEOUT
                    or now-odom.stamp > app.ODOM_STALE_TIMEOUT):
                vx, wz, state = servo.hold("ARM_SENSOR_STALE")
            else:
                _, front_min, samples = app.raw_front_metrics(scan)
                if samples < app.FRONT_ROBUST_MIN_POINTS or front_min < servo.profile.front_stop_m:
                    vx, wz, state = servo.hold("ARM_RAW_FRONT_STOP")
                else:
                    vx, wz, state = servo.step(box, box_stamp, now, confidence)
            if servo.pending_pose is not None and state == "ARM_POSE_SWITCH":
                motor.stop()
                # Drain the GUI emergency key before sending a new joint movement.
                if cv2.waitKey(1) & 255 in (27, 32):
                    return False
                reached = (motor.view("down20", cancel=cancel_pose) if profile.bottom_switch_v is not None
                           else motor.observe(servo.pending_pose, cancel=cancel_pose))
                servo.confirm_pose(reached["servo_deg"][1], time.time())
                vx, wz, state = 0.0, 0.0, "ARM_POSE_SETTLING"
                print("[ARM_POSE] stage={} S2={:g}; fresh vision required".format(
                    servo.index+1, servo.profile.observation_s2_deg), flush=True)
            if now - last_control >= 1/10:
                motor.velocity(vx, wz)
                last_control = now
            if now-last_print >= .5:
                print("[ARM_PIXELS] {} vx={:+.3f} wz={:+.3f}".format(state, vx, wz), flush=True)
                last_print = now
            if ok:
                panel = frame.copy()
                cv2.drawMarker(panel, (int(servo.profile.target_u), int(servo.profile.target_v)), (0, 255, 0),
                               markerType=cv2.MARKER_CROSS, markerSize=24, thickness=2)
                if box is not None:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(panel, (x1, y1), (x2, y2), (0, 255, 255), 2)
                label = "{} | S2={:g} stage {}/{}".format(state, servo.profile.observation_s2_deg,
                                                        servo.index+1, len(profile.stages))
                cv2.putText(panel, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1)
                last_frame = panel
                cv2.imshow("Sugarbox Arm Pixels", panel)
            if cv2.waitKey(10) & 255 in (27, 32):
                return False
            if servo.ready and state in ("ARM_READY", "ARM_READY_LATCHED"):
                motor.stop()
                ready = True
                break
        if not ready:
            raise RuntimeError("Arm-pixel approach did not reach a confirmed ready point")
        # Return to travel before v23's checked travel->E1 path.
        if profile.bottom_switch_v is not None:
            motor.view("travel", cancel=cancel_pose)
        else:
            motor.observe(140, cancel=cancel_pose)
    except ArmMotionCancelled:
        return False
    finally:
        try:
            motor.close()
        finally:
            if detector:
                detector.close()
            if reader:
                reader.release()
            if ros:
                ros.close()
    if ready:
        if cv2.waitKey(1) & 255 in (27, 32):
            return False
        task = GraspTask(host)
        task.thread.start()
        try:
            while not task.done.is_set():
                if last_frame is not None:
                    cv2.imshow("Sugarbox Arm Pixels", last_frame)
                if cv2.waitKey(20) & 255 in (27, 32):
                    task.cancel()
            task.thread.join()
        except KeyboardInterrupt:
            task.cancel()
            task.thread.join(timeout=190)
            raise
        if task.error:
            raise RuntimeError(task.error)
        print("[GRASP_CONFIRMED] " + json.dumps(task.result), flush=True)
    return ready


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("ROUTE_B_IP", "10.29.101.252"))
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--i-am-beside-the-robot", action="store_true")
    parser.add_argument("--arm-empty-and-clear", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    ipaddress.IPv4Address(args.host)
    if args.real and (args.profile is None or not args.i_am_beside_the_robot or not args.arm_empty_and_clear):
        parser.error("Real motion requires --profile, --i-am-beside-the-robot and --arm-empty-and-clear")
    profile = load_profile(args.profile) if args.profile else None
    if args.check:
        if profile is None:
            parser.error("--check requires a confirmed profile")
        print("[PROFILE_OK] No hardware opened")
        return 0
    if not args.real:
        print("Use diagnostics/preview_arm_camera.py for read-only preview; no motion started")
        return 0
    env = os.environ.copy()
    env.update(ROUTE_B_IP=args.host, SUGARBOX_RUN_MODE="DRIVE", SUGARBOX_TRAINING_CONTRACT="1",
               SUGARBOX_CONTRACT_ALLOW_DRIVE="1", SUGARBOX_TARGET_INITIAL_ONLY="1",
               SUGARBOX_PPO_ONLY="0", SUGARBOX_FOOTPRINT_GUARD="0", SUGARBOX_ARM_HANDOFF="1")
    if profile.rear_handoff_v is not None:
        env["SUGARBOX_REAR_HANDOFF_V"] = str(profile.rear_handoff_v)
    else:
        env.pop("SUGARBOX_REAR_HANDOFF_V", None)
    os.environ.update(env)
    import cv2
    import sugarbox_rl_approach_final2 as app
    # Verify capability without changing the travel pose before the PPO leg.
    probe = MotorConnection(args.host)
    try:
        initialized = probe.initialize()
        if initialized.get("initialized") is not True or not near_pose(initialized.get("servo_deg"), NAV_POSE, 3):
            raise RuntimeError("Travel initialization did not reach its pose; PPO has not started")
        status = probe.request("arm_status")
        if (status.get("protocol") != "sugarbox-arm-observe-v1"
                or status.get("allow_arm_observation") is not True
                or not near_pose(status.get("servo_deg"), NAV_POSE)):
            raise RuntimeError("Travel pose/observation capability not confirmed; PPO has not started")
        if getattr(profile, "bottom_switch_v", None) is not None:
            probe.view("forward", oem_path_checked=profile.forward_view_confirmed)
    finally:
        probe.close()
    check_arm_camera(app)
    try:
        result = subprocess.run([sys.executable, "-u", str(HERE / "sugarbox_rl_approach_final2.py")], env=env)
    except BaseException:
        stopper = MotorConnection(args.host)
        stopper.close()
        raise
    if result.returncode != HANDOFF_EXIT_CODE:
        print("[NO_GRASP] PPO ended without an explicit stopped arm handoff")
        return result.returncode
    try:
        return 0 if run_pixel_stage(args.host, profile, cv2, app) else 2
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
