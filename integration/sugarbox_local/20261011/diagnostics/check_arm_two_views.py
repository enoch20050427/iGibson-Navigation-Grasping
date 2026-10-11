"""Stationary forward -> travel -> down20 camera check; no drive or grasp."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integration"))
from sugarbox_arm_approach import ArmMotionCancelled, MotorConnection, near_pose
from sugarbox_arm_pixels import DOWN20_POSE, FORWARD_VIEW_POSE, NAV_POSE


def check_views(motor, cancel, capture):
    if cancel():
        raise ArmMotionCancelled("Cancelled before arm movement")
    status = motor.request("arm_status")
    if not near_pose(status.get("servo_deg"), FORWARD_VIEW_POSE, 1):
        raise RuntimeError("Expected actual forward pose; no initialization or motion")
    for name, pose in (("travel", NAV_POSE), ("down20", DOWN20_POSE)):
        reached = motor.view(name, cancel=cancel, oem_path_checked=True)
        if not near_pose(reached.get("servo_deg"), pose, 1):
            raise RuntimeError("Incorrect named-view feedback")
        capture(name, reached)


def main():
    import cv2
    import numpy as np
    from sugarbox_rl_approach_final2 import LatestFrameReader

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=7001)
    parser.add_argument("--seconds", type=float, default=90)
    parser.add_argument("--capture-prefix", type=Path, required=True)
    for flag in ("i-am-beside-the-robot", "empty-arm-and-clear", "oem-path-checked"):
        parser.add_argument("--" + flag, action="store_true", required=True)
    args = parser.parse_args()
    if not 0 < args.seconds <= 180:
        parser.error("seconds must be in (0,180]")
    names = ("Rear Camera - Two View Check", "Arm Camera - Two View Check")
    readers = []
    motor = None
    moving_started = False
    cancelled = False
    label = "SETUP | CHASSIS STOPPED"

    def read_pair():
        pair = []
        for index, reader in enumerate(readers):
            ok, frame, stamp = reader.read()
            if ok and frame.shape[:2] == (480, 640) and 0 <= time.time()-stamp <= .35:
                pair.append((index, frame, stamp))
                panel = frame.copy()
                cv2.line(panel, (320, 0), (320, 479), (0, 255, 0), 1)
                cv2.putText(panel, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                            .48, (255, 255, 255), 1)
                cv2.imshow(names[index], panel)
        return pair

    def cancel():
        nonlocal cancelled
        if moving_started and len(read_pair()) != 2:
            cancelled = True
        if (cv2.waitKey(1) & 255 in (27, 32)
                or any(cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1 for name in names)):
            cancelled = True
        return cancelled

    def capture(view, reached):
        nonlocal label
        label = "S2={:g} | STOPPED | ESC closes".format(reached["servo_deg"][1])
        finished_at = time.time()
        deadline = time.monotonic()+10
        while time.monotonic() < deadline:
            if cancel():
                raise ArmMotionCancelled("Cancelled after named-view movement")
            pair = read_pair()
            if len(pair) == 2 and all(stamp > finished_at+.35 for _, _, stamp in pair):
                break
            time.sleep(.01)
        else:
            raise RuntimeError("No fresh post-movement camera pair")
        actual = motor.request("arm_status")["servo_deg"]
        if not near_pose(actual, reached["servo_deg"], 1):
            raise RuntimeError("Arm changed before image capture")
        record = dict(view=view, servo_deg=actual, oem_path_checked=True,
                      confirmed_grasp_point=False, frames={}, saved_at=time.time())
        for index, frame, stamp in pair:
            camera = "rear" if index == 0 else "arm"
            path = args.capture_prefix.with_name(args.capture_prefix.name+"_"+view+"_"+camera+".png")
            if path.exists() or not cv2.imwrite(str(path), frame):
                raise RuntimeError("Cannot save a new camera image")
            record["frames"][camera] = dict(path=str(path.resolve()), receiver_stamp=stamp)
        path = args.capture_prefix.with_name(args.capture_prefix.name+"_"+view+".json")
        with path.open("x", encoding="utf-8") as file:
            json.dump(record, file, indent=2)
        print("[VIEW_CAPTURED] " + json.dumps(record), flush=True)
        if view == "travel":
            deadline = time.monotonic()+2
            while time.monotonic() < deadline:
                if cancel():
                    raise ArmMotionCancelled("Cancelled before down20 movement")
                time.sleep(.01)

    try:
        for index, port in enumerate((5600, 5602)):
            readers.append(LatestFrameReader(udp_port=port))
            cv2.namedWindow(names[index], cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow(names[index], 30+index*670, 60)
            cv2.imshow(names[index], np.zeros((480, 640, 3), dtype=np.uint8))
        deadline = time.monotonic()+20
        while time.monotonic() < deadline:
            if cancel():
                raise ArmMotionCancelled("Cancelled before camera setup")
            if len(read_pair()) == 2:
                break
            time.sleep(.01)
        else:
            raise RuntimeError("Both cameras must be fresh before arm movement")
        motor = MotorConnection(args.host, args.port)
        moving_started = True
        check_views(motor, cancel, capture)
        deadline = time.monotonic()+args.seconds
        while time.monotonic() < deadline and not cancel():
            time.sleep(.01)
    except ArmMotionCancelled as exc:
        print("[CANCELLED] " + str(exc), flush=True)
    finally:
        try:
            if motor is not None:
                motor.close()
        finally:
            for reader in readers:
                reader.release()
            cv2.destroyAllWindows()
            print("[DONE] Arm left in place; no driving or grasp started.", flush=True)


if __name__ == "__main__":
    main()
