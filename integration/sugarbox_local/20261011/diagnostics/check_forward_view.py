"""Supervised stationary forward-camera view check; no driving or grasp."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integration"))
from sugarbox_arm_approach import MotorConnection, ArmMotionCancelled, near_pose
from sugarbox_arm_pixels import NAV_POSE, FORWARD_VIEW_POSE


def main():
    import cv2
    import numpy as np
    from sugarbox_rl_approach_final2 import LatestFrameReader

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=7001)
    parser.add_argument("--seconds", type=float, default=90)
    parser.add_argument("--capture-prefix", type=Path, required=True)
    parser.add_argument("--i-am-beside-the-robot", action="store_true", required=True)
    parser.add_argument("--empty-arm-and-clear", action="store_true", required=True)
    parser.add_argument("--oem-path-checked", action="store_true",
                        help="OEM S2-only forward path has already been physically validated")
    parser.add_argument("--validate-oem-path", action="store_true",
                        help="supervised first validation via small S2 steps, no chassis motion")
    args = parser.parse_args()
    if not 0 < args.seconds <= 180:
        parser.error("seconds must be in (0,180]")
    if not args.oem_path_checked and not args.validate_oem_path:
        parser.error("OEM pose identified, but model conflicts with it; validate S2 path before this preview")
    motor = MotorConnection(args.host, args.port)
    readers = []
    cancelled = False
    moved = False
    path_confirmed = args.oem_path_checked
    readings = []
    names = ("Rear Camera - Forward Pose Check", "Arm Camera - Forward Pose Check")

    def cancel():
        nonlocal cancelled
        fresh = 0
        for index, reader in enumerate(readers):
            ok, frame, stamp = reader.read()
            if ok and 0 <= time.time()-stamp <= .35:
                fresh += 1
                panel = frame.copy()
                cv2.putText(panel, "S2 PATH CHECK | CHASSIS STOPPED | ESC cancels", (8, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, .47, (255, 255, 255), 1)
                cv2.imshow(names[index], panel)
        if moved and fresh != 2:
            cancelled = True
        if (cv2.waitKey(1) & 255 in (27, 32)
                or any(cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1 for name in names)):
            cancelled = True
        return cancelled

    try:
        for index, port in enumerate((5600, 5602)):
            readers.append(LatestFrameReader(udp_port=port))
            cv2.namedWindow(names[index], cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow(names[index], 30+index*670, 60)
            panel = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(panel, "STOPPED | FORWARD VIEW SETUP | ESC cancels", (8, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
            cv2.imshow(names[index], panel)
        deadline = time.monotonic()+20
        while time.monotonic() < deadline:
            if cancel():
                raise ArmMotionCancelled("Cancelled before initial path validation")
            values = [reader.read() for reader in readers]
            if all(ok and frame.shape[:2] == (480, 640) and 0 <= time.time()-stamp <= .35
                   for ok, frame, stamp in values):
                break
            time.sleep(.02)
        else:
            raise RuntimeError("Both bound cameras must be fresh before any new pose motion")
        status = motor.request("arm_status")
        if not near_pose(status.get("servo_deg"), NAV_POSE, 2):
            raise RuntimeError("Actual travel pose is not confirmed; no new pose commanded")
        moved = True
        if args.validate_oem_path:
            readings = motor.validate_oem_path(cancel=cancel)
            reached = dict(servo_deg=readings[-1], path_validated=False)
        else:
            reached = motor.view("forward", cancel=cancel, oem_path_checked=True)
        print("[FORWARD_READBACK] " + json.dumps(reached), flush=True)
        deadline = time.monotonic()+args.seconds
        saved = False
        while time.monotonic() < deadline and not cancelled:
            now = time.time()
            frames = []
            for index, reader in enumerate(readers):
                ok, frame, stamp = reader.read()
                if ok and frame.shape[:2] == (480, 640) and 0 <= now-stamp <= .35:
                    frames.append((index, frame, stamp))
                    panel = frame.copy()
                    cv2.putText(panel, "FORWARD | S confirms clear path/view | ESC cancels", (8, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, .47, (255, 255, 255), 1)
                    cv2.imshow(names[index], panel)
            key = cv2.waitKey(10) & 255
            if key in (27, 32) or any(cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1 for name in names):
                cancelled = True
            elif len(frames) == 2 and (not saved or key in (ord("s"), ord("S"))):
                actual = motor.request("arm_status")["servo_deg"]
                if not near_pose(actual, FORWARD_VIEW_POSE, 1):
                    raise RuntimeError("Forward pose changed during image capture")
                suffix = "initial" if not saved else str(int(now*1000))
                operator_confirmed = key in (ord("s"), ord("S"))
                evidence = dict(servo_deg=actual, physically_viewed_only=True,
                                oem_path_checked=operator_confirmed or path_confirmed,
                                occlusion_confirmed=operator_confirmed, step_readings=readings,
                                frames={}, saved_at=time.time())
                for index, frame, stamp in frames:
                    camera = "rear" if index == 0 else "arm"
                    path = args.capture_prefix.with_name(args.capture_prefix.name+"_"+camera+"_"+suffix+".png")
                    if path.exists() or not cv2.imwrite(str(path), frame):
                        raise RuntimeError("Cannot save a new view image")
                    evidence["frames"][camera] = dict(path=str(path.resolve()), receiver_stamp=stamp)
                path = args.capture_prefix.with_name(args.capture_prefix.name+"_"+suffix+".json")
                with path.open("x", encoding="utf-8") as file:
                    json.dump(evidence, file, indent=2)
                saved = True
                print("[FORWARD_PAIR_SAVED] " + str(path), flush=True)
                if key in (ord("s"), ord("S")):
                    path_confirmed = True
                    break
        if not saved:
            print("[NO_FRESH_PAIR]", flush=True)
    except ArmMotionCancelled:
        cancelled = True
        print("[CANCELLED] No further arm movements.", flush=True)
    except BaseException:
        cancelled = True
        raise
    finally:
        try:
            motor.stop()
            if moved and not cancelled and path_confirmed:
                print("[TRAVEL_READBACK] " + json.dumps(motor.view("travel", cancel=cancel, oem_path_checked=True)), flush=True)
            elif moved:
                print("[ARM_LEFT_IN_PLACE] Confirm travel pose before driving.", flush=True)
        finally:
            try:
                motor.close()
            finally:
                for reader in readers:
                    reader.release()
                cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
