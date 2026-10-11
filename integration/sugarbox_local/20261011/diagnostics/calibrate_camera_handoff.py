"""Stationary dual-camera mouse marks. No chassis motion or grasp commands."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integration"))
from sugarbox_arm_approach import MotorConnection, ArmMotionCancelled
from handoff_calibration import HandoffDraft


def main():
    import cv2
    import numpy as np
    from sugarbox_rl_approach_final2 import LatestFrameReader

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=7001)
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--template", type=Path, default=Path(__file__).with_name("arm_staged_profile.template.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initialize-empty-clear-arm", action="store_true")
    parser.add_argument("--i-am-beside-the-robot", action="store_true", required=True)
    args = parser.parse_args()
    if not 0 < args.seconds <= 300:
        parser.error("seconds must be in (0,300]")
    evidence_path = args.output.with_name(args.output.stem + "_evidence.json")
    images = {camera: args.output.with_name(args.output.stem + "_" + camera + ".png")
              for camera in ("rear", "arm")}
    if any(path.exists() for path in (args.output, evidence_path, *images.values())):
        parser.error("Use a new output name; calibration evidence is never overwritten")
    draft = HandoffDraft(json.loads(args.template.read_text(encoding="utf-8")))
    motor = MotorConnection(args.host, args.port)
    readers, latest, selected = {}, {}, {}
    pose = None
    pose_time = 0.0
    moved = False
    cancelled = False
    saved = False
    names = {"rear": "Rear Handoff - click Sugarbox bottom", "arm": "Arm Overlap - click Sugarbox bottom"}

    def cancel():
        nonlocal cancelled
        if (cv2.waitKey(1) & 255 in (27, 32)
                or any(cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1 for name in names.values())):
            cancelled = True
        return cancelled

    def click(event, x, y, flags, camera):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        try:
            frame, stamp = latest[camera]
            now = time.time()
            if now-pose_time > 2.5:
                raise ValueError("Servo feedback is not fresh; wait for the next reading")
            draft.mark(camera, x, y, stamp, now, frame.shape, pose)
            selected[camera] = frame.copy()
            print("[MARK] " + camera + " " + str((x, y)), flush=True)
        except (KeyError, ValueError) as exc:
            print("[MARK_REFUSED] " + str(exc), flush=True)

    try:
        for camera, port in (("rear", 5600), ("arm", 5602)):
            readers[camera] = LatestFrameReader(udp_port=port)
            cv2.namedWindow(names[camera], cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow(names[camera], 30 if camera == "rear" else 700, 60)
            cv2.setMouseCallback(names[camera], click, camera)
            panel = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(panel, "ARM SETUP | ESC/SPACE cancels between steps", (8, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
            cv2.imshow(names[camera], panel)
        if cancel():
            raise ArmMotionCancelled("Cancelled before initialization")
        if args.initialize_empty_clear_arm:
            print("[ARM_INITIALIZED] " + json.dumps(motor.initialize()), flush=True)
            moved = True
            print("[ARM_HANDOFF_POSE] " + json.dumps(motor.observe(130, cancel=cancel)), flush=True)
        pose = motor.request("arm_status")["servo_deg"]
        HandoffDraft.check_pose(pose)
        pose_time = time.time()
        print("[READY] Click box bottom in both views; S saves draft and closes. ESC/SPACE cancels.", flush=True)
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline and not cancelled:
            if time.time()-pose_time >= 1:
                pose = motor.request("arm_status")["servo_deg"]
                HandoffDraft.check_pose(pose)
                pose_time = time.time()
            now = time.time()
            for camera, reader in readers.items():
                ok, frame, stamp = reader.read()
                if ok and 0 <= now-stamp <= .35 and frame.shape[:2] == (480, 640):
                    latest[camera] = (frame, stamp)
                    panel = frame.copy()
                    mark = draft.marks.get(camera)
                    if mark:
                        cv2.drawMarker(panel, (mark["u"], mark["v"]), (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
                        if camera == "rear":
                            cv2.line(panel, (0, mark["v"]), (639, mark["v"]), (0, 255, 0), 1)
                    cv2.putText(panel, "S2=130 | STOPPED | draft only | S save | ESC cancel", (8, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
                    cv2.imshow(names[camera], panel)
                else:
                    latest.pop(camera, None)
                    # Do not leave a frozen camera image available for clicking.
                    if frame is not None:
                        panel = frame.copy()
                        panel[:] = 0
                        cv2.putText(panel, "WAITING FOR FRESH 640x480 FRAME", (8, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2)
                        cv2.imshow(names[camera], panel)
            key = cv2.waitKey(10) & 255
            if key in (27, 32) or any(cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1 for name in names.values()):
                cancelled = True
            elif key in (ord("s"), ord("S")):
                try:
                    profile, evidence = draft.export(motor.request("arm_status")["servo_deg"])
                except ValueError as exc:
                    print("[SAVE_REFUSED] " + str(exc), flush=True)
                    continue
                args.output.parent.mkdir(parents=True, exist_ok=True)
                for camera, path in images.items():
                    if not cv2.imwrite(str(path), selected[camera]):
                        raise RuntimeError("Cannot save camera evidence")
                evidence["images"] = {camera: str(path.resolve()) for camera, path in images.items()}
                evidence["saved_at"] = time.time()
                evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
                args.output.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
                saved = True
                print("[DRAFT_SAVED] " + str(args.output), flush=True)
                break
        if not saved:
            print("[NOT_SAVED] No profile authorized or saved.", flush=True)
    except ArmMotionCancelled:
        cancelled = True
        print("[CANCELLED] No further arm movement.", flush=True)
    except BaseException:
        cancelled = True
        raise
    finally:
        try:
            motor.stop()
            if moved and not cancelled:
                print("[ARM_RESTORE] " + json.dumps(motor.observe(140, cancel=cancel)), flush=True)
            elif moved:
                print("[ARM_LEFT_IN_PLACE] Cancelled: travel pose must be checked before driving.", flush=True)
        finally:
            try:
                motor.close()
            finally:
                for reader in readers.values():
                    reader.release()
                cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
