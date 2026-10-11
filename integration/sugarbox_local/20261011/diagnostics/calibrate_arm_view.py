"""Supervised S2-only view check. No chassis velocity or jaw commands."""
import argparse
import json
from pathlib import Path
import sys
import time

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integration"))
from sugarbox_arm_approach import MotorConnection
from sugarbox_rl_approach_final2 import LatestFrameReader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=7001)
    parser.add_argument("--seconds", type=float, default=90)
    parser.add_argument("--s2", type=float, default=137)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--i-am-beside-the-robot", action="store_true", required=True)
    parser.add_argument("--initialize-empty-arm", action="store_true",
                        help="operator has confirmed an empty gripper and clear arm workspace")
    args = parser.parse_args()
    if not 0 < args.seconds <= 120:
        parser.error("seconds must be in (0,120]")
    if not 130 <= args.s2 <= 140:
        parser.error("s2 must be in the checked 130..140 range")
    motor = MotorConnection(args.host, args.port)
    reader = None
    tilted = False
    captured = False
    try:
        if args.initialize_empty_arm:
            print("[ARM_INITIALIZED] " + json.dumps(motor.initialize()), flush=True)
        print("[ARM_BEFORE] " + json.dumps(motor.request("arm_status")), flush=True)
        tilted = True
        result = motor.observe(args.s2)
        print("[ARM_OBSERVE] " + json.dumps(result), flush=True)
        reader = LatestFrameReader(udp_port=5602)
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            ok, frame, stamp = reader.read()
            if ok and 0 <= time.time() - stamp <= .35:
                if not captured:
                    if not cv2.imwrite(str(args.capture), frame):
                        raise RuntimeError("Cannot save observation image")
                    captured = True
                    print("[ARM_FRAME_OK] " + str(args.capture), flush=True)
                panel = frame.copy()
                cv2.line(panel, (320, 0), (320, 479), (0, 255, 0), 1)
                cv2.putText(panel, "S2={:g} | view only | no grip-ready point set".format(args.s2), (8, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1)
                cv2.imshow("Arm View Check - Chassis Stopped", panel)
            if cv2.waitKey(10) & 255 in (27, 32):
                print("[VIEW_CLOSED]", flush=True)
                break
        if not captured:
            raise RuntimeError("No fresh arm-camera frame")
    finally:
        try:
            if tilted:
                print("[ARM_RESTORE] " + json.dumps(motor.observe(140)), flush=True)
        finally:
            try:
                motor.close()
            finally:
                if reader:
                    reader.release()
                cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
