"""Bound arm-camera preview. This tool never opens a motor connection."""
import argparse
from pathlib import Path
import sys
import time

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integration"))
from sugarbox_rl_approach_final2 import LatestFrameReader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--port", type=int, default=5602)
    args = parser.parse_args()
    if not 0 < args.seconds <= 120:
        parser.error("seconds must be in (0,120]")
    reader = LatestFrameReader(udp_port=args.port)
    deadline = time.monotonic() + args.seconds
    captured = False
    try:
        while time.monotonic() < deadline:
            ok, frame, stamp = reader.read()
            if ok:
                if args.capture and not captured:
                    if not cv2.imwrite(str(args.capture), frame):
                        raise RuntimeError("Could not save camera frame")
                    captured = True
                    print("ARM_FRAME_OK", flush=True)
                cv2.imshow("Arm Camera Preview - No Motion", frame)
            if cv2.waitKey(10) & 255 == 27:
                break
        if args.capture and not captured:
            raise RuntimeError("No bound arm-camera frames received")
    finally:
        reader.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
