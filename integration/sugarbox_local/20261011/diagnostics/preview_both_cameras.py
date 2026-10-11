"""Two bound camera views for a stationary handoff check; no motor connection."""
import argparse
from pathlib import Path
import sys
import time

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integration"))
from sugarbox_rl_approach_final2 import LatestFrameReader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--capture-prefix", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.seconds <= 300:
        parser.error("seconds must be in (0,300]")
    readers = []
    frames = {}
    try:
        for name, port in (("Rear Camera - UDP5600", 5600), ("Arm Camera - UDP5602", 5602)):
            readers.append((name, LatestFrameReader(udp_port=port)))
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(name, 640, 480)
        cv2.moveWindow(readers[0][0], 30, 60)
        cv2.moveWindow(readers[1][0], 700, 60)
        deadline = time.monotonic() + args.seconds
        saved = False
        while time.monotonic() < deadline:
            now = time.time()
            fresh = {}
            for name, reader in readers:
                ok, frame, stamp = reader.read()
                if ok and 0 <= now-stamp <= .35:
                    frames[name] = frame
                    fresh[name] = frame
                    cv2.imshow(name, frame)
            key = cv2.waitKey(10) & 255
            if key in (27, 32):
                break
            if len(fresh) == 2 and (not saved or key == ord("c")):
                suffix = "initial" if not saved else str(int(now * 1000))
                for name, frame in fresh.items():
                    camera = "rear" if name.startswith("Rear") else "arm"
                    output = args.capture_prefix.with_name(args.capture_prefix.name + "_" + camera + "_" + suffix + ".png")
                    if not cv2.imwrite(str(output), frame):
                        raise RuntimeError("Cannot save paired camera frame")
                print("[CAMERA_PAIR_OK] " + suffix, flush=True)
                saved = True
        if not saved:
            raise RuntimeError("Did not receive fresh frames from both bound cameras")
    finally:
        for _, reader in readers:
            reader.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
