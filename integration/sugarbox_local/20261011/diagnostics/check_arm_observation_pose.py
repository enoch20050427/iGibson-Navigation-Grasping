"""Offline FK/collision comparison only; no hardware or network access."""
import json
import os
import argparse
from pathlib import Path
import sys

import numpy as np
import pybullet as p

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "grasp/v23/validate_nav_to_grasp_transition.py").is_file())
REFERENCE = ROOT / "grasp" / "v23"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integration"))
from sugarbox_arm_pixels import FORWARD_VIEW_POSE, DOWN20_POSE
sys.path.insert(0, str(REFERENCE))
from validate_nav_to_grasp_transition import E1, Model, TRAVEL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s2", nargs="+", type=float, default=[140, 137, 143])
    parser.add_argument("--pose", nargs=6, type=float, action="append",
                        help="additional offline-only six-servo candidate; never sent to hardware")
    parser.add_argument("--named-routes", action="store_true",
                        help="check all empty-arm coordinated routes between travel/forward/down20")
    args = parser.parse_args()
    # PyBullet's Windows loader cannot open some non-ASCII absolute paths.
    old_cwd, loader = Path.cwd(), p.loadURDF
    try:
        os.chdir(REFERENCE.parent / "x3plus")
        p.loadURDF = lambda path, *a, **kw: loader(os.path.relpath(path), *a, **kw)
        model = Model()
        if args.named_routes:
            model.path = lambda a, b: [tuple(np.array(a)+(np.array(b)-a)*fraction)
                                      for fraction in np.linspace(0, 1, 181)]
        model.set(E1)
        rotation = np.array(p.getMatrixFromQuaternion(
            p.getLinkState(model.robot, model.link["mono_link"])[1])).reshape(3, 3)
        expected = np.array([.024432, 0, -.999701])
        axes = [(float(np.dot(rotation[:, i] * sign, expected)), i, sign)
                for i in range(3) for sign in (-1, 1)]
        score, index, sign = max(axes)
        if score < .999:
            raise RuntimeError("Could not match the documented E1 optical axis")
        records = []
        poses = [tuple(TRAVEL[:1]) + (s2,) + tuple(TRAVEL[2:]) for s2 in args.s2]
        poses.extend(tuple(pose) for pose in (args.pose or []))
        for pose in poses:
            distances = model.check([(TRAVEL, pose)], carry=False)
            model.set(pose)
            state = p.getLinkState(model.robot, model.link["mono_link"])
            rotation = np.array(p.getMatrixFromQuaternion(state[1])).reshape(3, 3)
            axis = rotation[:, index] * sign
            depression = np.degrees(np.arctan2(-axis[2], axis[0]))
            nearest = min((lo, pair) for pair, (lo, _) in distances.items()
                          if not pair.startswith("FLOOR:"))
            floor = min(lo for pair, (lo, _) in distances.items() if pair.startswith("FLOOR:"))
            records.append(dict(servo_deg=list(pose), s2_deg=pose[1], optical_depression_deg=float(depression),
                                optical_axis=axis.tolist(), floor_min_mm=floor * 1000,
                                nearest_pair=nearest[1], nearest_mm=nearest[0] * 1000))
        routes = []
        if args.named_routes:
            named = {"travel": TRAVEL, "forward": FORWARD_VIEW_POSE, "down20": DOWN20_POSE}
            for start, a in named.items():
                for end, b in named.items():
                    if start == end:
                        continue
                    distances = model.check([(a, b)], carry=False)
                    gap, pair = min((lo, pair) for pair, (lo, _) in distances.items()
                                    if not pair.startswith("FLOOR:"))
                    floor = min(lo for pair, (lo, _) in distances.items() if pair.startswith("FLOOR:"))
                    routes.append(dict(start=start, end=end, nearest_mm=gap*1000, nearest_pair=pair,
                                       floor_min_mm=floor*1000, model_nonpenetrating=gap >= 0 and floor >= 0))
        print(json.dumps({"offline_only": True, "poses": records, "named_routes": routes}, indent=2))
    finally:
        if p.isConnected():
            p.disconnect()
        p.loadURDF = loader
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
