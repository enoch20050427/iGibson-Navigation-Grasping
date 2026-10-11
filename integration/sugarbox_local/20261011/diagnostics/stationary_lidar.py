"""Subscribe to LiDAR/odom only; never connect to or publish motor commands."""
import argparse
import json
import math
from pathlib import Path
import statistics
import threading
import time

import roslibpy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    client = roslibpy.Ros(host=args.host, port=9090)
    scans, odoms = [], []
    lock = threading.Lock()

    def capture_scan(msg):
        with lock:
            scans.append({"received_at": time.time(), "message": dict(msg)})

    def capture_odom(msg):
        with lock:
            odoms.append(dict(msg))

    scan_topic = roslibpy.Topic(client, "/scan", "sensor_msgs/LaserScan")
    odom_topic = roslibpy.Topic(client, "/odom_setmotor", "nav_msgs/Odometry")
    try:
        client.run(timeout=5)
        scan_topic.subscribe(capture_scan)
        odom_topic.subscribe(capture_odom)
        time.sleep(6)
    finally:
        scan_topic.unsubscribe()
        odom_topic.unsubscribe()
        client.terminate()

    with lock:
        scans, odoms = list(scans), list(odoms)
    output = Path(args.output)
    output.write_text(json.dumps({"scans": scans, "odoms": odoms}, ensure_ascii=True),
                      encoding="utf-8")
    print("Read-only capture: {} scans, {} odom samples".format(len(scans), len(odoms)))
    if not scans:
        raise RuntimeError("No LiDAR scans received")
    first = scans[0]["message"]
    print("Scan metadata:", json.dumps({key: first.get(key) for key in
          ("header", "angle_min", "angle_max", "angle_increment", "range_min", "range_max")}))
    bins, tight_counts, padded_counts = {}, [], []
    for entry in scans:
        msg = entry["message"]
        tight, padded = 0, 0
        seen = set()
        for i, value in enumerate(msg["ranges"]):
            if value is None:
                continue
            distance = float(value)
            if not math.isfinite(distance) or distance <= max(0.0, msg["range_min"]):
                continue
            if msg["range_max"] > 0 and distance > msg["range_max"]:
                continue
            angle = math.pi - (msg["angle_min"] + i * msg["angle_increment"])
            angle = math.atan2(math.sin(angle), math.cos(angle))
            x, y = distance * math.cos(angle), distance * math.sin(angle)
            is_tight = -0.32 <= x <= 0.0 and abs(y) <= 0.125
            is_padded = -0.38 <= x <= 0.06 and abs(y) <= 0.185
            tight += is_tight
            padded += is_padded
            if is_padded:
                key = int(math.floor(math.degrees(angle) / 5.0)) * 5
                item = bins.setdefault(key, {"ranges": [], "x": [], "y": [], "frames": 0})
                item["ranges"].append(distance)
                item["x"].append(x)
                item["y"].append(y)
                seen.add(key)
        for key in seen:
            bins[key]["frames"] += 1
        tight_counts.append(tight)
        padded_counts.append(padded)
    print("Points inside unpadded chassis per frame min/median/max:",
          min(tight_counts), statistics.median(tight_counts), max(tight_counts))
    print("Points inside padded chassis per frame min/median/max:",
          min(padded_counts), statistics.median(padded_counts), max(padded_counts))
    for angle, item in sorted(bins.items()):
        print("body_angle=[{}, {})deg frames={}/{} median_range={:.3f}m xy=({:+.3f},{:+.3f})m".format(
            angle, angle + 5, item["frames"], len(scans),
            statistics.median(item["ranges"]), statistics.median(item["x"]),
            statistics.median(item["y"])))
    if odoms:
        velocities = [msg["twist"]["twist"] for msg in odoms]
        print("Max odom abs vx/wz:", max(abs(v["linear"]["x"]) for v in velocities),
              max(abs(v["angular"]["z"]) for v in velocities))
    print("Saved:", output)


if __name__ == "__main__":
    main()
