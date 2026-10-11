"""Offline inspection only: no ROS, camera, socket, or motor initialization."""
import ast
import argparse
import hashlib
import json
import math
import pickle
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--training-contract", action="store_true",
                    help="Include offline training-plant and observation comparison")
args = parser.parse_args()
SOURCE = ROOT / "integration" / "sugarbox_rl_approach_final2.py"
tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
paths = {}
for node in tree.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in ("RL_MODEL_PATH", "VECNORMALIZE_PATH"):
                paths[target.id] = Path(ast.literal_eval(node.value))


def fingerprint(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


with zipfile.ZipFile(paths["RL_MODEL_PATH"]) as archive:
    metadata = json.loads(archive.read("data"))
    model_version = archive.read("_stable_baselines3_version").decode().strip()

# This is the same trusted local normalization artifact used by the controller.
with paths["VECNORMALIZE_PATH"].open("rb") as handle:
    normalizer = pickle.load(handle)
mean = np.asarray(normalizer.obs_rms.mean)
var = np.asarray(normalizer.obs_rms.var)
normalizer.training = False
normalizer.norm_reward = False

example = np.zeros((1, 55), dtype=np.float32)
example[0, :7] = [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
example[0, 7:] = 4.0
example[0, 29:33] = 0.12
normalized = normalizer.normalize_obs(example.copy())

# Execute only pure adapter functions, never the controller's main/imports.
names = {"wrap_pi", "angle_to_scan_index", "sanitize_policy_range", "scan_to_policy_rays"}
adapter_tree = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
adapter = {"math": math, "np": np, "NUM_LIDAR_RAYS": 48,
           "LIDAR_MAX_M": 4.0, "POLICY_ANGLES_DEG": np.linspace(-90, 90, 48),
           "LIDAR_REFLECT_FRONT_BACK": True, "PPO_LIDAR_DISTANCE_SCALE": 1.0}
exec(compile(adapter_tree, str(SOURCE), "exec"), adapter)
angles = np.linspace(-math.pi, math.pi, 2031)
body_angles = np.arctan2(np.sin(math.pi - angles), np.cos(math.pi - angles))
ranges = np.full(angles.shape, 4.0)
ranges[np.abs(body_angles) < math.radians(1.0)] = 0.3
scan = SimpleNamespace(ranges=ranges, angle_min=float(angles[0]),
                       angle_increment=float(angles[1] - angles[0]))
sampled = adapter["scan_to_policy_rays"](scan)

model = PPO.load(str(paths["RL_MODEL_PATH"]), device="cpu")
responses = {}
for label, ray_slice in (("clear", None), ("front_12cm", slice(19, 29)),
                         ("right_12cm", slice(0, 16)), ("left_12cm", slice(32, 48))):
    obs = example.copy()
    obs[0, 7:] = 4.0
    if ray_slice is not None:
        obs[0, 7:][ray_slice] = 0.12
    action, _ = model.predict(normalizer.normalize_obs(obs), deterministic=True)
    responses[label] = np.asarray(action).reshape(-1).tolist()

report = {
    "model": fingerprint(paths["RL_MODEL_PATH"]),
    "normalizer": fingerprint(paths["VECNORMALIZE_PATH"]),
    "model_sb3_version": model_version,
    "model_metadata": {k: metadata.get(k) for k in ("num_timesteps", "n_steps", "gamma", "n_envs")},
    "obs_shape": list(mean.shape),
    "norm_obs": normalizer.norm_obs,
    "clip_obs": normalizer.clip_obs,
    "epsilon": normalizer.epsilon,
    "stats_count": float(normalizer.obs_rms.count),
    "finite_stats": bool(np.isfinite(mean).all() and np.isfinite(var).all()),
    "positive_variances": bool((var > 0).all()),
    "mean_goal_and_feedback": mean[:7].tolist(),
    "lidar_mean_min_max": [float(mean[7:].min()), float(mean[7:].max())],
    "example_normalized_finite": bool(np.isfinite(normalized).all()),
    "example_front_normalized": normalized[0, 29:33].tolist(),
    "deployed_action_examples": [
        {"a0": a, "policy_vx_before_limit": a * 0.5, "client_vx": min(a * 0.5, 0.15)}
        for a in (0.1, 0.3, 0.5, 1.0)
    ],
    "ray_spacing_degrees": 180.0 / 47.0,
    "model_obs_shape": list(model.observation_space.shape),
    "model_action_shape": list(model.action_space.shape),
    "synthetic_narrow_obstacle_raw_min_m": float(ranges.min()),
    "synthetic_narrow_obstacle_policy_min_m": float(sampled.min()),
    "synthetic_policy_responses_not_physical_validation": responses,
}
if args.training_contract:
    sys.path.insert(0, str(ROOT / "integration"))
    from sugarbox_training_contract import (
        GOAL_ORIGIN_OFFSET_X_M, LIDAR_CENTER_OFFSET_X_M, TrainingActionReference,
        center_relative_goal, center_relative_hit,
        limited_reference_twist, training_observation,
    )

    comparisons = []
    for minimum in (4.0, 0.66, 0.33):
        ref = TrainingActionReference()
        for _ in range(3):
            shaped = ref.step([1, 1], minimum, 0.8, 0, 0.1, 0)
        limited = limited_reference_twist(shaped["vx"], shaped["wz"])
        comparisons.append({
            "assumed_center_ray_min_m": minimum,
            "training_twist": [shaped["vx"], shaped["wz"]],
            "offline_proportional_limit_twist": list(limited),
            "current_client_twist": [0.15, 0.78],
        })
    training_obs = training_observation(1, 0, -0.03, 0.04, 0,
                                         [0, 0], np.full(48, 0.33))
    report["offline_training_contract_comparison"] = {
        "not_wired_to_robot": True,
        "user_reported_lidar_center_offset_x_m": LIDAR_CENTER_OFFSET_X_M,
        "user_reported_goal_origin_offset_x_m": GOAL_ORIGIN_OFFSET_X_M,
        "goal_front_80cm_center_xy": list(center_relative_goal(0.8, 0.0)),
        "front_hit_20cm_center_polar": list(center_relative_hit(0.2, 0.0)),
        "twist_examples": comparisons,
        "training_reverse_lateral_speed_obs": float(training_obs[3]),
        "current_signed_forward_speed_obs": -0.03,
        "live_motor_velocity_mapping_not_calibrated": True,
    }
print(json.dumps(report, indent=2))
