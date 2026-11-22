"""CPU-only recovery of a robot path clock; contains no fluid replay data.

Read the owned legacy rig export and official URDF. The path coordinate u is
the sum of per-edge max_j(abs(delta_q_j) / velocity_limit_j), in normalized
robot-speed seconds. This is a velocity-only bound, not a five-second proof.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lab_paths import require_isaac_sim_root
HERE = ROOT / "_cache/pour_training/fast_v3"
DEFAULT_BANK = HERE / "legacy_kinematic_bank.npz"
DEFAULT_CACHE = HERE / "fast_path_cache.npz"
URDF = require_isaac_sim_root() / "exts/isaacsim.asset.importer.urdf/data/urdf/robots/franka_description/robots/panda_arm_hand.urdf"
GEOMETRY = ROOT / "assets/robot_testtube_pour/scripts/geometry.py"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def origin_matrix(joint):
    node = joint.find("origin")
    xyz = [float(x) for x in node.get("xyz", "0 0 0").split()]
    r, p, y = [float(x) for x in node.get("rpy", "0 0 0").split()]
    rx = np.array(((1, 0, 0), (0, math.cos(r), -math.sin(r)), (0, math.sin(r), math.cos(r))))
    ry = np.array(((math.cos(p), 0, math.sin(p)), (0, 1, 0), (-math.sin(p), 0, math.cos(p))))
    rz = np.array(((math.cos(y), -math.sin(y), 0), (math.sin(y), math.cos(y), 0), (0, 0, 1)))
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = rz @ ry @ rx, xyz
    return matrix


def prepare(bank=DEFAULT_BANK, output=DEFAULT_CACHE):
    bank, output = Path(bank).resolve(), Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError("Use a new cache destination; existing path artifacts are immutable.")
    with np.load(bank, allow_pickle=False) as archive:
        times = archive["times"].copy()
        world = archive["world_matrices_gf_row"].transpose(0, 1, 3, 2).copy()
        names = [str(path).rsplit("/", 1)[-1] for path in archive["link_paths"]]
    assert times.ndim == 1 and np.all(np.diff(times) > 0)
    assert np.isfinite(world).all() and len(world) == len(times)
    indices = {name: index for index, name in enumerate(names)}
    urdf = ET.parse(URDF).getroot()
    joints = [joint for joint in urdf.findall("joint") if joint.get("type") == "revolute"]
    assert len(joints) == 7
    columns, limits, lower, upper, origins, translation_errors, rotation_errors = [], [], [], [], [], [], []
    for joint in joints:
        axis = np.array([float(x) for x in joint.find("axis").get("xyz").split()])
        assert np.allclose(axis, (0, 0, 1)), "This reviewed Panda extractor expects URDF local Z axes."
        parent = world[:, indices[joint.find("parent").get("link")]]
        child = world[:, indices[joint.find("child").get("link")]]
        relative = np.linalg.inv(parent) @ child
        origin = origin_matrix(joint)
        rotation = origin[:3, :3].T @ relative[:, :3, :3]
        q = np.unwrap(np.arctan2(rotation[:, 1, 0], rotation[:, 0, 0]))
        expected_rotation = np.zeros_like(rotation)
        expected_rotation[:, 0, 0] = expected_rotation[:, 1, 1] = np.cos(q)
        expected_rotation[:, 1, 0] = np.sin(q)
        expected_rotation[:, 0, 1] = -np.sin(q)
        expected_rotation[:, 2, 2] = 1
        translation_errors.append(float(np.max(np.linalg.norm(relative[:, :3, 3] - origin[:3, 3], axis=1))))
        rotation_errors.append(float(np.max(np.linalg.norm(rotation - expected_rotation, axis=(1, 2)))))
        limit = joint.find("limit")
        columns.append(q)
        limits.append(float(limit.get("velocity")))
        lower.append(float(limit.get("lower")))
        upper.append(float(limit.get("upper")))
        origins.append(origin)
    q, limits = np.array(columns).T, np.asarray(limits)
    assert max(translation_errors) < 1e-6 and max(rotation_errors) < 1e-5
    assert np.all(q >= np.asarray(lower) - 1e-6) and np.all(q <= np.asarray(upper) + 1e-6)
    edge = np.max(np.abs(np.diff(q, axis=0)) / limits, axis=1)
    cumulative = np.r_[0.0, np.cumsum(edge)]
    outbound = times <= 8.4 + 1e-10
    source, path_q, u = times[outbound], q[outbound], cumulative[outbound]
    assert abs(source[-1] - 8.4) < 1e-10 and np.all(np.diff(u) > 1e-12)
    peak_ratio = np.max(np.abs(np.diff(q, axis=0)) / np.diff(times)[:, None] / limits, axis=0)
    fixed = {joint.get("name"): joint for joint in urdf.findall("joint")}
    hand_fixed = origin_matrix(fixed["panda_joint8"]) @ origin_matrix(fixed["panda_hand_joint"])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, u=u, source=source, joint_q=path_q,
             joint_names=np.asarray([joint.get("name") for joint in joints]),
             velocity_limits_rad_s=limits, lower_limits_rad=np.asarray(lower), upper_limits_rad=np.asarray(upper),
             origin_matrices_col=np.asarray(origins), hand_fixed_matrix_col=hand_fixed,
             base_world_matrix_col=world[0, indices["panda_link0"]],
             geometry_sha256=np.asarray(sha(GEOMETRY)), urdf_sha256=np.asarray(sha(URDF)),
             legacy_bank_sha256=np.asarray(sha(bank)), source_time_scale=np.asarray(2.0))
    report = {"schema_version": 1, "kind": "robot_path_parameterization_not_liquid_replay",
              "cache_file": output.name, "cache_sha256": sha(output),
              "inputs": {"legacy_bank_sha256": sha(bank), "geometry_sha256": sha(GEOMETRY), "urdf_sha256": sha(URDF)},
              "source_time_scale": 2.0, "source_outbound_max": 8.4, "u_max": float(u[-1]),
              "robot_joint_position_bounds_passed": True,
              "max_urdf_translation_residual_m": max(translation_errors),
              "max_joint_rotation_residual_frobenius": max(rotation_errors),
              "recovered_peak_legacy_velocity_limit_ratio": float(max(peak_ratio)),
              "velocity_only_outbound_seconds": float(u[-1]),
              "velocity_only_out_and_reverse_home_seconds": float(2 * u[-1]),
              "velocity_only_forward_upright_then_reverse_transfer_home_seconds": float(cumulative[-1] + np.interp(4.4, times, cumulative)),
              "old_source_rate_one_lower_bound_to_78_degrees_s": 7.07577159,
              "stage_coordinates": [{"old_source": value, "u": float(np.interp(value, source, u))}
                                    for value in (0.0, 1.4, 4.4, 6.4, 7.07577159, 7.239, 8.4)],
              "limitations": ["Velocity-only bound ignores acceleration, jerk, fluid and dose/settle time.",
                  "Joint q is piecewise linear in u. Its derivative has bank-edge corners; controller u(t) must stop smoothly at domain ends.",
                  "No joint acceleration limit exists in this URDF; report sampled acceleration in rad/s^2 without inventing a manufacturer limit.",
                  "Original analytic mouth/hand geometry is evaluated through piecewise-linear u-to-source mapping; dense FK/grasp verification remains required."]}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--output", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()
    print(json.dumps(prepare(args.bank, args.output), indent=2))


if __name__ == "__main__":
    main()
