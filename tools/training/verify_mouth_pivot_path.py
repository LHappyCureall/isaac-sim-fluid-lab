"""CPU-only verification of the two independently cached fixed-mouth paths."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "projects/pour_policy_training_v3"
SOURCE = ROOT / "assets/robot_testtube_pour/scripts/mouth_pivot_fast_geometry.py"


def load():
    spec = importlib.util.spec_from_file_location("pivot_cpu_geometry", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rectangle_sdf(values):
    return np.linalg.norm(np.maximum(values, 0), axis=1) + np.minimum(np.max(values, axis=1), 0)


def verify(g):
    vertices, faces = g.rounded_testtube_mesh()
    triangles = np.asarray(faces).reshape(-1, 3)
    tube_samples = np.r_[vertices, np.mean(vertices[triangles], axis=1)]
    center, radius = np.asarray(g.BEAKER_CENTER), g.BEAKER_RADIUS
    thickness, height = g.BEAKER_THICKNESS, g.BEAKER_HEIGHT
    hold = 4.8 - 2 * g.U_MAX
    if hold <= 0:
        raise ValueError("No nominal five-second roundtrip budget remains.")
    times = np.linspace(0, 5, 1201)
    us = np.where(times <= g.U_MAX, times, np.where(times <= g.U_MAX + hold, g.U_MAX,
                  np.maximum(0, 2 * g.U_MAX + hold - times)))
    qs = np.array([g.joint_positions(u) for u in us])
    qd = np.diff(qs, axis=0) * 240
    qdd = np.diff(qs, n=2, axis=0) * 240 ** 2
    min_table = min_cup = math.inf
    max_grasp = max_dose_mouth = max_dose_rotation = max_joint_acc = max_joint_velocity = 0.
    max_prepare_tilt = 0.
    for u in us:
        matrix = g.tube_matrix(u)
        sample = tube_samples @ matrix[:3, :3].T + matrix[:3, 3]
        delta = sample - center
        radial, z = np.linalg.norm(delta[:, :2], axis=1), delta[:, 2]
        wall = rectangle_sdf(np.c_[np.abs(radial - (radius + thickness / 2)) - thickness / 2,
                                   np.abs(z - height / 2) - height / 2])
        floor = rectangle_sdf(np.c_[radial - radius - thickness, np.abs(z + thickness / 2) - thickness / 2])
        min_cup = min(min_cup, float(min(np.minimum(wall, floor))))
        min_table = min(min_table, float(min(sample[:, 2] - g.TABLE_TOP_Z)))
        hand, hand_r = g.hand_fk(u)
        tcp = hand + hand_r @ np.array((0., 0., g.PANDA_HAND_TO_TCP))
        tube_tcp = matrix[:3, 3] + matrix[:3, :3] @ np.array((0., 0., g.TUBE_LENGTH - g.TCP_BELOW_MOUTH))
        max_grasp = max(max_grasp, float(np.linalg.norm(tcp - tube_tcp)))
        mouth, tilt = g.mouth_pose(u)
        if u >= g.PREPARE_END_U:
            max_dose_mouth = max(max_dose_mouth, float(np.linalg.norm(mouth - g.PIVOT_M)))
            max_dose_rotation = max(max_dose_rotation, float(np.linalg.norm(matrix[:3, :3] - g.rotation_y(tilt))))
        else:
            max_prepare_tilt = max(max_prepare_tilt, math.degrees(tilt))
        max_joint_acc = max(max_joint_acc, float(max(np.abs(g.joint_second_derivative(u)))))
        max_joint_velocity = max(max_joint_velocity, float(max(np.abs(g.joint_derivative(u)) / g.JOINT_VELOCITY_LIMITS_RAD_S)))
        _, xyzw = g.tube_pose(u)
        x, y, zq, w = xyzw
        reconstructed = np.array(((1-2*(y*y+zq*zq), 2*(x*y-zq*w), 2*(x*zq+y*w)),
                                  (2*(x*y+zq*w), 1-2*(x*x+zq*zq), 2*(y*zq-x*w)),
                                  (2*(x*zq-y*w), 2*(y*zq+x*w), 1-2*(x*x+y*y))))
        assert np.max(np.abs(reconstructed - matrix[:3, :3])) < 1e-10
    # Dense nominal-u analytic sampling also covers interpolation cells missed by 240 Hz.
    for u in np.linspace(0, g.U_MAX, 12001):
        max_joint_acc = max(max_joint_acc, float(max(np.abs(g.joint_second_derivative(u)))))
        max_joint_velocity = max(max_joint_velocity, float(max(np.abs(g.joint_derivative(u)) / g.JOINT_VELOCITY_LIMITS_RAD_S)))
    points, groups = g.initial_water_points()
    old_points, old_groups = g._STATIC.initial_water_points()
    water_same = np.array_equal(points, old_points) and np.array_equal(groups, old_groups)
    initial = g.classify_water(points, groups, 0)
    q_bounds = bool(np.all(qs >= g.JOINT_LOWER_LIMITS_RAD - 1e-10) and np.all(qs <= g.JOINT_UPPER_LIMITS_RAD + 1e-10))
    ratio = float(max(np.max(np.abs(qd), axis=0) / g.JOINT_VELOCITY_LIMITS_RAD_S))
    maximum_qdd = float(np.max(np.abs(qdd)))
    passed = (water_same and q_bounds and ratio <= 1 + 1e-8 and maximum_qdd <= 25
              and max_joint_acc <= 25 and max_joint_velocity <= 1 and min_cup > 0 and min_table > 0
              and max_dose_mouth < 1e-5 and max_dose_rotation < 1e-5
              and initial["tube_remaining"] == 1198 and initial["total"] == 5330)
    return {"schema_version": 1, "kind": "CPU_fixed_mouth_route_not_liquid_validation", "cpu_passed": bool(passed),
            "geometry_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "cache_file": g.CACHE_FILE.name, "cache_sha256": g.CACHE_SHA256,
            "mouth_pivot_m": g.PIVOT_M.tolist(), "nominal_coordinate": "u seconds of embedded C2 joint clock",
            "scalar_profile": "C2 sine-ramped trapezoid velocity", "ramp_s": g.RAMP_SECONDS,
            "prepare_s": g.PREPARE_END_U, "dose_78_to_125_s": g.DOSE_LEG_SECONDS,
            "outbound_s": g.U_MAX, "roundtrip_s": 2 * g.U_MAX, "hold_available_s": hold,
            "home_arrival_s": 4.8, "home_quiet_s": .2, "deadline_s": 5., "sampling_hz": 240,
            "joint_velocity_FD_limit_ratio": ratio, "joint_acceleration_FD_rad_s2": maximum_qdd,
            "nominal_u_analytic_joint_velocity_limit_ratio": max_joint_velocity,
            "nominal_u_analytic_joint_acceleration_rad_s2": max_joint_acc,
            "joint_position_bounds_passed": q_bounds, "tube_surface_samples": len(tube_samples),
            "minimum_sampled_table_clearance_m": min_table,
            "minimum_sampled_cup_glass_signed_clearance_m": min_cup,
            "maximum_nominal_grasp_error_m": max_grasp,
            "maximum_dose_mouth_position_error_m": max_dose_mouth,
            "maximum_dose_orientation_error_vs_pure_Ry_frobenius": max_dose_rotation,
            "maximum_prepare_tilt_deg": max_prepare_tilt,
            "initial_water_bitwise_unchanged": water_same, "initial_tube_particles": initial["tube_remaining"],
            "initial_total_particles": initial["total"],
            "initial_tube_pose_position_error_m": float(np.linalg.norm(g.tube_pose(0)[0] - g._STATIC.tube_pose(0)[0])),
            "home_joint_error_rad": float(max(np.abs(qs[-1] - qs[0]))),
            "limitations": ["No PBF or GPU simulation ran. No dose accuracy or spill improvement is established.",
                "Five-second budget is a nominal full out-and-back clock plus hold; live controller rate/stop changes require rechecking.",
                "Mesh vertex/triangle-center clearance sampling is not continuous collision proof or robot self-collision checking.",
                "Joint acceleration cap25rad/s2 is an experiment limit, not an official URDF specification.",
                "Prepare still combines transport and tilt; early startup ejection needs a separate live comparison."]}


def main():
    module = load()
    results = []
    for label, sha in (("026", module.DEFAULT_CACHE_SHA256),
                        ("022", "f0dec80ef890b2ef8946627af98be8d736e2a3d684e7f8f7bf2adb120aa46d40")):
        g = module.load_geometry(HERE / ("mouth_pivot_z" + label + "_cache.npz"), sha)
        report = verify(g)
        path = HERE / ("mouth_pivot_z" + label + "_cpu_report.json")
        if path.exists():
            raise FileExistsError("Keep previous engineering reports immutable: " + str(path))
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        results.append(report)
    print(json.dumps(results, indent=2, allow_nan=False))
    if not all(row["cpu_passed"] for row in results):
        raise SystemExit("At least one independent pivot path failed CPU checks.")


if __name__ == "__main__":
    main()
