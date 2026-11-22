"""Independent CPU comparison: upright 2/2.5/3 cm lift before joint transport.

This does not change the default pivot route. The lift is dense Cartesian IK,
not a claim that joint interpolation alone preserves upright orientation.
"""

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

from prepare_mouth_pivot_path import coefficients, evaluate, forward, solve
from verify_mouth_pivot_path import load, rectangle_sdf


HERE = Path(__file__).resolve().parents[2] / "projects/pour_policy_training_v3"


def duration_for_joint_line(delta, limits, ramp=.2):
    return max(2 * ramp, ramp + float(max(np.abs(delta) / (.95 * limits))),
               ramp + float(max(np.abs(delta))) * math.pi / (2 * ramp * 25))


def lift_curve(g, height):
    grid = np.linspace(0, 1, 41)
    seed, positions = g.Q_HOME.copy(), []
    inverse_grasp = np.linalg.inv(g._HAND_TO_TUBE)
    initial = g.tube_matrix(0)
    maximum_error = 0.
    for fraction in grid:
        tube = initial.copy()
        tube[2, 3] += height * fraction
        seed, _, position_error, rotation_error = solve(tube @ inverse_grasp, seed, g._CACHE)
        positions.append(seed.copy())
        maximum_error = max(maximum_error, position_error, rotation_error)
    positions = np.asarray(positions)
    coeff = coefficients(grid, positions)
    query = np.linspace(0, 1, 8001)
    first, second = evaluate(grid, coeff, query, 1), evaluate(grid, coeff, query, 2)
    low, high = 0., min(5., float(np.min(.95 * g.JOINT_VELOCITY_LIMITS_RAD_S /
                                       np.maximum(np.max(np.abs(first), axis=0), 1e-12))))
    for _ in range(48):
        speed = .5 * (low + high)
        bound = np.max(np.abs(second) * speed ** 2 + np.abs(first) * speed * math.pi / .4)
        if bound <= 24:
            low = speed
        else:
            high = speed
    duration = 1 / low + .2
    return grid, coeff, positions[-1], duration, maximum_error


def compare(g, lift_height, progress):
    grid, coeff, qlift, lift_duration, ik_error = lift_curve(g, lift_height)
    transfer_duration = duration_for_joint_line(g.Q_78 - qlift, g.JOINT_VELOCITY_LIMITS_RAD_S)
    prepare_duration = lift_duration + transfer_duration
    outbound = prepare_duration + g.DOSE_LEG_SECONDS
    hold = 4.8 - 2 * outbound

    def state(u):
        if u <= lift_duration:
            p, v, a = progress(u, lift_duration, .2)
            q = evaluate(grid, coeff, np.asarray([p]), 0)[0]
            first = evaluate(grid, coeff, np.asarray([p]), 1)[0]
            second = evaluate(grid, coeff, np.asarray([p]), 2)[0]
            return q, first * v, second * v * v + first * a
        if u <= prepare_duration:
            p, v, a = progress(u - lift_duration, transfer_duration, .2)
            delta = g.Q_78 - qlift
            return qlift + delta * p, delta * v, delta * a
        return g._joint_state(g.PREPARE_END_U + u - prepare_duration)

    vertices, faces = g.rounded_testtube_mesh()
    triangles = np.asarray(faces).reshape(-1, 3)
    surface = np.r_[vertices, np.mean(vertices[triangles], axis=1)]
    minimum_cup = minimum_table = math.inf
    max_lift_tilt = max_lift_xy = max_lift_position_error = 0.
    max_analytic_acc = max_analytic_v = 0.
    qs = []
    times = np.linspace(0, 5, 1201)
    if hold < 0:
        times = np.linspace(0, 2 * outbound, math.ceil(2 * outbound * 240) + 1)
        us = np.minimum(times, 2 * outbound - times)
    else:
        us = np.where(times <= outbound, times,
                      np.where(times <= outbound + hold, outbound, np.maximum(0, 2 * outbound + hold - times)))
    for u in us:
        q, qd, qdd = state(float(u))
        qs.append(q)
        max_analytic_acc = max(max_analytic_acc, float(max(np.abs(qdd))))
        max_analytic_v = max(max_analytic_v, float(max(np.abs(qd) / g.JOINT_VELOCITY_LIMITS_RAD_S)))
        tube = forward(q, g._CACHE) @ g._HAND_TO_TUBE
        points = surface @ tube[:3, :3].T + tube[:3, 3]
        delta = points - np.asarray(g.BEAKER_CENTER)
        radial, z = np.linalg.norm(delta[:, :2], axis=1), delta[:, 2]
        r, h, thick = g.BEAKER_RADIUS, g.BEAKER_HEIGHT, g.BEAKER_THICKNESS
        wall = rectangle_sdf(np.c_[np.abs(radial - r - thick / 2) - thick / 2, np.abs(z - h / 2) - h / 2])
        floor = rectangle_sdf(np.c_[radial - r - thick, np.abs(z + thick / 2) - thick / 2])
        minimum_cup = min(minimum_cup, float(min(np.minimum(wall, floor))))
        minimum_table = min(minimum_table, float(min(points[:, 2] - g.TABLE_TOP_Z)))
        if u <= lift_duration:
            max_lift_tilt = max(max_lift_tilt, math.degrees(math.acos(float(np.clip(tube[2, 2], -1, 1)))))
            max_lift_xy = max(max_lift_xy, float(np.linalg.norm(tube[:2, 3] - g.tube_matrix(0)[:2, 3])))
            p = progress(float(u), lift_duration, .2)[0]
            wanted = g.tube_matrix(0)[:3, 3] + (0, 0, lift_height * p)
            max_lift_position_error = max(max_lift_position_error, float(np.linalg.norm(tube[:3, 3] - wanted)))
    qs = np.asarray(qs)
    dt = float(times[1] - times[0])
    velocity = float(np.max(np.abs(np.diff(qs, axis=0)) / dt / g.JOINT_VELOCITY_LIMITS_RAD_S))
    acceleration = float(np.max(np.abs(np.diff(qs, n=2, axis=0))) / dt ** 2)
    q_bounds = bool(np.all(qs >= g.JOINT_LOWER_LIMITS_RAD) and np.all(qs <= g.JOINT_UPPER_LIMITS_RAD))
    return {"lift_height_m": lift_height, "mouth_pivot_m": g.PIVOT_M.tolist(), "cache_sha256": g.CACHE_SHA256,
            "upright_lift_s": lift_duration, "lifted_to_78_s": transfer_duration, "prepare_s": prepare_duration,
            "dose_s": g.DOSE_LEG_SECONDS, "outbound_s": outbound, "roundtrip_s": 2 * outbound,
            "available_hold_with_homequiet_0_2_s": hold, "five_second_nominal_budget_possible": hold >= 0,
            "IK_max_error_m_or_rad": ik_error, "lift_tilt_max_deg": max_lift_tilt,
            "lift_xy_deviation_max_m": max_lift_xy, "lift_position_error_max_m": max_lift_position_error,
            "joint_position_bounds_passed": q_bounds, "sampling_hz": 1 / dt,
            "joint_velocity_FD_limit_ratio": velocity, "joint_acceleration_FD_rad_s2": acceleration,
            "sampled_analytic_joint_velocity_limit_ratio": max_analytic_v,
            "sampled_analytic_joint_acceleration_rad_s2": max_analytic_acc,
            "minimum_sampled_cup_glass_clearance_m": minimum_cup, "minimum_sampled_table_clearance_m": minimum_table,
            "cpu_constraints_passed": bool(q_bounds and velocity <= 1 + 1e-8 and acceleration <= 25
                                            and max_analytic_acc <= 25 and minimum_cup > 0 and minimum_table > 0)}


def main():
    module = load()
    cases = []
    for label, sha in (("026", module.DEFAULT_CACHE_SHA256),
                       ("022", "f0dec80ef890b2ef8946627af98be8d736e2a3d684e7f8f7bf2adb120aa46d40")):
        g = module.load_geometry(HERE / ("mouth_pivot_z" + label + "_cache.npz"), sha)
        for height in (.02, .025, .03):
            cases.append(compare(g, height, module._progress))
    report = {"schema_version": 1, "kind": "CPU_only_optional_upright_lift_not_installed", "cases": cases,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "geometry_sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
              "limitations": ["No liquid solver or GPU ran; startup ejection and spill benefits are unknown.",
                  "This optional three-leg route has not replaced the two-leg default runtime or any active geometry.",
                  "A pure upright Cartesian lift uses dense IK; transfer afterward remains a joint-space shortcut.",
                  "Clearance checks are sampled tube-vs-table/cup tests, not robot self-collision proofs.",
                  "Available hold is after nominal full-speed joint clocks; feedback slow tilting/retracting changes the budget."]}
    path = HERE / "upright_lift_pivot_cpu_report.json"
    if path.exists():
        raise FileExistsError("Keep previous engineering reports immutable.")
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
