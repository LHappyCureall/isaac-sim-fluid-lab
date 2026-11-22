"""Offline CPU IK for an independently versioned fixed-mouth pouring route."""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "projects/pour_policy_training_v3"
SCRIPTS = ROOT / "assets/robot_testtube_pour/scripts"
RAMP = .20


def load_reference():
    path = SCRIPTS / "direct_fast_geometry.py"
    spec = importlib.util.spec_from_file_location("pivot_preparation_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rotation_log(rotation):
    cosine = float(np.clip((np.trace(rotation) - 1) / 2, -1, 1))
    angle = math.acos(cosine)
    vee = np.array((rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0],
                    rotation[1, 0] - rotation[0, 1]))
    return .5 * vee if angle < 1e-7 else angle / (2 * math.sin(angle)) * vee


def forward(q, data, jacobian=False):
    matrix = data["base_world_matrix_col"].copy()
    axes, pivots = [], []
    for origin, angle in zip(data["origin_matrices_col"], q):
        matrix = matrix @ origin
        axes.append(matrix[:3, 2].copy())
        pivots.append(matrix[:3, 3].copy())
        cosine, sine = math.cos(float(angle)), math.sin(float(angle))
        rotation = np.array(((cosine, -sine, 0, 0), (sine, cosine, 0, 0),
                             (0, 0, 1, 0), (0, 0, 0, 1)), dtype=float)
        matrix = matrix @ rotation
    matrix = matrix @ data["hand_fixed_matrix_col"]
    if not jacobian:
        return matrix
    jac = np.zeros((6, 7))
    for index, (axis, pivot) in enumerate(zip(axes, pivots)):
        jac[:3, index] = np.cross(axis, matrix[:3, 3] - pivot)
        jac[3:, index] = axis
    return matrix, jac


def solve(target, seed, data):
    q, damping = np.asarray(seed).copy(), .001
    lower, upper = data["lower_limits_rad"], data["upper_limits_rad"]
    for iteration in range(160):
        actual, jacobian = forward(q, data, True)
        translation = target[:3, 3] - actual[:3, 3]
        rotation = rotation_log(target[:3, :3] @ actual[:3, :3].T)
        if np.linalg.norm(translation) < 2e-9 and np.linalg.norm(rotation) < 2e-8:
            return q, iteration + 1, float(np.linalg.norm(translation)), float(np.linalg.norm(rotation))
        error = np.r_[translation, .25 * rotation]
        jacobian[3:] *= .25
        step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping ** 2 * np.eye(6), error)
        length = np.linalg.norm(step)
        if length > .15:
            step *= .15 / length
        cost = float(error @ error)
        accepted = False
        for fraction in (1., .5, .25, .125, .0625):
            candidate = np.clip(q + fraction * step, lower + 1e-7, upper - 1e-7)
            matrix = forward(candidate, data)
            e = np.r_[target[:3, 3] - matrix[:3, 3],
                      .25 * rotation_log(target[:3, :3] @ matrix[:3, :3].T)]
            if float(e @ e) < cost:
                q, damping, accepted = candidate, max(1e-7, damping / 2), True
                break
        if not accepted:
            damping *= 5
    raise ValueError("Offline bounded IK did not converge for a pivot target.")


def coefficients(grid, values):
    first = np.gradient(values, grid, axis=0, edge_order=2)
    second = np.gradient(first, grid, axis=0, edge_order=2)
    h = np.diff(grid)[:, None]
    result = np.zeros((len(grid) - 1, 6, 7))
    result[:, 0], result[:, 1], result[:, 2] = values[:-1], h * first[:-1], .5 * h * h * second[:-1]
    rhs = np.stack((values[1:] - np.sum(result[:, :3], axis=1),
                    h * first[1:] - result[:, 1] - 2 * result[:, 2],
                    h * h * second[1:] - 2 * result[:, 2]), axis=1)
    result[:, 3:] = np.linalg.solve(np.array(((1., 1., 1.), (3., 4., 5.), (6., 12., 20.))), rhs)
    return result


def evaluate(grid, coeff, values, derivative):
    values = np.asarray(values)
    index = np.clip(np.searchsorted(grid, values, side="right") - 1, 0, len(grid) - 2)
    h = grid[index + 1] - grid[index]
    x = (values - grid[index]) / h
    result = np.zeros((len(values), 7))
    for power in range(derivative, 6):
        factor = math.factorial(power) / math.factorial(power - derivative)
        result += coeff[index, power] * (factor * x ** (power - derivative) / h ** derivative)[:, None]
    return result


def prepare(z, output):
    if not .20 <= z <= .28:
        raise ValueError("Use a separately checked pivot height in [.20,.28] m.")
    output = Path(output)
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError("Use a new pivot-cache destination; prior experiments are immutable.")
    reference = load_reference()
    data = {name: value.copy() for name, value in reference._CACHE.items() if name in
            ("base_world_matrix_col", "origin_matrices_col", "hand_fixed_matrix_col",
             "velocity_limits_rad_s", "lower_limits_rad", "upper_limits_rad", "joint_names")}
    hand_to_tube = reference._HAND_TO_TUBE.copy()
    pivot = np.array((.43, .10, z))
    angles = np.radians(np.linspace(78, 125, 189))
    table, diagnostics = [], []
    seed = reference.Q_78.copy()
    for angle in angles:
        tube = np.eye(4)
        tube[:3, :3] = reference.rotation_y(angle)
        tube[:3, 3] = pivot - tube[:3, :3] @ np.array((0, 0, reference.TUBE_LENGTH))
        target = tube @ np.linalg.inv(hand_to_tube)
        seed, count, position_error, rotation_error = solve(target, seed, data)
        table.append(seed.copy())
        diagnostics.append((count, position_error, rotation_error))
    table = np.asarray(table)
    coeff = coefficients(angles, table)
    query = np.linspace(angles[0], angles[-1], 20001)
    first, second = evaluate(angles, coeff, query, 1), evaluate(angles, coeff, query, 2)
    velocity = data["velocity_limits_rad_s"]
    low, high = 0., float(np.min(.95 * velocity / np.maximum(np.max(np.abs(first), axis=0), 1e-12)))
    high = min(high, float(angles[-1] - angles[0]) / RAMP)
    for _ in range(48):
        omega = .5 * (low + high)
        bound = np.max(np.abs(second) * omega ** 2 + np.abs(first) * omega * math.pi / (2 * RAMP))
        if bound <= 24:
            low = omega
        else:
            high = omega
    dose_duration = float(angles[-1] - angles[0]) / low + RAMP
    delta = np.abs(table[0] - reference.Q_HOME)
    prepare_duration = max(2 * RAMP, RAMP + float(max(delta / (.95 * velocity))),
                           RAMP + float(max(delta)) * math.pi / (2 * RAMP * 25))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **data, theta_rad=angles, theta_joint_q=table, theta_joint_coefficients=coeff,
             q_home=reference.Q_HOME, hand_to_tube=hand_to_tube, pivot_m=pivot,
             prepare_duration_s=np.asarray(prepare_duration), dose_duration_s=np.asarray(dose_duration),
             ramp_s=np.asarray(RAMP), original_geometry_sha256=np.asarray(
                 hashlib.sha256((SCRIPTS / "geometry.py").read_bytes()).hexdigest()),
             direct_geometry_sha256=np.asarray(hashlib.sha256((SCRIPTS / "direct_fast_geometry.py").read_bytes()).hexdigest()),
             reference_cache_sha256=np.asarray(reference.CACHE_SHA256))
    result = {"schema_version": 1, "kind": "offline_fixed_mouth_IK_not_liquid_validation",
        "pivot_m": pivot.tolist(), "cache_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "cache_file": output.name, "IK_poses": len(angles),
        "maximum_IK_iterations": max(row[0] for row in diagnostics),
        "maximum_IK_position_error_m": max(row[1] for row in diagnostics),
        "maximum_IK_orientation_error_rad": max(row[2] for row in diagnostics),
        "prepare_78_s": prepare_duration, "dose_78_to125_s": dose_duration,
        "outbound_s": prepare_duration + dose_duration, "roundtrip_s": 2 * (prepare_duration + dose_duration),
        "available_hold_at_5s_with_home_quiet_0_2": 4.8 - 2 * (prepare_duration + dose_duration),
        "dose_joint_acceleration_sufficient_bound_rad_s2": 24.0,
        "reference_geometry_sha256": str(np.asarray(reference.REFERENCE_GEOMETRY_SHA256)),
        "reference_cache_sha256": reference.CACHE_SHA256,
        "notice": "CPU IK/timing only; actual jet, spill, dose and startup pressure remain untested."}
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mouth-z", type=float, default=.26)
    parser.add_argument("--output", type=Path, default=HERE / "mouth_pivot_z026_cache.npz")
    args = parser.parse_args()
    print(json.dumps(prepare(args.mouth_z, args.output), indent=2))
