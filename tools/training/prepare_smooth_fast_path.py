"""CPU engineering search for a C1 path clock with unchanged spatial geometry.

The old source coordinate remains the argument of the original analytic pose.
Only the clock and the auxiliary cached joint interpolation are smoothed.
This is robot timing calibration, not an action-conditioned liquid rollout.
"""

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "_cache/pour_training/fast_v3"
RAW_CACHE = DIRECTORY / "fast_path_cache.npz"
OUTPUT = DIRECTORY / "fast_path_smooth_cache.npz"
U_RATE_LIMIT = 0.95
U_ACCELERATION_LIMIT = 3.0
JOINT_ACCELERATION_CAP = 25.0


def polynomial(grid, coefficients, values, derivative=0):
    values = np.asarray(values, dtype=float).reshape(-1)
    indices = np.clip(np.searchsorted(grid, values, side="right") - 1, 0, len(grid) - 2)
    h = grid[indices + 1] - grid[indices]
    x = (values - grid[indices]) / h
    result = np.zeros((len(values), coefficients.shape[-1]))
    for power in range(derivative, coefficients.shape[1]):
        factor = 1
        for step in range(derivative):
            factor *= power - step
        result += coefficients[indices, power] * (factor * x ** (power - derivative) / h ** derivative)[:, None]
    return result


def joint_coefficients(source, q):
    first = np.gradient(q, source, axis=0, edge_order=2)
    second = np.gradient(first, source, axis=0, edge_order=2)
    for stop in (0.0, 1.4, 4.4, 6.4, 8.4):
        indices = np.isclose(source, stop, atol=1e-10)
        first[indices], second[indices] = 0.0, 0.0
    h = np.diff(source)[:, None]
    coefficients = np.zeros((len(source) - 1, 6, 7))
    coefficients[:, 0] = q[:-1]
    coefficients[:, 1] = h * first[:-1]
    coefficients[:, 2] = .5 * h * h * second[:-1]
    rhs = np.stack((q[1:] - np.sum(coefficients[:, :3], axis=1),
                    h * first[1:] - coefficients[:, 1] - 2 * coefficients[:, 2],
                    h * h * second[1:] - 2 * coefficients[:, 2]), axis=1)
    coefficients[:, 3:] = np.linalg.solve(np.array(((1., 1., 1.), (3., 4., 5.), (6., 12., 20.))), rhs)
    return coefficients


def inverse_clock_coefficients(source, clock, density):
    slope = 1.0 / density
    # Fritsch-Carlson sufficient monotonicity restriction, preserving C1.
    for index, secant in enumerate(np.diff(source) / np.diff(clock)):
        norm = np.hypot(slope[index] / secant, slope[index + 1] / secant)
        if norm > 3:
            slope[index:index + 2] *= 3 / norm
    h = np.diff(clock)
    delta = np.diff(source)
    coefficients = np.zeros((len(clock) - 1, 4, 1))
    coefficients[:, 0, 0] = source[:-1]
    coefficients[:, 1, 0] = h * slope[:-1]
    coefficients[:, 2, 0] = 3 * delta - h * (2 * slope[:-1] + slope[1:])
    coefficients[:, 3, 0] = -2 * delta + h * (slope[:-1] + slope[1:])
    return coefficients


def evaluate_clock(raw, q_coefficients, source_grid, density, clock=None):
    if clock is None:
        clock = np.r_[0., np.cumsum(.5 * (density[:-1] + density[1:]) * np.diff(source_grid))]
    coefficients = inverse_clock_coefficients(source_grid, clock, density)
    query = np.linspace(0, clock[-1], 50001)
    source = polynomial(clock, coefficients, query)[:, 0]
    ds = polynomial(clock, coefficients, query, 1)[:, 0]
    d2s = polynomial(clock, coefficients, query, 2)[:, 0]
    dq = polynomial(raw["source"], q_coefficients, source, 1)
    d2q = polynomial(raw["source"], q_coefficients, source, 2)
    q_u = dq * ds[:, None]
    q_uu = d2q * ds[:, None] ** 2 + dq * d2s[:, None]
    velocity_ratio = np.max(np.abs(q_u) * U_RATE_LIMIT / raw["velocity_limits_rad_s"])
    unit_velocity_ratio = np.max(np.abs(q_u) / raw["velocity_limits_rad_s"])
    unit_acceleration = np.max(np.abs(q_uu))
    # Sufficient for any controller |u_dot|<=.95, |u_ddot|<=3.
    joint_acceleration_bound = np.max(np.abs(q_uu) * U_RATE_LIMIT ** 2
                                      + np.abs(q_u) * U_ACCELERATION_LIMIT)
    one_way = clock[-1] / U_RATE_LIMIT + U_RATE_LIMIT / U_ACCELERATION_LIMIT
    return clock, coefficients, {"u_max": float(clock[-1]),
        "velocity_limit_ratio_bound": float(velocity_ratio),
        "unit_rate_joint_velocity_ratio": float(unit_velocity_ratio),
        "unit_rate_joint_acceleration_rad_s2": float(unit_acceleration),
        "unit_rate_velocity_acceleration_gate_passed": bool(unit_velocity_ratio <= 1.00001 and unit_acceleration <= JOINT_ACCELERATION_CAP),
        "joint_acceleration_bound_rad_s2": float(joint_acceleration_bound),
        "velocity_and_acceleration_gate_passed": bool(velocity_ratio <= 1.00001 and joint_acceleration_bound <= JOINT_ACCELERATION_CAP),
        "rest_to_rest_out_and_home_seconds": float(2 * one_way),
        "available_hold_before_5s_with_home_quiet_0_2s": float(5 - .2 - 2 * one_way)}


def constrained_clock(raw, coefficients, grid, acceleration):
    """Backward feasible speed intervals, then forward maximal feasible speed.

    y=(d source/du)^2. Midpoint joint acceleration inequalities are linear
    in adjacent y values, so eliminating the next y gives its feasible range.
    All final interpolated bounds are checked independently at 50,001 u's.
    """
    first = polynomial(raw["source"], coefficients, grid, 1)
    velocity = .98 * raw["velocity_limits_rad_s"]
    maximum = np.min((velocity / np.maximum(np.abs(first), 1e-12)) ** 2, axis=1)
    maximum = np.minimum(maximum, 50.0 ** 2)
    midpoints, ds = .5 * (grid[:-1] + grid[1:]), np.diff(grid)
    first_mid = polynomial(raw["source"], coefficients, midpoints, 1)
    second_mid = polynomial(raw["source"], coefficients, midpoints, 2)
    second = polynomial(raw["source"], coefficients, grid, 2)
    left = np.c_[-.5 * first_mid / ds[:, None] + .5 * second_mid,
                 -.5 * first[:-1] / ds[:, None] + second[:-1], -.5 * first[1:] / ds[:, None]]
    right = np.c_[.5 * first_mid / ds[:, None] + .5 * second_mid,
                  .5 * first[:-1] / ds[:, None], .5 * first[1:] / ds[:, None] + second[1:]]
    left, right = np.c_[left, -left], np.c_[right, -right]
    feasible = maximum.copy()
    for index in range(len(grid) - 2, -1, -1):
        a, b = left[index], right[index]
        positive, negative, zero = b > 1e-12, b < -1e-12, np.abs(b) <= 1e-12
        upper_m = np.r_[-a[positive] / b[positive], 0.]
        upper_c = np.r_[acceleration / b[positive], feasible[index + 1]]
        lower_m = np.r_[-a[negative] / b[negative], 0.]
        lower_c = np.r_[acceleration / b[negative], 0.]
        coefficient = lower_m[:, None] - upper_m[None, :]
        constant = upper_c[None, :] - lower_c[:, None]
        mask = coefficient > 1e-12
        bounds = constant[mask] / coefficient[mask]
        if np.any(zero & (a > 1e-12)):
            bounds = np.r_[bounds, acceleration / a[zero & (a > 1e-12)]]
        if len(bounds):
            feasible[index] = min(feasible[index], max(0.0, float(min(bounds))))
    y = np.zeros_like(grid)
    y[0] = feasible[0] * (1 - 1e-10)
    for index in range(len(grid) - 1):
        a, b = left[index], right[index]
        positive, negative = b > 1e-12, b < -1e-12
        upper = min(feasible[index + 1], float(min((acceleration - a[positive] * y[index]) / b[positive], default=np.inf)))
        lower = max(0.0, float(max((acceleration - a[negative] * y[index]) / b[negative], default=0.0)))
        if upper < lower - 1e-5:
            raise ValueError("Discrete clock feasible-interval propagation failed.")
        y[index + 1] = max(0.0, upper) * (1 - 1e-10)
    speed = np.sqrt(y)
    if np.any(speed <= 0):
        raise ValueError("A clock interval has zero traversable source speed.")
    clock = np.r_[0., np.cumsum(2 * ds / (speed[:-1] + speed[1:]))]
    return clock, 1 / speed


def prepare():
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists() or OUTPUT.with_suffix(".json").exists():
        raise FileExistsError("Preserve existing robot path evidence; use a fresh local cache directory.")
    with np.load(RAW_CACHE, allow_pickle=False) as archive:
        raw = {name: archive[name].copy() for name in archive.files}
    joint_coeff = joint_coefficients(raw["source"], raw["joint_q"])
    source_grid = np.unique(np.round(np.r_[np.linspace(0, raw["source"][-1], 8401), raw["source"]], 12))
    q_s = polynomial(raw["source"], joint_coeff, source_grid, 1)
    q_ss = polynomial(raw["source"], joint_coeff, source_grid, 2)
    required = np.max(np.abs(q_s) / raw["velocity_limits_rad_s"], axis=1)
    candidates, accepted = [], []
    for mode in ("hypot", "add"):
        for floor in (.02, .04, .06, .08, .10, .12, .16, .20, .25, .30, .40):
            density = np.hypot(required, floor) if mode == "hypot" else required + floor
            clock, coefficients, metrics = evaluate_clock(raw, joint_coeff, source_grid, density)
            metrics.update(mode=mode, old_source_clock_floor=floor)
            candidates.append(metrics)
            if metrics["velocity_and_acceleration_gate_passed"]:
                accepted.append((metrics["u_max"], metrics, clock, coefficients))
    for floor in (.005, .01, .02, .04):
        density = np.maximum(np.hypot(required, floor),
                             np.sqrt(np.max(np.abs(q_ss), axis=1) / 17.0))
        weights = np.hanning(63)
        weights /= sum(weights)
        for iteration in range(120):
            current_clock = np.r_[0., np.cumsum(.5 * (density[:-1] + density[1:]) * np.diff(source_grid))]
            current_coefficients = inverse_clock_coefficients(source_grid, current_clock, density)
            query = .5 * (current_clock[:-1] + current_clock[1:])
            source_query = polynomial(current_clock, current_coefficients, query)[:, 0]
            ds = polynomial(current_clock, current_coefficients, query, 1)[:, 0]
            d2s = polynomial(current_clock, current_coefficients, query, 2)[:, 0]
            first = polynomial(raw["source"], joint_coeff, source_query, 1)
            second = polynomial(raw["source"], joint_coeff, source_query, 2)
            q_u = first * ds[:, None]
            q_uu = second * ds[:, None] ** 2 + first * d2s[:, None]
            edge_bound = np.max(np.abs(q_uu) * U_RATE_LIMIT ** 2
                                + np.abs(q_u) * U_ACCELERATION_LIMIT, axis=1)
            bound = np.maximum(np.r_[edge_bound[0], edge_bound], np.r_[edge_bound, edge_bound[-1]])
            if max(bound) <= 22.0:
                break
            factor = np.clip(np.sqrt(bound / 22.0), 1.0, 1.12)
            density *= factor
            smoothed = np.convolve(np.pad(density, (31, 31), mode="edge"), weights, mode="valid")
            density = np.maximum(density, smoothed)
        clock, coefficients, metrics = evaluate_clock(raw, joint_coeff, source_grid, density)
        metrics.update(mode="adaptive_local_acceleration_envelope", old_source_clock_floor=floor,
                       iterations=iteration + 1)
        candidates.append(metrics)
        if metrics["velocity_and_acceleration_gate_passed"]:
            accepted.append((metrics["u_max"], metrics, clock, coefficients))
    for acceleration in (14., 18., 20., 22., 24.):
        clock, density = constrained_clock(raw, joint_coeff, source_grid, acceleration)
        clock, coefficients, metrics = evaluate_clock(raw, joint_coeff, source_grid, density, clock=clock)
        metrics.update(mode="backward_forward_joint_constraints", unit_clock_joint_acceleration_cap=acceleration)
        candidates.append(metrics)
        if metrics["unit_rate_velocity_acceleration_gate_passed"]:
            accepted.append((metrics["u_max"], metrics, clock, coefficients))
    report = {"schema_version": 1, "kind": "cpu_smooth_clock_engineering_comparison_not_fluid_training",
        "u_rate_limit": U_RATE_LIMIT, "u_acceleration_limit": U_ACCELERATION_LIMIT,
        "joint_acceleration_cap_rad_s2": JOINT_ACCELERATION_CAP, "dense_samples_per_candidate": 50001,
        "joint_interpolation": "C2 quintic Hermite in original source; zero first/second derivatives at original phase endpoints",
        "source_interpolation": "C1 monotone cubic Hermite from u to original analytic source",
        "spatial_pose_path_changed": False, "candidates": candidates,
        "limitations": ["Sufficient bound covers arbitrary signed u rates/acceleration within the declared caps; may be conservative.",
            "Timing comparison is CPU robot-path engineering only; liquid quantity and stability are untested."]}
    if accepted:
        _, metrics, clock, coefficients = min(accepted, key=lambda value: value[0])
        data = dict(raw)
        data.update(u=clock, source=source_grid,
                    joint_q=polynomial(raw["source"], joint_coeff, source_grid),
                    source_coefficients=coefficients, joint_spline_source=raw["source"],
                    joint_coefficients=joint_coeff, interpolation_version=np.asarray(2))
        np.savez(OUTPUT, **data)
        report.update(selected=metrics, cache_file=OUTPUT.name,
                      cache_sha256=hashlib.sha256(OUTPUT.read_bytes()).hexdigest())
        OUTPUT.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (DIRECTORY / "smooth_clock_comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    report = prepare()
    print(json.dumps({"selected": report.get("selected"), "cache_sha256": report.get("cache_sha256"),
                      "candidates": report["candidates"]}, indent=2))
