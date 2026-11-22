"""CPU-only dense joint/FK checks of a proposed five-second path clock.

The test clock is an engineering trapezoid, not a dosing policy. No simulator,
fluid, robot UI or recorded liquid trajectory is imported or executed.
"""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "_cache/pour_training/fast_v3"
SOURCE = ROOT / "assets/robot_testtube_pour/scripts/fast_geometry.py"


def _load_geometry():
    spec = importlib.util.spec_from_file_location("fast_path_cpu_geometry", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify():
    geometry = _load_geometry()
    a, vmax = 3.0, 0.95
    ta = vmax / a
    cruise = (geometry.U_MAX - vmax * vmax / a) / vmax
    assert cruise >= 0
    duration = 2 * ta + cruise
    dwell = max(0.0, min(1.2, 5.0 - .2 - 2 * duration))

    def outbound(t):
        if t <= ta:
            return 0.5 * a * t * t
        if t <= ta + cruise:
            return 0.5 * a * ta * ta + vmax * (t - ta)
        if t <= duration:
            left = duration - t
            return geometry.U_MAX - 0.5 * a * left * left
        return geometry.U_MAX

    def coordinate(t):
        if t <= duration:
            return outbound(t)
        if t <= duration + dwell:
            return geometry.U_MAX
        if t <= 2 * duration + dwell:
            return geometry.U_MAX - outbound(t - duration - dwell)
        return 0.0

    def rate_acceleration(t):
        sign = 1.0
        if t > duration:
            if t <= duration + dwell:
                return 0.0, 0.0
            t -= duration + dwell
            sign = -1.0
        if t < ta:
            return sign * a * t, sign * a
        if t < ta + cruise:
            return sign * vmax, 0.0
        if t < duration:
            return sign * a * (duration - t), -sign * a
        return 0.0, 0.0

    points, labels = geometry.initial_water_points()
    old_points, old_labels = geometry._OLD.initial_water_points()
    assert np.array_equal(points, old_points) and np.array_equal(labels, old_labels)
    rows = []
    for fps in (120, 240):
        times = np.linspace(0, 5, 5 * fps + 1)
        us = np.array([coordinate(float(t)) for t in times])
        qs = np.array([geometry.joint_positions(u) for u in us])
        velocities = np.diff(qs, axis=0) * fps
        accelerations = np.diff(velocities, axis=0) * fps
        physical_analytic = []
        for t, u in zip(times, us):
            rate, acceleration = rate_acceleration(float(t))
            physical_analytic.append(geometry.joint_second_derivative(u) * rate ** 2
                                     + geometry.joint_derivative(u) * acceleration)
        hand_errors, orientation_errors, grasp_errors = [], [], []
        mouths, tilts = [], []
        for u in us:
            mouth, angle = geometry.mouth_pose(u)
            expected_hand, _ = geometry.hand_pose(u)
            actual_hand, rotation = geometry.hand_fk(u)
            target_rotation = geometry.rotation_y(angle + math.pi / 2)
            cosine = float(np.clip((np.trace(target_rotation.T @ rotation) - 1) / 2, -1, 1))
            hand_errors.append(float(np.linalg.norm(actual_hand - expected_hand)))
            orientation_errors.append(math.acos(cosine))
            expected_tcp = mouth - geometry.rotation_y(angle) @ np.array([0, 0, geometry.TCP_BELOW_MOUTH])
            actual_tcp = actual_hand + rotation @ np.array([0, 0, geometry.PANDA_HAND_TO_TCP])
            grasp_errors.append(float(np.linalg.norm(actual_tcp - expected_tcp)))
            mouths.append(mouth)
            tilts.append(angle)
        ratio = np.max(np.abs(velocities) / geometry.JOINT_VELOCITY_LIMITS_RAD_S, axis=0)
        acceleration_peaks = np.max(np.abs(accelerations), axis=0)
        rows.append({"fps": fps, "samples": len(times),
                     "max_joint_velocity_limit_ratio": float(max(ratio)),
                     "peak_joint_velocity_rad_s": np.max(np.abs(velocities), axis=0).tolist(),
                     "peak_joint_acceleration_rad_s2": acceleration_peaks.tolist(),
                     "max_joint_acceleration_rad_s2": float(max(acceleration_peaks)),
                     "max_analytic_joint_acceleration_on_clock_rad_s2": float(np.max(np.abs(physical_analytic))),
                     "joint_velocity_gate_passed": bool(max(ratio) <= 1.00001),
                     "joint_acceleration_25_rad_s2_sim_gate_passed": bool(max(acceleration_peaks) <= 25.00001),
                     "max_hand_position_error_m": max(hand_errors),
                     "max_hand_orientation_error_rad": max(orientation_errors),
                     "max_grasp_position_error_m": max(grasp_errors),
                     "pose_grasp_gate_passed": bool(max(hand_errors) <= .002 and max(orientation_errors) <= .02 and max(grasp_errors) <= .003),
                     "max_mouth_speed_m_s": float(np.max(np.linalg.norm(np.diff(mouths, axis=0) * fps, axis=1))),
                     "max_tilt_speed_rad_s": float(np.max(np.abs(np.diff(tilts) * fps))),
                     "max_tilt_acceleration_rad_s2": float(np.max(np.abs(np.diff(tilts, n=2) * fps * fps))),
                     "return_home_q_error_rad": float(np.max(np.abs(qs[-1] - qs[0])))})
    return {"schema_version": 1, "kind": "cpu_robot_path_feasibility_not_live_dose_validation",
            "geometry_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "cache_sha256": geometry.CACHE_SHA256,
            "path_u_max": geometry.U_MAX, "u_units": "normalized robot-speed seconds",
            "u_max_rate": vmax, "u_max_acceleration": a,
            "engineering_clock": {"outbound_end_s": duration, "dose_hold_s": dwell,
                                  "reverse_start_s": duration + dwell, "home_arrival_s": 2 * duration + dwell,
                                  "action_deadline_s": 5.0},
            "particle_initial_state_bitwise_equal": True, "particle_count": len(points),
            "initial_source_particle_count": int(np.count_nonzero(labels == 1)),
            "dense_checks": rows,
            "limitations": ["This clock does not dose 20/30/40mL; no liquid solver was run.",
                "The 25rad/s^2 acceleration gate is a declared simulation engineering cap, not an official URDF specification.",
                "C2 joint/C1 source interpolation removes velocity snaps; arbitrary policy u-rate changes still need signed physical joint-acceleration checks.",
                "Analytic derivatives are evaluated at the listed dense sample times; this is not a formal interval certificate.",
                "Visible USD robot/grasp readback must still be verified in Isaac Sim before claiming GUI acceptance."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DIRECTORY / "fast_path_cpu_report.json")
    args = parser.parse_args()
    report = verify()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
